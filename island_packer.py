"""
Island Mask Packer
==================
Extracts island blobs from a B&W PNG and repacks them with controlled
edge-to-edge spacing using:
  - Connected Component Labeling (extraction)
  - Force-Directed Separation (packing)
  - Per-island randomized min/max distances for organic spacing
  - Multi-box representation for large/elongated islands

Dependencies: pip install opencv-python numpy Pillow
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import random
import math
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PackerConfig:
    # Canvas size for the output image
    canvas_size: Tuple[int, int] = (2048, 2048)

    # Per-island minimum edge distance is randomly chosen from this range.
    # For a pair of islands, their effective min distance = average of both.
    # Note: BB-based repulsion underestimates ~5px for irregular shapes,
    # so these values should be ~5px above your actual desired minimum.
    min_edge_distance_range: Tuple[int, int] = (7, 13)

    # Per-island maximum edge distance is randomly chosen from this range.
    # Islands whose nearest neighbor exceeds their max distance get pulled in.
    # Set both to 0 to disable max distance enforcement.
    max_edge_distance_range: Tuple[int, int] = (15, 20)

    # Islands with area >= this threshold AND aspect ratio >= split_aspect_ratio
    # get recursively split along their longest axis up to max_splits times.
    # This lets other islands nestle closer to large shapes.
    # 0 = disabled (no splitting)
    split_area_threshold: int = 2561

    # Minimum aspect ratio (max_dim / min_dim) to trigger splitting.
    # Set to 1.0 to split all islands above the area threshold regardless of shape.
    split_aspect_ratio: float = 1.0

    # Maximum number of recursive splits per island (each split doubles boxes).
    # 1 = up to 2 boxes, 2 = up to 4, 3 = up to 8, 4 = up to 16.
    max_splits: int = 2

    # Filter: skip islands smaller than this many pixels (noise removal)
    min_island_area: int = 50

    # Filter: skip islands larger than this (optional cap, 0 = no cap)
    max_island_area: int = 0

    # Padding around the canvas border — islands won't be placed within this
    canvas_border_padding: int = 30

    # After packing, crop output to this many pixels from the nearest island edge
    crop_padding: int = 5

    # Force-directed simulation settings
    max_iterations: int = 3000       # Max physics steps
    force_strength: float = 2.0      # How hard islands push each other apart
    attraction_strength: float = 3.0 # How hard islands pull toward neighbors (max dist)
    damping: float = 0.6             # Velocity damping per step (0-1)
    convergence_threshold: float = 0.1  # Stop when max movement < this (px)

    # Random seed (None = random each run)
    seed: Optional[int] = 42

    # Whether to allow island rotation (0, 90, 180, 270 degrees)
    allow_rotation: bool = True

    # Expansion: iteratively add copies of islands into large empty areas,
    # then re-run force sim, until no gaps larger than this remain.
    # Set to 0 to disable expansion.
    expand_min_gap_size: int = 40

    # Maximum expansion rounds (each round: find gaps, add copies, re-simulate)
    expand_max_rounds: int = 10

    # Debug: save intermediate steps
    debug_output: bool = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Island:
    id: int
    mask: np.ndarray          # Binary mask, tightly cropped (H x W, uint8 0/255)
    area: int                 # Pixel count
    original_centroid: Tuple[float, float]  # In the source image

    # Per-island distance constraints (assigned randomly from config ranges)
    my_min_dist: float = 20.0
    my_max_dist: float = 50.0

    # Sub-boxes: list of (local_x, local_y, w, h) relative to island top-left.
    # For single-box islands: [(0, 0, mask_w, mask_h)]
    # For split islands: multiple boxes from recursive splitting (up to 16).
    boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)

    # Placement state (set during packing)
    x: float = 0.0           # Top-left x on canvas
    y: float = 0.0           # Top-left y on canvas
    vx: float = 0.0          # Velocity for force sim
    vy: float = 0.0

    @property
    def w(self) -> int:
        return self.mask.shape[1]

    @property
    def h(self) -> int:
        return self.mask.shape[0]

    @property
    def cx(self) -> float:
        """Center x on canvas"""
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        """Center y on canvas"""
        return self.y + self.h / 2

    @property
    def approx_radius(self) -> float:
        """Bounding circle radius — used for broad-phase collision"""
        return math.sqrt(self.area / math.pi)


# ---------------------------------------------------------------------------
# Multi-box splitting
# ---------------------------------------------------------------------------

def _split_region(mask: np.ndarray, offset_x: int, offset_y: int,
                   splits_remaining: int) -> List[Tuple[int, int, int, int]]:
    """
    Recursively split a mask region along its longest dimension.
    Returns a list of (x, y, w, h) boxes in the island's local coordinate space.

    Each split halves the region and computes a tight bounding box for the
    actual pixels in each half, then recurses if splits remain.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []

    # Tight bounding box of pixels in this region
    bx = int(xs.min())
    by = int(ys.min())
    bw = int(xs.max()) - bx + 1
    bh = int(ys.max()) - by + 1

    if splits_remaining <= 0:
        return [(offset_x + bx, offset_y + by, bw, bh)]

    h, w = mask.shape

    # Split along the longest dimension of the MASK region (not the tight BB)
    if w >= h:
        mid = w // 2
        left_mask = mask[:, :mid]
        right_mask = mask[:, mid:]
        left_boxes = _split_region(left_mask, offset_x, offset_y, splits_remaining - 1)
        right_boxes = _split_region(right_mask, offset_x + mid, offset_y, splits_remaining - 1)
        return left_boxes + right_boxes
    else:
        mid = h // 2
        top_mask = mask[:mid, :]
        bottom_mask = mask[mid:, :]
        top_boxes = _split_region(top_mask, offset_x, offset_y, splits_remaining - 1)
        bottom_boxes = _split_region(bottom_mask, offset_x, offset_y + mid, splits_remaining - 1)
        return top_boxes + bottom_boxes


def compute_island_boxes(island: Island, config: PackerConfig) -> None:
    """
    Compute sub-boxes for an island. Large islands get recursively split
    along their longest dimension up to max_splits times. Each resulting
    region gets a tight bounding box of its actual pixels.

    With max_splits=4, a large island can have up to 16 sub-boxes.
    """
    h, w = island.mask.shape
    should_split = (
        config.split_area_threshold > 0
        and island.area >= config.split_area_threshold
        and max(w, h) / (min(w, h) + 1e-6) >= config.split_aspect_ratio
    )

    if not should_split:
        island.boxes = [(0, 0, w, h)]
        return

    island.boxes = _split_region(island.mask, 0, 0, config.max_splits)

    # Fallback: if splitting produced no boxes (shouldn't happen), use full BB
    if not island.boxes:
        island.boxes = [(0, 0, w, h)]


# ---------------------------------------------------------------------------
# Step 1: Extraction via Connected Component Labeling
# ---------------------------------------------------------------------------

def extract_islands(image_path: str, config: PackerConfig) -> List[Island]:
    """
    Load a B&W PNG and extract all distinct island blobs as Island objects.
    Uses OpenCV connected component analysis.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Threshold to strict binary (handles anti-aliasing / grey fringe)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Connected Component Labeling
    num_labels, label_map, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    islands = []
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < config.min_island_area:
            continue
        if config.max_island_area > 0 and area > config.max_island_area:
            continue

        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]

        roi = label_map[y:y+h, x:x+w]
        mask = np.where(roi == label_id, 255, 0).astype(np.uint8)

        cx, cy = centroids[label_id]

        island = Island(
            id=label_id,
            mask=mask,
            area=area,
            original_centroid=(cx, cy),
        )
        islands.append(island)

    print(f"Extracted {len(islands)} islands (filtered from {num_labels - 1} components)")
    return islands


# ---------------------------------------------------------------------------
# Step 2: Distance measurement utilities
# ---------------------------------------------------------------------------

def measure_cardinal_distances(
    islands: List[Island], canvas_size: Tuple[int, int]
) -> List[dict]:
    """
    For each island, cast rays from its center in 8 cardinal directions.
    Measure edge-to-edge distance to nearest other island in that direction.
    """
    W, H = canvas_size
    canvas = np.zeros((H, W), dtype=np.uint8)

    for isl in islands:
        x, y = int(round(isl.x)), int(round(isl.y))
        x2, y2 = min(x + isl.w, W), min(y + isl.h, H)
        mx2, my2 = x2 - x, y2 - y
        if mx2 > 0 and my2 > 0:
            canvas[y:y2, x:x2] = np.maximum(canvas[y:y2, x:x2], isl.mask[:my2, :mx2])

    label_canvas = np.zeros((H, W), dtype=np.int32)
    for idx, isl in enumerate(islands):
        x, y = int(round(isl.x)), int(round(isl.y))
        x2, y2 = min(x + isl.w, W), min(y + isl.h, H)
        mx2, my2 = x2 - x, y2 - y
        if mx2 > 0 and my2 > 0:
            mask_region = isl.mask[:my2, :mx2] > 0
            label_canvas[y:y2, x:x2][mask_region] = idx + 1

    directions = {
        "N":  ( 0, -1), "NE": ( 1, -1), "E":  ( 1,  0), "SE": ( 1,  1),
        "S":  ( 0,  1), "SW": (-1,  1), "W":  (-1,  0), "NW": (-1, -1),
    }

    results = []
    for idx, isl in enumerate(islands):
        label_id = idx + 1
        cx, cy = int(round(isl.cx)), int(round(isl.cy))
        entry = {"id": isl.id}

        for dir_name, (ddx, ddy) in directions.items():
            px, py = cx, cy
            exited = False
            while 0 <= px < W and 0 <= py < H:
                if label_canvas[py, px] != label_id:
                    exited = True
                    break
                px += ddx
                py += ddy

            if not exited:
                entry[dir_name] = float('inf')
                continue

            edge_x, edge_y = px - ddx, py - ddy

            found = False
            while 0 <= px < W and 0 <= py < H:
                if canvas[py, px] > 0 and label_canvas[py, px] != label_id:
                    found = True
                    break
                px += ddx
                py += ddy

            if not found:
                entry[dir_name] = float('inf')
            else:
                dist = math.sqrt((px - edge_x) ** 2 + (py - edge_y) ** 2)
                entry[dir_name] = dist

        results.append(entry)

    return results


def measure_all_distances(islands: List[Island], canvas_size: Tuple[int, int]) -> dict:
    """
    Measure nearest-neighbor edge-to-edge distances using multi-box BB gaps.
    For each pair of islands, compute the minimum gap across all box-to-box
    combinations.
    """
    n = len(islands)
    if n < 2:
        return {"min_gap": 0, "max_gap": 0, "mean_gap": 0, "islands_measured": 0}

    # Build flat arrays of all boxes with their parent island index
    all_box_x = []
    all_box_y = []
    all_box_w = []
    all_box_h = []
    box_owner = []  # which island each box belongs to

    for i, isl in enumerate(islands):
        for (bx, by, bw, bh) in isl.boxes:
            all_box_x.append(isl.x + bx)
            all_box_y.append(isl.y + by)
            all_box_w.append(float(bw))
            all_box_h.append(float(bh))
            box_owner.append(i)

    bxs = np.array(all_box_x)
    bys = np.array(all_box_y)
    bws = np.array(all_box_w)
    bhs = np.array(all_box_h)
    owners = np.array(box_owner)
    m = len(bxs)

    # Pairwise BB edge-to-edge gap (all boxes)
    gap_x = np.maximum(bxs[None, :] - (bxs[:, None] + bws[:, None]), 0) + \
            np.maximum(bxs[:, None] - (bxs[None, :] + bws[None, :]), 0)
    gap_y = np.maximum(bys[None, :] - (bys[:, None] + bhs[:, None]), 0) + \
            np.maximum(bys[:, None] - (bys[None, :] + bhs[None, :]), 0)
    edge_dist = np.sqrt(gap_x * gap_x + gap_y * gap_y)

    # Mask out same-island box pairs
    same_island = owners[:, None] == owners[None, :]
    edge_dist[same_island] = np.inf

    # For each island, find the minimum gap to any box of any OTHER island
    min_distances = np.zeros(n, dtype=np.float64)
    for i in range(n):
        my_boxes = np.where(owners == i)[0]
        other_boxes = np.where(owners != i)[0]
        if len(my_boxes) == 0 or len(other_boxes) == 0:
            min_distances[i] = np.inf
            continue
        min_distances[i] = edge_dist[np.ix_(my_boxes, other_boxes)].min()

    return {
        "min_gap": round(float(min_distances.min()), 1),
        "max_gap": round(float(min_distances.max()), 1),
        "mean_gap": round(float(min_distances.mean()), 1),
        "islands_measured": n,
    }


# ---------------------------------------------------------------------------
# Step 3: Force-Directed Packing
# ---------------------------------------------------------------------------

def initial_placement(islands: List[Island], config: PackerConfig) -> None:
    """
    Place islands in a scattered cluster near the canvas center.
    """
    rng = random.Random(config.seed)
    W, H = config.canvas_size

    avg_min = sum(config.min_edge_distance_range) / 2.0
    avg_max = sum(config.max_edge_distance_range) / 2.0
    target_gap = (avg_min + avg_max) / 2.0

    avg_size = math.sqrt(sum(isl.area for isl in islands) / len(islands))

    n = len(islands)
    cell_size = avg_size + target_gap
    cluster_area = n * cell_size * cell_size
    cluster_radius = math.sqrt(cluster_area / math.pi) * 0.85

    cx_canvas, cy_canvas = W / 2.0, H / 2.0

    for isl in islands:
        if config.allow_rotation:
            angle = rng.choice([0, 90, 180, 270])
            if angle != 0:
                isl.mask = np.rot90(isl.mask, k=angle // 90)

        # Compute sub-boxes after rotation (mask dimensions may have changed)
        compute_island_boxes(isl, config)

        while True:
            rx = rng.uniform(-cluster_radius, cluster_radius)
            ry = rng.uniform(-cluster_radius, cluster_radius)
            if rx * rx + ry * ry <= cluster_radius * cluster_radius:
                break

        isl.x = cx_canvas + rx - isl.w / 2.0
        isl.y = cy_canvas + ry - isl.h / 2.0
        isl.vx = 0.0
        isl.vy = 0.0

    for isl in islands:
        clamp_to_canvas(isl, config)


def clamp_to_canvas(isl: Island, config: PackerConfig) -> None:
    """Keep island within canvas bounds."""
    W, H = config.canvas_size
    pad = config.canvas_border_padding
    isl.x = max(pad, min(isl.x, W - isl.w - pad))
    isl.y = max(pad, min(isl.y, H - isl.h - pad))


def run_force_simulation(islands: List[Island], config: PackerConfig) -> None:
    """
    Force-directed layout using island-level bounding-box gap computation.
    All n x n pairwise operations are fully vectorized with NumPy.
    Uses sparse COO-style indexing for overlapping pairs to avoid dense n x n
    element-wise math on the full matrix.
    """
    n = len(islands)
    strength = config.force_strength
    damping = config.damping
    attr_str = config.attraction_strength

    W_canvas, H_canvas = config.canvas_size
    pad = config.canvas_border_padding

    m = sum(len(isl.boxes) for isl in islands)

    # Island-level arrays (float32 for speed — positions are pixel-scale)
    pos = np.empty((n, 2), dtype=np.float32)
    vel = np.zeros((n, 2), dtype=np.float32)
    dims = np.empty((n, 2), dtype=np.float32)
    min_dists = np.empty(n, dtype=np.float32)
    max_dists = np.empty(n, dtype=np.float32)

    for i, isl in enumerate(islands):
        pos[i, 0] = isl.x
        pos[i, 1] = isl.y
        dims[i, 0] = isl.w
        dims[i, 1] = isl.h
        min_dists[i] = isl.my_min_dist
        max_dists[i] = isl.my_max_dist

    half_dims = dims * 0.5
    use_max_dist = config.max_edge_distance_range[1] > 0

    # Pre-compute pair thresholds (static across iterations)
    half_min = min_dists * 0.5
    pair_min_x = half_min[:, None] + half_min[None, :]  # (n, n)
    pair_min_y = pair_min_x  # same values

    # Squared max_dists for comparison without sqrt
    max_dists_sq = max_dists * max_dists

    # Pre-allocate reusable n x n arrays
    gap_x = np.empty((n, n), dtype=np.float32)
    gap_y = np.empty((n, n), dtype=np.float32)
    tmp = np.empty((n, n), dtype=np.float32)
    forces = np.zeros((n, 2), dtype=np.float32)

    # Diagonal indices (static)
    diag_idx = np.arange(n)

    print(f"  {n} islands -> {m} boxes ({m - n} split)")

    for iteration in range(config.max_iterations):
        # BB edges
        x1 = pos[:, 0]
        y1 = pos[:, 1]
        x2 = x1 + dims[:, 0]
        y2 = y1 + dims[:, 1]

        # BB gap between all pairs (edge-to-edge, clamped >= 0)
        np.subtract(x1[None, :], x2[:, None], out=gap_x)
        np.maximum(gap_x, 0, out=gap_x)
        np.subtract(x1[:, None], x2[None, :], out=tmp)
        np.maximum(tmp, 0, out=tmp)
        gap_x += tmp

        np.subtract(y1[None, :], y2[:, None], out=gap_y)
        np.maximum(gap_y, 0, out=gap_y)
        np.subtract(y1[:, None], y2[None, :], out=tmp)
        np.maximum(tmp, 0, out=tmp)
        gap_y += tmp

        # --- Repulsion: sparse approach ---
        # Find pairs where gap < pair_min in BOTH axes
        overlapping = (gap_x < pair_min_x) & (gap_y < pair_min_y)
        overlapping[diag_idx, diag_idx] = False

        forces[:] = 0.0

        if overlapping.any():
            # Sparse indices of overlapping pairs
            oi, oj = np.where(overlapping)

            # Center-to-center direction for overlapping pairs only
            cx = x1 + half_dims[:, 0]
            cy = y1 + half_dims[:, 1]
            dx = cx[oi] - cx[oj]
            dy = cy[oi] - cy[oj]
            dist = np.sqrt(dx * dx + dy * dy)
            dist[dist < 1e-6] = 1e-6

            # Overlap amount
            sox = pair_min_x[oi, oj] - gap_x[oi, oj]
            soy = pair_min_y[oi, oj] - gap_y[oi, oj]
            mag = np.minimum(sox * soy, 1.0)

            # Accumulate forces (sparse -> dense via np.add.at)
            fx = (dx / dist) * mag * strength
            fy = (dy / dist) * mag * strength
            np.add.at(forces[:, 0], oi, fx)
            np.add.at(forces[:, 1], oi, fy)

        # --- Attraction (nearest neighbor, squared distance) ---
        if use_max_dist:
            # Squared edge distance to avoid sqrt
            np.multiply(gap_x, gap_x, out=tmp)
            edge_dist_sq = tmp.copy()
            np.multiply(gap_y, gap_y, out=tmp)
            edge_dist_sq += tmp
            edge_dist_sq[diag_idx, diag_idx] = np.inf

            nn_idx = edge_dist_sq.argmin(axis=1)
            nn_dist_sq = edge_dist_sq[diag_idx, nn_idx]

            needs_pull = nn_dist_sq > max_dists_sq
            if needs_pull.any():
                pull_idx = np.where(needs_pull)[0]
                nn_of_pull = nn_idx[pull_idx]

                cx = x1 + half_dims[:, 0]
                cy = y1 + half_dims[:, 1]
                attr_dx = cx[nn_of_pull] - cx[pull_idx]
                attr_dy = cy[nn_of_pull] - cy[pull_idx]
                attr_dist = np.sqrt(attr_dx * attr_dx + attr_dy * attr_dy)
                attr_dist[attr_dist < 1e-6] = 1e-6

                nn_edge = np.sqrt(nn_dist_sq[pull_idx])
                excess = nn_edge - max_dists[pull_idx]
                attr_force = excess / (max_dists[pull_idx] + 1e-6)

                forces[pull_idx, 0] += (attr_dx / attr_dist) * attr_force * attr_str * strength
                forces[pull_idx, 1] += (attr_dy / attr_dist) * attr_force * attr_str * strength

        # --- Apply forces ---
        vel = (vel + forces) * damping
        pos += vel

        # Clamp to canvas
        np.clip(pos[:, 0], pad, W_canvas - dims[:, 0] - pad, out=pos[:, 0])
        np.clip(pos[:, 1], pad, H_canvas - dims[:, 1] - pad, out=pos[:, 1])

        max_move = float(np.abs(vel).sum(axis=1).max())

        if iteration % 100 == 0:
            print(f"  Iteration {iteration:4d} | max_move={max_move:.3f}")

        if max_move < config.convergence_threshold:
            print(f"  Converged at iteration {iteration}")
            break

    # Write positions back to Island objects
    for i, isl in enumerate(islands):
        isl.x = float(pos[i, 0])
        isl.y = float(pos[i, 1])
        isl.vx = float(vel[i, 0])
        isl.vy = float(vel[i, 1])



# ---------------------------------------------------------------------------
# Step 4: Render output
# ---------------------------------------------------------------------------

def find_empty_regions(islands: List[Island], config: PackerConfig,
                       min_size: int,
                       bounds: Optional[Tuple[int, int, int, int]] = None,
                       ) -> List[Tuple[int, int, int, int]]:
    """
    Render the current island layout and find rectangular empty regions
    at least min_size x min_size within the given bounds.

    bounds: (x1, y1, x2, y2) to limit scanning. If None, uses the content BB.
    Returns list of (x, y, w, h) regions.
    """
    W, H = config.canvas_size
    canvas = np.zeros((H, W), dtype=np.uint8)
    for isl in islands:
        x, y = int(round(isl.x)), int(round(isl.y))
        x2, y2 = min(x + isl.w, W), min(y + isl.h, H)
        mx2, my2 = x2 - x, y2 - y
        if mx2 > 0 and my2 > 0:
            canvas[y:y2, x:x2] = np.maximum(canvas[y:y2, x:x2], isl.mask[:my2, :mx2])

    if bounds is not None:
        cx1, cy1, cx2, cy2 = bounds
    else:
        ys, xs = np.where(canvas > 0)
        if len(xs) == 0:
            return []
        cx1, cy1 = int(xs.min()), int(ys.min())
        cx2, cy2 = int(xs.max()), int(ys.max())

    step = max(1, min_size // 2)
    regions = []

    for sy in range(cy1, cy2 - min_size + 1, step):
        for sx in range(cx1, cx2 - min_size + 1, step):
            region = canvas[sy:sy + min_size, sx:sx + min_size]
            if not region.any():
                regions.append((sx, sy, min_size, min_size))
                # Mark this area so we don't find overlapping regions
                canvas[sy:sy + min_size, sx:sx + min_size] = 1

    return regions


def expand_into_gaps(islands: List[Island], source_islands: List[Island],
                     config: PackerConfig, rng: random.Random,
                     bounds: Optional[Tuple[int, int, int, int]] = None,
                     ) -> int:
    """
    Find large empty regions in the current layout (within bounds) and place
    randomly-rotated copies of source islands into them. The copies are added
    to the islands list (mutated in place) so a subsequent force sim can
    settle them.

    Returns the number of copies added.
    """
    min_size = config.expand_min_gap_size
    if min_size <= 0:
        return 0

    regions = find_empty_regions(islands, config, min_size, bounds)
    if not regions:
        return 0

    # Build pool of source islands that could fit in min_size x min_size
    # (checking both orientations since we randomly rotate)
    pool = [isl for isl in source_islands
            if min(isl.w, isl.h) <= min_size and max(isl.w, isl.h) <= min_size * 2]
    if not pool:
        # Fallback: any island whose smallest dimension fits
        pool = [isl for isl in source_islands if min(isl.w, isl.h) <= min_size]
    if not pool:
        return 0

    min_lo, min_hi = config.min_edge_distance_range
    max_lo, max_hi = config.max_edge_distance_range

    added = 0
    for (rx, ry, rw, rh) in regions:
        # Pick a random source island
        src = rng.choice(pool)

        # Randomly rotate
        mask = src.mask.copy()
        angle = rng.choice([0, 90, 180, 270])
        if angle != 0:
            mask = np.rot90(mask, k=angle // 90)

        h, w = mask.shape

        # Check if it fits in the region
        if w > rw or h > rh:
            continue

        copy = Island(
            id=src.id,
            mask=mask,
            area=src.area,
            original_centroid=src.original_centroid,
            my_min_dist=rng.uniform(min_lo, min_hi),
            my_max_dist=rng.uniform(max_lo, max_hi),
        )
        compute_island_boxes(copy, config)

        # Place centered in the region
        copy.x = float(rx + (rw - w) // 2)
        copy.y = float(ry + (rh - h) // 2)
        copy.vx = 0.0
        copy.vy = 0.0

        islands.append(copy)
        added += 1

    return added


def fill_gaps(islands: List[Island], config: PackerConfig) -> Tuple[int, List[Island]]:
    """
    After force-directed simulation, scan the rendered canvas for empty regions
    *between* existing islands and place COPIES of small islands into those gaps.
    Originals stay in place.

    A gap is valid only if the surrounding area (2x the box size) contains at least
    some island pixels — this prevents filling empty space at the edges of the
    cluster. The placed copy also gets a padding margin around it so subsequent
    copies don't touch it.

    Scans with a sliding box of size max_edge_distance_range[1] x max_edge_distance_range[1],
    stepping by half of min_edge_distance_range[0].

    Small islands are cycled through round-robin so the same island can be copied
    multiple times if there are more gaps than small islands.

    Returns (number_of_copies_placed, list_of_new_copy_islands).
    """
    max_hi = config.max_edge_distance_range[1]
    min_lo = config.min_edge_distance_range[0]

    box_size = max_hi
    step = max(1, min_lo // 2)
    # Minimum gap padding around each placed copy
    min_gap_pad = min_lo

    if box_size <= 0:
        return 0, []

    # Find small islands that fit inside box_size x box_size
    small_pool = [isl for isl in islands if isl.w <= box_size and isl.h <= box_size]

    if not small_pool:
        return 0, []

    # Sort by area descending (prefer larger fills first)
    small_pool.sort(key=lambda isl: isl.area, reverse=True)

    # Render canvas with ALL islands to find current occupied pixels
    W, H = config.canvas_size
    canvas = np.zeros((H, W), dtype=np.uint8)
    for isl in islands:
        x, y = int(round(isl.x)), int(round(isl.y))
        x2, y2 = min(x + isl.w, W), min(y + isl.h, H)
        mx2, my2 = x2 - x, y2 - y
        if mx2 > 0 and my2 > 0:
            canvas[y:y2, x:x2] = np.maximum(canvas[y:y2, x:x2], isl.mask[:my2, :mx2])

    # Find the content bounding box (tight) to limit scanning
    ys, xs = np.where(canvas > 0)
    if len(xs) == 0:
        return 0, []
    content_x1 = int(xs.min())
    content_y1 = int(ys.min())
    content_x2 = int(xs.max())
    content_y2 = int(ys.max())

    # Check radius: a gap is only valid if nearby pixels contain island content
    check_radius = box_size * 2

    copies_placed = 0
    new_islands = []
    pool_idx = 0

    # Padded check box: the placed island + min_gap_pad on each side must be black
    padded_size = box_size + 2 * min_gap_pad

    # Scan with sliding box (within content bounds only)
    for sy in range(content_y1, content_y2 - box_size + 1, step):
        for sx in range(content_x1, content_x2 - box_size + 1, step):
            # Check padded region is completely black (ensures spacing)
            px1 = max(0, sx - min_gap_pad)
            py1 = max(0, sy - min_gap_pad)
            px2 = min(W, sx + box_size + min_gap_pad)
            py2 = min(H, sy + box_size + min_gap_pad)
            padded_region = canvas[py1:py2, px1:px2]
            if padded_region.any():
                continue

            # Verify this is a gap BETWEEN islands, not empty fringe
            # Check a larger surrounding area for any island content
            cx1 = max(0, sx - check_radius)
            cy1 = max(0, sy - check_radius)
            cx2 = min(W, sx + box_size + check_radius)
            cy2 = min(H, sy + box_size + check_radius)
            surround = canvas[cy1:cy2, cx1:cx2]
            if not surround.any():
                continue

            # Cycle through small islands round-robin to find one that fits
            for attempt in range(len(small_pool)):
                src = small_pool[(pool_idx + attempt) % len(small_pool)]
                if src.w <= box_size and src.h <= box_size:
                    # Create a copy island
                    copy = Island(
                        id=src.id,
                        mask=src.mask,
                        area=src.area,
                        original_centroid=src.original_centroid,
                        my_min_dist=src.my_min_dist,
                        my_max_dist=src.my_max_dist,
                        boxes=[(0, 0, src.w, src.h)],
                    )
                    copy.x = float(sx + (box_size - src.w) // 2)
                    copy.y = float(sy + (box_size - src.h) // 2)

                    # Paint onto canvas (so subsequent scans see this copy)
                    x, y = int(round(copy.x)), int(round(copy.y))
                    x2, y2 = min(x + copy.w, W), min(y + copy.h, H)
                    mx2, my2 = x2 - x, y2 - y
                    if mx2 > 0 and my2 > 0:
                        canvas[y:y2, x:x2] = np.maximum(
                            canvas[y:y2, x:x2], copy.mask[:my2, :mx2]
                        )

                    new_islands.append(copy)
                    copies_placed += 1
                    pool_idx = (pool_idx + attempt + 1) % len(small_pool)
                    break

    return copies_placed, new_islands


def render_packed_canvas(islands: List[Island], config: PackerConfig) -> np.ndarray:
    """Paint all placed islands onto a black canvas and return it."""
    W, H = config.canvas_size
    canvas = np.zeros((H, W), dtype=np.uint8)

    for isl in islands:
        x, y = int(round(isl.x)), int(round(isl.y))
        x2, y2 = min(x + isl.w, W), min(y + isl.h, H)
        isl_x2, isl_y2 = x2 - x, y2 - y

        if isl_x2 <= 0 or isl_y2 <= 0:
            continue

        region = isl.mask[:isl_y2, :isl_x2]
        canvas[y:y2, x:x2] = np.maximum(canvas[y:y2, x:x2], region)

    return canvas


def crop_to_content(canvas: np.ndarray, padding: int = 5) -> np.ndarray:
    """
    Crop the canvas to the bounding box of all white pixels plus padding.
    Works with both single-channel (H,W) and multi-channel (H,W,C) arrays.
    """
    if canvas.ndim == 3:
        # Multi-channel: find any non-zero pixel across all channels
        any_nonzero = np.any(canvas > 0, axis=2)
        ys, xs = np.where(any_nonzero)
    else:
        ys, xs = np.where(canvas > 0)

    if len(xs) == 0:
        return canvas  # nothing to crop

    x_min = max(0, int(xs.min()) - padding)
    x_max = min(canvas.shape[1], int(xs.max()) + 1 + padding)
    y_min = max(0, int(ys.min()) - padding)
    y_max = min(canvas.shape[0], int(ys.max()) + 1 + padding)

    return canvas[y_min:y_max, x_min:x_max]


def render_debug_boxes(islands: List[Island], config: PackerConfig) -> np.ndarray:
    """
    Render a color debug image showing islands (white) with their
    sub-bounding boxes overlaid as colored rectangles.

    - Single-box islands: white only, no box outline
    - Multi-box islands: white pixels + colored box outlines (green)

    Returns a BGR image (3-channel).
    """
    W, H = config.canvas_size
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Draw islands in white
    for isl in islands:
        x, y = int(round(isl.x)), int(round(isl.y))
        x2, y2 = min(x + isl.w, W), min(y + isl.h, H)
        isl_x2, isl_y2 = x2 - x, y2 - y
        if isl_x2 <= 0 or isl_y2 <= 0:
            continue
        region = isl.mask[:isl_y2, :isl_x2]
        for c in range(3):
            canvas[y:y2, x:x2, c] = np.maximum(canvas[y:y2, x:x2, c], region)

    # Draw bounding box outlines for split islands
    for isl in islands:
        if len(isl.boxes) <= 1:
            continue
        ix, iy = int(round(isl.x)), int(round(isl.y))
        for (bx, by, bw, bh) in isl.boxes:
            # Box corners in canvas space
            rx1 = max(0, ix + bx)
            ry1 = max(0, iy + by)
            rx2 = min(W - 1, ix + bx + bw - 1)
            ry2 = min(H - 1, iy + by + bh - 1)
            # Draw rectangle outline in green (BGR: 0, 255, 0)
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (0, 255, 0), 1)

    return canvas


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def pack_islands(
    input_path: str,
    output_path: str,
    config: Optional[PackerConfig] = None,
) -> dict:
    """
    Full pipeline: extract -> place -> simulate -> render -> crop -> measure.
    Returns a stats dict with distance measurements.
    """
    if config is None:
        config = PackerConfig()

    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

    min_lo, min_hi = config.min_edge_distance_range
    max_lo, max_hi = config.max_edge_distance_range

    print(f"=== Island Packer ===")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Canvas: {config.canvas_size[0]}x{config.canvas_size[1]}")
    print(f"Min edge distance range: {min_lo}-{min_hi}px")
    if max_hi > 0:
        print(f"Max edge distance range: {max_lo}-{max_hi}px (8-cardinal)")
    if config.split_area_threshold > 0:
        print(f"Multi-box split: area >= {config.split_area_threshold} AND aspect >= {config.split_aspect_ratio}, max_splits={config.max_splits}")
    print(f"Crop padding: {config.crop_padding}px")
    print()

    # 1. Extract
    t0 = time.time()
    islands = extract_islands(input_path, config)
    print(f"Extraction: {time.time()-t0:.2f}s | {len(islands)} islands\n")

    if not islands:
        raise ValueError("No islands found -- check min_island_area or image threshold")

    # Shuffle randomly
    rng = random.Random(config.seed)
    rng.shuffle(islands)

    # Assign per-island random min/max distances
    for isl in islands:
        isl.my_min_dist = rng.uniform(min_lo, min_hi)
        isl.my_max_dist = rng.uniform(max_lo, max_hi)

    print("Per-island distance stats:")
    all_mins = [isl.my_min_dist for isl in islands]
    all_maxs = [isl.my_max_dist for isl in islands]
    print(f"  min_dist range: {min(all_mins):.1f} - {max(all_mins):.1f} (mean {sum(all_mins)/len(all_mins):.1f})")
    print(f"  max_dist range: {min(all_maxs):.1f} - {max(all_maxs):.1f} (mean {sum(all_maxs)/len(all_maxs):.1f})")
    print()

    # 2. Initial placement (scattered, not grid)
    print("Placing islands (scattered cluster)...")
    initial_placement(islands, config)

    split_count = sum(1 for isl in islands if len(isl.boxes) > 1)
    total_boxes = sum(len(isl.boxes) for isl in islands)
    max_boxes_per = max(len(isl.boxes) for isl in islands)
    print(f"  {split_count} islands split (max {max_boxes_per} boxes/island, {total_boxes} total boxes)")
    print("Done.\n")

    # 3. Force-directed simulation
    print("Running force-directed simulation...")
    t1 = time.time()
    run_force_simulation(islands, config)
    print(f"Simulation: {time.time()-t1:.2f}s\n")

    # 3b. Expansion loop: find large empty regions, add rotated copies, re-simulate
    #     Constrained to the initial cluster bounding box so it doesn't grow outward.
    source_islands = list(islands)  # snapshot of originals for copying
    total_expansion_copies = 0
    if config.expand_min_gap_size > 0:
        t_expand = time.time()

        # Capture the initial cluster bounding box (with some margin)
        margin = config.expand_min_gap_size
        all_x1 = min(isl.x for isl in islands)
        all_y1 = min(isl.y for isl in islands)
        all_x2 = max(isl.x + isl.w for isl in islands)
        all_y2 = max(isl.y + isl.h for isl in islands)
        expand_bounds = (
            max(0, int(all_x1) - margin),
            max(0, int(all_y1) - margin),
            min(config.canvas_size[0], int(all_x2) + margin),
            min(config.canvas_size[1], int(all_y2) + margin),
        )
        print(f"  Expansion bounds: ({expand_bounds[0]},{expand_bounds[1]})-({expand_bounds[2]},{expand_bounds[3]})")

        # Use a shorter sim for expansion rounds
        expand_cfg = PackerConfig(**{
            f.name: getattr(config, f.name)
            for f in config.__dataclass_fields__.values()
        })
        expand_cfg.max_iterations = 300
        expand_cfg.convergence_threshold = 3.0  # stop earlier, cluster is already mostly settled

        for round_num in range(config.expand_max_rounds):
            added = expand_into_gaps(islands, source_islands, config, rng, expand_bounds)
            if added == 0:
                print(f"  Expansion: no gaps >= {config.expand_min_gap_size}px remain after round {round_num + 1}")
                break
            total_expansion_copies += added
            print(f"  Expansion round {round_num + 1}: added {added} copies ({len(islands)} total), re-simulating...")
            run_force_simulation(islands, expand_cfg)
            if added < 15:
                print(f"  Expansion: diminishing returns ({added} < 15), stopping")
                break
        print(f"Expansion: {total_expansion_copies} copies added in {time.time()-t_expand:.2f}s\n")

    # 3c. Gap-filling pass: disabled
    gap_filled = 0

    # 4. Render
    canvas = render_packed_canvas(islands, config)

    # 5. Crop to content
    cropped = crop_to_content(canvas, config.crop_padding)
    cv2.imwrite(output_path, cropped)
    print(f"Saved: {output_path}")
    print(f"  Full canvas: {canvas.shape[1]}x{canvas.shape[0]}")
    print(f"  Cropped:     {cropped.shape[1]}x{cropped.shape[0]}")

    # 5b. Debug bounding box image
    if split_count > 0:
        debug_canvas = render_debug_boxes(islands, config)
        debug_cropped = crop_to_content(debug_canvas, config.crop_padding)
        # Build debug filename: insert _boxes before extension
        import os
        base, ext = os.path.splitext(output_path)
        debug_path = f"{base}_boxes{ext}"
        cv2.imwrite(debug_path, debug_cropped)
        print(f"Saved debug boxes: {debug_path}")

    # 6. Measure distances (multi-box BB nearest neighbor)
    print("\nMeasuring edge-to-edge distances (multi-box BB)...")
    stats = measure_all_distances(islands, config.canvas_size)
    stats["total_islands"] = len(islands)
    stats["expansion_copies"] = total_expansion_copies
    stats["gap_filled_copies"] = gap_filled
    stats["split_islands"] = split_count
    stats["total_boxes"] = total_boxes
    stats["canvas"] = f"{config.canvas_size[0]}x{config.canvas_size[1]}"
    stats["cropped"] = f"{cropped.shape[1]}x{cropped.shape[0]}"
    stats["min_edge_distance_range"] = f"{min_lo}-{min_hi}"
    stats["max_edge_distance_range"] = f"{max_lo}-{max_hi}"

    # 7. Measure 8-cardinal-direction distances
    if max_hi > 0:
        print("Measuring 8-cardinal-direction distances...")
        cardinal = measure_cardinal_distances(islands, config.canvas_size)

        dir_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        violations = 0
        min_cardinal_nearest = float('inf')
        max_cardinal_nearest = 0.0
        cardinal_nearests = []

        for idx, entry in enumerate(cardinal):
            finite_dists = [entry[d] for d in dir_names if entry[d] != float('inf')]
            if finite_dists:
                nearest = min(finite_dists)
                cardinal_nearests.append(nearest)
                min_cardinal_nearest = min(min_cardinal_nearest, nearest)
                max_cardinal_nearest = max(max_cardinal_nearest, nearest)

                isl_max = islands[idx].my_max_dist
                all_exceed = all(entry[d] > isl_max for d in dir_names)
                if all_exceed:
                    violations += 1

        stats["cardinal_min_nearest"] = round(min_cardinal_nearest, 1)
        stats["cardinal_max_nearest"] = round(max_cardinal_nearest, 1)
        stats["cardinal_mean_nearest"] = round(
            float(np.mean(cardinal_nearests)) if cardinal_nearests else 0, 1
        )
        stats["cardinal_max_dist_violations"] = violations

    print("\n=== Results ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.1f}")
        else:
            print(f"  {k}: {v}")

    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    input_file = sys.argv[1] if len(sys.argv) > 1 else "IslandMaskBWV11.png"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "IslandMaskPacked.png"

    # Parse optional CLI args: min_lo min_hi max_lo max_hi canvas_w canvas_h
    min_lo = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    min_hi = int(sys.argv[4]) if len(sys.argv) > 4 else 13
    max_lo = int(sys.argv[5]) if len(sys.argv) > 5 else 15
    max_hi = int(sys.argv[6]) if len(sys.argv) > 6 else 20
    canvas_w = int(sys.argv[7]) if len(sys.argv) > 7 else 4096
    canvas_h = int(sys.argv[8]) if len(sys.argv) > 8 else 4096

    cfg = PackerConfig(
        canvas_size=(canvas_w, canvas_h),
        min_edge_distance_range=(min_lo, min_hi),
        max_edge_distance_range=(max_lo, max_hi),
        min_island_area=50,
        canvas_border_padding=30,
        crop_padding=5,
        split_area_threshold=2561,
        split_aspect_ratio=1.0,
        max_splits=2,
        expand_min_gap_size=40,
        expand_max_rounds=10,
        max_iterations=3000,
        force_strength=2.0,
        attraction_strength=3.0,
        damping=0.6,
        convergence_threshold=0.1,
        allow_rotation=True,
        seed=42,
    )

    stats = pack_islands(input_file, output_file, cfg)
