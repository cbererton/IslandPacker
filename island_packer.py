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
    min_edge_distance_range: Tuple[int, int] = (20, 35)

    # Per-island maximum edge distance is randomly chosen from this range.
    # Islands whose nearest neighbor exceeds their max distance get pulled in.
    # Set both to 0 to disable max distance enforcement.
    max_edge_distance_range: Tuple[int, int] = (45, 70)

    # Islands with area >= this threshold AND aspect ratio >= split_aspect_ratio
    # get split into 2 bounding boxes along their longest axis.
    # This lets other islands nestle closer to elongated shapes.
    # 0 = disabled (no splitting)
    split_area_threshold: int = 2000

    # Minimum aspect ratio (max_dim / min_dim) to trigger splitting.
    # Only islands that are both large AND elongated get split.
    split_aspect_ratio: float = 1.8

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
    # For split islands: two boxes covering each half's tight bounding box.
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

def compute_island_boxes(island: Island, config: PackerConfig) -> None:
    """
    Compute sub-boxes for an island. Large elongated islands get split into
    two bounding boxes along their longest dimension. Each box is a tight
    bounding rect of the actual pixels in that half.
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

    # Split along the longest dimension
    if w >= h:
        # Split horizontally: left half and right half
        mid = w // 2
        left_mask = island.mask[:, :mid]
        right_mask = island.mask[:, mid:]

        island.boxes = []
        for offset_x, half_mask in [(0, left_mask), (mid, right_mask)]:
            # Find tight bounding box of actual pixels in this half
            ys, xs = np.where(half_mask > 0)
            if len(xs) == 0:
                continue
            bx = int(xs.min())
            by = int(ys.min())
            bw = int(xs.max()) - bx + 1
            bh = int(ys.max()) - by + 1
            island.boxes.append((offset_x + bx, by, bw, bh))
    else:
        # Split vertically: top half and bottom half
        mid = h // 2
        top_mask = island.mask[:mid, :]
        bottom_mask = island.mask[mid:, :]

        island.boxes = []
        for offset_y, half_mask in [(0, top_mask), (mid, bottom_mask)]:
            ys, xs = np.where(half_mask > 0)
            if len(xs) == 0:
                continue
            bx = int(xs.min())
            by = int(ys.min())
            bw = int(xs.max()) - bx + 1
            bh = int(ys.max()) - by + 1
            island.boxes.append((bx, offset_y + by, bw, bh))

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
    Force-directed layout with per-island distance constraints and multi-box
    overlap detection.

    Each island may have 1 or 2 sub-boxes. Repulsion checks overlap between
    all box pairs across different islands. Forces are accumulated per island.

    Attraction uses per-island min edge distance across all box pairs to find
    nearest neighbor.
    """
    n = len(islands)
    strength = config.force_strength
    damping = config.damping
    attr_str = config.attraction_strength

    W_canvas, H_canvas = config.canvas_size
    pad = config.canvas_border_padding

    # --- Build box-level arrays ---
    # Each island has 1 or 2 boxes. We flatten all boxes into arrays
    # and track which island each box belongs to.
    box_local_x = []  # box x offset relative to island top-left
    box_local_y = []
    box_w = []
    box_h = []
    box_island = []  # index of parent island

    for i, isl in enumerate(islands):
        for (bx, by, bw, bh) in isl.boxes:
            box_local_x.append(float(bx))
            box_local_y.append(float(by))
            box_w.append(float(bw))
            box_h.append(float(bh))
            box_island.append(i)

    blx = np.array(box_local_x, dtype=np.float64)  # (m,)
    bly = np.array(box_local_y, dtype=np.float64)
    bw_arr = np.array(box_w, dtype=np.float64)
    bh_arr = np.array(box_h, dtype=np.float64)
    bowner = np.array(box_island, dtype=np.int64)
    m = len(blx)

    # Pre-compute same-island mask (m x m boolean)
    same_island = bowner[:, None] == bowner[None, :]

    # Island-level arrays
    pos = np.zeros((n, 2), dtype=np.float64)
    vel = np.zeros((n, 2), dtype=np.float64)
    dims = np.zeros((n, 2), dtype=np.float64)
    min_dists = np.zeros(n, dtype=np.float64)
    max_dists = np.zeros(n, dtype=np.float64)

    for i, isl in enumerate(islands):
        pos[i, 0] = isl.x
        pos[i, 1] = isl.y
        dims[i, 0] = isl.w
        dims[i, 1] = isl.h
        min_dists[i] = isl.my_min_dist
        max_dists[i] = isl.my_max_dist

    half_dims = dims / 2.0

    use_max_dist = config.max_edge_distance_range[1] > 0

    # Per-box min_dist (inherited from parent island)
    box_min_dist = min_dists[bowner]  # (m,)
    half_box_min = box_min_dist / 2.0

    print(f"  {n} islands -> {m} boxes ({m - n} split)")

    for iteration in range(config.max_iterations):
        # --- Compute absolute box positions from island positions ---
        # box_abs_x[b] = pos[owner[b], 0] + box_local_x[b]
        box_abs_x = pos[bowner, 0] + blx  # (m,)
        box_abs_y = pos[bowner, 1] + bly

        # Island centers (for force direction)
        centers = pos + half_dims  # (n, 2)

        # --- Multi-box repulsion ---
        # Expanded bounding boxes per box (padded by half min_dist)
        bb_x1 = box_abs_x - half_box_min
        bb_y1 = box_abs_y - half_box_min
        bb_x2 = box_abs_x + bw_arr + half_box_min
        bb_y2 = box_abs_y + bh_arr + half_box_min

        # Pairwise overlap (m x m)
        ox = np.minimum(bb_x2[:, None], bb_x2[None, :]) - np.maximum(bb_x1[:, None], bb_x1[None, :])
        oy = np.minimum(bb_y2[:, None], bb_y2[None, :]) - np.maximum(bb_y1[:, None], bb_y1[None, :])

        overlapping = (ox > 0) & (oy > 0)
        overlapping[same_island] = False  # ignore boxes from the same island

        # Direction: from island center i toward island center j
        # Map box pairs back to island pairs for direction
        owner_i = bowner[:, None].repeat(m, axis=1)  # (m, m)
        owner_j = bowner[None, :].repeat(m, axis=0)

        dx_ij = centers[owner_i, 0] - centers[owner_j, 0]  # (m, m)
        dy_ij = centers[owner_i, 1] - centers[owner_j, 1]
        dist_ij = np.sqrt(dx_ij * dx_ij + dy_ij * dy_ij) + 1e-6

        # Force magnitude based on overlap area
        overlap_area = np.maximum(ox, 0) * np.maximum(oy, 0)
        exp_w = bw_arr + box_min_dist
        exp_h = bh_arr + box_min_dist
        exp_area = exp_w * exp_h
        min_area = np.minimum(exp_area[:, None], exp_area[None, :])
        force_mag = np.minimum(overlap_area / (min_area + 1e-6), 1.0)
        force_mag[~overlapping] = 0.0

        # Force vectors at box level
        rep_fx = (dx_ij / dist_ij) * force_mag
        rep_fy = (dy_ij / dist_ij) * force_mag

        # Accumulate forces per island (sum over all boxes belonging to each island)
        forces = np.zeros((n, 2), dtype=np.float64)
        # Sum forces on all boxes belonging to island i
        box_force_x = rep_fx.sum(axis=1)  # (m,) — total force on each box
        box_force_y = rep_fy.sum(axis=1)
        np.add.at(forces[:, 0], bowner, box_force_x)
        np.add.at(forces[:, 1], bowner, box_force_y)
        forces *= strength

        # --- Attraction (nearest neighbor using multi-box edge distance) ---
        if use_max_dist:
            # BB edge-to-edge gap between all box pairs
            gap_x = np.maximum(box_abs_x[None, :] - (box_abs_x[:, None] + bw_arr[:, None]), 0) + \
                    np.maximum(box_abs_x[:, None] - (box_abs_x[None, :] + bw_arr[None, :]), 0)
            gap_y = np.maximum(box_abs_y[None, :] - (box_abs_y[:, None] + bh_arr[:, None]), 0) + \
                    np.maximum(box_abs_y[:, None] - (box_abs_y[None, :] + bh_arr[None, :]), 0)
            box_edge_dist = np.sqrt(gap_x * gap_x + gap_y * gap_y)
            box_edge_dist[same_island] = np.inf

            # For each island, find nearest other island (min edge dist across all box pairs)
            nn_edge = np.full(n, np.inf, dtype=np.float64)
            nn_idx = np.zeros(n, dtype=np.int64)

            for i in range(n):
                my_boxes = np.where(bowner == i)[0]
                other_boxes = np.where(bowner != i)[0]
                if len(my_boxes) == 0 or len(other_boxes) == 0:
                    continue
                sub = box_edge_dist[np.ix_(my_boxes, other_boxes)]
                min_val = sub.min()
                nn_edge[i] = min_val
                # Find which island the nearest box belongs to
                flat_idx = sub.argmin()
                other_box_idx = other_boxes[flat_idx % len(other_boxes)]
                nn_idx[i] = bowner[other_box_idx]

            needs_pull = nn_edge > max_dists
            if needs_pull.any():
                pull_idx = np.where(needs_pull)[0]
                nn_of_pull = nn_idx[pull_idx]

                attr_dx = centers[nn_of_pull, 0] - centers[pull_idx, 0]
                attr_dy = centers[nn_of_pull, 1] - centers[pull_idx, 1]
                attr_dist = np.sqrt(attr_dx * attr_dx + attr_dy * attr_dy) + 1e-6

                excess = nn_edge[pull_idx] - max_dists[pull_idx]
                attr_force = excess / (max_dists[pull_idx] + 1e-6)

                forces[pull_idx, 0] += (attr_dx / attr_dist) * attr_force * attr_str * strength
                forces[pull_idx, 1] += (attr_dy / attr_dist) * attr_force * attr_str * strength

        # --- Apply forces ---
        vel = (vel + forces) * damping
        pos += vel

        # Clamp to canvas
        pos[:, 0] = np.clip(pos[:, 0], pad, W_canvas - dims[:, 0] - pad)
        pos[:, 1] = np.clip(pos[:, 1], pad, H_canvas - dims[:, 1] - pad)

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
    """
    ys, xs = np.where(canvas > 0)
    if len(xs) == 0:
        return canvas  # nothing to crop

    x_min = max(0, int(xs.min()) - padding)
    x_max = min(canvas.shape[1], int(xs.max()) + 1 + padding)
    y_min = max(0, int(ys.min()) - padding)
    y_max = min(canvas.shape[0], int(ys.max()) + 1 + padding)

    return canvas[y_min:y_max, x_min:x_max]


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
        print(f"Multi-box split: area >= {config.split_area_threshold} AND aspect >= {config.split_aspect_ratio}")
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
    print(f"  {split_count} islands split into 2 boxes ({total_boxes} total boxes)")
    print("Done.\n")

    # 3. Force-directed simulation
    print("Running force-directed simulation...")
    t1 = time.time()
    run_force_simulation(islands, config)
    print(f"Simulation: {time.time()-t1:.2f}s\n")

    # 4. Render
    canvas = render_packed_canvas(islands, config)

    # 5. Crop to content
    cropped = crop_to_content(canvas, config.crop_padding)
    cv2.imwrite(output_path, cropped)
    print(f"Saved: {output_path}")
    print(f"  Full canvas: {canvas.shape[1]}x{canvas.shape[0]}")
    print(f"  Cropped:     {cropped.shape[1]}x{cropped.shape[0]}")

    # 6. Measure distances (multi-box BB nearest neighbor)
    print("\nMeasuring edge-to-edge distances (multi-box BB)...")
    stats = measure_all_distances(islands, config.canvas_size)
    stats["total_islands"] = len(islands)
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
    min_lo = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    min_hi = int(sys.argv[4]) if len(sys.argv) > 4 else 35
    max_lo = int(sys.argv[5]) if len(sys.argv) > 5 else 45
    max_hi = int(sys.argv[6]) if len(sys.argv) > 6 else 70
    canvas_w = int(sys.argv[7]) if len(sys.argv) > 7 else 4096
    canvas_h = int(sys.argv[8]) if len(sys.argv) > 8 else 4096

    cfg = PackerConfig(
        canvas_size=(canvas_w, canvas_h),
        min_edge_distance_range=(min_lo, min_hi),
        max_edge_distance_range=(max_lo, max_hi),
        min_island_area=50,
        canvas_border_padding=30,
        crop_padding=5,
        split_area_threshold=2000,
        split_aspect_ratio=1.8,
        max_iterations=3000,
        force_strength=2.0,
        attraction_strength=3.0,
        damping=0.6,
        convergence_threshold=0.1,
        allow_rotation=True,
        seed=42,
    )

    stats = pack_islands(input_file, output_file, cfg)
