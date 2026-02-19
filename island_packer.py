"""
Island Mask Packer
==================
Extracts island blobs from a B&W PNG and repacks them with controlled
edge-to-edge spacing using:
  - Connected Component Labeling (extraction)
  - Force-Directed Separation (packing)
  - Per-island randomized min/max distances for organic spacing

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

    # Filter: skip islands smaller than this many pixels (noise removal)
    min_island_area: int = 50

    # Filter: skip islands larger than this (optional cap, 0 = no cap)
    max_island_area: int = 0

    # Padding around the canvas border — islands won't be placed within this
    canvas_border_padding: int = 30

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
    # Returns: num_labels, label_map, stats, centroids
    num_labels, label_map, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    islands = []
    # Label 0 is background — skip it
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        # Area filters
        if area < config.min_island_area:
            continue
        if config.max_island_area > 0 and area > config.max_island_area:
            continue

        # Extract tight crop
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]

        # Crop the label map to this component's bounding box
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
    For each island, cast rays from its center in 8 cardinal directions
    (N, NE, E, SE, S, SW, W, NW). Measure the edge-to-edge distance from
    this island's edge to the nearest other island in that direction.

    Returns a list (one entry per island) of dicts with keys like:
      {"id": 5, "N": 32.0, "NE": inf, "E": 18.5, ...}
    where inf means no island was found in that direction within the canvas.
    """
    W, H = canvas_size
    canvas = np.zeros((H, W), dtype=np.uint8)

    # Paint all islands
    for isl in islands:
        x, y = int(round(isl.x)), int(round(isl.y))
        x2, y2 = min(x + isl.w, W), min(y + isl.h, H)
        mx2, my2 = x2 - x, y2 - y
        if mx2 > 0 and my2 > 0:
            canvas[y:y2, x:x2] = np.maximum(canvas[y:y2, x:x2], isl.mask[:my2, :mx2])

    # Build a label map so we can tell which island a pixel belongs to
    label_canvas = np.zeros((H, W), dtype=np.int32)
    for idx, isl in enumerate(islands):
        x, y = int(round(isl.x)), int(round(isl.y))
        x2, y2 = min(x + isl.w, W), min(y + isl.h, H)
        mx2, my2 = x2 - x, y2 - y
        if mx2 > 0 and my2 > 0:
            mask_region = isl.mask[:my2, :mx2] > 0
            label_canvas[y:y2, x:x2][mask_region] = idx + 1  # 1-indexed

    # 8 cardinal unit direction vectors
    directions = {
        "N":  ( 0, -1),
        "NE": ( 1, -1),
        "E":  ( 1,  0),
        "SE": ( 1,  1),
        "S":  ( 0,  1),
        "SW": (-1,  1),
        "W":  (-1,  0),
        "NW": (-1, -1),
    }

    results = []
    for idx, isl in enumerate(islands):
        label_id = idx + 1
        cx, cy = int(round(isl.cx)), int(round(isl.cy))
        entry = {"id": isl.id}

        for dir_name, (ddx, ddy) in directions.items():
            # Step 1: Walk outward from center until we leave this island's pixels
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

            # (px, py) is now the first pixel outside this island along this ray
            edge_x, edge_y = px - ddx, py - ddy  # last pixel inside island

            # Step 2: Continue walking until we hit another island pixel or leave canvas
            gap_start_x, gap_start_y = px, py
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
                # Distance from island edge to the hit pixel
                dist = math.sqrt((px - edge_x) ** 2 + (py - edge_y) ** 2)
                entry[dir_name] = dist

        results.append(entry)

    return results


def measure_all_distances(islands: List[Island], canvas_size: Tuple[int, int]) -> dict:
    """
    After packing, measure nearest-neighbor edge-to-edge distances using
    vectorized bounding box gap computation. Cardinal ray measurement
    provides the precise pixel-level distances.
    """
    n = len(islands)
    if n < 2:
        return {"min_gap": 0, "max_gap": 0, "mean_gap": 0, "islands_measured": 0}

    # Pack into arrays
    xs = np.array([isl.x for isl in islands])
    ys = np.array([isl.y for isl in islands])
    ws = np.array([isl.w for isl in islands], dtype=np.float64)
    hs = np.array([isl.h for isl in islands], dtype=np.float64)

    # BB edge-to-edge gaps (vectorized)
    gap_x = np.maximum(xs[None, :] - (xs[:, None] + ws[:, None]), 0) + \
            np.maximum(xs[:, None] - (xs[None, :] + ws[None, :]), 0)
    gap_y = np.maximum(ys[None, :] - (ys[:, None] + hs[:, None]), 0) + \
            np.maximum(ys[:, None] - (ys[None, :] + hs[None, :]), 0)
    edge_dist = np.sqrt(gap_x * gap_x + gap_y * gap_y)
    np.fill_diagonal(edge_dist, np.inf)

    min_distances = edge_dist.min(axis=1)

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
    Uses a Poisson-disk-like jittered placement to avoid grid artifacts.
    Each island is placed at a random position within a loose cluster,
    with enough spread for forces to separate them organically.
    """
    rng = random.Random(config.seed)
    W, H = config.canvas_size

    # Midpoint of the distance ranges — used to size the cluster
    avg_min = sum(config.min_edge_distance_range) / 2.0
    avg_max = sum(config.max_edge_distance_range) / 2.0
    target_gap = (avg_min + avg_max) / 2.0

    # Estimate average island size
    avg_size = math.sqrt(sum(isl.area for isl in islands) / len(islands))

    n = len(islands)

    # Cluster radius: pack islands into a roughly circular region
    # Area needed ≈ n * (avg_island_size + target_gap)^2
    cell_size = avg_size + target_gap
    cluster_area = n * cell_size * cell_size
    cluster_radius = math.sqrt(cluster_area / math.pi) * 0.85  # tighter packing

    cx_canvas, cy_canvas = W / 2.0, H / 2.0

    for isl in islands:
        # Apply random rotation
        if config.allow_rotation:
            angle = rng.choice([0, 90, 180, 270])
            if angle != 0:
                isl.mask = np.rot90(isl.mask, k=angle // 90)

        # Place at random position within the circular cluster
        # Use rejection sampling for uniform distribution in a circle
        while True:
            rx = rng.uniform(-cluster_radius, cluster_radius)
            ry = rng.uniform(-cluster_radius, cluster_radius)
            if rx * rx + ry * ry <= cluster_radius * cluster_radius:
                break

        isl.x = cx_canvas + rx - isl.w / 2.0
        isl.y = cy_canvas + ry - isl.h / 2.0
        isl.vx = 0.0
        isl.vy = 0.0

    # Clamp all to canvas
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
    Force-directed layout with per-island distance constraints.

    Each island has its own my_min_dist and my_max_dist. For any pair (i, j):
      - effective_min = (my_min_dist_i + my_min_dist_j) / 2
      - effective_max = (my_max_dist_i + my_max_dist_j) / 2

    This creates organic, non-uniform spacing.

    Vectorized with NumPy — all pairwise distances and forces computed as
    bulk array operations.
    """
    n = len(islands)
    strength = config.force_strength
    damping = config.damping
    attr_str = config.attraction_strength

    W_canvas, H_canvas = config.canvas_size
    pad = config.canvas_border_padding

    # Pack island geometry into contiguous arrays
    pos = np.zeros((n, 2), dtype=np.float64)
    vel = np.zeros((n, 2), dtype=np.float64)
    dims = np.zeros((n, 2), dtype=np.float64)  # w, h per island
    min_dists = np.zeros(n, dtype=np.float64)   # per-island min distance
    max_dists = np.zeros(n, dtype=np.float64)   # per-island max distance

    for i, isl in enumerate(islands):
        pos[i, 0] = isl.x
        pos[i, 1] = isl.y
        dims[i, 0] = isl.w
        dims[i, 1] = isl.h
        min_dists[i] = isl.my_min_dist
        max_dists[i] = isl.my_max_dist

    half_dims = dims / 2.0

    # Pairwise effective distances: average of both islands' values
    # eff_min[i,j] = (min_dists[i] + min_dists[j]) / 2
    eff_min = (min_dists[:, None] + min_dists[None, :]) / 2.0   # (n, n)
    eff_max = (max_dists[:, None] + max_dists[None, :]) / 2.0   # (n, n)
    half_eff_min = eff_min / 2.0  # half of effective min distance per pair

    # Check if max distance enforcement is enabled
    use_max_dist = config.max_edge_distance_range[1] > 0

    for iteration in range(config.max_iterations):
        # Centers: pos + half_dims
        centers = pos + half_dims  # (n, 2)

        # --- Vectorized repulsion (bounding box overlap with per-pair padding) ---
        # Each pair has its own padding: half_eff_min[i,j]
        # For efficiency, use per-island half-padding and sum for pairs
        half_min_per_island = min_dists / 2.0  # (n,)

        bb_x1 = pos[:, 0] - half_min_per_island
        bb_y1 = pos[:, 1] - half_min_per_island
        bb_x2 = pos[:, 0] + dims[:, 0] + half_min_per_island
        bb_y2 = pos[:, 1] + dims[:, 1] + half_min_per_island

        # Pairwise overlap
        ox = np.minimum(bb_x2[:, None], bb_x2[None, :]) - np.maximum(bb_x1[:, None], bb_x1[None, :])
        oy = np.minimum(bb_y2[:, None], bb_y2[None, :]) - np.maximum(bb_y1[:, None], bb_y1[None, :])

        overlapping = (ox > 0) & (oy > 0)
        np.fill_diagonal(overlapping, False)

        # Direction vectors
        dx = centers[:, 0:1] - centers[:, 0:1].T
        dy = centers[:, 1:2] - centers[:, 1:2].T
        dist = np.sqrt(dx * dx + dy * dy) + 1e-6

        # Force magnitude: overlap_area / min_expanded_area
        overlap_area = np.maximum(ox, 0) * np.maximum(oy, 0)
        expanded_w = dims[:, 0:1] + min_dists[:, None]  # use island i's min_dist for expansion
        expanded_h = dims[:, 1:2] + min_dists[:, None]
        expanded_area_i = expanded_w * expanded_h
        expanded_area_j = (dims[:, 0:1].T + min_dists[None, :]) * (dims[:, 1:2].T + min_dists[None, :])
        min_area = np.minimum(expanded_area_i, expanded_area_j)
        force_mag = np.minimum(overlap_area / (min_area + 1e-6), 1.0)
        force_mag[~overlapping] = 0.0

        rep_fx = (dx / dist) * force_mag
        rep_fy = (dy / dist) * force_mag

        forces = np.zeros((n, 2), dtype=np.float64)
        forces[:, 0] = rep_fx.sum(axis=1) * strength
        forces[:, 1] = rep_fy.sum(axis=1) * strength

        # --- Vectorized attraction (nearest neighbor, per-island max dist) ---
        if use_max_dist:
            # BB edge-to-edge gap
            gap_x = np.maximum(pos[:, 0:1].T - (pos[:, 0:1] + dims[:, 0:1]), 0) + \
                    np.maximum(pos[:, 0:1] - (pos[:, 0:1].T + dims[:, 0:1].T), 0)
            gap_y = np.maximum(pos[:, 1:2].T - (pos[:, 1:2] + dims[:, 1:2]), 0) + \
                    np.maximum(pos[:, 1:2] - (pos[:, 1:2].T + dims[:, 1:2].T), 0)
            edge_dist = np.sqrt(gap_x * gap_x + gap_y * gap_y)

            np.fill_diagonal(edge_dist, np.inf)
            nn_idx = np.argmin(edge_dist, axis=1)
            nn_edge = edge_dist[np.arange(n), nn_idx]

            # Each island uses its own max_dist threshold
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def pack_islands(
    input_path: str,
    output_path: str,
    config: Optional[PackerConfig] = None,
) -> dict:
    """
    Full pipeline: extract → place → simulate → render → measure.
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
    print()

    # 1. Extract
    t0 = time.time()
    islands = extract_islands(input_path, config)
    print(f"Extraction: {time.time()-t0:.2f}s | {len(islands)} islands\n")

    if not islands:
        raise ValueError("No islands found — check min_island_area or image threshold")

    # Shuffle randomly — no size bias in placement
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
    print("Done.\n")

    # 3. Force-directed simulation
    print("Running force-directed simulation...")
    t1 = time.time()
    run_force_simulation(islands, config)
    print(f"Simulation: {time.time()-t1:.2f}s\n")

    # 4. Render
    canvas = render_packed_canvas(islands, config)
    cv2.imwrite(output_path, canvas)
    print(f"Saved: {output_path}")

    # 5. Measure distances (BB nearest neighbor)
    print("\nMeasuring edge-to-edge distances (BB)...")
    stats = measure_all_distances(islands, config.canvas_size)
    stats["total_islands"] = len(islands)
    stats["canvas"] = f"{config.canvas_size[0]}x{config.canvas_size[1]}"
    stats["min_edge_distance_range"] = f"{min_lo}-{min_hi}"
    stats["max_edge_distance_range"] = f"{max_lo}-{max_hi}"

    # 6. Measure 8-cardinal-direction distances
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

                # Check if ALL 8 directions exceed this island's max distance
                isl_max = islands[idx].my_max_dist
                all_exceed = all(
                    entry[d] > isl_max for d in dir_names
                )
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
        max_iterations=3000,
        force_strength=2.0,
        attraction_strength=3.0,
        damping=0.6,
        convergence_threshold=0.1,
        allow_rotation=True,
        seed=42,
    )

    stats = pack_islands(input_file, output_file, cfg)
