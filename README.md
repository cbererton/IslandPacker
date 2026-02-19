# Island Packer

Extracts island blobs from a black & white PNG mask and repacks them onto a new canvas with organic, non-uniform spacing using force-directed simulation.

## Features

- **Connected Component Labeling** — Extracts individual islands from a B&W mask image using OpenCV
- **Force-Directed Packing** — Pushes overlapping islands apart (repulsion) and pulls distant islands closer (attraction) using vectorized NumPy computation
- **Multi-Box Splitting** — Large elongated islands are split into 2 tight bounding boxes so other islands can nestle closer, reducing wasted space
- **Per-Island Distance Ranges** — Each island gets its own randomly chosen min/max distance from configurable ranges, creating organic non-uniform spacing
- **8-Cardinal Ray Measurement** — Measures edge-to-edge distances in N/NE/E/SE/S/SW/W/NW from each island center
- **Auto-Crop** — Output is automatically cropped to the content bounding box with configurable padding
- **Scattered Placement** — Islands are randomly shuffled and placed in a circular cluster (no grid artifacts)
- **Rotation Support** — Random 90° rotation of islands (0/90/180/270°)
- **Diagnostic Tool** — `check_gaps.py` provides ground-truth pixel-level gap verification via local EDT

## Installation

```bash
pip install opencv-python numpy Pillow
```

## Usage

```bash
# Basic usage (defaults: min_range=5-10, max_range=15-20, canvas=4096x4096)
python island_packer.py input.png output.png

# Custom distance ranges: min_lo min_hi max_lo max_hi
python island_packer.py input.png output.png 5 10 15 20

# Custom distance ranges + canvas size
python island_packer.py input.png output.png 5 10 15 20 4096 4096
```

### Arguments

| Position | Description | Default |
|----------|-------------|---------|
| 1 | Input PNG (B&W island mask) | `IslandMaskBWV11.png` |
| 2 | Output PNG | `IslandMaskPacked.png` |
| 3 | Min edge distance range low (px) | `5` |
| 4 | Min edge distance range high (px) | `10` |
| 5 | Max edge distance range low (px) | `15` |
| 6 | Max edge distance range high (px) | `20` |
| 7 | Canvas width (px) | `4096` |
| 8 | Canvas height (px) | `4096` |

### Diagnostics

```bash
# Verify true pixel-level gaps in the packed output
python check_gaps.py
```

## How It Works

1. **Extract** — Load the B&W PNG and run connected component labeling to identify individual islands
2. **Shuffle & Assign** — Randomly order the islands and assign each a random min/max distance from the configured ranges
3. **Place** — Scatter islands randomly within a circular cluster near canvas center (no grid)
4. **Split** — Large elongated islands (area ≥ threshold AND aspect ≥ ratio) are split into 2 tight bounding boxes along their longest axis
5. **Simulate** — Run force-directed simulation with per-island distances:
   - **Repulsion**: Box-to-box overlap checks across different islands; forces accumulate per island
   - **Attraction**: Nearest neighbor edge distance computed across all box pairs; islands pulled closer when too far
   - For each pair, the effective distance = average of both islands' values
6. **Render** — Paint all islands onto the output canvas
7. **Crop** — Auto-crop the canvas to content bounding box plus padding
8. **Measure** — Report multi-box BB gap statistics and 8-cardinal-direction distances

## Configuration

Key parameters in `PackerConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_edge_distance_range` | (5, 10) | Range for per-island minimum gap (px) |
| `max_edge_distance_range` | (15, 20) | Range for per-island maximum gap (px) |
| `split_area_threshold` | 1600 | Split islands with area ≥ this into 2 boxes (0 = disabled) |
| `split_aspect_ratio` | 1.0 | Only split islands with aspect ratio ≥ this (1.0 = all large islands) |
| `crop_padding` | 5 | Pixels of padding around content in cropped output |
| `min_island_area` | 50 | Filter out islands smaller than this |
| `canvas_border_padding` | 30 | Keep islands this far from canvas edges |
| `max_iterations` | 3000 | Force simulation iteration cap |
| `force_strength` | 2.0 | Repulsion force multiplier |
| `attraction_strength` | 3.0 | Attraction force multiplier |
| `damping` | 0.6 | Velocity damping per step |
| `convergence_threshold` | 0.1 | Stop when max movement < this |
| `seed` | 42 | Random seed (None = random) |
| `allow_rotation` | True | Randomly rotate islands 0/90/180/270° |

## Performance

Tested with 857 islands on a 4096×4096 canvas:
- Extraction: ~0.06s
- Force simulation: ~60-140s (vectorized NumPy, depends on convergence)
- Cardinal measurement: ~10-15s
- Output auto-cropped from 4096×4096 to ~2677×2538

## License

MIT
