# Island Packer

Extracts island blobs from a black & white PNG mask and repacks them onto a new canvas with controlled edge-to-edge spacing using force-directed simulation.

## Features

- **Connected Component Labeling** — Extracts individual islands from a B&W mask image using OpenCV
- **Force-Directed Packing** — Pushes overlapping islands apart (repulsion) and pulls distant islands closer (attraction) using vectorized NumPy computation
- **Min & Max Distance Constraints** — Enforces both a minimum and maximum edge-to-edge gap between islands
- **8-Cardinal Ray Measurement** — Measures edge-to-edge distances in N/NE/E/SE/S/SW/W/NW from each island center
- **Random Placement** — Islands are randomly shuffled before placement (no size bias)
- **Rotation Support** — Optional random 90° rotation of islands
- **Diagnostic Tool** — `check_gaps.py` provides ground-truth pixel-level gap verification via local EDT

## Installation

```bash
pip install opencv-python numpy Pillow
```

## Usage

```bash
# Basic usage (defaults: min_gap=20, max_gap=50, canvas=4096x4096)
python island_packer.py input.png output.png

# Custom gap distances
python island_packer.py input.png output.png 25 55

# Custom gap distances + canvas size
python island_packer.py input.png output.png 25 55 4096 4096
```

### Arguments

| Position | Description | Default |
|----------|-------------|---------|
| 1 | Input PNG (B&W island mask) | `IslandMaskBWV11.png` |
| 2 | Output PNG | `IslandMaskPacked.png` |
| 3 | Min edge-to-edge gap (px) | `20` |
| 4 | Max edge-to-edge gap (px) | `50` |
| 5 | Canvas width (px) | `4096` |
| 6 | Canvas height (px) | `4096` |

### Diagnostics

```bash
# Verify true pixel-level gaps in the packed output
python check_gaps.py
```

## How It Works

1. **Extract** — Load the B&W PNG and run connected component labeling to identify individual islands
2. **Shuffle** — Randomly order the islands (no size-based sorting)
3. **Place** — Arrange islands in a tight grid cluster near canvas center
4. **Simulate** — Run force-directed simulation:
   - **Repulsion**: Islands whose bounding boxes (padded by `min_edge_distance`) overlap get pushed apart
   - **Attraction**: Islands whose nearest neighbor exceeds `max_edge_distance` get pulled closer
5. **Render** — Paint all islands onto the output canvas
6. **Measure** — Report bounding-box gap statistics and 8-cardinal-direction distances

## Configuration

Key parameters in `PackerConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_edge_distance` | 20 | Minimum px gap between island edges |
| `max_edge_distance` | 50 | Maximum px gap (8-cardinal measurement) |
| `min_island_area` | 50 | Filter out islands smaller than this |
| `canvas_border_padding` | 30 | Keep islands this far from canvas edges |
| `max_iterations` | 3000 | Force simulation iteration cap |
| `force_strength` | 1.5 | Repulsion force multiplier |
| `attraction_strength` | 3.0 | Attraction force multiplier |
| `damping` | 0.65 | Velocity damping per step |
| `convergence_threshold` | 0.1 | Stop when max movement < this |
| `seed` | 42 | Random seed (None = random) |
| `allow_rotation` | True | Randomly rotate islands 0/90/180/270° |

## Performance

Tested with 857 islands on a 4096×4096 canvas:
- Extraction: ~0.06s
- Force simulation: ~54s (vectorized NumPy)
- Cardinal measurement: ~10-15s

## License

MIT
