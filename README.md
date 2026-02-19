# Island Packer

Extracts island blobs from a black & white PNG mask and repacks them onto a new canvas with organic, non-uniform spacing using force-directed simulation.

## Features

- **Connected Component Labeling** — Extracts individual islands from a B&W mask image using OpenCV
- **Force-Directed Packing** — Pushes overlapping islands apart (repulsion) and pulls distant islands closer (attraction) using vectorized NumPy computation
- **Per-Island Distance Ranges** — Each island gets its own randomly chosen min/max distance from configurable ranges, creating organic non-uniform spacing
- **8-Cardinal Ray Measurement** — Measures edge-to-edge distances in N/NE/E/SE/S/SW/W/NW from each island center
- **Scattered Placement** — Islands are randomly shuffled and placed in a circular cluster (no grid artifacts)
- **Rotation Support** — Random 90° rotation of islands (0/90/180/270°)
- **Diagnostic Tool** — `check_gaps.py` provides ground-truth pixel-level gap verification via local EDT

## Installation

```bash
pip install opencv-python numpy Pillow
```

## Usage

```bash
# Basic usage (defaults: min_range=20-35, max_range=45-70, canvas=4096x4096)
python island_packer.py input.png output.png

# Custom distance ranges: min_lo min_hi max_lo max_hi
python island_packer.py input.png output.png 20 35 45 70

# Custom distance ranges + canvas size
python island_packer.py input.png output.png 20 35 45 70 4096 4096
```

### Arguments

| Position | Description | Default |
|----------|-------------|---------|
| 1 | Input PNG (B&W island mask) | `IslandMaskBWV11.png` |
| 2 | Output PNG | `IslandMaskPacked.png` |
| 3 | Min edge distance range low (px) | `20` |
| 4 | Min edge distance range high (px) | `35` |
| 5 | Max edge distance range low (px) | `45` |
| 6 | Max edge distance range high (px) | `70` |
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
4. **Simulate** — Run force-directed simulation with per-island distances:
   - **Repulsion**: Islands whose bounding boxes (padded by their individual min distance) overlap get pushed apart
   - **Attraction**: Islands whose nearest neighbor exceeds their individual max distance get pulled closer
   - For each pair, the effective distance = average of both islands' values
5. **Render** — Paint all islands onto the output canvas
6. **Measure** — Report bounding-box gap statistics and 8-cardinal-direction distances

## Configuration

Key parameters in `PackerConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_edge_distance_range` | (20, 35) | Range for per-island minimum gap (px) |
| `max_edge_distance_range` | (45, 70) | Range for per-island maximum gap (px) |
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

Tested with 857 islands on a 4096x4096 canvas:
- Extraction: ~0.06s
- Force simulation: ~60-120s (vectorized NumPy, depends on convergence)
- Cardinal measurement: ~10-15s

## License

MIT
