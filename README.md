# Island Packer

Extracts island blobs from a black & white PNG mask and repacks them onto a new canvas with organic, non-uniform spacing using force-directed simulation.

## Features

- **Connected Component Labeling** — Extracts individual islands from a B&W mask image using OpenCV
- **Force-Directed Packing** — Pushes overlapping islands apart (repulsion) and pulls distant islands closer (attraction) using vectorized NumPy computation
- **Recursive Multi-Box Splitting** — Large islands (top 10%) are recursively split along their longest axis (up to 4 tight bounding boxes per island), letting other islands nestle closer
- **Expansion Loop** — Iteratively finds large empty regions within the packed cluster and fills them with randomly-rotated copies of source islands, re-running the force sim each round until the cluster is dense
- **Gap Filling** — After expansion, a final pass places copies of small islands into remaining tiny gaps between islands
- **Debug Box Visualization** — Generates a second output image showing bounding boxes used for spacing overlaid in green
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
# Basic usage (defaults: min_range=7-13, max_range=15-20, canvas=4096x4096)
python island_packer.py input.png output.png

# Custom distance ranges: min_lo min_hi max_lo max_hi
python island_packer.py input.png output.png 7 13 15 20

# Custom distance ranges + canvas size
python island_packer.py input.png output.png 7 13 15 20 4096 4096
```

### Arguments

| Position | Description | Default |
|----------|-------------|---------|
| 1 | Input PNG (B&W island mask) | `IslandMaskBWV11.png` |
| 2 | Output PNG | `IslandMaskPacked.png` |
| 3 | Min edge distance range low (px) | `7` |
| 4 | Min edge distance range high (px) | `13` |
| 5 | Max edge distance range low (px) | `15` |
| 6 | Max edge distance range high (px) | `20` |
| 7 | Canvas width (px) | `4096` |
| 8 | Canvas height (px) | `4096` |

### Outputs

The packer generates two output files:
- **`output.png`** — The packed island image (B&W, auto-cropped)
- **`output_boxes.png`** — Debug visualization with green bounding box outlines showing the sub-boxes used for spacing on split islands

### Diagnostics

```bash
# Verify true pixel-level gaps in the packed output
python check_gaps.py
```

## How It Works

1. **Extract** — Load the B&W PNG and run connected component labeling to identify individual islands
2. **Shuffle & Assign** — Randomly order the islands and assign each a random min/max distance from the configured ranges
3. **Place** — Scatter islands randomly within a circular cluster near canvas center (no grid)
4. **Split** — Large islands (area >= threshold, top ~10%) are recursively split up to `max_splits` times along their longest axis, producing tight bounding boxes for each sub-region
5. **Simulate** — Run force-directed simulation with per-island distances:
   - **Repulsion**: Island-level BB gap checks; sparse pair indexing for efficiency
   - **Attraction**: Nearest neighbor edge distance (squared, no sqrt); islands pulled closer when too far
   - For each pair, the effective distance = average of both islands' values
6. **Expand** — Iteratively find large empty regions (>= `expand_min_gap_size`) within the initial cluster bounding box, add randomly-rotated copies of source islands, and re-run a short force sim. Repeats until gaps are filled or diminishing returns
7. **Gap Fill** — Final pass scans for small empty regions and places copies of small islands into them (originals stay in place)
8. **Render** — Paint all islands onto the output canvas + generate debug box visualization
9. **Crop** — Auto-crop both outputs to content bounding box plus padding
10. **Measure** — Report multi-box BB gap statistics and 8-cardinal-direction distances

## Configuration

Key parameters in `PackerConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_edge_distance_range` | (7, 13) | Range for per-island minimum gap (px) |
| `max_edge_distance_range` | (15, 20) | Range for per-island maximum gap (px) |
| `split_area_threshold` | 2561 | Recursively split islands with area >= this (0 = disabled) |
| `split_aspect_ratio` | 1.0 | Only split islands with aspect ratio >= this (1.0 = all large islands) |
| `max_splits` | 2 | Max recursive splits per island (2 = up to 4 boxes) |
| `expand_min_gap_size` | 40 | Minimum empty region size (px) to trigger expansion (0 = disabled) |
| `expand_max_rounds` | 10 | Maximum expansion iterations |
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

Tested with 857 islands on a 4096x4096 canvas:
- Extraction: ~0.06s
- Initial force simulation: ~7s (sparse COO + float32 vectorized NumPy)
- Expansion (10 rounds): ~17s
- Gap filling: ~1.4s
- Total: ~30s

## License

MIT
