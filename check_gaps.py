"""Fast diagnostic: check minimum edge-to-edge gaps using label map and local EDT."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import time

img = cv2.imread('IslandMaskPacked.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

num_labels, label_map, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
n = num_labels - 1
print(f'Components in packed image: {n}')

# For each island, use a LOCAL window (bounding box + search radius)
# to compute EDT without this island, then sample at boundary
search_radius = 80  # only look this far
kernel = np.ones((3,3), np.uint8)
H, W = binary.shape
min_gaps = []
t0 = time.time()

for lbl in range(1, num_labels):
    x = stats[lbl, cv2.CC_STAT_LEFT]
    y = stats[lbl, cv2.CC_STAT_TOP]
    w = stats[lbl, cv2.CC_STAT_WIDTH]
    h = stats[lbl, cv2.CC_STAT_HEIGHT]

    # Expand window by search_radius
    rx1 = max(0, x - search_radius)
    ry1 = max(0, y - search_radius)
    rx2 = min(W, x + w + search_radius)
    ry2 = min(H, y + h + search_radius)

    # Local label map and binary
    local_labels = label_map[ry1:ry2, rx1:rx2]
    local_binary = binary[ry1:ry2, rx1:rx2].copy()

    # Remove this island from the local binary
    local_binary[local_labels == lbl] = 0

    # EDT in local window — distance from each pixel to nearest remaining island
    edt = distance_transform_edt(local_binary == 0)

    # Get this island's boundary pixels (in local coords)
    mask = (local_labels == lbl).astype(np.uint8) * 255
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = (mask > 0) & (eroded == 0)

    if boundary.any():
        vals = edt[boundary]
        if vals.size > 0:
            min_gaps.append((lbl, float(vals.min())))

print(f'Computed in {time.time()-t0:.1f}s')

print(f'\nIslands with smallest true gaps:')
for lbl, gap in sorted(min_gaps, key=lambda x: x[1])[:15]:
    cx, cy = centroids[lbl]
    area = stats[lbl, cv2.CC_STAT_AREA]
    print(f'  Island {lbl}: min_gap={gap:.1f}px area={area} center=({cx:.0f},{cy:.0f})')

print(f'\nGap distribution:')
gaps = [g for _, g in min_gaps]
for threshold in [5, 10, 15, 20, 25, 30, 40, 50]:
    count = sum(1 for g in gaps if g < threshold)
    print(f'  < {threshold}px: {count}/{len(gaps)} islands')
print(f'  Total islands: {len(gaps)}')
print(f'  Min gap: {min(gaps):.1f}px')
print(f'  Max gap: {max(gaps):.1f}px')
print(f'  Mean gap: {np.mean(gaps):.1f}px')
