# Baseline validation

Validation command:

```powershell
python tools/evaluate_samples.py
```

Dataset: 21 supplied 6000 x 4000 Sony JPEGs. The two empty-table images have
an expected count of zero. Every non-empty test image contains the full legal
22-ball inventory.

| Scenario | Exact-count images | Notes |
| --- | ---: | --- |
| `01_empty_table` | 2/2 | Zero false detections |
| `02_random_balls` | 5/5 | Exact count on every image |
| `03_near_cushions` | 5/5 | Exact count and legal color inventory |
| `04_near_pockets` | 4/4 | Padded warp retains balls centered beyond cushion-nose boundaries |
| `05_clusters` | 5/5 | Dense red triangle produces all 15 red candidates |

Overall:

- Exact expected count: 21/21 images
- Mean absolute count error: 0.000 balls/image
- Both empty-table references: 0 detections

These numbers measure count only. The dataset does not yet include annotated
ground-truth centers, so millimeter coordinate error has not been measured.
The checked-in corners are an estimated calibration and lens distortion is not
yet corrected. Coordinate-accuracy claims would therefore be premature.

