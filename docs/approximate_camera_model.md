# Approximate camera model before ChArUco calibration

This project now supports a temporary camera model:

```yaml
pipeline:
  camera_model:
    mode: approximate_pinhole_from_corners
```

It is meant for review/debugging while waiting for the CALITAR ChArUco board.
It is not calibrated geometry.

Inputs:

- camera image size;
- approximate focal length;
- approximate sensor size;
- principal point, defaulting to image center;
- manual table corner pixels;
- known table dimensions;
- known ball radius.

How it works:

1. Build an approximate pinhole intrinsic matrix from focal length, sensor size,
   and image resolution.
2. Use `solvePnP` with the four manually clicked table corners to estimate
   camera pose relative to the table.
3. Cast source-image rays through selected ball centers.
4. Intersect those rays with Z planes such as `Z=0`, `Z=26.25`, and `Z=52.5`.
5. Project a predicted sphere silhouette in the review UI.

What it can help with:

- visualizing why `Z=0` and `Z=26.25` produce different table XY coordinates;
- comparing observed ball boundary points against an expected physical sphere
  projection;
- making the review UI more evidence-based before real calibration exists.

What it cannot guarantee:

- true lens distortion correction;
- accurate focal length after focus/crop/exif assumptions;
- accurate camera pose if the four table corners are imperfect;
- correct sphere silhouette for edge balls at calibrated accuracy.

## What is needed besides ChArUco calibration

ChArUco gives the camera intrinsics, distortion coefficients, and a strong
camera pose estimate. To turn the physical sphere projection from a scoring aid
into the main ball position solver, the project still needs:

- a stable table coordinate definition: which table corner is `(0, 0)` and how
  cushion/cloth boundaries map to world coordinates;
- reliable table-plane pose for each camera setup;
- known ball radius and table dimensions;
- good source-image segmentation of each visible ball blob;
- an optimizer that adjusts the candidate 3D ball center until the projected
  sphere silhouette best matches the observed boundary/mask evidence;
- physical validation constraints such as touching-ball distance, cushion-touch
  distance, rack distances, and known spots.

What can be done before the board arrives:

- use `approximate_pinhole_from_corners`;
- project the physical sphere silhouette approximately;
- score radial/edge observed ellipse evidence against the physical projection;
- benchmark legacy circle-first confidence against experimental physics-first
  and physical-model-first confidence with `tools/benchmark_model_scoring.py`.

Once ChArUco images are available, replace this mode with:

```yaml
pipeline:
  camera_model:
    mode: calibrated_pinhole
    camera_matrix: [...]
    distortion_coefficients: [...]
    rotation_world_to_camera: [...]
    translation_world_to_camera: [...]
```

The review UI intentionally labels the current sphere projection as approximate
when this mode is active.
