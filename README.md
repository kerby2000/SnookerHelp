# Snooker Vision

First classical-OpenCV prototype for recognizing a fixed overhead snooker
table from full-resolution Sony still images.

The current pipeline:

1. Loads a 6000 x 4000 Sony JPEG.
2. Applies the saved four-corner table homography.
3. Warps the playing surface to a 1 px/mm top-down image.
4. Compares it with a warped empty-table reference.
5. Finds ball-sized circles in the difference image.
6. Refines final ball circles in the original source image.
7. Classifies their colors from robust HSV/Lab samples.
8. Writes source/warped overlays, a difference image, and JSON table state.

The checked-in calibration is an initial estimate for the supplied `Media`
images. Re-click the four cushion-nose intersections before using coordinates
for measurement.

## Setup

```powershell
python -m pip install -e ".[dev]"
```

## Process one image

```powershell
python tools/process_single_image.py `
  --image Media/02_random_balls/DSC00524.JPG
```

v1 wrapper:

```powershell
python -m snookerhelp.tools.process_image `
  --image Media/02_random_balls/DSC00524.JPG
```

Outputs are written to `data/debug_outputs/<image-name>/`.

## Recalibrate table corners

Click in this order:

1. top-left
2. top-right
3. bottom-right
4. bottom-left

Use the virtual intersection of the straight cushion noses, not the outside
wooden rail and not the deepest point of a pocket opening.

```powershell
python tools/click_table_corners.py `
  --image Media/01_empty_table/DSC00543.JPG
```

Left click adds a point, Backspace removes the last point, `R` resets, Enter
saves, and Escape cancels.

## Process the latest Sony image

```powershell
python tools/process_latest_image.py --folder "C:/Users/lukin/Pictures"
```

## Watch the Sony Remote folder

```powershell
python capture/watch_sony_folder.py --folder "C:/Users/lukin/Pictures"
```

The watcher waits until a new JPEG's size is stable before processing it.

## Evaluate supplied samples

```powershell
python tools/evaluate_samples.py
```

v1 wrapper:

```powershell
python -m snookerhelp.tools.validate --kind samples
```

Each non-empty sample contains the full 22-ball inventory. The evaluator uses
that known count only for reporting; the detector's inventory pruning applies
maximum legal counts and does not invent missing detections.

Current baseline on the supplied 21 images:

- 21/21 images have the exact expected count.
- Mean absolute count error is 0.000 balls/image.
- Both empty-table images produce zero detections.
- All five random-ball images produce 22 detections.
- All cushion, pocket-mouth, and dense-cluster samples produce 22 detections.

See `docs/baseline_validation.md` for the scenario breakdown.

## Coordinate accuracy

Create manual ball-center ground truth:

```powershell
python tools/annotate_ball_centers.py `
  --image Media/02_random_balls/DSC00524.JPG `
  --coordinate-system warped_px
```

Then evaluate the detector:

```powershell
python tools/evaluate_accuracy.py `
  --image Media/02_random_balls/DSC00524.JPG
```

The evaluator writes per-ball CSV and JSON errors plus an error-vector overlay.
Statistics include mean, median, 95th percentile, and maximum error in both
warped pixels and millimeters.

Important geometry note: the warped image is a cloth-plane rectification. It is
valid for flat table features, but balls are 3D objects above the cloth, so ball
shapes near sides/corners are not expected to remain circular in the warped
view. The JSON now includes both warped centers and source-image refined
centers; current table millimeter coordinates are still marked approximate
until a real camera model is calibrated.

For the camera/ray geometry, Z-plane projection model, and calibrated-pinhole
config shape, see `docs/ball_geometry_model.md`.

For animated Manim explainers of the same geometry, see
`docs/manim_geometry_visualizations.md`.

For per-image visual reports tied to real photos, see
`docs/image_debug_reports.md`.

Run the v1 review UI:

```powershell
python -m snookerhelp.tools.review --reports outputs/reports --port 8770
```

For repeatability, use photos captured without moving any balls:

```powershell
python tools/evaluate_repeatability.py `
  repeated/DSC01001.JPG `
  repeated/DSC01002.JPG `
  repeated/DSC01003.JPG
```

See `docs/coordinate_accuracy_validation.md` for annotation keys, schemas, and
measurement constraints.

For lower-effort physical checks, use:

- `tools/evaluate_touching_balls.py` for touching-pair distance and red-rack
  nearest-neighbor validation, including `--center-mode compare` for warped vs
  source-refined center comparison and `--center-mode z-planes` for per-height
  projection comparison.
- `tools/evaluate_cushion_touch.py` for ball-radius-to-cushion checks.
- `tools/evaluate_spot_positions.py` for known table spot checks.

See `docs/physical_validation_tools.md` for command examples and YAML formats.

## Current limitations

- The background-reference method assumes the camera, table, and lighting stay
  fixed. Re-capture the empty reference after moving the camera or changing
  lighting.
- The supplied calibration does not correct lens distortion.
- Homography coordinates are approximate because ball centers are above the
  cloth plane and camera intrinsics/pose are not yet modeled.
- Deeply occluded balls in pocket mouths and dense clusters can still be
  missed or confused.
- Color rules are tuned to the supplied Sony images and should later be
  replaced by measured per-camera color prototypes.

For coordinate-accuracy calibration inputs, see
`docs/precision_calibration_inputs.md`.
