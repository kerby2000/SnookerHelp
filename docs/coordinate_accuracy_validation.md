# Coordinate accuracy validation

## Subpixel center output

Each detected ball now includes:

```json
{
  "raw_hough_center_px": [500.0, 1019.0],
  "warped_center_px": [500.12, 1018.87],
  "refined_center_px": [500.12, 1018.87],
  "table_xy_mm": [380.12, 879.13],
  "table_xy_mm_approximate": true,
  "source_rough_center_px": [910.4, 1822.1],
  "source_refined_center_px": [911.2, 1821.7],
  "source_refined_table_xy_by_z_mm": {
    "z_0_00": {
      "z_mm": 0.0,
      "xy_mm": [379.8, 879.4],
      "approximate": true
    },
    "z_26_25": {
      "z_mm": 26.25,
      "xy_mm": [379.8, 879.4],
      "approximate": true
    }
  },
  "source_radius_px": 41.8,
  "source_fit_residual_px": 1.9,
  "source_refinement_success": true,
  "raw_hough_radius_px": 27.0,
  "radius_px": 26.42,
  "radius_mm": 26.42,
  "fit_residual_px": 0.48,
  "color_label": "white",
  "color_confidence": 0.88,
  "detection_confidence": 0.95
}
```

The fitter samples the outward foreground boundary around the Hough estimate
and applies robust least-squares circle fitting. If the edge support, radius,
center displacement, or residual is invalid, the raw Hough result is retained
and `fit_residual_px` is `null`.

The warped view is a cloth-plane rectification. It is valid for flat table
features, but ball shapes near the sides and corners are not expected to remain
circular because ball centers are above the cloth. Current `table_xy_mm` values
are therefore marked approximate. The detector now also stores source-image
rough/refined centers so later camera calibration can project ball centers
through a real 3D camera model.

`source_refined_table_xy_by_z_mm` stores the source-image center projected to
multiple effective Z planes. In the current development default,
`approximate_pinhole_from_corners`, these values come from approximate
ray/plane intersections derived from lens/sensor metadata and manual table
corners. They are useful for debugging height/parallax effects, but they are
not calibrated truth. In future `calibrated_pinhole` mode, these values will
come from ChArUco-derived intrinsics, distortion, and camera pose.

## Manual annotation

Recommended command:

```powershell
python tools/annotate_ball_centers.py `
  --image Media/02_random_balls/DSC00524.JPG `
  --coordinate-system warped_px
```

Controls:

| Key | Action |
| --- | --- |
| `1` | white |
| `2` | red |
| `3` | yellow |
| `4` | green |
| `5` | brown |
| `6` | blue |
| `7` | pink |
| `8` | black |
| `9` | unknown |
| Left click | add center using selected label |
| Shift+left click | relabel nearest annotation |
| Right click | delete nearest annotation |
| Backspace | undo last annotation |
| `+` / `-` | zoom |
| arrows or `w`/`a`/`s`/`d` | pan |
| `0` | reset view |
| `R` | reset |
| Shift+`A` | accept detector-seeded points and save |
| Enter or Shift+`S` | save |
| Escape | cancel |

Default output:

```text
data/annotations/<image_stem>.json
```

Supported coordinate systems:

- `source_px`: clicks are stored in original-camera pixels.
- `warped_px`: clicks are stored in the padded metric warp.
- `table_mm`: the warped view is shown but points are stored in table
  millimeters.

Example schema:

```json
{
  "version": 1,
  "image_name": "Media/02_random_balls/DSC00524.JPG",
  "coordinate_system": "warped_px",
  "notes": null,
  "balls": [
    {
      "id": 1,
      "label": "white",
      "x": 500.2,
      "y": 1018.9,
      "notes": "optional"
    }
  ]
}
```

## Accuracy report

```powershell
python tools/evaluate_accuracy.py `
  --image Media/02_random_balls/DSC00524.JPG
```

Optional explicit paths:

```powershell
python tools/evaluate_accuracy.py `
  --image Media/02_random_balls/DSC00524.JPG `
  --annotations data/annotations/DSC00524.json `
  --detector-output data/debug_outputs/DSC00524/DSC00524_state.json
```

Outputs:

```text
data/accuracy_reports/DSC00524/
  DSC00524_accuracy.csv
  DSC00524_accuracy.json
  DSC00524_accuracy_overlay.jpg
  DSC00524_detector_state.json
```

The overlay uses:

- green cross: annotated center;
- red point: detected center;
- yellow arrow: error vector;
- magenta cross: missed annotation;
- red circle: extra detection.

At the current 1 px/mm warp scale, one warped pixel equals one table
millimeter. This is a coordinate-scale conversion, not proof of physical
accuracy. Physical accuracy still depends on exact table dimensions, clicked
corners, lens distortion, and annotation quality.

## Repeatability

Capture several images without moving the camera, table, lighting, or balls:

```powershell
python tools/evaluate_repeatability.py `
  repeated/DSC01001.JPG `
  repeated/DSC01002.JPG `
  repeated/DSC01003.JPG
```

Or process a folder:

```powershell
python tools/evaluate_repeatability.py `
  --folder repeated `
  --pattern "DSC*.JPG"
```

The report includes `std_x_mm`, `std_y_mm`, radial standard deviation, and
coordinate range for each matched ball. Images with changed ball layouts are
not valid repeatability inputs.
