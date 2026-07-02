# ChArUco calibration workflow

This project is ready for the CALITAR `CALI100020TAR.5` board:

- target: `15 x 20`;
- OpenCV board size used by this project: `squares_x=20`, `squares_y=15`;
- checker size: `32.0 mm`;
- marker size: `24.0 mm`;
- dictionary: `DICT_5X5_1000`;
- board config: `configs/charuco_calitar_cali100020tar5.yaml`.

The board photo set is not available yet, so the current development config
uses `approximate_pinhole_from_corners`. That gives approximate ray/plane and
sphere-projection evidence before calibration, but it does not replace real
ChArUco intrinsics/distortion/pose. The tools below can be run once the board
arrives.

## 1. Capture intrinsics images

Capture 15-30 sharp stills of the board at different positions and tilts. Keep
the same camera, lens, focal length, aperture, and focus that will be used over
the table.

Example folder:

```powershell
Media/calibration/charuco_intrinsics/*.JPG
```

Run:

```powershell
python tools/calibrate_camera_charuco.py `
  --images "Media/calibration/charuco_intrinsics/*.JPG" `
  --board configs/charuco_calitar_cali100020tar5.yaml `
  --output configs/camera_intrinsics_charuco.yaml
```

Output:

```text
configs/camera_intrinsics_charuco.yaml
```

This contains:

- image size;
- camera matrix;
- distortion coefficients;
- RMS reprojection error;
- per-frame detected ChArUco corner counts.

## 2. Estimate camera pose relative to the table

Place the ChArUco board flat on the cloth. Measure where the board coordinate
origin is in table coordinates.

Important: OpenCV's ChArUco board origin is the board's top-left square corner
in board coordinates. Board `+X` follows the long board direction and board
`+Y` follows the short board direction.

Example, if the board origin is at table coordinate `(500, 400, 0)` and the
board is aligned with table X/Y:

```powershell
python tools/estimate_table_pose_charuco.py `
  --image Media/calibration/charuco_table_pose/pose_001.JPG `
  --intrinsics configs/camera_intrinsics_charuco.yaml `
  --board configs/charuco_calitar_cali100020tar5.yaml `
  --board-origin-table-mm 500 400 0 `
  --board-x-axis-table 1 0 0 `
  --board-y-axis-table 0 1 0 `
  --output configs/camera_model_charuco_table.yaml
```

If the board is rotated on the table, change the board axes. For example, if
board `+X` points along table `+Y` and board `+Y` points toward table `-X`:

```powershell
--board-x-axis-table 0 1 0 --board-y-axis-table -1 0 0
```

## 3. Enable calibrated camera mode

In `configs/sony_dev.yaml`, replace the manual camera model with:

```yaml
pipeline:
  camera_model:
    config_file: "configs/camera_model_charuco_table.yaml"
```

Then regenerate reports:

```powershell
python tools/generate_dataset_reports.py --glob "Media/**/*.JPG" --output outputs/reports
```

## What changes after calibration

The review UI's physical sphere projection changes from approximate to
calibrated.

The projected sphere curve is the image projection of a known-radius snooker
ball under the calibrated camera model. The report will score observed
mask/boundary points against that predicted curve. With real calibration, this
becomes a much stronger confidence signal for difficult edge and cushion balls
instead of relying only on manual visual judgment.
