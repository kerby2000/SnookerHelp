# Snooker Vision Architecture Document

## Document status

This document is the project/hardware architecture reference. It still contains
the original first-prototype plan for historical context.

For the current ball-fitting implementation, start with:

- [docs/ball_geometry_model.md](ball_geometry_model.md)
- [docs/image_debug_reports.md](image_debug_reports.md)
- [docs/approximate_camera_model.md](approximate_camera_model.md)

Current implementation summary:

```text
source photo
  -> warped cloth-plane view for rough ball detection
  -> source-image crop refinement for each ball
  -> source-image edge/mask/ellipse evidence
  -> approximate physical sphere projection
  -> physical-model-first confidence scores
  -> source center projected through approximate_pinhole_from_corners
  -> report.json + review UI
```

The warped image is now treated as a debug/rough-detection view, not as the
final physical measurement surface for ball shapes.

## 1. Project Goal

The goal is to build a computer-vision system for a snooker training assistant.

The first milestone is not shot recommendation or projection. The first milestone is:

**Recognize the full snooker table state from an overhead camera image: table boundaries, cushions, pockets, and all visible balls with their positions in table coordinates.**

Later milestones will use this table state to calculate and project shot trajectories.

The system should eventually work with a fixed overhead industrial camera, but while hardware is still arriving, development is done using high-resolution still images from a Sony Alpha 6000 camera.

---

## 2. Current Development Hardware

### 2.1 Temporary development camera: Sony Alpha 6000

Used while waiting for final Basler lens/hardware.

Camera:

* Sony Alpha 6000 / ILCE-6000
* APS-C sensor
* 24 MP still images
* Still image resolution: 6000 × 4000 px
* Lens: Sony E 11 mm F1.8 APS-C E-mount lens
* Remote capture via Sony Imaging Edge Remote
* HDMI capture gives only Full HD preview and is not used for final measurement

Current workflow:

1. Sony camera is mounted above the snooker table.
2. HDMI preview may be used for live aiming/framing.
3. Sony Remote app triggers full-resolution still capture.
4. JPEG image is automatically copied to a PC folder.
5. Python software should watch this folder, load the latest image, and process it.

Important limitation:

* HDMI preview is only for live preview.
* 24 MP stills come through Sony Remote / file transfer.
* Sony is useful for static image development, but not ideal for the final real-time installation.

Recommended Sony settings for development:

* Mode: Manual
* Aperture: F4 baseline
* Shutter: approximately 1/25 s is acceptable for static table images
* ISO: fixed, low if possible
* White balance: fixed, not auto
* Focus: manual
* JPEG quality: Fine or Extra Fine
* RAW: not required for normal processing
* DRO/HDR/auto image enhancement: off if possible
* Camera position: fixed after calibration

Why F4:

* F1.8 gives brighter images, but more vignetting and softer corners.
* F4 produces cleaner edges and better calibration/recognition data.
* Since the first milestone processes static table state after balls stop moving, longer exposure is acceptable.

---

## 3. Future / Final Camera Hardware

### 3.1 Final industrial camera

Camera:

* Basler ace 2 a2A5320-23ucPRO Color USB3
* Resolution: 16.1 MP
* Image size: 5320 × 3032 px
* Sensor format: 1.1"
* Sensor active size approximately 14.6 × 8.3 mm
* USB3 industrial camera
* Controlled through Basler pylon / pypylon
* Intended final use: deterministic live/still image acquisition

Advantages over Sony:

* Direct SDK control
* More deterministic exposure/gain/white balance
* Full-resolution streaming or triggered capture
* Better integration with final installed system
* More machine-vision-friendly workflow

### 3.2 Existing Basler lens, not sufficient

Existing lens:

* BASLER C11-0824-12M
* 8 mm / 8.5 mm class
* 1.1" C-mount
* 12 MP class

Problem:

* It does not cover the full snooker table from the available camera height.
* It is too narrow for the required field of view.

### 3.3 Ordered future lens

Ordered eBay lens:

* C-ML-U0618SR-18C
* Moritex / Cognex industrial lens
* 6 mm class, actual focal length approximately 6.25 mm
* C-mount
* 1.1" image circle
* 12 MP class
* Expected to cover the table at camera height approximately 1600–1800 mm
* Best mounting height should be as close as practical to 1750–1800 mm for comfortable margin

Expected behavior:

* This lens is not a fisheye lens.
* It is a wide-angle industrial lens.
* Lens distortion correction will still be necessary.
* Final accuracy depends on calibration, optical sharpness, lighting, and rigid mounting.

---

## 4. Computing Hardware

Current project PC:

* MINISFORUM Mercury EM680 Mini PC
* Ryzen 7 6800U class CPU
* Integrated AMD Radeon graphics
* No dedicated NVIDIA GPU

Expected suitability:

Good for:

* Basler image capture
* Sony folder-watcher workflow
* OpenCV processing
* camera calibration
* table homography
* classical ball detection
* color classification
* static image processing
* low/medium frame-rate live analysis
* debug UI

Not ideal for:

* local training of large neural networks
* real-time high-resolution YOLO segmentation
* SAM-style segmentation at full resolution
* heavy multi-camera AI inference

Software should therefore begin with classical OpenCV methods and only add neural networks later if needed.

---

## 5. Lighting

Lighting should be treated as part of the measurement system.

Target lighting:

* Fixed brightness
* Fixed color temperature
* Flicker-free
* Diffuse
* Dimmable if possible
* No automatic variation during capture

Important notes:

* Uneven illumination and vignetting are visible in current Sony images.
* Do not rely on global RGB thresholds.
* Use local color sampling, HSV/Lab color spaces, or brightness normalization.
* Ball highlights and shadows must be handled robustly.
* Future table lights should preferably be 4000–5000K and fixed during calibration and operation.

---

## 6. Calibration Targets and Reference Markers

### 6.1 Large ChArUco board

Planned / recommended:

* Large ChArUco target, ideally 800 × 600 mm
* Used for lens calibration and distortion correction
* Should be rigid and flat
* Matte surface preferred

Purpose:

* Estimate camera intrinsics
* Estimate lens distortion
* Validate image sharpness and calibration quality

### 6.2 Small fixed table markers

Planned / recommended:

* Round reflective survey targets
* Preferred size: 40 mm if physically acceptable
* 30 mm acceptable if space is limited
* Neutral grey/silver reflective with black cross/circle pattern preferred

Placement:

* Easier to place on wooden rail / near pockets than on cloth
* Should not interfere with play
* Should be rigidly attached and not move

Important geometric issue:

Markers placed on the wooden rail are not in the same plane as the balls. The balls roll on the cloth plane. Rail markers may be 20–50 mm above the cloth plane.

Therefore:

* Do not use rail markers as simple 2D homography points on the cloth plane.
* Treat rail marker centers as known 3D points.
* Measure each marker center in table coordinates: X, Y, Z.
* Use them for camera pose estimation / drift correction.
* Intersect detected ball image rays with the cloth plane Z=0 to get corrected ball positions.

Simpler temporary method:

Before implementing full 3D pose correction, allow manual 4-corner table calibration from clicked points on the playing surface. This is acceptable for the first prototype.

---

## 7. Coordinate Systems

The software should clearly separate coordinate systems.

### 7.1 Image coordinates

Raw camera image coordinates:

```text
u, v in pixels
origin: top-left of image
u: horizontal pixel coordinate
v: vertical pixel coordinate
```

### 7.2 Undistorted image coordinates

Image coordinates after lens distortion correction.

### 7.3 Table coordinates

Physical table coordinate system:

```text
X, Y in millimeters
Z in millimeters

Z = 0: cloth / ball rolling plane
X: table length direction
Y: table width direction
```

Suggested origin:

```text
origin = one inner corner of the playing surface
X axis = along table length
Y axis = across table width
```

Approximate snooker playing surface dimensions:

```text
length ≈ 3569 mm
width  ≈ 1778 mm
```

Exact values should be measured from the actual table and stored in configuration.

### 7.4 Warped table image

For easier processing, create a top-down rectified table image using a configurable mm-to-pixel scale.

Possible scale:

```text
1 px/mm for full table debug image
2 px/mm for more precise processing if memory/performance allows
```

At 1 px/mm:

```text
warped image ≈ 3569 × 1778 px
```

This is manageable and gives direct metric interpretation.

---

## 8. Software Design Principles

### 8.1 Camera-agnostic pipeline

The processing pipeline should not depend on whether the image came from Sony or Basler.

Define image sources:

1. Sony folder watcher source
2. Single image file source
3. Basler pypylon source, later
4. Video/HDMI source, only for preview/experiments

Common processing pipeline:

```text
image source
→ load/capture image
→ optional undistort when calibrated intrinsics are available
→ table calibration / cloth-plane warp
→ rough ball detection in warped view
→ source-image evidence extraction per ball
→ candidate model comparison
→ ball classification
→ coordinate projection through camera model
→ overlay/debug output
→ JSON table state output
```

### 8.2 Classical OpenCV first

Do not start with YOLO.

The first version should use classical computer vision because:

* Camera is fixed
* Table is fixed
* Lighting is controlled
* Balls are circular
* Ball colors are known
* The goal is precise center measurement, not generic object recognition

Deep learning can be added later for difficult cases.

### 8.3 Detection accuracy over visual beauty

The system should optimize for:

* repeatable ball centers
* correct table coordinates
* stable calibration
* clear debug output

It should not optimize for pretty photographic rendering.

---

## 9. Proposed Repository Structure

```text
snooker_vision/
  README.md
  pyproject.toml

  docs/
    snooker_vision_architecture.md

  configs/
    sony_dev.yaml
    basler_ace2.yaml
    table_model.yaml
    detector_classical.yaml

  data/
    raw/
      sony/
      basler/
    calibration/
    processed/
    debug_outputs/

  calibration/
    calibrate_lens_charuco.py
    calibrate_table_manual.py
    calibrate_table_markers_3d.py
    validate_calibration.py

  capture/
    watch_sony_folder.py
    capture_basler.py
    basler_live_view.py

  vision/
    image_source.py
    undistort.py
    table_model.py
    table_warp.py
    ball_detect_classical.py
    ball_color_classifier.py
    circle_fit.py
    marker_detect.py
    state_estimator.py
    overlay.py

  tools/
    process_latest_image.py
    process_single_image.py
    click_table_corners.py
    inspect_colors.py
    measure_pixel_scale.py

  tests/
    test_geometry.py
    test_circle_fit.py
    test_color_classifier.py
```

---

## 10. Configuration Files

Use YAML files for user-editable settings.

### 10.1 Sony development config

```yaml
image_source:
  type: folder_watch
  folder: "C:/Users/lukin/Pictures"
  filename_pattern: "DSC*.JPG"

camera:
  name: "Sony Alpha 6000"
  lens: "Sony E 11mm F1.8"
  image_width: 6000
  image_height: 4000
  mode: "development_stills"
```

### 10.2 Basler future config

```yaml
image_source:
  type: basler_pypylon

camera:
  name: "Basler ace 2 a2A5320-23ucPRO"
  image_width: 5320
  image_height: 3032
  sensor_format: "1.1 inch"
  lens: "C-ML-U0618SR-18C"
  interface: "USB3"

capture:
  exposure_us: null
  gain: 0
  white_balance: "fixed"
  pixel_format: "BGR8"
```

### 10.3 Table model config

```yaml
table:
  name: "home_snooker_table"
  playing_surface:
    length_mm: 3569
    width_mm: 1778

coordinates:
  origin: "bottom_left_inner_playing_surface"
  x_axis: "table_length"
  y_axis: "table_width"
  z_axis: "up_from_cloth"

balls:
  diameter_mm: 52.5
  radius_mm: 26.25

warp:
  px_per_mm: 1.0
```

---

## 11. First Implementation Milestone

Historical note: this milestone is already implemented and has been extended
with source-image refinement, report generation, review UI, physical validation,
and approximate pinhole camera geometry.

The first implementation milestone should be:

**Given one full-resolution Sony JPEG, manually click the four playing-surface corners once, then detect balls and output their approximate table coordinates.**

This milestone does not require:

* ChArUco calibration
* Basler camera
* YOLO
* projector
* real-time tracking

Inputs:

* One Sony full-resolution JPEG
* Manual table corner clicks
* Approximate table size in mm

Outputs:

1. Debug image with:

   * detected table region
   * detected balls
   * ball center points
   * ball color labels
2. JSON file with:

   * image filename
   * table corners
   * ball list
   * ball coordinates in mm
   * confidence/debug metrics

Example JSON:

```json
{
  "source_image": "DSC00524.JPG",
  "table": {
    "length_mm": 3569,
    "width_mm": 1778,
    "corner_points_px": [[0, 0], [1, 0], [1, 1], [0, 1]]
  },
  "balls": [
    {
      "id": 1,
      "class": "white",
      "x_mm": 421.3,
      "y_mm": 832.7,
      "radius_mm": 26.2,
      "confidence": 0.94
    },
    {
      "id": 2,
      "class": "red",
      "x_mm": 1120.5,
      "y_mm": 455.1,
      "radius_mm": 26.5,
      "confidence": 0.90
    }
  ]
}
```

---

## 12. Classical Ball Detection Pipeline

Historical note: the rough detection still uses the warped cloth-plane view, but
final source evidence is now measured in the original source image. See
[ball_geometry_model.md](ball_geometry_model.md) for the current fitting model.

Initial algorithm:

1. Load image.
2. Optionally resize for preview only.
3. Use full resolution for measurement.
4. Apply manual table homography.
5. Warp image to metric top-down image.
6. Create a table/cloth mask.
7. Detect non-green objects on the cloth.
8. Filter candidate blobs by:

   * size
   * circularity
   * expected ball radius
   * color saturation
   * distance from rail/pockets
9. Refine ball evidence in the original source image:

   * source-image circle baseline/fallback;
   * mask contour/centroid diagnostics;
   * radial/edge observed ellipse;
   * approximate or calibrated sphere projection.
10. Classify ball color:

* sample inner region near ball center
* avoid specular highlight
* use HSV/Lab color features
* classify into known snooker colors

11. Output ball list.

Known snooker ball classes:

```text
white
red
yellow
green
brown
blue
pink
black
unknown
```

Notes:

* Multiple red balls do not need unique identities in the static state.
* White ball must be reliably distinguished.
* Pink/brown/red may be challenging under changing lighting.
* Use real ball samples from the actual table, not only a color chart.

---

## 13. Table Detection and Calibration Strategy

### 13.1 First version: manual table corners

Implement a simple click tool:

```text
click four inner playing-surface corners
save to calibration/table_manual.yaml
```

Use these four points to compute a homography from image pixels to table mm.

This is sufficient for early development.

### 13.2 Later version: ChArUco lens calibration

Use ChArUco images to estimate:

* camera matrix
* distortion coefficients
* reprojection error

Save as:

```text
calibration/camera_intrinsics.yaml
```

### 13.3 Later version: fixed marker pose estimation

Use fixed survey targets on rails.

Each marker has known 3D coordinates:

```yaml
markers:
  - id: "M01"
    x_mm: 100
    y_mm: -50
    z_mm: 35
  - id: "M02"
    x_mm: 3460
    y_mm: -50
    z_mm: 35
```

Use detected marker centers and known 3D points to solve camera pose.

Then for each detected ball center pixel:

1. Undistort the pixel.
2. Cast a ray from camera through pixel.
3. Intersect ray with Z=0 cloth plane.
4. Return X/Y table coordinates.

This is the correct method when reference markers are not on the ball plane.

---

## 14. Color Calibration Strategy

A cheap 24-patch color card can be used for practical debugging, but final ball classification should be trained from real snooker balls under actual table lighting.

Recommended process:

1. Capture image with all ball colors placed separately.
2. Manually or automatically sample each ball.
3. Store representative HSV/Lab values.
4. Use these values as class prototypes.
5. Optionally normalize image color using gray/white patch from color card.

Do not rely on printed RGB values from a cheap color card as absolute truth.

---

## 15. Performance Strategy

The first implementation should process one still image at a time.

Later live behavior:

* use low-resolution live preview for UI
* run full-resolution detection only when needed
* for snooker, final table state after balls stop is more important than every moving frame

Possible final operating mode:

```text
preview stream: downscaled
measurement frame: full resolution
processing trigger: manual button or automatic "balls stopped moving"
```

The EM680 should be able to process stills and low-frame-rate live recognition with OpenCV. Do not assume full-resolution 23 fps heavy processing.

---

## 16. Future Deep Learning Stage

Add YOLO/segmentation only after classical pipeline is working.

Possible model:

* YOLO11 segmentation
* classes: ball colors, pockets, cushion/playing area
* use masks to improve ball candidate detection
* still use classical circle fitting for final center accuracy

Important:

Do not use YOLO bounding box centers as final ball centers. They are too crude for trajectory projection.

Correct future pipeline:

```text
YOLO/SAM/segmentation mask
→ contour extraction
→ circle fitting
→ center in pixels
→ table coordinate conversion
```

---

## 17. Acceptance Criteria for First Prototype

A successful first prototype should:

1. Watch Sony Remote output folder.
2. Detect new JPEG files.
3. Load latest full-resolution image.
4. Allow manual table-corner calibration.
5. Warp table to metric top-down view.
6. Detect most separated balls.
7. Classify basic colors.
8. Save overlay image.
9. Save JSON table state.

Minimum accuracy target for first prototype:

* visually correct detections
* ball center error roughly within a few millimeters
* robustness enough for spread-out balls

Do not require perfect clustered-ball detection in version 1.

---

## 18. Known Challenges

1. Uneven illumination across table.
2. Ball shadows.
3. Specular highlights on balls.
4. Red/pink/brown color confusion.
5. Balls touching or close together.
6. Balls close to cushions or pockets.
7. Camera/lens distortion.
8. Table rail markers are not on the ball plane.
9. Sony and Basler images will have different color response.
10. Final projector alignment is a later separate calibration problem.

---

## 19. Immediate Next Codex Task

Historical note: this immediate task was the original baseline task. The current
implementation already has an experimental physics-first model-scoring layer
that compares legacy circle-first confidence against Candidate-D-first
confidence. The remaining next step is to let calibrated sphere geometry solve
the final center, not only score the current center.

Implement the first working OpenCV prototype using Sony still images.

Focus on:

* folder watcher
* manual table-corner calibration
* table warp
* first classical ball detector
* debug overlay
* JSON output

Do not implement:

* YOLO
* projector
* Basler pypylon
* 3D rail-marker pose
* ChArUco calibration

Those come after the first baseline works.
