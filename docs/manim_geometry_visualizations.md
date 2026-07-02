# Manim geometry visualizations

These scenes explain why cloth-plane warping is not a correct final ball
geometry model, and how source-image ray/plane projection should replace it.

Scene file:

```text
visualizations/ball_geometry_scenes.py
```

## Install

Manim is optional:

```powershell
python -m pip install -e ".[viz]"
```

If the system is missing render dependencies, follow the Manim Community
installation instructions for Windows. Manim uses Cairo/Pango/FFmpeg through
its Python package stack.

## Rendering status

The old one-off Manim renderer script has been removed from the active v1
codebase. The scene source remains documented here as historical design input;
rendering can be reintroduced later as a v1 visualization command if these
animations become useful again.

Scene names from the original plan:

- `CoordinateSystems`
- `WhyHomographyFails`
- `RayIntersection`
- `TouchingBallValidation`
- `EndToEndPipeline`

## Scene plan

### 1. CoordinateSystems

Shows table X/Y/Z, cloth plane `Z=0`, intermediate test plane `Z=13.1`,
ball-center plane `Z=26.25`, and ball-top plane `Z=52.5`.

### 2. WhyHomographyFails

Shows source image plane, camera, cloth plane, one center ball, one edge ball,
and a warped debug panel where the edge ball becomes shifted/stretched. The
point is explicit: the homography is valid for the cloth plane, not for a
sphere.

### 3. RayIntersection

Shows one source pixel, the image ray, intersections with `Z=0`, `Z=26.25`,
and `Z=52.5`, and the XY difference caused by using the wrong plane.

### 4. TouchingBallValidation

Shows two touching balls with expected center distance `52.5 mm`, then a small
comparison table of distance error under several Z assumptions.

### 5. EndToEndPipeline

Shows the target future pipeline:

```text
source image
→ undistort
→ detect/refine source center
→ image ray
→ world intersection
→ table_state.json
→ top-down table visualization
```

The key takeaway is that final table position should come from:

```text
source image point + camera model + ray/plane intersection
```

not from:

```text
warped-image circle center alone
```
