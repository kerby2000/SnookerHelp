# Physical validation tools

These tools validate detector geometry using physical constraints instead of
manual center annotation.

## Repeatability

Use this only for repeated photos of an unchanged ball layout:

```powershell
python tools/evaluate_repeatability.py `
  Media/02_random_balls/DSC00524.JPG `
  Media/02_random_balls/DSC00525.JPG `
  --output-dir data/repeatability_reports/my_repeatability
```

The report includes per-ball `std_x_mm`, `std_y_mm`, `radial_std_mm`,
`max_range_mm`, and region grouping. It warns when frames look like different
layouts because of low match fraction or large median displacement.

## Touching-ball pairs

Auto-find likely touching pairs from processed detector JSON:

```powershell
python tools/evaluate_touching_balls.py `
  data/debug_outputs/physical_validation_samples/DSC00529/DSC00529_state.json `
  data/debug_outputs/physical_validation_samples/DSC00534/DSC00534_state.json `
  --auto `
  --center-mode compare
```

`--center-mode warped` uses the original warped-image fitted centers.
`--center-mode source-refined` uses source-image centers projected through the
current camera model. In development this is usually
`approximate_pinhole_from_corners`, so the result is still approximate, but it
does model height-dependent ray/plane projection. `--center-mode compare`
reports warped and source-refined results for the same physical pairs and
summarizes whether source refinement improves the distance error.

For Z-plane projection comparison:

```powershell
python tools/evaluate_touching_balls.py `
  data/debug_outputs/physical_validation_samples/DSC00529/DSC00529_state.json `
  data/debug_outputs/physical_validation_samples/DSC00534/DSC00534_state.json `
  --auto `
  --center-mode z-planes
```

`--center-mode z-planes` compares the configured source-image projections for
`Z=0`, `13.1`, `26.25`, `39.4`, and `52.5 mm`. `--center-mode all` includes
warped centers, the default source-refined coordinate, and all Z planes.

When Z-plane rows are present, outputs include:

```text
touching_pairs_z_plane_summary.csv
touching_pairs_z_plane_region_heatmap.png
```

The CSV ranks Z planes by median absolute touching-distance error per table
region. The heatmap highlights the best Z for each region. In
`manual_homography` mode all Z planes intentionally produce the same XY. In
`approximate_pinhole_from_corners` and future `calibrated_pinhole` modes, the
table should show which height best explains the physical constraints.

Explicit pair file:

```yaml
pairs:
  - file: DSC00529
    ball_a: 8
    ball_b: 9
    notes: bottom cushion pair
  - file: DSC00534
    ball_a: 1
    ball_b: 2
```

Run it with:

```powershell
python tools/evaluate_touching_balls.py `
  data/debug_outputs/physical_validation_samples/DSC00529/DSC00529_state.json `
  --pairs-file pairs.yaml
```

## Rack / cluster red-ball nearest-neighbor validation

For triangle/rack images:

```powershell
python tools/evaluate_touching_balls.py `
  data/debug_outputs/physical_validation_samples/DSC00540/DSC00540_state.json `
  --rack-reds
```

This reports nearest-neighbor distances among detected red balls. The expected
touching distance defaults to `52.5 mm`.

The warped overlay is a cloth-plane rectification. It is useful for debugging
candidate locations, but ball outlines near edges/corners are not expected to
remain circular in that view.

See `docs/ball_geometry_model.md` for the ray/plane geometry and camera-model
configuration shape.

## Cushion-touch validation

Inline:

```powershell
python tools/evaluate_cushion_touch.py `
  data/debug_outputs/physical_validation_samples/DSC00529/DSC00529_state.json `
  --touch 8:bottom `
  --touch 1:left
```

YAML:

```yaml
touches:
  - file: DSC00529
    ball_id: 8
    cushion: bottom
  - file: DSC00529
    ball_id: 1
    cushion: left
```

Expected center-to-cushion distance defaults to the ball radius, `26.25 mm`.

## Spot-position validation

Spot coordinates use the table coordinate system in millimeters:

```yaml
spots:
  blue:
    x_mm: 1784.5
    y_mm: 889.0
  pink:
    x_mm: 2676.75
    y_mm: 889.0
```

Mappings can be separate:

```yaml
mappings:
  - file: DSC00540
    spot: blue
    ball_id: 6
  - file: DSC00540
    spot: pink
    class: pink
```

Run:

```powershell
python tools/evaluate_spot_positions.py `
  data/debug_outputs/physical_validation_samples/DSC00540/DSC00540_state.json `
  --spots spots.yaml `
  --mappings mappings.yaml
```

If `ball_id` is omitted and `class` is provided, the nearest detected ball of
that class to the expected spot is used.

## Region grouping

All validation reports group errors by:

- `center`
- `left_edge`
- `right_edge`
- `top_edge`
- `bottom_edge`
- `pockets/corners`

The default edge margin is two ball diameters. Override it with
`--region-margin-mm`.

## Detector-seeded annotation

The annotation tool now starts from detections by default:

```powershell
python tools/annotate_ball_centers.py --image Media/02_random_balls/DSC00524.JPG
```

Useful controls:

- `Shift+A`: accept current detections and save
- left click: add a point
- right click: delete nearest point
- `Shift+left click`: relabel nearest point with the selected color
- `+` / `-`: zoom
- arrows or `w`/`a`/`s`/`d`: pan
- `0`: reset view

Use `--no-detector-seed` to start from an empty annotation set.
