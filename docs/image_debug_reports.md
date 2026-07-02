# Per-image visual debug reports

The report generator creates one folder per real input photo:

```text
outputs/reports/<image_stem>/
  report.html
  report.json
  01_source_detection.png
  02_warped_detection.png
  03_source_zoom_grid.png
  04_geometry_selected_ball.png
  05_error_comparison.png
  06_physical_validation.png
  07_pipeline_summary.png
```

## Single image

```powershell
python tools/generate_image_report.py `
  --image Media/03_near_cushions/DSC00529.JPG `
  --output outputs/reports `
  --selected-ball auto
```

Equivalent v1 package entrypoint:

```powershell
python -m snookerhelp.tools.generate_reports image `
  --image Media/03_near_cushions/DSC00529.JPG `
  --output outputs/reports
```

Select a specific ball:

```powershell
python tools/generate_image_report.py `
  --image Media/03_near_cushions/DSC00529.JPG `
  --ball-id 8
```

## Dataset

```powershell
python tools/generate_dataset_reports.py `
  --glob "Media/**/*.JPG" `
  --output outputs/reports
```

This also writes:

```text
outputs/reports/index.html
outputs/reports/dataset_reports.json
```

Use `--limit 3` for a quick smoke run.

Equivalent v1 package entrypoint:

```powershell
python -m snookerhelp.tools.generate_reports dataset `
  --glob "Media/**/*.JPG" `
  --output outputs/reports
```

## Visual OK/NOK feedback workflow

The primary workflow is the schema-driven v1 local interactive review UI:

```powershell
python tools/review_reports.py --reports outputs/reports --port 8770
```

Equivalent module entrypoint:

```powershell
python -m snookerhelp.tools.review --reports outputs/reports --port 8770
```

This opens a v1 review tool with product-language sections:

- Pixels
- Image evidence
- Physical model
- Final estimate
- Confidence
- Manual correction

The UI saves human feedback beside each immutable report:

```text
outputs/reports/<image_stem>/report.json   # algorithm output, do not edit
outputs/reports/<image_stem>/review_v1.json # v1 human review and manual corrections
```

`report.json` is still the immutable machine report. The v1 server adapts it to
`snookerhelp.table_state.v1` before the browser sees it. Legacy `review.json`
files remain readable, but new v1 review saves go to `review_v1.json`.

`review_v1.json` stores the human layer: OK/NOK/needs-review/missing decisions, issue
tags, confidence, comments, manual center/radius/ellipse/cushion-line edits, and
missing-ball hints. Manual edits do not overwrite algorithm output.

For elongated edge/cushion balls, do not treat any 2D overlay as ground truth.
Sometimes the correct physical center is ambiguous in the source image. The v1
UI groups the underlying evidence into image evidence, physical model, final
estimate, and confidence. It should not ask the user to approve prototype
candidate names.

The interactive UI also shows both `legacy_review_confidence` and the
experimental `physics_first_review_confidence` and
`physics_c_only_review_confidence`. The displayed auto confidence uses a
physics score only when the sphere projection is plausible.

See [ball_geometry_model.md](ball_geometry_model.md) for the current fitting
model and why image evidence and physical model evidence affect confidence
differently.

### Missing balls

Use the full-table source panel:

1. choose the label in the `Add missing ball` controls;
2. optionally type a note;
3. click `Add missing ball: off` so it changes to `click table`;
4. click the missing ball center in the source image.

The v1 UI adds an `M1`, `M2`, ... marker and stores missing-ball feedback under
`missing_balls` in `review_v1.json`.

### Cushion-line edits

For a selected ball:

1. click `Line` or open the `Cushion Line` tab;
2. drag the cyan line handles shown inside the zoom crop;
3. save the review.

The handles are local to the visible crop. The saved correction is stored as
`manual_correction.cushion_line_px`.

### Confidence

The confidence slider is optional. Auto confidence is shown only as context.
`human_confidence` is exported only after the slider is explicitly moved.
If you do not use the slider, exported feedback contains
`"human_confidence": null`.

The static `report.html` page is now a read-only QA artifact. It does not store
review feedback and does not export browser-local feedback. Use
`tools/review_reports.py` for all OK/NOK decisions, missing balls, and manual
corrections.

### Diagnostic evidence maps

Regenerated v1 reports include per-ball diagnostic map PNGs under:

```text
outputs/reports/<image_stem>/evidence_maps/
```

In the v1 review UI, select a ball and use the selected-ball crop background
selector to inspect:

- source image;
- grayscale edge;
- Lab Delta-E;
- chroma difference;
- ball-vs-cloth probability;
- physical projection band;
- combined boundary score.

These maps are diagnostic and scoring evidence. They do not create recovered
cyan boundary points. The current active boundary layer is still accepted white
points plus rejected red outliers.

## Export reviewed feedback as a dataset

After reviewing reports:

```powershell
python tools/export_review_feedback.py `
  --reports outputs/reports `
  --output data/review_feedback/dataset_feedback_v1.jsonl
```

The normal export command now writes v1 `ReviewFeedback` records. The old row-
per-ball legacy exporter source has been removed; migrate old JSONL feedback
with the command below if needed.

Existing exported feedback can be migrated to v1 schema:

```powershell
python -m snookerhelp.tools.migrate_feedback `
  --input data/review_feedback/dataset_feedback.jsonl `
  --output data/review_feedback/dataset_feedback_v1.jsonl
```

For new v1 review files, export directly:

```powershell
python tools/export_review_feedback.py `
  --reports outputs/reports `
  --output data/review_feedback/dataset_feedback_v1.jsonl
```

## Benchmark circle-first vs physics-first scoring

After regenerating reports:

```powershell
python tools/benchmark_model_scoring.py `
  --reports outputs/reports `
  --output outputs/model_scoring_benchmark
```

This summarizes whether physical-model-first scores raise or lower automatic
confidence across the dataset.

## Scenario metadata

Physical validation becomes more useful when you provide scenario ground truth:

```powershell
python tools/generate_image_report.py `
  --image Media/03_near_cushions/DSC00529.JPG `
  --scenario data/scenarios/cushion_touch.yaml
```

Example:

```yaml
touching_pairs:
  - ball_a: 8
    ball_b: 9
    expected_distance_mm: 52.5
    notes: visually touching pair

cushion_touches:
  - ball_id: 8
    cushion: bottom
    expected_radius_mm: 26.25

spots:
  blue:
    x_mm: 1784.5
    y_mm: 889.0

spot_tests:
  - spot: blue
    ball_id: 6
```

Supported validation kinds:

- touching-ball distance: expected center distance defaults to `52.5 mm`;
- cushion touch: expected center-to-cushion distance defaults to `26.25 mm`;
- spot test: expected coordinate comes from `spots`.

If no scenario is supplied, the physical validation panel shows automatically
found candidate touching pairs. Those are useful for visual inspection but are
not ground truth.

## How to read the panels

1. `01_source_detection.png`: real source image with rough source centers,
   refined source centers, fitted source radii, labels, residuals, and the
   source-image table/cushion boundary used by the homography.
2. `02_warped_detection.png`: cloth-plane debug warp. This is not a physical
   ball-shape view.
3. `03_source_zoom_grid.png`: close crops for every detected ball. Each tile
   shows rough/refined center, fitted circle, residual, source boundary sample
   count, rough-to-refined shift, nearest cushion, and whether the source fit
   was accepted or the detector fell back.
4. `04_geometry_selected_ball.png`: static source ROI plus ray/plane geometry
   and Z-plane coordinate comparison for one selected ball. The HTML page also
   has an interactive selected-ball geometry section where you can switch balls.
5. `05_error_comparison.png`: arrows from old warped-derived coordinates to
   source-refined projected coordinates.
6. `06_physical_validation.png`: pass/warn/fail physical constraints.
7. `07_pipeline_summary.png`: one-page pipeline overview.

The report is intended to answer:

- Did the detector find the balls?
- Where did it place the source-image centers?
- How are those image points converted into table coordinates?
- How different are old warped-derived and source-refined projected positions?
- Do the coordinates satisfy physical constraints?
