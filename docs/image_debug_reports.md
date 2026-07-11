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

## Interactive evidence and ground-truth workflow

The primary workflow is the schema-driven v1 local interactive review UI:

```powershell
python tools/review_reports.py --reports outputs/reports --port 8770
```

Equivalent module entrypoint:

```powershell
python -m snookerhelp.tools.review --reports outputs/reports --port 8770
```

This opens the schema-driven v1 workbench with:

- Pixels
- Evidence layers
- Image evidence
- Physical model
- Final estimate
- Confidence
- Ball statistics
- Live evidence experiment
- Perfect ellipse ground truth

The machine report remains immutable:

```text
outputs/reports/<image_stem>/report.json   # algorithm output, do not edit
```

`report.json` is still the immutable machine report. The v1 server adapts it to
`snookerhelp.table_state.v1` before the browser sees it.

The perfect-ellipse editor stores independent image-space ground truth under:

```text
benchmarks/annotations/<image_stem>.json
```

Use `Copy selected fit` to initialize the annotation, then drag the cyan center,
major-axis, and minor-axis handles over the source-color crop. Saving records a
`snookerhelp.ground_truth.v1` ellipse; it never changes `report.json` or the
production source center. This is useful before camera calibration because it
measures image boundary extraction directly. It is not automatically the true
3D sphere center when the silhouette is occluded.

The live experiment panel calls the backend and recomputes the evidence map,
accepted/rejected boundary points, ellipse, and decomposed diagnostic score.
Controls include evidence map, selected/red/specific-ball color reference,
probability scaling, radial search range, angular sample count, outlier filtering,
and neighbor filtering. Its result appears as a transient evidence row and does
not overwrite machine output.

The two scores have different meanings:

- evidence-view score: internal image/physics agreement; not ground truth;
- annotation score: measured ellipse agreement with the saved perfect ellipse.

For elongated edge/cushion balls, do not treat any 2D overlay as ground truth.
Sometimes the correct physical center is ambiguous in the source image. The v1
UI groups the underlying evidence into image evidence, physical model, final
estimate, and confidence. It should not ask the user to approve prototype
candidate names.

See [ball_geometry_model.md](ball_geometry_model.md) for the current fitting
model and why image evidence and physical model evidence affect confidence
differently.

### Confidence

Current confidence is algorithmic and decomposed. It is not calibrated accuracy
and it does not use the perfect ellipse. Ground-truth comparison is reported
separately so an algorithm cannot award itself a high score merely by agreeing
with its own physical prior.

The static `report.html` page remains a read-only QA artifact. Use the local v1
workbench for experiments and perfect-ellipse annotations.

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

## Benchmark detector ellipses against perfect ellipses

```powershell
python tools/benchmark_ellipse_annotations.py `
  --report outputs/reports/DSC00540/report.json `
  --annotations benchmarks/annotations/DSC00540.json `
  --output benchmarks/results/DSC00540 `
  --tolerance-px 3
```

The JSON and CSV report source-center error, ellipse contour RMS, axis/angle
errors, annotation score, and per-evidence-map summaries. These are measured
against human image-space ground truth and are the promotion gate for future
evidence/cluster changes.

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
