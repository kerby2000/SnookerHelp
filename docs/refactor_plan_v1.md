# SnookerHelp v1 clean architecture and migration plan

Status: Phase 1 approved; Phase 2 migration started. See
[v1_migration_status.md](v1_migration_status.md) for current implementation
status and gates.

Audit date: 2026-06-27.

## 1. Executive summary

The current codebase has accumulated useful experiments, but the architecture is
still prototype-shaped:

- rough detection, source evidence, physical projection, confidence scoring,
  JSON schema, overlays, reports, and review UI are tightly coupled;
- older concepts still leak into UI/data names: `Candidate A/B/C/D`,
  `circle_radial`, `fallback_radial`, `manual_homography`, `Hough`, and
  `source_refined_center_px`;
- there are two report/review paths:
  - static per-image report HTML and generated PNG panels;
  - interactive review server/UI;
- validation tools are useful but duplicated and inconsistent;
- several files are oversized and mix unrelated responsibilities.

The validated learning should be preserved:

- supplied `Media/` images;
- current tests;
- generated benchmark data;
- review feedback JSON;
- ChArUco calibration work;
- physical validation tooling;
- the current best recognition insight: physical model first, supported by
  radial/edge image evidence, with circle/mask evidence demoted.

The v1 migration should not be a blind rewrite. The correct path is to create a
clean v1 package beside the current implementation, add adapters, migrate one
workflow at a time, and delete old paths only after tests and benchmarks pass.

## 2. Current codebase observations

### Oversized / mixed-responsibility files

| File | Approx. LOC | Issue |
|---|---:|---|
| `vision/review_app.py` | 1999 | One Python function returns the full HTML/CSS/JS review app. Hard to test, version, or split UI concepts from data. |
| `tools/evaluate_touching_balls.py` | 814 | CLI parsing, center-mode logic, pair matching, rack validation, Z-plane comparison, summaries, and overlays in one file. |
| `vision/review_evidence.py` | 765 | Builds review JSON, crops, old candidate payloads, warnings, confidence, disagreement, cushion lines, and legacy model decisions. |
| `vision/report_html.py` | 755 | Static report renderer plus old browser-local feedback path. Overlaps with interactive review UI. |
| `vision/report_views.py` | 600 | Useful static QA panels, but mixed drawing helpers and old report concepts. |
| `vision/state_estimator.py` | 575 | Orchestrates config, warp, detector, source refine, camera projection, state schema, overlays, and physical projection. |
| `vision/source_ball_refine.py` | 544 | Valuable image evidence logic, but mixes circle baseline, radial/edge ellipse, mask evidence, fallback behavior, and ROI handling. |
| `vision/ball_detect_classical.py` | 502 | Rough detection, Hough, foreground, legal inventory pruning, color classification, and source-output fields all in one detector class. |
| `vision/model_scoring.py` | 444 | Contains multiple experiments: legacy physics-first, C-only score, confidence combining, and private grading rules. |
| `vision/validation.py` | 453 | General validation utilities, drawing helpers, state loading, and detector invocation are mixed. |

### Duplicate / overlapping paths

1. Report generation:
   - `vision/report_html.py`
   - `vision/report_views.py`
   - `vision/report_metrics.py`
   - `vision/reporting.py`
   - `tools/generate_image_report.py`
   - `tools/generate_dataset_reports.py`

2. Review feedback:
   - old static `report.html` localStorage export path;
   - interactive review server `tools/review_reports.py`;
   - feedback export `tools/export_review_feedback.py`;
   - annotation tool `tools/annotate_ball_centers.py`.

3. Validation:
   - `tools/evaluate_accuracy.py`;
   - `tools/evaluate_repeatability.py`;
   - `tools/evaluate_touching_balls.py`;
   - `tools/evaluate_cushion_touch.py`;
   - `tools/evaluate_spot_positions.py`;
   - shared functions in `vision/validation.py`;
   - report-specific validation in `vision/report_metrics.py`.

4. Geometry:
   - `vision/table_warp.py`;
   - `vision/camera_model.py`;
   - `vision/sphere_projection.py`;
   - `manual_homography` fallback;
   - report geometry rendering in `vision/report_views.py`.

### Legacy concepts still present

These concepts are useful as implementation history, but should not remain
primary v1 concepts:

- Candidate A / Candidate B / Candidate C / Candidate D;
- radial circle as a final model;
- mask centroid as a final model;
- Hough circle as a user-facing detector concept;
- `circle_radial`;
- `fallback_radial`;
- `manual_homography` as a product-facing geometry model;
- `source_refined_center_px` as a product-facing final estimate;
- static `report.html` as the main review workflow;
- “accepted” based only on a circle or image model without physical validation.

### Dead or near-dead implementation paths

These are not necessarily safe to delete today, but they are strong candidates
for legacy isolation:

- `vision/report_html.py` old static feedback/export UI;
- `vision/review_evidence.py::_consensus_ellipse_fit_payload()` as a final
  shape concept; Candidate C is now radial/edge-only, so consensus should not be
  a primary v1 concept;
- `vision/model_scoring.py::physics_first_score()` that mixes B/mask and C
  evidence; the current better experiment is D + C-only;
- `vision/review_evidence.py` model candidate keys named
  `candidate_a_*`, `candidate_b_*`, etc.;
- `HomographyCameraModel` as a user-facing camera model. It may remain as a
  bootstrap/test adapter, but not as v1 physical geometry;
- static generated PNG panels as a product review UI. They may remain as QA
  artifacts.

## 3. Clean v1 product language

The UI should not expose prototype candidate names. Use these terms:

- Pixels
- Image evidence
- Physical model
- Final estimate
- Confidence
- Manual correction

### Explicit do-not-keep list for user-facing UI terms

Do not use these as primary UI labels:

- Candidate A
- Candidate B
- Candidate C
- Candidate D
- circle baseline
- radial circle
- Hough
- fallback radial
- `circle_radial`
- `fallback_radial`
- mask centroid as a model name
- manual homography
- source refined center
- raw Hough center
- warped circle
- old warped-derived coordinate

Allowed only in developer/debug JSON or advanced diagnostic mode:

- Hough candidate;
- radial boundary points;
- mask contour;
- homography;
- source pixel;
- Z-plane projection;
- sphere projection residual.

## 4. Clean v1 concepts

### `BallEvidence`

Raw and fitted evidence from the source image.

Contains:

- source crop bounds;
- pixel observations;
- radial/edge boundary points;
- radial/edge ellipse fit;
- optional segmentation mask evidence;
- color samples and color-class evidence;
- nearest cushion/pocket context;
- evidence quality metrics.

Does not contain:

- final table coordinate;
- UI decision;
- physical truth claim.

### `ImageModel`

A fitted 2D explanation of observed pixels.

Examples:

- radial/edge ellipse;
- visible contour;
- color support;
- boundary support count/residual.

In v1, the preferred image evidence model is the radial/edge ellipse. Circle and
mask can remain internal diagnostics but should not drive final position.

### `PhysicalModel`

The calibrated or approximate model that maps between image pixels and the table
world.

Contains:

- camera intrinsics and distortion;
- camera pose relative to table;
- table dimensions and coordinate frame;
- ball radius;
- ray/plane projection;
- projected sphere silhouette;
- physical constraint evaluators.

The manual four-corner homography can remain as a bootstrap or legacy adapter,
but v1 should treat it as approximate calibration, not the main physical model.

### `BallEstimate`

The computed ball position and uncertainty.

Contains:

- ball id;
- label/color;
- source pixel estimate;
- table XY estimate;
- Z/effective-height assumption;
- uncertainty;
- provenance: image evidence + physical model version;
- validation status.

This replaces user-facing references to `source_refined_center_px`,
`circle_radial`, and fallback radial center.

### `Confidence`

A structured confidence result, not only a float.

Contains:

- level: high / medium / low / needs_review;
- numeric score;
- reasons;
- residuals;
- which evidence was used;
- which constraints passed/failed.

### `TableState`

The canonical v1 output for one image.

Contains:

- image metadata;
- camera/table model references;
- list of `BallEstimate`;
- global warnings;
- validation summary;
- schema version.

### `ReviewFeedback`

Human review layer stored separately from algorithm output.

Contains:

- OK/NOK/manual correction;
- issue tags;
- comments;
- missing-ball hints;
- manual center/ellipse/cushion corrections;
- audit trail;
- schema version.

### `GroundTruthBall`

Manual or physical ground truth.

Contains:

- label/color;
- source pixel point or table XY;
- coordinate system;
- uncertainty of annotation;
- scenario/source metadata.

## 5. Proposed v1 package structure

Create a new package beside `vision/` first. Do not rename everything in one
commit.

```text
snookerhelp/
  core/
    config.py
    geometry.py
    image_io.py
    schema.py
    table.py
    units.py

  recognition/
    rough_detector.py
    classical_rough_detector.py
    evidence.py
    image_model.py
    physical_model.py
    estimator.py
    color.py
    inventory.py

  calibration/
    camera.py
    charuco.py
    table_pose.py
    homography_bootstrap.py

  review/
    schema.py
    evidence_export.py
    feedback.py
    server.py
    static/
      index.html
      app.js
      styles.css

  qa/
    validation.py
    physical_constraints.py
    repeatability.py
    accuracy.py
    benchmark.py
    report_assets.py

  tools/
    process_image.py
    generate_reports.py
    review.py
    validate.py
    calibrate.py
```

Keep `vision/` during migration as a compatibility layer. When v1 passes tests,
move old modules to `legacy/vision_v0/` or delete them according to the
classification below.

## 6. File classification

### `vision/` modules

| File | Classification | v1 action |
|---|---|---|
| `vision/__init__.py` | refactor | Eventually expose v1 API or become compatibility shim. |
| `vision/accuracy.py` | keep | Move algorithms to `snookerhelp/qa/accuracy.py`; keep CLI adapter. |
| `vision/ball_color_classifier.py` | refactor | Move to `snookerhelp/recognition/color.py`; keep current Sony rules as profile. |
| `vision/ball_detect_classical.py` | refactor | Split into `classical_rough_detector.py`, `inventory.py`, and detection schema. Hough remains rough detection only. |
| `vision/camera_model.py` | refactor | Move calibrated pinhole logic to `calibration/camera.py`; keep homography as `homography_bootstrap.py`. |
| `vision/charuco_calibration.py` | keep | Move to `calibration/charuco.py`. |
| `vision/circle_fit.py` | move_to_legacy | Keep temporarily for rough diagnostic tests; should not be final model. |
| `vision/config.py` | keep | Move to `core/config.py`; preserve path resolution behavior. |
| `vision/ellipse_fit.py` | keep | Move to `recognition/image_model.py` or `core/geometry.py`. |
| `vision/model_scoring.py` | replace | Replace with `Confidence` service using physical model + image evidence. Preserve C-only benchmark learning. |
| `vision/overlay.py` | refactor | Move generic drawing to `qa/report_assets.py`; avoid product UI dependency. |
| `vision/report_html.py` | move_to_legacy | Static report HTML is superseded by review UI. Keep until v1 report/review tests pass. |
| `vision/report_metrics.py` | refactor | Split physical validation metrics into `qa/physical_constraints.py`; report selection helpers into review/report module. |
| `vision/report_views.py` | refactor | Keep as QA static panel generator, not product UI. Move to `qa/report_assets.py`. |
| `vision/reporting.py` | replace | Replace with v1 report builder that emits schema + optional QA artifacts. |
| `vision/review_app.py` | replace | Split into real static frontend files + API server. Remove candidate vocabulary. |
| `vision/review_evidence.py` | replace | Replace with typed `ReviewEvidence`/`BallEvidence` exporter. Avoid candidate A/B/C/D schema. |
| `vision/source_ball_refine.py` | refactor | Move radial/edge evidence extraction to `recognition/evidence.py`; remove final-circle semantics. |
| `vision/sphere_projection.py` | keep | Move to `recognition/physical_model.py` or `calibration/camera.py`; keep tests. |
| `vision/state_estimator.py` | replace | Replace with v1 orchestrator using typed concepts. Keep as adapter until migration complete. |
| `vision/table_model.py` | keep | Move to `core/table.py`. |
| `vision/table_warp.py` | refactor | Move to `calibration/homography_bootstrap.py`; rough detection can use warp, final model should not depend on it. |
| `vision/validation.py` | refactor | Split data loading, region classification, drawing, and physical constraints into `qa/`. |

### `tools/` scripts

| File | Classification | v1 action |
|---|---|---|
| `tools/annotate_ball_centers.py` | refactor | Merge useful manual annotation pieces into review workflow. Keep GroundTruthBall export. |
| `tools/benchmark_model_scoring.py` | keep | Move to `snookerhelp/qa/benchmark.py`; make it benchmark v1 confidence. |
| `tools/calibrate_camera_charuco.py` | keep | Thin CLI over `calibration/charuco.py`. |
| `tools/click_table_corners.py` | refactor | Keep as homography bootstrap utility; rename language away from final calibration. |
| `tools/estimate_table_pose_charuco.py` | keep | Thin CLI over `calibration/table_pose.py`. |
| `tools/evaluate_accuracy.py` | refactor | Move core logic to `qa/accuracy.py`; keep CLI. |
| `tools/evaluate_cushion_touch.py` | refactor | Merge into unified `tools/validate.py --kind cushion-touch`. |
| `tools/evaluate_repeatability.py` | refactor | Merge into unified validation CLI; keep repeatability output schema. |
| `tools/evaluate_samples.py` | keep | Keep as core regression benchmark. Move to `qa/benchmark.py`. |
| `tools/evaluate_spot_positions.py` | refactor | Merge into unified validation CLI. |
| `tools/evaluate_touching_balls.py` | refactor | Split into touching-pair, rack, z-plane comparison, overlay modules. Keep algorithms. |
| `tools/export_review_feedback.py` | keep | Move to `review/feedback.py` CLI wrapper. Preserve existing JSONL. |
| `tools/generate_dataset_reports.py` | refactor | Replace with v1 dataset report CLI after report schema stabilizes. |
| `tools/generate_image_report.py` | refactor | Thin CLI over v1 report builder. |
| `tools/process_latest_image.py` | refactor | Thin CLI over v1 process command. |
| `tools/process_single_image.py` | refactor | Thin CLI over v1 process command. |
| `tools/render_geometry_scenes.py` | move_to_legacy | Optional explanatory visualization, not core recognition. |
| `tools/review_reports.py` | replace | Replace with v1 review server/API and static frontend assets. |

### Data, tests, and outputs

| Path | Classification | v1 action |
|---|---|---|
| `Media/` | keep | Preserve as regression dataset. |
| `data/review_feedback/` | keep | Preserve; add schema version migration if needed. |
| `data/annotations/` | keep | Preserve as GroundTruthBall examples. |
| `outputs/model_scoring_benchmark/` | keep | Preserve as v0 benchmark baseline. |
| `outputs/reports/` | keep temporarily | Useful for review continuity; regenerate with v1 later. |
| `outputs/reports_*_smoke/` | delete_after_v1_passes_tests | Smoke outputs can be regenerated. |
| `SnookerHelp_v0.*.zip` | move_to_legacy | Archive outside source tree or under `legacy/releases/`. |
| `tests/` | keep | Expand with v1 schema and migration tests. |
| `configs/` | keep | Preserve configs; split calibration profiles later. |
| `visualizations/` | move_to_legacy | Optional explanatory assets. |

## 7. What should survive

Keep the validated core behavior:

- rough detection can still use warped cloth-plane images;
- final evidence should come from source-image pixels;
- physical model should be highest trust;
- radial/edge ellipse evidence is the preferred image support;
- mask evidence is secondary diagnostic evidence;
- legal snooker inventory pruning is useful but must remain test-covered;
- ChArUco calibration path should continue;
- physical validation tools should continue.

Keep current regression tests and add more before deletion:

- exact sample count tests;
- DSC00526 duplicate suppression;
- DSC00528 true-green preservation;
- DSC00529 cushion duplicate suppression;
- DSC00540 dense cluster inventory;
- DSC00542 difficult cluster/edge evidence regression.

## 8. Migration order

### Step 0 — freeze baseline

Before structural migration:

```powershell
python -m pytest -q
python tools/evaluate_samples.py
python tools/benchmark_model_scoring.py --reports outputs/reports --output outputs/model_scoring_benchmark
```

Record:

- test count and pass status;
- sample-count exactness;
- benchmark confidence summary;
- current report schema version;
- review feedback schema version.

### Step 1 — add v1 schemas only

Add typed dataclasses or typed dictionaries under:

```text
snookerhelp/core/schema.py
snookerhelp/recognition/evidence.py
snookerhelp/review/schema.py
```

No algorithm change.

Add adapters:

- current state JSON -> `TableState`;
- current review JSON -> `ReviewFeedback`;
- current ball JSON -> `BallEvidence` and `BallEstimate`.

### Step 2 — isolate physical model

Move/copy physical projection logic behind:

```text
snookerhelp/recognition/physical_model.py
snookerhelp/calibration/camera.py
```

Keep `vision/sphere_projection.py` as a compatibility shim until tests are
ported.

### Step 3 — isolate image evidence

Move radial/edge evidence extraction from `source_ball_refine.py` into:

```text
snookerhelp/recognition/image_model.py
snookerhelp/recognition/evidence.py
```

Do not expose circle/mask as final models. Keep them only as diagnostics.

### Step 4 — replace confidence scoring

Replace `model_scoring.py` with a v1 `Confidence` service:

```text
Physical model score
  + Image evidence score
  + Physical validation score
  + Region/cushion/pocket risk
  -> Confidence
```

Remove old A/B/C/D names from score outputs.

### Step 5 — create v1 estimator

Create:

```text
snookerhelp/recognition/estimator.py
```

The v1 estimator should output `TableState`, not prototype state JSON. Keep
`vision/state_estimator.py` as an adapter until all tools use v1.

### Step 6 — replace review evidence and UI language

Replace:

- `vision/review_evidence.py`;
- `vision/review_app.py`;
- `tools/review_reports.py`.

With:

```text
snookerhelp/review/evidence_export.py
snookerhelp/review/server.py
snookerhelp/review/static/index.html
snookerhelp/review/static/app.js
snookerhelp/review/static/styles.css
```

UI sections should be:

1. Pixels
2. Image evidence
3. Physical model
4. Final estimate
5. Confidence
6. Manual correction

### Step 7 — unify validation tools

Create one validation entrypoint:

```powershell
python -m snookerhelp.tools.validate --kind touching
python -m snookerhelp.tools.validate --kind cushion
python -m snookerhelp.tools.validate --kind spot
python -m snookerhelp.tools.validate --kind repeatability
python -m snookerhelp.tools.validate --kind accuracy
```

Keep old scripts as wrappers until v1 validation outputs match current outputs.

### Step 8 — move legacy code

Only after v1 tests and benchmarks pass:

```text
legacy/
  vision_v0/
  static_reports_v0/
  ui_v0/
  tools_v0/
```

Move old code there before final deletion. Keep shims for one migration cycle.

### Step 9 — delete obsolete paths

Delete only after:

- tests pass;
- all sample reports regenerate;
- review feedback still loads;
- benchmark confidence and physical validation are not worse;
- CLI replacements exist.

## 9. Tests and benchmarks required before deletion

Minimum required checks:

```powershell
python -m pytest -q
python tools/evaluate_samples.py
python tools/generate_dataset_reports.py --glob "Media/**/*.JPG" --output outputs/reports_v1
python tools/benchmark_model_scoring.py --reports outputs/reports_v1 --output outputs/model_scoring_benchmark_v1
python tools/export_review_feedback.py --reports outputs/reports_v1 --output data/review_feedback/v1_smoke.jsonl
```

Required result gates:

- all tests pass;
- supplied sample count remains exact or any regression is explicitly accepted;
- DSC00526/DSC00528/DSC00529/DSC00542 regressions remain covered;
- v1 report generation completes for all sample images;
- review UI can load at least DSC00524, DSC00529, DSC00540, DSC00542;
- existing `data/review_feedback/dataset_feedback.jsonl` can still be read or
  migrated;
- benchmark confidence is not worse than the current C-only baseline unless
  physical validation proves the lower confidence is more correct;
- touching/cushion/spot/repeatability tools still emit CSV and JSON;
- ChArUco board config test still passes.

Physical validation gates before trusting v1 coordinates:

- touching-ball errors by region;
- cushion-touch radius errors by region;
- rack red nearest-neighbor distribution;
- spot-position errors;
- repeatability standard deviation across repeated unchanged layouts;
- calibrated ChArUco reprojection error once board images are available.

## 10. Risk list

| Risk | Impact | Mitigation |
|---|---|---|
| Confidence improved but physical accuracy did not | High | Use touching/cushion/spot/repeatability validation before deleting old coordinate paths. |
| Removing warped pipeline too early breaks rough detection | High | Keep warp as internal rough-detection implementation until source-only rough detection is proven. |
| Removing manual homography too early blocks current camera bootstrap | High | Rename to bootstrap/approximate calibration, not delete until ChArUco flow is operational. |
| Review feedback schema breaks | High | Add migration tests for existing `review.json` and JSONL feedback. |
| UI loses useful diagnostics by hiding candidate names | Medium | Keep advanced developer/debug panel, but use product language by default. |
| Dense-cluster behavior regresses | High | Keep DSC00540/DSC00542 tests and add physical rack validation. |
| Legal inventory pruning hides real detector misses | Medium | Track raw/gated/final counts in QA, not product UI. |
| Static reports are removed before replacement | Medium | Keep `report_views.py` as QA artifact generator until v1 review UI exports equivalent evidence. |
| ChArUco assumptions differ from CALITAR board reality | Medium | Keep board config and validation workflow separate from recognition code. |
| Big-bang package rename causes broken tools | High | Add v1 package with adapters first; old scripts become wrappers. |

## 11. Suggested first Phase 2 task after approval

Do not start this until approved.

Recommended first implementation task:

```text
Create v1 schema package and adapters only.

Add:
snookerhelp/core/schema.py
snookerhelp/recognition/evidence.py
snookerhelp/review/schema.py

Implement:
- BallEvidence
- ImageModel
- PhysicalModelSummary
- BallEstimate
- Confidence
- TableState
- ReviewFeedback
- GroundTruthBall

Add adapter functions that convert the current report/state JSON into these
objects.

No algorithm change.
No UI rewrite.
No deletion.
```

Reason: this creates the stable target architecture without risking the current
working detector, reports, or review feedback.

## 12. Phase 2 execution status

Phase 2 has been approved and started.

Current execution status is tracked in
[v1_migration_status.md](v1_migration_status.md). That document is the source of
truth for which wrappers, legacy moves, v1 UI pieces, and acceptance gates have
already been completed.

Do not use the remaining historical sections above as proof that a path has not
yet moved. They describe the audit result and intended migration order.
