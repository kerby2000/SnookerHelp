# SnookerHelp v1 migration status

Last updated: 2026-07-02.

## Completed

- Added `snookerhelp/` v1 package skeleton.
- Added typed v1 schemas:
  - `BallEvidence`
  - `ImageModel`
  - `PhysicalModel`
  - `BallEstimate`
  - `Confidence`
  - `TableState`
  - `ReviewFeedback`
  - `GroundTruthBall`
- Added adapters from current `report.json` / legacy review data to v1 schema.
- Added a new v1 review server and static frontend:
  - `snookerhelp/review/server.py`
  - `snookerhelp/review/static/index.html`
  - `snookerhelp/review/static/app.js`
  - `snookerhelp/review/static/styles.css`
- Added stable v1 facade modules over validated legacy implementation.
- Added wrapper CLIs:
  - `python -m snookerhelp.tools.review`
  - `python -m snookerhelp.tools.validate`
  - `python -m snookerhelp.tools.generate_reports`
  - `python -m snookerhelp.tools.process_image`
  - `python -m snookerhelp.tools.calibrate`
  - `python -m snookerhelp.tools.migrate_feedback`
  - `python -m snookerhelp.tools.export_feedback`
- Added tests for v1 schema, review server payloads, static UI language, and
  feedback migration.
- Added v1 review-feedback API support for missing balls, manual source-pixel
  correction, manual ellipse correction, and manual cushion-line correction.
  The current focused UI hides the manual editor and uses a clickable ball
  statistics/evidence table instead.
- Added v1-native feedback export from `review_v1.json` / legacy `review.json`.
- Deleted the old 2000-line review UI and old review-server compatibility
  scripts after the v1 review UI save/export gates passed.
- Replaced `tools/review_reports.py` with a v1 review UI wrapper.
- Replaced the old static browser-feedback report renderer with a v1-owned
  read-only QA report renderer in `snookerhelp.qa.report_html`.
- Deleted the old row-per-ball feedback exporter after v1 export and migration
  gates passed.
- Replaced `tools/export_review_feedback.py` with a v1 feedback export wrapper.
- Replaced active report-generation commands with v1 package entrypoints:
  - `tools/generate_image_report.py`
  - `tools/generate_dataset_reports.py`
  - `snookerhelp/tools/generate_reports.py`
- Replaced all active root `tools/*.py` commands with v1 package entrypoint
  wrappers.
- Moved package-owned CLI logic into:
  - `snookerhelp.tools.generate_reports`
  - `snookerhelp.tools.process_image`
  - `snookerhelp.tools.validate`
  - `snookerhelp.tools.calibrate`
  - `snookerhelp.qa.samples`
  - `snookerhelp.qa.benchmark`
  - `snookerhelp.qa.*_cli` validation modules
  - `snookerhelp.calibration.charuco`
  - `snookerhelp.calibration.homography_bootstrap`
  - `snookerhelp.review.annotation`
- Moved additional backend helper ownership into v1 packages:
  - `snookerhelp.core.config`
  - `snookerhelp.core.table`
  - `snookerhelp.calibration.homography_bootstrap.TableWarp`
  - `snookerhelp.calibration.charuco_core`
  - `snookerhelp.calibration.camera_core`
  - `snookerhelp.recognition.sphere_projection`
  - `snookerhelp.recognition.image_model`
  - `snookerhelp.recognition.color`
  - `snookerhelp.qa.accuracy`
  - `snookerhelp.qa.validation`
  - `snookerhelp.qa.reporting`
- Moved the remaining detector/report-evidence backend ownership into v1
  packages:
  - `snookerhelp.recognition.estimator`
  - `snookerhelp.recognition.classical_rough_detector`
  - `snookerhelp.recognition.circle_fit`
  - `snookerhelp.recognition.source_refinement`
  - `snookerhelp.recognition.confidence`
  - `snookerhelp.review.overlay`
  - `snookerhelp.review.evidence_builder`
  - `snookerhelp.qa.report_html`
  - `snookerhelp.qa.report_metrics`
  - `snookerhelp.qa.report_views`
- Removed active `snookerhelp.*` and active root `tools/*.py` imports from
  `vision.*`, then deleted the `vision/` compatibility source tree after the
  v1 replacement gates passed.
- Rewrote the main fitting explanation doc to use v1 product language:
  Pixels, image evidence, physical model, final estimate, confidence, and
  manual correction.
- Added v1.3 local evidence maps and physical optimization:
  - `snookerhelp.recognition.evidence_maps`
  - `snookerhelp.recognition.physical_optimize`
  - accepted white edge points, rejected red outliers, cream observed ellipse,
    blue forward/optimized physical sphere projection, and selectable
    diagnostic evidence-map crop backgrounds in the v1 review UI.
- Added v1.3.1 adjacent-ball scene constraints:
  - `snookerhelp.recognition.cluster_optimize`
  - close-neighbor cluster diagnostics with equal-radius, non-overlap, and
    touching-distance checks;
  - scene-constraint rows in the v1 review UI and benchmark CSV/JSON.
- Added confidence components for image evidence, physical model, scene
  constraints, and final confidence.
- Added v1.3.3 diagnostic evidence-map UI and removed the cyan
  projection-band recovered-points experiment from the active pipeline:
  - every regenerated ball report now has six evidence-map crop backgrounds;
  - physical projection band remains diagnostic/scoring evidence only;
  - no recovered-point layer is exposed in the v1 UI or v1 schema.
- Added v1.3.4 evidence-map-specific boundary variants:
  - every evidence-map crop background can now have its own accepted white
    boundary points, rejected red outliers, and cream observed ellipse;
  - the v1 UI switches the dots and cream ellipse when the crop background
    selector changes;
  - final `source_px`, table coordinates, and confidence are intentionally not
    changed by this diagnostic feature until a variant benchmark promotes a
    stable map policy.
- Added v1.3.5 review UI layout and overlay controls:
  - Image evidence, Physical model, and Confidence panels are arranged along
    the bottom of the Final Estimate panel;
  - crop background selection is decoupled from overlay selection;
  - evidence rows now expose a checkbox matrix for accepted dots, rejected
    dots, dashed outline, and fit-center cross;
  - clicking an evidence row resets the matrix to that row and switches the
    background when that row has a background asset;
  - explanatory legend moved into a draggable floating help panel;
  - 10 mm scale overlay removed from the crop controls.
- Added v1.3.6 final-position evidence-map policy:
  - `ball_vs_cloth_probability` is the default final image-evidence map;
  - `chroma_difference` is used for green, blue, and brown balls;
  - all other evidence maps remain visible diagnostics and do not affect final
    source/table position;
  - v1 UI now defaults the evidence-layer matrix to the policy-selected map.
- Added v1.3.7 evidence-view scoring and UI explanations:
  - every evidence-map boundary variant now carries a diagnostic `view_score`;
  - source-boundary evidence also carries a diagnostic view score;
  - the v1 overlay matrix shows a Score column for side-by-side map
    comparison;
  - Help/legend explains confidence, absence of ground truth, evidence-view
    score, and final source-center policy.
- Added v1.3.8 global cloth-reference color model:
  - B/C/L evidence maps now use a per-image global table-cloth Lab reference by
    default instead of the local annulus;
  - the old local annulus model remains visible as a diagnostic for
    cluster/neighbor contamination;
  - the Image evidence panel shows active ball Lab, active cloth Lab,
    Lab/chroma separation, sample counts, low-contrast flag, and active
    parameters;
  - `tools/analyze_cloth_reference.py` benchmarks local-vs-global cloth
    behavior across regenerated reports;
  - final map promotion now has a plausibility guard for implausible ellipse
    aspect ratio, size, or center shift.
- Deleted the old `legacy/static_reports_v0/report_html.py` browser-local
  feedback renderer. Static `report.html` is now a read-only QA artifact; all
  OK/NOK, missing-ball, and manual-correction feedback goes through the v1
  review UI.
- Deleted the remaining `legacy/tools_v0/*.py` source snapshots after active
  validation/calibration/process/report/export commands passed smoke gates.

## Current gates

Most recent local checks:

```text
python -m pytest -q
108 passed in the current migration gate; v1.3.1 targeted and full gates are
listed below.

python -m snookerhelp.tools.validate --kind samples
Exact counts: 21/21; mean absolute count error: 0.000

python -m snookerhelp.tools.validate --kind architecture
active snookerhelp/tools code has no vision.* or legacy.* imports;
vision source tree is deleted;
v1 review UI uses product language;
active static report renderer is v1-owned;
old browser-local static feedback renderer is deleted;
old review UI/server files are deleted;
legacy source tree is deleted

rg -n "from vision\.|import vision\.|vision\." snookerhelp tools -g "*.py"
no matches

vision source tree audit
vision/ deleted

python tools/generate_dataset_reports.py --glob "Media/**/*.JPG" --output outputs/reports_v1_entrypoint_gate
21/21 reports generated

python tools/generate_dataset_reports.py --glob "Media/**/*.JPG" --output outputs/reports_v1_vision_shim_gate
21/21 reports generated

python tools/generate_dataset_reports.py --glob "Media/**/*.JPG" --output outputs/reports_v1_static_qa_gate
21/21 reports generated

python tools/generate_dataset_reports.py --glob "Media/**/*.JPG" --output outputs/reports_v1_legacy_deleted_gate
21/21 reports generated

python tools/generate_dataset_reports.py --glob "Media/**/*.JPG" --output outputs/reports_v1_vision_deleted_gate
21/21 reports generated

python tools/benchmark_model_scoring.py --reports outputs/reports_v1_gate --output outputs/model_scoring_benchmark_v1_entrypoint_gate
Displayed mean confidence: 0.694
Confidence improved by >=10 points: 281
Confidence reduced by >=10 points: 0

python tools/benchmark_model_scoring.py --reports outputs/reports_v1_vision_shim_gate --output outputs/model_scoring_benchmark_v1_vision_shim_gate
Displayed mean confidence: 0.694
Confidence improved by >=10 points: 281
Confidence reduced by >=10 points: 0

python tools/benchmark_model_scoring.py --reports outputs/reports_v1_static_qa_gate --output outputs/model_scoring_benchmark_v1_static_qa_gate
Displayed mean confidence: 0.694
Confidence improved by >=10 points: 281
Confidence reduced by >=10 points: 0

python tools/benchmark_model_scoring.py --reports outputs/reports_v1_legacy_deleted_gate --output outputs/model_scoring_benchmark_v1_legacy_deleted_gate
Displayed mean confidence: 0.694
Confidence improved by >=10 points: 281
Confidence reduced by >=10 points: 0

python tools/benchmark_model_scoring.py --reports outputs/reports_v1_vision_deleted_gate --output outputs/model_scoring_benchmark_v1_vision_deleted_gate
Displayed mean confidence: 0.694
Confidence improved by >=10 points: 281
Confidence reduced by >=10 points: 0

python tools/export_review_feedback.py --reports outputs/reports --output data/review_feedback/dataset_feedback_v1_entrypoint_gate.jsonl
normal feedback export command writes v1 ReviewFeedback JSONL

python tools/export_review_feedback.py --reports outputs/reports --output data/review_feedback/dataset_feedback_v1_static_qa_gate.jsonl
exported 2 image feedback records, 44 ball reviews, 0 missing-ball hints

active validation/process smoke gates
accuracy, touching, rack, cushion, spot, repeatability, and process-image
commands wrote JSON/CSV/state outputs under data/*/v1_legacy_delete_*,
data/*/v1_vision_deleted_*, outputs/process_v1_legacy_delete_gate, and
outputs/process_v1_vision_deleted_gate

review server smoke
tools/review_reports.py loaded outputs/reports_v1_legacy_deleted_gate;
/api/table-state/DSC00542 returned snookerhelp.table_state.v1 with 22 balls
and snookerhelp.review_feedback.v1

tools/review_reports.py loaded outputs/reports_v1_vision_deleted_gate;
/api/table-state/DSC00542 returned snookerhelp.table_state.v1 with 22 balls
and /api/review/DSC00542 returned snookerhelp.review_feedback.v1 with 22 ball
reviews

python tools/generate_image_report.py --image Media/01_empty_table/DSC00543.JPG --output outputs/reports_v1_entrypoint_smoke
active root command generated report.json and static QA report.html

python tools/generate_image_report.py --image Media/01_empty_table/DSC00543.JPG --output outputs/reports_v1_module_move_smoke
package-owned report builder generated report.json and static QA report.html

python tools/generate_image_report.py --image Media/01_empty_table/DSC00543.JPG --output outputs/reports_v1_estimator_move_smoke
active root command generated report.json and static QA report.html

python tools/generate_dataset_reports.py --glob "Media/01_empty_table/*.JPG" --output outputs/reports_v1_dataset_entrypoint_smoke --limit 1
active root command generated dataset_reports.json with schema snookerhelp.dataset_reports.v1

node --check snookerhelp/review/static/app.js
passed

python -m compileall -q snookerhelp tools tests
passed

v1.3.1 evidence-map / physical-optimization / cluster-constraint gates:

```text
python -m pytest tests/test_cluster_optimize.py tests/test_evidence_maps.py tests/test_physical_optimize.py tests/test_sphere_projection.py tests/test_v1_schema.py tests/test_model_scoring.py tests/test_v1_review_contract.py -q
23 passed

python tools/generate_dataset_reports.py --glob "Media/**/*.JPG" --output outputs/reports_v1_evidence_maps
21/21 reports generated

python tools/benchmark_model_scoring.py --reports outputs/reports_v1_evidence_maps --output outputs/model_scoring_benchmark_v1_3
Rows: 418
Displayed mean confidence: 0.756
Confidence improved by >=10 points: 314
Confidence reduced by >=10 points: 0
Mean accepted/rejected boundary points: 117.5 / benchmark-dependent
Physical optimization statuses: {'no_better_solution': 223, 'optimized': 195}
Physical projection modes: {'forward': 223, 'optimized': 195}
Joint cluster statuses: {'not_in_cluster': 316, 'optimized': 98, 'no_better_solution': 2, 'no_contact_constraints': 2}
Mean joint cluster pair-distance improvement: 2.359 mm

node --check snookerhelp/review/static/app.js
passed
```

v1.3.3 diagnostic evidence-map / no recovered-points gates:

```text
python -m pytest tests/test_evidence_maps.py tests/test_v1_schema.py tests/test_v1_review_contract.py tests/test_sample_pipeline.py::test_dsc00542_green_blue_and_red_cluster_evidence_regression tests/test_sample_pipeline.py::test_weak_green_blue_boundary_recovery_regression -q
14 passed

python tools/generate_dataset_reports.py --glob "Media/**/*.JPG" --output outputs/reports_v1_evidence_maps
21/21 reports generated

python tools/benchmark_model_scoring.py --reports outputs/reports_v1_evidence_maps --output outputs/model_scoring_benchmark_v1_3
Rows: 418
Displayed mean confidence: 0.789
Confidence improved by >=10 points: 321
Confidence reduced by >=10 points: 0
Mean accepted/rejected boundary points: 117.5 / 8.4
Evidence map statuses: {'computed': 418}
Mean evidence-map assets per ball: 6.0
Physical optimization statuses: {'no_better_solution': 124, 'optimized': 294}
Physical projection modes: {'forward': 124, 'optimized': 294}
Joint cluster statuses: {'not_in_cluster': 315, 'optimized': 99, 'no_better_solution': 4}
Duplicate-warning false-positive proxy count: 0

python -m pytest -q
107 passed

python -m snookerhelp.tools.validate --kind architecture
all architecture checks passed

python -m snookerhelp.tools.validate --kind samples
Exact counts: 21/21; mean absolute count error: 0.000

node --check snookerhelp/review/static/app.js
passed
```

v1.3.4 evidence-map boundary-variant gates:

```text
python -m pytest tests/test_evidence_maps.py tests/test_v1_schema.py tests/test_v1_review_contract.py -q
13 passed

python -m pytest -q
108 passed

python tools/generate_dataset_reports.py --glob "Media/**/*.JPG" --output outputs/reports_v1_evidence_maps
21/21 reports generated

report payload check
418/418 detected balls had six evidence-map boundary variant entries:
gray_edge, lab_delta_e, chroma_difference, ball_vs_cloth_probability,
physical_projection_band, combined_boundary_score

python -m snookerhelp.tools.validate --kind architecture
all architecture checks passed

python -m snookerhelp.tools.validate --kind samples
Exact counts: 21/21; mean absolute count error: 0.000

python tools/benchmark_model_scoring.py --reports outputs/reports_v1_evidence_maps --output outputs/model_scoring_benchmark_v1_3
Rows: 418
Displayed mean confidence: 0.789
Confidence improved by >=10 points: 321
Confidence reduced by >=10 points: 0
Mean accepted/rejected boundary points: 117.5 / 8.4
Evidence map statuses: {'computed': 418}
Mean evidence-map assets per ball: 6.0
Physical optimization statuses: {'no_better_solution': 124, 'optimized': 294}
Physical projection modes: {'forward': 124, 'optimized': 294}
Joint cluster statuses: {'not_in_cluster': 315, 'optimized': 99, 'no_better_solution': 4}
Mean joint cluster pair-distance improvement: 2.452 mm
Duplicate-warning false-positive proxy count: 0

node --check snookerhelp/review/static/app.js
passed
```

v1.3.5 overlay-matrix UI gates:

```text
python -m pytest tests/test_v1_review_contract.py tests/test_v1_schema.py tests/test_evidence_maps.py -q
13 passed

python -m pytest -q
108 passed

python -m compileall -q snookerhelp tools tests
passed

node --check snookerhelp/review/static/app.js
passed
```

v1.3.6 final-position evidence-map policy gates:

```text
python -m pytest tests/test_v1_review_contract.py tests/test_v1_schema.py tests/test_evidence_maps.py -q
13 passed

python -m pytest tests/test_sample_pipeline.py -q
10 passed

python -m pytest -q
108 passed

python -m compileall -q snookerhelp tools tests
passed

node --check snookerhelp/review/static/app.js
passed

python tools/generate_dataset_reports.py --glob "Media/**/*.JPG" --output outputs/reports_v1_evidence_maps
21/21 reports generated

policy payload audit over outputs/reports_v1_evidence_maps
418/418 detected balls had the configured selected_map:
ball_vs_cloth_probability for black/pink/red/white/yellow and
chroma_difference for blue/brown/green; violations: 0

python tools/benchmark_model_scoring.py --reports outputs/reports_v1_evidence_maps --output outputs/model_scoring_benchmark_v1_3
Rows: 418
Displayed mean confidence: 0.792
Confidence improved by >=10 points: 297
Confidence reduced by >=10 points: 0
Physical optimization statuses: {'no_better_solution': 354, 'optimized': 64}
Physical projection modes: {'forward': 354, 'optimized': 64}
Joint cluster statuses: {'not_in_cluster': 313, 'optimized': 97, 'no_better_solution': 6, 'no_contact_constraints': 2}
Duplicate-warning false-positive proxy count: 0

python -m snookerhelp.tools.validate --kind architecture
all architecture checks passed

python -m snookerhelp.tools.validate --kind samples
Exact counts: 21/21; mean absolute count error: 0.000
```

v1.3.7 evidence-view scoring / confidence-explanation gates:

```text
python -m pytest tests/test_v1_review_contract.py tests/test_v1_schema.py tests/test_evidence_maps.py -q
13 passed

python -m pytest -q
108 passed

python -m compileall -q snookerhelp tools tests
passed

node --check snookerhelp/review/static/app.js
passed

python tools/generate_dataset_reports.py --glob "Media/**/*.JPG" --output outputs/reports_v1_evidence_maps
21/21 reports generated

python tools/benchmark_model_scoring.py --reports outputs/reports_v1_evidence_maps --output outputs/model_scoring_benchmark_v1_3
Rows: 418
Displayed mean confidence: 0.792
Confidence improved by >=10 points: 297
Confidence reduced by >=10 points: 0

python -m snookerhelp.tools.validate --kind architecture
all architecture checks passed

python -m snookerhelp.tools.validate --kind samples
Exact counts: 21/21; mean absolute count error: 0.000
```

v1.3.8 global-cloth reference gates:

```text
python -m pytest tests/test_evidence_maps.py -q
3 passed

node --check snookerhelp/review/static/app.js
passed

python tools/generate_dataset_reports.py --glob "Media/**/*.JPG" --output outputs/reports_v1_global_cloth
21/21 reports generated

python tools/analyze_cloth_reference.py --reports outputs/reports_v1_global_cloth --output outputs/cloth_reference_analysis_global
Rows: 418
Mean local-vs-global Delta-E: 27.32
Active low contrast rows: 0
Local-annulus low contrast rows: 11

python -m pytest -q
108 passed

python -m snookerhelp.tools.validate --kind samples
Exact counts: 21/21; mean absolute count error: 0.000

python -m snookerhelp.tools.validate --kind architecture
all architecture checks passed
```

The v1 review server was smoke-tested over HTTP at:

```text
http://127.0.0.1:8775/
http://127.0.0.1:8787/
```

with 21 regenerated reports. `DSC00542` loaded through
`/api/table-state/DSC00542` as `snookerhelp.table_state.v1` with 22 balls, and
`/api/review/DSC00542` returned `snookerhelp.review_feedback.v1`.

v1.3.9 neighbor-ellipse ownership filtering:

- Added source-space neighboring ellipses as review diagnostics.
- Added a conservative boundary filter that rejects sampled points lying inside
  nearby neighbor ellipses when enough selected-ball points remain.
- Added a v1 UI reference overlay for purple dashed neighbor ellipses.
- Kept rejected neighbor-owned points visible as red dots.
- The v1 UI now shows the promoted final evidence-map payload when the active
  row is the same map used for final position, so dots/ellipse/filter counters
  match the exported final source center.
- Kept the final map `maximum_axis_ratio` guard as a fallback; neighbor
  ownership is now the primary cluster-contamination defense.

Focused `DSC00542` spot check after regenerating the image report:

```text
ball #9 final B-map:  9 neighbor-owned rejects / 11 candidates
ball #14 final B-map: 20 neighbor-owned rejects / 27 candidates
```

Fast gates run before full report regeneration:

```text
node --check snookerhelp/review/static/app.js
passed

python -m py_compile snookerhelp\recognition\source_refinement.py snookerhelp\recognition\estimator.py snookerhelp\review\evidence_builder.py snookerhelp\recognition\evidence.py
passed

python -m pytest tests/test_evidence_maps.py tests/test_v1_review_contract.py -q
10 passed

python -m pytest tests/test_v1_review_contract.py -q
7 passed after final-map UI alignment
```

Full v1.3.9 gates:

```text
python tools/generate_dataset_reports.py --glob "Media/**/*.JPG" --output outputs/reports_v1_global_cloth
21/21 reports generated

python tools/benchmark_model_scoring.py --reports outputs/reports_v1_global_cloth --output outputs/model_scoring_benchmark_v1_3_neighbor
Rows: 418
Displayed mean confidence: 0.793
Confidence improved by >=10 points: 300
Confidence reduced by >=10 points: 0
Mean accepted/rejected boundary points: 117.5 / 8.4

neighbor-ellipse final-policy coverage:
rows: 418
neighbor candidate rows: 47
neighbor rejected rows: 35
total neighbor rejected: 413

python -m pytest -q
108 passed

python -m compileall -q snookerhelp tools tests
passed

python -m snookerhelp.tools.validate --kind architecture
all architecture checks passed

python -m snookerhelp.tools.validate --kind samples
Exact counts: 21/21; mean absolute count error: 0.000
```

## Remaining refactor cleanup

No active legacy or `vision/` compatibility source paths remain intentionally
available.

The static `report.html` path remains as a v1-owned read-only QA artifact, not
as a feedback workflow.

The v1 refactor migration cleanup is complete under the current acceptance
gates. Further work should be product/recognition work, not legacy migration:
real ChArUco calibration, stronger physical model selection, detector tuning,
and review-feedback-driven recognition improvements.
