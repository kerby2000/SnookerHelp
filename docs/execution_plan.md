# SnookerHelp recognition execution plan

Status: active

This is the single implementation plan for the recognition project. Historical
plans and migration diaries are not active requirements. Git checkpoint
`155e727` preserves the pre-plan v1 evidence and cluster experiments.

## Product objective

For each fixed-camera table image, produce a complete and auditable table state:

- registered table, cushion, and pocket geometry;
- one explicit hypothesis for every visible ball;
- ball label, source-image center, table XY, radius, and uncertainty;
- explicit missing, duplicate, suppressed, and unresolved hypotheses;
- image, physical, and scene evidence explaining every final estimate.

The warped cloth-plane image remains a rough-detection/debug aid. Final ball
coordinates come from source/undistorted pixels and the camera model.

## Non-negotiable rules

1. Algorithm output, experiment output, and human annotation are separate data.
2. The review UI does not contain private recognition algorithms.
3. Production recognition and interactive experiments call the same evidence
   functions.
4. A physical prior is never counted as observed image evidence.
5. Dense clusters are solved jointly; traversal order is diagnostic only.
6. Confidence is decomposed and benchmarked. It is not a self-generated
   probability.
7. No experimental result is promoted without a before/after benchmark.
8. Generated maps and large point arrays are assets, not canonical JSON blobs.
9. Obsolete plans are removed after their useful requirements are incorporated
   here or in the maintained topic documents.

## Maintained document set

- `architecture.md`: system boundary, runtime components, and data flow.
- `ball_geometry_model.md`: source-image evidence and physical ball geometry.
- `boundary_filtering_strategy.md`: current evidence algorithms and parameters.
- `charuco_calibration_workflow.md`: calibrated camera workflow.
- `coordinate_accuracy_validation.md` and `physical_validation_tools.md`: QA.
- `image_debug_reports.md`: review and experiment workbench.
- `v2_requirements/`: detailed contracts that have not yet passed acceptance.
- this file: implementation order, ownership, and gates.

## Phase 0 - preserve the current baseline

Status: complete

- Checkpoint current source, tests, configuration, and active experiments.
- Push the checkpoint to `origin/main`.
- Preserve the 21 sample images and generated count benchmark.

Evidence:

- checkpoint: `155e727`;
- 126 tests passing before the checkpoint;
- 21/21 sample images have the expected count, 418/418 balls total.

The count result is not coordinate ground truth because legal-inventory limits
can hide one duplicate plus one missed ball.

## Phase 1 - evidence experiment and image-space ground truth

Status: implementation complete; acceptance data in progress

### Deliverables

1. A tracked annotation schema under `benchmarks/annotations/`.
2. A perfect-ellipse editor in the v1 review workbench.
3. A stateless backend experiment endpoint that recomputes:
   - selected evidence map;
   - raw/accepted/rejected boundary points;
   - ellipse fit;
   - decomposed diagnostic score;
   - comparison with the production baseline;
   - comparison with a saved manual ellipse when available.
4. Interactive controls for ball/cloth reference policy, map parameters,
   radial sampling, filtering, and neighbor rejection.
5. A reusable evidence service called by both the estimator and experiment API.

### Perfect-ellipse annotation

The editor starts from a copy of the current observed ellipse but stores an
independent human annotation. The user can move the center, change major and
minor axes, rotate the ellipse, record visible/occluded arcs, and save notes.

This is valuable before camera calibration. It gives image-space truth for
boundary extraction and fitting. It is not automatically treated as the true
3D sphere center, especially for an occluded ball.

### Experiment safety

- Experiments never overwrite `report.json` or production table state.
- UI changes are debounced and cancellable.
- Every response includes the complete effective parameter set.
- A parameter set may be promoted only after evaluation across the gold set.

### Acceptance gate

- Saved annotations survive server restart and remain separate from review data.
- Copy/edit/save/reload works for at least DSC00524 and DSC00540.
- Parameter changes recompute data on the server, not only CSS display filters.
- Baseline parameters reproduce the production evidence within numeric tolerance.
- Unit/API/UI contract tests pass.

Current evidence:

- DSC00540 has 22/22 saved perfect ellipses;
- production median center error is 0.80 px, but mean is 8.65 px because five
  interior reds fail badly;
- worst cluster fits are #12, #14, #17, #19, and #21 with 25-29 px contour RMS;
- one loose-ball annotation set is still required before closing this gate.

## Phase 2 - tracked gold benchmark

Status: in progress

### Initial gold images

- loose balls: DSC00524 and DSC00525;
- near cushion: DSC00529;
- near pocket: DSC00534;
- intact rack cluster: DSC00540;
- arbitrary cluster: DSC00542;
- empty table: DSC00543.

### Required annotations

- accepted, missing, and duplicate ball identities;
- image-space center and ellipse where visually measurable;
- visible and occluded arcs for cluster balls;
- label and reviewer uncertainty;
- physical touching/cushion/spot constraints where known.

### Metrics

- detection precision/recall before inventory pruning;
- center error in source pixels and table millimeters;
- ellipse center/axis/angle/contour error;
- touching and non-overlap violations;
- unexplained foreground in cluster ROIs;
- repeatability and calibrated-camera reprojection error;
- confidence reliability by error bucket.

### Acceptance gate

No recognition change is promoted merely because its own heuristic score rises.
It must improve a ground-truth or physical metric without regressing the loose,
cushion, and pocket subsets.

## Phase 3 - loose-ball physical silhouette solver

Status: pending

### Design

- Variables are table coordinates `(X, Y)` at ball-center height.
- The camera model projects the known-radius sphere silhouette to source pixels.
- Source edge/color evidence scores the predicted silhouette.
- A free ellipse is an observation/diagnostic, not final geometry.
- Several initial positions are evaluated to expose ambiguity.

### Acceptance gate

- Lower annotated center error than the current final-map ellipse center.
- No count regression on any sample image.
- Uncertainty rises when image evidence is weak or multiple minima exist.
- Approximate-camera mode caps physical/final confidence.

## Phase 4 - joint cluster solver

Status: pending

### Replace, do not extend, the late repair stack

The current contact adjustment, shape consensus, rejected-arc combinations, and
promotion gates remain diagnostics until the replacement passes its gate.

The replacement operates on a complete connected component with:

- center and existence variables per hypothesis;
- hard non-overlap constraints;
- uncertain/latent contact edges;
- projected per-ball silhouettes;
- a whole-cluster foreground/union objective;
- global boundary-arc ownership;
- duplicate and missing hypotheses;
- multiple starting solutions and an explicit abstain result.

Perimeter/interior classification may initialize weights, but no sequential
walk is allowed to become the primary solver.

### Acceptance gate

- No promoted component contains impossible overlaps.
- DSC00540 explains all 15 red balls without oversized free ellipses.
- DSC00542 does not regress.
- Missing and duplicate decisions are explicit.
- Component objective and physical validation improve over the independent-ball
  baseline.

## Phase 5 - confidence calibration

Status: pending

Expose separate values for:

- existence confidence;
- label confidence;
- image-boundary support;
- physical projection agreement;
- scene/cluster consistency;
- multi-start solution stability;
- final positional uncertainty.

Map these components to user-facing levels only after comparison with the gold
benchmark. Human confidence remains null unless explicitly entered.

## Phase 6 - calibrated camera

Status: waiting for board images

- Calibrate Sony/lens intrinsics and distortion with the CALITAR ChArUco board.
- Estimate and validate camera pose relative to the table.
- Record reprojection errors and calibration provenance.
- Replace approximate projection evidence without changing recognition APIs.

Calibration improves physical projection and millimeter accuracy. It does not
replace image evidence, annotations, or the cluster solver.

## Phase 7 - production cleanup

Status: pending

- Make the canonical schema native estimator output; remove legacy report
  adaptation from the normal path.
- Split oversized modules into evidence, loose solver, cluster solver,
  confidence, API, and asset-writing ownership.
- Delete old wrappers, traversal experiments, duplicate confidence paths, and
  static report implementations after their tests migrate.
- Update package version and package discovery.
- Keep generated reports outside Git and retain only benchmark summaries.

## Immediate implementation order

1. Ground-truth schema and storage API.
2. Perfect-ellipse editor.
3. Shared evidence scoring/service.
4. Experiment API.
5. Interactive experiment controls and baseline comparison.
6. Gold benchmark runner.
7. Loose-ball solver.
8. Joint cluster solver.
9. Confidence calibration and final cleanup.

## Commit policy

- Commit at every passing acceptance gate.
- Keep human annotations in their own commits where practical.
- Do not mix generated report assets with source commits.
- Every algorithm commit records the benchmark command and before/after result.
