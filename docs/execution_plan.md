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

Status: complete

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
- DSC00524 has 22/22 saved perfect ellipses and is the loose-ball regression
  baseline;
- saved annotations remain independent of detector/report output;
- the experiment endpoint, ellipse editor, reload behavior, and benchmark
  contracts are covered by the passing test suite.

## Phase 2 - tracked gold benchmark

Status: in progress

Current annotation coverage:

- complete: DSC00524 and DSC00540;
- being reviewed: DSC00529, DSC00534, and DSC00542;
- not yet part of an algorithm promotion gate: DSC00525 and DSC00543.

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

Status: intact-rack slice complete; arbitrary clusters pending

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

### Implemented intact-rack path

The promoted DSC00540 path is deliberately narrower than the final arbitrary
cluster design:

1. Build the physical contact graph and require one connected 15-red component.
2. Estimate a robust shared ellipse size and orientation from plausible
   independently fitted members.
3. Estimate hexagonal lattice phase and spacing from repeated contact vectors.
4. Enumerate triangular rack orientations and use a global Hungarian assignment.
5. Select the lattice with the largest consensus of accurate independent
   anchors before minimizing residual. Grossly wrong members cannot translate
   the whole rack.
6. Deduplicate the union of every member's raw boundary samples and assign each
   sample to at most one proposed silhouette. Ambiguous contact pixels and
   interior highlights remain unowned diagnostics.
7. Refine every center against its uniquely owned arcs while keeping the shared
   shape and lattice displacement bound.
8. Promote all 15 members together only when anchor, per-node boundary support,
   and non-overlap gates pass. Otherwise the result is diagnostic-only.

The solver consumes the same independent state for every member. It does not
walk clockwise/counter-clockwise and cannot propagate one promoted fit into the
next ball. The old per-ball arc-combination and sequential joint-center
promotion switches are disabled by default.

### Measured gate (2026-07-11)

Perfect-ellipse benchmark, 22/22 balls in both images:

| Image | Metric | Before | Joint solver |
|---|---:|---:|---:|
| DSC00540 | mean source-center error | 8.648 px | 1.217 px |
| DSC00540 | median source-center error | 0.799 px | 0.763 px |
| DSC00540 | mean contour RMS | 7.514 px | 1.773 px |
| DSC00540 | median contour RMS | 1.091 px | 1.110 px |
| DSC00524 | mean source-center error | 0.666 px | 0.666 px |
| DSC00524 | mean contour RMS | 0.841 px | 0.841 px |

The DSC00540 rack gate used nine independent lattice anchors at 3.893 px RMS,
unique boundary support on all 15 red nodes, zero hard world overlaps, and
1.354 mm RMS touching-distance error under the approximate camera model.

Ellipse QA now matches detections to annotations one-to-one by class and nearest
source position. Canonical red IDs are display slots, not persistent physical
identities; a subpixel update may exchange two nearly level red slots without
changing geometry.

Remaining work:

- generalize promotion beyond the exact 15-red rack;
- use DSC00542 annotations as the first arbitrary-cluster gate;
- add explicit existence, duplicate, and missing-hypothesis variables;
- benchmark cushion and pocket subsets before broad promotion.

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

1. Complete DSC00529, DSC00534, and DSC00542 gold annotations.
2. Generalize global ownership/joint fitting to arbitrary connected clusters,
   gated first by DSC00542.
3. Implement the loose-ball physical silhouette solver.
4. Add existence/duplicate/missing hypotheses to scene optimization.
5. Calibrate confidence against the expanded gold set.
6. Finish calibrated-camera and production-cleanup phases.

## Commit policy

- Commit at every passing acceptance gate.
- Keep human annotations in their own commits where practical.
- Do not mix generated report assets with source commits.
- Every algorithm commit records the benchmark command and before/after result.
