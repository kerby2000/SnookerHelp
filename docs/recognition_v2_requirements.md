# SnookerHelp Recognition v2 Fresh Implementation Requirements

Status: draft requirements for a fresh implementation  
Date: 2026-07-07  
Scope: recognition, geometry, confidence, review evidence, and validation  
Explicit decision: do not continue growing the current prototype path as the main algorithm

## 1. Purpose

This document defines a clean recognition v2 implementation for SnookerHelp.

The current v1 prototype produced useful learning, but it accumulated too many competing ideas:

- per-ball circle/radial fitting;
- mask contour fitting;
- many evidence maps;
- candidate A/B/C/D terminology;
- dense-cluster traversal experiments;
- add-back experiments;
- fixed-rack assumptions;
- review UI overlays that expose internal algorithm details too early.

The v2 goal is not to patch all of that. The goal is to build a smaller, cleaner recognition system with explicit contracts, measurable validation, and a review UI that explains the result without leaking obsolete prototype concepts.

## 2. Main conclusion from v1

The central problem is not just “find a better ellipse per ball.”

For isolated balls, per-ball evidence works well enough. For dense clusters, especially `DSC00540`, the visible contour of an inner ball is incomplete and contaminated by neighboring balls, highlights, shadows, and contact regions. A per-ball ellipse can look mathematically valid while being physically impossible.

The v2 implementation must therefore solve dense clusters as a group.

The main model should be:

source image evidence + physical ball constraints + cluster graph optimization

not:

one independent ellipse fit per ball, then hope the cluster is consistent

## 3. Non-negotiable product rules

### 3.1 User-facing language

The v2 UI must not expose these as primary user concepts:

- Candidate A
- Candidate B
- Candidate C
- Candidate D
- radial model
- mask model
- add-back model
- traversal model
- fixed-rack model
- arbitrary reject-color taxonomy

The user-facing concepts are:

- Pixels
- Image evidence
- Physical model
- Final estimate
- Confidence
- Manual correction
- Validation

Internal debug names may exist in JSON, but the main UI must use the product language above.

### 3.2 Main overlay colors

The main selected-ball overlay must remain simple:

| Visual | Meaning |
| --- | --- |
| White dots | accepted image boundary evidence used by the current estimate |
| Red dots | rejected or unused image boundary evidence |
| Cream outline | final observed/estimated ball outline used for the selected estimate |
| Green cross | final source-pixel center used for table coordinates |
| Optional blue outline | physical projection prediction, only when explicitly enabled |

Do not add multiple colored reject categories in the primary UI. Rejection reasons can be inspected in text tables or advanced debug panels.

### 3.3 Confidence language

Do not call a result “high confidence” because a displayed ellipse looks smooth.

Confidence must come from measurable agreement between:

- image evidence;
- physical projection;
- ball size/shape plausibility;
- cluster graph consistency;
- neighbor/contact constraints;
- duplicate/missing-ball checks;
- validation tests when available.

## 4. Scope

### 4.1 In scope

Recognition v2 must handle:

- loose single balls;
- balls near cushions;
- balls near pockets;
- touching pairs;
- small arbitrary clusters;
- large arbitrary clusters;
- intact or partly broken red triangle/rack clusters;
- multiple evidence maps for diagnostics and scoring;
- current Sony sample images;
- future calibrated camera model;
- approximate camera model before ChArUco calibration is available.

### 4.2 Out of scope for this implementation

Do not add:

- YOLO or other neural object detector;
- pool/snooker physics simulation;
- projector code;
- Basler camera capture;
- game-state reasoning;
- shot planning;
- automatic pocket detection beyond existing table geometry;
- hard-coded fixed 15-red rack solver as the main cluster algorithm.

A rack-specific helper may exist later as one optional prior, but not as the primary solution.

## 5. Required inputs

The recognition v2 pipeline must accept these inputs:

| Input | Required now | Notes |
| --- | --- | --- |
| Source image | Yes | Full-resolution camera photo |
| Table geometry | Yes | Table dimensions and corner pixels or calibrated pose |
| Ball diameter | Yes | Default 52.5 mm |
| Ball radius | Yes | Default 26.25 mm |
| Approximate camera model | Yes | May be homography/pinhole approximation before ChArUco |
| Calibrated camera model | Future | Intrinsics, distortion, extrinsics |
| Detector config | Yes | Evidence thresholds, cluster parameters, scoring weights |
| Scenario metadata | Optional | Touching pairs, cushion contacts, spots, repeatability groups |
| Manual review feedback | Optional | Corrections and accept/reject decisions |

## 6. Required outputs

Recognition v2 must produce one canonical output schema per processed image.

### 6.1 Image-level output

Required fields:

- `schema_version`
- `image_name`
- `image_path`
- `image_size_px`
- `camera_model`
- `table_model`
- `ball_diameter_mm`
- `ball_radius_mm`
- `estimates`
- `clusters`
- `diagnostics`
- `validation`

### 6.2 Ball estimate output

Each ball estimate must include:

- stable image-independent ball ID where possible;
- image-local raw detector ID;
- color label;
- color confidence;
- source-pixel center;
- table XY estimate;
- effective Z plane used;
- projected physical shape;
- observed image evidence shape;
- final selected outline;
- confidence score;
- confidence reasons;
- cluster membership;
- duplicate/missing status;
- manual correction state if present.

Example shape:

```json
{
  "canonical_id": "red_07",
  "raw_detector_id": 13,
  "label": "red",
  "source_center_px": [1234.5, 2048.2],
  "table_xy_mm": [1820.1, 763.4],
  "effective_z_mm": 26.25,
  "confidence": {
    "score": 0.82,
    "level": "medium",
    "reasons": [
      "image_boundary_supported",
      "cluster_contact_consistent",
      "physical_projection_residual_medium"
    ]
  },
  "cluster": {
    "cluster_id": "cluster_03",
    "role": "interior",
    "contact_degree": 4
  }
}
```

### 6.3 Debug output

Debug output must be structured, not UI-specific.

It must include:

- evidence maps available for each ball crop;
- accepted/rejected boundary samples;
- boundary arc components;
- fit candidates evaluated;
- cluster graph nodes and edges;
- optimization energy terms;
- duplicate hypotheses;
- missing-ball hypotheses;
- rejected solution reasons;
- selected solution reason.

## 7. Coordinate systems

The implementation must explicitly distinguish:

| Coordinate system | Meaning |
| --- | --- |
| `source_px` | Original camera image pixels |
| `undistorted_px` | Lens-undistorted camera pixels, once calibration exists |
| `warped_px` | Cloth-plane debug rectification only |
| `table_mm` | Physical table coordinates |
| `world_mm` | Full 3D table coordinate frame with Z |

Important rule:

Final ball fitting must happen in source or undistorted image coordinates, not in the warped cloth-plane image.

The warped view may remain only for:

- rough detection;
- debug visualization;
- table masking;
- approximate region grouping;
- compatibility with current sample pipeline.

It must not be used as evidence that ball shapes should be circular.

## 8. Camera model requirements

### 8.1 Current approximate model

Until ChArUco calibration exists, v2 may use an approximate camera model based on:

- image resolution;
- table corner pixels;
- table dimensions;
- approximate camera height;
- approximate camera XY relative to table;
- approximate focal length;
- sensor size;
- ball radius.

This approximate model is a weak prior. It is not ground truth.

### 8.2 Future calibrated model

The interface must support replacing the approximate model with real ChArUco calibration:

- camera matrix;
- distortion coefficients;
- camera pose relative to table;
- reprojection error;
- calibration timestamp;
- calibration image set;
- board specification.

The bought ChArUco board specification to support:

- CALITAR target;
- 15 x 20 checkerboard;
- checker size 32.00 mm;
- marker size 24.00 mm;
- dictionary `DICT_5X5_1000`.

### 8.3 Required camera-model API

The API must include:

```python
undistort_points(points_px) -> points_px
image_point_to_world_ray(point_px) -> Ray3D
intersect_ray_with_z(ray, z_mm) -> point_xyz_mm
world_point_to_image(point_xyz_mm) -> point_px
project_sphere(center_xyz_mm, radius_mm) -> ProjectedShape
```

`project_sphere()` is important. It is the source of the physical projected outline used as a prior and validation check.

## 9. Core v2 data concepts

### 9.1 ImageModel

Represents image-space evidence for a crop or full image:

- source RGB crop;
- evidence maps;
- boundary samples;
- boundary arcs;
- observed shape candidates;
- local evidence quality metrics.

### 9.2 PhysicalModel

Represents table/camera/ball physical constraints:

- ball radius;
- table dimensions;
- camera model;
- expected projected sphere outline;
- expected contact distance;
- expected no-overlap constraints;
- expected cushion distances where applicable.

### 9.3 BallHypothesis

A possible ball before final acceptance:

- approximate center;
- label/color estimate;
- rough radius;
- source crop;
- evidence map results;
- possible physical projection;
- detector provenance.

### 9.4 BoundarySample

A sampled image point that may belong to the visible boundary:

- source pixel coordinate;
- evidence map name;
- evidence strength;
- radial angle around candidate;
- local gradient direction;
- accepted/rejected status;
- rejection reason list;
- neighbor ownership score;
- associated boundary arc ID.

### 9.5 BoundaryArc

A connected or angularly coherent group of boundary samples:

- arc ID;
- sample IDs;
- angular start/end;
- arc length;
- mean evidence strength;
- fit residual to candidate outline;
- ownership probability.

Important: v2 must reason over arcs, not only individual dots.

### 9.6 ClusterGraph

A graph describing a group of nearby or touching balls:

- nodes are ball hypotheses;
- edges are possible physical relationships;
- graph supports unknown/missing nodes;
- graph supports duplicate hypotheses;
- graph supports contact and non-overlap constraints.

### 9.7 BallEstimate

Final accepted estimate:

- source-pixel center;
- table position;
- selected image evidence;
- selected physical interpretation;
- confidence;
- diagnostics.

### 9.8 ReviewFeedback

Human feedback must be stored separately from algorithm output:

- OK/NOK decision;
- manual center if entered;
- missing-ball marker;
- duplicate marker;
- comment;
- confidence from human only if user explicitly sets it.

Do not auto-fill human confidence with a default that looks like user input.

## 10. Pipeline overview

Recognition v2 must run in these stages.

### Stage 1 — Load image and geometry

Inputs:

- source image;
- table geometry;
- camera model;
- detector config.

Outputs:

- normalized image context;
- table mask;
- cloth reference model.

### Stage 2 — Rough ball hypotheses

Generate initial ball hypotheses.

The implementation may reuse existing rough detector code as a temporary source of hypotheses, but v2 must treat it as replaceable.

Requirements:

- preserve raw detector IDs;
- keep all plausible hypotheses before duplicate suppression;
- mark low-confidence rough detections instead of silently deleting them;
- allow future alternate detectors to provide hypotheses through the same interface.

### Stage 3 — Evidence maps

Compute diagnostic and scoring maps per crop.

Required maps:

- source image;
- grayscale edge;
- Lab Delta-E;
- chroma difference;
- ball-vs-cloth probability;
- physical projection band;
- combined boundary score.

Default evidence policy learned from samples:

| Ball class | Default evidence priority |
| --- | --- |
| red | ball-vs-cloth probability, then source image/chroma as fallback |
| yellow | ball-vs-cloth probability |
| white | ball-vs-cloth probability or grayscale edge |
| black | source image and edge/chroma depending on exposure |
| brown | chroma difference often better |
| blue | chroma difference often better |
| green | chroma difference often better |

Only the selected evidence map should influence the current final estimate. Other maps are diagnostics unless explicitly used by a configured ensemble.

### Stage 4 — Boundary sampling

For each ball hypothesis, sample boundary evidence around the expected ball outline.

Requirements:

- sample along rays or local normals around the candidate;
- keep multiple candidate points per angular sector before final filtering;
- group samples into boundary arcs;
- store all samples with reasoned accepted/rejected status;
- do not permanently discard points before cluster solving.

### Stage 5 — Loose-ball solver

If a ball is isolated, solve it with a per-ball image model plus physical projection prior.

Loose-ball acceptance must check:

- sufficient arc coverage;
- plausible projected size;
- plausible ellipse/circle shape for its table region;
- agreement with physical projection;
- no impossible overlap with neighbors;
- color-label plausibility.

Loose-ball output can be accepted without cluster optimization if all gates pass.

### Stage 6 — Cluster detection

Build clusters from rough hypotheses.

Cluster membership is based on:

- source-pixel distance;
- table-mm distance where available;
- projected shape overlap;
- color/class expectations;
- local table region;
- detected duplicate/overlap signs.

Cluster types:

- no adjacent cluster;
- touching pair;
- small cluster;
- dense cluster;
- red-rack-like cluster;
- arbitrary large cluster.

The implementation must not assume that a large red cluster is always an intact 15-red triangle.

### Stage 7 — Cluster graph construction

For each cluster, build a graph:

- node per detected ball hypothesis;
- optional missing-node hypotheses;
- duplicate-node hypotheses;
- contact edges;
- near-contact edges;
- overlap edges;
- non-contact edges;
- evidence ownership edges between arcs and balls.

Edges must carry soft weights, not only hard decisions.

Example edge fields:

```json
{
  "node_a": "red_05",
  "node_b": "red_09",
  "relationship": "possible_contact",
  "distance_mm": 54.2,
  "expected_mm": 52.5,
  "weight": 0.72,
  "reason": ["near_expected_touching_distance", "visible_gap_small"]
}
```

### Stage 8 — Cluster graph optimization

Dense clusters must be solved jointly.

The optimizer must estimate:

- ball centers;
- boundary ownership;
- duplicate suppressions;
- missing-ball hypotheses;
- cluster shape consistency;
- final confidence per ball and per cluster.

The energy function should include these terms:

| Term | Purpose |
| --- | --- |
| Image evidence support | selected centers/shapes must explain boundary arcs |
| Boundary ownership | one boundary arc should not explain multiple incompatible balls |
| Equal-radius constraint | all balls have same physical radius |
| Shared local projected-shape prior | nearby balls should have similar projected ellipse size/orientation |
| Contact distance | touching candidates should be near 52.5 mm center distance |
| No-overlap | centers must not imply impossible ball intersections |
| Cluster mask support | union of projected balls should explain the observed ball-colored region |
| Duplicate penalty | overlapping hypotheses for one ball should collapse |
| Missing penalty | unexplained cluster evidence should create a missing-ball hypothesis |
| Camera prior | physical projection should be plausible but weak before calibration |

Important:

- The optimizer must evaluate alternative cluster solutions.
- It must not trust one sequential traversal order as the main algorithm.
- Clockwise/counterclockwise traversal may be diagnostics or initialization only.
- Perimeter/interior classification may influence weights, but must not be the solver itself.

### Stage 9 — Final estimate selection

For each ball:

- use loose-ball result if isolated and high quality;
- use cluster-optimized result if clustered;
- suppress duplicate detections;
- mark missing hypotheses separately;
- export final source center and table XY.

If no reliable estimate exists, mark as `needs_review` instead of forcing a wrong ellipse.

### Stage 10 — Validation and reporting

Generate:

- machine-readable JSON;
- visual review report;
- physical validation metrics;
- benchmark summary.

## 11. Cluster-specific requirements

### 11.1 Dense clusters cannot rely on cloth boundary

Interior balls may have little or no visible cloth around them. Their estimate must come from:

- contacts with neighboring balls;
- shared projected shape;
- visible partial arcs;
- cluster mask/union support;
- known equal ball radius.

### 11.2 Neighbor highlights must not become boundary evidence

The solver must treat specular highlight rectangles and lamp reflections as possible false arcs.

Detection requirements:

- identify high-brightness rectangular highlight regions;
- down-weight samples inside highlight interiors;
- down-weight thin straight-line highlight edges;
- avoid using another ball’s highlight as the selected ball boundary.

This does not require perfect highlight segmentation, but highlight evidence must not dominate dense-cluster fitting.

### 11.3 Arc-combination fitting

For each ball, the implementation may generate several arc combinations.

The combinations must be selected at the arc level, not individual point level.

For `N` arc clusters, evaluate candidate subsets with guardrails:

- all single arcs only for diagnostics;
- pairs and triples when enough angular coverage exists;
- larger combinations when they remain physically plausible;
- cap total combinations to prevent exponential blowup;
- rank candidates by image support plus physical plausibility.

The selected result must not be promoted unless it also passes cluster-level consistency.

### 11.4 Shared shape prior

Balls close together on the table must have similar projected shape.

Use local consensus from reliable neighbors:

- median major axis;
- median minor axis;
- median orientation with 180-degree wrap handling;
- robust spread estimate;
- confidence based on number and quality of supporting neighbors.

A ball is suspicious if:

- its projected size is far larger or smaller than local consensus;
- its orientation differs materially from local consensus;
- its center implies impossible overlap/contact geometry;
- it uses arcs owned by another ball.

This should be a correction signal, not merely a warning.

### 11.5 Duplicate handling

If two hypotheses describe the same physical ball, v2 must explicitly mark one as duplicate/suppressed.

Duplicate criteria:

- source centers too close;
- projected outlines overlap too much;
- same label/color;
- evidence arcs largely shared;
- contact graph cannot place both without impossible overlap.

The output must distinguish:

- accepted ball;
- suppressed duplicate;
- unresolved duplicate requiring review.

### 11.6 Missing-ball handling

A missing-ball hypothesis must be created when:

- cluster mask has unexplained ball-colored area;
- expected contact graph has a gap;
- physical validation indicates expected touching distance cannot be satisfied;
- manual review marks a missing ball.

Missing hypotheses must not be silently inserted as accepted balls. They must be flagged as:

- `missing_hypothesis`;
- `needs_review`;
- or `accepted_by_physical_prior` only after strict evidence gates.

## 12. Evidence-map requirements

### 12.1 Global cloth reference

Use a global cloth reference as default.

Rationale:

- local annulus is often contaminated by neighboring balls in clusters;
- local cloth is absent for inner cluster balls;
- global cloth gives stable comparison across the image.

Local annulus may still be recorded as a diagnostic contamination indicator.

### 12.2 Ball-vs-cloth probability

This is the default evidence map for most loose balls and many red balls.

Required diagnostics:

- ball Lab estimate;
- global cloth Lab;
- local annulus Lab;
- ball-cloth separation;
- sample counts;
- low-contrast flag;
- saturation/exposure flag;
- active weights.

### 12.3 Chroma difference

Chroma difference is often better for:

- green;
- blue;
- brown;
- some red cases where ball-vs-cloth probability is polluted by highlights or neighbors.

The selector must allow class-specific preference.

### 12.4 Lab Delta-E

Lab Delta-E is useful as an alternate diagnostic and sometimes fits red clusters well, but it must not override the default policy without scoring evidence.

### 12.5 Grayscale edge

Grayscale edge is diagnostic and may help white/black/high-contrast edges. It should not be the primary final source for colored balls unless it wins explicit scoring gates.

### 12.6 Physical projection band

The physical projection band is a prior and diagnostic, not image evidence.

Do not manufacture boundary samples from the physical projection band and then count them as observed image evidence.

## 13. Confidence requirements

Confidence must be explainable.

Each score must report components:

- image evidence score;
- arc coverage score;
- fit residual score;
- physical projection agreement score;
- local shape consensus score;
- contact graph consistency score;
- duplicate/missing penalty;
- calibration quality penalty;
- manual override state.

Suggested levels:

| Level | Score range | Meaning |
| --- | --- | --- |
| high | >= 0.85 | trusted automatically for normal downstream use |
| medium | 0.60 to 0.85 | likely correct but review recommended for difficult cases |
| low | < 0.60 | not trusted; review or alternate solution required |

These thresholds are initial defaults and must be calibrated with benchmarks.

Confidence must not use manual feedback unless the report explicitly labels it as human feedback.

## 14. Review UI requirements

### 14.1 Main layout

The v2 review UI must show:

- full source image with pan/zoom;
- selected-ball crop;
- evidence selector;
- simple overlay toggles;
- ball statistics table;
- confidence explanation;
- physical model summary;
- validation summary when available.

### 14.2 Full image panel

Requirements:

- pan and zoom;
- readable labels at all zoom levels;
- click ball to select;
- fit selected view to the selected ball/cluster;
- show canonical ball IDs;
- show suppressed duplicates and missing hypotheses differently from accepted balls.

### 14.3 Selected-ball crop

Requirements:

- large crop;
- selectable background: source image or evidence map;
- overlay toggles:
  - accepted dots;
  - rejected dots;
  - final outline;
  - final center;
  - rough center;
  - physical projection;
  - neighbor outlines;
- no scale overlay by default unless requested.

### 14.4 Evidence table

For each evidence view, show:

- score;
- selected/unused status;
- accepted point count;
- rejected point count;
- fit RMS;
- arc coverage;
- whether it contributes to final estimate.

The table may allow checking overlays per evidence view, but final estimate selection must be clearly marked.

### 14.5 Cluster view

For clusters, show:

- cluster ID;
- member balls;
- cluster graph edges;
- accepted contacts;
- rejected impossible contacts;
- duplicate suppressions;
- missing hypotheses;
- final cluster solution score;
- local shared-shape statistics.

This is more important than showing many per-ball experimental colors.

### 14.6 Manual correction

Manual correction should be minimal:

- accept estimate;
- reject estimate;
- mark duplicate;
- mark missing ball;
- optional manual source center;
- optional note.

Manual correction must not dominate the UI. The user primarily wants to diagnose the algorithm, not hand-label every ball.

## 15. Validation and benchmarks

The fresh implementation must be benchmark-driven.

### 15.1 Required benchmark image groups

| Group | Sample images | Purpose |
| --- | --- | --- |
| Loose/random balls | `DSC00524`, `DSC00525`, `DSC00526`, `DSC00527` | Ensure normal cases do not regress |
| Near cushions | `DSC00529` and similar | Edge geometry and cushion proximity |
| Near pockets | `DSC00534` and similar | Pocket/corner edge cases |
| Dense triangle | `DSC00540` | Hard cluster/rack case |
| Arbitrary cluster | `DSC00541`, `DSC00542` | Non-rack dense cluster behavior |
| Empty table | `DSC00543` and similar | Cloth reference, false positives |

### 15.2 Required metrics

Per image:

- number of accepted balls;
- number of suppressed duplicates;
- number of missing hypotheses;
- number of low-confidence estimates;
- mean confidence;
- per-ball confidence;
- physical overlap violations;
- touching-distance errors where known;
- cushion-distance errors where known;
- table-coordinate repeatability where repeated images exist.

Per cluster:

- cluster size;
- graph edge count;
- accepted contact count;
- rejected contact count;
- duplicate suppressions;
- missing hypotheses;
- shared-shape spread;
- cluster solution energy;
- unresolved nodes.

Per ball:

- selected evidence map;
- accepted boundary samples;
- rejected boundary samples;
- arc coverage;
- fit RMS;
- physical residual;
- local consensus residual;
- final confidence.

### 15.3 Acceptance gates before replacing v1

Minimum gates:

1. Loose-ball images must not regress versus the current best v1 reports.
2. Dense-cluster images must not promote physically impossible oversized ellipses as accepted final estimates.
3. Duplicate detections must be explicitly suppressed or marked unresolved.
4. Missing balls must be explicitly represented, not hidden.
5. The UI must make it clear why a ball is trusted or not trusted.
6. The same report must be reproducible from CLI.
7. Tests must cover loose solving, cluster solving, duplicate suppression, missing hypotheses, and schema output.

Suggested initial numeric targets:

| Metric | Initial target |
| --- | --- |
| Loose-ball accepted count | expected image inventory, no obvious misses |
| Dense-cluster impossible overlaps | 0 accepted hard violations |
| Dense-cluster duplicate accepted pairs | 0 obvious duplicates |
| Touching pair distance error | report in mm; target to be calibrated |
| Report generation time | < 30 s per image in development mode |
| Cached selected-ball UI update | < 1 s |

Do not treat these targets as final scientific thresholds until more ground truth exists.

## 16. Testing requirements

### 16.1 Unit tests

Required unit tests:

- evidence map generation;
- global cloth reference selection;
- boundary sampling;
- arc extraction;
- robust ellipse fitting;
- projected sphere shape;
- cluster graph construction;
- graph energy terms;
- duplicate suppression;
- missing hypothesis creation;
- schema serialization.

### 16.2 Synthetic tests

Synthetic tests must include:

- isolated ellipse with known center;
- partial arcs;
- neighboring ball contamination;
- highlight-like rectangular false arcs;
- duplicate hypothesis pair;
- missing node in a small cluster;
- local shared-shape outlier.

### 16.3 Regression tests

Regression tests must run selected real images and compare:

- accepted count;
- duplicate count;
- missing count;
- confidence distribution;
- selected evidence map;
- cluster solution diagnostics.

Do not require exact pixel-perfect overlays for regression unless the rendering is the subject of the test.

## 17. Performance requirements

Dense-cluster solving can become combinatorial.

Requirements:

- cap arc-combination enumeration;
- use robust pruning;
- solve connected components independently;
- cache evidence maps;
- cache boundary samples;
- cache projected shapes;
- support debug level configuration;
- export timing per stage.

The implementation must report when it skipped expensive alternatives because of configured limits.

## 18. Fresh implementation structure

Use a clean implementation namespace.

Suggested structure:

```text
snookerhelp/
  core/
    schema.py
    geometry.py
    ball_ids.py
  calibration/
    camera_model.py
    approximate_camera.py
    charuco_calibration.py
  recognition_v2/
    pipeline.py
    rough_hypotheses.py
    evidence_maps.py
    boundary_sampling.py
    arc_extraction.py
    loose_solver.py
    cluster_graph.py
    cluster_solver.py
    duplicate_missing.py
    confidence.py
    outputs.py
  review_v2/
    server.py
    static/
    schema.py
  qa/
    benchmark.py
    validation.py
    report.py
```

If keeping the package name `snookerhelp/recognition/`, the v2 code must still be separated clearly from old prototype modules. Do not import prototype modules into v2 unless the dependency is explicitly approved and covered by tests.

## 19. What may be reused

Reusable with caution:

| Existing asset | Reuse condition |
| --- | --- |
| Sample media | Keep |
| Table corner calibration inputs | Keep |
| Manual homography code | Temporary rough detection/debug only |
| Approximate camera model interface | Keep and clean |
| ChArUco calibration plan | Keep |
| Global cloth reference learning | Keep |
| Evidence map code | Reuse only behind clean v2 interface |
| Physical validation tools | Keep and adapt |
| Review feedback files | Keep as evidence, not training truth |
| Existing tests | Keep, update for v2 schema |

## 20. What must not be carried forward as main design

Do not carry forward as primary v2 design:

- candidate A/B/C/D user model;
- independent per-ball radial-circle model as final authority;
- mask centroid as final authority;
- fixed 15-red rack traversal as the main solver;
- perimeter-first sequential recognition as the main solver;
- physical projection band as fake observed evidence;
- multiple reject colors in the main UI;
- local annulus cloth reference as default;
- monolithic review app file;
- static report-only workflow as the main review experience;
- confidence based on visual smoothness;
- silent duplicate removal;
- silent missing-ball failure.

## 21. Migration plan

### Phase 0 — Freeze prototype

Stop adding recognition logic to the current v1 prototype except for small bug fixes needed to inspect existing evidence.

### Phase 1 — Create v2 schema and adapters

Deliverables:

- v2 data classes;
- JSON schema;
- adapter from current detector output to v2 rough hypotheses;
- schema tests.

No algorithm changes yet.

### Phase 2 — Evidence and arc extraction

Deliverables:

- global cloth reference;
- evidence maps behind one interface;
- boundary sampling;
- arc extraction;
- per-evidence scoring;
- selected-ball visual debug.

### Phase 3 — Loose-ball solver

Deliverables:

- isolated-ball solution path;
- physical projection prior;
- confidence components;
- benchmark on loose images.

Acceptance:

- loose-ball images match or exceed current v1 quality.

### Phase 4 — Cluster graph model

Deliverables:

- cluster graph builder;
- neighbor/contact edges;
- duplicate hypotheses;
- missing hypotheses;
- boundary ownership model;
- cluster diagnostics UI.

No final promotion until benchmarked.

### Phase 5 — Cluster optimizer

Deliverables:

- joint center solver;
- shared-shape prior;
- contact/no-overlap constraints;
- cluster mask support;
- solution ranking;
- dense-cluster benchmarks.

Acceptance:

- `DSC00540` no longer accepts obviously oversized wrong ellipses as final estimates.
- `DSC00542` remains good.

### Phase 6 — Review UI v2

Deliverables:

- clean v2 UI against v2 schema;
- no old candidate terminology;
- pan/zoom full image;
- selected-ball crop;
- cluster view;
- confidence explanation.

Do not refactor the old 2000-line review UI into this. Build the v2 UI against the v2 schema.

### Phase 7 — Replace default pipeline

Deliverables:

- CLI default uses v2;
- current v1 prototype moved behind compatibility flag or legacy path;
- benchmark report attached.

### Phase 8 — Delete/archive old code

Only after v2 passes gates:

- move old prototype modules to `legacy/` or delete;
- update docs;
- update README;
- remove obsolete UI terms.

## 22. Open technical decisions

These must be decided by experiment, not assumption:

1. Whether cluster mask support can be robust enough without a learned segmentation model.
2. Whether approximate camera model is good enough to improve dense clusters before ChArUco calibration.
3. How strict local shared-shape consensus should be near table edges.
4. How to tune duplicate/missing penalties.
5. How to score inner balls with almost no visible cloth boundary.
6. Whether a small amount of manual scenario metadata is acceptable for difficult validation images.
7. Whether a later ML segmentation step is justified after the classical v2 limit is measured.

## 23. Immediate next action

Before writing v2 code, create a small design review package from this document:

- one page explaining the cluster graph model;
- one page with `DSC00540` failure examples;
- one page with expected v2 output JSON;
- one page with benchmark gates.

Then implement Phase 1 only.

The next implementation step should not be another heuristic inside the current prototype.

