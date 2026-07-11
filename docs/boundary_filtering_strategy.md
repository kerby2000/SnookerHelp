# Boundary filtering and evidence-map strategy

This document describes the current v1 boundary-evidence pipeline used by the
review UI and by physical-model scoring.

Current status: v1.6.0 plus generic cluster graph / boundary-ownership
diagnostics, full-table global evidence-map normalization, live backend
experiments, and tracked perfect-ellipse ground truth.

## Short version

The active pipeline is now:

```text
source crop
  -> radial source edge-boundary samples
  -> conservative outlier filtering
  -> accepted white points + rejected red points
  -> cream observed ellipse fitted from accepted white points only
  -> optional evidence-map-specific boundary variants for review
  -> active cloth reference for color maps, global by default
  -> full-table global Lab/chroma/edge maps cropped back to each ROI
  -> blue projected sphere outline from the camera/table/ball model
  -> large adjacent-cluster shell diagnostics
  -> generic cluster graph + boundary ownership diagnostics
  -> gated arc-combination promotion for dense-cluster outliers
  -> diagnostic evidence maps written for review
  -> configured final-position evidence-map policy
  -> physical optimizer scores projected outlines against accepted image evidence
```

The cyan recovered-points experiment is removed from the active pipeline.

The cyan solid ellipse in v1.6.0 has a different meaning: it is a human
perfect-ellipse annotation stored under `benchmarks/annotations/`. It is never
fed into production recognition. It provides an independent target for testing
map and filtering parameters.

The physical projection band still exists, but only as a diagnostic map and a
weak score term. It no longer creates extra cyan boundary points and it no
longer adds points to the cream ellipse fit.

v1.3.6 promotes exactly one evidence-map boundary variant into the final
source/table position:

```text
default balls:             ball_vs_cloth_probability
green / blue / brown:      chroma_difference
all other evidence maps:   diagnostic only
```

This policy came from visual review of free-standing balls in DSC00524,
DSC00525, DSC00526, and DSC00527. It is intentionally simple and explicit so
it can be benchmarked and rolled back if it hurts clustered, cushion, or pocket
cases.

DSC00542 shows the next limitation clearly: the free-ball policy is useful for
loose balls, but it is not sufficient for dense clusters where the crop contains
other balls, shadows, and very little real cloth.

The first measured intact-rack baseline is `DSC00540`. With 22 manually fitted
ellipses and a default 3 px annotation tolerance, most fits are already close,
but interior reds #12, #14, #17, #19, and #21 have roughly 25-29 px contour RMS
and 32-36 px center error. This isolates the primary remaining failure as
cluster boundary ownership/joint fitting rather than global cloth estimation.
See `benchmarks/results/DSC00540/ellipse_benchmark.json`.

v1.3.8 changes the color reference used by `Lab Delta-E`,
`chroma_difference`, and `ball_vs_cloth_probability`:

```text
old default: cloth Lab estimated from a local annulus around each ball
new default: one global cloth Lab estimated from the table cloth per image
```

The old local annulus is still computed and shown as diagnostic data because it
is useful for finding contaminated crops. It is no longer the default reference
for B/C/L maps.

## What the review UI should show

The v1 UI should use product language, not prototype candidate names:

- Pixels
- Image evidence
- Physical model
- Final estimate
- Confidence
- Manual correction, only where needed

For one selected ball:

- white dots = accepted boundary pixels for the selected evidence view;
- red dots = rejected boundary outliers for the selected evidence view;
- cream dashed line = observed edge ellipse fitted from the selected accepted
  white dots;
- blue dashed line = projected sphere outline from camera/table/ball geometry;
- green cross = final source-pixel estimate used for table XY;
- evidence-map backgrounds = maps that explain where the algorithm
  sees edges/color contrast/projection-band support.

When the evidence background is `Source image`, the white/red points are the
default source-boundary sampler. When the evidence background is an evidence
map, the white/red points and cream ellipse are recomputed from that selected
map.

Do not interpret the blue dashed line as calibrated truth yet. With the current
approximate camera model it is a physics-based prior, not a final authority.

## Current implementation locations

- `snookerhelp/recognition/source_refinement.py`
  - radial source-boundary sampling;
  - accepted/rejected boundary filtering;
  - observed edge ellipse fitting.
- `snookerhelp/recognition/evidence_maps.py`
  - grayscale/color/probability/projection-band maps;
  - full-table global map normalization;
  - global and local cloth-reference diagnostics.
- `snookerhelp/recognition/physical_optimize.py`
  - local physical X/Y search and scoring against source evidence.
- `snookerhelp/recognition/cluster_optimize.py`
  - adjacent-ball diagnostics for close/rack balls.
- `snookerhelp/recognition/cluster_graph.py`
  - generic node/edge graph for arbitrary touching or near-touching clusters;
  - duplicate/overlap/touching/near-touching relationship diagnostics.
- `snookerhelp/recognition/boundary_ownership.py`
  - classifies current boundary samples as target-owned, contact seam,
    neighbor-owned, weak target boundary, or unowned outlier.
- `snookerhelp/recognition/cluster_optimizer.py`
  - combines the legacy adjacent-cluster optimizer, the generic graph, and
    ownership diagnostics into the scene-constraint payload.
- `snookerhelp/review/evidence_builder.py`
  - writes crop-aligned diagnostic evidence-map PNGs into report folders;
  - writes map-specific boundary-point and ellipse variants for the v1 UI.
- `snookerhelp/review/static/`
  - v1 browser UI; lets the user switch evidence background to each evidence map,
    and switches the white/red points plus cream ellipse with it.

## Active filtering stages

The filter is deliberately conservative. It removes obvious bad samples without
inventing new boundary points.

### 1. Angular segment endpoint trimming

Broken boundary arcs often create bad points at the first or last sample of an
arc. The filter detects large angular gaps and trims a small number of samples
near the gap endpoints.

Relevant config:

```yaml
source_refinement:
  boundary_outlier_segment_endpoint_trim_points: 1
  boundary_outlier_segment_gap_factor: 2.25
```

### 2. Local radial-radius consistency

For each angular sample, the filter compares its radius with nearby angular
samples. A point that jumps far inward/outward compared with its local
neighbors is treated as a likely highlight, texture edge, shadow, cushion, or
neighbor artifact.

Relevant config:

```yaml
source_refinement:
  boundary_outlier_window_points: 9
  boundary_outlier_radius_factor: 0.085
  boundary_outlier_min_radius_px: 3.0
```

### 3. Ellipse residual consistency

After local radius filtering, the filter fits a temporary ellipse and rejects
points with excessive residual from that ellipse.

Relevant config:

```yaml
source_refinement:
  boundary_outlier_ellipse_passes: 2
  boundary_outlier_ellipse_radius_factor: 0.08
  boundary_outlier_min_ellipse_residual_px: 3.25
```

This is not the final physical model. It is a robustifying pass before the
cream observed ellipse is fitted from accepted points.

### 4. Neighbor-ellipse ownership filtering

v1.3.9 adds a cluster-specific ownership pass before the final ellipse residual
filter. The problem it addresses is visible on `DSC00542` balls #9 and #14:
the radial sampler can grab lamp-reflection edges or internal texture on
adjacent balls, then use those points as if they belong to the selected ball.

The current rule is intentionally simple and auditable:

1. collect nearby detected ball ellipses in source-pixel coordinates;
2. mark a sampled boundary point as neighbor-owned if it lies inside a nearby
   neighbor ellipse scaled slightly inward;
3. reject those points only if enough selected-ball points remain;
4. keep the rejected points visible as red dots;
5. keep ownership categories in diagnostics; the default UI still uses only
   white accepted dots and red rejected dots.

Relevant config:

```yaml
evidence_maps:
  neighbor_ellipse_rejection_enabled: true
  neighbor_ellipse_rejection_axis_scale: 0.92
  neighbor_ellipse_rejection_distance_factor: 3.2
```

This is more physically meaningful than relying only on a final ellipse
axis-ratio guard. The `maximum_axis_ratio` policy guard still exists, but it is
now an emergency fallback for obviously implausible promoted ellipses, not the
primary defense against cluster contamination.

The report exports:

- `neighbor_ellipses_px`: nearby ellipses used for review/ownership checks;
- `neighbor_ellipse_candidate_count`: sampled points inside neighbor ellipses;
- `neighbor_ellipse_rejected_count`: neighbor-owned points actually removed
  before fitting.

v1 also exports a higher-level ownership payload under each ball's
`source_joint_cluster_optimization.boundary_ownership`. This does not introduce
new UI colors. It is used to explain and score whether a candidate fit is
mostly supported by the selected ball, by a contact seam, or by a neighboring
ball.

### 4b. Arc-combination promotion

For dense clusters, the first filter can reject useful arcs because they are
near neighboring ellipses. The current promoted path is:

1. start from all raw radial boundary samples;
2. split them into angular/spatial arc groups;
3. fit every useful non-empty group combination, capped when there are too many
   groups;
4. compare each candidate ellipse to the same-color cluster consensus size and
   angle;
5. score residual, point coverage, multi-arc support, and boundary ownership;
6. promote only when the best candidate is not a cluster-shape outlier and
   passes conservative residual/coverage gates.

This is the first implementation of the "try combinations, then promote only if
physics/cluster consistency improves" strategy. It is intentionally not a
triangle-only solver.

### 5. Large-cluster shell diagnostics

v1.4 adds a generic shell classifier for any large adjacent-ball component.
This is a scene-level diagnostic, not a rack-only special case.

The current method:

1. build adjacent-ball components from table-coordinate anchors;
2. for components with at least `shell_classification_min_size` balls, compute
   a convex hull;
3. mark balls near the current hull as the current shell;
4. remove that shell and repeat on the remaining balls;
5. expose `perimeter shell 1` / `interior shell 2+` in the report and v1 UI.

Relevant config:

```yaml
cluster_optimization:
  shell_classification_enabled: true
  shell_classification_min_size: 5
  shell_perimeter_distance_factor: 0.42
```

On `DSC00540`, this produces one 15-red cluster with 12 perimeter balls and
3 interior balls. That is the expected first split for the starting triangle:
outside balls have at least one cloth-facing side, while inside balls have
little or no cloth boundary to sample.

The output fields are diagnostic:

- `cluster_shell_status`;
- `cluster_shell`;
- `cluster_role`;
- `cluster_perimeter_distance_mm`;
- `cluster_neighbor_degree`.

Current limitation: shell role is not yet used to change final fitting. The
next step is to let perimeter balls provide stronger neighbor/ownership
constraints before attempting interior-ball estimates.

### 6. Same-color cluster shape prior

v1.4.6 adds a conservative shape-prior diagnostic for dense clusters such as
`DSC00540`.

The failed strategy was to let the perimeter/interior traversal influence the
fit directly. That remains disabled for final fitting because the current
outside-in path does not yet provide enough evidence to decide which interior
arcs are trustworthy.

The active v1.4.6 rule is narrower and easier to audit:

1. find adjacent-ball components;
2. group cluster members by color/label;
3. for same-label groups with enough members, compute a robust consensus
   ellipse size and angle from the available per-ball observed ellipses;
4. compare every ball in that same-color group against the consensus;
5. flag a ball if its ellipse axes or orientation are physically inconsistent
   with the rest of the cluster.

This directly targets the `DSC00540` failure mode where red balls #9 and #12
grab neighboring reflections/edges, producing ellipses roughly 1.3x too large
and rotated away from the cluster consensus. In a real rack, neighboring balls
can occlude arcs, but they cannot make one red ball physically project much
larger than adjacent red balls at the same table region.

Relevant config:

```yaml
cluster_optimization:
  perimeter_weighted_fit_enabled: false
  shape_prior_enabled: true
  shape_prior_min_cluster_size: 5
  shape_prior_min_label_count: 5
  shape_prior_min_consensus_members: 4
  shape_prior_major_scale_limit: 1.22
  shape_prior_minor_scale_limit: 1.22
  shape_prior_angle_delta_deg: 12.0
```

Exported per-ball diagnostics include:

- `cluster_shape_prior`;
- `cluster_shape_outlier`;
- `cluster_shape_reasons`;
- `cluster_shape_consensus_major_axis_px`;
- `cluster_shape_consensus_minor_axis_px`;
- `cluster_shape_consensus_angle_deg`;
- `cluster_shape_major_scale`;
- `cluster_shape_minor_scale`;
- `cluster_shape_angle_delta_deg`.

The review UI shows these diagnostics under `Cluster shape`, and the confidence
logic now penalizes:

- `cluster_shape_outlier`;
- `cluster_ellipse_size_outlier`;
- `cluster_ellipse_angle_outlier`;
- `neighbor_ellipse_ownership_conflict`.

Current limitation: v1.4.6 only flags and scores the inconsistency. It does not
yet repair the fit. The next fitting strategy should be a constrained ellipse
fit: use the same-color cluster consensus for axes/angle and solve mainly for
the center from the visible arcs.

### 7. Per-point rejection reasons and add-back diagnostics

v1.4.7 makes rejected boundary points auditable at point level. A rejected
point can have multiple reasons, but the UI assigns one primary color:

| Primary reason | UI color | Meaning |
|---|---|---|
| `angular_segment_endpoint` | orange | likely bad point at the end of a broken arc |
| `local_radius_spike` | red | local radial jump compared with nearby samples |
| `neighbor_ellipse_overlap` | purple | point lies inside a nearby detected ball ellipse |
| `ellipse_residual_outlier` | blue | point is far from the temporary robust ellipse |
| `other_rejected` / `unknown_rejected` | pink | rejected without a more specific reason |

The report exports:

- `rejected_point_reasons`;
- `rejected_reason_counts`.

v1.4.7 also exports diagnostic add-back fits for each evidence view. These
temporarily add selected rejected categories back and refit the ellipse:

- accepted points only;
- add arc endpoints;
- add local radius spikes;
- add neighbor-overlap rejects;
- add ellipse-residual rejects;
- add endpoints + local radius spikes;
- add all rejected points.

The UI reports the best shape-match scenario against the same-color cluster
shape prior. This is diagnostic only. It is not yet used as the final estimate,
because the add-back fit can still be wrong when the visible evidence is mostly
from neighboring highlights or occluded arcs.

v1.4.8 introduced a finer diagnostic requested from DSC00540 balls #9 and #12:
`consensus_reject_refit`. Instead of adding a whole rejection category back,
it grouped rejected points by local angular/spatial continuity, tried small
group combinations, and kept the combination whose refitted ellipse best
matched the same-color cluster consensus size and angle.

v1.4.9 promotes the useful part of that idea into the final estimator. The
active promoted path is now `arc_combination_refit`:

1. start from all raw radial boundary samples, not only the first accepted
   white dots;
2. split the raw samples into angular/spatial arc groups;
3. try useful group combinations;
4. fit two candidate families:
   - free ellipse from the selected arc points;
   - shared-shape fixed ellipse, where axes/angle come from the same-color
     cluster consensus and only the center is solved from the selected arcs;
5. score candidates by cluster-shape agreement, residual, point coverage,
   multi-arc support, center shift, and boundary ownership;
6. promote only if the candidate passes the conservative promotion gate.

When promoted, the selected arc clusters become the visible white dots, the
remaining raw samples become red rejected dots, and the cream outline becomes
the promoted final ellipse. The UI does not need separate add-back colors for
normal review. The richer rejection reasons and top candidate lists remain in
JSON/debug data.

Current DSC00540 behavior:

- red #9 promotes to a `cluster_shape_fixed` arc-combination fit;
- red #12 promotes to a `cluster_shape_fixed` arc-combination fit;
- red #14 stays with its baseline fit because the candidate improvement is too
  small to justify changing the final estimate.

This is still a bridge toward the generic cluster graph optimizer, not the full
joint solver. It can repair obvious per-ball cluster mistakes, but it does not
yet optimize every node, contact edge, missing hypothesis, and duplicate
hypothesis jointly.

### Safeguards

- Rejected points remain visible as red dots.
- Rejected points do not affect the cream ellipse.
- If filtering would leave too few points, the filter falls back to the safer
  unfiltered set.
- The report exports raw/accepted/rejected counts, per-point rejection reasons,
  and add-back fit scenarios.

## Diagnostic evidence maps

For every ball, the system can write crop-aligned map images into:

```text
outputs/reports_v1_global_cloth/<image_stem>/evidence_maps/
```

The v1 review UI exposes them through the selected-ball evidence background
selector.

Implemented maps:

| Map | Meaning | Current use |
|---|---|---|
| `gray_edge` | Sobel luminance edge strength | diagnostic background + boundary variant |
| `lab_delta_e` | Lab color distance from active cloth reference | diagnostic background + boundary variant |
| `chroma_difference` | Lab a*/b* chroma distance from active cloth reference | final by policy for green/blue/brown; diagnostic for other balls |
| `ball_vs_cloth_probability` | learned ball-vs-cloth probability using active cloth reference | default final-position map; diagnostic when not selected |
| `physical_projection_band` | Gaussian band around the projected sphere outline | diagnostic prior background + boundary variant; not image-only evidence |
| `combined_boundary_score` | weighted edge/color/probability/physical map | diagnostic/scoring background + boundary variant |

### Evidence-map-specific boundary variants

v1.3.4 adds a separate boundary sampler for every evidence map. This is the
feature behind the UI behavior where the white dots and cream ellipse change
when the evidence-background row changes.

The per-map variant is stored under:

```text
ball.evidence.diagnostics.evidence_maps.boundary_variants.<map_key>
```

Each variant contains:

- accepted boundary points;
- rejected outlier points;
- the ellipse fitted from accepted points;
- filter statistics;
- the sampler type used by that map.

Sampler types:

- `peak_response`: choose the strongest response along each radial line. This
  is used for direct edge-like maps such as `gray_edge` and
  `physical_projection_band`.
- `outward_drop`: choose the strongest inward-to-outward drop along each radial
  line. This is used for probability/color maps where the interior is bright
  and the exterior cloth is darker.

v1.3.6 limitation: all variants remain visible, but only the configured final
variant is promoted to final `source_px` and table coordinates. Non-selected
maps do not affect final position. The current promoted rule is:

- `ball_vs_cloth_probability` for most balls;
- `chroma_difference` for green, blue, and brown balls.

The next benchmark step is to measure whether this rule improves physical
validation without increasing missed balls, duplicate detections, or bad
cluster/cushion fits.

### Cloth/color model

v1.3.8 has two color references:

1. global cloth reference, active by default;
2. local annulus reference, retained as a diagnostic/fallback.

The global reference is estimated once per image:

1. start from the table polygon;
2. erode inward so cushion/wood/pocket pixels are avoided;
3. exclude neighborhoods around detected balls;
4. ignore very dark pixels and strong highlights;
5. use the median remaining cloth Lab value.

For each ball crop, the ball interior is still sampled from the inner disk. The
maps then compare this ball sample against the active cloth reference.

Since v1.5.5, global-cloth diagnostic maps are computed over the full source
image first, then cropped to the selected ball ROI for display and sampling.
This matters because the previous implementation used the global cloth Lab
value but still normalized Lab Delta-E, chroma, and grayscale edge inside each
ball ROI. That made different balls appear to have different "background"
scales even when they shared the same cloth reference.

The current rule is:

```text
global cloth reference -> full-table Lab/chroma/edge maps -> ROI crop for each ball
```

not:

```text
global cloth reference -> per-ball ROI Lab/chroma/edge normalization
```

The ball-vs-cloth probability map remains partly ball-specific because it uses
the selected ball's sampled interior Lab. However, it uses the full-table
Lab/chroma maps as supporting inputs, so the underlying color-map scale is now
stable across all balls in one image.

The review UI shows:

- active cloth reference mode;
- evidence map source and normalization scope;
- ball Lab;
- active cloth Lab;
- active Lab/chroma separation;
- local annulus cloth Lab and separation;
- global cloth Lab;
- sample counts;
- active parameter knobs.

This is implemented in `compute_full_table_evidence_maps()`,
`compute_ball_evidence_maps()`, and `estimate_global_cloth_reference()`.

Since v1.5.6, the review UI also has display-only brightness/contrast/invert
controls for the selected evidence background. These controls are deliberately
not part of the recognition algorithm. They answer one narrow question:

> Is this evidence map actually bad, or is the grayscale display range making it
> look overexposed?

Changing these sliders does not recalculate white boundary points, red rejected
points, fitted ellipses, confidence, or exported table state. Real algorithm
tuning will need a backend recomputation endpoint that accepts evidence
parameters, regenerates the selected map/points/ellipse, and stores that run as a
separate experiment rather than mutating the immutable report JSON.

Current limitation: global cloth fixes contaminated local-annulus references,
but it does not solve boundary ownership in clusters. In a tight rack, a map can
now correctly separate red ball from green cloth and still sample the contour of
neighboring balls or narrow shadow gaps.

### Class-specific weighting

The combined map uses different weights by ball class.

For green/blue balls:

```text
edge        0.16
Lab         0.22
chroma      0.24
probability 0.28
physical    0.10
```

For other balls:

```text
edge        0.34
Lab         0.18
chroma      0.14
probability 0.24
physical    0.10
```

Reason: green/blue balls often have weak luminance edges on green cloth, so
Lab/chroma and local ball-vs-cloth probability are more useful than raw edge
strength alone.

## Physical projection as a weak prior

The blue dashed outline is the predicted projection of a known-radius snooker
ball under the active camera model.

Today the camera model is still approximate, so the physical projection is used
as:

- a visual diagnostic;
- a weak score term in `combined_boundary_score`;
- a residual target for physical optimization.

It is not used to recover extra boundary points anymore.

The removed experiment was:

```text
search near blue projection band -> create cyan recovered points -> add them to scoring
```

That made some cases worse because the approximate camera model can place the
band in the wrong location. The current safer rule is:

```text
show the physical band as a map;
score agreement;
do not manufacture new boundary points from it.
```

## Robust ellipse fitting status

Currently implemented:

- endpoint trimming;
- local radius outlier rejection;
- ellipse-residual outlier rejection;
- final OpenCV ellipse fit from accepted white points;
- rejected red points remain visible.

Not yet implemented:

- true RANSAC ellipse fitting;
- IRLS/Huber/Tukey robust ellipse fitting;
- arc-coverage scoring;
- promoting evidence-map-specific boundary variants into the production final
  estimate.

This is the next high-value recognition step after diagnostic-map review.

## Which proposed strategies are implemented?

| Strategy | Status |
|---|---|
| Diagnostic maps: gray edge, Lab Delta-E, chroma, ball-vs-cloth probability, physical projection band | Implemented and visualized as evidence-map backgrounds |
| Per-map boundary dots and cream ellipse | Implemented in v1.3.4; v1.3.6 promotes one configured variant to final source/table position |
| Global cloth reference | Implemented in v1.3.8 and active by default for Lab Delta-E, chroma, and ball-vs-cloth probability |
| Full-table global evidence-map normalization | Implemented in v1.5.5 for global-cloth Lab/chroma/edge maps; per-ball views are ROI crops of those full-table maps |
| Local color model | Retained as diagnostic/fallback; no longer the default cloth reference |
| Class-specific weighting | Implemented in the combined evidence map |
| Blue physical projection as a weak prior | Implemented for diagnostic/scoring/optimization; recovered-point generation removed |
| Robust ellipse fitting | Partially implemented via outlier filtering and per-map ellipse variants; full RANSAC/IRLS/arc coverage still pending |

## How to see the maps

Regenerate reports:

```powershell
python tools/generate_dataset_reports.py --glob "Media/**/*.JPG" --output outputs/reports_v1_global_cloth
```

Start the review UI:

```powershell
python tools/review_reports.py --reports outputs/reports_v1_global_cloth --host 127.0.0.1 --port 8771
```

Open:

```text
http://127.0.0.1:8771/
```

Select a ball. In the selected-ball panel, use the evidence-background selector to
switch between:

- source image;
- grayscale edge;
- Lab Delta-E;
- chroma difference;
- ball-vs-cloth probability;
- physical projection band;
- combined boundary score.

When you switch the selector, three things should change together:

1. the evidence background image;
2. the white/red sampled boundary points;
3. the cream observed ellipse fitted from those points.

The correct human review question is:

```text
Does the map highlight the real ball boundary, or is it highlighting cloth,
shadows, highlights, cushion, or neighbors?
```

The matrix `Score` column is a diagnostic evidence-view score. It is computed
per row from accepted points, rejected points, ellipse availability, and
agreement with the current physical sphere outline:

```text
view score =
  45% physical-outline residual
+ 30% accepted boundary point count
+ 20% inlier ratio
+  5% ellipse availability
```

It is not ground truth and it does not by itself validate table coordinates.
It is there to make side-by-side evidence-map comparison less subjective.

## DSC00542 cluster-review conclusions

Manual review of DSC00542 gives a concrete rule split:

- loose or mostly isolated balls usually work well with `source image` and
  `ball_vs_cloth_probability`;
- green, blue, brown, and some shadow-biased balls often need
  `chroma_difference`;
- tightly touching red clusters are a different problem: color/probability maps
  can become misleading because the local crop is no longer a clean
  ball-versus-cloth experiment.

Observed good-fit evidence for DSC00542:

| Ball ids | Best-looking evidence views |
|---|---|
| 1 | ball-vs-cloth probability, chroma difference |
| 2 | ball-vs-cloth probability, chroma difference, source image |
| 3 | ball-vs-cloth probability, source image, Lab Delta-E |
| 4 | source image, Lab Delta-E, chroma difference, ball-vs-cloth probability |
| 5 | source image, chroma difference, ball-vs-cloth probability |
| 6, 8, 10, 13, 15 | source image, ball-vs-cloth probability |
| 16 | chroma difference only |
| 17 | chroma difference only; other views confuse shadow with contour |
| 18 | source image, chroma difference, ball-vs-cloth probability |
| 19 | source image, chroma difference |
| 20 | chroma difference |
| 21 | source image, chroma difference, ball-vs-cloth probability |
| 22 | source image, chroma difference |

Hard cluster cases:

- Ball 7: source image is the only useful evidence view. Other maps amplify
  neighbor edges and shadow gaps. The source view still produces loose dots
  because current filtering does not yet know which neighboring-ball edge owns
  each point.
- Ball 9: no current evidence map is clearly useful. `ball_vs_cloth_probability`
  is too binary, and `Lab Delta-E` / `chroma_difference` can appear inverted
  relative to easier balls.
- Balls 11, 12, and 14 show similar cluster behavior to ball 9.

Important interpretation:

```text
The per-view score is useful for loose balls, but it can be fooled in clusters.
It currently rewards point count, inlier ratio, and physical-outline residual.
It does not yet know whether those points belong to the selected ball or to a
neighboring ball, shadow, highlight, or narrow cloth gap.
```

### Why DSC00542 #7/#9 looked different from #8 before v1.3.8

The code did not apply special per-ball settings for red ball #7/#9 versus red
ball #8. They used the same red-ball evidence-map policy.

The difference came from local context:

- #8 has clean ball/cloth contrast and enough visible outward boundary;
- #7 and #9 sit inside the rack/cluster, so the annulus used to estimate cloth
  often contains neighboring red balls, shadows, and small cloth gaps;
- the inner ball sample can also be contaminated by specular highlights and
  occlusion.

That breaks the assumption behind `ball_vs_cloth_probability`:

```text
local inner disk  = this ball
local outer ring  = cloth
```

In clusters, the outer ring is often not cloth. The old local annulus therefore
estimated a red-neighbor/shadow color as "cloth". For DSC00542 #9, the local
cloth Lab was almost red-ball-colored:

```text
#8 local cloth Lab:  138,  83, 154
#8 ball Lab:         136, 196, 187
#8 local separation: 117.7

#9 local cloth Lab:  129, 192, 183
#9 ball Lab:         129, 196, 187
#9 local separation:   5.7
```

That is the smoking gun: #9's local annulus was not cloth. `Lab Delta-E`,
`chroma_difference`, and `ball_vs_cloth_probability` were all being built from
a bad reference.

With the v1.3.8 global cloth reference on DSC00542:

```text
global cloth Lab:    146,  78, 153
#8 active separation: 123.2
#9 active separation: 124.0
```

This fixes the color-reference problem. It does not fully fix the cluster
ownership problem: #9 can still collect valid-looking points from neighboring
balls or narrow gaps. The final-position policy therefore includes an ellipse
quality guard that rejects implausible map fits, such as a selected ellipse
with axis ratio above the configured limit.

### Why loose dots survive

The active filter removes:

- angular segment endpoints;
- local radius spikes;
- high ellipse-residual points.

It does not yet remove points based on:

- connected arc ownership;
- expected neighboring-ball silhouettes;
- whether a short arc belongs to the selected ball or an adjacent ball;
- whether the local color model is contaminated.

Therefore small arcs from neighboring balls, specular highlights, and shadows
can survive when they are locally consistent enough to pass radius/ellipse
checks.

### Engineering conclusion

The next confidence improvement should not be another generic evidence map.
The next step should be cluster-aware evidence ownership:

1. detect cluster/touching context from nearest-neighbor distances and overlap;
2. estimate neighbor exclusion zones in the selected crop;
3. split accepted points into connected angular arcs;
4. reject or down-weight short isolated arcs and arcs inside neighbor zones;
5. score local color-model health before trusting `B`, `C`, or `L`;
6. keep physical projection as a weak prior, not a selector by itself.

Until that exists, the safe final-policy rule is:

```text
loose balls:
  default to ball-vs-cloth probability;
  use chroma difference for green/blue/brown and visually confirmed exceptions.

cluster/touching balls:
  prefer source image or chroma only when ownership is clear;
  do not let physical projection band or per-view score alone raise confidence;
  mark low confidence when local color model is contaminated.
```

## Priority order for next experiments

1. Add cluster-aware diagnostics before changing final coordinates:
   - cluster/touching context flag;
   - local color-model health;
   - neighbor-overlap rejection;
   - connected arc count and largest visible arc span;
   - short/noisy arc rejection shown as a separate visible layer.
2. Benchmark the promoted final-position map policy:
   - default `ball_vs_cloth_probability`;
   - `chroma_difference` for green, blue, brown;
   - all other maps diagnostic only.
3. Keep evidence maps visible and inspect them on hard balls:
   - DSC00524 #12 blue;
   - DSC00524 #21 green;
   - DSC00525 #8 green;
   - DSC00542 edge/cluster examples.
4. Compare map-driven boundary samplers for green/blue:
   - inspect `ball_vs_cloth_probability`, `chroma_difference`, and
     `combined_boundary_score`;
   - reject maps that improve one ball but leak into highlights, cloth,
     cushions, or neighbors on other balls;
   - keep old white edge points and new map-driven points as separate visible
     layers during the experiment.
5. Add robust ellipse fitting:
   - RANSAC or IRLS;
   - explicit inlier/outlier sets;
   - arc coverage score.
6. Add occlusion-aware scoring for clusters:
   - do not penalize hidden arcs;
   - do penalize impossible overlap.
7. Promote the physical model only after ChArUco calibration:
   - real intrinsics;
   - lens distortion;
   - camera pose;
   - reprojection error gates.

## Benchmark gates

Every new strategy should be compared against fixed sample reports:

```text
baseline/no filter
geometry filter
diagnostic maps only
map-driven green/blue boundary sampler
robust ellipse fitter
calibrated physical model
```

Metrics:

- count accuracy per image;
- mean confidence;
- high/medium/low confidence counts;
- physical residual against projected sphere;
- touching-ball distance error where metadata exists;
- cushion-touch error where metadata exists;
- number of cases that became worse;
- manual review examples from known hard images.

Important rule:

```text
Do not raise confidence just because an overlay looks smooth.
Confidence should rise only when image evidence, physical model, and physical
validation agree.
```

## Current v1.3.6 benchmark snapshot

Generated reports:

```text
outputs/reports_v1_evidence_maps
```

Benchmark output:

```text
outputs/model_scoring_benchmark_v1_3/model_scoring_benchmark.json
outputs/model_scoring_benchmark_v1_3/model_scoring_benchmark.csv
```

Latest local benchmark:

```text
rows:                                  418
images with detected balls:             19
legacy mean confidence:              47.9%
physics-first mean confidence:        59.9%
physical + observed ellipse mean:     78.0%
displayed mean confidence:            79.2%
confidence improved by >=10 points:    297
confidence reduced by >=10 points:       0
mean accepted/rejected boundary pts: 117.5 / 8.4
evidence-map statuses:                 computed for 418/418
mean evidence-map assets per ball:       6.0
physical optimization optimized:         64
physical optimization no better:        354
joint cluster optimized:                 97
duplicate-warning false-positive proxy:   0
final-position map policy violations:      0
evidence-map boundary variants:     computed for all six maps on detected balls
```

Important interpretation: the numbers below must be refreshed after every
policy change. v1.3.6 promotes one map-specific boundary variant into final
coordinates, so the benchmark must now be read as a position-policy benchmark,
not just a diagnostic-map benchmark.

## Current v1.3.8 global-cloth benchmark snapshot

Generated reports:

```text
outputs/reports_v1_global_cloth
```

Cloth-reference analysis:

```text
outputs/cloth_reference_analysis_global/cloth_reference_by_ball.csv
outputs/cloth_reference_analysis_global/cloth_reference_summary.json
```

Latest local benchmark:

```text
rows:                             418
images with detected balls:        19
mean local-vs-global Delta-E:    27.3
max local-vs-global Delta-E:    124.4
active low-contrast rows:           0
local-annulus low-contrast rows:   11
```

Interpretation:

- global cloth removes the worst local-annulus contamination cases;
- the old local model thought several red cluster balls had almost no
  ball-vs-cloth separation because the annulus sampled neighboring red balls;
- the active B/C/L maps now use a stable table-cloth reference;
- cluster ownership remains unsolved, so final map fits still need plausibility
  guards and later neighbor-aware arc ownership.

## Summary for ChatGPT Pro

Use this summary when asking for external strategy ideas:

```text
Project: SnookerHelp, classical OpenCV snooker-ball recognition.

Current active pipeline:
- rough ball candidates still come from a warped cloth-plane view;
- final evidence is measured in the original source image crop;
- on the source image, white dots are accepted radial source edge-boundary pixels;
- on an evidence-map background, white dots are sampled from that selected map;
- red dots are rejected outliers for the selected source/map view;
- cream dashed ellipse is fitted from the selected accepted white dots only;
- blue dashed outline is the approximate projected sphere from camera/table/ball geometry;
- no cyan recovered projection-band points are active anymore.
- final source/table position uses one configured image-evidence map:
  ball-vs-cloth probability by default, chroma difference for green/blue/brown;
- all other maps remain visible diagnostics and do not affect final position.

Current evidence maps:
- grayscale edge;
- Lab Delta-E from active cloth reference;
- chroma-only difference from active cloth reference;
- ball-vs-cloth probability from active cloth reference;
- physical projection band;
- combined boundary score.

Color model:
- active cloth reference defaults to one global table-cloth Lab estimate per
  image;
- the old local annulus model is still computed as a contamination diagnostic;
- ball color is estimated from the inner disk;
- highlights and deep shadows are ignored;
- green/blue balls weight chroma/probability more strongly than grayscale edge.

Current limitation:
- only one manually reviewed map policy is currently promoted;
- evidence-map-specific boundary variants remain visible so the policy can be
  challenged per image;
- robust ellipse fitting is still only partial outlier filtering, not full RANSAC/IRLS;
- neighbor-ellipse ownership rejects obvious adjacent-ball contamination, but
  it is not yet a full cluster solver that processes the rack from outer
  visible edges inward;
- camera model is approximate until ChArUco calibration.

Need advice:
Design a classical OpenCV strategy for low-contrast green/blue balls on green
cloth. Candidate directions:
1. local ball-vs-cloth color model in Lab/chroma/normalized RGB;
2. class-specific radial boundary sampling on probability maps;
3. robust ellipse fitting with RANSAC/IRLS and arc coverage;
4. occlusion-aware scoring for clusters/touching balls;
5. using projected sphere outline as weak prior only;
6. benchmark gates that avoid false confidence.

Constraint:
No YOLO/neural network yet. Keep evidence auditable in UI:
accepted points, rejected points, maps, observed ellipse, physical projection,
final estimate, confidence.
```
