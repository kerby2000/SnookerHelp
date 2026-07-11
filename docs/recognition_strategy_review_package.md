# SnookerHelp recognition strategy review package

This is a compact package to paste into ChatGPT Pro or another reviewer when asking for algorithm strategy advice.

## Problem

We need accurate snooker ball centers from one overhead camera. Loose balls are now mostly good. The hard case is a tight red-ball cluster/rack, especially `DSC00540`, where inner balls have little or no visible green cloth boundary. The current per-ball image model can accidentally fit:

- lamp reflections on neighboring balls;
- boundary arcs from neighboring balls;
- partial arcs that produce an ellipse about 1.5x too large;
- an ellipse angle that disagrees with the local cluster.

Bad fitted ellipses should not become constraints for later balls, otherwise errors propagate through the cluster.

## Current pipeline

1. Classical detector finds rough balls.
2. Rough positions are mapped back to the source image.
3. Source-image crops are built for every ball.
4. Several evidence maps are computed per crop:
   - source image boundary;
   - grayscale edge;
   - Lab Delta-E from cloth;
   - chroma difference;
   - ball-vs-cloth probability;
   - physical projection band;
   - combined boundary score.
5. For each evidence map, boundary points are sampled.
6. Points are filtered into:
   - accepted boundary points, shown as white dots;
   - rejected boundary points, shown as red dots.
7. An observed ellipse is fitted from accepted points.
8. A final source pixel center is selected from the chosen image evidence.
9. The approximate physical model projects the final source pixel to table coordinates.

Current UI rule: users should see only white accepted dots, red rejected dots, and the cream dashed fitted ellipse. Extra diagnostic colors should not be exposed as primary concepts.

## Current default evidence selection

Based on manual review:

- Most loose balls: ball-vs-cloth probability is best.
- Green/blue/brown balls: chroma difference is often better.
- Clustered red balls: source image or ball-vs-cloth can work for perimeter balls, but inner balls are unstable.

Other maps are currently diagnostic unless selected by policy.

## Current filtering

Boundary points can be rejected because of:

- angular segment endpoint;
- local radius spike;
- ellipse residual outlier;
- neighbor ellipse overlap;
- other/unknown reject.

The user no longer wants these reason categories shown as separate colors. They can stay in text/debug data, but visually all rejected points should be red.

## Current cluster shape prior

For adjacent same-color balls, the system estimates a local consensus ellipse shape:

- consensus major axis;
- consensus minor axis;
- consensus ellipse angle.

Hypothesis: in a tight cluster/rack, neighboring balls should have nearly the same projected ellipse size and orientation. If one ball has a much larger ellipse or a very different angle, that fit is likely wrong.

Example observation on `DSC00540`:

- good perimeter balls have roughly similar axes and angle near `0°/180°`;
- problematic inner fits were around 1.5x too large and at angles like `18°` or `26°`.

## Current arc-cluster combination path

For cluster shape outliers, the system now has a reusable refit step:

1. Take all raw boundary samples before final filtering.
2. Split them into spatial/angular arc clusters.
3. Try combinations of these clusters.
   - For `n` clusters, theoretical combinations = `2^n - 1`.
   - Tiny combinations are skipped.
4. Fit an ellipse to each combination.
5. Rank each fitted ellipse using:
   - agreement with cluster consensus shape;
   - ellipse residual/RMS;
   - point coverage;
   - number of arcs used.
6. Promote the best combination only if it passes a strict gate.

Promotion gate currently requires:

- refit status is improved;
- ellipse RMS <= 2.5 px;
- point fraction >= 0.18;
- at least 2 arc clusters;
- at least 18 points;
- shape-score improvement >= 8;
- best combination is not a cluster-shape outlier.

The shape gate has two paths:

- high confidence: cluster-shape score >= 82;
- moderate confidence: cluster-shape score >= 72, ellipse RMS <= 1.6 px, at least 36 points, at least 2 arc clusters, shape-score improvement >= 20, and the best combination is not a cluster-shape outlier.

If promoted, the promoted arc-combination ellipse becomes the final image model and the JSON/table coordinates are regenerated from it. If not promoted, it remains diagnostics only.

Current behavior after the gate change:

- `DSC00540` ball #9 now promotes from a shared-shape arc combination:
  approximately `94.5 x 78.2 px @ 2.1 deg`.
- `DSC00540` ball #12 still gets an arc-combination candidate, but its final
  center is now selected by the graph-joint promotion path described below.

## Current failure mode

For some rack balls, the raw point clusters contain both useful hidden/occluded boundary arcs and wrong lamp-reflection/neighbor arcs. A human can often see that a subset of clusters gives the right ellipse, but the algorithm may rank the wrong subset unless the shape prior and point grouping are strong enough.

Key examples:

- A “good” cluster-combination should use multiple real ball-boundary arcs and exclude lamp-reflection arcs.
- A “bad” combination may have a low RMS but still be physically impossible because the ellipse is too large or rotated relative to neighboring balls.

## Generic contact-graph strategy now partially implemented

The strongest proposal so far is to stop treating the triangle/rack as a
special template and instead solve every dense touching group as a generic
physical contact graph:

- nodes are detected balls;
- edges are possible contacts, near contacts, or occlusion relations;
- boundary ownership assigns image edge samples to the most plausible ball;
- perimeter balls have stronger image evidence;
- interior balls use weaker image evidence plus contact/shape constraints;
- same local camera region should imply similar projected ellipse size and orientation;
- cluster-level results should be promoted only when both image score and physical score improve.

The first promoted slice is implemented:

- `cluster_graph.py` builds arbitrary contact/near-contact graph components;
- `cluster_optimizer.py` runs the existing equal-radius joint cluster optimizer
  and attaches graph/boundary ownership context;
- `cluster_promotion.py` is the explicit estimator-level gate;
- final source centers may now be replaced by `cluster_graph_joint_center`
  for weak interior balls in clusters of 4+ when contact-distance RMS improves
  and movement is bounded.

This is not yet a complete global cluster solver. It does not yet create
missing-ball hypotheses, suppress duplicate nodes by graph optimization, or
solve a single whole-cluster silhouette. It is the first safe promotion path
for the physical contact graph.

## What we need advice on

Please evaluate the strategy and propose a robust next algorithm for tight snooker-ball clusters without using YOLO or learned object detectors.

Questions:

1. Should cluster balls be solved jointly instead of per-ball?
2. Should we fit a shared-shape ellipse model for all balls in a local cluster?
3. Should we impose rack/touching constraints, e.g. center-to-center distance near one ball diameter?
4. How should raw boundary point clusters be selected or rejected?
5. How can we avoid using lamp reflections as contour evidence?
6. How can we prevent bad early fits from propagating into later balls?
7. Should perimeter balls constrain interior balls, or should all cluster balls be optimized simultaneously?
8. Is the right model:
   - per-ball ellipse + consensus shape prior;
   - global cluster graph optimization;
   - known triangular rack lattice fit;
   - contour segmentation of the whole cluster;
   - something else?

## Constraints

- Use the existing single camera and existing photos for now.
- ChArUco calibration will come later, but do not depend on it for this step.
- Do not add YOLO, pool physics simulation, projector code, or Basler capture yet.
- Keep the UI explainable: white accepted dots, red rejected dots, fitted ellipse, final center, confidence.

## Desired output

We need a practical algorithm that can be implemented incrementally and benchmarked against current reports. The ideal answer should specify:

- data structures;
- fitting order or joint optimization method;
- scoring function;
- fallback logic;
- how to validate improvement on `DSC00540`;
- when a result should be promoted to final estimator versus left as diagnostics.
