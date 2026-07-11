# 06 — Cluster Graph Solver

Dense clusters are the main reason for v2.

The cluster graph solver must solve a group of balls jointly. It must not rely on a sequential traversal as the primary algorithm.

## Cluster nodes

Each node represents:

- a detected ball hypothesis;
- an optional missing-ball hypothesis;
- an optional duplicate/suppressed hypothesis.

Node fields:

- initial center;
- optimized center;
- label;
- projected shape;
- evidence arcs;
- status;
- confidence.

## Cluster edges

Edges represent soft physical relationships:

- possible contact;
- near contact;
- impossible overlap;
- duplicate;
- non-contact;
- unknown.

Edges store:

- measured distance;
- expected distance;
- weight;
- reason list.

## Boundary ownership

Boundary arcs must be assigned to the most plausible owner.

Ownership uses:

- proximity to projected shape;
- arc angle;
- neighbor overlap;
- evidence strength;
- shared-shape consistency;
- contact geometry.

An arc can be:

- owned by one ball;
- shared/ambiguous;
- rejected as highlight/noise;
- assigned to a missing hypothesis.

## Shared shape prior

Nearby balls should have similar projected ellipse size and orientation.

Use robust local consensus:

- median major axis;
- median minor axis;
- median angle with 180-degree wrap;
- spread;
- supporting-neighbor count.

This prior should correct suspicious oversized/rotated fits when enough reliable neighbors exist.

## Duplicate hypotheses

Two hypotheses should not both be accepted when they represent one physical ball.

Duplicate evidence:

- centers too close;
- outlines overlap heavily;
- same color;
- shared boundary arcs;
- graph cannot satisfy both without overlap.

Output must mark:

- accepted node;
- suppressed duplicate;
- unresolved duplicate.

## Missing hypotheses

Missing-ball hypotheses are created from:

- unexplained cluster mask area;
- contact graph gaps;
- expected ball inventory;
- manual review feedback;
- physical validation failures.

Missing hypotheses are not automatically accepted. They are marked as unresolved unless evidence is strong.

## Joint optimization

The optimizer scores candidate cluster solutions using:

- image evidence support;
- boundary ownership;
- equal radius;
- shared local projected shape;
- contact distance;
- no-overlap;
- cluster mask support;
- duplicate penalty;
- missing penalty;
- camera-model prior.

## Abstention

The cluster solver must be allowed to refuse promotion.

If all candidate cluster solutions are worse than the current per-ball estimates, or if the solver cannot explain the evidence without impossible overlaps, the cluster solution remains `diagnostic_only` and affected balls are marked `needs_review`.

## Promotion rules

A cluster solution can promote final ball estimates only if:

- it improves physical consistency;
- it does not worsen strong image evidence;
- duplicate/missing decisions are explicit;
- no hard overlap violation remains;
- confidence reasons are exported.

Required promotion gates:

- no hard overlap violations;
- duplicate conflicts resolved or marked unresolved;
- cluster energy improves over baseline;
- no high-confidence loose/perimeter ball is degraded;
- shared-shape residual is within configured limit;
- cluster mask support improves or remains acceptable;
- movement from rough hypotheses is explainable;
- missing hypotheses are not counted as accepted balls unless strict gates pass.

## Optional arc-combination candidate generation

Arc-combination fitting is a candidate-generation method, not a top-level truth source.

Arc-combination candidates remain diagnostic until cluster-level and physical-consistency promotion gates pass.

## Rack/triangle handling

The intact 15-red triangle is one case of a generic cluster.

The solver may use rack-like priors, but the primary implementation must support arbitrary clusters from real games:

- intact triangle;
- partly broken triangle;
- two-ball contacts;
- random multi-ball clusters;
- near-cushion clusters;
- mixed-color clusters.
