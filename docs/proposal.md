A real snooker game can produce:

intact 15-red triangle,
one or more reds knocked out,
deformed rack,
touching pairs,
curved mini-clusters,
mixed-color clusters,
partial contacts,
balls almost touching but not actually touching,
occlusions where one ball hides another ball’s visible boundary,
clusters where only perimeter balls have reliable contours.

So the correct general solution is not:

fit known 15-ball triangle

It is:

fit an arbitrary physical contact graph of equal-radius balls

The rack triangle then becomes just one possible graph, not the algorithm.

Your own review package describes the current weakness well: loose balls are mostly good, but dense clusters like DSC00540 fail because inner balls have little or no cloth boundary, and per-ball image models can fit lamp reflections, neighboring-ball arcs, or ellipses that are too large/rotated; once a bad ellipse is promoted, it can propagate errors through the cluster.

Better general strategy
Use a cluster contact-graph optimizer

Think of a cluster as:

nodes = balls
edges = possible contacts / near contacts / occlusions
observations = rough centers, evidence maps, boundary arcs, color labels
constraints = equal radius, non-overlap, contact distances, inventory, local projection shape

So instead of solving each ball independently, solve the whole local cluster as a graph.

A cluster can be:

2 balls touching
3 balls in a line
5 balls in an irregular blob
13 reds from a broken rack
15 intact reds
mixed red + color cluster

Same algorithm.

Core model

For each cluster, optimize:

center_i = source image center of ball i
shape_i  = projected ball outline, usually constrained by local shared shape
label_i  = red / color class, usually already known
active_i = whether hypothesis is real or rejected

The objective should combine:

image evidence
+ physical ball constraints
+ contact graph constraints
+ cluster mask support
+ boundary visibility/ownership
+ color/inventory constraints

In formula-like form:

Loss =
  Σ unary_image_loss(ball_i)
+ Σ pairwise_physical_loss(ball_i, ball_j)
+ Σ cluster_mask_union_loss(cluster)
+ Σ visible_boundary_loss(cluster)
+ Σ label_inventory_loss(cluster)
+ Σ movement_prior(ball_i)

This is generic. A triangular rack is just a case where many pairwise distances are close to one ball diameter.

What replaces “rack lattice”?
Contact graph

Build a graph from approximate ball centers.

Edges represent possible relationships:

touching
near_touching
not_touching
occluding/overlapping in image
same-color adjacent
different-color adjacent

Start with k-nearest neighbors or Delaunay triangulation, then classify edges.

For every pair of nearby balls:

distance ≈ diameter       → likely touching/contact edge
distance > diameter       → near but separate
distance < diameter       → overlap/occlusion or bad centers
distance much too small   → duplicate or impossible

The edge itself should be uncertain at first. The optimizer can decide whether it is a real contact or just nearby.

This handles:

one red knocked out,
deformed rack,
missing node,
extra nearby color,
partial cluster.

No fixed 15-node template required.

General cluster pipeline
1. Detect cluster components

Use current rough detections and distance graph.

if balls are closer than ~1.25–1.4 diameters:
    connect them

Connected components become clusters.

Examples:

single ball       → per-ball mode
two touching      → pair cluster mode
3–8 arbitrary     → graph cluster mode
10–15 reds        → dense graph mode, optional rack prior

Do not use rack logic unless the graph strongly resembles a rack.

2. Build multiple candidate centers per ball

For each rough detection, keep candidate centers from several sources:

rough detector center
best evidence-map ellipse center
physical projection center
local mask/contour center
neighbor-consensus-adjusted center

Do not immediately choose one.

Each candidate gets a score:

image support
boundary support
color support
physical plausibility
cluster consistency

Then the cluster optimizer chooses/fuses.

3. Estimate local projected ball shape

Inside one small cluster, all balls should have nearly the same projected shape:

similar ellipse axes
similar ellipse angle
similar source-pixel scale

That does not mean every individual ball must have a good ellipse. It means the cluster should have a shared local projection model.

So estimate shape from trusted balls:

perimeter balls with good exterior arcs
loose nearby balls
current physical projection

Then use that shape as a prior for weak/inner balls.

Current code already has the right instinct with consensus major/minor axes and angle; the package notes that outlier ellipses in DSC00540 were about 1.5× too large or rotated differently, which is exactly the kind of thing a shared-shape prior should reject.

4. Optimize centers jointly

For each cluster, solve all centers together.

Unary terms

For each ball:

stay near rough center if rough detection is reliable
match accepted boundary points if visible
match color/ball probability map
match predicted physical sphere projection
Pairwise terms

For each nearby pair:

non-overlap:
  distance >= diameter - tolerance

touching edge:
  distance ≈ diameter

near edge:
  distance > diameter

duplicate penalty:
  two centers cannot explain same image blob
Cluster terms

For the whole cluster:

predicted union of balls should cover ball-probability mask
outer boundary of predicted union should match observed cluster boundary
unexplained mask regions should propose missing hypotheses
predicted centers should not create impossible overlaps

This is much more robust than fitting one ellipse at a time.

Important: perimeter vs interior evidence

This is the key for clusters.

Perimeter balls

Use image evidence strongly:

visible exterior arcs
cloth-vs-ball transition
cluster outer contour
physical projection band
Interior balls

Use image evidence weakly:

few/no exterior arcs
maybe contact seams
rough detector center
neighbor/contact constraints
shared shape prior

For an interior ball, the system should not require a good individual boundary ellipse. It should say:

position source: joint cluster graph
individual boundary evidence: weak
position confidence: medium/high if graph constraints are strong

This is not a failure. It is the physically correct interpretation.

Boundary ownership is the missing concept

Current workflow lets each ball independently sample arcs. That is why neighbor arcs and reflections can be stolen.

After a provisional cluster solution, every boundary point should be assigned to an owner:

which ball does this point belong to?
is it on an exterior visible arc?
is it inside another ball?
is it likely a highlight?
is it a seam between touching balls?

A boundary point is accepted for ball i only if:

it is close to ball_i predicted boundary
ball_i is the best owner
the point is not inside another predicted ball
the local gradient/color transition is compatible
the arc is expected to be visible

That prevents one ball’s edge from becoming another ball’s contour.

Treat contact seams differently from exterior boundaries

In a cluster, not all visible edges are useful in the same way.

There are:

exterior boundary:
  ball vs cloth
  strong evidence for center

contact seam:
  ball vs ball
  useful but weaker
  can be reflection/shadow-sensitive

highlight edge:
  not boundary
  should be rejected

neighbor outer arc:
  belongs to another ball
  should not be stolen

So the evidence map should classify arcs:

exterior_arc
contact_seam_arc
highlight_arc
neighbor_owned_arc
unknown

The UI can still show accepted points white and rejected points red, as you want, but the JSON/debug can keep the reason.

Generic missing-ball handling

A broken rack may have missing detections. Do not assume all expected positions are present.

Instead:

Add hypothesis generation

If the cluster mask or contact graph suggests a missing ball, propose a hidden/weak hypothesis.

Sources for new hypotheses:

holes in close-packed graph
local maxima in ball probability map
gaps between touching neighbors
expected equilateral position from two touching balls
unexplained cluster mask area
snooker inventory count

Each proposed ball starts as:

status = hypothesis
confidence = low

It becomes final only if:

it improves cluster loss
does not violate inventory
has enough image/color/physical support
does not create impossible overlaps

This handles examples like “one ball knocked out,” “one hidden interior ball,” or “rough detector missed one.”

Generic extra/duplicate handling

Similarly, if there are too many rough detections:

two detections explain same physical ball
center distance too small
same mask region
same boundary ownership

Then the optimizer can deactivate one.

Use an active variable or post-hoc duplicate suppression:

active_i = true/false

with penalty for removing a detection, but strong penalty for impossible overlap.

Mixed-color clusters

Do not restrict cluster mode to same-color reds.

A cluster graph can include:

red-red
red-pink
red-black
cue-red
green-red

The pairwise physical constraints are the same:

same radius
non-overlap
possible contact

The difference is in color evidence:

each node has its own label/color probability
same-color seams are harder
different-color seams are easier

So the graph optimizer should support mixed labels from the start.

For example:

{
  "nodes": [
    {"id": 3, "label": "red"},
    {"id": 17, "label": "pink"},
    {"id": 8, "label": "red"}
  ],
  "edges": [
    {"a": 3, "b": 17, "type": "possible_contact"},
    {"a": 17, "b": 8, "type": "near_touching"}
  ]
}
Scoring function

A practical score could be:

cluster_score =
  0.25 * rough_anchor_score
+ 0.25 * exterior_boundary_score
+ 0.20 * pair_distance_score
+ 0.15 * mask_union_score
+ 0.10 * physical_projection_score
+ 0.05 * inventory/color_score
- penalties

Penalties:

impossible overlap
unexplained mask area
stolen neighbor arcs
highlight arcs used as boundary
large individual ellipse scale outlier
large movement from rough center without evidence

This should be configurable and benchmarked, not hard-coded forever.

Fitting order

A robust order:

1. Loose-ball estimates.
2. Build proximity graph.
3. Identify cluster components.
4. For each cluster:
   a. estimate shared local shape from reliable perimeter evidence
   b. create candidate centers and missing hypotheses
   c. initialize contact graph
   d. optimize centers jointly
   e. assign boundary ownership
   f. refit/score visible arcs
   g. iterate 2–3 times
5. Compare cluster solution vs old per-ball solution.
6. Promote only if cluster solution improves physical and image scores.

Important: cluster solution should not be allowed to wreck good loose balls. It only applies to connected components above a risk threshold.

Promotion rules

A cluster solution becomes final only if:

non-overlap violations reduced
pair distances become more plausible
cluster mask/outer boundary score improves or stays good
center movements are reasonable
no high-confidence loose/perimeter ball is made worse
sample inventory stays valid
physical residual does not worsen significantly

Otherwise it remains diagnostic.

For a selected ball, final position source can be:

loose_image_evidence
cluster_graph_joint_fit
cluster_graph_with_weak_boundary
manual_review

That is much more honest than “ellipse accepted.”

Validation metrics

For arbitrary clusters, validate with these metrics:

center-to-center distance histogram
number of impossible overlaps
minimum separation
contact-edge residual
cluster mask IoU
outer boundary residual
unexplained mask area
duplicate detections removed
missing hypotheses promoted
movement from baseline
repeatability across repeated captures

For DSC00540, add:

dense red component detected
cluster graph size
number of active red nodes
pair-distance RMS
overlap violation count
bad ellipse promotions = 0

Your broader validation plan already values physical constraints like touching-ball distance, rack/cluster nearest-neighbor distances, cushion-touch distances, and repeatability; these are exactly the right gates for cluster work.

What to tell Codex

Here is the corrected task. This avoids overfitting to a 15-ball triangle.

Task: Implement generic cluster graph optimizer for arbitrary snooker ball clusters.

Context:
A special triangular rack solver is too narrow. In real games, clusters can be incomplete, deformed, mixed-color, partially touching, or have one or more balls knocked out. Current per-ball ellipse fitting fails in dense clusters because interior balls have weak/no cloth boundary and per-ball fits can steal neighbor arcs or lamp reflections.

Goal:
Create a generic cluster solver that optimizes arbitrary ball clusters jointly using image evidence, shared local projection shape, physical equal-radius constraints, contact/near-contact graph constraints, and cluster mask support.

Do not implement a hard-coded 15-red rack solver as the primary algorithm.
A triangular rack prior may be an optional special prior only when the graph strongly matches it.

Implement:

1. New modules
- snookerhelp/recognition/cluster_graph.py
- snookerhelp/recognition/cluster_optimizer.py
- snookerhelp/recognition/boundary_ownership.py

2. Cluster detection
- Build a proximity graph from current rough/final ball estimates.
- Connect balls if center distance is below configurable multiple of ball diameter.
- Connected components with size >= 2 become cluster candidates.
- Classify cluster risk:
  loose_pair, touching_pair, arbitrary_cluster, dense_cluster, possible_rack_like.

3. Node model
Each cluster node stores:
- ball id / label
- rough source center
- candidate centers from evidence maps
- current final center
- projected/shared ellipse shape
- confidence
- active flag
- whether node is observed or hypothesized

4. Edge model
Each graph edge stores:
- node ids
- initial distance
- relation probability:
  touching / near_touching / separated / duplicate
- same-color or mixed-color
- visibility/occlusion relation
- pairwise residual

5. Candidate hypotheses
Generate optional missing-ball hypotheses from:
- unexplained cluster mask regions
- gaps in close-packed graph
- equilateral predictions from neighboring contacts
- ball probability local maxima
- inventory constraints

Generate duplicate-removal candidates for centers that are physically too close.

6. Shared local shape prior
Estimate a local projected ball shape for the cluster from:
- reliable perimeter balls
- physical projection if available
- robust median of good ellipses

Use this as a prior, not a hard replacement.

7. Joint optimization
Optimize all active centers in the cluster jointly.

Loss terms:
- rough center anchor loss
- evidence-map candidate loss
- exterior boundary loss
- cluster mask union loss
- pairwise non-overlap loss
- touching-edge distance loss
- near-edge separation loss
- shared shape consistency loss
- physical projection residual
- color/inventory consistency
- movement prior

Use robust losses so lamp reflections/outlier arcs do not dominate.

8. Boundary ownership
After initial cluster solution:
- assign every raw boundary point to the most plausible ball boundary
- reject points inside another ball's predicted interior
- reject points whose best owner is another ball
- classify arcs as exterior boundary, contact seam, highlight, neighbor-owned, unknown
- only exterior/contact-compatible arcs can influence refinement

9. Highlight/reflection rejection
Add highlight mask:
- high brightness
- low saturation
- white/near-white
- inside predicted ball area
- not near expected exterior boundary

Exclude these edges from boundary ownership/refinement.

10. Iteration
Run:
- initial cluster optimize
- boundary ownership
- local refinement
- rescore
for 2–3 iterations.

11. Final estimate selection
If cluster solution passes promotion gates:
- use cluster_graph_joint_fit as final position source for affected balls
- individual per-ball ellipses become diagnostics only if they disagree with cluster graph
- do not promote large/rotated per-ball ellipses that violate shared shape or graph constraints

If cluster solution fails gates:
- keep existing per-ball solution
- mark cluster solution diagnostic only

12. JSON/report output
For each cluster:
- cluster_id
- cluster_type
- active nodes
- hypothesized nodes
- removed duplicates
- edge count
- contact edges
- pair_distance_rms
- overlap violations
- cluster_mask_iou
- outer_boundary_residual
- confidence
- promotion status

For each ball:
- position_source
- cluster_id if any
- individual_boundary_confidence
- cluster_position_confidence
- nearest-neighbor residuals
- active/hypothesis/duplicate status

13. UI
Keep user-facing view simple:
- white accepted dots
- red rejected dots
- cream fitted outline
- final center
- confidence

Add optional advanced cluster overlay:
- contact graph edges
- cluster-constrained centers
- hidden/hypothesized nodes
- pair-distance residuals

14. Benchmarks
Add tests/reports for:
- intact DSC00540
- deformed rack / missing red if available
- arbitrary cluster DSC00542
- loose balls DSC00524/25/26 must not regress
- green/blue/brown loose cases must not regress

Metrics:
- sample count/inventory
- pair distance RMS
- overlap violations
- duplicate count
- unexplained mask area
- bad per-ball ellipse promotions
- confidence distribution
- physical validation metrics

Acceptance criteria:
- The solver handles arbitrary clusters, not only full triangular racks.
- It reduces impossible overlaps and bad ellipse promotions in DSC00540.
- It does not force missing nodes when a ball has genuinely moved out.
- It can handle mixed-color clusters.
- Loose-ball benchmark does not regress.
- Cluster solutions are promoted only when both physical and image evidence improve.