# Ball geometry and fitting model

Start here to understand how SnookerHelp fits balls today and how that should
evolve into the calibrated v1 geometry pipeline.

## Short version

SnookerHelp now separates four ideas that used to be mixed together:

```text
Pixels          what the camera sees
Image evidence  edges, masks, contours, ellipses measured in the source image
Physical model  camera rays, ball radius, table plane, sphere projection
Final estimate  the selected source point and table XY with confidence
```

The important rule:

```text
The warped image is a cloth-plane debug and rough-detection view.
The source image is where ball evidence is measured.
The camera model is how source pixels become table coordinates.
```

## Current pipeline

The current implementation still uses a practical bridge from the prototype:

```text
source photo
  -> cloth-plane warp for rough ball candidates
  -> map each rough candidate back to the source photo
  -> collect source-image evidence in a local crop
  -> fit image evidence models
  -> compare evidence against the physical sphere projection
  -> project the selected source point through the camera model
  -> output table coordinates, uncertainty, and review evidence
```

The warped view remains useful for background subtraction, table masking,
coarse location, and debug display. It is not used as the final place to judge
whether a ball is round.

## Why warped ball shapes are misleading

The four-corner table homography is a cloth-plane transform. It is valid for
flat objects on:

- cloth;
- cushion/table boundary points;
- table lines/spots.

It is not valid for a 3D sphere above the cloth.

```text
cloth plane:       Z = 0 mm
ball center plane: Z = 26.25 mm
ball top plane:    Z = 52.5 mm
```

When a ball near a side/corner is warped as if every visible point lies on the
cloth, the visible blob becomes stretched and shifted. That is expected. The
correct final position must come from source-image evidence plus camera
geometry, not from a warped-image circle center alone.

## Image evidence

For every detected ball crop, the code gathers several measured facts from the
source image.

### Radial boundary points

The algorithm samples outward from a provisional center and looks for likely
ball boundary points. These points are useful when the visible boundary is
clean. They are weaker near cushions, pockets, shadows, highlights, and dense
clusters.

Since v1.2, these points are split into accepted and rejected evidence:

- accepted edge points are drawn as white dots;
- rejected edge outliers are drawn as red dots;
- the observed cream ellipse is fitted only from accepted white points;
- rejected red points remain visible in the review UI.

The current filter removes angular arc endpoints, local radius spikes, and
large ellipse-residual outliers. It does not try to invent missing edge points.
For low-contrast blue/green balls on green cloth, the next step is better
color-aware evidence rather than only stronger outlier rejection.

See [boundary_filtering_strategy.md](boundary_filtering_strategy.md) for the
current filter details, evidence maps, benchmark numbers, and the planned
green/blue strategy.

### Local evidence maps

v1.3 adds evidence maps for every ball crop:

- grayscale edge magnitude;
- Lab Delta-E from the active cloth reference;
- chroma-only difference from the active cloth reference;
- ball-vs-cloth probability;
- physical projection band score.

For green/blue balls, the map weights Lab/chroma and ball-vs-cloth probability
more heavily than raw grayscale edge strength. This is the first step toward
recovering weak boundaries that ordinary edge detection misses.

The v1 review UI shows these maps as selectable crop backgrounds, so failures
can be inspected visually instead of only through summary numbers.

Since v1.3.8, the active cloth reference for the color maps is global by
default. The system estimates one table-cloth Lab value per image from the
table polygon, excluding detected ball neighborhoods and invalid dark/highlight
pixels. The older local annulus around each ball is still computed and shown as
a diagnostic because it exposes exactly when a cluster crop is contaminated by
neighboring balls or shadows.

v1.3.4 also samples boundary points from each map separately. In the UI, the
crop background selector now controls:

1. the displayed diagnostic map;
2. the accepted/rejected boundary points;
3. the cream observed ellipse fitted from those points.

This makes it possible to compare whether `gray_edge`, `Lab Delta-E`,
`chroma_difference`, `ball_vs_cloth_probability`,
`physical_projection_band`, or `combined_boundary_score` gives the most
plausible observed ellipse on a difficult ball.

v1.3.6 promotes one configured map-specific variant to the final source-pixel
estimate:

```text
default balls:             ball_vs_cloth_probability
green / blue / brown:      chroma_difference
all other evidence maps:   diagnostic only
```

This is not a physical law. It is a reviewed heuristic based on free-standing
sample images. It should be judged by physical validation and may be replaced
by a better per-ball policy later.

Cluster note from DSC00542: this policy is much weaker inside tight red-ball
clusters. v1.3.8 fixes the worst local-cloth-reference failure by using global
cloth for B/C/L maps. v1.3.9 adds the first cluster-aware ownership filter:
nearby ball ellipses are drawn as purple dashed reference outlines, and sampled
boundary points that fall inside those neighbor ellipses can be rejected before
the selected ball's cream ellipse is fitted.

This is still not a full rack/cluster solver. The remaining problem is deciding
which visible edge belongs to which ball when several balls touch. The correct
next improvement is not simply to choose the highest map score; it is to extend
cluster-aware evidence ownership:

- neighbor exclusion zones and neighbor-owned point rejection;
- connected visible-arc scoring;
- local color-model health checks;
- explicit low-confidence output when the evidence belongs to multiple balls.

Concrete example: DSC00542 #9 previously had a local annulus cloth Lab almost
identical to the ball Lab, so B/C/L maps looked wrong. Global cloth gives #8
and #9 similar ball-vs-cloth separation again. The remaining error is that
neighbor-ball arcs can still be sampled as if they belonged to #9; v1.3.9 makes
that failure visible and rejects obvious neighbor-owned points, but it is not
yet a complete inside-out cluster reasoning algorithm.

### Mask contour and centroid

The algorithm also builds a rough ball mask from color and local contrast. The
mask contour and centroid are diagnostic evidence:

- if the mask agrees with the radial boundary, confidence improves;
- if the mask leaks into a neighbor or cushion, confidence drops;
- the mask centroid is not automatically the physical ball center.

### Observed ellipse

The most useful current image-shape model is the observed ellipse from edge and
boundary evidence. This is a 2D description of the visible blob in the source
image. It often explains elongated balls better than a circle.

Observed ellipse evidence is still not ground truth. Highlights, occlusions,
touching balls, cushions, and shadows can make the apparent ellipse incomplete
or biased.

## Physical model

The physical model asks a different question:

```text
Given the camera model, table coordinate system, and ball radius,
what should a known-radius ball look like at this table position?
```

The current development mode is `approximate_pinhole_from_corners`. It uses:

- source image size;
- approximate focal length;
- approximate sensor size;
- principal point;
- manually clicked table corners;
- known table dimensions;
- ball radius.

It can:

- cast camera rays through source pixels;
- intersect rays with `Z=0`, `Z=26.25`, and `Z=52.5` planes;
- project an approximate ball silhouette into the source image;
- compare that projected silhouette with observed image evidence;
- expose a physical projection band as diagnostic/scoring evidence;
- run a bounded local X/Y optimization that can move the blue physical
  projection when evidence supports it;
- run an adjacent-ball scene-constraint check for close clusters using
  equal-radius, non-overlap, and touching-distance constraints.

It cannot yet:

- correct real lens distortion;
- guarantee exact focal length;
- replace ChArUco calibration;
- make the projected sphere silhouette the final truth.

The blue curve should be read as:

```text
forward mode:
  current estimated 3D ball center -> projected sphere silhouette

optimized mode:
  local X/Y search -> projected sphere silhouette that better matches evidence
```

Because the current camera is approximate, optimized mode is evidence, not
automatic ground truth.

For adjacent balls, v1.3.1 adds a conservative joint cluster diagnostic. It
does not replace final coordinates; it reports whether close balls can be made
more physically plausible with a small bounded movement. This is useful for
rack/cluster cases where missing arcs are expected.

After ChArUco calibration, the same interface should use real intrinsics,
distortion, and camera pose. At that point the projected sphere model can carry
much more trust.

## Final estimate

Today the final table coordinate is still conservative:

1. choose the configured final image-evidence map for that ball class;
2. fit the observed ellipse from that map's accepted boundary points;
3. check that the ellipse has plausible aspect ratio, size, and center shift;
4. use that ellipse center as the final source-pixel estimate when available;
5. fall back to the older source point if the selected map fails or violates
   the plausibility guard;
6. project that source point through the active camera model;
7. report table XY and uncertainty;
8. compute physical optimization as supporting evidence;
9. use image/physical disagreement to decide confidence and review priority.

The next major geometry step is to make the physical sphere model solve the
source center more directly:

```text
observed boundary evidence
  -> calibrated camera
  -> known ball radius
  -> optimize 3D ball center
  -> final table XY
```

That requires ChArUco calibration plus a robust optimizer that fits the
projected sphere silhouette to the observed boundary/mask evidence.

v1.3 has the first bounded optimizer, but it does not yet replace the final
source center by default. That is intentional until calibrated-camera validation
and physical validation scenarios prove the movement is trustworthy.

## Why a circle baseline still exists internally

The old prototype used circle fitting as the primary source refinement. In v1,
that circle fit should be treated only as:

- an internal fallback;
- a quick baseline for clean center-region balls;
- a regression benchmark against newer image/physical evidence.

It should not be exposed as the main user-facing model, and it should not be
trusted automatically for elongated edge/cushion balls.

## Confidence

Confidence should answer:

```text
How much should we trust this final table position?
```

Current v1 confidence is not measured against ground truth. Unless a manual
annotation, touching-ball scenario, cushion-touch scenario, rack constraint, or
spot validation is provided, there is no true external target in the score.

The displayed automatic score is an internal agreement score:

```text
displayed confidence =
  max(
    legacy detector/review baseline,
    usable image-evidence + physical-projection agreement scores
  )
```

Important guardrail: if the blue physical sphere projection has a low-quality
residual against the selected boundary points, the physical score is not
allowed to raise confidence.

The per-evidence-view score in the v1 UI is different. It is only a diagnostic
map-comparison score:

```text
view score =
  45% physical-outline residual
+ 30% accepted boundary point count
+ 20% inlier ratio
+  5% ellipse availability
```

This helps compare `ball_vs_cloth_probability` vs `chroma_difference` vs other
views for the same ball. It is not exported physical accuracy.

The Image evidence panel also shows the active color-model parameters:

- active reference mode, currently global by default;
- ball Lab;
- active cloth Lab;
- active Lab/chroma separation;
- whether the active separation is low contrast;
- local annulus Lab/separation as a contamination diagnostic;
- global cloth Lab and sample count;
- active radius/value/exclusion factors.

If the local annulus separation is very low but global separation is high, the
crop is probably contaminated by a neighbor, shadow, cushion, or pocket.

Useful confidence signals:

- enough source-image boundary points exist;
- observed ellipse has plausible size and aspect ratio;
- observed ellipse agrees with the projected sphere silhouette;
- mask contour does not leak into a neighbor/cushion;
- table position is not in a known difficult region;
- no duplicate/overlapping detection is nearby;
- physical validations are satisfied.

Confidence should drop when:

- source evidence is missing or partial;
- observed ellipse and physical projection disagree;
- mask centroid is far from the selected center;
- the ball is near a cushion or pocket mouth;
- detections overlap too strongly;
- touching-ball or cushion-touch validation fails.

## Physical validation is the external truth

When the image evidence is ambiguous, the strongest practical evidence is
physical validation:

- touching balls should be `52.5 mm` center-to-center;
- a ball touching a cushion should be `26.25 mm` from the cushion line;
- rack/cluster red-ball nearest-neighbor distances should cluster near
  `52.5 mm`;
- spot balls should land on known spot coordinates;
- repeated unchanged photos should have low XY standard deviation.

These checks are how we decide whether the camera model and final estimates are
improving, not just whether an overlay looks plausible.

## Review UI language

The v1 UI should use these concepts:

- Pixels
- Image evidence
- Physical model
- Final estimate
- Confidence
- Manual correction

It should not ask the user to approve a named prototype candidate. The user
should see what pixels were used, which evidence agrees or disagrees, which
final position is being used, and why the confidence is high/medium/low.

## What ChArUco calibration adds

The CALITAR ChArUco board is needed to replace approximate geometry with a real
camera model:

- camera matrix;
- distortion coefficients;
- camera pose relative to the table;
- reprojection-error validation.

Once available, the physical model can be promoted from a scoring aid to the
main solver for difficult elongated balls.

## Current implementation map

Important modules:

- `snookerhelp/recognition/source_refinement.py`: source-crop refinement and
  boundary evidence;
- `snookerhelp/recognition/evidence_maps.py`: local edge/color/probability
  evidence maps, global/local cloth-reference diagnostics, and physical
  projection-band diagnostics;
- `snookerhelp/recognition/physical_optimize.py`: bounded physical X/Y
  optimizer for the projected sphere model;
- `snookerhelp/recognition/cluster_optimize.py`: adjacent-ball scene-constraint
  diagnostics for equal radius, non-overlap, and touching distance;
- `docs/boundary_filtering_strategy.md`: accepted/rejected boundary-point
  filtering strategy and low-contrast green/blue plan;
- `snookerhelp/review/evidence_builder.py`: report/review evidence
  construction;
- `snookerhelp/recognition/sphere_projection.py`: approximate/calibrated sphere
  projection;
- `snookerhelp/recognition/confidence.py`: confidence scoring experiments;
- `snookerhelp/core/schema.py`: v1 data schema;
- `snookerhelp/review/server.py`: v1 review API;
- `snookerhelp/review/static/`: v1 browser UI.

Longer-term target modules:

- `snookerhelp/recognition/image_model.py`: source-image evidence extraction;
- `snookerhelp/recognition/physical_model.py`: calibrated sphere model;
- `snookerhelp/calibration/charuco.py`: real calibration workflow;
- `snookerhelp/review/`: schema-driven review and feedback tooling.
