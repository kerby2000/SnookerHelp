# SnookerHelp documentation

This directory contains maintained product and engineering documentation. Start
with these documents, in order:

1. [execution_plan.md](execution_plan.md) - active implementation phases,
   acceptance gates, and cleanup policy.
2. [architecture.md](architecture.md) - project boundary, runtime components,
   hardware, and current data flow.
3. [ball_geometry_model.md](ball_geometry_model.md) - image-space ellipse
   evidence, physical sphere projection, final center, and uncertainty.
4. [boundary_filtering_strategy.md](boundary_filtering_strategy.md) - current
   evidence maps, sampling, filtering, and cluster limitations.
5. [image_debug_reports.md](image_debug_reports.md) - report/review/experiment
   workbench.
6. [coordinate_accuracy_validation.md](coordinate_accuracy_validation.md) and
   [physical_validation_tools.md](physical_validation_tools.md) - QA methods.
7. [charuco_calibration_workflow.md](charuco_calibration_workflow.md) - real
   camera calibration workflow.

Detailed contracts that have not yet passed their acceptance gates remain in
[v2_requirements/](v2_requirements/README.md). They support the active execution
plan; they are not a separate rewrite project.

Specialized references:

- [approximate_camera_model.md](approximate_camera_model.md)
- [precision_calibration_inputs.md](precision_calibration_inputs.md)
- [baseline_validation.md](baseline_validation.md)
- [manim_geometry_visualizations.md](manim_geometry_visualizations.md)

## Current implementation summary

```text
rough detection:          warped cloth-plane image
final source evidence:    source image crops
image evidence:           filtered edge-boundary ellipse, with rejected outliers visible in red
diagnostic maps:          grayscale edge, Lab/chroma contrast, ball-vs-cloth probability, physical band
cloth reference:          global table-cloth Lab by default; local annulus retained as contamination diagnostic
map variants:             selected map can drive its own visible dots and cream ellipse for review
final map policy:          ball-vs-cloth default; chroma for green/blue/brown; other maps diagnostic
cluster filtering:         neighbor-owned boundary points rejected when they fall inside nearby ball ellipses
cluster shells:            large adjacent clusters show perimeter/interior shell diagnostics
cluster traversal:         large adjacent clusters show outside-in CW/CCW diagnostic traversal ranks
review viewport:           full-table source panel supports wheel zoom, drag pan, reset, and Fit selected
review numbering:          #1 white, #2 yellow, #3 green, #4 brown, #5 blue, #6 pink, #7 black, #8-22 table-ordered reds
physics evidence:         approximate or calibrated projected sphere model, forward or optimized
scene constraints:        adjacent-ball equal-radius/non-overlap/contact diagnostics
final estimate:           source pixel + camera/table projection + confidence
auto confidence:          image evidence + physical model + scene constraints, with benchmark scores retained
camera model:             approximate_pinhole_from_corners until ChArUco calibration
human review:             v1 schema review feedback; legacy review.json is still readable
```

The most important conceptual rule:

```text
The warped image is a cloth-plane debug view.
The source image is where ball evidence is measured.
The camera model is how source pixels become table coordinates.
```
