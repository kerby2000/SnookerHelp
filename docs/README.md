# SnookerHelp docs map

Start with:

1. [ball_geometry_model.md](ball_geometry_model.md)  
   Main document for current ball fitting, image evidence, physical model,
   final estimate, confidence, and why circle/ellipse/sphere evidence can
   disagree.

2. [image_debug_reports.md](image_debug_reports.md)  
   How to generate reports and use the interactive review UI.

3. [boundary_filtering_strategy.md](boundary_filtering_strategy.md)  
   Current v1.3.9 accepted/rejected edge-point filtering, per-map boundary
   variants, promoted final-position evidence-map policy, global cloth
   reference, neighbor-ellipse cluster ownership filtering, diagnostic
   evidence maps, physical optimization, adjacent-ball scene constraints,
   benchmark gates, and the plan for low-contrast
   blue/green balls on green cloth.

4. [approximate_camera_model.md](approximate_camera_model.md)  
   Current temporary camera model used before ChArUco calibration.

5. [charuco_calibration_workflow.md](charuco_calibration_workflow.md)  
   What to do when the CALITAR board arrives.

6. [physical_validation_tools.md](physical_validation_tools.md)  
   Touching-ball, cushion-touch, rack/cluster, spot, and repeatability
   validation tools.

7. [refactor_plan_v1.md](refactor_plan_v1.md)  
   v1 architecture plan and migration status.

Secondary / reference docs:

- [architecture.md](architecture.md): hardware and project-level architecture.
  Some early milestone text is historical; use `ball_geometry_model.md` for the
  current fitting model.
- [coordinate_accuracy_validation.md](coordinate_accuracy_validation.md):
  manual annotation, accuracy, and repeatability tool reference.
- [baseline_validation.md](baseline_validation.md): early baseline notes.
- [precision_calibration_inputs.md](precision_calibration_inputs.md):
  manual corner calibration command notes.
- [manim_geometry_visualizations.md](manim_geometry_visualizations.md):
  abstract animation plan, not the operational report workflow.

Current implementation summary:

```text
rough detection:          warped cloth-plane image
final source evidence:    source image crops
image evidence:           filtered edge-boundary ellipse, with rejected outliers visible in red
diagnostic maps:          grayscale edge, Lab/chroma contrast, ball-vs-cloth probability, physical band
cloth reference:          global table-cloth Lab by default; local annulus retained as contamination diagnostic
map variants:             selected map can drive its own visible dots and cream ellipse for review
final map policy:          ball-vs-cloth default; chroma for green/blue/brown; other maps diagnostic
cluster filtering:         neighbor-owned boundary points rejected when they fall inside nearby ball ellipses
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
