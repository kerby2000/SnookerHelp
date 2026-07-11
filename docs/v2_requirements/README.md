# Detailed recognition requirements

These files contain detailed contracts for acceptance gates that are not yet
complete. They are implemented incrementally in the existing codebase; this is
not a parallel rewrite. The authoritative order and status live in
[`../execution_plan.md`](../execution_plan.md).

## Reading order

1. [00_principles.md](00_principles.md)
2. [01_data_contracts.md](01_data_contracts.md)
3. [02_scenarios_and_validation.md](02_scenarios_and_validation.md)
4. [03_pipeline_overview.md](03_pipeline_overview.md)
5. [04_evidence_model.md](04_evidence_model.md)
6. [05_loose_ball_solver.md](05_loose_ball_solver.md)
7. [06_cluster_graph_solver.md](06_cluster_graph_solver.md)
8. [07_confidence_model.md](07_confidence_model.md)
9. [08_review_ui.md](08_review_ui.md)
10. [09_implementation_plan.md](09_implementation_plan.md)
Implementation order is defined only in `docs/execution_plan.md`.

## Existing docs used as source material

| Existing doc | Borrow into |
| --- | --- |
| `docs/architecture.md` | coordinate systems, camera-agnostic rules, classical OpenCV constraint |
| `docs/approximate_camera_model.md` | approximate pinhole mode and current limitations |
| `docs/charuco_calibration_workflow.md` | future calibrated camera model and CALITAR board details |
| `docs/ball_geometry_model.md` | source-image fitting, sphere projection, confidence language, ball numbering |
| `docs/boundary_filtering_strategy.md` | evidence maps, global cloth reference, cluster failure findings |
| `docs/physical_validation_tools.md` | touching, cushion, spot, repeatability validation |
| `docs/coordinate_accuracy_validation.md` | manual annotation, accuracy metrics, repeatability |
| `docs/image_debug_reports.md` | visual report/review workflow lessons |
| `docs/execution_plan.md` | active implementation order, evidence from earlier audits, and promotion gates |

## Rule for old docs

Completed plans and migration diaries are removed from the active tree. Their
history remains available in Git checkpoint `155e727`. If a statement here
conflicts with `docs/execution_plan.md`, the execution plan wins.
