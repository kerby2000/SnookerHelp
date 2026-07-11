# SnookerHelp Recognition v2 Requirements Index

This folder is the working requirements set for a fresh recognition v2 implementation.

The older monolithic document, `docs/recognition_v2_requirements.md`, remains as a capture/archive document. Implementation work should use the smaller files in this folder.

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
11. [10_codex_task_01_schema_and_benchmark.md](10_codex_task_01_schema_and_benchmark.md)

## Implementation order

The implementation order is intentionally not solver-first.

1. [Task 01](10_codex_task_01_schema_and_benchmark.md): data contracts, scenario files, and benchmark schema check.
2. Scenario/validation formats and benchmark runner extensions.
3. Pipeline skeleton.
4. Evidence model.
5. Loose-ball solver.
6. Cluster graph solver.
7. Confidence model.
8. Review UI.

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
| `docs/refactor_plan_v1.md` | clean architecture and UI language rules |
| `docs/proposal.md` | generic cluster graph strategy from external review |
| `docs/recognition_strategy_review_package.md` | current failure modes and constraints |

## Rule for old docs

Old docs are evidence and history. They are not automatically requirements.

If a statement conflicts with these v2 files, the v2 requirement files win.
