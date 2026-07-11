# 01 — Data Contracts

This file defines the first v2 implementation target. No solver complexity should be implemented until these contracts exist with tests.

## Coordinate systems

| Name | Meaning |
| --- | --- |
| `source_px` | Original source image pixels |
| `undistorted_px` | Lens-undistorted image pixels after calibration |
| `warped_px` | Cloth-plane debug rectification only |
| `table_mm` | 2D table coordinate system in millimeters |
| `world_mm` | 3D table coordinate system in millimeters |

Final estimates must always identify which coordinate system each point uses.

## ImageContext

Represents one processed image.

Required fields:

- `image_name`
- `image_path`
- `image_size_px`
- `capture_metadata`
- `table_model`
- `camera_model`
- `ball_radius_mm`
- `cloth_reference`
- `rough_hypotheses`

## ClothModel

The cloth model is a first-class input to evidence maps.

Required fields:

- `source_image_id`
- `mask_source`
- `cloth_lab_mean`
- `cloth_lab_covariance`
- `valid_region_mask`
- `excluded_regions`
- `exposure_stats`
- `confidence`

The global cloth model must exclude:

- balls;
- rails;
- pockets;
- highlights;
- strong shadows;
- outside-table regions.

## CameraModel

Required fields:

- `mode`
- `calibrated`
- `image_size_px`
- `camera_matrix`
- `distortion_coefficients`
- `world_to_camera`
- `camera_to_world`
- `quality`

Required modes:

- `manual_homography_compat`
- `approximate_pinhole_from_corners`
- `calibrated_pinhole`

Required methods:

```python
undistort_points(points_px) -> points_px
image_point_to_world_ray(point_px) -> Ray3D
intersect_ray_with_z(ray, z_mm) -> point_xyz_mm
world_point_to_image(point_xyz_mm) -> point_px
project_sphere(center_xyz_mm, radius_mm) -> ProjectedShape
```

## TableModel

Required fields:

- `width_mm`
- `height_mm`
- `cloth_corners_source_px`
- `cloth_corners_table_mm`
- `cushion_lines_table_mm`
- `pocket_regions_table_mm`
- `coordinate_origin`
- `coordinate_axes`

## BallHypothesis

A rough possible ball before final acceptance.

Required fields:

- `estimate_id`
- `hypothesis_id`
- `raw_detector_id`
- `source_center_px`
- `rough_radius_px`
- `label`
- `label_confidence`
- `source`
- `estimate_status`
- `position_source`
- `cluster_id`

Allowed `estimate_status` values:

- `candidate`
- `accepted`
- `needs_review`
- `suppressed_duplicate`
- `unresolved_duplicate`
- `missing_hypothesis`
- `rejected_hypothesis`
- `diagnostic_only`
- `manual_corrected`

Allowed `position_source` values:

- `loose_image_evidence`
- `cluster_graph_joint_fit`
- `physical_projection_optimized`
- `rough_detector_only`
- `manual_correction`
- `missing_hypothesis`

ID rules:

- `estimate_id` is always unique within one processed image.
- `canonical_id` is stable only for uniquely identifiable colored balls: white, yellow, green, brown, blue, pink, black.
- Red `canonical_id` is frame-local unless a future tracking module assigns a `track_id`.
- `track_id` is optional and reserved for future temporal tracking.

## EvidenceMap

Required fields:

- `map_id`
- `display_name`
- `coordinate_system`
- `image`
- `normalization`
- `source_features`
- `diagnostics`

Required map IDs:

- `source_image`
- `grayscale_edge`
- `lab_delta_e`
- `chroma_difference`
- `ball_vs_cloth_probability`
- `physical_projection_band`
- `combined_boundary_score`

## BoundarySample

A possible boundary point.

Required fields:

- `sample_id`
- `ball_hypothesis_id`
- `evidence_map_id`
- `point_source_px`
- `angle_rad`
- `radius_px`
- `strength`
- `accepted`
- `rejection_reasons`
- `arc_id`
- `owner_hypothesis_id`

Rejected samples remain visible as red dots in the review UI.

## BoundaryArc

A group of related boundary samples.

Required fields:

- `arc_id`
- `ball_hypothesis_id`
- `sample_ids`
- `evidence_map_id`
- `angle_start_rad`
- `angle_end_rad`
- `coverage_rad`
- `mean_strength`
- `fit_residual_px`
- `ownership_score`

v2 solvers operate on arcs first, points second.

## ProjectedShape

Represents a physical camera/table/ball projection.

Required fields:

- `shape_id`
- `source_center_px`
- `ellipse_px`
- `contour_points_source_px`
- `z_mm`
- `ball_radius_mm`
- `camera_model_mode`
- `calibrated`
- `residual_to_evidence_px`

## ObservedShape

Represents an image-derived shape.

Required fields:

- `shape_id`
- `evidence_map_id`
- `ellipse_px`
- `supporting_arc_ids`
- `supporting_sample_ids`
- `rms_residual_px`
- `arc_coverage_rad`
- `quality`

## ClusterGraph

Required fields:

- `cluster_id`
- `node_ids`
- `edge_ids`
- `cluster_type`
- `solver_status`
- `selected_solution_id`

Cluster types:

- `none`
- `touching_pair`
- `small_cluster`
- `dense_cluster`
- `rack_like_cluster`
- `arbitrary_large_cluster`

## ClusterNode

Required fields:

- `node_id`
- `hypothesis_id`
- `label`
- `initial_center_source_px`
- `optimized_center_source_px`
- `accepted`
- `suppression_reason`
- `missing_reason`

## ClusterEdge

Required fields:

- `edge_id`
- `node_a`
- `node_b`
- `relationship`
- `distance_mm`
- `expected_distance_mm`
- `weight`
- `reasons`

Relationships:

- `possible_contact`
- `near_contact`
- `overlap`
- `duplicate`
- `non_contact`
- `unknown`

## ClusterSolution

Required fields:

- `solution_id`
- `cluster_id`
- `node_solutions`
- `edge_solutions`
- `energy_terms`
- `score`
- `accepted`
- `rejection_reasons`

## BallEstimate

Final output per accepted or unresolved ball.

Required fields:

- `estimate_id`
- `canonical_id`
- `track_id`
- `raw_detector_id`
- `label`
- `source_center_px`
- `table_xy_mm`
- `effective_z_mm`
- `observed_shape`
- `projected_shape`
- `selected_evidence_map_id`
- `confidence`
- `cluster_id`
- `estimate_status`
- `position_source`
- `manual_feedback`

Example red identity:

```json
{
  "estimate_id": "estimate_014",
  "canonical_id": "red_frame_014",
  "track_id": null,
  "label": "red"
}
```

## Confidence

Required fields:

- `image`
- `physical`
- `scene`
- `final`
- `components`
- `reasons`
- `penalties`
- `calibration_quality`

Each score group contains:

- `score`
- `level`
- `reasons`

Levels:

- `high`
- `medium`
- `low`
- `unknown`

## ReviewFeedback

Required fields:

- `image_name`
- `canonical_id`
- `raw_detector_id`
- `decision`
- `manual_source_center_px`
- `manual_label`
- `missing_ball`
- `duplicate_of`
- `comment`
- `human_confidence`
- `timestamp`

Human confidence must be `null` unless the user explicitly sets it.

## ScenarioSpec

Required fields:

- `scenario_id`
- `image_names`
- `scenario_type`
- `expected_inventory`
- `checks`
- `touching_pairs`
- `cushion_touches`
- `spot_mappings`
- `repeatability_group`
- `notes`

## BenchmarkResult

Required fields:

- `benchmark_id`
- `image_group`
- `pipeline_version`
- `metrics`
- `regressions`
- `artifacts`
- `timestamp`

## Debug artifact references

Canonical output JSON must stay reasonably small.

Large debug data should be stored as separate artifacts referenced by path:

- evidence maps as PNG/NPY files;
- crop images as PNG/JPG;
- boundary samples as JSONL/CSV/Parquet;
- cluster graph diagnostics as JSON;
- optimizer traces as JSONL.

The canonical JSON should include summaries and artifact references, not embed every large debug array.
