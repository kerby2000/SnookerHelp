# 10 — Codex Task 01: Schema and Benchmark Foundations

This file is the handoff for a fresh Codex window.

Do not give the new Codex window the whole previous discussion. Give it this file and the linked requirement files.

## Copy/paste prompt for the new Codex window

```text
We are starting a fresh SnookerHelp recognition v2 implementation.

Read docs/v2_requirements/10_codex_task_01_schema_and_benchmark.md first, then read only the files it lists.

Implement Task 01 only: v2 data contracts, explicit enums, ScenarioSpec YAML loader, initial scenario files, BenchmarkResult model, and an empty benchmark CLI that validates scenarios and writes JSON.

Do not implement recognition heuristics, dense cluster solving, review UI, or old prototype refactors.

Keep new code in a clean v2 namespace and add tests.
```

## Task objective

Create the v2 data-contract and benchmark foundation.

Do not implement recognition heuristics, cluster solving, review UI, or dense-cluster optimization in this task.

The output of this task should make later solver work measurable and constrained.

## Files to read first

Read these files in order:

1. `docs/v2_requirements/00_principles.md`
2. `docs/v2_requirements/01_data_contracts.md`
3. `docs/v2_requirements/02_scenarios_and_validation.md`
4. `docs/v2_requirements/03_pipeline_overview.md`
5. `docs/v2_requirements/09_implementation_plan.md`

Optional reference only:

- `docs/recognition_v2_requirements.md`
- `docs/approximate_camera_model.md`
- `docs/charuco_calibration_workflow.md`
- `docs/physical_validation_tools.md`
- `docs/boundary_filtering_strategy.md`
- `docs/ball_geometry_model.md`

The split v2 files are authoritative if any older document disagrees.

## Non-negotiable rules

1. Freeze the old prototype recognition path.
2. Do not add new heuristics to current v1/v1.4 recognition modules.
3. Do not refactor the old review UI.
4. Do not use Candidate A/B/C/D terminology.
5. Do not implement a fixed 15-red rack solver.
6. Do not use physical projection as manufactured image evidence.
7. Do not silently delete duplicates or missing balls.
8. Keep new code small and isolated.

## New source-tree rule

Use a clean v2 namespace.

Preferred structure for this task:

```text
snookerhelp/
  v2/
    __init__.py
    core/
      __init__.py
      schema.py
      ids.py
      coordinates.py
    qa/
      __init__.py
      scenario.py
      benchmark.py
```

Do not put v2 schemas into the old `snookerhelp/core/schema.py` unless a thin adapter is required.

Do not import old prototype modules into v2 unless the import is wrapped and tested.

## Required deliverables

### 1. v2 schema models

Implement data models for:

- `ImageContext`
- `CameraModel`
- `TableModel`
- `ClothModel`
- `BallHypothesis`
- `EvidenceMap`
- `BoundarySample`
- `BoundaryArc`
- `ProjectedShape`
- `ObservedShape`
- `ClusterGraph`
- `ClusterNode`
- `ClusterEdge`
- `ClusterSolution`
- `BallEstimate`
- `Confidence`
- `ReviewFeedback`
- `ScenarioSpec`
- `BenchmarkResult`

Use dataclasses or Pydantic. Pick the simplest option that supports:

- type checking;
- JSON serialization;
- stable defaults;
- schema round-trip tests.

### 2. Explicit enums

Implement enums for:

```text
CoordinateSystem:
  source_px
  undistorted_px
  warped_px
  table_mm
  world_mm

CameraModelMode:
  manual_homography_compat
  approximate_pinhole_from_corners
  calibrated_pinhole

EstimateStatus:
  candidate
  accepted
  needs_review
  suppressed_duplicate
  unresolved_duplicate
  missing_hypothesis
  rejected_hypothesis
  diagnostic_only
  manual_corrected

PositionSource:
  loose_image_evidence
  cluster_graph_joint_fit
  physical_projection_optimized
  rough_detector_only
  manual_correction
  missing_hypothesis

ConfidenceLevel:
  high
  medium
  low
  unknown

ClusterType:
  none
  touching_pair
  small_cluster
  dense_cluster
  rack_like_cluster
  arbitrary_large_cluster
```

### 3. ID rules

Implement helper functions for IDs:

- `estimate_id` is unique within one processed image.
- colored balls may use stable canonical IDs:
  - `white`
  - `yellow`
  - `green`
  - `brown`
  - `blue`
  - `pink`
  - `black`
- reds get frame-local canonical IDs such as `red_frame_014`.
- `track_id` is optional and remains `None` until a future temporal tracker exists.

Do not imply stable physical identity for reds across different shots.

### 4. Split confidence object

Implement confidence as:

```json
{
  "image": {"score": 0.91, "level": "high", "reasons": []},
  "physical": {"score": 0.63, "level": "medium", "reasons": ["approximate_camera"]},
  "scene": {"score": 0.84, "level": "medium", "reasons": []},
  "final": {"score": 0.76, "level": "medium", "reasons": []},
  "components": {},
  "penalties": {},
  "calibration_quality": {}
}
```

Do not collapse this to one score.

### 5. ScenarioSpec YAML loader

Implement a loader for scenario YAML files.

For v2 benchmark images, scenario files are mandatory.

Minimum supported format:

```yaml
image: Media/05_clusters/DSC00540.JPG
scenario_type: dense_red_cluster
expected_inventory:
  total: 22
  red: 15
checks:
  - no_accepted_impossible_overlaps
  - duplicate_hypotheses_reported
  - missing_hypotheses_reported_if_present
  - dense_cluster_report_present
  - no_oversized_cluster_ellipses_promoted
notes:
  exact_centers_are_ground_truth: false
```

### 6. Initial scenario files

Create scenario files for the initial benchmark set.

Suggested location:

```text
data/scenarios/v2/
```

Required initial files:

- `DSC00524.yaml`
- `DSC00525.yaml`
- `DSC00526.yaml`
- `DSC00527.yaml`
- `DSC00529.yaml`
- `DSC00534.yaml`
- `DSC00540.yaml`
- `DSC00541.yaml`
- `DSC00542.yaml`
- `DSC00543.yaml`

These files do not need manual centers. They must define:

- image path;
- scenario type;
- expected inventory where known;
- required regression checks;
- notes.

### 7. BenchmarkResult model and empty benchmark CLI

Implement a benchmark CLI that can:

- load scenario files;
- validate their structure;
- write a `BenchmarkResult` JSON summary;
- report missing scenario files;
- not run any solver yet.

Suggested CLI:

```powershell
python -m snookerhelp.v2.qa.benchmark --scenarios data/scenarios/v2 --output outputs/v2_benchmark/schema_check.json
```

Expected first behavior:

- validate scenario YAML files;
- emit scenario count;
- emit missing/invalid files;
- write a benchmark summary JSON.

## Tests required

Add tests for:

- enum values;
- schema JSON round trip;
- red ID rules;
- colored ball ID rules;
- confidence split structure;
- scenario YAML load;
- missing scenario detection;
- benchmark summary output.

Suggested tests:

```text
tests/v2/test_schema_contracts.py
tests/v2/test_ids.py
tests/v2/test_scenarios.py
tests/v2/test_benchmark_cli.py
```

Run at least:

```powershell
python -m pytest tests/v2
```

If broader tests are cheap, run:

```powershell
python -m pytest
```

## Acceptance criteria

Task 01 is complete when:

1. v2 schema models exist in the clean v2 namespace.
2. Explicit enums exist and are tested.
3. Red IDs are frame-local by default.
4. Split confidence object is implemented and tested.
5. Scenario YAML loader exists.
6. Initial scenario files exist for the benchmark images.
7. Empty benchmark CLI validates scenarios and writes JSON.
8. Tests pass for the new v2 schema/benchmark layer.
9. No old recognition heuristics were added or changed.
10. No review UI code was modified.

## Expected final response from Codex

When done, report:

- files added;
- tests run;
- benchmark CLI command tested;
- any assumptions made;
- what Phase 2 should do next.

Do not claim recognition quality improved. This task only builds the contracts and benchmark foundation.
