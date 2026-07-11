# 02 — Scenarios and Validation

Validation must exist before dense solver complexity. Otherwise there is no defensible way to decide whether a heuristic improved recognition or only improved one screenshot.

For v2 development, every benchmark image must have a `ScenarioSpec`. Full manual ball centers are not required, but expected inventory and validation checks are required.

## Scenario YAML format

Scenario metadata is optional but should have one stable format.

```yaml
scenario_id: DSC00540_rack_cluster
images:
  - DSC00540.JPG
type: dense_cluster
expected:
  total_balls: 22
  reds: 15
  colors: [white, yellow, green, brown, blue, pink, black]
touching_pairs:
  - image: DSC00540.JPG
    ball_a: red_01
    ball_b: red_02
    expected_distance_mm: 52.5
cushion_touches:
  - image: DSC00529.JPG
    ball: red_04
    cushion: bottom
    expected_distance_mm: 26.25
spot_mappings:
  - image: DSC00540.JPG
    spot: blue
    ball: blue
repeatability_group:
  group_id: random_layout_01
  images: []
notes: ""
```

Minimal scenario example:

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

## Real-image benchmark groups

| Group | Images | Purpose |
| --- | --- | --- |
| Empty table | `DSC00543` and similar | false positives and cloth reference |
| Loose/random balls | `DSC00524`, `DSC00525`, `DSC00526`, `DSC00527` | normal recognition, no regression |
| Near cushions | `DSC00529` and similar | source geometry near edges |
| Near pockets | `DSC00534` and similar | pocket/corner artifacts |
| Rack/triangle | `DSC00540` | dense red triangle |
| Arbitrary cluster | `DSC00541`, `DSC00542` | non-rack cluster behavior |

## Synthetic tests

Required synthetic cases:

- isolated projected ellipse with known center;
- partial visible arcs;
- neighboring ball overlap;
- missing inner boundary;
- duplicate detections for one ball;
- highlight rectangle near a ball boundary;
- same-color cluster with shared local ellipse shape;
- low-contrast green/blue/brown ball against cloth.

## Acceptance gates

Before v2 replaces v1:

1. Loose-ball images must not regress in accepted count or obvious center quality.
2. Dense clusters must not accept impossible oversized ellipses as final estimates.
3. Obvious duplicates must be suppressed or marked unresolved.
4. Missing hypotheses must be represented explicitly.
5. The UI must explain why a ball is trusted or not trusted.
6. Validation reports must be reproducible from CLI.
7. Benchmarks must run on the agreed image groups.

## What counts as regression

Regression examples:

- accepted ball count drops on loose images without marking missing hypotheses;
- new duplicate accepted as a real ball;
- confidence increases while physical validation worsens;
- dense-cluster wrong ellipse becomes accepted final estimate;
- old clear source evidence disappears from UI;
- benchmark output becomes non-reproducible;
- review UI introduces old candidate terminology.
- benchmark image has no scenario spec;
- missing hypotheses are counted as accepted balls without strict gates.

## Required metrics

Per image:

- accepted ball count;
- suppressed duplicate count;
- missing hypothesis count;
- low-confidence count;
- mean confidence;
- physical overlap violation count;
- touching-distance error where scenario data exists;
- cushion-distance error where scenario data exists.

Per cluster:

- cluster size;
- edge count;
- contact count;
- duplicate hypotheses;
- missing hypotheses;
- shared-shape spread;
- selected solution score;
- unresolved nodes.

Per ball:

- selected evidence map;
- accepted/rejected boundary samples;
- arc coverage;
- fit residual;
- physical projection residual;
- local shape consensus residual;
- confidence components.
