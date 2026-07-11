# 07 — Confidence Model

Confidence must be explainable, split by source, and benchmarked.

## Inputs

Confidence uses:

- image evidence score;
- boundary arc coverage;
- fit residual;
- physical projection residual;
- local shared-shape consistency;
- contact graph consistency;
- duplicate/missing status;
- calibration quality;
- validation results when available.

## Output

Each `Confidence` object contains:

- `image`
- `physical`
- `scene`
- `final`
- `components`
- `reasons`
- `penalties`
- `calibration_quality`

Example:

```json
{
  "confidence": {
    "image": {"score": 0.91, "level": "high"},
    "physical": {"score": 0.63, "level": "medium", "reasons": ["approximate_camera"]},
    "scene": {"score": 0.84, "level": "medium"},
    "final": {"score": 0.76, "level": "medium"}
  }
}
```

## Levels

Initial levels:

| Level | Score |
| --- | --- |
| high | `>= 0.85` |
| medium | `0.60 - 0.85` |
| low | `< 0.60` |
| unknown | insufficient data |

Thresholds must be recalibrated after benchmark results.

## Calibration penalty

Approximate camera mode must reduce confidence when physical projection is important.

Modes:

- `manual_homography_compat`: weak geometry only;
- `approximate_pinhole_from_corners`: useful but approximate;
- `calibrated_pinhole`: strongest physical projection.

If `camera_model.mode` is approximate, geometry confidence cannot be high for:

- near-cushion balls;
- near-pocket balls;
- dense clusters;
- balls whose image evidence and physical projection disagree.

The system may report high `image` confidence while capping `physical` and `final` confidence.

## Manual feedback separation

Manual review is not algorithm confidence.

Human input fields must remain separate:

- human OK/NOK;
- manual center;
- manual missing mark;
- manual duplicate mark;
- human confidence if explicitly set.

The system must not fill human confidence by default.

## Required explanation

The UI and JSON must explain confidence in plain language.

Example:

```text
Score 0.72 medium.
Image evidence is good, physical projection residual is medium, cluster contact
graph is consistent, but camera model is approximate and one neighboring arc is
ambiguous.
```
