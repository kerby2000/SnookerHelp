# 05 — Loose Ball Solver

The loose-ball solver handles isolated balls and simple non-touching cases.

## Inputs

- `BallHypothesis`;
- evidence maps;
- boundary samples;
- boundary arcs;
- projected physical shape;
- table/camera model.

## Acceptance requirements

A loose ball can be accepted when:

- enough boundary arc coverage exists;
- selected evidence map has good score;
- fitted observed shape has plausible size;
- physical projection residual is plausible;
- no nearby duplicate is present;
- no impossible overlap exists;
- color label is plausible.

## Final center

The final source center must be derived from the selected observed/physical agreement, not from warped cloth-plane circle fitting.

The output must identify:

- selected evidence map;
- observed shape;
- projected shape;
- final source center;
- final table XY;
- confidence components.

## Rejection / review cases

Loose-ball estimate must be marked `needs_review` when:

- evidence maps disagree strongly;
- boundary coverage is too low;
- shape is implausible;
- physical residual is high;
- color is ambiguous;
- duplicate hypothesis exists.

## Confidence components

Loose-ball confidence uses:

- image evidence score;
- arc coverage score;
- residual score;
- physical projection agreement;
- duplicate/no-overlap checks;
- calibration quality.

## Non-goals

The loose-ball solver must not contain dense-cluster special cases.

If cluster membership is detected, hand off to the cluster graph solver.

