# 00 — Principles

## Why v2 exists

Recognition v1 taught useful lessons, but it accumulated incompatible prototype paths:

- warped-image circle fitting;
- source-image radial/circle fitting;
- mask/centroid experiments;
- evidence-map experiments;
- add-back experiments;
- fixed-rack/traversal experiments;
- large review UI overlays exposing internal candidates.

The main failure is dense clusters. In clusters such as `DSC00540`, an inner ball often has little or no visible cloth boundary. A per-ball ellipse can fit a wrong subset of pixels and still look plausible. That bad estimate then contaminates neighboring balls.

v2 exists to restart around a clean model:

source pixels + image evidence + physical camera/table/ball model + cluster graph consistency

## Core product language

The user-facing product language is:

- Pixels
- Image evidence
- Physical model
- Final estimate
- Confidence
- Manual correction
- Validation

Do not expose these as primary UI concepts:

- Candidate A/B/C/D
- radial candidate
- mask candidate
- add-back candidate
- traversal candidate
- fixed rack candidate
- multi-color reject categories

Internal debug JSON may contain algorithm names, but the primary UI must not be organized around them.

## Architecture rules

1. Final ball measurement happens in source or undistorted image coordinates.
2. Warped cloth-plane images are debug/rough-detection aids only.
3. Ball geometry is 3D. The cloth plane is `Z=0`; the ball center is approximately `Z=26.25 mm`.
4. The physical projection is a weak prior until real ChArUco calibration exists.
5. Dense clusters are solved jointly, not by one independent ellipse per ball.
6. Missing and duplicate balls are first-class hypotheses, not silent failures.
7. Confidence is a measured agreement score, not visual smoothness.
8. Human feedback is stored separately from algorithm output.
9. Evidence maps produce candidate evidence. A per-color evidence policy is a baseline prior, not final authority.
10. The cluster optimizer may abstain instead of forcing a plausible-looking but wrong result.

## Small-codebase rules

The fresh implementation must avoid another monolith.

- Keep schema, evidence, solvers, confidence, UI, and validation in separate modules.
- Keep core data classes small and serializable.
- Prefer explicit adapters over importing legacy prototype modules directly.
- Every solver promotion rule must have a benchmark or test.
- If a diagnostic does not affect final output, label it diagnostic.
- Do not add UI controls before the data exists in the schema.
- No module over 500 lines without explicit approval.
- No function over 80 lines without explicit approval.
- No HTML/JS generated as one giant Python string.
- No UI dependency inside recognition modules.
- No recognition dependency inside review UI except through the v2 JSON schema.
- No old prototype import unless wrapped by a v2 interface and covered by tests.

## What not to carry forward as main design

Do not carry forward as primary v2 design:

- independent circle/radial model as final authority;
- mask centroid as final authority;
- fixed 15-red rack solver as the main solver;
- perimeter-first traversal as the main solver;
- physical projection band as fake observed evidence;
- local annulus cloth reference as default;
- per-color evidence-map policy as final authority;
- multi-color reject reasons in the main overlay;
- silent duplicate deletion;
- silent missing-ball failure;
- long static report pages as the only review mechanism.
