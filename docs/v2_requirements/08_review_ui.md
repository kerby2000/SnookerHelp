# 08 — Review UI

The v2 UI is a new implementation against the v2 schema.

Do not refactor the old review UI into v2.

## Main purpose

For each image, the UI must answer:

1. Did the system find the balls?
2. What image pixels support the selected estimate?
3. What physical model was used?
4. Why is the estimate trusted or not trusted?
5. Are there duplicates or missing balls?

## Main layout

Required panels:

- full source image with pan/zoom;
- selected-ball crop;
- evidence table;
- physical model summary;
- confidence summary;
- ball statistics table;
- cluster view when selected ball is in a cluster.

## Full source image

Requirements:

- pan and zoom;
- readable labels at zoom;
- click ball to select;
- fit selected ball/cluster;
- show accepted, duplicate, missing, and unresolved states clearly.

## Selected-ball crop

Requirements:

- large crop;
- selectable background image/evidence map;
- overlay toggles for accepted dots, rejected dots, final outline, final center, rough center, physical projection, neighbor outlines;
- no cluttered primary legend over the ball.

## Overlay rules

Primary overlays:

| Overlay | Meaning |
| --- | --- |
| White dots | accepted image boundary evidence |
| Red dots | rejected/unused image boundary evidence |
| Cream outline | final selected outline |
| Green cross | final source center |
| Optional blue outline | physical projection prediction |

No multi-color reject categories in the primary view.

## Evidence table

For each evidence map show:

- map name;
- score;
- whether it contributes to final estimate;
- accepted point count;
- rejected point count;
- fit residual;
- arc coverage.

Clicking a map should switch the crop background and overlay that map's points/outline.

## Cluster view

When a selected ball is clustered, show:

- cluster ID;
- cluster members;
- contact edges;
- duplicate hypotheses;
- missing hypotheses;
- shared shape statistics;
- selected cluster solution score;
- unresolved reasons.

## Manual correction

Manual correction is secondary.

Allowed controls:

- OK;
- NOK;
- mark duplicate;
- mark missing ball;
- optional source center correction;
- optional comment.

Do not force the user into hand-labeling every ball.

## Advanced mode

Advanced mode may show:

- rejection reason table;
- optimizer energy terms;
- rejected cluster solutions;
- arc-combination candidates;
- timing/debug data.

Advanced diagnostics must not dominate the default UI.

