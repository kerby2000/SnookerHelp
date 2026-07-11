# 03 — Pipeline Overview

This file describes stages only. Solver math belongs in later files.

## Stage 1 — Load image and geometry

Inputs:

- source image;
- detector config;
- table model;
- camera model;
- ball radius/diameter.

Outputs:

- `ImageContext`;
- cloth/table mask;
- global cloth reference;
- camera/table transform.

## Stage 2 — Generate rough hypotheses

Create initial `BallHypothesis` objects.

Temporary implementation may reuse existing rough detector output, but v2 must treat the rough detector as replaceable.

Requirements:

- preserve raw detector IDs;
- keep plausible low-confidence hypotheses;
- do not silently suppress duplicates here;
- assign initial labels and rough centers.

## Stage 3 — Compute evidence maps

For each selected crop and/or full image region, compute:

- source image view;
- grayscale edge;
- Lab Delta-E;
- chroma difference;
- ball-vs-cloth probability;
- physical projection band;
- combined boundary score.

Only configured evidence maps influence final estimates. Others are diagnostics.

## Stage 4 — Sample boundary evidence

Generate `BoundarySample` objects around each hypothesis.

Requirements:

- sample candidate boundary points;
- keep accepted and rejected points;
- group points into arcs;
- keep rejection reasons in data, not as primary UI colors.

## Stage 5 — Solve isolated balls

If a ball has no meaningful neighbors, use the loose-ball solver.

The loose solver checks:

- image evidence support;
- arc coverage;
- projected shape plausibility;
- residuals;
- no-overlap constraints.

## Stage 6 — Build cluster graph

Nearby/touching/overlapping hypotheses become `ClusterGraph` components.

The graph includes:

- ball nodes;
- possible contact edges;
- duplicate edges;
- overlap edges;
- missing-ball hypotheses;
- boundary ownership links.

## Stage 7 — Solve cluster jointly

Dense clusters are solved as a group.

The solver estimates:

- centers;
- boundary ownership;
- duplicate suppressions;
- missing hypotheses;
- confidence.

## Stage 8 — Select final estimates

For each physical ball:

- select loose or cluster solution;
- assign canonical ID;
- compute table coordinates;
- compute confidence;
- mark unresolved cases.

## Stage 9 — Export reports and validation

Outputs:

- canonical JSON;
- review UI data;
- validation reports;
- benchmark results.

