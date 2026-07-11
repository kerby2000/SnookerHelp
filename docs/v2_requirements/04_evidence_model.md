# 04 — Evidence Model

The evidence model converts source pixels into boundary samples and arcs. It does not decide the final physical solution alone.

## Global cloth reference

Use global cloth reference by default.

Reason:

- local annuli are contaminated in dense clusters;
- inner cluster balls may have no visible cloth;
- global cloth gives stable image-wide color comparison.

Local annulus values remain diagnostics:

- local cloth Lab;
- sample count;
- contamination estimate;
- difference from global cloth.

## Evidence maps

Required maps:

| Map | Purpose | Baseline use |
| --- | --- | --- |
| Source image | visual truth for human review | diagnostic/default visual |
| Grayscale edge | luminance boundary | limited |
| Lab Delta-E | color distance from cloth | alternate |
| Chroma difference | color-only separation from cloth | preferred for green/blue/brown |
| Ball-vs-cloth probability | learned/local score against cloth | default for most balls |
| Physical projection band | geometric prior | prior only |
| Combined boundary score | ensemble diagnostic | experimental |

## Class-specific evidence baseline

Initial baseline from observed samples:

| Ball class | Preferred map |
| --- | --- |
| red | ball-vs-cloth probability |
| white | ball-vs-cloth probability or grayscale edge |
| yellow | ball-vs-cloth probability |
| green | chroma difference |
| blue | chroma difference |
| brown | chroma difference |
| black | source/edge/chroma depending on exposure |

This is not a final decision rule.

Evidence maps produce candidate evidence. Final selection must be based on:

- evidence-map score;
- boundary arc quality;
- physical projection agreement;
- local shape consensus;
- duplicate/missing checks;
- cluster graph consistency.

For loose balls, a single evidence map may win. For clusters, multiple maps may contribute to the graph solution.

## Boundary sampling

Boundary sampling must:

- sample around the expected projected outline;
- keep multiple candidates per angle sector;
- record strength and source map;
- store both accepted and rejected points;
- group points into arcs.

The sampler must not permanently discard points before cluster solving.

## Arc extraction

Boundary points must be grouped into arcs using:

- spatial proximity;
- angular continuity;
- evidence strength;
- residual to projected/observed shape;
- ownership score.

Arc-level reasoning is required for cluster solving.

## Highlight rejection

Specular highlights and lamp reflections are not ball boundaries.

The evidence model must down-weight:

- rectangular bright highlight interiors;
- straight highlight edges;
- saturated regions;
- repeated lamp reflection patterns on neighboring balls.

Rejected highlight samples remain visible as red dots.

## Physical projection band

The physical projection band is not observed image evidence.

It may:

- guide search windows;
- score residuals;
- provide expected shape/scale/orientation;
- help cluster graph optimization.

It must not:

- create fake white boundary samples;
- be counted as image support;
- override real evidence while the camera model is approximate.

Hard invariant:

Every accepted boundary sample must reference an observed image/evidence-map source. A physical projection may guide where to search, but it must not manufacture accepted boundary samples by itself.
