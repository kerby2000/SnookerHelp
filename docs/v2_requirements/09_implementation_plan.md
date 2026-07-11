# 09 — Implementation Plan

## Phase 0 — Freeze prototype

Stop adding main recognition logic to the current v1 prototype.

Allowed:

- bug fixes needed to inspect current data;
- documentation;
- adapters for migration.

Not allowed:

- new dense-cluster heuristics in the old path;
- new candidate terminology in the UI;
- more prototype-only overlays as primary features.

## Phase 1 — Data contracts

Deliverables:

- v2 dataclasses or Pydantic models;
- JSON serialization;
- schema tests;
- adapter from current detector output to `BallHypothesis`;
- one sample exported v2 JSON.
- explicit estimate status and position-source enums;
- split confidence object;
- `ClothModel`;
- debug artifact references.

Acceptance:

- schema round-trip tests pass;
- no solver logic required yet.

## Phase 2 — Scenarios and benchmarks

Deliverables:

- scenario YAML loader;
- scenario files for every benchmark image;
- benchmark runner;
- benchmark result schema;
- real-image benchmark groups;
- regression comparison output.

Acceptance:

- loose, cushion, pocket, rack, and arbitrary-cluster groups can run through the benchmark harness.

## Phase 3 — Pipeline skeleton

Deliverables:

- staged pipeline object;
- timing per stage;
- cached intermediate output;
- CLI entrypoint.

Acceptance:

- pipeline runs from source image to rough hypotheses and empty placeholder estimates.

## Phase 4 — Evidence model

Deliverables:

- global cloth reference;
- evidence maps behind one interface;
- boundary sampling;
- arc extraction;
- evidence-map scoring;
- visual debug export.

Acceptance:

- selected-ball evidence can be inspected for the benchmark images.

## Phase 5 — Loose-ball solver

Deliverables:

- isolated-ball solver;
- projected shape scoring;
- confidence components;
- no-overlap/duplicate checks for simple cases.

Acceptance:

- loose/random images do not regress against current best known reports.

## Phase 6 — Cluster graph model

Deliverables:

- cluster graph builder;
- edge classification;
- boundary ownership model;
- duplicate/missing hypotheses;
- cluster diagnostics.

Acceptance:

- `DSC00540` and `DSC00542` produce inspectable cluster graphs.

## Phase 7 — Cluster optimizer

Deliverables:

- joint center optimization;
- shared-shape prior;
- contact/no-overlap constraints;
- cluster mask support;
- solution ranking and promotion.

Acceptance:

- dense-cluster wrong oversized ellipses are not accepted as final estimates;
- duplicate/missing status is explicit;
- arbitrary clusters do not regress.

## Phase 8 — Confidence model

Deliverables:

- component score calculation;
- calibrated level thresholds;
- explanation strings;
- benchmark confidence summary.

Acceptance:

- confidence changes can be justified from components and benchmark deltas.

## Phase 9 — Review UI v2

Deliverables:

- new UI against v2 schema;
- pan/zoom full image;
- selected-ball crop;
- evidence table;
- cluster view;
- confidence explanation;
- manual feedback storage.

Acceptance:

- UI has no Candidate A/B/C/D terminology;
- primary overlay uses white/red/cream/green and optional blue only.

## Phase 10 — Replace default path

Deliverables:

- default CLI uses v2;
- old prototype path behind compatibility flag;
- docs updated;
- benchmark report attached.

Acceptance:

- all v2 gates pass.

## Phase 11 — Archive/delete obsolete paths

Only after v2 passes:

- move old prototype code to `legacy/` or delete;
- remove obsolete docs from the main docs map;
- keep media, scenarios, feedback, and benchmark artifacts.
