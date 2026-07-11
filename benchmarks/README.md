# Tracked recognition benchmarks

This directory contains small, reviewable ground-truth and scenario files.
Generated images, evidence maps, and reports remain under `outputs/` and are not
committed.

- `annotations/<image_stem>.json`: source-image centers, manually fitted
  ellipses, visible/occluded arcs, and reviewer uncertainty.
- `scenarios/`: physical touching, cushion, spot, rack, and repeatability facts.
- `results/`: compact benchmark summaries only.

The annotation schema is `snookerhelp.ground_truth.v1`. Human annotations never
overwrite detector output.

Run the measured ellipse benchmark with:

```powershell
python tools/benchmark_ellipse_annotations.py `
  --report outputs/reports/DSC00540/report.json `
  --annotations benchmarks/annotations/DSC00540.json `
  --output benchmarks/results/DSC00540 `
  --tolerance-px 3
```

The default tolerance is not a detector allowance. It records the expected
human placement uncertainty for blurred/occluded source boundaries.
