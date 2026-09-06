# Changelog

## 0.2.0

Schedule construction now preserves rounding residuals when computing a small uniform
remainder beside much larger scheduled sources. This corrects misplaced minority
elements, including constant schedules such as `Sampling::delayed(0.0)`, and can
change previously generated orders. Keep the same crate version when resuming an
existing ordering. This minor version change follows the crate's pre-1.0 ordering
stability policy.

- Shuffled cursors retain seek buffers for mixes reached beneath the shuffle.
  Initialized `last()` calls also reuse state, and selecting an empty range defers
  tree repositioning while preserving buffers.
- `Seq::try_shard` provides fallible worker configuration. `Seq::validate` consumes
  configurations and safely disposes of rejected trees; `Seq::dispose` exposes
  iterative disposal for callers using borrowed validation.
- Independent, reproducible weight and schedule fixtures cover exact quotas,
  piecewise-linear schedules, and minority-element placement at large lengths.
- Phase benchmarks sample positions within the named phase. Timings use warmup,
  calibrated batches and five samples. Campaigns alternate revision order, report
  medians and ranges, and validate structured reports and workload identities.
