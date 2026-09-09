# Changelog

## Unreleased

- **Breaking (0.3.0):** simplify the rising-rate inverse using remaining area from
  the segment endpoint. This removes numerical repair searches and changes some
  item orders near ties. Resume existing checkpoints with their original crate version.
- Schedule construction uses the precise peak directly and fused product residuals,
  removing correction-factor arithmetic and custom significand splitting. Smooth
  rate curves, exact quotas, and seek/walk agreement are preserved.

- Checkpoints accept worker shards that fit on 32-bit targets even when their original
  sequence is larger. Cloned mix cursors preserve reusable vector capacity after moving
  to smaller concat children.
- Compiler verification checks an independently resolved rustup compiler path, rejecting
  custom launchers named `rustc` and older verification records without an override.
  Failed benchmark commands terminate descendants, including after the parent exits;
  Windows subprocesses start suspended and enter a job before running.
- Checkpoint batches stream through one cursor and commit successful progress after
  resuming. Worker creation rejects configurations the checkpoint parser cannot read.
  `ORDERING_VERSION` exposes the linked crate's version for conservative checkpoints.
- Tree traversal avoids temporary singleton child vectors and uses shared rebuild
  frames. Concat cursors recycle compatible child buffers across epochs and seeks.
- Independent schedule oracles now test continuous walks and contiguous large-order
  windows. Allocation regressions cover unary traversal and concat buffer recycling.
- Benchmark campaigns mark wrapped compiler invocations as unverified and require an
  explicit override. Commands have configurable deadlines, terminate timed-out process
  trees and retain failure diagnostics without replacing completed campaign results.

- Mapping and cleanup after callback errors or panics use an explicit heap stack,
  including previously mapped and unvisited deep sibling branches.
- Mix cursors box their largest state, reducing per-part storage. Lifecycle benchmarks
  measure requested, retained and peak cursor memory and worker scaling, and include
  wide weight exponents, remainder ties and near-capacity schedules.
- Preparation reports include compiled transform parameters, original source paths,
  lengths and salts, and successful schedule capacity/tolerance diagnostics.
  `Error::sampling_detail` distinguishes invalid breakpoint and coefficient failures
  and reports the values behind excessive steepness.
- Independent schedule fixtures cover complementary schedules without uniform parts,
  interacting ramps, reordered minorities, nearby boundaries, exact ties and the mix
  length limit. Stateful cursor regressions minimize failing operation histories.
- Benchmark campaigns build in isolated directories, capture Cargo-observed compiler
  settings, check environment compatibility and commit comparison labels atomically.
- A versioned checkpoint example validates configuration, immutable source metadata,
  seeds, worker settings, crate version and resume position before restoring.

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
