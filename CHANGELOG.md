# Changelog

## Unreleased (0.4.0)

- **Breaking:** remove `Seq::Weighted`, `Seq::weighted`, `Seq::weighted_with`,
  `WeightedPart`, and the `InvalidWeight`, `ZeroWeights` and `EmptyWeightedPart`
  error kinds. Choose exact counts with `Seq::mix([a.cycle(a_count), b.cycle(b_count)])`
  or attach schedules with `Seq::mix_with`. Remove the exact floating-point quota
  allocator and its fixtures and benchmarks. Serialized `Weighted` configurations
  must be rewritten using explicit counts.

- **Breaking:** remove `Order::sources_mut`. Open or transform source handles with
  `Seq::map` or `Seq::try_map` before compilation.
- **Breaking:** make checked access the default and remove the corresponding `try_`
  aliases. `Order::get` and `get_indexed` return `Option`; `Order::iter`, cursor
  `seek`/`set_range`, and `Seq::slice`/`shard` return `Result`. Invalid cursor
  operations preserve its state. `Order::source_index` returns `None` for foreign
  references or zero-sized source types; use indexed results for explicit ordinals.

- **Breaking:** reduce `MAX_DEPTH` from 256 to 16 and remove `Seq::dispose`.
  `Seq` remains an enum. Compilation, mapping and cleanup now use ordinary recursion
  within the supported depth limit;
  arbitrarily deep hand-built trees are unsupported.

- **Breaking:** remove `Order::prepare` and the `Preparation`, `PreparedNode`,
  `PreparedKind`, `PreparedParameters`, `PreparedSource`, `PreparedMix` and
  `WeightedAllocation` report types. Use `Order::new` or `Order::with_seed` to
  compile orders. Configuration errors retain their paths and sampling details.
- **Breaking:** schedules now use independent curves on a shared virtual clock.
  `Uniform` is constant in virtual time, like `delayed(0.0)` or `until(1.0)`.
  Start/full/fade/off values no longer denote fractions of the final output;
  all curves adapt when merged, so linear ramps generally become nonlinear in
  output progress. Overlaps and gaps work without a uniform filler.
- Remove the uniform-remainder sweep, compensated sum/expansion arithmetic,
  combined capacity checks and tolerance/clamping policy. Profiles now have at
  most five segments and compile independently in linear time in the part count.
  Individual parameter, coefficient and numerical-resolution checks remain.
- Remove `ErrorKind::Overcommitted`, `ErrorKind::SamplingOverflow`,
  `SamplingDiagnostics` and `PreparedMix::diagnostics`.
- Seek by summing integer counts below virtual-time keys and bounded bisection.
  Counts, source-local order and seek/walk agreement remain exact;
  shuffle arithmetic is unchanged. Scheduled order fingerprints
  change: resume existing checkpoints with their original crate version.

## 0.3.0

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
