# Changelog

## Unreleased (0.4.0)

- **Breaking:** derive shuffle salts from the original configuration during compilation.
  Sources contribute their salts and original lengths, including empty or discarded
  sources. Unary operations pass salts through; concatenations combine child salts in
  configuration order before flattening. Nested grouping can affect the result.
  Return salts with compiler summaries and remove the source-salt table and traversal
  of pruned nodes. Single-source shuffle arithmetic and epoch reseeding are unchanged;
  shuffles over concatenations can change, including when selections retain only one
  source. Resume existing orders with their original crate version.

- **Breaking:** reject shuffles whose input configuration contains any mix, including
  empty or single-part mixes and mixes beneath concatenations, repetitions or selections.
  `Order::new` reports `ErrorKind::ShuffleContainsMix` at the nearest enclosing shuffle.
  Shuffle each input before mixing instead. Remove the shuffled cursor's mix-seek cache
  and the callback-based random traversal. Accepted configurations keep their ordering.

- **Breaking:** remove `Eq` and `Hash` from `Seq`, `MixPart` and `Schedule`.
  `PartialEq` now uses ordinary `f64` equality for schedule parameters: signed
  zeros compare equal, and configurations containing NaN compare unequal even
  to themselves. Remove the custom floating-point bit comparison. Ordering,
  validation and serialized configurations are unchanged.

- **Breaking:** rename `Schedule::fading` to `Schedule::fade` to pair with
  `Schedule::ramp`. Replace `fading(fade, off)` calls with `fade(fade, off)`.
  Parameters, validation, ordering and serialized configurations are unchanged.

- **Breaking:** rename `Seq::map` and `Seq::try_map` to `Seq::map_sources` and
  `Seq::try_map_sources` to make clear that they transform source handles.
  Update mapping calls to use the new names. Ordering, callback behavior and
  serialized configurations are unchanged.

- Compare `Item` source ordinals and record indices before source values, avoiding
  source comparisons when either index differs. Document that equality includes
  source values and that their comparison cost depends on the source type.

- **Breaking:** rename `ErrorKind::TooSteep` to `ErrorKind::ScheduleTooSteep`.
  Update matches and constructors to use the new variant name. Its diagnostic
  fields, error paths, display messages and ordering behavior are unchanged.

- Correct the `Source` documentation to describe the returned `Item`, including
  its source ordinal, source reference and record index.

- **Breaking:** rename `Seq::cycle(len)` to `Seq::cycle_to(len)` to make its
  exact finite target length explicit. Replace `.cycle(len)` calls with
  `.cycle_to(len)`. The `Seq::Cycle` variant, serialized configurations and
  ordering behavior are unchanged.

- **Breaking:** replace `Cursor::seek` and `Cursor::set_range` with
  `Cursor::reset(range)`. Each reset replaces the remaining range and moves to its
  start, using absolute order positions. Replace `set_range(range)` with
  `reset(range)`; replace `seek(pos)` with `reset(pos..end)` to retain a chosen
  endpoint, or `reset(pos..)` to read through the order's end. `reset(..)` restarts
  the whole order. Remove `BoundsError::SeekOutOfBounds`; resets use the same
  range validation as `Order::cursor`, and failures leave the cursor unchanged.
  Buffer reuse, ordering and serialized configurations are unchanged.

- **Breaking:** rename `ORDERING_VERSION` to `CRATE_VERSION`. Its value remains
  the linked crate's package version, including patches that preserve ordering.
  Use the renamed constant for conservative exact-version checkpoint checks.

- **Breaking:** split full-order iteration from ranged cursor construction.
  `Order::iter()` now returns a `Cursor` directly; replace `iter(..)?` or
  `iter(..).unwrap()` with `iter()`. Use `Order::cursor(range)` for checked ranges;
  it returns `Result<Cursor, BoundsError>`. Both cursors remain seekable, and
  `IntoIterator for &Order` uses `iter()`. Ordering, bounds errors, allocation
  behavior and serialized configurations are unchanged.

- **Breaking:** rename `Sampling` to `Schedule`, `MixPart.sampling` to
  `MixPart.schedule`, `SamplingReason` to `ScheduleReason`, and
  `ErrorKind::InvalidSampling { sampling, reason }` to
  `ErrorKind::InvalidSchedule { schedule, reason }`. Rename the `sampling` field
  to `schedule` in serialized mix parts; the old field is rejected. Schedule
  variants, parameters, error messages and ordering outputs are unchanged.

- **Breaking:** use `usize` for lengths, positions, offsets, strides and element
  counts throughout compilation and iteration. The `len` fields in
  `ErrorKind::SkipOutOfRange`, `TakeOutOfRange` and `ScheduleTooSteep` now use `usize`.
  Length growth uses checked `usize` arithmetic. Seeds, salts, shuffle arithmetic
  and the numerical mix limit remain `u64`; supported orders are unchanged.

- **Breaking:** remove `Sampling::DelayedLinear`. `delayed(at)` and
  `ramp(start, full)` now return `Trapezoid` with `fade: 1.0` and `off: 1.0`.
  Replace serialized `DelayedLinear` variants with `Trapezoid`, preserving `start`
  and `full` and adding those two fields; the old variant is rejected.
  Equivalent constructor and explicit trapezoid values now compare
  alike; `Uniform` remains distinct. Ordering outputs are unchanged. Debug output
  and error messages that include these schedules now show `Trapezoid`.

- **Breaking:** move schedule diagnostics into `ErrorKind`. `InvalidSchedule` now
  contains `schedule` and `reason: ScheduleReason`; `ScheduleTooSteep` contains `len`,
  `peak_rate` and `limit`. Remove `Error::sampling_detail` and `SamplingDetail`.
  Read the fields through `kind()` or `into_kind()`, which now preserves schedule
  diagnostics. Error paths, full error messages and ordering are unchanged.

- **Breaking:** merge `Seq::mix_with` into `Seq::mix`, which now accepts bare
  sequences, `(seq, schedule)` pairs or `MixPart` values. Replace `mix_with` calls
  with `mix`. Empty inputs need an explicit element type, such as
  `Seq::mix(std::iter::empty::<Seq<usize>>())`. Ordering and serialized
  configurations are unchanged.

- **Breaking:** require every sequence node's length to fit in `usize`, including
  intermediates later truncated or discarded. Oversized intermediates on 32-bit
  targets are rejected at their node. Remove `ErrorKind::OrderTooLong`; length
  overflow uses `ErrorKind::LengthOverflow` on all targets.

- **Breaking:** remove `salt_path`. Choose the dataset identity and its byte
  representation in the caller, then pass those bytes to `salt`.

- **Breaking:** remove `Cursor::remaining`; use `ExactSizeIterator::len()`
  (`cursor.len()`) to read the remaining element count.

- Return lengths and `u8` repetition levels up recursive compiler and folding
  visits. These summaries are temporary; repeat nodes retain their own level for
  reseeding. Ordering outputs and public signatures are unchanged by this refactor.

- **Breaking:** number repetition levels from the inside out instead of by enclosing
  depth. Adding an outer repeat or extending a cycle now preserves the entire first
  pass, including nested epochs. Later passes use the repeat's epoch and level to
  reseed existing shuffles. Non-nested repetition and source salts are unchanged;
  nested repetition orders change. Resume existing orders with their original crate
  version. Remove the compiler's repeat-depth repair traversal.
- Dataset examples derive salts from stable dataset names rather than storage paths.

- Position cursors when constructed and when moved to empty ranges. Remove deferred
  root initialization and separate tracking of the tree's previous position.
  `last()` now uses a direct lookup. Construction, empty-range transitions and
  `last()` can allocate; seeks within an initialized mix still reuse its buffers.
  Item order and public signatures are unchanged.

- Simplify cursor allocation behavior: clones no longer preserve spare buffer
  capacity, and concat transitions create fresh child state. Repeated seeks within
  an initialized mix still reuse its buffers. Item order and public signatures are
  unchanged; clones and transitions may allocate on subsequent seeks.

- Remove the checkpoint worker example.

- Replace the custom benchmark harness and campaign runner with a small Criterion
  suite. Run `cargo bench --bench ordering`; use Criterion's saved baselines for
  comparisons.

- **Breaking:** remove `Seq::check` and `Seq::validate`; all sequence builders
  accept any `T` and defer configuration checks to `Order::new` or
  `Order::with_seed`, which require `T: Source`. `BoundsError` is reserved for
  order and cursor access.

- **Breaking:** replace `Seq::stride(step, offset)` and `Seq::Stride` with
  `skip(offset).step_by(step)` and `Seq::StepBy { step, inner }`. Remove
  `Seq::slice` and `Seq::Slice`; use `skip(start).take(len)` for a range.
  Each operation adds a level of configuration depth. `skip` rejects an offset
  past the end, while the former `stride` returned an empty sequence.
  Serialized `Stride` and `Slice` configurations must be rewritten using these
  operations; they are rejected during deserialization.

- **Breaking:** remove `Seq::shard`, `Seq::Shard` and `ErrorKind::InvalidShard`.
  Partition worker positions with `skip(index).step_by(count)` and check
  `index < count` in the caller. Each operation adds a level of configuration
  depth. Unlike the former `shard`, a skip past the end is an error; for a known
  length, use `skip(index.min(len))` when those workers should be empty.
  Serialized `Shard` configurations are rejected.

- **Breaking:** use `Item { source_ordinal, source, record_index }` for both
  `Order::get` and `Cursor` iteration. Remove `Order::get_indexed`,
  `Order::source_index`, `Cursor::indexed` and `IndexedCursor`. Every result
  identifies its source explicitly, including equal and zero-sized handles.

- **Breaking:** remove `Seq::Weighted`, `Seq::weighted`, `Seq::weighted_with`,
  `WeightedPart`, and the `InvalidWeight`, `ZeroWeights` and `EmptyWeightedPart`
  error kinds. Choose exact counts with `Seq::mix([a.cycle_to(a_count), b.cycle_to(b_count)])`
  or attach schedules with `Seq::mix`. Remove the exact floating-point quota
  allocator and its fixtures and benchmarks. Serialized `Weighted` configurations
  must be rewritten using explicit counts.

- **Breaking:** remove `Order::sources_mut`. Open or transform source handles with
  `Seq::map_sources` or `Seq::try_map_sources` before compilation.
- **Breaking:** make checked access the default and remove the corresponding `try_`
  aliases. `Order::get` returns `Option`; `Order::cursor` and cursor
  `seek`/`set_range` return `Result`. Invalid cursor
  operations preserve its state.

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
