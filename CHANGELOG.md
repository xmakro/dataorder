# Changelog

## Unreleased (0.4.0)

This release reworks the API around explicit items, checked access and a single
order seed, and changes how shuffles and schedules are computed. Every order that
contains a shuffle or a scheduled mix differs from 0.3.0; resume such orders with the
crate version that produced them.

- **Breaking:** shuffles take no seed of their own, and `Seq::Shuffle` is gone.
  `shuffle()` replaces `shuffle(seed)` and builds a `Seq::Repeat` with one shuffled
  pass; select the run with `Order::with_seed` or `Order::set_seed`. A permutation
  depends on that order seed and on the input's configuration salt, which
  follows the original configuration: every source contributes its `Source::salt` and
  original length, even when empty or excluded by a selection; plain repetitions and
  selections pass the salt through; each shuffled layer advances it once, so nested
  shuffles use distinct keys; and concatenations combine child salts independently of
  grouping. Key derivation hashes the order seed, pass number and salt in one step.

- **Breaking:** `repeat(times)` and `cycle_to(len)` (renamed from `cycle`) preserve
  their input's record order on every pass, including nested shuffles, and there is no
  repetition context: enclosing repeats never reseed inner shuffles, and
  `x.repeat(3).repeat(2)` orders like `x.repeat(6)`. Add `repeat_shuffled(times)` and
  `cycle_to_shuffled(len)`, which permute the immediate input separately on each pass,
  including the first. `Seq::Repeat` and `Seq::Cycle` gain a `shuffled` field.

- **Breaking:** a shuffle or shuffled repetition rejects any mix in its input
  configuration, including empty or single-part mixes and mixes under other operations,
  with `ErrorKind::ShuffleContainsMix` at the enclosing shuffle. Shuffle each input
  before mixing.

- **Breaking:** schedules are independent curves on a shared virtual clock. Rename
  `Sampling` to `Schedule`, `MixPart.sampling` to `schedule` and `fading` to `fade`.
  `Uniform` is constant in virtual time, like `delayed(0.0)`; it no longer fills the
  capacity other schedules leave, and there is no capacity check, so schedules may
  overlap or leave gaps. Breakpoints are virtual times rather than output fractions,
  and merged ramps are generally nonlinear in output position. Remove `DelayedLinear`;
  `delayed` and `ramp` return `Trapezoid` with `fade` and `off` at 1. Seeks count the
  elements below a virtual time and bisect. Remove `Eq` and `Hash` from `Seq`,
  `MixPart` and `Schedule`; `PartialEq` uses ordinary `f64` equality.

- **Breaking:** remove weighted mixes: `Seq::Weighted`, `Seq::weighted`,
  `Seq::weighted_with`, `WeightedPart` and their error kinds. Choose exact counts with
  `Seq::mix([a.cycle_to(a_count), b.cycle_to(b_count)])`. Merge `mix_with` into `mix`,
  which accepts sequences, `(seq, schedule)` pairs or `MixPart` values.

- **Breaking:** replace `stride`, `slice` and `shard`, their `Seq` variants and their
  `try_` forms with `skip`, `take` and `step_by` (`Seq::StepBy`). Partition workers
  with `skip(index).step_by(count)`. Skipping past the end is an error, where `stride`
  and `shard` produced empty sequences. A stride validates its input before its step.

- **Breaking:** every result is an `Item { source_ordinal, source, record_index }`,
  and `Order::get` returns `Option<Item>`. Remove `get_indexed`, `source_index`,
  `sources_mut`, the `try_` access aliases, `Cursor::indexed` and `IndexedCursor`.
  Item equality compares the ordinal and record index before the source value.
  Rename `Seq::map` and `try_map` to `map_sources` and `try_map_sources`; use them
  to open source handles before compiling.

- **Breaking:** `Order::iter()` returns a `Cursor` over the whole order and
  `Order::cursor(range)` a checked `Result<Cursor, BoundsError>`. `Cursor::reset(range)`
  replaces `seek`, `set_range` and their `try_` forms, using absolute order positions;
  a failed reset leaves the cursor unchanged. Remove `Cursor::remaining`; use `len()`.
  Remove `TryFrom<Seq<T>> for Order<T>`; use `Order::new`.
  `BoundsError` gains `StartOutOfBounds`, merges `StartOverflow` and `EndOverflow`
  into `Overflow`, and loses `InvalidShard` and `SeekOutOfBounds`. Cursors are
  positioned when constructed, `last()` uses a direct lookup, clones keep no spare
  capacity and entering another concat child creates fresh state; seeks within an
  initialized mix still reuse its buffers.

- **Breaking:** remove `Order::prepare` with its `Preparation`, `PreparedNode`,
  `PreparedKind`, `PreparedParameters`, `PreparedSource`, `PreparedMix`,
  `SamplingDiagnostics` and `WeightedAllocation` report types, and `Seq::check`,
  `validate` and `dispose`. Builders accept any `T` and defer validation to
  `Order::new`, which requires `T: Source`.

- **Breaking:** schedule diagnostics live in `ErrorKind`: `InvalidSchedule { schedule }`
  and `ScheduleTooSteep { len, peak_rate, limit }` replace `InvalidSampling`,
  `TooSteep`, `SamplingDetail` and `Error::sampling_detail`. An invalid schedule has a
  non-finite breakpoint, breakpoints out of range or order, no time at a positive rate,
  or breakpoints too close together for finite coefficients; the message says so and
  the schedule itself is in the error. Remove `Error::into_kind`, which `kind`
  covers. Remove `Overcommitted`, `SamplingOverflow`, `InvalidWeight`, `ZeroWeights`,
  `EmptyWeightedPart`, `OrderTooLong`, `TooManySources`, `TooManyMixParts` and
  `TooDeep`.

- **Breaking:** lengths, positions and counts are `usize` throughout, including the
  `len` fields of `ErrorKind`, and every sequence node must fit, including
  intermediates a parent later truncates; overflow is `LengthOverflow` at that node.
  Sources and mix parts are indexed with `usize`. Remove `MAX_DEPTH`: configurations of any depth
  compile, though compilation, mapping, cloning, comparison and destruction still
  recurse with depth.

- **Breaking:** serialized configurations change accordingly. A shuffle is a `Repeat`
  with `times: 1` and `shuffled: true`, and `Repeat` and `Cycle` accept a `shuffled`
  field that defaults to false; mix parts use `schedule`; schedules are `Uniform` or
  `Trapezoid`; `StepBy { step, inner }` is new. `Shuffle`, `Weighted`, `Stride`,
  `Slice`, `Shard`, `DelayedLinear` and `sampling` fields are rejected rather than
  reinterpreted.

- **Breaking:** remove `salt_path`; pass the dataset identity's bytes to `salt`.
  Rename `ORDERING_VERSION` to `CRATE_VERSION`.

- Store each shuffled node's first-pass key in the compiled order and derive later
  passes' keys only when a cursor enters them; `set_seed` re-derives the stored keys
  in time proportional to the number of shuffled nodes. Keep concat and repetition
  cursor state in dedicated structs, compile a selection of a concatenation as a unit
  stride over it, and select a profile's segment with a short scan instead of a
  cached hint.

- Replace the custom benchmark harness and campaign runner with a Criterion suite
  (`cargo bench --bench ordering`), remove the checkpoint worker example, derive the
  examples' salts from dataset names, and give each documented rule one home in the
  crate, `Schedule` and `Seq` docs.

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
