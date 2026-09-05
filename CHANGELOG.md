# Changelog

Orders are part of the API: a release that changes the elements any configuration yields,
or the serialized form of a configuration, is a breaking change (a new minor version while
the crate is 0.x) and says so here. Golden tests in `tests/golden.rs` pin the orders.

## 0.1.0 (unreleased)

Pre-release review fixes (order changes relative to earlier development snapshots):

- Weighted shares now compare exact binary64 quotas and remainders, including terms below
  floating-point summation precision. Ordinary integer weights such as `[1, 1, 7]` over
  three positions now give `[1, 0, 2]`, respecting the lowest-index tie rule. This changes
  affected weighted orders. Common weights use `u128`; extreme ranges use bounded exact
  arithmetic without a dependency.
- Cursor state is built on the first draw, avoiding allocation for empty ranges and
  `count`. Clones preserve reserved seek capacity, and a tournament with one remaining
  part stops comparing against exhausted parts. Seeking a mix to position zero skips
  numerical rank estimation.
- Independent name/path salt vectors and golden orders protect the stable identity hash.
  JSON depth guidance now covers the extra nesting of mixed and weighted parts. Benchmark
  modes cover schedule phases, continuous exhausted tails, compilation, reused seeks and
  requested cursor-allocation bytes.

- Shuffles now use six Feistel rounds with the full `SplitMix64` finalizer and independently
  derived round keys. The previous seven multiply-only rounds retained strong adjacency
  patterns for some ordinary seeds and lengths. This changes seeded shuffle orders;
  public regressions, broader statistical checks and updated golden fingerprints cover it.
- A schedule quantile exactly on a flat cumulative-share interval now selects its left
  endpoint, matching the specified inverse. This changes affected mixed orders.
- Tournament buffers reserve room for parts that a backward seek can revive, so seeking
  a cursor first entered near the end no longer allocates to grow the tournament.
- Repeat and cycle documentation explains that adding an effective outer repetition can
  change the existing epochs of nested repeats. Mapping and cloning remain recursive;
  their documented stack limits now account for the source type's inline size.
- Seek benchmarks use deterministic random positions. The campaign harness resolves
  relative directories, rejects empty parsed runs and removes stale results for reused labels.
- Schedule construction preserves small slope contributions across narrow transitions;
  non-finite derived coefficients are rejected (`InvalidSampling` for an individual
  schedule, `SamplingOverflow` for the combined profile). The rising inverse avoids
  cancellation and is canonicalized to preserve monotonic keys. Seek correction and
  equal-key runs have bounded searches instead of unbounded loops or linear corrections.
- Nested slices remove unreachable boundary sources from shuffle salts. Weight allocation
  uses compensated summation, preserving small weights beside a dominant weight. These
  fixes can change affected schedules, shuffles and weighted counts; they left the
  then-existing golden examples unchanged, and regressions pin the newly corrected cases.
- Compilation moves source values out before recursing, so large inline sources respect
  the depth limit without overflowing a thread stack. Empty mix parts retain their source
  handles and validation errors but no longer occupy runtime cursor state.
- JSON users must enable `serde_json/float_roundtrip` to preserve weights and breakpoints;
  the configuration's serialized shape is unchanged. Cost documentation includes seek
  replay, concat search and average cycle-walking cost, and explains the schedule
  feasibility constraint when sharding individual parts.

First release: `Seq` (sources, concat, mix with sampling schedules that start, ramp, fade and
stop, weighted mix, shuffle, repeat, cycle, skip, take, stride), `Source` with a length and a
salt (`salt` and `salt_path` derive one), `Order` with random access, `source_index`,
reseeding and seekable, re-rangeable cursors, the `serde` feature.
