# dataorder

Shuffle and mix billions of records. Seek anywhere, then stream from there.

`dataorder` provides deterministic ordering for datasets too large to keep a full
index array in memory. Each lookup tells you **which dataset to read and the record's
index within it**, leaving record loading to you. Ordering memory grows with the
number of datasets and the sequence structure, not the number of records.

In the measurements below, shuffled random access takes about **31 ns**, a random
lookup in a mix of 100 datasets about **2.7 µs**, and walking that mix after
a seek about **35 ns per item**. See the [performance highlights](#performance).

The default build has **no dependencies**.

- **Shuffle on demand.** Compute each shuffled index in **O(1) time on average**
  and O(1) space, without generating or storing the full permutation.
- **Jump into a mix.** Counting and binary searches locate the position within each
  input sequence, without replaying the preceding records. Seek cost depends on the
  input sequences and their schedules, rather than how far into the dataset you go.
- **Walk cheaply after seeking.** A mix chooses which sequence to read next with
  **O(log k) comparisons** for `k` input sequences. Pay for the seek once, then
  iterate from there.

Combine these operations with sampling schedules, repeated epochs and worker sharding.
The same configuration and seed reproduce the same order, including after a restart.

[API documentation](https://docs.rs/dataorder) · [Runnable example](examples/demo.rs) ·
[Performance](#performance)

## Getting started

```toml
[dependencies]
dataorder = "0.2"
```

Start with `Seq::source(dataset)`, add ordering operations, then validate and prepare
the sequence with `Order::new`. A `usize` can stand in for a dataset when you only
need its length:

```rust
use dataorder::{Order, Seq};

fn main() -> Result<(), dataorder::Error> {
    // Two passes over a billion records, with a fresh shuffle for each pass.
    let seq = Seq::source(1_000_000_000).shuffle(42).repeat(2);
    let order = Order::new(seq)?;
    assert_eq!(order.len(), 2_000_000_000);

    // Resume deep into the second epoch without replaying the earlier positions.
    let resume = 1_200_000_000;
    for (dataset_len, index) in order.iter(resume..resume + 10) {
        println!("record {index} from a dataset of {dataset_len} records");
    }
    assert_eq!(order.iter(resume..).next(), Some(order.get(resume)));
    Ok(())
}
```

The position in an order differs from the index within a dataset: position
1,200,000,000 above selects one of the original billion records. `get(pos)` returns
a dataset handle and a record index; `iter(range)` returns the same pairs in order.

## Using your datasets

Implement `Source`, the dataset trait, for your own handle type. Only its length is
required. A stable `salt` distinguishes its shuffle from those of other datasets
with the same length and seed.

```rust
use dataorder::{Order, Seq, Source};

struct Dataset {
    path: &'static str,
    records: usize,
}

impl Source for Dataset {
    fn len(&self) -> usize { self.records }
    fn salt(&self) -> u64 { dataorder::salt(self.path) }
}

fn main() -> Result<(), dataorder::Error> {
    let web = Seq::source(Dataset { path: "web.bin", records: 1000 }).shuffle(1);
    let code = Seq::source(Dataset { path: "code.bin", records: 200 }).shuffle(2);
    let order = Order::new(Seq::mix([web, code]))?;

    for (dataset, index) in order.iter(..10) {
        // Use your own loader to read this record.
        println!("{}: record {index}", dataset.path);
    }
    Ok(())
}
```

`Order` owns your dataset handles and yields references to them. Slices, arrays and
vectors also implement `Source`, as do references to any type implementing the trait.
Keep dataset lengths stable after building an order: it uses the lengths recorded
at construction.

## Combining sequences

Each input sequence can represent a single dataset or combine several datasets,
so you can nest mixes and concatenations.

| To… | Use |
| --- | --- |
| Read sequences one after another | `Seq::concat(sequences)` |
| Interleave sequences, preserving the order within each | `Seq::mix(sequences)` |
| Choose a total length and relative proportions | `Seq::weighted(total, [(seq, weight), …])` |
| Control when a sequence contributes records | `Seq::mix_with` or `Seq::weighted_with`, with a `Sampling` schedule |
| Shuffle positions | `.shuffle(seed)` |
| Repeat whole epochs, reseeding existing shuffles | `.repeat(times)` |
| Repeat or truncate to an exact length | `.cycle(len)` |
| Keep a range or every nth position | `.slice(range)` or `.stride(step, offset)` |
| Assign every nth position to a worker | `.shard(worker_count, worker_index)` |

A plain mix uses every input element once, drawing more often from longer sequences.
A weighted mix repeats or truncates each input sequence to its assigned count; the
counts sum exactly to `total`. Each input keeps its order unless you add `.shuffle(seed)`.
Largest-remainder rounding can reduce a part's count when `total` grows: weights
`[5, 3, 1]` receive `[2, 1, 1]` at total 4 and `[3, 2, 0]` at total 5. Changing the
total or weights need not preserve the prefix. Keep the original configuration and
concatenate additional data when the existing prefix must stay fixed.

Schedules control **when** elements appear, while lengths or weights control **how
many** appear. For example, this order draws 75% from one dataset and introduces
the other around halfway through the run:

```rust
use dataorder::{Order, Sampling, Seq};

fn main() -> Result<(), dataorder::Error> {
    let seq = Seq::weighted_with(1000, [
        (Seq::source(300).shuffle(1), 3.0, Sampling::Uniform),
        (Seq::source(100).shuffle(2), 1.0, Sampling::delayed(0.5)),
    ]);
    let order = Order::new(seq)?;
    assert_eq!(order.len(), 1000);
    Ok(())
}
```

`Sampling::Uniform` spreads a sequence's records across the space left by the other
schedules. A schedule that requires more records than can fit in an interval is rejected. See
[`Sampling`](https://docs.rs/dataorder/latest/dataorder/enum.Sampling.html) for ramps,
fade-outs and rounding at schedule boundaries.

## Things to know

- **Composition matters.** Shuffle each input sequence before mixing for efficient
  iteration. To make a schedule span several epochs, repeat its input sequence;
  repeating the whole mix restarts its schedules each epoch.
- **Workers partition positions.** Apply `.shard(count, index)` to the completed
  sequence to divide its positions without overlap. The global schedule is preserved
  collectively; each worker need not receive a balanced dataset mix. Two equal
  interleaved datasets split across two workers send one dataset to each worker,
  even when both inputs are shuffled. Shuffling the completed mix breaks that pattern
  but scatters its scheduled phases and adds a mix seek per element. Sharding the input sequences
  before mixing produces a different order and can make an otherwise valid schedule
  infeasible. Shard lengths can differ by one; callers needing equal worker lengths
  must choose their truncation or padding policy.
- **Seeds are reproducible.** The same configuration and seed give the same order on
  supported platforms. `Order::with_seed` and `set_seed` reseed all existing shuffles.
  Adding an outer repeat can change later epochs of repeats inside it; see the
  [shuffle and repetition rules](https://docs.rs/dataorder/latest/dataorder/#shuffles-and-repetitions).
- **Bounds are checked.** `Order::new` reports invalid configurations with an error
  kind and node path. `take` and `skip` past the end are errors. Accessing an invalid
  position with `get`, or an invalid range with `iter`, panics. Use `try_get` for an
  optional result and `try_iter`, `try_seek`, `try_set_range`, or `Seq::try_slice` for
  fallible range operations. `Seq::try_shard` checks worker counts and indices.
  Failed cursor operations leave their state unchanged.
- **Reuse cursors.** `iter` is best for consecutive positions. For repeated seeks or
  ranges, reuse its `Cursor` with `seek` or `set_range` to reuse allocated buffers.
  `offset()` reports the next absolute position; `position(predicate)` is the usual
  consuming iterator search.

Use `get_indexed(pos)` or `iter(range).indexed()` to obtain
`(source_ordinal, dataset, record_index)`. The ordinal indexes `order.sources()` and
distinguishes equal and zero-sized handles. `Order::prepare(seq, seed)` returns an
order together with a report of exact weighted quotas and compiled node lengths,
without enumerating records. The report distinguishes original configuration paths
for quotas from paths in the simplified compiled tree. Ordinary constructors do not
collect it. `Seq::check` performs compilation to validate a borrowed configuration;
calling it before `Order::new` repeats that work.
For configuration trees of unknown depth, use consuming `Seq::validate` to return
the tree on success and dispose of it safely on error. After a borrowed check rejects
a deep tree, call `Seq::dispose`; ordinary enum destruction is recursive.

`Seq` can be cloned, compared, hashed and mapped to another dataset handle type with
`map` or `try_map`. The optional `serde` feature adds configuration serialization. When
using JSON, also enable `serde_json/float_roundtrip` to preserve weights and schedules.
See the [feature documentation](https://docs.rs/dataorder/latest/dataorder/#feature-flags)
for details.

## Performance

Measured on macOS ARM64 with Rust 1.98.1, release build, on 2026-09-05.
Each cell is the median [minimum..maximum] of five samples after warmup and batch
calibration to at least 20 ms. Timings exclude record I/O.

| Order | Positions | Random lookup | Seek + first item | Walk / item |
| --- | --- | --- | --- | --- |
| Shuffled dataset | 1 billion | 31.3 [31.2..31.3] ns | 0.046 [0.046..0.047] µs | 18.3 [18.2..18.3] ns |
| Mix of 100 shuffled datasets | 100 million | 2.71 [2.71..2.71] µs | 3.109 [3.108..3.111] µs | 35.2 [35.1..35.2] ns |
| Mix of 1,000 shuffled datasets, 20% scheduled | 100 million | 64.61 [64.53..64.65] µs | 67.163 [67.102..67.269] µs | 63.4 [63.4..63.4] ns |
| Nested mix of 1,100 shuffled datasets, 2–4 epochs | 4.1 billion | 44.74 [44.72..44.82] µs | 47.780 [47.743..47.799] µs | 84.1 [84.1..84.2] ns |

Random lookup measures `get(pos)`. Seek measures `iter(pos..).next()`, including
cursor construction and destruction. Walk measures batches of five million items,
including the initial seek. Random positions are precomputed outside timing; reading
the positions, loop control and optimization barriers remain inside. Phase mode uses
positions within each named phase window. The benchmark runs on one thread without
CPU affinity. Run it locally with:

```sh
cargo run --release --example bench
```

See [the benchmark code](examples/bench.rs) for the measured configurations and
the [cost model](https://docs.rs/dataorder/latest/dataorder/#cost) for how composition
affects performance.

## Development

Requires Rust 1.89 or newer.

```sh
cargo run --release --example demo
cargo test --locked --all-features
cargo test --locked --all-features --examples
cargo doc --locked --no-deps --all-features
python3 tests/fixtures/generate_weight_oracle.py --check
python3 tests/fixtures/generate_schedule_oracle.py --check
```

The README's Rust examples are tested with the crate's documentation examples.
The fixture generators use Python's standard library and fixed seeds. Weight quotas
use exact integer ratios; schedule expectations use rational CDFs and 96-digit
inverse calculations independent of the Rust implementation. Omit `--check` to
regenerate the fixtures after changing a generator. CI checks both generated files.

For repeated benchmark runs and comparisons, run
`cargo run --release --example bench_campaign -- --help`.
Use `compare before /path/to/before after /path/to/after all` to run six rounds,
alternating which revision runs first. Both checkouts must contain identical
`examples/bench.rs` and `examples/support/measurements.rs`; copy the harness into
the older checkout when comparing implementations. The runner checks harness and
workload fingerprints, complete row sets, and measurement columns before saving.
It snapshots each executable so shared build directories cannot replace a revision
between rounds. `table 1 before after` selects labels and the walk column.

Campaigns run unpinned by default. Set `BENCH_CORE=N` to request CPU affinity through
`taskset`; unavailable affinity is reported before building. Results retain each raw
run and its five calibrated samples per metric, revision and working-tree status,
compiler, machine information and affinity. Tables show the median of all retained
samples and their minimum-to-maximum range. Legacy results remain readable one label
at a time, with their original statistic. Unavailable machine fields are stored as
`null`. Concurrent campaigns merge their results under a file lock and replace the
results file atomically.

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
