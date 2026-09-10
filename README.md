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
The same configuration, source metadata and seed reproduce the same order within the
crate's ordering compatibility policy. For restarts, also preserve the crate version
and worker settings; see the [checkpoint example](examples/checkpoint.rs).

[API documentation](https://docs.rs/dataorder) · [Runnable example](examples/demo.rs) ·
[Performance](#performance)

## Getting started

```toml
[dependencies]
dataorder = "0.4"
```

Start with `Seq::source(dataset)`, add ordering operations, then validate and prepare
the sequence with `Order::new`. A `usize` can stand in for a dataset when you only
need its length:

```rust
use dataorder::{Order, Seq};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Two passes over a billion records, with a fresh shuffle for each pass.
    let seq = Seq::source(1_000_000_000).shuffle(42).repeat(2);
    let order = Order::new(seq)?;
    assert_eq!(order.len(), 2_000_000_000);

    // Resume deep into the second epoch without replaying the earlier positions.
    let resume = 1_200_000_000;
    for item in order.iter(resume..resume + 10)? {
        println!("record {} from a dataset of {} records", item.record_index, item.source);
    }
    assert_eq!(order.iter(resume..)?.next(), order.get(resume));
    Ok(())
}
```

The position in an order differs from the index within a dataset: position
1,200,000,000 above selects one of the original billion records. `get(pos)` returns
`Option<Item>`; `iter(range)?` yields the same `Item` values in order. Each item
contains `source_ordinal`, `source` (a reference to the dataset handle), and `record_index`.

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let web = Seq::source(Dataset { path: "web.bin", records: 1000 }).shuffle(1);
    let code = Seq::source(Dataset { path: "code.bin", records: 200 }).shuffle(2);
    let order = Order::new(Seq::mix([web, code]))?;

    for item in order.iter(..10)? {
        // Use your own loader to read this record.
        println!("{}: record {}", item.source.path, item.record_index);
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
| Choose exact counts for each dataset | `Seq::mix([a.cycle(a_count), b.cycle(b_count), …])` |
| Control when a sequence contributes records | `Seq::mix_with`, with a `Sampling` schedule |
| Shuffle positions | `.shuffle(seed)` |
| Repeat whole epochs, reseeding existing shuffles | `.repeat(times)` |
| Repeat or truncate to an exact length | `.cycle(len)` |
| Keep a range or every nth position | `.slice(range)` or `.stride(step, offset)` |
| Assign every nth position to a worker | `.shard(worker_count, worker_index)` |

A mix uses every input element once, drawing more often from longer sequences.
Choose each dataset's count with `.cycle(count)`, which repeats or truncates the
sequence as needed. This gives a 75/25 mixture of one million records:

```rust
use dataorder::Seq;

let web = Seq::source(100_000);
let code = Seq::source(500_000);
let seq = Seq::mix([
    web.shuffle(1).cycle(750_000),
    code.shuffle(2).cycle(250_000),
]);
assert_eq!(seq.check(), Ok(1_000_000));
```

Each input keeps its order; existing shuffles are reseeded for additional epochs.
Changing a part's count can change the mixed order's prefix. Keep the original
configuration and concatenate additional data when the existing prefix must stay fixed.

Schedules assign each part's elements positions on a **shared virtual clock** from
0 to 1. Part lengths control **how many** elements each part contributes.
Each curve is normalized independently, and the mix merges its virtual-time keys.
`Sampling::Uniform` has a constant rate on that clock, just like `delayed(0.0)`.

For example, this order draws 75% from one dataset and introduces the other at
virtual time 0.5:

```rust
use dataorder::{Order, Sampling, Seq};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let seq = Seq::mix_with([
        (Seq::source(300).shuffle(1).cycle(750), Sampling::Uniform),
        (Seq::source(100).shuffle(2).cycle(250), Sampling::delayed(0.5)),
    ]);
    let order = Order::new(seq)?;
    assert_eq!(order.len(), 1000);
    let first_delayed = order.iter(..)?.position(|item| item.source_ordinal == 1).unwrap();
    // Half of the 750 uniform items have appeared by virtual time 0.5.
    // The delayed source starts around output position 375, not 500.
    assert!((374..=377).contains(&first_delayed));
    Ok(())
}
```

**Virtual time is not output progress.** With normalized cumulative curves `F_i`,
part counts `n_i` and total count `N`, the continuous model reaches output progress
`sum(n_i * F_i(t)) / N` at virtual time `t`. A part's fraction of the output rate is
`n_i * rate_i(t) / sum(n_j * rate_j(t))` when the combined rate is positive.
Changing another part's count or schedule can therefore move its actual start or
end position. Linear ramps remain smooth in virtual time but generally become
nonlinear against output progress; constant curves adapt in the same way.
Discrete items approximate the curves, while counts and source-local order remain exact.

Schedules can overlap or leave gaps, including mixes with no uniform parts. Clock
intervals with no active parts produce no output. There is no shared capacity check
or special filler source. Individual breakpoint and numerical-resolution limits still
apply. See [`Sampling`](https://docs.rs/dataorder/latest/dataorder/enum.Sampling.html)
for ramps, fade-outs, limits and the virtual-clock model.

**Migration from 0.3:** scheduled orders change in 0.4. `Uniform` no longer fills
other schedules' unused capacity, and start/full/fade/off values now refer to
virtual time. Revisit schedules that relied on output-percentage deadlines, and
resume existing checkpoints with their original crate version.

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
  before mixing produces a different order and can change how virtual time maps
  to output positions. Shard lengths can differ by one; callers needing equal worker lengths
  must choose their truncation or padding policy.
- **Seeds are reproducible.** The same configuration and seed give the same order on
  supported platforms. `Order::with_seed` and `set_seed` reseed all existing shuffles.
  Adding an outer repeat can change later epochs of repeats inside it; see the
  [shuffle and repetition rules](https://docs.rs/dataorder/latest/dataorder/#shuffles-and-repetitions).
- **Bounds are checked.** `Order::new` reports invalid configurations with an error
  kind and node path. `take` and `skip` past the end are errors. `get`
  returns `None` for invalid positions. `iter`, `seek`, `set_range`,
  and `Seq::slice` return `Result` for range operations. `Seq::shard` returns
  `Result` after checking worker counts and indices.
  Failed cursor operations leave their state unchanged.
- **Reuse cursors.** `iter` is best for consecutive positions. For repeated seeks or
  ranges, reuse its `Cursor` with `seek` or `set_range` to reuse allocated buffers.
  `offset()` reports the next absolute position; `position(predicate)` is the usual
  consuming iterator search.

Every item's `source_ordinal` indexes `order.sources()` and distinguishes equal and
zero-sized handles. Ordinals follow the original configuration, including sources
whose nodes were removed during compilation. They are local to the order, not persistent dataset IDs.

Invalid schedules expose further context through `Error::sampling_detail()`:
non-finite parameters, invalid breakpoints, coefficient overflow, or the length,
peak rate and limit behind excessive steepness.

`Seq::check` performs compilation to validate a borrowed configuration;
calling it before `Order::new` repeats that work.
Use consuming `Seq::validate` to return the tree on success. Configurations support
up to 16 levels (`MAX_DEPTH`); tree operations and ordinary Rust cleanup recurse
with depth. Arbitrarily deep hand-built trees are unsupported.

`Seq` can be cloned, compared, hashed and mapped to another dataset handle type with
`map` or `try_map`. The optional `serde` feature adds configuration serialization. When
using JSON, also enable `serde_json/float_roundtrip` to preserve schedule parameters.
See the [feature documentation](https://docs.rs/dataorder/latest/dataorder/#feature-flags)
for details.

For a complete restart pattern, run `cargo run --example checkpoint --features serde`.
The example stores the whole configuration as its identity, immutable source versions,
lengths and salts, seeds, shard count/index, the next worker-local offset, and checkpoint
and crate versions, using `dataorder::ORDERING_VERSION` to identify the linked crate.
Restore compares these against independently loaded current metadata and rejects
mismatches. Processing seeks once per batch and commits each successful callback,
including after resume. Worker creation checks that its checkpoint can be parsed by
the restore format: JSON's depth limit can reject configurations that the crate accepts.
Applications should persist committed work rather than prefetched positions.

## Performance

Historical measurements below predate the 0.4 virtual-clock change; scheduled-mix
construction and seek timings do not describe the new engine. They were measured
on macOS ARM64 with Rust 1.98.1, release build, on 2026-09-05.
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
positions within each named output-progress window. The benchmark runs on one
thread without CPU affinity. Run it locally with:

```sh
cargo run --release --example bench
```

See [the benchmark code](examples/bench.rs) for the measured configurations and
the [cost model](https://docs.rs/dataorder/latest/dataorder/#cost) for how composition
affects performance.

Cursor memory scales with the number of live parts and independent workers, as well
as their reached child states. Boxing the mix state reduces storage reserved for every
child slot, at the cost of one allocation per active mix. On this macOS ARM64 host,
a 100,000-source cursor retained about 23.9 MB after warmup, down from 31.9 MB before
that change. In a six-round alternating comparison on the same host, the five-source
mix's fresh seek increased from 0.305 to 0.350 µs; its walk stayed about 11.9 ns/item.
The 100-source shuffled mix walked at 24.1 vs 25.0 ns/item. This is a memory/latency
tradeoff, not a general speedup. CPU-model access was unavailable in the sandbox;
this comparison explicitly allowed that missing field and used matching compiler
settings. Six lifecycle rounds on 2026-09-06 measured:

| Mix sources | Worker cursors | Retained bytes | Peak live bytes |
| --- | --- | --- | --- |
| 10,000 | 1 | 2,546,816 | 2,546,816 |
| 10,000 | 8 | 20,374,528 | 20,374,528 |
| 10,000 | 32 | 81,498,112 | 81,498,112 |
| 100,000 | 1 | 23,891,840 | 23,891,840 |

These allocator measurements include the cursor vector and cursor-owned allocations,
exclude the shared order and allocator overhead, and are not RSS. Peak live bytes count
Rust allocation layouts; they cannot measure a system allocator's internal realloc copy.
`bench --lifecycle` reports cumulative requests, retained bytes and peak live bytes.
It also covers large scheduled mixes with minority sources.

Concat cursors recycle compatible child buffers across boundaries, seeks and repeated
epochs. Their retained capacities can reflect larger previously visited children,
but they do not cache every visited child. Switching node kinds or dropping nested
child states can still allocate on a later visit. Lifecycle benchmarks also cover
repeated concatenations and compilation of many identity transforms.

## Development

Requires Rust 1.89 or newer.

```sh
cargo run --release --example demo
cargo test --locked --all-features
cargo test --locked --all-features --examples
cargo doc --locked --no-deps --all-features
python3 tests/fixtures/generate_schedule_oracle.py --check
```

The README's Rust examples are tested with the crate's documentation examples.
The schedule fixture generator uses Python's standard library and a fixed seed.
Schedule expectations use rational CDFs and 96-digit
inverse calculations independent of the Rust implementation. Omit `--check` to
regenerate the fixtures after changing the generator. The fixtures include
independent overlapping schedules without uniform parts, gaps, interacting ramps,
reordered minorities, nearby boundaries, exact ties and lengths up to `MAX_MIX_LEN`.
CI checks the generated file. Small oracle fixtures check complete continuous walks; large fixtures include
independently computed contiguous windows. Stateful cursor tests combine seeks, range
changes, clones, skips, exhaustion and failed operations; failures print a reproducible
`DATAORDER_STATE_SEED` and minimize the operation history.

For repeated benchmark runs and comparisons, run
`cargo run --release --example bench_campaign -- --help`.
Use `compare before /path/to/before after /path/to/after all` to run six rounds,
alternating which revision runs first. Both checkouts must contain identical
`examples/bench.rs` and `examples/support/measurements.rs`; copy the harness into
the older checkout when comparing implementations. The runner checks harness and
workload fingerprints, complete row sets, and measurement columns before saving.
Each compilation uses its own target and intermediate directories, overriding shared
Cargo output directories from the start of the build. The runner records an executable
fingerprint, Cargo-selected compiler version and target configuration, and the observed
profile and compiler arguments for the library and benchmark. `table 1 before after`
selects labels and the walk column; columns 5, 6 and 7 select requested, retained and
peak cursor bytes.

Campaigns run unpinned by default. Set `BENCH_CORE=N` to request CPU affinity through
`taskset`; unavailable affinity is reported before building. Results retain each raw
run and its five calibrated samples per metric, revision and working-tree status,
compiler, machine information and affinity. Tables show the median of all retained
samples and their minimum-to-maximum range. Legacy results remain readable one label
at a time, with their original statistic. Unavailable machine fields are stored as
`null`. Comparisons reject differing CPU, system, affinity or effective build settings,
and missing environment provenance. Use `--allow-environment-differences` when such a
difference is intentional; the runner prints the differences. This does not bypass
harness or workload checks. Machine metadata cannot account for thermal state or other
processes, so run timing campaigns on an otherwise idle machine.

Compiler verification trusts the toolchain independently resolved by `rustup which
rustc` and checks that Cargo invokes that compiler directly. Compiler wrappers, custom
launchers (including those named `rustc`), and builds without a resolvable rustup
compiler require `--allow-environment-differences`: Cargo's displayed arguments cannot
establish which flags a launcher actually passed to the compiler. These builds are
recorded as unverified, even when their displayed settings match. Comparisons with
older provenance lacking compiler path verification also require the override.

`BENCH_TIMEOUT_SECS` sets a positive deadline for each external command (default:
1800 seconds). Process trees are terminated on timeouts and command failures. Failed commands retain their
stdout and stderr under `target/bench-diagnostics`; the error prints the directory.
Incomplete campaigns do not replace saved comparison results.

Concurrent campaigns commit both comparison labels under one lock and atomically replace
the results file. Each command prints its own committed snapshot. A later table rejects
paired labels if one has subsequently been overwritten by a different campaign.

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
