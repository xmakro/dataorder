# dataorder

Shuffle and mix billions of records. Seek anywhere, then stream from there.

`dataorder` provides deterministic ordering for datasets too large to keep a full
index array in memory. Each lookup tells you **which dataset to read and the record's
index within it**, leaving record loading to you. Ordering memory grows with the
number of datasets and the sequence structure, not the number of records.

The default build has **no dependencies**.

- **Shuffle on demand.** Compute each shuffled index in **O(1) time on average**
  and O(1) space, without generating or storing the full permutation.
- **Jump into a mix.** Counting and binary searches locate the position within each
  input sequence, without replaying the preceding records. Seek cost depends on the
  input sequences and their schedules, rather than how far into the dataset you go.
- **Walk cheaply after seeking.** A mix chooses which sequence to read next with
  **O(log k) comparisons** for `k` input sequences. Pay for the seek once, then
  iterate from there.

Combine these operations with schedules, repeated epochs and worker sharding.
The same configuration, source metadata and seed reproduce the same order within the
crate's ordering compatibility policy. For restarts, also preserve the crate version
and worker settings.

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
    for item in order.cursor(resume..resume + 10)? {
        println!("record {} from a dataset of {} records", item.record_index, item.source);
    }
    assert_eq!(order.cursor(resume..)?.next(), order.get(resume));
    Ok(())
}
```

The position in an order differs from the index within a dataset: position
1,200,000,000 above selects one of the original billion records. `get(pos)` returns
`Option<Item>`; `iter()` visits the whole order, while `cursor(range)?` selects a range.
Both return a seekable `Cursor` yielding the same `Item` values in order. Each item
contains `source_ordinal`, `source` (a reference to the dataset handle), and `record_index`.

## Using your datasets

Implement `Source`, the dataset trait, for your own handle type. Only its length is
required. A stable `salt` distinguishes its shuffle from those of other datasets
with the same length and seed. Use a stable dataset name so moving its files does
not change its shuffle.

```rust
use dataorder::{Order, Seq, Source};

struct Dataset {
    name: &'static str,
    records: usize,
}

impl Source for Dataset {
    fn len(&self) -> usize { self.records }
    fn salt(&self) -> u64 { dataorder::salt(self.name) }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let web = Seq::source(Dataset { name: "web", records: 1000 }).shuffle(1);
    let code = Seq::source(Dataset { name: "code", records: 200 }).shuffle(2);
    let order = Order::new(Seq::mix([web, code]))?;

    for item in order.cursor(..10)? {
        // Use your own loader to read this record.
        println!("{}: record {}", item.source.name, item.record_index);
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
| Choose exact counts for each dataset | `Seq::mix([a.cycle_to(a_count), b.cycle_to(b_count), …])` |
| Control when a sequence contributes records | `Seq::mix`, with a `Schedule` for each part |
| Shuffle positions | `.shuffle(seed)` |
| Repeat whole epochs, reseeding existing shuffles | `.repeat(times)` |
| Repeat or truncate to an exact length | `.cycle_to(len)` |
| Keep a range of positions | `.skip(start).take(len)` |
| Keep every nth position from an offset | `.skip(offset).step_by(step)` |
| Assign every nth position to a worker | `.skip(worker_index).step_by(worker_count)` |

A mix uses every input element once, drawing more often from longer sequences.
Choose each dataset's count with `.cycle_to(count)`, which repeats or truncates the
sequence as needed. This gives a 75/25 mixture of one million records:

```rust
use dataorder::{Order, Seq};

let web = Seq::source(100_000);
let code = Seq::source(500_000);
let seq = Seq::mix([
    web.shuffle(1).cycle_to(750_000),
    code.shuffle(2).cycle_to(250_000),
]);
assert_eq!(Order::new(seq)?.len(), 1_000_000);
# Ok::<(), dataorder::Error>(())
```

Each input keeps its order; existing shuffles are reseeded for additional epochs.
Changing a part's count can change the mixed order's prefix. Keep the original
configuration and concatenate additional data when the existing prefix must stay fixed.

Schedules assign each part's elements positions on a **shared virtual clock** from
0 to 1. Part lengths control **how many** elements each part contributes.
Each curve is normalized independently, and the mix merges its virtual-time keys.
`Schedule::Uniform` has a constant rate on that clock, just like `delayed(0.0)`.

For example, this order draws 75% from one dataset and introduces the other at
virtual time 0.5:

```rust
use dataorder::{Order, Schedule, Seq};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let seq = Seq::mix([
        (Seq::source(300).shuffle(1).cycle_to(750), Schedule::Uniform),
        (Seq::source(100).shuffle(2).cycle_to(250), Schedule::delayed(0.5)),
    ]);
    let order = Order::new(seq)?;
    assert_eq!(order.len(), 1000);
    let first_delayed = order.iter().position(|item| item.source_ordinal == 1).unwrap();
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
apply. See [`Schedule`](https://docs.rs/dataorder/latest/dataorder/enum.Schedule.html)
for ramps, fade-outs, limits and the virtual-clock model.

**Migration from 0.3:** scheduled orders change in 0.4. `Uniform` no longer fills
other schedules' unused capacity, and start/full/fade/off values now refer to
virtual time. Revisit schedules that relied on output-percentage deadlines, and
resume existing checkpoints with their original crate version.

## Things to know

- **Composition matters.** Shuffle each input sequence before mixing for efficient
  iteration. To make a schedule span several epochs, repeat its input sequence;
  repeating the whole mix restarts its schedules each epoch.
- **Workers partition positions.** Apply `.skip(index).step_by(count)` to the
  completed sequence to divide its positions without overlap. Check `index < count`
  in your calling code. For a known sequence length `len`, use `skip(index.min(len))`
  if workers past its end should receive no positions. The global schedule is preserved
  collectively; each worker need not receive a balanced dataset mix. Two equal
  interleaved datasets split across two workers send one dataset to each worker,
  even when both inputs are shuffled. Shuffling the completed mix breaks that pattern
  but scatters its scheduled phases and adds a mix seek per element. Sharding the input sequences
  before mixing produces a different order and can change how virtual time maps
  to output positions. Shard lengths can differ by one; callers needing equal worker lengths
  must choose their truncation or padding policy.
- **Seeds are reproducible.** The same configuration and seed give the same order on
  supported platforms. `Order::with_seed` and `set_seed` reseed all existing shuffles.
  Adding an outer repeat preserves the entire first pass, including nested epochs;
  later outer passes reseed the shuffles inside it. See the
  [shuffle and repetition rules](https://docs.rs/dataorder/latest/dataorder/#shuffles-and-repetitions).
- **Bounds are checked.** `Order::new` reports invalid configurations with an error
  kind and node path. `take` and `skip` past the end are errors. `get`
  returns `None` for invalid positions. `cursor` and `reset` return
  `Result` for range operations. `step_by(0)` is an error when the order is built.
  Failed cursor operations leave their state unchanged.
- **Lengths must fit `usize`.** Every intermediate sequence must fit, even when a
  later `take`, `cycle_to`, or `step_by` would shorten it. Overflow is reported at the
  offending node.
- **Reuse cursors.** Use `iter()` for the whole order or `cursor(range)?` for a range.
  Use `reset(range)?` to replace the remaining range and reuse allocated buffers.
  Every range uses absolute order positions: `reset(pos..end)` keeps a chosen
  endpoint, `reset(pos..)` reads through the order's end, and `reset(..)` restarts
  the whole order. The previous range does not constrain the new one.
  Construction and moves to empty ranges can allocate. `last()` uses a direct lookup
  and can allocate independently of the cursor's buffers.
  Changing concat children creates fresh state. Clones copy current state without
  preserving spare capacity, so later seeks can allocate.
  `offset()` reports the next absolute position; `position(predicate)` is the usual
  consuming iterator search.

Every item's `source_ordinal` indexes `order.sources()` and distinguishes equal and
zero-sized handles. Ordinals follow the original configuration, including sources
whose nodes were removed during compilation. They are local to the order, not persistent dataset IDs.

Schedule diagnostics are part of `ErrorKind`: `InvalidSchedule { schedule, reason }`
includes a `ScheduleReason` for non-finite parameters, invalid breakpoints or
coefficient overflow. `TooSteep { len, peak_rate, limit }` carries the values behind
excessive steepness. Both `Error::kind()` and `Error::into_kind()` retain these details.

`Seq<T>` accepts any `T`, including unresolved dataset names or paths. All builders
defer configuration validation to `Order::new`, where `T: Source` is required.
Use `map` or `try_map` to resolve sources before compiling. Configurations support
up to 16 levels (`MAX_DEPTH`); tree operations and ordinary Rust cleanup recurse
with depth. Arbitrarily deep hand-built trees are unsupported.

`Seq` can be cloned, compared, hashed and mapped to another dataset handle type with
`map` or `try_map`. The optional `serde` feature adds configuration serialization. When
using JSON, also enable `serde_json/float_roundtrip` to preserve schedule parameters.
See the [feature documentation](https://docs.rs/dataorder/latest/dataorder/#feature-flags)
for details.

## Performance

Run the [Criterion](https://criterion-rs.github.io/book/) benchmarks:

```sh
cargo bench --bench ordering
cargo bench --bench ordering -- mix_100
```

The [benchmark suite](benches/ordering.rs) covers a billion-record shuffle, a mix
of 100 shuffled datasets, and a mix of 1,000 shuffled datasets with 20% scheduled.
Each measures construction, random lookup, fresh seek, reused cursor seek, and a
100,000-item walk. Sources are lengths only; no record I/O is included.
Configuration cloning is excluded from construction timing. Lookup positions are
precomputed. Both seek measurements include the first item; fresh seek also
includes cursor construction and destruction. Walk timing includes its initial
seek and reports throughput in elements per second.

The `selection` workloads measure slices over shuffles, repeats and mixes; strides
over sources, shuffles, repeats and mixes; and a mix whose parts combine these
selections. Each uses the same construction, lookup, seek and walk measurements.
Run them with `cargo bench --bench ordering -- selection`.

The `cursor_state` group measures cloning after entering a smaller concat child,
seeking the clone back, repeated seeks between children, and walking across children
and epochs. Clone setup is excluded from `cloned_seek_back`; that measurement includes
the seek, first item, and destruction of the clone.

Criterion handles warmup, sampling and comparison with the previous run. To keep
a baseline across changes, run these before and after the change on the same
machine with the same toolchain and benchmark workloads:

```sh
cargo bench --bench ordering -- --save-baseline before
# Make the change, then compare against the saved baseline.
cargo bench --bench ordering -- --baseline before
```

Results are stored under `target/criterion`. For a quick smoke test without timing,
run `cargo bench --bench ordering -- --test`. Allocation behavior is checked by
[regression tests](tests/cost.rs).

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

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
