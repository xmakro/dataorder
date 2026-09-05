# dataorder

Shuffle and mix billions of records. Seek anywhere, then stream from there.

`dataorder` provides deterministic ordering for datasets too large to keep a full
index array in memory. Each lookup computes a **source and an index within that
source**, leaving record loading to you. Ordering memory grows with the sources and
sequence structure, not the number of records.

- **Shuffle on demand.** Compute each shuffled index in **O(1) time on average**
  and O(1) space, without generating or storing the full permutation.
- **Jump into a mix.** Counting and binary searches locate the position within each
  part, without replaying the preceding records. Seek cost depends on the parts and
  their schedules, rather than how far into the dataset you go.
- **Walk cheaply after seeking.** A mix keeps a tournament tree, choosing each next
  part with **O(log k) comparisons** for `k` parts. Pay for the seek once, then
  iterate from there.

Combine these operations with sampling schedules, repeated epochs and worker sharding.
The same configuration and seed reproduce the same order, including after a restart.

[API documentation](https://docs.rs/dataorder) · [Runnable example](examples/demo.rs) ·
[Benchmarks](docs/benchmarks.md)

## Performance

Current implementation on an Apple M2 Pro, release build, minimum of two runs:

| Order | Positions | Seek + first item | Walk / item |
| --- | --- | --- | --- |
| Shuffled source | 1 billion | 0.04 µs | 14.2 ns |
| Mix of 100 shuffled sources | 100 million | 2.20 µs | 26.7 ns |
| Mix of 1,000 shuffled sources, 20% scheduled | 100 million | 47.40 µs | 45.1 ns |
| Nested mix of 1,100 shuffled sources, 2–4 epochs | 4.1 billion | 33.86 µs | 62.4 ns |

Seek includes creating a cursor and returning the first item. Walk averages a
five-million-item range, including its initial seek. Timings measure ordering and
exclude record I/O. See the [full results and methodology](docs/benchmarks.md#current-measurements).

## Getting started

```toml
[dependencies]
dataorder = "0.1"
```

Describe the sequence with `Seq`, then validate and prepare it with `Order::new`.
A `usize` can stand in for a dataset when you only need its length:

```rust
use dataorder::{Order, Seq};

fn main() -> Result<(), dataorder::Error> {
    // Two passes over a billion records, with a fresh shuffle for each pass.
    let seq = Seq::source(1_000_000_000).shuffle(42).repeat(2);
    let order = Order::new(seq)?;
    assert_eq!(order.len(), 2_000_000_000);

    // Resume deep into the second epoch without replaying the earlier positions.
    let resume = 1_200_000_000;
    for (source, index) in order.iter(resume..resume + 10) {
        println!("record {index} from a source of {source} records");
    }
    assert_eq!(order.iter(resume..).next(), Some(order.get(resume)));
    Ok(())
}
```

An order's position and a source's index are different: position 1,200,000,000 above
selects one of the original billion records. `get(pos)` returns its source and index;
`iter(range)` returns the same pairs in order.

## Using your datasets

Implement `Source` for a dataset handle. Only its length is required. A stable `salt`
distinguishes its shuffle from those of other datasets with the same length and seed.

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

`Order` owns the handles and yields references to them. Slices, arrays, vectors and
references to sources also implement `Source`. Keep source lengths stable after
building an order: it uses the lengths recorded at construction.

## Combining sequences

| To… | Use |
| --- | --- |
| Read parts one after another | `Seq::concat(parts)` |
| Interleave parts, preserving each part's order | `Seq::mix(parts)` |
| Choose a total length and relative proportions | `Seq::weighted(total, [(seq, weight), …])` |
| Change when a part appears | `Seq::mix_with` or `Seq::weighted_with`, with a `Sampling` schedule |
| Shuffle positions | `.shuffle(seed)` |
| Repeat whole epochs, reseeding existing shuffles | `.repeat(times)` |
| Repeat or truncate to an exact length | `.cycle(len)` |
| Keep a range or every nth position | `.slice(range)` or `.stride(step, offset)` |
| Assign every nth position to a worker | `.shard(worker_count, worker_index)` |

A plain mix uses every element once, so longer parts appear more often. A weighted
mix repeats or truncates each part to its assigned count; the counts sum exactly
to `total`. Neither operation shuffles a part unless you add `.shuffle(seed)`.

Schedules control **when** elements appear, while lengths or weights control **how
many** appear. For example, this order draws 75% from one source and introduces
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

Uniform parts fill the space left by scheduled parts. A schedule that requires more
records than can fit in an interval is rejected. See
[`Sampling`](https://docs.rs/dataorder/latest/dataorder/enum.Sampling.html) for ramps,
fade-outs and rounding at schedule boundaries.

## Things to know

- **Composition matters.** Shuffle parts before mixing for efficient iteration.
  Repeat the parts to make a schedule span several epochs; repeating the mix restarts
  its schedules each epoch.
- **Workers partition positions.** Apply `.shard(count, index)` to the completed
  sequence to divide its positions without overlap. Sharding parts before mixing
  produces a different order and can make an otherwise valid schedule infeasible.
- **Seeds are reproducible.** The same configuration and seed give the same order on
  supported platforms. `Order::with_seed` and `set_seed` reseed all existing shuffles.
  Adding an outer repeat can change later epochs of repeats inside it; see the
  [shuffle and repetition rules](https://docs.rs/dataorder/latest/dataorder/#shuffles-and-repetitions).
- **Bounds are checked.** `Order::new` reports invalid configurations with an error
  kind and node path. `take` and `skip` past the end are errors. Accessing an invalid
  position with `get`, or an invalid range with `iter`, panics.
- **Reuse cursors.** `iter` is best for consecutive positions. For repeated seeks or
  ranges, reuse its `Cursor` with `seek` or `set_range` to reuse allocated buffers.

`Seq` can be cloned, compared, hashed and mapped to another source type with `map`
or `try_map`. The optional `serde` feature adds configuration serialization. When
using JSON, also enable `serde_json/float_roundtrip` to preserve weights and schedules.
See the [feature documentation](https://docs.rs/dataorder/latest/dataorder/#feature-flags)
for details.

## Development

```sh
cargo run --release --example demo
cargo test --locked --all-features
cargo doc --locked --no-deps --all-features
```

The README's Rust examples are tested with the crate's documentation examples.
For performance measurements and comparisons, see the [benchmark guide](docs/benchmarks.md).

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
