# dataorder

Deterministic, seekable dataset ordering for training.

Shuffle datasets, mix them in chosen proportions, repeat epochs and split the result
across workers. `dataorder` computes each position as a **source and an index within
that source**. It does not load records or build a list of all indices.

[API documentation](https://docs.rs/dataorder) · [Runnable example](examples/demo.rs) ·
[Benchmarks](docs/benchmarks.md)

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
    // Two passes over 1,000 records, with a fresh shuffle for each pass.
    let seq = Seq::source(1000).shuffle(42).repeat(2);
    let order = Order::new(seq)?;
    assert_eq!(order.len(), 2000);

    // Resume at position 1,200 without replaying the earlier positions.
    for (source, index) in order.iter(1200..1210) {
        println!("record {index} from a source of {source} records");
    }
    assert_eq!(order.iter(1200..).next(), Some(order.get(1200)));
    Ok(())
}
```

An order's position and a source's index are different: position 1,200 above selects
one of the original 1,000 records. `get(pos)` returns that record's source and index;
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
