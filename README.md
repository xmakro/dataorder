# dataorder

Shuffle and mix billions of records. Seek anywhere, then iterate from there.

`dataorder` computes the order in which to read records from your datasets, without
building an index array. Describe the order you want: shuffle a dataset, mix several,
repeat for a few epochs, shard across workers. Each lookup then tells you which dataset
to read and which record within it. Loading the record is up to you.

- **Shuffle on demand.** Each shuffled index is computed when needed, in O(1) time on
  average, so the permutation is never stored.
- **Seek anywhere.** Jumping to any position costs about the same wherever it is;
  nothing before it is replayed.
- **Iterate cheaply.** After a seek, a mix picks the next dataset with O(log k)
  comparisons for k datasets.
- **Reproduce exactly.** The same configuration, dataset lengths and seed give the
  same order on every supported platform.

No dependencies by default.
[API documentation](https://docs.rs/dataorder) · [Runnable example](examples/demo.rs)

## Getting started

```toml
[dependencies]
dataorder = "0.4"
```

Implement `Source` for your dataset handle. Only the length is required; the salt,
derived here from a distinct key per dataset, makes datasets of equal length shuffle
differently. Describe the order with `Seq`, then compile it into an `Order`:

```rust
use dataorder::{Order, Seq, Source};

struct Dataset {
    key: &'static str,
    records: usize,
}

impl Source for Dataset {
    fn len(&self) -> usize { self.records }
    fn salt(&self) -> u64 { dataorder::salt(self.key) }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Two passes over each dataset, shuffled differently on each pass, interleaved.
    let web = Seq::source(Dataset { key: "web", records: 1_000_000_000 }).repeat_shuffled(2);
    let code = Seq::source(Dataset { key: "code", records: 200_000_000 }).repeat_shuffled(2);
    let order = Order::with_seed(Seq::mix([web, code]), 42)?;
    assert_eq!(order.len(), 2_400_000_000);

    // Resume deep into the run. Nothing before it is computed.
    let resume = 1_500_000_000;
    for item in order.cursor(resume..resume + 10)? {
        // Load the record with your own reader.
        println!("{}: record {}", item.source.key, item.record_index);
    }
    Ok(())
}
```

`get(pos)` returns a single position and `iter()` the whole order. In the examples
below, a `usize` stands in for a dataset of that length.

## Building sequences

| To… | Use |
| --- | --- |
| Shuffle | `.shuffle()` |
| Read sequences one after another | `Seq::concat([a, b])` |
| Interleave sequences, each keeping its own order | `Seq::mix([a, b])` |
| Mix in chosen proportions | `Seq::mix([a.cycle_to_shuffled(750), b.cycle_to_shuffled(250)])` |
| Control when a part of a mix contributes | `Seq::mix([(a, Schedule::Uniform), (b, Schedule::delayed(0.5))])` |
| Repeat, in the same order every pass | `.repeat(times)` |
| Repeat, reshuffled on every pass | `.repeat_shuffled(times)` |
| Repeat or cut to an exact length | `.cycle_to(len)` or `.cycle_to_shuffled(len)` |
| Keep a range of positions | `.skip(start).take(len)` |
| Split across workers | `.skip(worker).step_by(workers)` |

Sequences nest freely, with one rule: shuffle before mixing. A shuffle's input must
not contain a mix, and `Order::new` rejects one that does.

## Mixing

A mix uses every element of every part once, so longer parts appear more often. To
choose the proportions, give each part an exact count: `cycle_to_shuffled` repeats or
cuts the part over freshly shuffled passes, and `cycle_to` keeps the part's order.
A 75/25 mixture of one million records:

```rust
use dataorder::{Order, Seq};

fn main() -> Result<(), dataorder::Error> {
    let seq = Seq::mix([
        Seq::source(100_000).cycle_to_shuffled(750_000), // 7.5 shuffled passes
        Seq::source(500_000).cycle_to_shuffled(250_000), // half of a shuffled pass
    ]);
    assert_eq!(Order::new(seq)?.len(), 1_000_000);
    Ok(())
}
```

Changing a part's count changes the whole mix, including the elements before the
change. To extend a run without disturbing its prefix, concatenate the new data after
the original configuration; to change the mixture partway through, mix what remains:
`Seq::mix([old.skip(consumed), new])`.

A schedule controls when a part contributes, on a shared virtual clock from 0 to 1.
`Schedule::Uniform` draws at a constant rate throughout, `delayed(0.5)` starts halfway
along the clock, `ramp(0.2, 0.6)` rises from zero, and `until`, `fade` and `trapezoid`
stop or fall off. The clock is not output progress: at virtual time 0.5 a uniform part
has contributed half of its elements, so the delayed part below starts after about 375
of the 750 uniform elements, not at position 500.

```rust
use dataorder::{Order, Schedule, Seq};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let seq = Seq::mix([
        (Seq::source(300).cycle_to_shuffled(750), Schedule::Uniform),
        (Seq::source(100).cycle_to_shuffled(250), Schedule::delayed(0.5)),
    ]);
    let order = Order::new(seq)?;
    let first_delayed = order.iter().position(|item| item.source_ordinal == 1).unwrap();
    assert!((374..=377).contains(&first_delayed));
    Ok(())
}
```

Repeating a mix restarts its clock on every pass; to run a schedule across several
epochs, repeat the parts and mix them once, as in the [example](examples/demo.rs).
See [`Schedule`](https://docs.rs/dataorder/latest/dataorder/enum.Schedule.html) for
the model, ramps, fade-outs and limits.

## Reproducibility

The same configuration, dataset lengths and salts, and seed produce the same order.
`Order::with_seed(seq, seed)` and `order.set_seed(seed)` set the seed for every
shuffle at once; the
[shuffle rules](https://docs.rs/dataorder/latest/dataorder/#shuffles-and-repetitions)
say what a permutation depends on and which configuration changes keep it.

To resume a run, save the configuration, the seed and the position, then rebuild the
order and open a cursor there. A release that changes any order is a breaking change,
so resume with the crate version that produced the run. The optional `serde` feature
serializes configurations; with JSON, enable `serde_json/float_roundtrip` so that
schedule parameters survive a round trip.

## Performance

Building an order depends only on the configuration, never on the data. A lookup walks
from the root of the sequence to a dataset: a binary search per concat, one permutation
per shuffle and a bounded seek per mix. A cursor pays for one seek and then iterates,
so prefer `cursor(range)` over repeated `get` calls, and reuse a cursor with
`reset(range)` when seeking often. The
[cost model](https://docs.rs/dataorder/latest/dataorder/#cost) has the details.

## Development

Requires Rust 1.89 or newer.

```sh
cargo test --locked --all-features
cargo test --locked --all-features --examples
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo doc --locked --no-deps --all-features
cargo bench --bench ordering
python3 tests/fixtures/generate_schedule_oracle.py --check
```

The README's examples run as doctests. The schedule tests compare against fixtures
from an independent Python oracle; run the generator without `--check` to regenerate
them after changing it. To compare benchmarks across a change, run them with
`-- --save-baseline before` on the old code and `-- --baseline before` on the new.

Licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
