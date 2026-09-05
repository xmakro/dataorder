# dataorder

Deterministic, seekable data order for training, without materializing anything. A `Seq<T>` is
a tree of sequence expressions over `Source` leaves (any `T` that is a `Source`: something with
a length): `Concat`, `Mix` (balanced, order-preserving interleaving with per-part sampling
schedules), `Weighted` (a mix in given proportions), `Shuffle`, `Repeat`, `Skip`, `Take` and
`Stride`. `Order::new` validates it and precomputes what iteration needs; elements,
`(&source, index in the source)`, are never materialized: `get` computes any position, `iter`
walks any range.

```rust
use dataorder::{Order, Sampling, Seq, Source};

struct Shard { path: &'static str, len: usize }
impl Source for Shard {
    fn len(&self) -> usize { self.len }
}

fn main() -> Result<(), dataorder::Error> {
    let seq = Seq::mix_with([
        (Seq::source(Shard { path: "web.bin", len: 1_000_000 }).shuffle(1).repeat(3), Sampling::Uniform),
        (Seq::source(Shard { path: "code.bin", len: 200_000 }).shuffle(2), Sampling::delayed(0.5)),
    ])
    .shard(8, 0); // worker 0 of 8
    let order = Order::new(seq)?;
    for (shard, index) in order.iter(1000..1010) {
        println!("element {index} of {}", shard.path);
    }
    let (shard, index) = order.get(1005);
    println!("position 1005 is element {index} of {}", shard.path);
    Ok(())
}
```

Builders: `Seq::source`, `Seq::concat`, `Seq::mix` (all uniform), `Seq::mix_with`,
`Seq::weighted(total, [(seq, weight), …])` and `Seq::weighted_with` (each part is repeated and
cut to its share of `total`, epochs reshuffled), and on a `Seq`: `.shuffle(seed)`,
`.repeat(times)`, `.slice(range)`, `.take(n)`, `.skip(n)`, `.stride(step, offset)`,
`.shard(count, index)`, `.map(f)` (the same structure over other sources: handles become loaded
datasets), `.check()` (validate and get the length without building the order).
`Order::with_seed(seq, seed)` reseeds every shuffle at once. A `Seq` is plain data (clone,
compare, hash; the `serde` feature derives `Serialize` and `Deserialize`); building the order
consumes it, and the order owns the sources, yields references to them and gives them back with
`into_sources`. A bare `usize` is a source too, when only the order matters.

The precise semantics of every node, what compilation rejects and folds, and the stability
policy are in the [crate documentation](https://docs.rs/dataorder). In short: every node maps its
positions to positions of its children; a shuffle depends on its seed, the order's seed and the
repetition it is in, so `x.shuffle(s).repeat(3)` is three different orders of `x`; shards
partition a sequence position by position, so a mix's schedule is preserved across workers;
`iter(a..b)` yields exactly `get(a)..get(b)`; and an order is a pure function of the
configuration and the seed on every platform. A release that changes any order is a breaking
change and is listed in [CHANGELOG.md](CHANGELOG.md).

## Mix

```text
  ramp(d0, d1)                       delayed(d)                      Uniform
            ________________                 ________________     ________________
           /                                 |
  ________/                         ________|
          d0     d1                          d
```

Progress `τ` is the position in the mix divided by its length `N`. A scheduled part follows
a share function `F(τ)`, the fraction of it drawn by progress `τ`: `Sampling::ramp(d0, d1)`
(`DelayedLinear { start: d0, full: d1 }`) is the integral of a rate that is zero until `d0`,
rises linearly until `d1` and stays constant afterwards; `Sampling::delayed(d)` switches the
rate on at `d`. Every position holds exactly one element, so the uniform parts absorb the
slack: they keep a constant rate relative to each other and take whatever share the scheduled
parts leave free. If the scheduled parts alone would need more than 100% of the draw rate at
some progress, `Order::new` fails with `ErrorKind::Overcommitted`.

Every element gets an ideal progress `F⁻¹((j + φ)/n)` and the mix is the sort by that value
(ties by part index; `φ` staggers the non-empty parts so that equal ones round-robin). Each
part follows its schedule to within about one element at any position, and the position of an
element is within `k` (typically `√k`) of `progress·N`, the same warp for all parts. Nothing is
materialized: a seek counts, per part, the elements below the target progress (`O(k log s)`
for `k` parts, `s` scheduled), and the walk takes the minimum of a tournament tree over the
parts' next elements, `⌈log2 k⌉` branch-free comparisons per element. Seeks are exact
whatever was walked before, because the order is defined as a sort by keys that are monotone
within a part by construction.

| k | seek (uniform / 20% scheduled) | per element (uniform / scheduled) |
|---|---|---|
| 100 | 5 µs / 6 µs | 15 ns / 17 ns |
| 1 000 | 54 µs / 75 µs | 20 ns / 23 ns |
| 10 000 | 0.75 ms / 0.87 ms | 31 ns / 32 ns |

A mix is at most 2⁴⁶ long (`MAX_MIX_LEN`), and a scheduled part must satisfy
`length × final_rate ≤ 2⁴⁶`. `cargo test --release -- --ignored --nocapture` runs this table
and the tournament tree against `BinaryHeap`.

## Shuffle

A seeded permutation of `0..n` in O(1) per element and no state: a six-round Feistel network
on the `k`-bit numbers (`2^(k−1) < n ≤ 2^k`) with cycle walking to `0..n`. The round function
adds the round key to half the bits, multiplies by the round's odd multiplier and keeps the top
bits of the product. It passes joint-distribution (grid and low bits), serial-correlation and
fixed-point checks at every size from 2 to 10⁶ (`src/perm.rs` tests); there is no security
claim. A masked multiply–xorshift mixer (MurmurHash3's finalizer cut to `k` bits) is twice as
fast but fails badly as a permutation: consecutive inputs map to outputs with a nearly constant
difference.

## Cost

Per element and per operation, measured 2026-09 on one core of a Ryzen 9 9950X3D
(`cargo run --release --example bench`; the benchmark harness in the repository,
`scripts/bench_campaign.py`, pins the core and keeps the minimum of repeated runs). *walk*: one
element by `next` after a seek. *seek*: `order.iter(pos..)` at a random position (builds and
positions a cursor). *get*: `order.get(pos)` at a random position. The first row is a
realistic training order: two mixes, of 1000 and 100 shuffled sources of 0.5–2 million
elements, each source repeated 2–4 epochs, mixed together.

| order | walk | seek | get |
|---|---|---|---|
| `mix(mix(1000 × shuffled, 2–4 epochs), mix(100 × same))` | 36.6 ns | 47 µs | 27.7 µs |
| `shuffle(source)` | 7.8 ns | 0.02 µs | 12.2 ns |
| `shuffle(source 10⁶).repeat(1000)` | 7.7 ns | 0.04 µs | 17.4 ns |
| `shuffle(concat(100 × source))` | 18.3 ns | 0.02 µs | 24.3 ns |
| `mix(5 × source)` | 7.1 ns | 0.27 µs | 206 ns |
| `mix(60% shuffled + 9 × 4.4% shuffled)` | 20.4 ns | 0.50 µs | 376 ns |
| `mix(100 × source)` | 10.8 ns | 2.81 µs | 2.1 µs |
| `mix(100 × shuffled)` | 17.0 ns | 2.96 µs | 2.1 µs |
| `mix(100 × shuffled, 20% scheduled)` | 21.7 ns | 5.02 µs | 4.1 µs |
| `mix(1000 × shuffled, 20% scheduled)` | 28.0 ns | 44 µs | 37.1 µs |
| `mix(100 × shuffled, 20% scheduled).shard(8, 0)` | 117 ns | 4.60 µs | 4.1 µs |
| `repeat(3, mix(3 nested)).shard(4, 1)` | 56.3 ns | 0.26 µs | 192 ns |
| `shuffle(mix(100 × source))` | 2.1 µs | 0.02 µs | 2.1 µs |

`get` and a seek walk the path from the root to a source: constant work per node, except that
a `Mix` costs a seek of the interleave (`O(k log s)` for `k` parts, `s` scheduled). A walk keeps a
cursor per node on the active path: a `Mix` costs `⌈log2 k⌉` comparisons per element plus one
key computation, a `Shuffle` one permutation (about 4.5 ns) plus a `get`-style descent into its
child (its positions are scattered, so a shuffle *over* a mix pays the interleave seek per
element: shuffle the parts, not the mix), a `Stride` skips `step − 1` elements of its child,
re-seeking a mix when that is cheaper than stepping. Seeking an existing cursor reuses its
buffers; `Iterator::nth` skips without visiting. How these numbers came about, and what was
tried and rejected, is in [docs/optimization-notes.md](docs/optimization-notes.md).

## Layout

| file | contents |
|---|---|
| `src/lib.rs` | crate docs: semantics, cost, feature flags, stability |
| `src/seq.rs` | `Seq`, `MixPart`, `WeightedPart` and the builder methods |
| `src/source.rs` | `Source` |
| `src/error.rs` | `Error` (kind and path) and `ErrorKind` |
| `src/order.rs` | `Order`: construction with folds, the node tree, random access |
| `src/cursor.rs` | `Cursor`: per-node cursors with seek, next and skip |
| `src/perm.rs` | seeded permutations of `0..n` and context derivation |
| `src/interleave/` | the mix: `mod.rs` model and construction, `iter.rs` seek and walk, `profile.rs` rate profiles and their integrals, `tournament.rs` loser tree, `sampling.rs` schedules, `tests.rs` merge against brute force, exact seeks, balance and schedule bounds |
| `src/tests.rs` | random configurations against a materializing reference evaluator |
| `tests/golden.rs`, `tests/api.rs` | pinned orders and the public surface, as a downstream crate sees them |
| `scripts/bench_campaign.py` | pinned, repeated benchmark runs tabulated across labelled steps (repository only) |

Run `cargo run --release --example demo` for a small schedule, `--example bench` for the table, and
`cargo test --release -- --ignored --nocapture` for the interleave, tournament tree and permutation
micro-benchmarks.
