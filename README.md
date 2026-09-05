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
    fn salt(&self) -> u64 { dataorder::salt(self.path) }
}

fn main() -> Result<(), dataorder::Error> {
    // Worker 0 of 8: each part is sharded, then the shards are mixed (see Cost).
    let seq = Seq::mix_with([
        (Seq::source(Shard { path: "web.bin", len: 1_000_000 }).shuffle(1).repeat(3).shard(8, 0), Sampling::Uniform),
        (Seq::source(Shard { path: "code.bin", len: 200_000 }).shuffle(2).shard(8, 0), Sampling::delayed(0.5)),
    ]);
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
`.repeat(times)`, `.cycle(len)` (repeated as often as `len` positions need and cut there;
`cycle(usize::MAX)` never runs out), `.slice(range)`, `.take(n)`, `.skip(n)`, `.stride(step, offset)`,
`.shard(count, index)`, `.map(f)` and `.try_map(f)` (the same structure over other sources:
handles become loaded datasets), `.check()` (validate and get the length without consuming the
configuration). `Order::with_seed(seq, seed)` and `order.set_seed(seed)` reseed every shuffle at once;
a cursor is repositioned with `seek(pos)` and re-ranged with `set_range(range)`. A `Seq` is
plain data (clone, compare, hash; the `serde` feature derives `Serialize` and `Deserialize`);
building the order consumes it, and the order owns the sources, yields references to them
(`order.source_index(&s)` says which one, also when sources compare equal) and gives them back
with `into_sources`. JSON consumers should enable `serde_json/float_roundtrip` to preserve
floating-point weights and schedules exactly; `dataorder/serde` alone does not enable it.
A bare `usize` is a source too, when only the order matters, as are
slices, arrays and vectors; `dataorder::salt` and `salt_path` turn a name or a path into a salt.

The precise semantics of every node, what compilation rejects and folds, and the stability
policy are in the [crate documentation](https://docs.rs/dataorder). In short: every node maps its
positions to positions of its children; a shuffle depends on its seed, the order's seed, the
repetition it is in and the sources under it that it can draw from (their salts and lengths),
so `x.shuffle(s).repeat(3)` is three different orders of `x`, sources with different salts
never shuffle alike, and an empty shard in a listing, or one beyond a `take`, changes nothing;
shards
partition a sequence position by position, so a mix's schedule is preserved across workers;
`iter(a..b)` yields exactly `get(a)..get(b)`; and an order is a pure function of the
configuration and the seed on every platform. A release that changes any order is a breaking
change.

Repeat contexts also depend on nesting depth. Adding an outer repeat changes the later
epochs of inner repeats, even during the outer repeat's first epoch; the same applies when
a cycle or weighted share needs to repeat its part. Extending a nested sequence this way
can change its existing prefix. A single repetition or a cycle within its current length
preserves that prefix.

## Mix

```text
  ramp(d0, d1)               delayed(d)              trapezoid(d0, d1, d2, d3)     Uniform
            ________                 ________              ____
           /                         |                    /    \                ________
  ________/                  ________|            _______/      \_______
          d0     d1                  d                  d0  d1  d2  d3
```

Progress `τ` is the position in the mix divided by its length `N`. A scheduled part follows
a share function `F(τ)`, the fraction of it drawn by progress `τ`: `Sampling::ramp(d0, d1)`
(`DelayedLinear { start: d0, full: d1 }`) is the integral of a rate that is zero until `d0`,
rises linearly until `d1` and stays constant afterwards; `Sampling::delayed(d)` switches the
rate on at `d`. `Sampling::trapezoid(d0, d1, d2, d3)` (`Trapezoid { start, full, fade, off }`)
also falls back to zero between `d2` and `d3`, so a source can be phased out; `until(d)` and
`fading(d2, d3)` are its constant-then-off forms. Every position holds exactly one element, so
the uniform parts absorb the slack: they keep a constant rate relative to each other and take
whatever share the scheduled parts leave free. If the scheduled parts alone would need more
than 100% of the draw rate at some progress, `Order::new` fails with
`ErrorKind::Overcommitted`. Schedules are relative to their mix: repeat the parts, not the mix,
to schedule over a run of several epochs.

Every element gets an ideal progress `F⁻¹((j + φ)/n)` and the mix is the sort by that value
(ties by part index; `φ` staggers the non-empty parts so that equal ones round-robin). Each
part follows its schedule with discrete rounding error; for a feasible profile with exact
arithmetic, the position of an element is within `k` of `progress·N`, the same warp for all
parts. The feasibility check accepts excess demand up to `10⁻⁹` for numerical rounding.
For a mix near the length limit, that tolerance can add thousands of positions of drift,
so the `k` bound is not a guarantee for every accepted configuration. Nothing is
materialized: a seek counts the elements below a target progress and replays at most `2k`
heads, normally `O(k log(S + 1) + k log(k + 1))` for `k` non-empty parts and `S` distinct
schedule breakpoints. Poor numerical guesses use bounded binary searches (see the crate's
Cost documentation). Building the mix is `O(k + s log s)` for `s` scheduled parts.
The walk takes the minimum of a tournament tree
over the parts' next elements, `⌈log2 k⌉` branch-free comparisons per element. Seeks are exact
whatever was walked before, because the order is defined as a sort by keys that are monotone
within a part by construction.

| k | seek (uniform / 20% scheduled, 7 distinct starts / 20% scheduled, all starts distinct) | per element (same three) |
|---|---|---|
| 100 | 2.8 µs / 4.8 µs / 6.0 µs | 10 ns / 13 ns / 12 ns |
| 1 000 | 29 µs / 51 µs / 91 µs | 15 ns / 19 ns / 18 ns |
| 10 000 | 0.44 ms / 0.62 ms / 1.24 ms | 22 ns / 29 ns / 26 ns |

A mix is at most 2⁴⁶ long (`MAX_MIX_LEN`), and a scheduled part must satisfy
`length × peak rate ≤ 2⁴⁶`. Extremely narrow transitions whose derived coefficients overflow are
rejected; use equal adjacent breakpoints for an abrupt change. `cargo test --release -- --ignored --nocapture` runs this table
and the tournament tree against `BinaryHeap`.

These Ryzen measurements precede the later numerical and shuffle fixes.

## Shuffle

A seeded permutation of `0..n` in O(1) per element on average and no state: a six-round
Feistel network on the `k`-bit numbers (`2^(k−1) < n ≤ 2^k`) with cycle walking to `0..n`.
Each round adds its independently derived key to one half, applies the full `SplitMix64`
finalizer, and keeps the low bits needed by the other half. A previous seven-round network
with one multiplication per round retained strong adjacency patterns for some seeds and
lengths, including `source(56_444).shuffle(1)`.

Tests check bijectivity, joint distribution, serial correlation, consecutive differences,
fixed points and small-domain coverage across seeds. They include the reported weak public
configurations and independently chosen seeds and lengths. These checks provide evidence
of statistical quality, not a guarantee for every configuration or a security claim.

## Cost

Per element and per operation, measured 2026-09 on one core of a Ryzen 9 9950X3D
(`cargo run --release --example bench`; the benchmark harness in the repository,
`examples/bench_campaign.rs`, pins the core and keeps the minimum of repeated runs). *walk*: one
element by `next` after a seek. *seek*: `order.iter(pos..).next()` at a random position (builds
and positions a cursor and draws its first element, which is what enters the parts of a mix).
*get*: `order.get(pos)` at a random position. The first row is a realistic training order: two mixes,
of 1000 and 100 shuffled sources of 0.5–2 million elements, each source repeated 2–4 epochs,
mixed together.

These are the Ryzen measurements before the numerical and shuffle review fixes.

| order | walk | seek | get |
|---|---|---|---|
| `mix(mix(1000 × shuffled, 2–4 epochs), mix(100 × same))` | 38.4 ns | 30 µs | 27.6 µs |
| `source` | 1.4 ns | 0.02 µs | 3.1 ns |
| `shuffle(source)` | 9.0 ns | 0.03 µs | 16.2 ns |
| `shuffle(source 10⁶).repeat(1000)` | 9.2 ns | 0.04 µs | 19.4 ns |
| `concat(100 × shuffle(source))` | 9.2 ns | 0.06 µs | 29.9 ns |
| `shuffle(concat(100 × source))` | 20.3 ns | 0.05 µs | 29.1 ns |
| `mix(5 × source)` | 6.9 ns | 0.19 µs | 142.7 ns |
| `mix(80% source + 4 × 5% source)` | 7.3 ns | 0.18 µs | 155.2 ns |
| `mix(60% shuffled + 9 × 4.4% shuffled)` | 22.3 ns | 0.36 µs | 284.0 ns |
| `mix(100 × source)` | 10.5 ns | 2.07 µs | 1.9 µs |
| `mix(100 × shuffled)` | 18.3 ns | 2.10 µs | 1.9 µs |
| `mix(100 × shuffled, 20% scheduled)` | 24.2 ns | 4.04 µs | 3.8 µs |
| `mix(1000 × shuffled, 20% scheduled)` | 29.5 ns | 37 µs | 36.4 µs |
| `mix(100 × shuffled, 20% scheduled).shard(8, 0)` | 114.8 ns | 3.76 µs | 3.8 µs |
| `mix(100 × shuffled, 20% scheduled).shard(512, 0)` | 4.3 µs | 7.58 µs | 3.8 µs |
| `repeat(3, mix(3 nested)).shard(4, 1)` | 56.7 ns | 0.26 µs | 170.6 ns |
| `mix(mix(100 × shuffled), mix(100 × shuffled))` | 23.6 ns | 2.23 µs | 2.0 µs |
| `mix(mix(100 × shuffled), mix(100 × shuffled)).shard(8, 0)` | 106.0 ns | 2.27 µs | 2.0 µs |
| `shuffle(mix(100 × source))` | 1.9 µs | 1.96 µs | 1.9 µs |

`get` and a seek walk the path from the root to a source: a `Concat` searches its offsets
in `O(log k)`, a `Mix` seeks its interleave as described above, and a `Shuffle` cycle-walks
its permutation at constant average cost (an individual position can take longer). A walk keeps a
cursor per node on the active path: a `Mix` costs `⌈log2 k⌉` comparisons per element plus one
key computation, a `Shuffle` one permutation plus a `get`-style descent into its
child (its positions are scattered, so a shuffle *over* a mix pays the interleave seek per
element: shuffle the parts, not the mix), a `Stride` skips `step − 1` elements of its child
(a mix steps its interleave, or re-seeks it when that is cheaper, and its parts skip along, a
nested mix stepping its own interleave, so a shard of a mix of mixes costs about what a shard
of a flat mix does: the two `.shard(8, 0)` rows). Seeking an existing cursor reuses its
buffers, including after cloning; `Iterator::nth` skips without visiting. Cursor state is
built on the first draw, so empty ranges and `count` allocate nothing. A mix with just one
remaining part stops replaying the tournament. Sharding each part before mixing reduces work
only when the resulting worker schedules remain feasible: rounding each part
independently changes proportions and can make a worker overcommitted. Shard the global
mix when its exact position partition must be preserved.

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
| `src/weight.rs` | exact largest-remainder allocation of binary64 weights |
| `src/interleave/` | the mix: `mod.rs` model and construction, `iter.rs` seek and walk, `profile.rs` rate profiles and their integrals, `tournament.rs` loser tree, `sampling.rs` schedules, `tests.rs` merge against brute force, exact seeks, balance and schedule bounds |
| `src/tests.rs` | random configurations against a materializing reference evaluator |
| `tests/golden.rs`, `tests/api.rs`, `tests/cost.rs` | pinned orders, the public surface as a downstream crate sees it, and the cost model (allocation counts of seeks, `count` and `last`, shards of nested mixes) |
| `examples/bench_campaign.rs` | pinned, repeated benchmark runs tabulated across labelled steps |

Run `cargo run --release --example demo` for a small schedule, `--example bench` for the table, and
`cargo test --release -- --ignored --nocapture` for the interleave, tournament tree and permutation
micro-benchmarks.

For schedule-phase and continuous-tail measurements, run
`cargo run --release --example bench -- --phases`; `-- --lifecycle` adds compilation,
reused-cursor seeking and requested allocation bytes. `-- --all` runs every group. The
campaign harness accepts the corresponding `phases`, `lifecycle` or `all` mode after its
directory argument:

```sh
cargo run --release --example bench_campaign -- run baseline . all
cargo run --release --example bench_campaign -- table 1
```

Each campaign runs twice and saves per-column minima in `target/bench-campaign.json`.
`BENCH_MODE` sets the default mode; `BENCH_CORE` selects a core (default `2`), or `none`
on systems without `taskset`. Table columns are `0` seek, `1` walk (default), `2` get,
`3` build, `4` reused seek and `5` cursor requested bytes. Existing campaign JSON files
can be moved to `target/bench-campaign.json`.
