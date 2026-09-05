# dataorder

Deterministic, seekable data order for training, without materializing anything. A `Seq<T>` is
a tree of sequence expressions over `Source` leaves (any `T` that is a `Dataset`: something with
a length): `Concat`, `Mix` (balanced, order-preserving interleaving with per-part sampling
schedules), `Shuffle`, `Repeat`, `Skip`, `Take` and `Stride`. `Order::compile` validates it and
precomputes what iteration needs; elements, `(&source, index in the source)`, are never
materialized: `get` computes any position, `iter` walks any range.

```rust
use dataorder::{Dataset, Order, Sampling, Seq};

struct Shard { path: &'static str, len: usize }
impl Dataset for Shard {
    fn len(&self) -> usize { self.len }
}

let seq = Seq::mix_with([
    (Seq::source(Shard { path: "web.bin", len: 1_000_000 }).shuffle(1).repeat(3), Sampling::Uniform),
    (Seq::source(Shard { path: "code.bin", len: 200_000 }).shuffle(2), Sampling::DelayedLinear { start: 0.5, full: 0.5 }),
])
.shard(0, 8);                                              // worker 0 of 8
let order = Order::compile(seq)?;
for (shard, index) in order.iter(1000..1010) {              // positions 1000..1010, in order
    // element `index` of `shard.path`
}
let (shard, index) = order.get(1005);
```

Builders: `Seq::source`, `Seq::concat`, `Seq::mix` (all uniform), `Seq::mix_with`, and on a
`Seq`: `.shuffle(seed)`, `.repeat(times)`, `.slice(range)`, `.take(n)`, `.skip(n)`,
`.stride(step, offset)`, `.shard(index, count)`, `.map(f)` (the same structure over other
sources: handles become loaded datasets), `.check()` (validate and get the length without
compiling). `Order::compile_seeded(seq, seed)` reseeds every
shuffle at once. A `Seq` is plain data (clone, compare, serialize with your own `T`); compiling
consumes it, and the order owns the sources and yields references to them. A bare `usize` is
a source too, when only the order matters. Lengths and positions are `usize` at the interface
and 64-bit inside.

## Semantics

Every node maps its positions to positions of its children:

| node | length | position `p` maps to |
|---|---|---|
| `Source(t)` | `t.len()` | element `p` of `t` |
| `Concat(parts)` | sum | the part containing `p`, at `p` minus the part's offset |
| `Mix(parts)` | sum | what the interleave of the parts' lengths puts at `p` |
| `Shuffle { seed, inner }` | `n` | `perm_seed(p)` of `inner` |
| `Repeat { times, inner }` | `times·n` | `p mod n` of `inner`, in the context of epoch `p div n` |
| `Skip { n, inner }` | `len − n` | `n + p` of `inner` |
| `Take { n, inner }` | `n` | `p` of `inner` |
| `Stride { step, offset, inner }` | `⌈(n − offset) / step⌉` | `offset + p·step` of `inner` |

A shuffle's permutation depends on its `seed`, the order's seed and the *context*, which every
`Repeat` on the path above derives afresh for each repetition after its first:
`x.shuffle(s).repeat(3)` is `x.shuffle(s)` followed by two other orders of `x`, and
`Seq::concat([x.shuffle(s), x.shuffle(s)])` repeats one order. Shards of one sequence partition it: shard `i` of `count` holds position `i` of every
block of `count` consecutive positions, so a mix's schedule is preserved across workers.
Everything is deterministic in the configuration and the order's seed, and `iter(a..b)` yields
exactly `get(a)..get(b)` whatever was iterated before.

Compilation rejects skips and takes past the end, zero strides, orders longer than `usize::MAX` (and
intermediate lengths beyond 64 bits), invalid or overcommitted schedules, and mixes longer
than 2⁴⁶. It folds what is exact: nested concats flatten, empty parts vanish, skips and takes
merge into sources, slices and strides, a mix or shuffle of a single element is the element,
and a single repetition is the sequence.

## Stability

An order is a pure function of the configuration and the seed: the same `Seq` and seed give the
same elements on every platform (the arithmetic is IEEE 754 binary64, correctly rounded, without
fused operations) and in every release that does not say otherwise. A release that changes any
order is a breaking change (a new minor version while the crate is 0.x). Golden tests in
`src/tests.rs` pin fingerprints of a dozen orders; CI runs the test suite on 64-bit and 32-bit
x86 (Linux, Windows) and on 64-bit ARM (macOS), and checks the declared minimum Rust version.

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
rate on at `d`. Every position holds exactly
one element, so the uniform parts absorb the slack: they keep a constant rate relative to
each other and take whatever share the scheduled parts leave free. If the scheduled parts
alone would need more than 100% of the draw rate at some progress, compilation fails with
`Error::Overcommitted`.

Every element gets an ideal progress `F⁻¹((j + φ)/n)` and the mix is the sort by that value
(ties by part index; `φ` staggers parts so that equal ones round-robin). Each part follows
its schedule to within about one element at any position, and the position of an element is
within `k` (typically `√k`) of `progress·N`, the same warp for all parts. Nothing is
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

A mix is at most 2⁴⁶ long, and a scheduled part must satisfy `length × final_rate ≤ 2⁴⁶`.
`cargo test --release -- --ignored --nocapture` runs this table and the tournament tree
against `BinaryHeap`.

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

`cargo run --release --example bench`, pinned to one core, minimum of two runs
(`scripts/bench_campaign.py`), on a Zen 5. *walk*: one element by `next` after a seek;
*walk before*: the same before the optimization round described below. *seek*:
`order.iter_from(pos)` at a random position (positions a cursor, allocating it). *get*:
`order.get(pos)` at a random position. The first row is a realistic training order: two
mixes, of 1000 and 100 shuffled sources of 0.5–2 million elements, each source repeated 2–4
epochs, mixed together.

| order | walk before | walk | seek | get |
|---|---|---|---|---|
| `mix(mix(1000 × shuffled, 2–4 epochs), mix(100 × same))` | 43.1 ns | 36.6 ns | 47 µs | 27.7 µs |
| `source` | 1.3 ns | 1.3 ns | 0.02 µs | 3.0 ns |
| `shuffle(source)` | 7.4 ns | 7.8 ns | 0.02 µs | 12.2 ns |
| `shuffle(source 10⁶).repeat(1000)` | 7.6 ns | 7.7 ns | 0.04 µs | 17.4 ns |
| `concat(100 × shuffle(source))` | 7.6 ns | 7.7 ns | 0.05 µs | 26.0 ns |
| `shuffle(concat(100 × source))` | 17.7 ns | 18.3 ns | 0.02 µs | 24.3 ns |
| `mix(5 × source)` | 12.6 ns | 7.1 ns | 0.27 µs | 206 ns |
| `mix(80% source + 4 × 5%)` | 13.1 ns | 7.4 ns | 0.28 µs | 224 ns |
| `mix(60% shuffled + 9 × 4.4% shuffled)` | 26.2 ns | 20.4 ns | 0.50 µs | 376 ns |
| `mix(100 × source)` | 16.6 ns | 10.8 ns | 2.81 µs | 2.1 µs |
| `mix(100 × shuffled)` | 24.1 ns | 17.0 ns | 2.96 µs | 2.1 µs |
| `mix(100 × shuffled, 20% scheduled)` | 24.7 ns | 21.7 ns | 5.02 µs | 4.1 µs |
| `mix(1000 × shuffled, 20% scheduled)` | 30.5 ns | 28.0 ns | 44 µs | 37.1 µs |
| `mix(100 × shuffled, 20% scheduled).shard(0, 8)` | 129 ns | 117 ns | 4.60 µs | 4.1 µs |
| `repeat(3, mix(3 nested)).shard(1, 4)` | 71.4 ns | 56.3 ns | 0.26 µs | 192 ns |
| `shuffle(mix(100 × source))` | 3.5 µs | 2.1 µs | 0.02 µs | 2.1 µs |

`get` and a seek walk the path from the root to a source: constant work per node, except that
a `Mix` costs a seek of the interleave (`O(k log s)` for `k` parts, `s` scheduled). A walk keeps a
cursor per node on the active path: a `Mix` costs `⌈log2 k⌉` comparisons per element plus one
key computation, a `Shuffle` one permutation (about 4.5 ns) plus a `get`-style descent into its
child (its positions are scattered, so a shuffle *over* a mix pays the interleave seek per
element: shuffle the parts, not the mix), a `Stride` skips `step − 1` elements of its child,
re-seeking a mix when that is cheaper than stepping.

### What the optimization round kept

Everything was measured on the table above; these four earned their place at little code:

- Keys of the interleave by multiplication with precomputed reciprocals instead of division, and
  no square root on constant-rate segments (a mix of 100 sources went from 16.6 to 12.3 ns on
  this alone).
- Inlining control: the interleave step and the mix step force-inlined into the per-node
  dispatcher, the shuffle step outlined into a non-inlined function so that the dispatcher
  stays small, and an explicit enum tag (`#[repr(u8)]`) so that no dispatch decodes a niche.
- A six-round Feistel network with one multiply per round instead of four rounds with two
  (same statistics, slightly cheaper).
- Tree slots of 24 bytes (`NaN` for "no next key") and a walk step without an `Option`.

### Measured and rejected

Not worth their complexity, though they won:

- Closed-form answers for mix parts that are a source or a shuffled source, without a child
  cursor: 1.6 ns per element on mixes of shuffled parts, ~60 lines.
- A bucket table for `Concat` instead of a binary search: a shuffle over a 100-part concat from
  18 to 9 ns, ~40 lines.
- Eight permutations at a time in AVX-512 lanes (runtime-detected) into a per-shuffle buffer, and
  a batch `fill` API writing whole blocks: a shuffled source from 7.4 to 4.5 ns per element by
  `next` and 2.2 ns by `fill`, mixes of shuffled parts 1–2 ns less; ~230 lines, the crate's only
  `unsafe`, platform-specific, and a second per-node code path to keep in sync.

Not worth it at all:

- A flat arena (nodes and per-node states in two vectors, the walk a loop over indices): 2–3 ns
  slower per element on shallow trees, equal where the interleave dominates, 8× slower to seek
  on wide trees.
- Run-length emission in the interleave (the runner-up from the losers on the winner's path,
  the run length from the share function): the bookkeeping costs 2–4 ns on every element of a
  balanced mix, and on a skewed one the closed-form count costs about what the replays it
  replaces cost, at `k ≤ 100`.
- Random access through cursor seeks instead of the stateless descent: 0.5–4 ns slower per
  shuffled element.
- A masked multiply–xorshift mixer as the permutation (serial correlation over 100σ) and
  three Feistel rounds (fails the grid test).

## Layout

| file | contents |
|---|---|
| `src/lib.rs` | crate docs |
| `src/error.rs` | `Error` |
| `src/seq.rs` | `Seq` and its builder methods |
| `src/dataset.rs` | `Dataset` |
| `src/order.rs` | `Order`: compilation with folds, the node tree, random access |
| `src/cursor.rs` | `Cursor`: per-node cursors with seek, next and skip |
| `src/perm.rs` | seeded permutations of `0..n` and context derivation |
| `src/interleave/` | the mix: `mod.rs` model and construction, `iter.rs` seek and walk, `profile.rs` rate profiles and their integrals, `tournament.rs` loser tree, `sampling.rs` schedules, `tests.rs` merge against brute force, exact seeks, balance and schedule bounds |
| `src/tests.rs` | random configurations against a materializing reference evaluator, golden orders |
| `scripts/bench_campaign.py` | pinned, repeated benchmark runs tabulated across labelled steps |

Run `cargo run --release --example demo` for a small schedule, `--example bench` for the table, and
`cargo test --release -- --ignored --nocapture` for the interleave, tournament tree and permutation
micro-benchmarks.
