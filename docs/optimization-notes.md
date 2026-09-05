# Optimization notes

An optimization round over the walk (2026-09-04), measured with `scripts/bench_campaign.py`:
one core (`taskset`), minimum of two runs per row, on a Ryzen 9 9950X3D. Unpinned numbers on
that machine scatter by ±20% between its two dies, which is why the harness pins. *walk
before* is the crate before the round; everything else is the crate as published.

| order | walk before | walk after | seek | get |
|---|---|---|---|---|
| `mix(mix(1000 × shuffled, 2–4 epochs), mix(100 × same))` | 43.1 ns | 36.6 ns | 47 µs | 27.7 µs |
| `shuffle(source)` | 7.4 ns | 7.8 ns | 0.02 µs | 12.2 ns |
| `shuffle(source 10⁶).repeat(1000)` | 7.6 ns | 7.7 ns | 0.04 µs | 17.4 ns |
| `shuffle(concat(100 × source))` | 17.7 ns | 18.3 ns | 0.02 µs | 24.3 ns |
| `mix(5 × source)` | 12.6 ns | 7.1 ns | 0.27 µs | 206 ns |
| `mix(60% shuffled + 9 × 4.4% shuffled)` | 26.2 ns | 20.4 ns | 0.50 µs | 376 ns |
| `mix(100 × source)` | 16.6 ns | 10.8 ns | 2.81 µs | 2.1 µs |
| `mix(100 × shuffled)` | 24.1 ns | 17.0 ns | 2.96 µs | 2.1 µs |
| `mix(100 × shuffled, 20% scheduled)` | 24.7 ns | 21.7 ns | 5.02 µs | 4.1 µs |
| `mix(1000 × shuffled, 20% scheduled)` | 30.5 ns | 28.0 ns | 44 µs | 37.1 µs |
| `mix(100 × shuffled, 20% scheduled).shard(0, 8)` | 129 ns | 117 ns | 4.60 µs | 4.1 µs |
| `repeat(3, mix(3 nested)).shard(1, 4)` | 71.4 ns | 56.3 ns | 0.26 µs | 192 ns |
| `shuffle(mix(100 × source))` | 3.5 µs | 2.1 µs | 0.02 µs | 2.1 µs |

## What the optimization round kept

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

## Measured and rejected

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


The full-optimization tree (AVX-512 permutations, buffers, `fill`, mix leaves) is archived
outside the repository and can be revived from these notes if the shuffle path ever matters
more than the code it costs.
