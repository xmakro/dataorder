# Optimization notes

An optimization round over the walk (2026-09-04), measured with `scripts/bench_campaign.py`:
one core (`taskset`), minimum of two runs per row, on a Ryzen 9 9950X3D. Unpinned numbers on
that machine scatter by ±20% between its two dies, which is why the harness pins. *walk
before* is the crate before the round; everything else is the crate as published. *seek* here
is `order.iter(pos..)` alone, which leaves the parts of a nested mix unentered (the first row's
47 µs is the eager cursor of the time); the README's table now times the first element too.

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
| `mix(100 × shuffled, 20% scheduled).shard(8, 0)` | 129 ns | 117 ns | 4.60 µs | 4.1 µs |
| `repeat(3, mix(3 nested)).shard(4, 1)` | 71.4 ns | 56.3 ns | 0.26 µs | 192 ns |
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

## Review round (2026-09-04)

Changes made for correctness or API reasons, checked against the table above with interleaved
runs of the old and new binaries (the only way to separate them from the ±1 ns drift between
sessions):

- Child cursors of a mix (and the current child of a concat) are built when first entered.
  Cursor creation for the nested realistic order went from 47 µs to 0.1 µs; the walk is
  unchanged, but only with the child seek marked `#[cold] #[inline(never)]`: inlined into the
  mix step it cost 0.3 to 0.6 ns per element on every mix row.
- The interleave's iterator seeks in place and the tournament tree is rebuilt into its
  existing allocations, so `Cursor::seek` and `Iterator::nth` reuse a cursor; `get` on a mix
  is 5 to 30 % faster from the same change (fewer, presized allocations).
- Rejected: a masked trade instead of `select_unpredictable` in the tree's replay (it would
  have lowered the minimum Rust version) costs about 3 ns per element on mixes, because it
  lengthens the dependency chain of every level; counting the repeat depth at run time while
  descending (it would have saved a compile-time pass) costs about 0.5 ns per element on
  shuffles. The depth stays in the node and weighted parts get a post-pass instead.

## Third review (2026-09-05)

- A shard of a nested mix re-seeked the inner mixes for every element it kept: the mix's
  `skip` stepped the outer interleave and left the parts' cursors behind, and the mismatch on
  the next draw *seeked* the part, a full interleave seek when the part is a mix. Now a part
  that has been entered skips forward to its next index, and a long hop (which re-seeks the
  outer interleave) leaves the parts where they are. The realistic order sharded eight ways
  went from 27.8 µs to 0.23 µs per element, a mix of two 100-part mixes from 1.85 µs to
  93 ns; flat mixes are unchanged. The hop at which a skip re-seeks the interleave instead of
  stepping it is twice the number of parts for uniform mixes and four times with schedules:
  at 100 parts a step costs 10 to 13 ns and a seek 1.8 µs uniform or 4.5 µs with a fifth of
  the parts scheduled, so the break-even is about 1.7 and 3.6 parts.
- Measured and reverted: a `u32` segment hint through the key computation (the tree's slot
  already stores it as `u32`) cost 0.6 ns per element on mixes of 100 parts and 2.3 ns on the
  realistic order in an interleaved three-way A/B (base, new, new with the hunk reverted);
  the hint stays `usize` in the walk and is narrowed at the slot.
- The seek column of the README's table times `iter(pos..).next()`: the parts of a mix are
  entered on the first element, so `iter(pos..)` alone measured 0.1 µs for the realistic
  order where positioning costs 30 µs, about a `get`.
