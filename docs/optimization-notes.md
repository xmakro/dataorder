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

## Fourth review (2026-09-05)

Measured with the interleaved harness (base and new binaries alternating, ABBA order, min of
three or four runs each, one core); an A/A run of the same binary against itself puts the noise
at ±0.3 ns on most rows and ±1 ns on the realistic order.

- A seek of a mix walked the profile's segments from segment 0 for every part, so with `S`
  distinct schedule breakpoints among the parts it cost `O(k · S)`: 9.5 ms for a mix of 10 000
  parts with 2000 distinct `delayed` starts, against 0.26 ms with one start. `quantile` now
  walks its hint at most 16 segments and then searches the shares (kept contiguous alongside
  the segment starts, so that the searches touch a few cache lines rather than one 72-byte
  segment each): 1.17 ms, and the rows with a few distinct starts gained 5 to 7 % on seeks
  and gets. The walk is unchanged, because a key moving to the next segment takes the same
  comparisons as before. The README's interleave table has the new column.
- The seventh Feistel round costs 0.9 ns per permuted element (bare shuffle 8.1 → 9.0 ns),
  0.7 to 0.9 ns on mixes of shuffled sources and about 1.8 ns on the realistic order. Six
  rounds left about one key in 300 with a visible structure in consecutive differences (a
  64-bin chi-square at 5 to 47σ), at every size tried, not only at powers of two.
- Measured and rejected: boxing the per-part cursor slots of a mix (16 bytes per part instead
  of about 220, 3× less memory for a cursor over 100 000 parts) costs 0.5 to 1.5 ns per
  element on every mix and 2.9 ns on a mix of mixes, one dependent load per level per element.
- Nested strides now fold into one (a shard of a shard walked 3 ns per level slower), a slice
  of a concatenation keeps only the parts it touches, and a prefix take of a repeat is the
  repeat cut short (a weighted part is one node less deep); none of these is measurable on
  the table's rows.

## Numerical review (2026-09-05)

Compared the `dab2636` development snapshot with the fixes on an unpinned Apple Silicon Mac,
using optimized builds of `examples/bench.rs` and small public-API probes. These are local
checks, not replacements for the pinned Ryzen measurements above. The ordinary benchmark
was run before and after the changes, then repeated after restoring the zero-start ramp's
fast inverse. The following ordinary rows are the last fixed/baseline pair:

| operation | before | after |
|---|---|---|
| `get` at the midpoint of a trillion uniform elements plus one fading element | 1.08 s | below 1 µs |
| cursor allocation, two live parts plus 100,000 empty parts | 29,794,880 bytes | 704 bytes |
| `get`, two live parts plus 100,000 empty parts | 686 µs | below 1 µs |
| `get`, 100 shuffled parts, 20% scheduled | 6.53 µs | 6.58 µs |
| `get`, 1000 shuffled parts, 20% scheduled | 62.7 µs | 63.5 µs |
| walk, 1000 shuffled parts, 20% scheduled | 58.5 ns | 58.1 ns |
| `get`, nested realistic order | 45.8 µs | 44.8 µs |

The allocation figures count requested bytes during cursor creation, not RSS. With the
fix, adding empty parts allocates exactly the same runtime state as the two-part order;
source handles still occupy storage in the compiled order. Ordinary lookup differences
at this scale are sensitive to scheduling and measurement noise. The first-row seek
timings varied substantially between runs, so they do not support a speedup claim.

- Rising profiles with a positive starting rate use a stable quotient and canonicalize it
  against their monotone integral. Zero-start ramps retain their square-root/multiply path:
  canonicalizing those too added about 14% to scheduled seeks without improving correctness.
- Slope events remove the previous contribution before adding the new one. An exact
  floating-point expansion retains even a third, much smaller slope across cancellation;
  a two-float accumulator alone was insufficient. This extra work occurs during compilation.
- Seek corrections have bounded local adjustments followed by bisection, and consume a
  long run of equal keys by counts. The numerical regression tests run in subprocesses
  with a timeout, so a recurrence of an infinite loop or stack abort fails the test suite.
- Compilation first moves sources into a vector and builds a tree of source indices.
  This adds a construction pass and temporary storage, while making recursive frames
  independent of the source type's size. A maximum-depth tree with 8 KiB array sources
  is tested on a 2 MiB thread stack in debug and release builds.

## Shuffle and API review (2026-09-05)

The seven multiply-only Feistel rounds still retained strong adjacency patterns:
`source(56_444).shuffle(1)` had serial correlation 0.0512 and consecutive-difference
chi-square 556.7, and `source(65_536).shuffle(18_437)` had difference chi-square 13,622.
The replacement uses six rounds of the full `SplitMix64` finalizer with independently
derived round keys. The reported difference statistics fall to 74.94 and 58.74. The
release suite checks 512 independent public seed/length pairs (79.4 million permutations),
including lengths on both sides of powers of two. A separate 10,000-pair holdout over
lengths 32,906–131,072 checked 767.6 million permutations: maximum difference chi-square
110.01 and maximum absolute serial correlation 3.67 standard deviations. These checks
support the change but do not prove statistical quality for every possible configuration.

Public API timings below are medians of five alternating before/after runs on the same
unpinned Apple Silicon Mac, comparing the previous development version with this change.
They measure a shuffled source: walk by `next`, random `get`, and construction/seeking of
a cursor followed by its first element. The key shrinks from 112 to 48 bytes, with no new
allocation or dependency, but the stronger round costs about 3 ns per walked element.

| source length | walk, before → after | get, before → after | seek, before → after |
|---|---|---|---|
| 65,536 | 8.63 → 11.32 ns | 16.89 → 18.95 ns | 24.96 → 27.50 ns |
| 1,000,000 | 10.00 → 13.14 ns | 18.53 → 20.59 ns | 26.73 → 29.45 ns |
| 1,000,000,000 | 10.71 → 14.09 ns | 19.02 → 21.61 ns | 27.23 → 30.57 ns |

- Quantiles now use a strict lower bound on cumulative-share segment starts. An exact
  plateau height selects its left endpoint in both the hint walk and binary search; the
  change adds no arithmetic or allocation to iteration. Regression tests exercise all
  hints across 128 segments, neighboring float values, and the resulting mix's seeks.
- Tournament node buffers reserve against the full part-count upper bound, matching their
  existing value and scratch buffers. A cursor entered near the end can then seek backward
  without allocating when completed parts become live again. The allocation test starts
  with one long part and 99 already-finished short parts.
- Repeat documentation now describes depth-dependent inner epochs. A repeat, cycle or
  weighted share that adds an effective repetition can change a nested part's existing
  prefix. The context scheme is unchanged. Mapping and cloning likewise remain recursive;
  their depth guidance now accounts for large inline source values.
- The benchmark campaign resolves directory arguments before changing its working
  directory, rejects runs with no recognized rows and replaces a reused label's results
  completely. The seek benchmark now uses deterministic random positions rather than
  regularly spaced positions that can align with epoch and concat boundaries.
