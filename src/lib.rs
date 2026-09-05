//! Deterministic, seekable data order for training, without materializing anything.
//!
//! A [`Seq`] is a tree: [`Source`](Seq::Source) leaves (anything that is a [`Dataset`]: a
//! length) combined by [`Concat`](Seq::Concat) and [`Mix`](Seq::Mix) (balanced,
//! order-preserving interleaving with per-part sampling schedules, see [`Sampling`]) and
//! transformed by [`Shuffle`](Seq::Shuffle), [`Repeat`](Seq::Repeat), [`Skip`](Seq::Skip),
//! [`Take`](Seq::Take) and [`Stride`](Seq::Stride). [`Order::compile`] checks it and precomputes what iteration
//! needs; the elements, `(&source, index in the source)`, are never materialized:
//! [`Order::get`] computes any position and [`Order::iter`] walks any range.
//!
//! ```
//! use dataorder::{Dataset, Order, Sampling, Seq};
//!
//! struct Shard { path: &'static str, len: usize }
//! impl Dataset for Shard {
//!     fn len(&self) -> usize { self.len }
//! }
//!
//! let seq = Seq::mix_with([
//!     (Seq::source(Shard { path: "web.bin", len: 1_000_000 }).shuffle(1).repeat(3), Sampling::Uniform),
//!     (Seq::source(Shard { path: "code.bin", len: 200_000 }).shuffle(2), Sampling::DelayedLinear { start: 0.5, full: 0.5 }),
//! ])
//! .shard(0, 8);
//! let order = Order::compile(seq)?;
//! for (shard, index) in order.iter(1000..1010) {
//!     // element `index` of `shard.path`
//! }
//! let (shard, index) = order.get(1005);
//! assert_eq!(order.iter(1000..1010).nth(5).map(|(s, i)| (s.path, i)), Some((shard.path, index)));
//! # Ok::<(), dataorder::Error>(())
//! ```
//!
//! A `Seq` is plain data: clone it, compare it, serialize it (with your own `T`), and
//! [`Seq::map`] its sources from handles to loaded datasets while keeping the structure.
//! A bare `usize` is a source too, when only the order matters. Lengths and positions are
//! `usize`; inside, the arithmetic is 64-bit, so on a 32-bit target an intermediate node may
//! exceed the address space as long as the order itself does not.
//!
//! # Semantics
//!
//! Every element has a position in the sequence of its node; a node maps its positions to
//! positions of its children:
//!
//! | node | length | position `p` maps to |
//! |---|---|---|
//! | `Source(t)` | `t.len()` | element `p` of `t` |
//! | `Concat(parts)` | sum | the part containing `p`, at `p` minus the part's offset |
//! | `Mix(parts)` | sum | what the interleave of the parts' lengths puts at `p` |
//! | `Shuffle { seed, inner }` | `n` | `perm_seed(p)` of `inner` |
//! | `Repeat { times, inner }` | `times·n` | `p mod n` of `inner`, in the context of epoch `p div n` |
//! | `Skip { n, inner }` | `len − n` | `n + p` of `inner` |
//! | `Take { n, inner }` | `n` | `p` of `inner` |
//! | `Stride { step, offset, inner }` | `⌈(n − offset) / step⌉` | `offset + p·step` of `inner` |
//!
//! Shuffles are seeded permutations of `0..n` (a keyed six-round Feistel network with cycle
//! walking, see `src/perm.rs`): O(1) per element, no state. A shuffle's permutation depends on its
//! `seed`, on the order's seed and on the *context*, which every `Repeat` on the path above
//! derives afresh for each repetition after its first, so `x.shuffle(s).repeat(3)` is
//! `x.shuffle(s)` followed by two other orders of `x`, while
//! `Seq::concat([x.shuffle(s), x.shuffle(s)])` repeats one order. Everything is
//! deterministic in the configuration and the order's seed.
//!
//! # Cost
//!
//! Compilation is linear in the configuration (plus `O(k + s²)` per mix of `k` parts, `s`
//! of them scheduled). [`Order::get`] walks the path from the root to a source: constant
//! work per node, except that a `Mix` costs a seek of the interleave (`O(k log s)`) and a
//! `Shuffle` a key derivation. [`Order::iter`] seeks once and then walks: a `Mix` costs
//! `⌈log2 k⌉` comparisons per element plus one key computation (10 ns at `k = 100`
//! sources, 16 ns with shuffled parts), a `Shuffle` one permutation (about 4.5 ns) plus a
//! [`Order::get`]-style descent into its child (so a shuffle *over* a mix pays the
//! interleave seek per element; shuffle the parts, not the mix), a `Stride` skips
//! `step − 1` elements of its child (re-seeking a mix when that is cheaper). `Concat`,
//! `Repeat`, `Skip` and `Take` add a few instructions. `cargo run --release --example bench`
//! measures these; the README has the table and what was measured and kept or rejected.
//!
//! # Layout
//!
//! | file | contents |
//! |---|---|
//! | `src/seq.rs` | [`Seq`] and its builder methods |
//! | `src/dataset.rs` | [`Dataset`] |
//! | `src/error.rs` | [`Error`] |
//! | `src/order.rs` | [`Order`]: compilation with folds, the node tree, random access |
//! | `src/cursor.rs` | [`Cursor`]: per-node cursors with seek, next and skip |
//! | `src/perm.rs` | seeded permutations of `0..n` and context derivation |
//! | `src/interleave/` | the mix: `mod.rs` model and construction, `iter.rs` seek and walk, `profile.rs` rate profiles, `tournament.rs` loser tree, `sampling.rs` schedules, `tests.rs` |
//! | `src/tests.rs` | whole-crate tests against a materializing reference evaluator, golden orders |
//!
//! # Stability
//!
//! An order is a pure function of the configuration and the seed: the same `Seq` and seed
//! give the same elements on every platform (the arithmetic is IEEE 754 binary64, correctly
//! rounded, without fused operations) and in every release that does not say otherwise. A
//! release that changes any order is a breaking change (a new minor version while the
//! crate is 0.x). Golden tests in `src/tests.rs` pin fingerprints of a dozen orders, and CI
//! runs them on 64-bit and 32-bit x86 and on 64-bit ARM.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod cursor;
mod dataset;
mod error;
mod interleave;
mod perm;
mod order;
mod seq;
#[cfg(test)]
mod tests;

pub use cursor::Cursor;
pub use dataset::Dataset;
pub use error::Error;
pub use interleave::Sampling;
pub use order::Order;
pub use seq::Seq;
