//! Deterministic, seekable data order for training, without materializing anything.
//!
//! A [`Seq`] is a tree: [`Source`](Seq::Source) leaves (anything that is a [`Source`]: a
//! length) combined by [`Concat`](Seq::Concat), [`Mix`](Seq::Mix) (balanced,
//! order-preserving interleaving with per-part sampling schedules, see [`Sampling`]) and
//! [`Weighted`](Seq::Weighted) (a mix in given proportions, repeating and cutting the parts)
//! and transformed by [`Shuffle`](Seq::Shuffle), [`Repeat`](Seq::Repeat), [`Skip`](Seq::Skip),
//! [`Take`](Seq::Take) and [`Stride`](Seq::Stride). [`Order::new`] validates it and
//! precomputes what iteration needs; the elements, `(&source, index in the source)`, are
//! never materialized: [`Order::get`] computes any position and [`Order::iter`] walks any
//! range.
//!
//! ```
//! use dataorder::{Order, Sampling, Seq, Source};
//!
//! struct Shard { path: &'static str, len: usize }
//! impl Source for Shard {
//!     fn len(&self) -> usize { self.len }
//!     fn salt(&self) -> u64 { dataorder::salt(self.path) }
//! }
//!
//! // Worker 0 of 8: each part is sharded, then the shards are mixed (see Cost).
//! let seq = Seq::mix_with([
//!     (Seq::source(Shard { path: "web.bin", len: 1_000_000 }).shuffle(1).repeat(3).shard(8, 0), Sampling::Uniform),
//!     (Seq::source(Shard { path: "code.bin", len: 200_000 }).shuffle(2).shard(8, 0), Sampling::delayed(0.5)),
//! ]);
//! let order = Order::new(seq)?;
//! for (shard, index) in order.iter(1000..1010) {
//!     println!("element {index} of {}", shard.path);
//! }
//! let (shard, index) = order.get(1005);
//! assert_eq!(order.iter(1000..1010).nth(5).map(|(s, i)| (s.path, i)), Some((shard.path, index)));
//! # Ok::<(), dataorder::Error>(())
//! ```
//!
//! A `Seq` is plain data: clone it, compare and hash it, serialize it (see
//! [feature flags](#feature-flags)), and [`Seq::map`] its sources from handles to loaded
//! datasets while keeping the structure. A bare `usize` is a source too, when only the order
//! matters. Lengths and positions are `usize`; inside, the arithmetic is 64-bit, so on a
//! 32-bit target an intermediate node may exceed the address space as long as the order
//! itself does not.
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
//! | `Weighted { total, parts }` | `total` | the mix of each part repeated and cut to `round(wᵢ/Σw · total)` |
//! | `Shuffle { seed, inner }` | `n` | `perm_seed(p)` of `inner` |
//! | `Repeat { times, inner }` | `times·n` | `p mod n` of `inner`, in the context of epoch `p div n` |
//! | `Skip { n, inner }` | `len − n` | `n + p` of `inner` |
//! | `Take { n, inner }` | `n` | `p` of `inner` |
//! | `Stride { step, offset, inner }` | `⌈(n − offset) / step⌉`, or 0 | `offset + p·step` of `inner` |
//!
//! Shuffles are seeded permutations of `0..n` (a keyed six-round Feistel network with cycle
//! walking, see `src/perm.rs`): O(1) per element, no state. A shuffle's permutation depends
//! on its `seed`, on the order's seed, on the *context*, which every `Repeat` of more than
//! one repetition on the path above derives afresh for each repetition after its first,
//! and on the sources under it that have elements, their [salts](Source::salt) and lengths
//! in order of appearance (an empty source, a part that is empty as a whole, or a part of a
//! concatenation that a skip or take above cuts away entirely, does not count; a stride, or
//! a slice of anything but a concatenation, does not cut parts away). So
//! `x.shuffle(s).repeat(3)` is `x.shuffle(s)` followed by two other orders
//! of `x`, `Seq::concat([x.shuffle(s), x.shuffle(s)])` repeats one order, `x.repeat(1)` is
//! `x`, and `Seq::mix([a.shuffle(s), b.shuffle(s)])` orders `a` and `b` alike only when
//! they have the same length and salt: give sources a salt, or shuffles their own seeds.
//! Everything is deterministic in the configuration and the order's seed, and `iter(a..b)`
//! yields exactly `get(a)..get(b)` whatever was iterated before.
//!
//! Compilation rejects skips and takes past the end, zero strides, orders longer than
//! `usize::MAX` (and intermediate lengths beyond 64 bits), invalid or overcommitted
//! schedules, invalid weights, mixes longer than [`MAX_MIX_LEN`] and nesting deeper than
//! [`MAX_DEPTH`]; the [`Error`] names the kind of problem and the path of the node. Two
//! builders panic instead, on mistakes no configuration can express (see [`Seq`]).
//! Compilation folds what is exact: nested concats flatten, empty parts vanish (an empty
//! part of a mix does not affect the order of the others, nor does an empty source or part
//! the shuffles above it), skips and takes merge into sources, slices and strides and narrow
//! a concatenation to the parts they touch, nested strides merge, a mix with a single
//! non-empty part is that part, a shuffle of at most one element is that element, and a
//! single repetition is the sequence.
//!
//! A schedule is relative to the mix it belongs to, so `mix.repeat(n)` restarts every
//! schedule in each repetition; to schedule over a whole run of several epochs, repeat the
//! parts and mix them once, as the example above does.
//!
//! # Cost
//!
//! Compilation is linear in the configuration (plus `O(k + s log s)` per mix of `k` parts,
//! `s` of them scheduled). [`Order::get`] walks the path from the root to a source: constant
//! work per node, except that a `Mix` costs a seek of the interleave (`O(k log s)`, which
//! allocates) and a `Shuffle` a key derivation. [`Order::iter`] seeks once and then walks:
//! a `Mix` costs `⌈log2 k⌉` comparisons per element plus one key computation, a `Shuffle`
//! one permutation plus a [`Order::get`]-style descent into its child (so a shuffle *over*
//! a mix pays the interleave seek per element; shuffle the parts, not the mix), a `Stride`
//! skips `step − 1` elements of its child (a mix steps its interleave, or re-seeks it when
//! that is cheaper, and its parts skip along, a nested mix stepping its own interleave), so
//! sharding a mix across `count` workers costs `count` times its interleaving in total
//! (shard the parts instead when that matters). `Concat`, `Repeat`, `Skip` and `Take` add a
//! few instructions. Creating a cursor allocates one cursor per node it enters and seeks;
//! [`Cursor::seek`] and [`Iterator::nth`] reuse the cursor's buffers. The README has
//! measured numbers.
//!
//! # Feature flags
//!
//! - `serde`: derives `Serialize` and `Deserialize` for [`Seq`], [`MixPart`],
//!   [`WeightedPart`] and [`Sampling`] (pulls in `serde` with `derive`). The format is
//!   serde's derived representation with the variant and field names as written here,
//!   `{"Shuffle":{"seed":1,"inner":{"Source":50}}}` for instance; unknown fields are
//!   rejected in every variant. It is stable under the same policy as the orders: a change
//!   to it is a breaking change. Only configurations [`Order::new`] accepts round-trip:
//!   `serde_json` writes an infinite or NaN weight or schedule parameter as `null`, which
//!   does not read back. Two more caveats: lengths and counts are `usize`, so a
//!   configuration written on a 64-bit machine need not read back on a 32-bit one; and
//!   every level of a `Seq` is two levels of nesting in a self-describing format, so
//!   `serde_json` reads at most 64 levels under its default recursion limit of 128
//!   (`Deserializer::disable_recursion_limit`, behind its `unbounded_depth` feature, lifts
//!   it).
//!
//! # Stability
//!
//! An order is a pure function of the configuration and the seed: the same `Seq` and seed
//! give the same elements on every platform and in every release that does not say
//! otherwise. The arithmetic is IEEE 754 binary64, correctly rounded and without fused
//! operations (Rust never contracts them), so it agrees on every target whose `f64` is
//! hardware or software binary64; the x87-only `i586` targets, which compute in extended
//! precision, are excluded. A release that changes any order, or the serialized form of a
//! configuration, is a breaking change (a new minor version while the crate is 0.x) and is
//! listed in `CHANGELOG.md`. Golden tests in `tests/golden.rs` pin fingerprints of a dozen
//! orders through the public API, and CI runs them on 64-bit and 32-bit x86 and on 64-bit
//! ARM.

#![forbid(unsafe_code)]
#![warn(missing_docs, unreachable_pub, clippy::doc_markdown, clippy::redundant_clone, clippy::use_self)]
// No `mul_add`, however clippy's pedantic group may put it: a fused multiply-add rounds
// differently from a multiply and an add, and the orders are promised to be the same on
// every target.
#![allow(clippy::suboptimal_flops)]
#![cfg_attr(docsrs, feature(doc_cfg))]

mod cursor;
mod error;
mod interleave;
mod order;
mod perm;
mod seq;
mod source;
#[cfg(test)]
mod tests;

pub use cursor::Cursor;
pub use error::{Error, ErrorKind};
pub use interleave::Sampling;
pub use order::Order;
pub use seq::{MixPart, Seq, WeightedPart};
pub use source::{Source, salt};

/// The bits of a float with `-0.0` taken as `0.0`: what equality and hashing of schedules
/// and weights compare.
pub(crate) fn float_bits(x: f64) -> u64 {
    (x + 0.0).to_bits()
}

/// Deepest nesting [`Order::new`] accepts, the root counting as level 1: a chain of
/// `MAX_DEPTH` nested transforms over a source is one level too many. Compilation recurses
/// once per level, and this keeps it well inside the default stack of a thread; a deeper
/// configuration is rejected without recursing into the rest of it (see [`Seq`]).
///
/// ```
/// use dataorder::{ErrorKind, MAX_DEPTH, Seq};
/// let chain = |levels: u32| (1..levels).fold(Seq::source(10), |s, _| s.take(10));
/// assert_eq!(chain(MAX_DEPTH).check(), Ok(10));
/// assert_eq!(chain(MAX_DEPTH + 1).check().unwrap_err().kind(), &ErrorKind::TooDeep);
/// ```
pub const MAX_DEPTH: u32 = 256;

/// Longest mix [`Order::new`] accepts: 2⁴⁶ elements. It keeps the gap between consecutive
/// keys of one part far above floating-point rounding and every count exact in `f64`. A
/// scheduled part must likewise satisfy `length × peak rate ≤ MAX_MIX_LEN`. Longer orders
/// are possible by repeating, concatenating or striding mixes.
///
/// ```
/// use dataorder::{ErrorKind, MAX_MIX_LEN, Seq};
/// assert_eq!(MAX_MIX_LEN, 1 << 46);
/// let long = Seq::mix([Seq::source(1 << 30).repeat(1 << 16), Seq::source(1)]);
/// assert_eq!(long.check().unwrap_err().kind(), &ErrorKind::MixTooLong);
/// let repeated = Seq::mix([Seq::source(1 << 30), Seq::source(1)]).repeat(3);
/// assert_eq!(repeated.check(), Ok(3 * (1 << 30) + 3));
/// ```
pub const MAX_MIX_LEN: u64 = interleave::MAX_TOTAL_LEN;

/// The README's code blocks, compiled as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
