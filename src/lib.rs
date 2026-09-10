//! Deterministic ordering for large datasets: shuffle, seek anywhere, then stream.
//!
//! Shuffle and mix billions of records without storing a full index array. Ordering
//! memory grows with the sources and sequence structure, not the number of records.
//! Each lookup returns an [`Item`] with a source ordinal, source reference and record
//! index, leaving record loading to you.
//!
//! - **Shuffle on demand:** each shuffled index takes O(1) time on average and O(1) space.
//! - **Seek into a mix:** counting and binary searches locate each part's position
//!   without replaying the preceding records.
//! - **Walk from there:** a mix maintains a tournament tree, choosing each next part
//!   with O(log k) comparisons for `k` parts, instead of seeking again for every element.
//!
//! See the [cost model](#cost) for composition costs and the
//! [README](https://github.com/xmakro/dataorder/blob/main/README.md#performance) for benchmarks.
//!
//! Three types make up the main API:
//!
//! - [`Source`] describes a dataset by its length and an optional shuffle salt.
//! - [`Seq`] describes how to combine and transform sources.
//! - [`Order`] validates a `Seq` and provides random access and iteration.
//!
//! # Example
//!
//! A `usize` can represent a source when only its length matters:
//!
//! ```
//! use dataorder::{Order, Seq};
//!
//! // Shuffle a billion records and repeat for two epochs.
//! let seq = Seq::source(1_000_000_000).shuffle(42).repeat(2);
//! let order = Order::new(seq)?;
//! assert_eq!(order.len(), 2_000_000_000);
//!
//! // Start anywhere, without replaying the earlier positions.
//! let resume = 1_200_000_000;
//! let mut cursor = order.iter(resume..resume + 10)?;
//! let item = cursor.next().unwrap();
//! assert_eq!(item.source_ordinal, 0);
//! assert_eq!(*item.source, 1_000_000_000);
//! assert!(item.record_index < 1_000_000_000);
//! assert_eq!(item, order.get(resume).unwrap());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Position 1,200,000,000 belongs to the *order*; the returned index belongs to the original
//! source. [`Order::iter`] returns the same items as calling [`Order::get`] at each
//! position in its range, regardless of previous iteration or seeks.
//!
//! Implement [`Source`] for your dataset handles, or use slices, arrays or vectors.
//! `Order` owns its sources and yields references to them. A `Seq` can be cloned,
//! compared, hashed, [mapped to another source type](Seq::map) and optionally
//! [serialized](#feature-flags). `Seq<T>` accepts any `T`; only constructing an
//! `Order<T>` requires `T: Source`.
//!
//! # Semantics
//!
//! Each node maps its positions to positions in its children. In this table, `n` is
//! the child's length and `p` is a position within the node.
//!
//! | Node | Length | Meaning |
//! | --- | --- | --- |
//! | [`Source`](Seq::Source) | Source length | Index `p` of the source |
//! | [`Concat`](Seq::Concat) | Sum of part lengths | Parts read one after another |
//! | [`Mix`](Seq::Mix) | Sum of part lengths | Parts interleaved, preserving each part's order |
//! | [`Shuffle`](Seq::Shuffle) | `n` | A seeded permutation of the child's positions |
//! | [`Repeat`](Seq::Repeat) | `times × n` | Index `p % n` in epoch `p / n` |
//! | [`Cycle`](Seq::Cycle) | `len` | Like repeat, with the last epoch truncated as needed |
//! | [`Skip`](Seq::Skip) | `n − skip` | Child position `skip + p` |
//! | [`Take`](Seq::Take) | `take` | Child position `p` |
//! | [`StepBy`](Seq::StepBy) | `⌈n / step⌉` | Child position `p × step` |
//!
//! A mix uses every element of every part once. Set each part's exact count with
//! [`Seq::cycle`] before mixing. A mix preserves the order within each part;
//! add a shuffle to a part to change that order.
//!
//! [`Sampling`] assigns each part's elements keys on a shared virtual clock.
//! Curves are independent: `Uniform` is constant in virtual time, and all parts
//! adapt equally when their keys are merged. Clock values are not fractions of
//! final output progress; a delay of 0.5 need not start halfway through the output,
//! and a linear ramp generally becomes nonlinear against output positions. See
//! [`Sampling`] for the conversion and an example. Overlaps and gaps are allowed.
//! Schedules belong to their mix:
//! repeating a mix restarts its schedules each epoch. To schedule over several epochs,
//! repeat the parts and mix them once. `skip(index).step_by(count)` partitions the
//! resulting positions among workers; see [`Seq::step_by`] for worker configuration,
//! costs, and the difference between partitioning a mix and partitioning its parts. The global position partition does not
//! guarantee a balanced dataset mix on each worker: two equal interleaved parts
//! sharded two ways send one part exclusively to each worker.
//!
//! # Shuffles and repetitions
//!
//! A shuffle visits every child position exactly once. Its permutation depends on:
//!
//! - The shuffle's seed and the order's seed.
//! - The repetition context, derived from enclosing repeats and their nesting depths.
//! - The salts and original lengths of sources retained under the shuffle, in order
//!   of appearance.
//!
//! Give datasets stable [`Source::salt`] values to distinguish their shuffles when
//! their lengths and seeds match. [`Order::set_seed`] changes the seed for all shuffles
//! without rebuilding the order. Shuffles use a six-round Feistel permutation with
//! cycle walking; they are intended for reproducible ordering, not cryptography.
//!
//! `x.shuffle(seed).repeat(3)` selects a shuffle for each epoch. The first epoch keeps
//! its original context. In contrast, concatenating three copies of `x.shuffle(seed)`
//! repeats the same order. Repetition alone does not add a shuffle.
//!
//! Nested repeats need care: adding an outer repeat with more than one epoch increases
//! the depth of inner repeats. Their later epochs can then change even during the
//! outer repeat's first epoch. Extending a sequence with `cycle` has the same effect
//! if it introduces another epoch. A single repetition, or a cycle
//! within the existing length, preserves the prefix.
//!
//! Empty sources and subtrees do not contribute to a shuffle's salt. A skip or take
//! also removes concatenation parts that it excludes entirely. Other combinations,
//! such as a stride over a concatenation or a slice of a mix, can retain sources even
//! when the selected positions do not reach them.
//!
//! # Validation and limits
//!
//! [`Order::new`] returns an [`Error`] with a kind and a path to the invalid node.
//! All sequence builders defer validation until an order is built, including parts
//! that would contribute no elements, such as the child of `repeat(0)`.
//! Configurations support up to [`MAX_DEPTH`] levels and use ordinary recursive
//! traversal and destruction. Arbitrarily deep hand-built trees are unsupported.
//!
//! Skips and takes must stay within the child sequence. `step_by` requires a nonzero
//! step, and an empty sequence cannot be cycled to a positive length. Schedules must
//! have valid parameters and satisfy their individual numerical limits; see
//! [`Sampling`] and [`ErrorKind`] for the full rules.
//!
//! Lengths and positions use `usize` in the public API and `u64` internally. The final
//! order must fit in `usize`; on a 32-bit target, intermediate nodes may be longer.
//! A mix is limited to [`MAX_MIX_LEN`] elements, and configuration depth is limited to
//! [`MAX_DEPTH`]. [`Seq`] documents stack use; its builders work with any source type.
//!
//! Compilation simplifies nodes without changing their order. It flattens nested
//! concatenations, removes empty parts, merges nested strides, and folds skips and
//! takes into sources or slices where possible. A mix with one non-empty part becomes
//! that part; a shuffle of at most one element and a single repetition need no wrapper.
//! A cycle that fits within one epoch becomes a take. Source handles remain available
//! through [`Order::sources`], including those whose nodes were removed.
//!
//! [`Order::get`] returns `None` for an invalid position. [`Order::iter`],
//! [`Cursor::seek`] and [`Cursor::set_range`] report [`BoundsError`] instead of
//! panicking on invalid bounds. Failed cursor operations
//! leave their state unchanged. [`Cursor::offset`] reads the next absolute position;
//! `position(predicate)` remains the standard consuming iterator search.
//! Every [`Item`] includes its source's ordinal in [`Order::sources`], so equal and
//! zero-sized sources can be distinguished. Ordinals are local to an order.
//!
//! # Cost
//!
//! Storage depends on the configuration and cursor state, not on the number of output
//! elements. Compilation can revisit subtrees when flattening concatenations, deriving
//! shuffle salts or adjusting repeat depths. Each mix builds independent profiles
//! in `O(k)` time for `k` parts.
//!
//! For random access, [`Order::get`] follows the path from the root to a source:
//!
//! | Node | Work at that node |
//! | --- | --- |
//! | Concat | `O(log k)` search over `k` part offsets |
//! | Shuffle | Constant average permutation cost; an individual position can take longer |
//! | Mix | A seek over its parts, with the cost described below |
//! | Repeat, skip, take, step by | Position arithmetic |
//!
//! For a mix with `k` non-empty parts, seeking counts elements below a trial virtual
//! time, then replays at most `2k` tournament steps. It tries `position / N` first,
//! which is a good estimate for uniform mixes. Scheduled mixes generally require
//! rank interpolation or bisection because virtual time differs from output progress.
//! After at most eight interpolation probes, there are at most 63 virtual-time
//! bisections, each counting all `k` parts. Each count starts with
//! a constant-time CDF estimate; correcting it against actual keys takes at most
//! 46 index bisections. Profiles have at most five segments. Building and replaying
//! the tournament costs `O(k log(k + 1))`; long equal-key runs are handled by counts.
//!
//! Sequential iteration keeps cursor state. A mix uses `⌈log2 k⌉` tournament comparisons
//! per element plus one key computation, with no comparisons once one part remains.
//! A shuffle reads scattered child positions, so **shuffling a mix pays for a mix seek
//! per element**. Its cursor retains seek buffers for every mix reached underneath
//! that shuffle, allocating on the first visit and reusing them thereafter. Shuffle
//! the parts before mixing when that is the order you need.
//!
//! Stepping through a sequence skips unselected child positions. Mixes advance their
//! interleave for short skips and seek for longer ones. Sharding a mix across `count`
//! workers can therefore multiply the total interleaving work by up to `count`;
//! see [`Seq::step_by`].
//!
//! Cursor allocations are deferred until needed. Empty ranges and `count()` allocate
//! nothing. [`Cursor::seek`], [`Cursor::set_range`] and [`Iterator::nth`] reuse existing
//! buffers, including in a cloned cursor, though entering a new child can allocate.
//! Concat transitions recycle compatible child buffers, including mix seek state.
//! Retained capacities can reflect the largest previously visited compatible child;
//! incompatible variants and removed child states are dropped rather than cached.
//! `last()` reuses initialized state; selecting an empty range defers repositioning.
//! For benchmark workloads and commands, see the
//! [README's performance section](https://github.com/xmakro/dataorder/blob/main/README.md#performance).
//!
//! # Feature flags
//!
//! The optional `serde` feature derives `Serialize` and `Deserialize` for [`Seq`],
//! [`MixPart`] and [`Sampling`]. It uses serde's derived representation,
//! with the documented variant and field names. For example:
//!
//! ```json
//! {"Shuffle":{"seed":1,"inner":{"Source":50}}}
//! ```
//!
//! Unknown fields are rejected. Changes to this format follow the [stability policy](#stability).
//! The source type must also support serialization and deserialization.
//!
//! For JSON, enable `float_roundtrip` on your own `serde_json` dependency:
//!
//! ```toml
//! [dependencies]
//! dataorder = { version = "0.4", features = ["serde"] }
//! serde_json = { version = "1", features = ["float_roundtrip"] }
//! ```
//!
//! Without it, parsing can change a breakpoint by one representable `f64`
//! step, affecting equality and possibly the order. `dataorder/serde` does not enable
//! this JSON parser feature. Also keep these limits in mind:
//!
//! - NaN and infinite parameters are invalid configurations. `serde_json` writes them
//!   as `null`, which does not deserialize back into an `f64`.
//! - Lengths and counts are `usize`; a configuration written on a 64-bit machine may
//!   not fit on a 32-bit machine.
//! - `serde_json`'s recursion limit also counts surrounding objects and arrays, so a
//!   configuration embedded in a larger document can reach that parser limit.
//!   Deserialization does not validate the configuration. Resolve its sources with
//!   [`Seq::map`] or [`Seq::try_map`] as needed, then compile it with [`Order::new`].
//!
//! # Stability
//!
//! The same configuration, source lengths and salts, and seed produce the same order
//! on supported platforms. A release that changes an order or the serialized
//! configuration format is a breaking change: a new minor version while the crate is 0.x.
//! Save [`ORDERING_VERSION`] in checkpoints for a conservative exact-version check.
//!
//! Calculations use IEEE 754 binary64 with separate multiply/add expressions.
//! Targets with hardware or software binary64 agree; x87-only `i586`
//! targets using extended precision are excluded. Golden tests pin order fingerprints
//! through the public API, and CI checks 64-bit and 32-bit x86 and 64-bit ARM.

#![forbid(unsafe_code)]
#![warn(missing_docs, unreachable_pub, clippy::doc_markdown, clippy::redundant_clone, clippy::use_self)]
// Fusing multiply/add expressions changes rounding and can change the order;
// Clippy must not suggest doing so.
#![allow(clippy::suboptimal_flops)]
#![cfg_attr(docsrs, feature(doc_cfg))]

mod bounds;
mod cursor;
mod error;
mod interleave;
mod order;
mod perm;
mod seq;
mod source;
#[cfg(test)]
mod tests;

pub use bounds::BoundsError;
pub use cursor::Cursor;
pub use error::{Error, ErrorKind, SamplingDetail};
pub use interleave::Sampling;
pub use order::{Item, Order};
pub use seq::{MixPart, Seq};
pub use source::{Source, salt, salt_path};

/// Float bits for equality and hashing, treating `-0.0` and `0.0` as equal.
pub(crate) fn float_bits(x: f64) -> u64 {
    let bits = x.to_bits();
    if bits << 1 == 0 { 0 } else { bits }
}

/// Maximum configuration depth accepted by [`Order::new`], counting the root as level 1.
///
/// A source alone has depth 1. Each enclosing transform adds a level, so a source
/// with `MAX_DEPTH` transforms is too deep. The limit bounds recursive compilation;
/// deeper configurations are rejected during compilation. Tree destruction and
/// other operations still recurse, so arbitrarily deep inputs are unsupported.
/// See [`Seq`] for stack use.
///
/// ```
/// use dataorder::{ErrorKind, MAX_DEPTH, Order, Seq};
/// let chain = |levels: u32| (1..levels).fold(Seq::source(10), |s, _| s.take(10));
/// assert_eq!(Order::new(chain(MAX_DEPTH)).map(|order| order.len()), Ok(10));
/// assert_eq!(Order::new(chain(MAX_DEPTH + 1)).unwrap_err().kind(), &ErrorKind::TooDeep);
/// ```
pub const MAX_DEPTH: u32 = 16;

/// Maximum mix length accepted by [`Order::new`]: 2⁴⁶ elements.
///
/// This limit leaves room for floating-point rounding between a part's consecutive
/// keys and keeps counts exactly representable in `f64`. Scheduled parts must also
/// satisfy `length × peak rate ≤ MAX_MIX_LEN`. Repeating or concatenating valid
/// mixes can produce longer orders.
///
/// ```
/// use dataorder::{ErrorKind, MAX_MIX_LEN, Order, Seq};
/// assert_eq!(MAX_MIX_LEN, 1 << 46);
/// let long = Seq::mix([Seq::source(1 << 30).repeat(1 << 16), Seq::source(1)]);
/// assert_eq!(Order::new(long).unwrap_err().kind(), &ErrorKind::MixTooLong);
/// let repeated = Seq::mix([Seq::source(1 << 30), Seq::source(1)]).repeat(3);
/// assert_eq!(Order::new(repeated).map(|order| order.len()), Ok(3 * (1 << 30) + 3));
/// ```
pub const MAX_MIX_LEN: u64 = interleave::MAX_TOTAL_LEN;

/// Version identifier for conservatively validating saved orders and checkpoints.
///
/// This is the version of the linked `dataorder` crate, including its patch version,
/// rather than the calling application's `CARGO_PKG_VERSION`. Save it alongside the
/// configuration, source metadata, seeds and worker settings. Require an exact match
/// on restore unless the application has explicitly verified a compatible migration.
/// A difference does not necessarily mean that the ordering changed; see the crate's
/// [stability policy](crate#stability).
pub const ORDERING_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The README's code blocks, compiled as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
