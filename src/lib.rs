//! Deterministic ordering for large datasets: shuffle, seek anywhere, then stream.
//!
//! Shuffle and mix billions of records without storing a full index array. Ordering
//! memory grows with the sources and sequence structure, not the number of records.
//! Each lookup returns an [`Item`] with a source ordinal, source reference, record
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
//! let seq = Seq::source(1_000_000_000).repeat_shuffled(2);
//! let order = Order::with_seed(seq, 42)?;
//! assert_eq!(order.len(), 2_000_000_000);
//!
//! // Start anywhere, without replaying the earlier positions.
//! let resume = 1_200_000_000;
//! let mut cursor = order.cursor(resume..resume + 10)?;
//! let item = cursor.next().unwrap();
//! assert_eq!(item.source_ordinal, 0);
//! assert_eq!(*item.source, 1_000_000_000);
//! assert!(item.record_index < 1_000_000_000);
//! assert_eq!(item, order.get(resume).unwrap());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Position 1,200,000,000 belongs to the *order*; the returned index belongs to the original
//! source. [`Order::cursor`] returns the same items as calling [`Order::get`] at each
//! position in its range, regardless of previous iteration or seeks. [`Order::iter`]
//! visits the whole order and returns its cursor directly.
//!
//! Implement [`Source`] for your dataset handles, or use slices, arrays or vectors.
//! `Order` owns its sources and yields references to them. A `Seq` can be cloned,
//! compared, [mapped to another source type](Seq::map_sources) and optionally
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
//! | [`Repeat`](Seq::Repeat) | `times × n` | Index `p % n` in the unchanged input, or in a separate permutation per pass when `shuffled` |
//! | [`Cycle`](Seq::Cycle) | `len` | Like repeat, with the last pass truncated as needed |
//! | [`Skip`](Seq::Skip) | `n − skip` | Child position `skip + p` |
//! | [`Take`](Seq::Take) | `take` | Child position `p` |
//! | [`StepBy`](Seq::StepBy) | `⌈n / step⌉` | Child position `p × step` |
//!
//! A mix uses every element of every part once. Set each part's exact count with
//! [`Seq::cycle_to`] or [`Seq::cycle_to_shuffled`] before mixing. A mix preserves
//! the order within each part; use the shuffled variant to shuffle each pass.
//!
//! [`Schedule`] assigns each part's elements keys on a shared virtual clock, which is
//! not output progress; see [`Schedule`] for the model, its limits and an example.
//! `skip(index).step_by(count)` partitions the resulting positions among workers; see
//! [`Seq::step_by`] for worker configuration and costs.
//!
//! # Shuffles and repetitions
//!
//! A shuffle visits every child position exactly once. Its permutation depends on:
//!
//! - The order's seed.
//! - The input configuration's ordered source salts, original source lengths and
//!   shuffled layers, before pruning or flattening.
//!
//! The input configuration of a shuffle or shuffled repetition must contain no mixes,
//! including empty or single-part mixes and mixes nested under other operations.
//! Shuffle each input before mixing. [`Order::new`] reports
//! [`ErrorKind::ShuffleContainsMix`] at the enclosing shuffle.
//!
//! Give datasets stable [`Source::salt`] values to distinguish their shuffles when
//! their lengths match. [`Order::set_seed`] changes the seed for all shuffles
//! without rebuilding the order. Shuffles use a six-round Feistel permutation with
//! cycle walking; they are intended for reproducible ordering, not cryptography.
//!
//! `x.repeat(3)` and `x.cycle_to(len)` preserve the input's record order on every pass,
//! including any nested shuffles. `x.shuffle().repeat(3)` repeats one fixed
//! permutation three times.
//!
//! `x.repeat_shuffled(3)` and `x.cycle_to_shuffled(len)` permute the immediate input's
//! positions separately on each pass, including the first. Their permutation uses
//! the order's seed, the local pass number and the input's configuration salt.
//! `x.shuffle()` produces the same order as `x.repeat_shuffled(1)`.
//! Use [`Order::with_seed`] or [`Order::set_seed`] to select the seed for all shuffled operations.
//! Nested shuffles and shuffled repetitions keep their own permutations; enclosing
//! repeats never reseed them. Each shuffled layer derives a new configuration salt
//! for enclosing shuffles, so `x.shuffle().shuffle()` uses distinct permutation keys.
//! Distinct keys can still produce the same permutation, especially for small inputs.
//! Every child receives the unchanged order seed, with no enclosing epoch.
//! A shortened final pass takes a prefix of its full permutation.
//!
//! Plain repetition of a shuffled repetition replays the same series of permutations.
//! Nested shuffled repetitions each permute their own input and therefore do not
//! flatten into one shuffled repetition.
//!
//! `x.repeat(3).repeat(2)` has the same order as `x.repeat(6)`.
//! Adding, reordering or nesting mixture
//! inputs does not reseed existing inputs, including those with nested repetitions.
//! Their own configurations and the order seed determine their permutations.
//!
//! Adding an outer repeat preserves the entire first pass. Extending an outermost
//! repeat or cycle, plain or shuffled, preserves the existing prefix. A selection fixes the positions
//! that subsequent operations can draw from: `x.repeat_shuffled(1).take(2).repeat_shuffled(2)`
//! permutes the same selected pair on both outer passes.
//!
//! Configuration salts are computed from the original tree. Each source contributes
//! its salt and original length, even when empty. Plain repetitions and selections
//! pass that value through unchanged. Each shuffle, shuffled repeat or shuffled cycle
//! advances the salt once, independent of its count or output length, even when its
//! runtime node is folded away. These three operations use the same salt step, so
//! their equivalent one-pass forms remain interchangeable inside larger sequences.
//! A concatenation combines its children's salts in order, independent of grouping.
//! Adding empty concatenations or changing `concat([a, b, c])` to
//! `concat([a, concat([b, c])])` preserves shuffling. An actual source of length zero
//! still contributes its identity. Compiler pruning and flattening never change
//! these salts. Modifying an excluded source can change a shuffle above the selection,
//! even though that source contributes no records.
//!
//! # Validation and limits
//!
//! [`Order::new`] returns an [`Error`] with a kind and a path to the invalid node.
//! All sequence builders defer validation until an order is built, including parts
//! that would contribute no elements, such as the child of `repeat(0)`.
//! Configurations use ordinary recursive traversal and destruction, so arbitrarily
//! deep hand-built trees are unsupported.
//!
//! Skips and takes must stay within the child sequence. `step_by` requires a nonzero
//! step, and an empty sequence cannot be cycled to a positive length. Schedules must
//! have valid parameters and satisfy their individual numerical limits; see
//! [`Schedule`] and [`ErrorKind`] for the full rules.
//!
//! Lengths and positions use `usize` throughout. Every sequence node must fit in
//! `usize`, even if a parent truncates or discards it. Seeds, salts and shuffle
//! arithmetic use fixed-width `u64` values for reproducibility across platforms.
//! A mix is limited to [`MAX_MIX_LEN`] elements. [`Seq`] documents stack use; its
//! builders work with any source type.
//!
//! Compilation simplifies nodes without changing their order. It flattens nested
//! concatenations, removes empty parts, merges nested selections and strides, and folds
//! skips and takes into sources where possible. A mix with one non-empty part becomes
//! that part; a shuffle of at most one element and a single plain repetition need no wrapper.
//! A plain cycle that fits within one epoch becomes a take. Source handles remain available
//! through [`Order::sources`], including those whose nodes were removed.
//!
//! [`Order::get`] returns `None` for an invalid position. [`Order::cursor`] and
//! [`Cursor::reset`] report [`BoundsError`] instead of panicking on invalid bounds.
//! Both take ranges in absolute order positions. Resetting replaces the remaining
//! range and moves to its start; unbounded endpoints refer to the whole order.
//! Failed cursor operations leave their state unchanged.
//! [`Cursor::offset`] reads the next absolute position;
//! `position(predicate)` remains the standard consuming iterator search.
//! Every [`Item`] includes its source's ordinal in [`Order::sources`], so equal and
//! zero-sized sources can be distinguished. Ordinals are local to an order.
//!
//! # Cost
//!
//! Storage depends on the configuration and cursor state, not on the number of output
//! elements. Compiler visits return lengths and configuration salts to their parents.
//! Compilation can revisit subtrees when flattening concatenations or folding selections.
//! Each mix builds independent profiles in `O(k)` time for `k` parts.
//!
//! For random access, [`Order::get`] follows the path from the root to a source:
//!
//! | Node | Work at that node |
//! | --- | --- |
//! | Concat | `O(log k)` search over `k` part offsets |
//! | Shuffle | Constant average permutation cost; an individual position can take longer |
//! | Shuffled repeat or cycle | Pass arithmetic and a permutation, as for shuffle |
//! | Mix | A seek over its parts, with the cost described below |
//! | Repeat, skip, take, step by | Position arithmetic |
//!
//! For a mix with `k` non-empty parts, seeking counts each part's elements below a
//! bounded number of trial virtual times, then replays at most `2k` tournament steps.
//! It tries `position / N` first, which is a good estimate for uniform mixes;
//! scheduled mixes generally need more probes because virtual time differs from
//! output progress. Building and replaying the tournament costs `O(k log(k + 1))`.
//!
//! Sequential iteration keeps cursor state. A mix uses `⌈log2 k⌉` tournament comparisons
//! per element plus one key computation, with no comparisons once one part remains.
//! A shuffle reads scattered child positions without allocating.
//!
//! Stepping through a sequence skips unselected child positions. Mixes advance their
//! interleave for short skips and seek for longer ones. Sharding a mix across `count`
//! workers can therefore multiply the total interleaving work by up to `count`;
//! see [`Seq::step_by`].
//!
//! Cursor construction positions its state immediately and can allocate, even for
//! an empty range or a cursor used only for `count()`. [`Cursor::reset`] and
//! [`Iterator::nth`] reuse existing buffers within the current child, including
//! when moving to empty ranges. Concat transitions replace
//! child state, and entering a new mix part can allocate. Cloning copies current state
//! without preserving spare buffer capacity; subsequent seeks may allocate new buffers.
//! `last()` uses [`Order::get`] and can allocate independently of the cursor's buffers.
//! For benchmark workloads and commands, see the
//! [README's performance section](https://github.com/xmakro/dataorder/blob/main/README.md#performance).
//!
//! # Feature flags
//!
//! The optional `serde` feature derives `Serialize` and `Deserialize` for [`Seq`],
//! [`MixPart`] and [`Schedule`]. It uses serde's derived representation,
//! with the documented variant and field names. For example:
//!
//! ```json
//! {"Mix":[{"seq":{"Shuffle":{"inner":{"Source":50}}},"schedule":"Uniform"}]}
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
//!   [`Seq::map_sources`] or [`Seq::try_map_sources`] as needed, then compile it with
//!   [`Order::new`].
//!
//! # Stability
//!
//! The same configuration, source lengths and salts, and seed produce the same order
//! on supported platforms. A release that changes an order or the serialized
//! configuration format is a breaking change: a new minor version while the crate is 0.x.
//! Save [`CRATE_VERSION`] in checkpoints for a conservative exact-version check.
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
pub use error::{BoundsError, Error, ErrorKind, ScheduleReason};
pub use interleave::Schedule;
pub use order::{Item, Order};
pub use seq::{MixPart, Seq};
pub use source::{Source, salt};

/// Numerical limit on mix length: 2⁴⁶ elements.
///
/// This limit leaves room for floating-point rounding between a part's consecutive
/// keys and keeps counts exactly representable in `f64`. Scheduled parts must also
/// satisfy `length × peak rate ≤ MAX_MIX_LEN`. Repeating or concatenating valid
/// mixes can produce longer orders. Every sequence length must also fit in `usize`.
///
/// ```
/// use dataorder::{ErrorKind, MAX_MIX_LEN, Order, Seq};
/// assert_eq!(MAX_MIX_LEN, 1 << 46);
/// # #[cfg(target_pointer_width = "64")]
/// # {
/// let long = Seq::mix([Seq::source(1 << 30).repeat(1 << 16), Seq::source(1)]);
/// assert_eq!(Order::new(long).unwrap_err().kind(), &ErrorKind::MixTooLong);
/// # }
/// let repeated = Seq::mix([Seq::source(1 << 30), Seq::source(1)]).repeat(3);
/// assert_eq!(Order::new(repeated).map(|order| order.len()), Ok(3 * (1 << 30) + 3));
/// ```
pub const MAX_MIX_LEN: u64 = interleave::MAX_TOTAL_LEN;

/// Version of the linked `dataorder` crate.
///
/// This is the version of the linked `dataorder` crate, including its patch version,
/// rather than the calling application's `CARGO_PKG_VERSION`. Save it alongside the
/// configuration, source metadata, seeds and worker settings. Require an exact match
/// on restore unless the application has explicitly verified a compatible migration.
/// A difference does not necessarily mean that the ordering changed; see the crate's
/// [stability policy](crate#stability).
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The README's code blocks, compiled as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
