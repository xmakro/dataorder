//! Deterministic ordering for large datasets: shuffle, seek anywhere, then stream.
//!
//! Shuffle and mix billions of records without storing an index array. Memory grows
//! with the sources and the sequence structure, not with the number of records. Each
//! lookup returns an [`Item`] naming a source and a record index within it; loading
//! the record is up to you.
//!
//! - **Shuffle on demand:** each shuffled index takes O(1) time on average and O(1) space.
//! - **Seek into a mix:** counting and binary searches locate each part's position
//!   without replaying the preceding records.
//! - **Walk from there:** a mix keeps a tournament tree and chooses each next part with
//!   O(log k) comparisons for `k` parts, instead of seeking again for every element.
//!
//! See the [cost model](#cost) for the cost of each operation and
//! [`benches/ordering.rs`](https://github.com/xmakro/dataorder/blob/main/benches/ordering.rs)
//! for the benchmarks.
//!
//! Three types make up the API:
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
//! Position 1,200,000,000 belongs to the *order*; the returned index belongs to the
//! source. [`Order::get`] returns one position, [`Order::iter`] returns a [`Cursor`]
//! over the whole order and [`Order::cursor`] one over a range. A cursor yields the same
//! items as `get` at each position in its range, whatever was iterated or seeked before.
//!
//! Implement [`Source`] for your dataset handles, or use slices, arrays or vectors.
//! `Order` owns its sources and yields references to them. A `Seq` can be cloned,
//! compared, [mapped to another source type](Seq::map_sources) and optionally
//! [serialized](#feature-flags). `Seq<T>` accepts any `T`; only building an `Order<T>`
//! requires `T: Source`.
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
//! | [`Mix`](Seq::Mix) | Sum of part lengths | Parts interleaved, each keeping its own order |
//! | [`Repeat`](Seq::Repeat) | `times × n` | Index `p % n` in the unchanged input, or in a separate permutation per pass when `shuffled`; [`shuffle`](Seq::shuffle) is one shuffled pass |
//! | [`Cycle`](Seq::Cycle) | `len` | Like repeat, with the last pass truncated as needed |
//! | [`Skip`](Seq::Skip) | `n − skip` | Child position `skip + p` |
//! | [`Take`](Seq::Take) | `take` | Child position `p` |
//! | [`StepBy`](Seq::StepBy) | `⌈n / step⌉` | Child position `p × step` |
//!
//! A mix uses every element of every part once, so longer parts appear more often. Set
//! each part's count with [`Seq::cycle_to`] or [`Seq::cycle_to_shuffled`] before mixing.
//! A [`Schedule`] spreads a part's elements over a shared virtual clock, which is not
//! output progress; see [`Schedule`] for the model and its limits.
//! `skip(index).step_by(count)` partitions the positions among workers; see
//! [`Seq::step_by`] for worker configuration and its costs.
//!
//! # Shuffles and repetitions
//!
//! A shuffle visits every position of its input exactly once. Its permutation depends
//! on the order's seed and on the input's configuration salt, which combines the
//! [`Source::salt`] values and original lengths of the sources below it and the
//! shuffled layers between them, as written in the original configuration. Give
//! datasets stable salts so that datasets of equal length shuffle differently.
//! [`Order::set_seed`] reseeds every shuffle without rebuilding the order. Shuffles use
//! a six-round Feistel permutation with cycle walking; they are meant for reproducible
//! ordering, not cryptography.
//!
//! The input of a shuffle or shuffled repetition must contain no mix, at any depth,
//! even an empty or single-part one. Shuffle each input before mixing; [`Order::new`]
//! reports [`ErrorKind::ShuffleContainsMix`] at the enclosing shuffle.
//!
//! `x.repeat(3)` and `x.cycle_to(len)` replay the input's order on every pass,
//! including any shuffle inside it: `x.shuffle().repeat(3)` repeats one permutation
//! three times, and `x.repeat(3).repeat(2)` orders like `x.repeat(6)`.
//! `x.repeat_shuffled(3)` and `x.cycle_to_shuffled(len)` permute the input's positions
//! separately on each pass, including the first, and `x.shuffle()` is
//! `x.repeat_shuffled(1)`. Nested shuffles keep their own permutations: an enclosing
//! repetition never reseeds them, and each shuffled layer uses a key distinct from the
//! layers below it. A shortened final pass takes a prefix of its full permutation.
//!
//! An existing shuffle survives these configuration changes:
//!
//! - Any change to a mix: adding, removing, reordering or nesting its parts, or
//!   changing their schedules. A shuffle lives inside a part and depends only on that
//!   part's configuration and the seed.
//! - Wrapping a sequence in a plain repeat, which keeps the whole first pass, or
//!   extending an outermost repeat or cycle, plain or shuffled, which keeps the
//!   existing prefix.
//! - Regrouping concatenations, such as changing `concat([a, b, c])` to
//!   `concat([a, concat([b, c])])`, or adding empty concatenations.
//!
//! A selection fixes the positions that later operations draw from:
//! `x.shuffle().take(2).repeat_shuffled(2)` permutes the same selected pair on both
//! outer passes. Because the salt follows the original configuration, a source
//! contributes its salt and original length even when it is empty or excluded by a
//! selection, so changing such a source changes the shuffles above it although it
//! contributes no records. The exact derivation is documented with the permutation code.
//!
//! # Validation and limits
//!
//! Builders only record the configuration. [`Order::new`] validates all of it,
//! including parts that contribute no elements, such as the child of `repeat(0)`, and
//! returns an [`Error`] with a kind and the path to the invalid node. Skips and takes
//! must stay within their input, `step_by` needs a nonzero step, a positive `cycle_to`
//! needs a non-empty input, and schedules must satisfy the rules on [`Schedule`]; see
//! [`ErrorKind`] for the full list.
//!
//! Lengths and positions are `usize`. Every node's length must fit, even when a parent
//! truncates it, and a mix holds at most [`MAX_MIX_LEN`] elements. Seeds, salts and
//! shuffle arithmetic use `u64`, so orders agree across platforms. Configurations are
//! traversed and dropped recursively, so arbitrarily deep hand-built trees are
//! unsupported.
//!
//! Compilation simplifies the tree without changing its order, dropping empty parts
//! and folding selections into sources where it can. Every source handle stays
//! available through [`Order::sources`], in configuration order, whether or not its
//! node survived, and every [`Item`] carries its source's ordinal there, so equal or
//! zero-sized sources can be told apart. Ordinals are local to an order.
//!
//! [`Order::get`] returns `None` for an invalid position. [`Order::cursor`] and
//! [`Cursor::reset`] take ranges of absolute order positions, with unbounded ends
//! meaning the whole order, and report a [`BoundsError`] instead of panicking; a failed
//! operation leaves the cursor unchanged. [`Cursor::offset`] reads the next absolute
//! position without consuming anything.
//!
//! # Cost
//!
//! Memory depends on the configuration and on cursor state, not on the number of
//! elements. Building an order costs time that depends on the configuration alone;
//! each mix builds its part profiles in `O(k)` time for `k` parts.
//!
//! [`Order::get`] follows the path from the root to a source:
//!
//! | Node | Work at that node |
//! | --- | --- |
//! | Concat | `O(log k)` search over `k` part offsets |
//! | Shuffle | Constant average permutation cost; an individual position can take longer |
//! | Shuffled repeat or cycle | Pass arithmetic and a permutation, as for shuffle |
//! | Mix | A seek over its parts, described below |
//! | Repeat, skip, take, step by | Position arithmetic |
//!
//! A mix with `k` non-empty parts seeks by counting each part's elements below a
//! bounded number of trial virtual times, then replaying at most `2k` tournament
//! steps. The first trial is `position / N`, which is close for uniform mixes;
//! scheduled mixes usually need more trials, because virtual time differs from output
//! progress. Building and replaying the tournament costs `O(k log(k + 1))`.
//!
//! Sequential iteration keeps cursor state. A mix spends `⌈log2 k⌉` tournament
//! comparisons plus one key computation per element, and none once a single part
//! remains. A shuffle reads scattered child positions without allocating.
//!
//! Stepping through a sequence skips the unselected child positions. A mix walks its
//! interleave over short skips and seeks over long ones, so sharding a mix across
//! `count` workers can multiply the total interleaving work by up to `count`.
//! Partitioning each part before mixing avoids that, at the price of a different
//! order; see [`Seq::step_by`].
//!
//! Constructing a cursor positions it immediately and can allocate, even for an empty
//! range or a cursor used only for `count()`: an entered mix reserves space for its
//! parts and builds each part's cursor when it first draws from it. [`Cursor::reset`]
//! and [`Iterator::nth`] skip forward or reposition backward, reusing the buffers of
//! the current child, including when moving to an empty range; entering another concat
//! child replaces the child state, and entering a new mix part can allocate. A clone
//! copies the initialized state without spare capacity, so its later seeks may
//! allocate. `last()` uses [`Order::get`] and allocates independently of the cursor's
//! buffers. The benchmarks in
//! [`benches/ordering.rs`](https://github.com/xmakro/dataorder/blob/main/benches/ordering.rs)
//! measure construction, random lookup, seeks and a sequential walk.
//!
//! # Feature flags
//!
//! The optional `serde` feature derives `Serialize` and `Deserialize` for [`Seq`],
//! [`MixPart`] and [`Schedule`] in serde's derived representation, with the documented
//! variant and field names; unknown fields are rejected. The source type must support
//! serialization too. A shuffled source in a mix serializes as:
//!
//! ```json
//! {"Mix":[{"seq":{"Repeat":{"times":1,"shuffled":true,"inner":{"Source":50}}},"schedule":"Uniform"}]}
//! ```
//!
//! Changes to this format follow the [stability policy](#stability). Deserialization
//! does not validate the configuration: resolve its sources with [`Seq::map_sources`]
//! or [`Seq::try_map_sources`] as needed, then compile it with [`Order::new`].
//!
//! With JSON, enable `float_roundtrip` on your own `serde_json` dependency. Without it,
//! parsing can move a schedule breakpoint by one representable `f64` step, which
//! changes equality and possibly the order:
//!
//! ```toml
//! [dependencies]
//! dataorder = { version = "0.4", features = ["serde"] }
//! serde_json = { version = "1", features = ["float_roundtrip"] }
//! ```
//!
//! NaN and infinite parameters are invalid in any case, and `serde_json` writes them as
//! `null`, which does not read back as an `f64`. Lengths are `usize`, so a
//! configuration written on a 64-bit machine may not fit on a 32-bit one.
//!
//! # Stability
//!
//! The same configuration, source lengths and salts, and seed produce the same order
//! on every supported platform. A release that changes an order or the serialized
//! configuration format is a breaking change: a new minor version while the crate is
//! 0.x.
//!
//! Calculations use IEEE 754 binary64 with separate multiply and add expressions, so
//! targets with hardware or software binary64 agree; x87-only `i586` targets using
//! extended precision are excluded. Golden tests pin order fingerprints through the
//! public API, and CI checks 64-bit and 32-bit x86 and 64-bit ARM.

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
pub use error::{BoundsError, Error, ErrorKind};
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

/// The README's code blocks, compiled as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
