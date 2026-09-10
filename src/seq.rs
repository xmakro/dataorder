//! Sequence configuration and builders. Each node is a source, a transform or a
//! combination of other sequences; [`Order::new`](crate::Order::new) validates and
//! compiles the tree.

use crate::Sampling;
use std::convert::Infallible;

/// A description of how to order sources of type `T`.
///
/// Build a sequence with [`source`](Seq::source) and the methods below, then pass it
/// to [`Order::new`](crate::Order::new). `T` must implement [`Source`](crate::Source)
/// when you build the order; before that, it can be any type, such as a path you
/// later [`map`](Seq::map) to a dataset.
/// You can also construct enum variants directly. Builders store the configuration
/// without validating it, loading records or generating indices.
///
/// `Seq` supports cloning, comparison, hashing and, with the `serde` feature,
/// serialization when `T` does. Schedule parameters are compared by their bits,
/// treating `-0.0` as `0.0`. NaN parameters are rejected during validation.
///
/// # Errors
///
/// [`Order::new`](crate::Order::new) reports invalid configurations as an
/// [`Error`](crate::Error) with the path to the invalid node. This includes
/// out-of-range skips and takes, zero steps, overflow, invalid schedules, and
/// excessive depth. All sequence builders accept any `T` and defer these checks
/// until the order is built.
///
/// # Depth
///
/// [`Order::new`](crate::Order::new) stops at [`MAX_DEPTH`](crate::MAX_DEPTH) (16 levels).
/// Keep configurations within this supported limit.
/// Tree operations and ordinary Rust destruction recurse with tree depth;
/// arbitrarily deep hand-built trees are unsupported. Stack use also depends on
/// the size of `T`; prefer small dataset handles over large inline sources.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(deny_unknown_fields))]
#[non_exhaustive]
pub enum Seq<T> {
    /// The elements `0..len()` of a source, in order.
    Source(T),
    /// The parts one after another.
    Concat(Vec<Self>),
    /// Every element of every part, interleaved according to each part's [`Sampling`].
    /// Each part keeps its own order. Empty parts do not affect the other parts' order.
    ///
    /// Schedules span this mix's length. Repeating the mix restarts them each epoch;
    /// repeat its parts instead to schedule over the whole run. The total length
    /// cannot exceed [`MAX_MIX_LEN`](crate::MAX_MIX_LEN).
    Mix(Vec<MixPart<T>>),
    /// Every position of `inner` once, in a seeded pseudorandom order.
    ///
    /// The permutation depends on `seed`, the order's seed, enclosing repetitions,
    /// and the salts and lengths of retained sources. See the crate's
    /// [shuffle rules](crate#shuffles-and-repetitions) for details.
    Shuffle {
        /// Selects the permutation.
        seed: u64,
        /// The sequence to permute.
        inner: Box<Self>,
    },
    /// `inner` repeated `times` times, reseeding existing shuffles after the first epoch.
    /// Repetition does not add shuffling. Any mix schedules inside restart each epoch.
    /// `x.repeat(1)` is `x`; `x.repeat(0)` is empty but still validates `inner`.
    ///
    /// The first pass preserves all of `inner`, including nested epochs. Repetition levels
    /// are numbered from the inside out, so an outer repeat does not change inner levels.
    /// Later passes reseed the existing shuffles using this repeat's epoch and level.
    Repeat {
        /// Number of repetitions.
        times: usize,
        /// The sequence to repeat.
        inner: Box<Self>,
    },
    /// Exactly `len` positions of `inner`, repeating or truncating it as needed.
    ///
    /// Each additional epoch reseeds existing shuffles, as [`Repeat`](Seq::Repeat)
    /// does. The entire first pass keeps its order, including nested epochs; increasing
    /// `len` preserves the existing prefix.
    /// When `len` fits within `inner`, this is equivalent to `inner.take(len)`.
    /// A positive `len` requires a non-empty child. `cycle(usize::MAX)` creates the
    /// longest supported order; it is still finite.
    Cycle {
        /// Length of the sequence.
        len: usize,
        /// The sequence to repeat and cut.
        inner: Box<Self>,
    },
    /// `inner` without its first `n` positions. Skipping more than there are is an error
    /// when the order is built, unlike `Iterator::skip`: configurations are validated, and a
    /// silently empty sequence hides a mistake.
    Skip {
        /// Positions dropped from the front.
        n: usize,
        /// The sequence to skip into.
        inner: Box<Self>,
    },
    /// The first `n` positions of `inner`. Taking more than there are is an error when the
    /// order is built, unlike `Iterator::take`.
    Take {
        /// Positions kept.
        n: usize,
        /// The sequence to take from.
        inner: Box<Self>,
    },
    /// Positions `0, step, 2·step, …` of `inner`.
    /// `step` must be positive when the order is built. Use [`skip`](Seq::skip)
    /// before this node to start at a different position.
    StepBy {
        /// Distance between kept positions.
        step: usize,
        /// The sequence to step through.
        inner: Box<Self>,
    },
}

/// A part of a [`Mix`](Seq::Mix): a sequence and its schedule. `(seq, sampling)` and a bare
/// `seq` (uniform) convert into it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(deny_unknown_fields))]
pub struct MixPart<T> {
    /// The sequence.
    pub seq: Seq<T>,
    /// How its elements are spread over the mix.
    pub sampling: Sampling,
}

impl<T> From<(Seq<T>, Sampling)> for MixPart<T> {
    fn from((seq, sampling): (Seq<T>, Sampling)) -> Self {
        Self { seq, sampling }
    }
}

impl<T> From<Seq<T>> for MixPart<T> {
    fn from(seq: Seq<T>) -> Self {
        Self { seq, sampling: Sampling::Uniform }
    }
}

impl<T> Seq<T> {
    /// The elements of `source`, in order.
    #[must_use]
    pub fn source(source: T) -> Self {
        Self::Source(source)
    }

    /// The parts one after another.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::concat([Seq::source(2), Seq::source(3)]))?;
    /// let elements: Vec<(usize, usize)> = order.iter(..)?.map(|item| (*item.source, item.record_index)).collect();
    /// assert_eq!(elements, [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn concat(parts: impl IntoIterator<Item = Self>) -> Self {
        Self::Concat(parts.into_iter().collect())
    }

    /// Interleaves parts, preserving each part's order.
    /// Accepts bare sequences (all [`Sampling::Uniform`]), `(seq, sampling)` pairs,
    /// or [`MixPart`] values. Schedules are independent on a shared virtual clock;
    /// see [`Sampling`] for how virtual time maps to output progress.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// // Each part keeps its order and is spread evenly: the longer one appears twice as often.
    /// let order = Order::new(Seq::mix([Seq::source(4), Seq::source(2)]))?;
    /// let sources: Vec<usize> = order.iter(..)?.map(|item| *item.source).collect();
    /// assert_eq!(sources, [4, 4, 2, 4, 4, 2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// Attach a schedule to each part to control when it contributes:
    ///
    /// ```
    /// use dataorder::{Order, Sampling, Seq};
    /// let seq = Seq::mix([
    ///     (Seq::source(700), Sampling::Uniform),
    ///     (Seq::source(300), Sampling::delayed(0.5)),
    /// ]);
    /// let order = Order::new(seq)?;
    /// // At virtual time 0.5, half of the 700 uniform elements have appeared.
    /// // The delayed source begins around output position 350, not 500.
    /// let first = order.iter(..)?.position(|item| *item.source == 300).unwrap();
    /// assert!((349..=351).contains(&first));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// Empty inputs need an explicit element type, for example
    /// `Seq::mix(std::iter::empty::<Seq<usize>>())`.
    #[must_use]
    pub fn mix(parts: impl IntoIterator<Item = impl Into<MixPart<T>>>) -> Self {
        Self::Mix(parts.into_iter().map(Into::into).collect())
    }

    /// This sequence in the pseudorandom order selected by `seed`.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::source(100).shuffle(1))?;
    /// let mut indices: Vec<usize> = order.iter(..)?.map(|item| item.record_index).collect();
    /// assert_ne!(indices[..5], [0, 1, 2, 3, 4]);
    /// indices.sort_unstable();
    /// assert_eq!(indices, (0..100).collect::<Vec<_>>());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn shuffle(self, seed: u64) -> Self {
        Self::Shuffle { seed, inner: Box::new(self) }
    }

    /// Repeats this sequence `times` times, reseeding existing shuffles after the first epoch.
    /// The first pass preserves the sequence, including all nested epochs;
    /// see [`Repeat`](Seq::Repeat).
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::source(1000).shuffle(1).repeat(2))?;
    /// let first: Vec<_> = order.iter(..1000)?.map(|item| item.record_index).collect();
    /// let mut second: Vec<_> = order.iter(1000..)?.map(|item| item.record_index).collect();
    /// assert_ne!(first, second);
    /// second.sort_unstable();
    /// assert_eq!(second, (0..1000).collect::<Vec<_>>());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn repeat(self, times: usize) -> Self {
        Self::Repeat { times, inner: Box::new(self) }
    }

    /// Repeats or truncates this sequence to exactly `len` positions.
    /// Existing shuffles are reseeded for each additional epoch; see [`Cycle`](Seq::Cycle).
    /// Even `cycle(usize::MAX)` is finite.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::source(1000).shuffle(1).cycle(2500))?;
    /// assert_eq!(order.len(), 2500);
    /// let epochs = Order::new(Seq::source(1000).shuffle(1).repeat(3))?;
    /// assert!(order.iter(..)?.eq(epochs.iter(..2500)?));
    /// let longest = Order::new(Seq::source(1000).shuffle(1).cycle(usize::MAX))?;
    /// assert_eq!(longest.len(), usize::MAX);
    /// assert!(longest.get(usize::MAX - 1).unwrap().record_index < 1000);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn cycle(self, len: usize) -> Self {
        Self::Cycle { len, inner: Box::new(self) }
    }

    /// The first `n` positions (an error when the order is built if there are fewer).
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let all = Order::new(Seq::source(10).shuffle(1))?;
    /// let first = Order::new(Seq::source(10).shuffle(1).take(3))?;
    /// assert!(first.iter(..)?.eq(all.iter(..3)?));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn take(self, n: usize) -> Self {
        Self::Take { n, inner: Box::new(self) }
    }

    /// Everything after the first `n` positions (an error when the order is built if there are fewer).
    /// Follow with `take(len)` to select a range, or `step_by(step)` to select
    /// every `step`-th position starting at `n`.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let all = Order::new(Seq::source(10).shuffle(1))?;
    /// let rest = Order::new(Seq::source(10).shuffle(1).skip(7))?;
    /// assert!(rest.iter(..)?.eq(all.iter(7..)?));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn skip(self, n: usize) -> Self {
        Self::Skip { n, inner: Box::new(self) }
    }

    /// Every `step`-th position, starting at the first position.
    /// Precede this with [`skip`](Seq::skip) to start at a different position.
    /// [`Order::new`](crate::Order::new) rejects a zero step, even for an empty sequence.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::source(10).step_by(4))?;
    /// assert_eq!(order.iter(..)?.map(|item| item.record_index).collect::<Vec<_>>(), [0, 4, 8]);
    /// let order = Order::new(Seq::source(10).skip(1).step_by(4))?;
    /// assert_eq!(order.iter(..)?.map(|item| item.record_index).collect::<Vec<_>>(), [1, 5, 9]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Workers
    ///
    /// Use `skip(index).step_by(count)` to assign positions `index, index + count, …`
    /// to a worker. Check `index < count` in the calling code. Since `skip` rejects
    /// positions past the end, use `skip(index.min(len))` when the sequence length
    /// is known and workers beyond it should receive an empty sequence.
    ///
    /// Applying this to a completed mix partitions its global positions exactly
    /// once across workers. Worker lengths can differ by one, and their dataset
    /// mixtures need not be balanced: two equal interleaved parts split across
    /// two workers send one part exclusively to each worker. Shuffling the mix
    /// first breaks that pattern but scatters its scheduled phases and adds a
    /// mix seek per element.
    ///
    /// Each worker advances the mix past unselected positions, or seeks for long
    /// skips. Across `count` workers this can cost up to `count` times the global
    /// mix's interleaving work. Partitioning each part before mixing can reduce
    /// this work, but changes the global order and how virtual time maps to output
    /// positions because each part's count is rounded separately.
    #[must_use]
    pub fn step_by(self, step: usize) -> Self {
        Self::StepBy { step, inner: Box::new(self) }
    }

    /// Transforms each source with `f`, preserving the sequence structure.
    /// Sources are visited in order of appearance. Use this to turn configuration
    /// values into dataset handles. The resulting order is unchanged if each source
    /// keeps its length and salt.
    ///
    /// ```
    /// use dataorder::Seq;
    /// use std::collections::HashMap;
    ///
    /// // Resolve dataset names to their known record counts.
    /// let counts = HashMap::from([("web", 1000usize), ("code", 200)]);
    /// let names = Seq::mix([Seq::source("web"), Seq::source("code").shuffle(1)]);
    /// let lengths = names.map(|name| counts[name]);
    /// assert_eq!(lengths, Seq::mix([Seq::source(1000), Seq::source(200).shuffle(1)]));
    /// ```
    #[must_use]
    pub fn map<U, F: FnMut(T) -> U>(self, mut f: F) -> Seq<U> {
        match self.try_map(|t| Ok::<U, Infallible>(f(t))) {
            Ok(seq) => seq,
            Err(never) => match never {},
        }
    }

    /// Transforms sources like [`map`](Seq::map), stopping at the first error.
    /// Sources after the error are not visited. Unvisited inputs and mapped outputs
    /// are dropped normally on error.
    ///
    /// ```
    /// use dataorder::Seq;
    ///
    /// // Parse record counts supplied as strings.
    /// let config = Seq::mix([Seq::source("1000"), Seq::source("200").shuffle(1)]);
    /// let lengths = config.try_map(str::parse::<usize>)?;
    /// assert_eq!(lengths, Seq::mix([Seq::source(1000), Seq::source(200).shuffle(1)]));
    /// assert!(Seq::source("unknown").try_map(str::parse::<usize>).is_err());
    /// # Ok::<(), std::num::ParseIntError>(())
    /// ```
    ///
    /// # Errors
    /// The first error `f` returns.
    pub fn try_map<U, E, F: FnMut(T) -> Result<U, E>>(self, mut f: F) -> Result<Seq<U>, E> {
        map_sources(self, &mut f)
    }
}

/// Map the supported, bounded-depth configuration using ordinary recursive ownership.
fn map_sources<T, U, E>(seq: Seq<T>, f: &mut impl FnMut(T) -> Result<U, E>) -> Result<Seq<U>, E> {
    Ok(match seq {
        Seq::Source(source) => Seq::source(f(source)?),
        Seq::Concat(parts) => Seq::concat(parts.into_iter().map(|p| map_sources(p, f)).collect::<Result<Vec<_>, _>>()?),
        Seq::Mix(parts) => Seq::Mix(
            parts.into_iter().map(|p| Ok(MixPart { seq: map_sources(p.seq, f)?, sampling: p.sampling })).collect::<Result<_, E>>()?,
        ),
        Seq::Shuffle { seed, inner } => map_sources(*inner, f)?.shuffle(seed),
        Seq::Repeat { times, inner } => map_sources(*inner, f)?.repeat(times),
        Seq::Cycle { len, inner } => map_sources(*inner, f)?.cycle(len),
        Seq::Skip { n, inner } => map_sources(*inner, f)?.skip(n),
        Seq::Take { n, inner } => map_sources(*inner, f)?.take(n),
        Seq::StepBy { step, inner } => map_sources(*inner, f)?.step_by(step),
    })
}
