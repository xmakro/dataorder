//! Sequence configuration and builders. Each node is a source, a transform or a
//! combination of other sequences; [`Order::new`](crate::Order::new) validates and
//! compiles the tree.

use crate::Schedule;
use std::convert::Infallible;

/// A description of how to order sources of type `T`.
///
/// Build a sequence with [`source`](Seq::source) and the methods below, then pass it
/// to [`Order::new`](crate::Order::new). `T` must implement [`Source`](crate::Source)
/// when you build the order; before that, it can be any type, such as a path you
/// later transform with [`map_sources`](Seq::map_sources) into a dataset.
/// You can also construct enum variants directly. Builders store the configuration
/// without validating it, loading records or generating indices.
///
/// `Seq` supports cloning, comparison and, with the `serde` feature,
/// serialization when `T` does. Schedule parameters use ordinary `f64` equality.
/// NaN parameters are rejected during validation.
///
/// # Errors
///
/// [`Order::new`](crate::Order::new) reports invalid configurations as an
/// [`Error`](crate::Error) with the path to the invalid node. This includes
/// out-of-range skips and takes, zero steps, overflow, invalid schedules,
/// excessive depth and shuffles containing mixes. All sequence builders accept
/// any `T` and defer these checks until the order is built.
///
/// # Depth
///
/// [`Order::new`](crate::Order::new) stops at [`MAX_DEPTH`](crate::MAX_DEPTH) (16 levels).
/// Keep configurations within this supported limit.
/// Tree operations and ordinary Rust destruction recurse with tree depth;
/// arbitrarily deep hand-built trees are unsupported. Stack use also depends on
/// the size of `T`; prefer small dataset handles over large inline sources.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(deny_unknown_fields))]
#[non_exhaustive]
pub enum Seq<T> {
    /// The elements `0..len()` of a source, in order.
    Source(T),
    /// The parts one after another.
    Concat(Vec<Self>),
    /// Every element of every part, interleaved according to each part's [`Schedule`].
    /// Each part keeps its own order. Empty parts do not affect the other parts' order.
    ///
    /// Schedules span this mix's length. Repeating the mix restarts them each epoch;
    /// repeat its parts instead to schedule over the whole run. The total length
    /// cannot exceed [`MAX_MIX_LEN`](crate::MAX_MIX_LEN).
    Mix(Vec<MixPart<T>>),
    /// Every position of `inner` once, in a seeded pseudorandom order.
    ///
    /// `inner` must contain no `Mix` nodes, including empty or single-part mixes
    /// and mixes nested beneath other operations. Shuffle inputs before mixing.
    /// [`Order::new`](crate::Order::new) rejects this combination with
    /// [`ErrorKind::ShuffleContainsMix`](crate::ErrorKind::ShuffleContainsMix).
    ///
    /// The permutation depends on the order's seed
    /// and the input configuration's ordered source salts, original lengths and shuffled layers.
    /// Concatenation grouping and empty concatenations do not affect it.
    /// Empty or discarded sources still contribute. See the crate's
    /// [shuffle rules](crate#shuffles-and-repetitions) for details.
    Shuffle {
        /// The sequence to permute.
        inner: Box<Self>,
    },
    /// `inner` repeated `times` times, preserving its record order on every pass.
    /// Any mix schedules inside restart each epoch.
    /// `x.repeat(1)` is `x`; `x.repeat(0)` is empty but still validates `inner`.
    ///
    /// Nested plain repeats compose: `x.repeat(3).repeat(2)` has the
    /// same order as `x.repeat(6)`. Adding an outer repeat preserves the first pass.
    Repeat {
        /// Number of repetitions.
        times: usize,
        /// The sequence to repeat.
        inner: Box<Self>,
    },
    /// Exactly `len` positions of `inner`, repeating or truncating it as needed.
    ///
    /// Each pass preserves the input's record order, as [`Repeat`](Seq::Repeat)
    /// does. Adding an outer cycle preserves the first pass, including nested epochs.
    /// Increasing its length preserves the existing prefix when it is outermost.
    /// When `len` fits within `inner`, this is equivalent to `inner.take(len)`.
    /// A positive `len` requires a non-empty child. `cycle_to(usize::MAX)` creates the
    /// longest supported order; it is still finite.
    Cycle {
        /// Length of the sequence.
        len: usize,
        /// The sequence to repeat and cut.
        inner: Box<Self>,
    },
    /// `inner` repeated `times` times, with a separate permutation of its positions
    /// on every pass, including the first. Nested shuffles keep their own permutations.
    ///
    /// Uses the order's seed, the local pass number and the input's configuration salt.
    /// Enclosing repetitions do not affect the permutation. Like [`Shuffle`](Seq::Shuffle),
    /// the input must contain no mixes, even when `times` is zero.
    ShuffledRepeat {
        /// Number of shuffled repetitions.
        times: usize,
        /// The sequence to permute on each pass.
        inner: Box<Self>,
    },
    /// Exactly `len` positions from separately shuffled passes over `inner`.
    ///
    /// Like [`ShuffledRepeat`](Seq::ShuffledRepeat), with the last pass truncated
    /// as needed. Shuffles before truncating, including when `len` fits in one pass.
    /// A positive length requires a non-empty input; the input must contain no mixes.
    ShuffledCycle {
        /// Length of the sequence.
        len: usize,
        /// The sequence to permute on each pass and cut.
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
    /// let elements: Vec<(usize, usize)> = order.iter().map(|item| (*item.source, item.record_index)).collect();
    /// assert_eq!(elements, [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn concat(parts: impl IntoIterator<Item = Self>) -> Self {
        Self::Concat(parts.into_iter().collect())
    }

    /// Interleaves parts, preserving each part's order.
    /// Accepts bare sequences (all [`Schedule::Uniform`]), `(seq, schedule)` pairs,
    /// or [`MixPart`] values. Schedules are independent on a shared virtual clock;
    /// see [`Schedule`] for how virtual time maps to output progress.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// // Each part keeps its order and is spread evenly: the longer one appears twice as often.
    /// let order = Order::new(Seq::mix([Seq::source(4), Seq::source(2)]))?;
    /// let sources: Vec<usize> = order.iter().map(|item| *item.source).collect();
    /// assert_eq!(sources, [4, 4, 2, 4, 4, 2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// Attach a schedule to each part to control when it contributes:
    ///
    /// ```
    /// use dataorder::{Order, Schedule, Seq};
    /// let seq = Seq::mix([
    ///     (Seq::source(700), Schedule::Uniform),
    ///     (Seq::source(300), Schedule::delayed(0.5)),
    /// ]);
    /// let order = Order::new(seq)?;
    /// // At virtual time 0.5, half of the 700 uniform elements have appeared.
    /// // The delayed source begins around output position 350, not 500.
    /// let first = order.iter().position(|item| *item.source == 300).unwrap();
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

    /// This sequence in a pseudorandom order selected by the order's seed and input salt.
    /// Produces the same order as `repeat_shuffled(1)`. Select the run with
    /// [`Order::with_seed`](crate::Order::with_seed) or [`Order::set_seed`](crate::Order::set_seed).
    /// Nested shuffled layers use distinct derived keys without reseeding their inputs.
    ///
    /// The sequence must contain no mixes, even beneath other operations or in
    /// subtrees that would fold away. Shuffle each input before mixing instead.
    /// This restriction is checked by [`Order::new`](crate::Order::new).
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::source(100).shuffle())?;
    /// let mut indices: Vec<usize> = order.iter().map(|item| item.record_index).collect();
    /// assert_ne!(indices[..5], [0, 1, 2, 3, 4]);
    /// indices.sort_unstable();
    /// assert_eq!(indices, (0..100).collect::<Vec<_>>());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn shuffle(self) -> Self {
        Self::Shuffle { inner: Box::new(self) }
    }

    /// Repeats this sequence `times` times, preserving its record order on every pass.
    /// The first pass preserves the sequence, including all nested epochs;
    /// see [`Repeat`](Seq::Repeat).
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::source(1000).shuffle().repeat(2))?;
    /// let first: Vec<_> = order.cursor(..1000)?.map(|item| item.record_index).collect();
    /// let second: Vec<_> = order.cursor(1000..)?.map(|item| item.record_index).collect();
    /// assert_eq!(first, second);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn repeat(self, times: usize) -> Self {
        Self::Repeat { times, inner: Box::new(self) }
    }

    /// Repeats or truncates this sequence to exactly `len` positions.
    /// Every pass preserves the input's record order; see [`Cycle`](Seq::Cycle).
    /// Even `cycle_to(usize::MAX)` is finite.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::source(1000).shuffle().cycle_to(2500))?;
    /// assert_eq!(order.len(), 2500);
    /// let epochs = Order::new(Seq::source(1000).shuffle().repeat(3))?;
    /// assert!(order.iter().eq(epochs.cursor(..2500)?));
    /// let longest = Order::new(Seq::source(1000).shuffle().cycle_to(usize::MAX))?;
    /// assert_eq!(longest.len(), usize::MAX);
    /// assert!(longest.get(usize::MAX - 1).unwrap().record_index < 1000);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn cycle_to(self, len: usize) -> Self {
        Self::Cycle { len, inner: Box::new(self) }
    }

    /// Repeats this sequence with a separate shuffle of its positions on every pass.
    /// Includes the first pass and uses the order's seed. Nested shuffles stay fixed;
    /// the input must contain no mixes. See [`ShuffledRepeat`](Seq::ShuffledRepeat).
    /// Only the unchanged order seed passes to the input. For example,
    /// `x.repeat_shuffled(1).take(2).repeat_shuffled(2)` permutes the same two
    /// selected positions on both outer passes.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::with_seed(Seq::source(1000).repeat_shuffled(3), 42)?;
    /// let first: Vec<_> = order.cursor(..1000)?.map(|item| item.record_index).collect();
    /// let mut second: Vec<_> = order.cursor(1000..2000)?.map(|item| item.record_index).collect();
    /// assert_ne!(first, second);
    /// second.sort_unstable();
    /// assert_eq!(second, (0..1000).collect::<Vec<_>>());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn repeat_shuffled(self, times: usize) -> Self {
        Self::ShuffledRepeat { times, inner: Box::new(self) }
    }

    /// Produces exactly `len` positions from separately shuffled passes over this sequence.
    /// Uses the order's seed and shuffles before truncating the last pass.
    /// See [`ShuffledCycle`](Seq::ShuffledCycle) for input restrictions.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::source(1000).cycle_to_shuffled(2500))?;
    /// let epochs = Order::new(Seq::source(1000).repeat_shuffled(3))?;
    /// assert_eq!(order.len(), 2500);
    /// assert!(order.iter().eq(epochs.cursor(..2500)?));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn cycle_to_shuffled(self, len: usize) -> Self {
        Self::ShuffledCycle { len, inner: Box::new(self) }
    }

    /// The first `n` positions (an error when the order is built if there are fewer).
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let all = Order::new(Seq::source(10).shuffle())?;
    /// let first = Order::new(Seq::source(10).shuffle().take(3))?;
    /// assert!(first.iter().eq(all.cursor(..3)?));
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
    /// let all = Order::new(Seq::source(10).shuffle())?;
    /// let rest = Order::new(Seq::source(10).shuffle().skip(7))?;
    /// assert!(rest.iter().eq(all.cursor(7..)?));
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
    /// assert_eq!(order.iter().map(|item| item.record_index).collect::<Vec<_>>(), [0, 4, 8]);
    /// let order = Order::new(Seq::source(10).skip(1).step_by(4))?;
    /// assert_eq!(order.iter().map(|item| item.record_index).collect::<Vec<_>>(), [1, 5, 9]);
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
    /// two workers send one part exclusively to each worker, even when both inputs
    /// are shuffled.
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

    /// Transforms each source handle with `f`, preserving the sequence structure.
    /// Calls `f` once per source node, not once per output record.
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
    /// let names = Seq::mix([Seq::source("web"), Seq::source("code").shuffle()]);
    /// let lengths = names.map_sources(|name| counts[name]);
    /// assert_eq!(lengths, Seq::mix([Seq::source(1000), Seq::source(200).shuffle()]));
    /// ```
    #[must_use]
    pub fn map_sources<U, F: FnMut(T) -> U>(self, mut f: F) -> Seq<U> {
        match self.try_map_sources(|t| Ok::<U, Infallible>(f(t))) {
            Ok(seq) => seq,
            Err(never) => match never {},
        }
    }

    /// Transforms sources like [`map_sources`](Seq::map_sources), stopping at the first error.
    /// Sources after the error are not visited. Unvisited inputs and mapped outputs
    /// are dropped normally on error.
    ///
    /// ```
    /// use dataorder::Seq;
    ///
    /// // Parse record counts supplied as strings.
    /// let config = Seq::mix([Seq::source("1000"), Seq::source("200").shuffle()]);
    /// let lengths = config.try_map_sources(str::parse::<usize>)?;
    /// assert_eq!(lengths, Seq::mix([Seq::source(1000), Seq::source(200).shuffle()]));
    /// assert!(Seq::source("unknown").try_map_sources(str::parse::<usize>).is_err());
    /// # Ok::<(), std::num::ParseIntError>(())
    /// ```
    ///
    /// # Errors
    /// The first error `f` returns.
    pub fn try_map_sources<U, E, F: FnMut(T) -> Result<U, E>>(self, mut f: F) -> Result<Seq<U>, E> {
        map_sources(self, &mut f)
    }
}

/// A part of a [`Mix`](Seq::Mix): a sequence and its schedule. `(seq, schedule)` and a bare
/// `seq` (uniform) convert into it.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(deny_unknown_fields))]
pub struct MixPart<T> {
    /// The sequence.
    pub seq: Seq<T>,
    /// How its elements are spread over the mix.
    pub schedule: Schedule,
}

impl<T> From<(Seq<T>, Schedule)> for MixPart<T> {
    fn from((seq, schedule): (Seq<T>, Schedule)) -> Self {
        Self { seq, schedule }
    }
}

impl<T> From<Seq<T>> for MixPart<T> {
    fn from(seq: Seq<T>) -> Self {
        Self { seq, schedule: Schedule::Uniform }
    }
}

/// Map the supported, bounded-depth configuration using ordinary recursive ownership.
fn map_sources<T, U, E>(seq: Seq<T>, f: &mut impl FnMut(T) -> Result<U, E>) -> Result<Seq<U>, E> {
    Ok(match seq {
        Seq::Source(source) => Seq::source(f(source)?),
        Seq::Concat(parts) => Seq::concat(parts.into_iter().map(|p| map_sources(p, f)).collect::<Result<Vec<_>, _>>()?),
        Seq::Mix(parts) => Seq::Mix(
            parts.into_iter().map(|p| Ok(MixPart { seq: map_sources(p.seq, f)?, schedule: p.schedule })).collect::<Result<_, E>>()?,
        ),
        Seq::Shuffle { inner } => map_sources(*inner, f)?.shuffle(),
        Seq::Repeat { times, inner } => map_sources(*inner, f)?.repeat(times),
        Seq::Cycle { len, inner } => map_sources(*inner, f)?.cycle_to(len),
        Seq::ShuffledRepeat { times, inner } => map_sources(*inner, f)?.repeat_shuffled(times),
        Seq::ShuffledCycle { len, inner } => map_sources(*inner, f)?.cycle_to_shuffled(len),
        Seq::Skip { n, inner } => map_sources(*inner, f)?.skip(n),
        Seq::Take { n, inner } => map_sources(*inner, f)?.take(n),
        Seq::StepBy { step, inner } => map_sources(*inner, f)?.step_by(step),
    })
}
