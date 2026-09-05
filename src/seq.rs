//! Sequence configuration and builders. Each node is a source, a transform or a
//! combination of other sequences; [`Order::new`] validates and compiles the tree.

use crate::{Error, MAX_DEPTH, Order, Sampling, Source, float_bits};
use std::convert::Infallible;
use std::hash::{Hash, Hasher};
use std::ops::{Bound, RangeBounds};

/// A description of how to order sources of type `T`.
///
/// Build a sequence with [`source`](Seq::source) and the methods below, then pass it
/// to [`Order::new`]. `T` must implement [`Source`] when you build the order; before
/// that, it can be any type, such as a path you later [`map`](Seq::map) to a dataset.
/// You can also construct enum variants directly. Builders store the configuration
/// without loading records or generating indices.
///
/// `Seq` supports cloning, comparison, hashing and, with the `serde` feature,
/// serialization when `T` does. Floating-point weights and schedule parameters are
/// compared by their bits, treating `-0.0` as `0.0`. NaN parameters are rejected
/// during validation.
///
/// # Errors and panics
///
/// [`Order::new`] and [`check`](Seq::check) report invalid configurations as an
/// [`Error`] with the path to the invalid node. This includes out-of-range skips
/// and takes, zero strides, overflow, invalid schedules or weights, and excessive depth.
///
/// Two builders check their arguments immediately and panic: [`slice`](Seq::slice)
/// for reversed or overflowing range bounds, and [`shard`](Seq::shard) for an index
/// outside `0..count`. See those methods for details.
///
/// # Depth
///
/// [`Order::new`] and [`check`](Seq::check) stop at [`MAX_DEPTH`]. Construction also
/// disposes of rejected trees without recursing through the remaining nodes.
///
/// Other tree operations, including cloning, mapping, serialization and ordinary
/// dropping, recurse once per level. Their stack use depends on depth and the size
/// of `T`. Large inline sources, such as arrays, can exhaust the stack even below
/// `MAX_DEPTH`; use handles or boxed sources for those trees.
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
    /// Exactly `total` elements, divided among the parts in proportion to their weights.
    ///
    /// Each part is [cycled](Seq::Cycle) to its assigned count, then interleaved
    /// according to its schedule. Cycling repeats or truncates the part as needed
    /// and reseeds any existing shuffles for each additional epoch.
    ///
    /// Counts use largest-remainder rounding: first round every exact quota
    /// `weight / sum_of_weights * total` down, then give the remaining positions to
    /// the largest fractional remainders. Ties go to the lowest part index. This
    /// uses the exact binary values of the weights and makes the counts sum to `total`.
    ///
    /// Weights must be finite and nonnegative. A positive `total` requires at least
    /// one positive weight, and a part assigned a positive count must not be empty.
    /// Like any mix, `total` is limited to [`MAX_MIX_LEN`](crate::MAX_MIX_LEN).
    Weighted {
        /// Length of this weighted sequence.
        total: usize,
        /// The parts with their weights and schedules.
        parts: Vec<WeightedPart<T>>,
    },
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
    /// With more than one repetition, any repeats inside `inner` become one level deeper.
    /// Their later epochs then shuffle differently even during this repeat's first epoch:
    /// adding an outer repeat does not preserve the whole prefix of a nested repeat.
    Repeat {
        /// Number of repetitions.
        times: usize,
        /// The sequence to repeat.
        inner: Box<Self>,
    },
    /// Exactly `len` positions of `inner`, repeating or truncating it as needed.
    ///
    /// Each additional epoch reseeds existing shuffles, as [`Repeat`](Seq::Repeat)
    /// does. If more than one epoch is needed, nested repeats also change depth.
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
    /// Positions `offset, offset + step, offset + 2·step, …` of `inner`.
    /// `step` must be positive. An offset at or past the end gives an empty sequence.
    /// Use [`shard`](Seq::shard) to partition positions among workers.
    Stride {
        /// Distance between kept positions.
        step: usize,
        /// First kept position.
        offset: usize,
        /// The sequence to stride over.
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

/// A part of a [`Weighted`](Seq::Weighted) mix: a sequence, its weight and its schedule.
/// `(seq, weight, sampling)` and `(seq, weight)` (uniform) convert into it. Equality and
/// hashing compare the weight bit for bit (with `-0.0` taken as `0.0`).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(deny_unknown_fields))]
pub struct WeightedPart<T> {
    /// The sequence.
    pub seq: Seq<T>,
    /// Its share of the total, relative to the other weights.
    pub weight: f64,
    /// How its elements are spread over the mix.
    pub sampling: Sampling,
}

impl<T> From<(Seq<T>, f64, Sampling)> for WeightedPart<T> {
    fn from((seq, weight, sampling): (Seq<T>, f64, Sampling)) -> Self {
        Self { seq, weight, sampling }
    }
}

impl<T> From<(Seq<T>, f64)> for WeightedPart<T> {
    fn from((seq, weight): (Seq<T>, f64)) -> Self {
        Self { seq, weight, sampling: Sampling::Uniform }
    }
}

impl<T: PartialEq> PartialEq for WeightedPart<T> {
    fn eq(&self, other: &Self) -> bool {
        self.seq == other.seq && float_bits(self.weight) == float_bits(other.weight) && self.sampling == other.sampling
    }
}

impl<T: Eq> Eq for WeightedPart<T> {}

impl<T: Hash> Hash for WeightedPart<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.seq.hash(state);
        float_bits(self.weight).hash(state);
        self.sampling.hash(state);
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
    /// let elements: Vec<(usize, usize)> = order.iter(..).map(|(&s, i)| (s, i)).collect();
    /// assert_eq!(elements, [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]);
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    #[must_use]
    pub fn concat(parts: impl IntoIterator<Item = Self>) -> Self {
        Self::Concat(parts.into_iter().collect())
    }

    /// The parts interleaved, all [`Sampling::Uniform`].
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// // Each part keeps its order and is spread evenly: the longer one appears twice as often.
    /// let order = Order::new(Seq::mix([Seq::source(4), Seq::source(2)]))?;
    /// let sources: Vec<usize> = order.iter(..).map(|(&s, _)| s).collect();
    /// assert_eq!(sources, [4, 4, 2, 4, 4, 2]);
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    #[must_use]
    pub fn mix(parts: impl IntoIterator<Item = Self>) -> Self {
        Self::Mix(parts.into_iter().map(MixPart::from).collect())
    }

    /// Interleaves parts with individual schedules.
    /// Accepts `(seq, sampling)` pairs or other values that convert into [`MixPart`].
    ///
    /// ```
    /// use dataorder::{Order, Sampling, Seq};
    /// let seq = Seq::mix_with([
    ///     (Seq::source(700), Sampling::Uniform),
    ///     (Seq::source(300), Sampling::delayed(0.5)),
    /// ]);
    /// let order = Order::new(seq)?;
    /// // This order draws only from the uniform source in its first half.
    /// assert!(order.iter(..500).all(|(&source, _)| source == 700));
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    #[must_use]
    pub fn mix_with(parts: impl IntoIterator<Item = impl Into<MixPart<T>>>) -> Self {
        Self::Mix(parts.into_iter().map(Into::into).collect())
    }

    /// The parts mixed by weight into `total` elements, all [`Sampling::Uniform`]; see
    /// [`Weighted`](Seq::Weighted).
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// // Repeat the small source to supply 60%; truncate the large source to supply 40%.
    /// let seq = Seq::weighted(3000, [
    ///     (Seq::source(100).shuffle(1), 0.6),
    ///     (Seq::source(5000).shuffle(2), 0.4),
    /// ]);
    /// let order = Order::new(seq)?;
    /// assert_eq!(order.len(), 3000);
    /// assert_eq!(order.iter(..).filter(|&(&source, _)| source == 100).count(), 1800);
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    #[must_use]
    pub fn weighted(total: usize, parts: impl IntoIterator<Item = (Self, f64)>) -> Self {
        Self::Weighted { total, parts: parts.into_iter().map(WeightedPart::from).collect() }
    }

    /// Mixes exactly `total` elements by weight, with an individual schedule for each part.
    /// Accepts `(seq, weight, sampling)` triples or other values that convert into
    /// [`WeightedPart`]. See [`Weighted`](Seq::Weighted) for count allocation.
    ///
    /// ```
    /// use dataorder::{Order, Sampling, Seq};
    /// // Draw 750 elements from the first source and 250 from the delayed source.
    /// let seq = Seq::weighted_with(1000, [
    ///     (Seq::source(300), 3.0, Sampling::Uniform),
    ///     (Seq::source(100), 1.0, Sampling::delayed(0.5)),
    /// ]);
    /// let order = Order::new(seq)?;
    /// assert_eq!(order.iter(..).filter(|&(&s, _)| s == 100).count(), 250);
    /// assert!(order.iter(..490).all(|(&s, _)| s == 300));
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    #[must_use]
    pub fn weighted_with(total: usize, parts: impl IntoIterator<Item = impl Into<WeightedPart<T>>>) -> Self {
        Self::Weighted { total, parts: parts.into_iter().map(Into::into).collect() }
    }

    /// This sequence in the pseudorandom order selected by `seed`.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::source(100).shuffle(1))?;
    /// let mut indices: Vec<usize> = order.iter(..).map(|(_, i)| i).collect();
    /// assert_ne!(indices[..5], [0, 1, 2, 3, 4]);
    /// indices.sort_unstable();
    /// assert_eq!(indices, (0..100).collect::<Vec<_>>());
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    #[must_use]
    pub fn shuffle(self, seed: u64) -> Self {
        Self::Shuffle { seed, inner: Box::new(self) }
    }

    /// Repeats this sequence `times` times, reseeding any existing shuffles each epoch.
    /// Adding more than one repetition also changes the contexts of nested repeats;
    /// see [`Repeat`](Seq::Repeat).
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::source(1000).shuffle(1).repeat(2))?;
    /// let epoch = |e: usize| order.iter(e * 1000..(e + 1) * 1000).map(|(_, i)| i).collect::<Vec<_>>();
    /// assert_ne!(epoch(0), epoch(1));
    /// let mut sorted = epoch(1);
    /// sorted.sort_unstable();
    /// assert_eq!(sorted, (0..1000).collect::<Vec<_>>());
    /// # Ok::<(), dataorder::Error>(())
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
    /// assert!(order.iter(..).eq(epochs.iter(..2500)));
    /// let longest = Order::new(Seq::source(1000).shuffle(1).cycle(usize::MAX))?;
    /// assert_eq!(longest.len(), usize::MAX);
    /// assert!(longest.get(usize::MAX - 1).1 < 1000);
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    #[must_use]
    pub fn cycle(self, len: usize) -> Self {
        Self::Cycle { len, inner: Box::new(self) }
    }

    /// The positions in `range` of this sequence: a [`Skip`](Seq::Skip) of its start and a
    /// [`Take`](Seq::Take) of its length, either omitted when trivial.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::source(10).slice(3..=5))?;
    /// assert_eq!(order.iter(..).map(|(_, i)| i).collect::<Vec<_>>(), [3, 4, 5]);
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    ///
    /// # Panics
    /// If the range's end lies before its start, or a bound is `usize::MAX` where one more
    /// would be needed (an exclusive start or an inclusive end at `usize::MAX`).
    #[must_use]
    pub fn slice(self, range: impl RangeBounds<usize>) -> Self {
        let bump = |x: usize| x.checked_add(1).expect("dataorder: slice bound overflows usize");
        let start = match range.start_bound() {
            Bound::Included(&s) => s,
            Bound::Excluded(&s) => bump(s),
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&e) => Some(bump(e)),
            Bound::Excluded(&e) => Some(e),
            Bound::Unbounded => None,
        };
        let skipped = if start == 0 { self } else { self.skip(start) };
        match end {
            Some(end) => skipped.take(end.checked_sub(start).expect("dataorder: slice end before start")),
            None => skipped,
        }
    }

    /// The first `n` positions (an error when the order is built if there are fewer).
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let all = Order::new(Seq::source(10).shuffle(1))?;
    /// let first = Order::new(Seq::source(10).shuffle(1).take(3))?;
    /// assert!(first.iter(..).eq(all.iter(..3)));
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    #[must_use]
    pub fn take(self, n: usize) -> Self {
        Self::Take { n, inner: Box::new(self) }
    }

    /// Everything after the first `n` positions (an error when the order is built if there are fewer).
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let all = Order::new(Seq::source(10).shuffle(1))?;
    /// let rest = Order::new(Seq::source(10).shuffle(1).skip(7))?;
    /// assert!(rest.iter(..).eq(all.iter(7..)));
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    #[must_use]
    pub fn skip(self, n: usize) -> Self {
        Self::Skip { n, inner: Box::new(self) }
    }

    /// Every `step`-th position starting at `offset`, as many as exist; `offset` may exceed
    /// `step`. See [`Stride`](Seq::Stride).
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::source(10).stride(4, 1))?;
    /// assert_eq!(order.iter(..).map(|(_, i)| i).collect::<Vec<_>>(), [1, 5, 9]);
    /// assert!(Order::new(Seq::source(10).stride(4, 12))?.is_empty());
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    #[must_use]
    pub fn stride(self, step: usize, offset: usize) -> Self {
        Self::Stride { step, offset, inner: Box::new(self) }
    }

    /// Shard `index` of `count`: positions `index, index + count, …`. All shards of one
    /// sequence together cover it exactly once, and shard `i` holds position `i` of every
    /// consecutive block of `count` positions, so a mix's schedule is preserved across
    /// workers. A shard of a sequence shorter than `count` may be empty.
    ///
    /// Sharding a mix keeps one in `count` interleaved positions. The mix advances
    /// past unselected positions, or seeks for long skips. Across `count` workers,
    /// this can cost up to `count` times the global mix's interleaving work.
    ///
    /// Sharding each part before mixing can reduce that work, but produces a
    /// different order. Rounding each part's length can also make a worker's
    /// schedule infeasible. For example, 3 uniform elements and 1 delayed until
    /// progress 0.75 fit globally. Splitting each part two ways gives worker 0
    /// lengths 2 and 1: its delayed element needs more than the available last
    /// quarter of its 3-position order. Shard the completed mix to preserve its
    /// global position partition and schedule.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let seq = Seq::source(10).shuffle(1);
    /// let all: Vec<usize> = Order::new(seq.clone())?.iter(..).map(|(_, i)| i).collect();
    /// let shard: Vec<usize> = Order::new(seq.shard(4, 1))?.iter(..).map(|(_, i)| i).collect();
    /// assert_eq!(shard, [all[1], all[5], all[9]]);
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    ///
    /// # Panics
    /// If `index >= count` (which includes `count == 0`): such a shard would duplicate
    /// another worker's data.
    #[must_use]
    pub fn shard(self, count: usize, index: usize) -> Self {
        assert!(index < count, "dataorder: shard index {index} out of range for {count} shards");
        self.stride(count, index)
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
    /// Sources after the error are not visited.
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
        self.try_map_with(&mut f)
    }

    /// Drops the tree with an explicit heap stack, avoiding recursive drop for
    /// configurations that exceed the depth limit.
    pub(crate) fn dismantle(self) {
        let mut stack = vec![self];
        while let Some(seq) = stack.pop() {
            match seq {
                Self::Source(_) => {}
                Self::Concat(parts) => stack.extend(parts),
                Self::Mix(parts) => stack.extend(parts.into_iter().map(|p| p.seq)),
                Self::Weighted { parts, .. } => stack.extend(parts.into_iter().map(|p| p.seq)),
                Self::Shuffle { inner, .. }
                | Self::Repeat { inner, .. }
                | Self::Cycle { inner, .. }
                | Self::Skip { inner, .. }
                | Self::Take { inner, .. }
                | Self::Stride { inner, .. } => stack.push(*inner),
            }
        }
    }

    fn try_map_with<U, E, F: FnMut(T) -> Result<U, E>>(self, f: &mut F) -> Result<Seq<U>, E> {
        Ok(match self {
            Self::Source(t) => Seq::Source(f(t)?),
            Self::Concat(parts) => Seq::Concat(parts.into_iter().map(|p| p.try_map_with(f)).collect::<Result<_, E>>()?),
            Self::Mix(parts) => Seq::Mix(
                parts.into_iter().map(|p| Ok(MixPart { seq: p.seq.try_map_with(f)?, sampling: p.sampling })).collect::<Result<_, E>>()?,
            ),
            Self::Weighted { total, parts } => Seq::Weighted {
                total,
                parts: parts
                    .into_iter()
                    .map(|p| Ok(WeightedPart { seq: p.seq.try_map_with(f)?, weight: p.weight, sampling: p.sampling }))
                    .collect::<Result<_, E>>()?,
            },
            Self::Shuffle { seed, inner } => Seq::Shuffle { seed, inner: Box::new(inner.try_map_with(f)?) },
            Self::Repeat { times, inner } => Seq::Repeat { times, inner: Box::new(inner.try_map_with(f)?) },
            Self::Cycle { len, inner } => Seq::Cycle { len, inner: Box::new(inner.try_map_with(f)?) },
            Self::Skip { n, inner } => Seq::Skip { n, inner: Box::new(inner.try_map_with(f)?) },
            Self::Take { n, inner } => Seq::Take { n, inner: Box::new(inner.try_map_with(f)?) },
            Self::Stride { step, offset, inner } => Seq::Stride { step, offset, inner: Box::new(inner.try_map_with(f)?) },
        })
    }
}

impl<T: Source> Seq<T> {
    /// Validates the configuration and returns its length without consuming it.
    /// Performs the same checks as [`Order::new`], using the sources' lengths.
    ///
    /// ```
    /// use dataorder::{ErrorKind, Seq};
    /// assert_eq!(Seq::source(10).skip(3).check(), Ok(7));
    /// let err = Seq::concat([Seq::source(10), Seq::source(5).take(6)]).check().unwrap_err();
    /// assert_eq!(err.kind(), &ErrorKind::TakeOutOfRange { n: 6, len: 5 });
    /// assert_eq!(err.path(), [1]); // the second part of the concat
    /// ```
    ///
    /// # Errors
    /// Whatever [`Order::new`] would report.
    pub fn check(&self) -> Result<usize, Error> {
        Order::new(self.lens()).map(|o| o.len())
    }

    /// The same expression over the sources' lengths, cut off where [`Order::new`] stops
    /// looking (one level beyond [`MAX_DEPTH`]), so that checking never recurses deeper than
    /// compiling does.
    pub(crate) fn lens(&self) -> Seq<usize> {
        self.lens_at(1)
    }

    fn lens_at(&self, level: u32) -> Seq<usize> {
        if level > MAX_DEPTH {
            return Seq::Source(0);
        }
        let inner = |s: &Self| Box::new(s.lens_at(level + 1));
        match self {
            Self::Source(t) => Seq::Source(t.len()),
            Self::Concat(parts) => Seq::Concat(parts.iter().map(|p| p.lens_at(level + 1)).collect()),
            Self::Mix(parts) => Seq::Mix(parts.iter().map(|p| MixPart { seq: p.seq.lens_at(level + 1), sampling: p.sampling }).collect()),
            Self::Weighted { total, parts } => Seq::Weighted {
                total: *total,
                parts: parts
                    .iter()
                    .map(|p| WeightedPart { seq: p.seq.lens_at(level + 1), weight: p.weight, sampling: p.sampling })
                    .collect(),
            },
            Self::Shuffle { seed, inner: i } => Seq::Shuffle { seed: *seed, inner: inner(i) },
            Self::Repeat { times, inner: i } => Seq::Repeat { times: *times, inner: inner(i) },
            Self::Cycle { len, inner: i } => Seq::Cycle { len: *len, inner: inner(i) },
            Self::Skip { n, inner: i } => Seq::Skip { n: *n, inner: inner(i) },
            Self::Take { n, inner: i } => Seq::Take { n: *n, inner: inner(i) },
            Self::Stride { step, offset, inner: i } => Seq::Stride { step: *step, offset: *offset, inner: inner(i) },
        }
    }
}
