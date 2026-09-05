//! The configuration: a tree of sequence expressions over sources, built by hand or with
//! the builder methods on [`Seq`].

use crate::{Error, MAX_DEPTH, Order, Sampling, Source, float_bits};
use std::convert::Infallible;
use std::hash::{Hash, Hasher};
use std::ops::{Bound, RangeBounds};

/// A sequence expression over sources of type `T` (anything that is a [`Source`]). Leaves
/// are [`Source`](Seq::Source)s; every other variant transforms or combines sequences.
/// Build its order with [`Order::new`]. It is plain data: clone it, compare and hash it,
/// serialize it (with the `serde` feature), or [`map`](Seq::map) its sources to another
/// type.
///
/// Equality and hashing compare the floating-point weights and schedule parameters bit for
/// bit (with `-0.0` taken as `0.0`), so `Seq` is `Eq` and `Hash` whenever `T` is; `Order::new`
/// rejects NaN in either place anyway.
///
/// # Errors and panics
///
/// Anything a hand-built configuration can get wrong is reported by [`Order::new`] as an
/// [`Error`] with the path of the node it was found at: skipping or taking past the end, a
/// zero stride, overflow, schedules, weights, and nesting deeper than [`MAX_DEPTH`]. Two
/// builders panic instead, on mistakes no configuration can express: a reversed
/// [`slice`](Seq::slice) range (a `Skip` and a `Take` would need a negative length) and a
/// [`shard`](Seq::shard) index at or beyond the count (a `Stride` with such an offset is a
/// valid, merely empty, sequence).
///
/// # Depth
///
/// Like any boxed tree, a `Seq` is cloned, compared, hashed, printed, mapped, serialized and
/// dropped by recursion, one stack frame per level. [`Order::new`] and [`check`](Seq::check)
/// cope with any depth: they stop at [`MAX_DEPTH`] and take the rest apart without
/// recursion. Keep the values themselves within a few thousand levels of a thread's stack
/// all the same; no order accepts them deeper.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(deny_unknown_fields))]
#[non_exhaustive]
pub enum Seq<T> {
    /// The elements `0..len()` of a source, in order.
    Source(T),
    /// The parts one after another.
    Concat(Vec<Self>),
    /// The parts interleaved: each part keeps its order and is drawn according to its
    /// [`Sampling`], balanced over the whole length. Schedules are relative to this mix (a
    /// repeated mix restarts them every repetition; repeat the parts to schedule over a
    /// whole run). Empty parts do not affect the order of the others. The total length of
    /// a mix is limited to [`MAX_MIX_LEN`](crate::MAX_MIX_LEN).
    Mix(Vec<MixPart<T>>),
    /// The parts mixed in the proportions of their weights, `total` elements in all: part `i`
    /// contributes `round(wᵢ / Σw · total)` elements (the largest remainders take the
    /// rounding up, so the counts sum to `total`), repeated as often as needed (reshuffling
    /// any shuffle inside for each repetition) and cut to that count, then mixed like
    /// [`Mix`](Seq::Mix) with the parts' schedules. Weights must be finite and nonnegative
    /// with a positive sum, a part with a positive share must have elements, and `total` is
    /// limited to [`MAX_MIX_LEN`](crate::MAX_MIX_LEN) like any mix.
    Weighted {
        /// Length of the order.
        total: usize,
        /// The parts with their weights and schedules.
        parts: Vec<WeightedPart<T>>,
    },
    /// `inner` in a pseudorandom order selected by `seed`, by the order's seed and by the
    /// sources under it that have elements (their [salts](Source::salt) and lengths, in
    /// order of appearance; an empty source, or a part that is empty as a whole, does not
    /// count). Inside a [`Repeat`](Seq::Repeat) the order also depends on the repetition,
    /// so every epoch is shuffled differently.
    Shuffle {
        /// Selects the permutation.
        seed: u64,
        /// The sequence to permute.
        inner: Box<Self>,
    },
    /// `inner`, `times` times over: first as it is, then reshuffled at every shuffle inside
    /// it for each further repetition. `x.repeat(1)` is `x`, and `x.repeat(0)` is empty
    /// (`inner` is validated all the same). Schedules of a mix inside restart every
    /// repetition.
    Repeat {
        /// Number of repetitions.
        times: usize,
        /// The sequence to repeat.
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
    /// Positions `offset, offset + step, offset + 2·step, …` of `inner`, as many as exist:
    /// shard `offset` of `step` shards. An offset at or past the end gives an empty
    /// sequence, not an error, because a shard of a short sequence is legitimately empty.
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
    #[must_use]
    pub fn concat(parts: impl IntoIterator<Item = Self>) -> Self {
        Self::Concat(parts.into_iter().collect())
    }

    /// The parts interleaved, all [`Sampling::Uniform`].
    #[must_use]
    pub fn mix(parts: impl IntoIterator<Item = Self>) -> Self {
        Self::Mix(parts.into_iter().map(MixPart::from).collect())
    }

    /// The parts interleaved, each with its own schedule.
    ///
    /// ```
    /// use dataorder::{Order, Sampling, Seq};
    /// let seq = Seq::mix_with([(Seq::source(700), Sampling::Uniform), (Seq::source(300), Sampling::delayed(0.5))]);
    /// let order = Order::new(seq)?;
    /// // Nothing of the delayed source in the first half.
    /// assert!(order.iter(..500).all(|(&source, _)| source == 700));
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    #[must_use]
    pub fn mix_with(parts: impl IntoIterator<Item = (Self, Sampling)>) -> Self {
        Self::Mix(parts.into_iter().map(MixPart::from).collect())
    }

    /// The parts mixed by weight into `total` elements, all [`Sampling::Uniform`]; see
    /// [`Weighted`](Seq::Weighted).
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// // 60% of a small source (repeated, reshuffled per repetition) and 40% of a large one (cut).
    /// let seq = Seq::weighted(3000, [(Seq::source(100).shuffle(1), 0.6), (Seq::source(5000).shuffle(2), 0.4)]);
    /// let order = Order::new(seq)?;
    /// assert_eq!(order.len(), 3000);
    /// assert_eq!(order.iter(..).filter(|&(&source, _)| source == 100).count(), 1800);
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    #[must_use]
    pub fn weighted(total: usize, parts: impl IntoIterator<Item = (Self, f64)>) -> Self {
        Self::Weighted { total, parts: parts.into_iter().map(WeightedPart::from).collect() }
    }

    /// The parts mixed by weight into `total` elements, each with its own schedule; see
    /// [`Weighted`](Seq::Weighted).
    #[must_use]
    pub fn weighted_with(total: usize, parts: impl IntoIterator<Item = (Self, f64, Sampling)>) -> Self {
        Self::Weighted { total, parts: parts.into_iter().map(WeightedPart::from).collect() }
    }

    /// This sequence in the pseudorandom order selected by `seed`.
    #[must_use]
    pub fn shuffle(self, seed: u64) -> Self {
        Self::Shuffle { seed, inner: Box::new(self) }
    }

    /// This sequence `times` times over: itself, then reshuffled for each further time.
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
    #[must_use]
    pub fn take(self, n: usize) -> Self {
        Self::Take { n, inner: Box::new(self) }
    }

    /// Everything after the first `n` positions (an error when the order is built if there are fewer).
    #[must_use]
    pub fn skip(self, n: usize) -> Self {
        Self::Skip { n, inner: Box::new(self) }
    }

    /// Every `step`-th position starting at `offset`, as many as exist; `offset` may exceed
    /// `step`. See [`Stride`](Seq::Stride).
    #[must_use]
    pub fn stride(self, step: usize, offset: usize) -> Self {
        Self::Stride { step, offset, inner: Box::new(self) }
    }

    /// Shard `index` of `count`: positions `index, index + count, …`. All shards of one
    /// sequence together cover it exactly once, and shard `i` holds position `i` of every
    /// consecutive block of `count` positions, so a mix's schedule is preserved across
    /// workers. A shard of a sequence shorter than `count` may be empty.
    ///
    /// Over a mix, a shard still steps through every element of the mix's interleave and
    /// keeps one in `count` (the parts' cursors skip past what is dropped; a part that is
    /// itself a mix steps its own interleave), so `count` workers sharding one mix do
    /// `count` times its interleaving work in total. When that matters, shard the parts and
    /// mix the shards: each worker then interleaves only its own share, with the same
    /// schedule.
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

    /// The same expression over the sources mapped by `f`, in order of appearance: a
    /// configuration over handles becomes one over loaded datasets. Its order is the same
    /// when the mapped sources keep their lengths and salts.
    ///
    /// ```
    /// use dataorder::Seq;
    /// let paths = Seq::mix([Seq::source("web.bin"), Seq::source("code.bin").shuffle(1)]);
    /// let lens = paths.map(|path| path.len()); // a Seq<usize>
    /// assert_eq!(lens, Seq::mix([Seq::source(7), Seq::source(8).shuffle(1)]));
    /// ```
    #[must_use]
    pub fn map<U, F: FnMut(T) -> U>(self, mut f: F) -> Seq<U> {
        match self.try_map(|t| Ok::<U, Infallible>(f(t))) {
            Ok(seq) => seq,
            Err(never) => match never {},
        }
    }

    /// [`map`](Seq::map) with a fallible function: the first error comes back and the
    /// sources after it are not visited.
    ///
    /// ```
    /// use dataorder::Seq;
    /// let open = |path: &str| if path.ends_with(".bin") { Ok(path.len()) } else { Err(path.to_string()) };
    /// let paths = Seq::mix([Seq::source("web.bin"), Seq::source("code.bin").shuffle(1)]);
    /// assert_eq!(paths.clone().try_map(open), Ok(Seq::mix([Seq::source(7), Seq::source(8).shuffle(1)])));
    /// assert_eq!(Seq::concat([paths, Seq::source("notes.txt")]).try_map(open), Err("notes.txt".to_string()));
    /// ```
    ///
    /// # Errors
    /// The first error `f` returns.
    pub fn try_map<U, E, F: FnMut(T) -> Result<U, E>>(self, mut f: F) -> Result<Seq<U>, E> {
        self.try_map_with(&mut f)
    }

    /// Takes the tree apart without recursion: what dropping it does, on the heap instead
    /// of the stack, for configurations too deep to drop the usual way.
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
            Self::Skip { n, inner } => Seq::Skip { n, inner: Box::new(inner.try_map_with(f)?) },
            Self::Take { n, inner } => Seq::Take { n, inner: Box::new(inner.try_map_with(f)?) },
            Self::Stride { step, offset, inner } => Seq::Stride { step, offset, inner: Box::new(inner.try_map_with(f)?) },
        })
    }
}

impl<T: Source> Seq<T> {
    /// Validates the configuration and returns the length of its order, without consuming
    /// it: the checks of [`Order::new`], over the sources' lengths.
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
            Self::Skip { n, inner: i } => Seq::Skip { n: *n, inner: inner(i) },
            Self::Take { n, inner: i } => Seq::Take { n: *n, inner: inner(i) },
            Self::Stride { step, offset, inner: i } => Seq::Stride { step: *step, offset: *offset, inner: inner(i) },
        }
    }
}
