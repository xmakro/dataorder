//! The configuration: a tree of sequence expressions over sources, built by hand or with
//! the builder methods on [`Seq`].

use crate::{Source, Error, Order, Sampling};
use std::ops::{Bound, RangeBounds};

/// A sequence expression over sources of type `T` (anything that is a
/// [`Source`](crate::Source)). Leaves are [`Source`](Seq::Source)s; every other variant
/// transforms or combines sequences. Build its order with
/// [`Order::new`](crate::Order::new). It is plain data: clone it, compare it,
/// serialize it (with the `serde` feature), or [`map`](Seq::map) its sources to another type.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Seq<T> {
    /// The elements `0..len()` of a source, in order.
    Source(T),
    /// The parts one after another.
    Concat(Vec<Self>),
    /// The parts interleaved: each part keeps its order and is drawn according to its
    /// [`Sampling`], balanced over the whole length. The total length of a mix is limited to
    /// 2⁴⁶.
    Mix(Vec<(Self, Sampling)>),
    /// The parts mixed in the proportions of their weights, `total` elements in all: part `i`
    /// contributes `round(wᵢ / Σw · total)` elements (the largest remainders take the
    /// rounding up, so the counts sum to `total`), repeated as often as needed (reshuffling
    /// any shuffle inside for each repetition) and cut to that count, then mixed like
    /// [`Mix`](Seq::Mix) with the parts' schedules. Weights must be finite and nonnegative
    /// with a positive sum, and a part with a positive share must have elements.
    Weighted {
        /// Length of the order.
        total: usize,
        /// The parts with their weights and schedules.
        parts: Vec<(Seq<T>, f64, Sampling)>,
    },
    /// `inner` in a pseudorandom order selected by `seed`. Inside a [`Repeat`](Seq::Repeat)
    /// the order also depends on the repetition, so every epoch is shuffled differently.
    Shuffle {
        /// Selects the permutation.
        seed: u64,
        /// The sequence to permute.
        inner: Box<Self>,
    },
    /// `inner`, `times` times over: first as it is, then reshuffled at every shuffle inside
    /// it for each further repetition. `x.repeat(1)` is `x`.
    Repeat {
        /// Number of repetitions.
        times: usize,
        /// The sequence to repeat.
        inner: Box<Self>,
    },
    /// `inner` without its first `n` positions. Skipping more than there are is an error at
    /// the time the order is built, unlike `Iterator::skip`: configurations are validated, and a silently
    /// empty sequence hides a mistake.
    Skip {
        /// Positions dropped from the front.
        n: usize,
        /// The sequence to skip into.
        inner: Box<Self>,
    },
    /// The first `n` positions of `inner`. Taking more than there are is an error at the
    /// time the order is built, unlike `Iterator::take`.
    Take {
        /// Positions kept.
        n: usize,
        /// The sequence to take from.
        inner: Box<Self>,
    },
    /// Positions `offset, offset + step, offset + 2·step, …` of `inner`: shard `offset` of
    /// `step` shards.
    Stride {
        /// Distance between kept positions.
        step: usize,
        /// First kept position.
        offset: usize,
        /// The sequence to stride over.
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
    #[must_use]
    pub fn concat(parts: impl IntoIterator<Item = Self>) -> Self {
        Self::Concat(parts.into_iter().collect())
    }

    /// The parts interleaved, all [`Sampling::Uniform`].
    #[must_use]
    pub fn mix(parts: impl IntoIterator<Item = Self>) -> Self {
        Self::Mix(parts.into_iter().map(|p| (p, Sampling::Uniform)).collect())
    }

    /// The parts interleaved, each with its own schedule.
    #[must_use]
    pub fn mix_with(parts: impl IntoIterator<Item = (Self, Sampling)>) -> Self {
        Self::Mix(parts.into_iter().collect())
    }

    /// The parts mixed by weight into `total` elements, all [`Sampling::Uniform`]; see
    /// [`Weighted`](Seq::Weighted).
    #[must_use]
    pub fn weighted(total: usize, parts: impl IntoIterator<Item = (Seq<T>, f64)>) -> Seq<T> {
        Self::Weighted { total, parts: parts.into_iter().map(|(p, w)| (p, w, Sampling::Uniform)).collect() }
    }

    /// The parts mixed by weight into `total` elements, each with its own schedule; see
    /// [`Weighted`](Seq::Weighted).
    #[must_use]
    pub fn weighted_with(total: usize, parts: impl IntoIterator<Item = (Seq<T>, f64, Sampling)>) -> Seq<T> {
        Self::Weighted { total, parts: parts.into_iter().collect() }
    }

    /// This sequence in the pseudorandom order selected by `seed`.
    #[must_use]
    pub fn shuffle(self, seed: u64) -> Self {
        Self::Shuffle { seed, inner: Box::new(self) }
    }

    /// This sequence `times` times over: itself, then reshuffled for each further time.
    #[must_use]
    pub fn repeat(self, times: usize) -> Self {
        Self::Repeat { times, inner: Box::new(self) }
    }

    /// The positions in `range` of this sequence: a [`Skip`](Seq::Skip) of its start and a
    /// [`Take`](Seq::Take) of its length, either omitted when trivial.
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

    /// Every `step`-th position starting at `offset`.
    #[must_use]
    pub fn stride(self, step: usize, offset: usize) -> Self {
        Self::Stride { step, offset, inner: Box::new(self) }
    }

    /// Shard `index` of `count`: positions `index, index + count, …`. All shards of one
    /// sequence together cover it exactly once, and shard `i` holds position `i` of every
    /// consecutive block of `count` positions.
    ///
    /// Over a mix, a shard still walks every element of the mix and keeps one in `count`,
    /// so `count` workers sharding one mix do `count` times its interleaving work in total.
    /// When that matters, shard the parts and mix the shards: each worker then interleaves
    /// only its own share, with the same schedule.
    #[must_use]
    pub fn shard(self, index: usize, count: usize) -> Self {
        self.stride(count, index)
    }

    /// The same expression over the sources mapped by `f`, in order of appearance: a
    /// configuration over handles becomes one over loaded datasets.
    #[must_use]
    pub fn map<U, F: FnMut(T) -> U>(self, mut f: F) -> Seq<U> {
        self.map_with(&mut f)
    }

    fn map_with<U, F: FnMut(T) -> U>(self, f: &mut F) -> Seq<U> {
        match self {
            Self::Source(t) => Seq::Source(f(t)),
            Self::Concat(parts) => Seq::Concat(parts.into_iter().map(|p| p.map_with(f)).collect()),
            Self::Mix(parts) => Seq::Mix(parts.into_iter().map(|(p, s)| (p.map_with(f), s)).collect()),
            Self::Weighted { total, parts } => Seq::Weighted { total, parts: parts.into_iter().map(|(p, w, s)| (p.map_with(f), w, s)).collect() },
            Self::Shuffle { seed, inner } => Seq::Shuffle { seed, inner: Box::new(inner.map_with(f)) },
            Self::Repeat { times, inner } => Seq::Repeat { times, inner: Box::new(inner.map_with(f)) },
            Self::Skip { n, inner } => Seq::Skip { n, inner: Box::new(inner.map_with(f)) },
            Self::Take { n, inner } => Seq::Take { n, inner: Box::new(inner.map_with(f)) },
            Self::Stride { step, offset, inner } => Seq::Stride { step, offset, inner: Box::new(inner.map_with(f)) },
        }
    }
}

impl<T: Source> Seq<T> {
    /// Validates the configuration and returns the length of its order, without consuming
    /// it: the checks of [`Order::new`], over the sources' lengths.
    ///
    /// # Errors
    /// Whatever [`Order::new`] would report.
    pub fn check(&self) -> Result<usize, Error> {
        Order::new(self.lens()).map(|o| o.len())
    }

    /// The same expression over the sources' lengths.
    pub(crate) fn lens(&self) -> Seq<usize> {
        match self {
            Self::Source(t) => Seq::Source(t.len()),
            Self::Concat(parts) => Seq::Concat(parts.iter().map(Self::lens).collect()),
            Self::Mix(parts) => Seq::Mix(parts.iter().map(|(p, s)| (p.lens(), *s)).collect()),
            Self::Weighted { total, parts } => Seq::Weighted { total: *total, parts: parts.iter().map(|(p, w, s)| (p.lens(), *w, *s)).collect() },
            Self::Shuffle { seed, inner } => Seq::Shuffle { seed: *seed, inner: Box::new(inner.lens()) },
            Self::Repeat { times, inner } => Seq::Repeat { times: *times, inner: Box::new(inner.lens()) },
            Self::Skip { n, inner } => Seq::Skip { n: *n, inner: Box::new(inner.lens()) },
            Self::Take { n, inner } => Seq::Take { n: *n, inner: Box::new(inner.lens()) },
            Self::Stride { step, offset, inner } => Seq::Stride { step: *step, offset: *offset, inner: Box::new(inner.lens()) },
        }
    }
}
