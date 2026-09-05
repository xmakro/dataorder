//! The configuration: a tree of sequence expressions over sources, built by hand or with
//! the builder methods on [`Seq`].

use crate::{Source, Error, Order, Sampling};
use std::ops::{Bound, RangeBounds};

/// A sequence expression over sources of type `T` (anything that is a
/// [`Source`](crate::Source)). Leaves are [`Source`](Seq::Source)s; every other variant
/// transforms or combines sequences. Compile it with
/// [`Order::compile`](crate::Order::compile). It is plain data: clone it, compare it,
/// serialize it (with the `serde` feature), or [`map`](Seq::map) its sources to another type.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Seq<T> {
    /// The elements `0..len()` of a source, in order.
    Source(T),
    /// The parts one after another.
    Concat(Vec<Seq<T>>),
    /// The parts interleaved: each part keeps its order and is drawn according to its
    /// [`Sampling`], balanced over the whole length. The total length of a mix is limited to
    /// 2⁴⁶.
    Mix(Vec<(Seq<T>, Sampling)>),
    /// `inner` in a pseudorandom order selected by `seed`. Inside a [`Repeat`](Seq::Repeat)
    /// the order also depends on the repetition, so every epoch is shuffled differently.
    Shuffle {
        /// Selects the permutation.
        seed: u64,
        /// The sequence to permute.
        inner: Box<Seq<T>>,
    },
    /// `inner`, `times` times over: first as it is, then reshuffled at every shuffle inside
    /// it for each further repetition. `x.repeat(1)` is `x`.
    Repeat {
        /// Number of repetitions.
        times: usize,
        /// The sequence to repeat.
        inner: Box<Seq<T>>,
    },
    /// `inner` without its first `n` positions. Skipping more than there are is an error at
    /// compile time, unlike `Iterator::skip`: configurations are validated, and a silently
    /// empty sequence hides a mistake.
    Skip {
        /// Positions dropped from the front.
        n: usize,
        /// The sequence to skip into.
        inner: Box<Seq<T>>,
    },
    /// The first `n` positions of `inner`. Taking more than there are is an error at compile
    /// time, unlike `Iterator::take`.
    Take {
        /// Positions kept.
        n: usize,
        /// The sequence to take from.
        inner: Box<Seq<T>>,
    },
    /// Positions `offset, offset + step, offset + 2·step, …` of `inner`: shard `offset` of
    /// `step` shards.
    Stride {
        /// Distance between kept positions.
        step: usize,
        /// First kept position.
        offset: usize,
        /// The sequence to stride over.
        inner: Box<Seq<T>>,
    },
}

impl<T> Seq<T> {
    /// The elements of `source`, in order.
    #[must_use]
    pub fn source(source: T) -> Seq<T> {
        Seq::Source(source)
    }

    /// The parts one after another.
    #[must_use]
    pub fn concat(parts: impl IntoIterator<Item = Seq<T>>) -> Seq<T> {
        Seq::Concat(parts.into_iter().collect())
    }

    /// The parts interleaved, all [`Sampling::Uniform`].
    #[must_use]
    pub fn mix(parts: impl IntoIterator<Item = Seq<T>>) -> Seq<T> {
        Seq::Mix(parts.into_iter().map(|p| (p, Sampling::Uniform)).collect())
    }

    /// The parts interleaved, each with its own schedule.
    #[must_use]
    pub fn mix_with(parts: impl IntoIterator<Item = (Seq<T>, Sampling)>) -> Seq<T> {
        Seq::Mix(parts.into_iter().collect())
    }

    /// This sequence in the pseudorandom order selected by `seed`.
    #[must_use]
    pub fn shuffle(self, seed: u64) -> Seq<T> {
        Seq::Shuffle { seed, inner: Box::new(self) }
    }

    /// This sequence `times` times over: itself, then reshuffled for each further time.
    #[must_use]
    pub fn repeat(self, times: usize) -> Seq<T> {
        Seq::Repeat { times, inner: Box::new(self) }
    }

    /// The positions in `range` of this sequence: a [`Skip`](Seq::Skip) of its start and a
    /// [`Take`](Seq::Take) of its length, either omitted when trivial.
    ///
    /// # Panics
    /// If the range's end lies before its start, or a bound is `usize::MAX` where one more
    /// would be needed (an exclusive start or an inclusive end at `usize::MAX`).
    #[must_use]
    pub fn slice(self, range: impl RangeBounds<usize>) -> Seq<T> {
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

    /// The first `n` positions (an error at compile time if there are fewer).
    #[must_use]
    pub fn take(self, n: usize) -> Seq<T> {
        Seq::Take { n, inner: Box::new(self) }
    }

    /// Everything after the first `n` positions (an error at compile time if there are fewer).
    #[must_use]
    pub fn skip(self, n: usize) -> Seq<T> {
        Seq::Skip { n, inner: Box::new(self) }
    }

    /// Every `step`-th position starting at `offset`.
    #[must_use]
    pub fn stride(self, step: usize, offset: usize) -> Seq<T> {
        Seq::Stride { step, offset, inner: Box::new(self) }
    }

    /// Shard `index` of `count`: positions `index, index + count, …`. All shards of one
    /// sequence together cover it exactly once, and shard `i` holds position `i` of every
    /// consecutive block of `count` positions.
    #[must_use]
    pub fn shard(self, index: usize, count: usize) -> Seq<T> {
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
            Seq::Source(t) => Seq::Source(f(t)),
            Seq::Concat(parts) => Seq::Concat(parts.into_iter().map(|p| p.map_with(f)).collect()),
            Seq::Mix(parts) => Seq::Mix(parts.into_iter().map(|(p, s)| (p.map_with(f), s)).collect()),
            Seq::Shuffle { seed, inner } => Seq::Shuffle { seed, inner: Box::new(inner.map_with(f)) },
            Seq::Repeat { times, inner } => Seq::Repeat { times, inner: Box::new(inner.map_with(f)) },
            Seq::Skip { n, inner } => Seq::Skip { n, inner: Box::new(inner.map_with(f)) },
            Seq::Take { n, inner } => Seq::Take { n, inner: Box::new(inner.map_with(f)) },
            Seq::Stride { step, offset, inner } => Seq::Stride { step, offset, inner: Box::new(inner.map_with(f)) },
        }
    }
}

impl<T: Source> Seq<T> {
    /// Validates the configuration and returns the length of its order, without consuming
    /// it: the checks of [`Order::compile`], over the sources' lengths.
    ///
    /// # Errors
    /// Whatever [`Order::compile`] would report.
    pub fn check(&self) -> Result<usize, Error> {
        Order::compile(self.lens()).map(|o| o.len())
    }

    /// The same expression over the sources' lengths.
    fn lens(&self) -> Seq<usize> {
        match self {
            Seq::Source(t) => Seq::Source(t.len()),
            Seq::Concat(parts) => Seq::Concat(parts.iter().map(Seq::lens).collect()),
            Seq::Mix(parts) => Seq::Mix(parts.iter().map(|(p, s)| (p.lens(), *s)).collect()),
            Seq::Shuffle { seed, inner } => Seq::Shuffle { seed: *seed, inner: Box::new(inner.lens()) },
            Seq::Repeat { times, inner } => Seq::Repeat { times: *times, inner: Box::new(inner.lens()) },
            Seq::Skip { n, inner } => Seq::Skip { n: *n, inner: Box::new(inner.lens()) },
            Seq::Take { n, inner } => Seq::Take { n: *n, inner: Box::new(inner.lens()) },
            Seq::Stride { step, offset, inner } => Seq::Stride { step: *step, offset: *offset, inner: Box::new(inner.lens()) },
        }
    }
}
