//! The configuration: a tree of sequence expressions over sources, built by hand or with
//! the builder methods on [`Seq`].

use crate::Sampling;
use std::ops::{Bound, RangeBounds};

/// A sequence expression over sources of type `T` (anything that is a
/// [`Dataset`](crate::Dataset)). Leaves are [`Source`](Seq::Source)s; every other variant
/// transforms or combines sequences. Compile it with
/// [`Order::compile`](crate::Order::compile). It is plain data: clone it, compare it,
/// serialize it, or [`map`](Seq::map) its sources to another type.
#[derive(Clone, Debug, PartialEq)]
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
    Shuffle { seed: u64, inner: Box<Seq<T>> },
    /// `inner`, `times` times over. Each repetition reshuffles every shuffle inside it
    /// (including the first: `x.repeat(n)` starts differently from `x`).
    Repeat { times: usize, inner: Box<Seq<T>> },
    /// Positions `start..end` of `inner`; `end = None` means up to the end.
    Slice { start: usize, end: Option<usize>, inner: Box<Seq<T>> },
    /// Positions `offset, offset + step, offset + 2·step, …` of `inner`: shard `offset` of
    /// `step` shards.
    Stride { step: usize, offset: usize, inner: Box<Seq<T>> },
}

impl<T> Seq<T> {
    /// The elements of `source`, in order.
    pub fn source(source: T) -> Seq<T> {
        Seq::Source(source)
    }

    /// The parts one after another.
    pub fn concat(parts: impl IntoIterator<Item = Seq<T>>) -> Seq<T> {
        Seq::Concat(parts.into_iter().collect())
    }

    /// The parts interleaved, all [`Sampling::Uniform`].
    pub fn mix(parts: impl IntoIterator<Item = Seq<T>>) -> Seq<T> {
        Seq::Mix(parts.into_iter().map(|p| (p, Sampling::Uniform)).collect())
    }

    /// The parts interleaved, each with its own schedule.
    pub fn mix_with(parts: impl IntoIterator<Item = (Seq<T>, Sampling)>) -> Seq<T> {
        Seq::Mix(parts.into_iter().collect())
    }

    /// This sequence in the pseudorandom order selected by `seed`.
    pub fn shuffle(self, seed: u64) -> Seq<T> {
        Seq::Shuffle { seed, inner: Box::new(self) }
    }

    /// This sequence `times` times over, reshuffled each time.
    pub fn repeat(self, times: usize) -> Seq<T> {
        Seq::Repeat { times, inner: Box::new(self) }
    }

    /// The positions in `range` of this sequence.
    pub fn slice(self, range: impl RangeBounds<usize>) -> Seq<T> {
        let start = match range.start_bound() {
            Bound::Included(&s) => s,
            Bound::Excluded(&s) => s + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&e) => Some(e + 1),
            Bound::Excluded(&e) => Some(e),
            Bound::Unbounded => None,
        };
        Seq::Slice { start, end, inner: Box::new(self) }
    }

    /// The first `n` positions.
    pub fn take(self, n: usize) -> Seq<T> {
        self.slice(..n)
    }

    /// Everything after the first `n` positions.
    pub fn skip(self, n: usize) -> Seq<T> {
        self.slice(n..)
    }

    /// Every `step`-th position starting at `offset`.
    pub fn stride(self, step: usize, offset: usize) -> Seq<T> {
        Seq::Stride { step, offset, inner: Box::new(self) }
    }

    /// Shard `index` of `count`: positions `index, index + count, …`. All shards of one
    /// sequence together cover it exactly once, and shard `i` holds position `i` of every
    /// consecutive block of `count` positions.
    pub fn shard(self, index: usize, count: usize) -> Seq<T> {
        self.stride(count, index)
    }

    /// The same expression over the sources mapped by `f`, in order of appearance: a
    /// configuration over handles becomes one over loaded datasets.
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
            Seq::Slice { start, end, inner } => Seq::Slice { start, end, inner: Box::new(inner.map_with(f)) },
            Seq::Stride { step, offset, inner } => Seq::Stride { step, offset, inner: Box::new(inner.map_with(f)) },
        }
    }
}
