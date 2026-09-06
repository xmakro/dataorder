//! Validation, compilation and random access.
//!
//! The compiler separates source handles from the configuration, then builds a tree
//! with lengths, concat offsets, interleave profiles and shuffle shapes. [`get`]
//! follows that tree to resolve a position without keeping iteration state.

use crate::bounds::{BoundsError, resolve};
use crate::cursor::Cursor;
use crate::interleave::{Interleave, MAX_TOTAL_LEN, Sampling};
use crate::perm::{self, Shape};
use crate::preparation::{CompilationReport, Preparation, PreparedMix, PreparedSource, SamplingDiagnostics, WeightedAllocation};
use crate::seq::{MixPart, WeightedPart};
use crate::{Error, ErrorKind, MAX_DEPTH, Seq, Source};
use std::fmt;
use std::ops::RangeBounds;

/// A compiled node. Empty subtrees are folded to [`Node::Empty`], so every child of a
/// `Concat`, `Mix` and every child of a transform has elements. Source indices still
/// refer to the original configuration, including sources whose nodes folded away.
#[derive(Clone, Debug)]
pub(crate) enum Node {
    Empty,
    /// Elements `offset..offset + len` of source `src` (an index into `Order::sources`).
    Source {
        src: u32,
        offset: u64,
        len: u64,
    },
    /// `offsets[i]` is the position of child `i`'s first element; `offsets[k]` the length.
    Concat {
        offsets: Vec<u64>,
        children: Vec<Self>,
    },
    Mix {
        il: Interleave,
        children: Vec<Self>,
    },
    /// `salt` folds the salts and lengths of the sources under it that have elements (see
    /// [`perm::shuffle_salt`] and [`sources`]).
    Shuffle {
        seed: u64,
        salt: u64,
        shape: Shape,
        child: Box<Self>,
    },
    /// `child` repeated: positions `0..len`, `child_len` per repetition, the last one cut
    /// short when `len` is not a multiple (a cycle). `depth` counts the repeats above this
    /// one; it salts the epoch contexts.
    Repeat {
        child_len: u64,
        len: u64,
        depth: u32,
        child: Box<Self>,
    },
    Slice {
        start: u64,
        len: u64,
        child: Box<Self>,
    },
    Stride {
        step: u64,
        offset: u64,
        len: u64,
        child: Box<Self>,
    },
}

impl Node {
    pub(crate) fn len(&self) -> u64 {
        match self {
            Self::Empty => 0,
            Self::Source { len, .. } | Self::Repeat { len, .. } | Self::Slice { len, .. } | Self::Stride { len, .. } => *len,
            Self::Concat { offsets, .. } => *offsets.last().unwrap(),
            Self::Mix { il, .. } => il.len(),
            Self::Shuffle { shape, .. } => shape.n,
        }
    }
}

/// A validated sequence with random access and seekable iteration.
///
/// Build one from a [`Seq`] with [`Order::new`]. It owns the source handles and
/// returns `(&source, index_within_source)` pairs through [`get`](Order::get) and
/// [`iter`](Order::iter). It stores the compiled structure, not the output elements.
///
/// Lengths and positions are `usize` in the API and `u64` internally. On a 32-bit
/// target, intermediate nodes may exceed `usize::MAX`, but the final order must fit.
/// `Debug` displays the length, seed and sources.
#[derive(Clone)]
pub struct Order<T> {
    pub(crate) root: Node,
    /// Context of the root (the order's seed); repetitions derive their own from it.
    pub(crate) ctx: u64,
    pub(crate) sources: Vec<T>,
}

impl<T: Source> Order<T> {
    /// Validates and compiles `seq` with an order seed of 0.
    /// Consumes the configuration and takes ownership of its sources. Clone the
    /// configuration to keep a copy, or use [`Seq::check`] to validate it by reference.
    ///
    /// ```
    /// use dataorder::{ErrorKind, Order, Seq};
    /// let order = Order::new(Seq::concat([Seq::source(3), Seq::source(2).shuffle(1)]))?;
    /// assert_eq!(order.len(), 5);
    /// let err = Order::new(Seq::source(3).skip(4)).unwrap_err();
    /// assert_eq!(err.kind(), &ErrorKind::SkipOutOfRange { n: 4, len: 3 });
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    ///
    /// # Errors
    /// Skips and takes past the end, a zero stride, lengths that overflow, nesting deeper
    /// than [`MAX_DEPTH`], and schedules or weights the mix rejects; see [`ErrorKind`]. The
    /// error names the node it was found at.
    pub fn new(seq: Seq<T>) -> Result<Self, Error> {
        Self::with_seed(seq, 0)
    }

    /// Validates and compiles `seq` with the given order seed.
    /// This seed is combined with each shuffle's own seed; it does not add shuffling
    /// to unshuffled sequences. Use [`Order::set_seed`] to reseed an existing order
    /// without rebuilding it.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let seq = Seq::source(100).shuffle(1);
    /// let (a, b) = (Order::with_seed(seq.clone(), 1)?, Order::with_seed(seq, 2)?);
    /// assert!(a.iter(..).ne(b.iter(..)));
    /// assert_eq!(b.seed(), 2);
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    ///
    /// # Errors
    /// As for [`Order::new`].
    pub fn with_seed(seq: Seq<T>, seed: u64) -> Result<Self, Error> {
        Self::build(seq, seed, None)
    }

    /// Compiles with `seed` and returns quotas, compiled parameters, original source
    /// metadata and schedule capacity diagnostics.
    /// The report is separate from the order and can be dropped after inspection.
    /// No records are enumerated. Unlike [`new`](Self::new), this collects diagnostic
    /// paths and metadata copies during compilation.
    ///
    /// ```
    /// use dataorder::{Order, PreparedKind, Seq};
    /// let seq = Seq::weighted(1_000_000_000, [(Seq::source(10), 3.0), (Seq::source(20), 1.0)]);
    /// let (order, report) = Order::prepare(seq, 42)?;
    /// assert_eq!(report.weighted[0].counts, [750_000_000, 250_000_000]);
    /// assert_eq!(report.nodes[0].kind, PreparedKind::Mix);
    /// assert_eq!(report.nodes[0].len, order.len() as u64);
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    ///
    /// # Errors
    /// As for [`Order::new`].
    pub fn prepare(seq: Seq<T>, seed: u64) -> Result<(Self, Preparation), Error> {
        let mut report = CompilationReport::default();
        let order = Self::build(seq, seed, Some(&mut report))?;
        let report = Preparation::new(&order.root, report);
        Ok((order, report))
    }

    fn build(seq: Seq<T>, seed: u64, report: Option<&mut CompilationReport>) -> Result<Self, Error> {
        let (seq, sources) = separate_sources(seq);
        let mut c = Compiler { sources, salts: Vec::new(), path: Vec::new(), report };
        let root = c.compile(seq, 0, 1)?;
        if usize::try_from(root.len()).is_err() {
            return Err(Error::new(ErrorKind::OrderTooLong { len: root.len() }, Vec::new()));
        }
        Ok(Self { root, ctx: seed, sources: c.sources })
    }
}

impl<T> Order<T> {
    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.root.len() as usize
    }

    /// `true` when there are no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The seed given to [`Order::with_seed`] or [`Order::set_seed`] (0 for [`Order::new`]).
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.ctx
    }

    /// Changes the seed used by all shuffles, without rebuilding the order.
    /// Takes constant time: shuffle keys are derived during access and iteration.
    /// Previously cloned orders keep their own seeds.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let seq = Seq::source(100).shuffle(1);
    /// let mut order = Order::new(seq.clone())?;
    /// order.set_seed(7);
    /// let reseeded = Order::with_seed(seq, 7)?;
    /// assert!(order.iter(..).eq(reseeded.iter(..)));
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    pub fn set_seed(&mut self, seed: u64) {
        self.ctx = seed;
    }

    /// Source handles in order of appearance in the original configuration.
    /// Includes sources whose nodes were removed during compilation.
    #[must_use]
    pub fn sources(&self) -> &[T] {
        &self.sources
    }

    /// Mutable access to source handles, for example to open a dataset in place.
    /// The order keeps the lengths and salts recorded at construction. Keep source
    /// lengths stable so that the stored order and returned indices remain valid.
    #[must_use]
    pub fn sources_mut(&mut self) -> &mut [T] {
        &mut self.sources
    }

    /// Consumes the order and returns its source handles in configuration order.
    #[must_use]
    pub fn into_sources(self) -> Vec<T> {
        self.sources
    }

    /// Returns a source's index in [`sources`](Order::sources).
    /// Uses the reference's location, so it takes constant time and distinguishes
    /// sources even when their values compare equal. For zero-sized source types,
    /// references cannot be distinguished and this always returns 0. Use
    /// [`get_indexed`](Self::get_indexed) or [`Cursor::indexed`] for explicit ordinals.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::mix([Seq::source(3), Seq::source(3)]))?;
    /// let parts: Vec<usize> = order.iter(..).map(|(s, _)| order.source_index(s)).collect();
    /// assert_eq!(parts, [0, 1, 0, 1, 0, 1]);
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    ///
    /// # Panics
    /// If `T` is not zero-sized and `source` is not a reference into this order's sources.
    #[must_use]
    pub fn source_index(&self, source: &T) -> usize {
        let size = std::mem::size_of::<T>();
        if size == 0 {
            return 0;
        }
        let (base, at) = (self.sources.as_ptr() as usize, std::ptr::from_ref(source) as usize);
        assert!(
            at >= base && at < base + size * self.sources.len() && (at - base).is_multiple_of(size),
            "dataorder: the source is not one of this order's"
        );
        (at - base) / size
    }

    /// Returns the source and source index at order position `pos`.
    ///
    /// Walks the path to a source. A concat searches its offsets in `O(log k)`; a mix
    /// seeks the interleave and allocates (see the crate's [cost model](crate#cost)), and
    /// a shuffle cycle-walks a permutation at constant average cost per position.
    /// Use [`Order::iter`] for consecutive positions. For many scattered positions,
    /// reuse a cursor with [`Cursor::seek`] to reuse its allocations.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::concat([Seq::source(3), Seq::source(5).shuffle(1)]))?;
    /// let (source, index) = order.get(2);
    /// assert_eq!((*source, index), (3, 2));
    /// assert_eq!(*order.get(3).0, 5);
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    ///
    /// # Panics
    /// If `pos >= self.len()`. Check [`len`](Order::len) first when the position
    /// might be out of range.
    #[must_use]
    pub fn get(&self, pos: usize) -> (&T, usize) {
        self.try_get(pos).unwrap_or_else(|| panic!("dataorder: position {pos} out of range"))
    }

    /// Returns the element at `pos`, or `None` when `pos >= len()`.
    /// Has the same lookup cost as [`get`](Self::get).
    #[must_use]
    pub fn try_get(&self, pos: usize) -> Option<(&T, usize)> {
        self.try_get_indexed(pos).map(|(_, source, index)| (source, index))
    }

    /// Returns `(source_ordinal, source, index_within_source)` at `pos`.
    /// The ordinal indexes [`sources`](Self::sources), including for zero-sized types.
    ///
    /// # Panics
    /// If `pos >= len()`; use [`try_get_indexed`](Self::try_get_indexed) for checked access.
    #[must_use]
    pub fn get_indexed(&self, pos: usize) -> (usize, &T, usize) {
        self.try_get_indexed(pos).unwrap_or_else(|| panic!("dataorder: position {pos} out of range"))
    }

    /// Checked [`get_indexed`](Self::get_indexed); returns `None` outside the order.
    #[must_use]
    pub fn try_get_indexed(&self, pos: usize) -> Option<(usize, &T, usize)> {
        if pos >= self.len() {
            return None;
        }
        let (s, i) = get(&self.root, pos as u64, self.ctx);
        Some((s as usize, &self.sources[s as usize], i as usize))
    }

    /// Returns a cursor over the positions in `range`.
    /// `iter(a..b)` yields `get(p)` for each `p` in `a..b`; the end is exclusive.
    /// `iter(..)` visits the whole order, as does `for item in &order`.
    ///
    /// Cursor state is allocated on the first draw. Each entered mix reserves
    /// space for its parts, then initializes child cursors as it draws from them.
    /// Empty ranges and `count()` allocate nothing. Prefer reusing a cursor with
    /// [`seek`](Cursor::seek) or [`set_range`](Cursor::set_range) when visiting many
    /// ranges, especially over large mixes.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::source(10).shuffle(3))?;
    /// let all: Vec<usize> = order.iter(..).map(|(_, i)| i).collect();
    /// assert_eq!(order.iter(4..7).map(|(_, i)| i).collect::<Vec<_>>(), all[4..7]);
    /// assert_eq!(order.iter(8..).count(), 2);
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    ///
    /// # Panics
    /// If the range ends after `len()`, ends before it starts, or has a bound at
    /// `usize::MAX` where one more would be needed.
    pub fn iter(&self, range: impl RangeBounds<usize>) -> Cursor<'_, T> {
        self.try_iter(range).unwrap_or_else(|e| panic!("dataorder: {e}"))
    }

    /// Returns a cursor, reporting invalid bounds without panicking.
    ///
    /// # Errors
    /// A reversed, overflowing or out-of-bounds range; see [`BoundsError`].
    pub fn try_iter(&self, range: impl RangeBounds<usize>) -> Result<Cursor<'_, T>, BoundsError> {
        Ok(Cursor::new(self, resolve(range, self.len())?))
    }
}

impl<T: fmt::Debug> fmt::Debug for Order<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Order").field("len", &self.len()).field("seed", &self.ctx).field("sources", &self.sources).finish()
    }
}

impl<T: Source> TryFrom<Seq<T>> for Order<T> {
    type Error = Error;

    /// [`Order::new`].
    fn try_from(seq: Seq<T>) -> Result<Self, Error> {
        Self::new(seq)
    }
}

impl<'a, T> IntoIterator for &'a Order<T> {
    type Item = (&'a T, usize);
    type IntoIter = Cursor<'a, T>;

    /// The whole order: `iter(..)`.
    fn into_iter(self) -> Cursor<'a, T> {
        self.iter(..)
    }
}

/// The element at `pos` of `node` in context `ctx`, as `(source index, index in it)`.
pub(crate) fn get(node: &Node, pos: u64, ctx: u64) -> (u32, u64) {
    get_with(node, pos, ctx, |il, pos| il.iter(pos..pos + 1).next().expect("dataorder: interleave yielded nothing"))
}

/// Random traversal with a caller-supplied mix seeker, allowing shuffles to retain
/// buffers for every mix they reach, including mixes in different concat children.
pub(crate) fn get_with<'a>(
    mut node: &'a Node,
    mut pos: u64,
    mut ctx: u64,
    mut mix: impl FnMut(&'a Interleave, u64) -> (usize, u64),
) -> (u32, u64) {
    loop {
        match node {
            Node::Empty => unreachable!("dataorder: position in an empty sequence"),
            Node::Source { src, offset, .. } => return (*src, offset + pos),
            Node::Concat { offsets, children } => {
                let i = offsets.partition_point(|&o| o <= pos) - 1;
                pos -= offsets[i];
                node = &children[i];
            }
            Node::Mix { il, children } => {
                let (s, j) = mix(il, pos);
                pos = j;
                node = &children[s];
            }
            Node::Shuffle { seed, salt, shape, child } => {
                pos = perm::permute(*shape, perm::key(*seed, ctx, *salt), pos);
                node = child;
            }
            Node::Repeat { child_len, depth, child, .. } => {
                let epoch = pos / child_len;
                pos -= epoch * child_len;
                ctx = perm::epoch_ctx(ctx, epoch, *depth);
                node = child;
            }
            Node::Slice { start, child, .. } => {
                pos += start;
                node = child;
            }
            Node::Stride { step, offset, child, .. } => {
                pos = offset + pos * step;
                node = child;
            }
        }
    }
}

struct Compiler<'a, T> {
    sources: Vec<T>,
    /// Salt and length of every source, by index, for the salts of the shuffles above them.
    salts: Vec<(u64, u64)>,
    /// Child indices from the root to the node being compiled, for error reports.
    path: Vec<usize>,
    /// Optional report; ordinary builds do not allocate diagnostic paths or quotas.
    report: Option<&'a mut CompilationReport>,
}

/// Move source values out before recursive compilation: a `Seq<T>` contains T inline, so
/// even a short tree of array sources otherwise puts megabytes in recursive frames.
/// This traversal uses a heap stack and stops at the compiler's depth boundary. A dummy
/// leaf there preserves the location of `TooDeep` without reading any source metadata.
fn separate_sources<T>(seq: Seq<T>) -> (Seq<usize>, Vec<T>) {
    enum Work<T> {
        Enter(Seq<T>, u32),
        Finish(Seq<usize>),
    }
    let mut work = vec![Work::Enter(seq, 1)];
    let (mut done, mut sources) = (Vec::new(), Vec::new());
    while let Some(item) = work.pop() {
        match item {
            Work::Finish(mut node) => {
                match &mut node {
                    Seq::Source(_) => unreachable!(),
                    Seq::Concat(parts) => {
                        *parts = done.split_off(done.len() - parts.len());
                    }
                    Seq::Mix(parts) => {
                        for p in parts.iter_mut().rev() {
                            p.seq = done.pop().unwrap();
                        }
                    }
                    Seq::Weighted { parts, .. } => {
                        for p in parts.iter_mut().rev() {
                            p.seq = done.pop().unwrap();
                        }
                    }
                    Seq::Shuffle { inner, .. }
                    | Seq::Repeat { inner, .. }
                    | Seq::Cycle { inner, .. }
                    | Seq::Skip { inner, .. }
                    | Seq::Take { inner, .. }
                    | Seq::Stride { inner, .. } => {
                        **inner = done.pop().unwrap();
                    }
                }
                done.push(node);
            }
            Work::Enter(seq, level) => {
                if level > MAX_DEPTH {
                    seq.dismantle();
                    done.push(Seq::Source(0));
                    continue;
                }
                let dummy = || Box::new(Seq::Source(0));
                let (node, children) = match seq {
                    Seq::Source(source) => {
                        done.push(Seq::Source(sources.len()));
                        sources.push(source);
                        continue;
                    }
                    Seq::Concat(parts) => (Seq::Concat(vec![Seq::Source(0); parts.len()]), parts),
                    Seq::Mix(parts) => {
                        let shape = parts.iter().map(|p| MixPart { seq: Seq::Source(0), sampling: p.sampling }).collect();
                        (Seq::Mix(shape), parts.into_iter().map(|p| p.seq).collect())
                    }
                    Seq::Weighted { total, parts } => {
                        let shape =
                            parts.iter().map(|p| WeightedPart { seq: Seq::Source(0), weight: p.weight, sampling: p.sampling }).collect();
                        (Seq::Weighted { total, parts: shape }, parts.into_iter().map(|p| p.seq).collect())
                    }
                    Seq::Shuffle { seed, inner } => (Seq::Shuffle { seed, inner: dummy() }, vec![*inner]),
                    Seq::Repeat { times, inner } => (Seq::Repeat { times, inner: dummy() }, vec![*inner]),
                    Seq::Cycle { len, inner } => (Seq::Cycle { len, inner: dummy() }, vec![*inner]),
                    Seq::Skip { n, inner } => (Seq::Skip { n, inner: dummy() }, vec![*inner]),
                    Seq::Take { n, inner } => (Seq::Take { n, inner: dummy() }, vec![*inner]),
                    Seq::Stride { step, offset, inner } => (Seq::Stride { step, offset, inner: dummy() }, vec![*inner]),
                };
                work.push(Work::Finish(node));
                work.extend(children.into_iter().rev().map(|child| Work::Enter(child, level + 1)));
            }
        }
    }
    (done.pop().unwrap(), sources)
}

impl<T: Source> Compiler<'_, T> {
    /// An error at the node being compiled, or at its child `part`.
    fn err_at(&self, kind: ErrorKind, part: Option<usize>) -> Error {
        let mut path = self.path.clone();
        path.extend(part);
        Error::new(kind, path)
    }

    fn err(&self, kind: ErrorKind) -> Error {
        self.err_at(kind, None)
    }

    /// Compiles child `i` of the node being compiled, `repeats` repeats deep, one nesting
    /// level below it.
    fn child(&mut self, i: usize, seq: Seq<usize>, repeats: u32, level: u32) -> Result<Node, Error> {
        self.path.push(i);
        let node = self.compile(seq, repeats, level + 1);
        self.path.pop();
        node
    }

    /// Compiles the children of a wide node in order. On an error, the parts not yet
    /// compiled are taken apart without recursion, so that a rejected configuration of any
    /// depth is dropped on the heap rather than the stack.
    fn children(&mut self, parts: impl IntoIterator<Item = Seq<usize>>, repeats: u32, level: u32) -> Result<Vec<Node>, Error> {
        let mut parts = parts.into_iter();
        let mut children = Vec::with_capacity(parts.size_hint().0);
        while let Some(part) = parts.next() {
            match self.child(children.len(), part, repeats, level) {
                Ok(node) => children.push(node),
                Err(e) => {
                    parts.for_each(Seq::dismantle);
                    return Err(e);
                }
            }
        }
        Ok(children)
    }

    /// `repeats` is the number of repeats above `seq`, `level` its nesting level (the root
    /// being 1). A `seq` that is rejected before it is consumed is dismantled without
    /// recursion (see [`Seq::dismantle`]). One method per variant keeps this frame, one per
    /// level of the recursion, small. Sources have already been replaced by indices, so
    /// recursive frame sizes are independent of T.
    fn compile(&mut self, seq: Seq<usize>, repeats: u32, level: u32) -> Result<Node, Error> {
        if level > MAX_DEPTH {
            seq.dismantle();
            return Err(self.err(ErrorKind::TooDeep));
        }
        match seq {
            Seq::Source(source) => self.source(source),
            Seq::Concat(parts) => self.concat(parts, repeats, level),
            Seq::Mix(parts) => self.mix_parts(parts, repeats, level),
            Seq::Weighted { total, parts } => self.weighted(total, parts, repeats, level),
            Seq::Shuffle { seed, inner } => self.shuffle(seed, *inner, repeats, level),
            Seq::Repeat { times, inner } => self.repeat(times, *inner, repeats, level),
            Seq::Cycle { len, inner } => self.cycled(len, *inner, repeats, level),
            Seq::Skip { n, inner } => self.skip(n, *inner, repeats, level),
            Seq::Take { n, inner } => self.take(n, *inner, repeats, level),
            Seq::Stride { step, offset, inner } => self.strided(step, offset, *inner, repeats, level),
        }
    }

    fn source(&mut self, index: usize) -> Result<Node, Error> {
        let source = &self.sources[index];
        let len = source.len() as u64;
        let src = u32::try_from(index).map_err(|_| self.err(ErrorKind::TooManySources))?;
        let salt = source.salt();
        self.salts.push((salt, len));
        if let Some(report) = &mut self.report {
            debug_assert_eq!(report.sources.len(), index);
            report.sources.push(PreparedSource { path: self.path.clone(), len, salt });
        }
        Ok(if len == 0 { Node::Empty } else { Node::Source { src, offset: 0, len } })
    }

    /// Nested concatenations are flattened and empty parts dropped: both keep the order and
    /// the context of every element.
    fn concat(&mut self, parts: Vec<Seq<usize>>, repeats: u32, level: u32) -> Result<Node, Error> {
        let mut children = Vec::new();
        for node in self.children(parts, repeats, level)? {
            match node {
                Node::Empty => {}
                Node::Concat { children: inner, .. } => children.extend(inner),
                node => children.push(node),
            }
        }
        Ok(match children.len() {
            0 => Node::Empty,
            1 => children.pop().unwrap(),
            _ => {
                let offsets = offsets(&children).ok_or_else(|| self.err(ErrorKind::LengthOverflow))?;
                Node::Concat { offsets, children }
            }
        })
    }

    fn mix_parts(&mut self, parts: Vec<MixPart<usize>>, repeats: u32, level: u32) -> Result<Node, Error> {
        let sampling: Vec<Sampling> = parts.iter().map(|p| p.sampling).collect();
        let children = self.children(parts.into_iter().map(|p| p.seq), repeats, level)?;
        self.mix(children, &sampling)
    }

    fn shuffle(&mut self, seed: u64, inner: Seq<usize>, repeats: u32, level: u32) -> Result<Node, Error> {
        let child = self.child(0, inner, repeats, level)?;
        if child.len() <= 1 {
            return Ok(child);
        }
        let mut under = Vec::new();
        sources(&child, &mut under);
        let salt = perm::shuffle_salt(under.into_iter().map(|src| self.salts[src as usize]));
        Ok(Node::Shuffle { seed, salt, shape: Shape::new(child.len()), child: Box::new(child) })
    }

    /// A single repetition is the sequence itself, so it does not count as a repeat above
    /// its child either.
    fn repeat(&mut self, times: usize, inner: Seq<usize>, repeats: u32, level: u32) -> Result<Node, Error> {
        let child = self.child(0, inner, if times > 1 { repeats + 1 } else { repeats }, level)?;
        let child_len = child.len();
        let len = (times as u64).checked_mul(child_len).ok_or_else(|| self.err(ErrorKind::LengthOverflow))?;
        Ok(if len == 0 {
            Node::Empty
        } else if times == 1 {
            child
        } else {
            Node::Repeat { child_len, len, depth: repeats, child: Box::new(child) }
        })
    }

    fn cycled(&mut self, len: usize, inner: Seq<usize>, repeats: u32, level: u32) -> Result<Node, Error> {
        let child = self.child(0, inner, repeats, level)?;
        if len > 0 && child.len() == 0 {
            return Err(self.err(ErrorKind::EmptyCycle));
        }
        Ok(cycle(child, len as u64, repeats))
    }

    fn skip(&mut self, n: usize, inner: Seq<usize>, repeats: u32, level: u32) -> Result<Node, Error> {
        let child = self.child(0, inner, repeats, level)?;
        let len = child.len();
        if n as u64 > len {
            return Err(self.err(ErrorKind::SkipOutOfRange { n, len }));
        }
        Ok(slice(child, n as u64, len - n as u64))
    }

    fn take(&mut self, n: usize, inner: Seq<usize>, repeats: u32, level: u32) -> Result<Node, Error> {
        let child = self.child(0, inner, repeats, level)?;
        let len = child.len();
        if n as u64 > len {
            return Err(self.err(ErrorKind::TakeOutOfRange { n, len }));
        }
        Ok(slice(child, 0, n as u64))
    }

    fn strided(&mut self, step: usize, offset: usize, inner: Seq<usize>, repeats: u32, level: u32) -> Result<Node, Error> {
        if step == 0 {
            inner.dismantle();
            return Err(self.err(ErrorKind::ZeroStep));
        }
        let child = self.child(0, inner, repeats, level)?;
        let (step, offset) = (step as u64, offset as u64);
        let n = child.len();
        let len = if offset >= n { 0 } else { (n - offset - 1) / step + 1 };
        Ok(stride(child, step, offset, len))
    }

    /// The mix of compiled `children` with their schedules; validates the schedules even
    /// when the mix folds away.
    fn mix(&mut self, mut children: Vec<Node>, sampling: &[Sampling]) -> Result<Node, Error> {
        // The tournament tree indexes parts with u32 and needs a spare bit.
        if children.len() >= u32::MAX as usize / 2 {
            return Err(self.err(ErrorKind::TooManyMixParts));
        }
        let lens: Vec<u64> = children.iter().map(Node::len).collect();
        let mut diagnostics = SamplingDiagnostics::default();
        let result = if self.report.is_some() {
            Interleave::with_diagnostics(&lens, sampling, Some(&mut diagnostics))
        } else {
            Interleave::with_sampling(&lens, sampling)
        };
        let mut il = result.map_err(|e| {
            let detail = e.detail();
            let (kind, part) = e.into_kind();
            self.err_at(kind, part).with_sampling_detail(detail)
        })?;
        if let Some(report) = &mut self.report {
            report.mixes.push(PreparedMix { path: self.path.clone(), counts: lens, sampling: sampling.to_vec(), diagnostics });
        }
        children.retain(|c| c.len() > 0);
        children.shrink_to_fit();
        il.remove_empty();
        Ok(match children.len() {
            0 => Node::Empty,
            // A single sequence is interleaved with nothing: it stays in order.
            1 => children.pop().unwrap(),
            _ => Node::Mix { il, children },
        })
    }

    /// A weighted mix: every part repeated as often as its share needs and cut to it, then
    /// mixed. A part is compiled once; when it turns out to need repeating, the repeats
    /// inside it move one level deeper after the fact.
    fn weighted(&mut self, total: usize, parts: Vec<WeightedPart<usize>>, repeats: u32, level: u32) -> Result<Node, Error> {
        let weights: Vec<f64> = parts.iter().map(|p| p.weight).collect();
        let sampling: Vec<Sampling> = parts.iter().map(|p| p.sampling).collect();
        let shares = match weighted_shares(total as u64, &weights) {
            Ok(shares) => shares,
            Err((kind, part)) => {
                parts.into_iter().for_each(|p| p.seq.dismantle());
                return Err(self.err_at(kind, part));
            }
        };
        if let Some(report) = &mut self.report {
            report.weighted.push(WeightedAllocation { path: self.path.clone(), counts: shares.clone() });
        }
        let mut children = Vec::with_capacity(parts.len());
        let mut parts = parts.into_iter().zip(shares);
        while let Some((part, share)) = parts.next() {
            let i = children.len();
            match self.weighted_part(i, part.seq, share, repeats, level) {
                Ok(node) => children.push(node),
                Err(e) => {
                    parts.for_each(|(p, _)| p.seq.dismantle());
                    return Err(e);
                }
            }
        }
        self.mix(children, &sampling)
    }

    /// Part `i` of a weighted mix, cycled to its `share`.
    fn weighted_part(&mut self, i: usize, seq: Seq<usize>, share: u64, repeats: u32, level: u32) -> Result<Node, Error> {
        let child = self.child(i, seq, repeats, level)?;
        if share > 0 && child.len() == 0 {
            return Err(self.err_at(ErrorKind::EmptyWeightedPart, Some(i)));
        }
        Ok(cycle(child, share, repeats))
    }
}

/// `child` repeated as often as `len` positions need and cut there (a [`Seq::Cycle`], or a
/// part of a weighted mix): within one repetition it is a slice; beyond, a repeat whose
/// last repetition is cut short. The child was compiled `repeats` deep, once, before it
/// was known whether it repeats; when it does, the repeats inside it move one level
/// deeper after the fact, as compiling it under a repeat would have put them. `child` has
/// elements whenever `len > 0`.
fn cycle(mut child: Node, len: u64, repeats: u32) -> Node {
    let child_len = child.len();
    if len <= child_len {
        return slice(child, 0, len);
    }
    deepen(&mut child);
    Node::Repeat { child_len, len, depth: repeats, child: Box::new(child) }
}

/// Exact largest-remainder shares, with the lowest index first on equal remainders.
pub(crate) fn weighted_shares(total: u64, weights: &[f64]) -> Result<Vec<u64>, (ErrorKind, Option<usize>)> {
    for (i, &w) in weights.iter().enumerate() {
        if !(w.is_finite() && w >= 0.0) {
            return Err((ErrorKind::InvalidWeight { weight: w }, Some(i)));
        }
    }
    if total > MAX_TOTAL_LEN {
        return Err((ErrorKind::MixTooLong, None));
    }
    if total == 0 {
        return Ok(vec![0; weights.len()]);
    }
    if !weights.iter().any(|&w| w > 0.0) {
        return Err((ErrorKind::ZeroWeights, None));
    }
    Ok(crate::weight::shares(total, weights))
}

/// The sources of `node` that can contribute elements, in order of appearance: every
/// `Source` of the compiled tree, since a subtree without elements folds to `Empty`, a
/// slice of a concatenation keeps only the parts it touches (see [`slice()`]), and a source
/// with elements stays wherever any other node keeps it (a stride over a concatenation, or a
/// slice of a mix, may still never reach some of them). What a shuffle above is salted with.
fn sources(node: &Node, out: &mut Vec<u32>) {
    match node {
        Node::Empty => {}
        Node::Source { src, .. } => out.push(*src),
        Node::Concat { children, .. } | Node::Mix { children, .. } => children.iter().for_each(|c| sources(c, out)),
        Node::Shuffle { child, .. } | Node::Repeat { child, .. } | Node::Slice { child, .. } | Node::Stride { child, .. } => {
            sources(child, out)
        }
    }
}

/// Moves every repeat in `node` one repeat deeper: what compiling it under one more repeat
/// would have produced.
fn deepen(node: &mut Node) {
    match node {
        Node::Empty | Node::Source { .. } => {}
        Node::Concat { children, .. } | Node::Mix { children, .. } => children.iter_mut().for_each(deepen),
        Node::Repeat { depth, child, .. } => {
            *depth += 1;
            deepen(child);
        }
        Node::Shuffle { child, .. } | Node::Slice { child, .. } | Node::Stride { child, .. } => deepen(child),
    }
}

/// `offsets[i]` is the position of `children[i]`'s first element and `offsets[k]` the total
/// length; `None` when that does not fit in 64 bits.
fn offsets(children: &[Node]) -> Option<Vec<u64>> {
    let mut offsets = Vec::with_capacity(children.len() + 1);
    let mut total = 0u64;
    for child in children {
        offsets.push(total);
        total = total.checked_add(child.len())?;
    }
    offsets.push(total);
    Some(offsets)
}

/// Positions `start..start + len` of `child`, folded into the child where that is exact: an
/// offset into a source, a slice or a stride, a prefix of a repeat is the repeat cut short
/// (or its first repetition alone, which keeps the context), and a concatenation is narrowed
/// to the parts the slice touches, so that a shuffle above is salted only with sources it
/// can draw from and the cursor searches fewer parts.
fn slice(child: Node, start: u64, len: u64) -> Node {
    if len == 0 {
        return Node::Empty;
    }
    if start == 0 && len == child.len() {
        return child;
    }
    match child {
        Node::Source { src, offset, .. } => Node::Source { src, offset: offset + start, len },
        Node::Repeat { child_len, depth, child, .. } if start == 0 => {
            if len <= child_len {
                slice(*child, 0, len)
            } else {
                Node::Repeat { child_len, len, depth, child }
            }
        }
        Node::Slice { start: inner, child, .. } => slice(*child, inner + start, len),
        Node::Stride { step, offset, child, .. } => {
            let offset = offset + start * step;
            if len == 1 { slice(*child, offset, 1) } else { Node::Stride { step, offset, len, child } }
        }
        Node::Concat { offsets: at, mut children } => {
            // The parts holding the first and the last position.
            let first = at.partition_point(|&o| o <= start) - 1;
            let last = at.partition_point(|&o| o < start + len) - 1;
            let last_len = start + len - at[last];
            let start = start - at[first];
            if first == last {
                return slice(children.swap_remove(first), start, len);
            }
            children.truncate(last + 1);
            children.drain(..first);
            // Boundary children may themselves be sliced concatenations. Trim them too,
            // so their unreachable inner sources cannot salt a shuffle above this slice.
            let end = children.len() - 1;
            let tail = std::mem::replace(&mut children[end], Node::Empty);
            children[end] = slice(tail, 0, last_len);
            let head = std::mem::replace(&mut children[0], Node::Empty);
            let head_len = head.len() - start;
            children[0] = slice(head, start, head_len);
            Node::Concat { offsets: offsets(&children).expect("dataorder: a slice of a concat is shorter than it"), children }
        }
        child => Node::Slice { start, len, child: Box::new(child) },
    }
}

/// Positions `offset, offset + step, …` of `child`, `len` of them (as many as exist, which
/// the caller has counted), folded where that is exact: one position or a step of one is a
/// slice, and a stride over a slice or over another stride is one stride (the products
/// cannot overflow, since with `len ≥ 2` they are bounded by the child's length).
fn stride(child: Node, step: u64, offset: u64, len: u64) -> Node {
    if len == 0 {
        return Node::Empty;
    }
    if step == 1 || len == 1 {
        return slice(child, offset, len);
    }
    match child {
        Node::Slice { start, child, .. } => Node::Stride { step, offset: start + offset, len, child },
        Node::Stride { step: inner, offset: base, child, .. } => {
            Node::Stride { step: step * inner, offset: base + offset * inner, len, child }
        }
        child => Node::Stride { step, offset, len, child: Box::new(child) },
    }
}
