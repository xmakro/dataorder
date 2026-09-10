//! Validation, compilation and random access.
//!
//! The compiler separates source handles from the configuration, then builds a tree
//! with lengths, concat offsets, interleave profiles and shuffle shapes. [`get`]
//! follows that tree to resolve a position without keeping iteration state.

use crate::bounds::{BoundsError, resolve};
use crate::cursor::Cursor;
use crate::interleave::{Interleave, Sampling};
use crate::perm::{self, Shape};
use crate::seq::MixPart;
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

/// A record selected by [`Order::get`] or [`Cursor`].
///
/// The ordinal identifies the source within this order's [`sources`](Order::sources),
/// even when source values are equal or zero-sized. It is local to the order, not a
/// persistent dataset ID. The item borrows its source and is cheap to copy.
#[derive(Debug, PartialEq, Eq)]
pub struct Item<'a, T> {
    /// Index into [`Order::sources`], including sources removed during compilation.
    pub source_ordinal: usize,
    /// The source handle owned by the order.
    pub source: &'a T,
    /// Index of the record within this source, not its position in the order.
    pub record_index: usize,
}

impl<T> Copy for Item<'_, T> {}

impl<T> Clone for Item<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

/// A validated sequence with random access and seekable iteration.
///
/// Build one from a [`Seq`] with [`Order::new`]. It owns the source handles and
/// returns [`Item`] values through [`get`](Order::get) and
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
    /// configuration to keep a copy. Sources only need to implement [`Source`]
    /// when constructing the order.
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
    /// Invalid shard bounds, skips and takes past the end, a zero step,
    /// lengths that overflow, nesting deeper than [`MAX_DEPTH`], and schedules the
    /// mix rejects; see [`ErrorKind`]. The error names the node it was found at.
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
    /// assert!(a.iter(..)?.ne(b.iter(..)?));
    /// assert_eq!(b.seed(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    /// As for [`Order::new`].
    pub fn with_seed(seq: Seq<T>, seed: u64) -> Result<Self, Error> {
        let mut c = Compiler { sources: Vec::new(), salts: Vec::new(), path: Vec::new() };
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
    /// assert!(order.iter(..)?.eq(reseeded.iter(..)?));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
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

    /// Consumes the order and returns its source handles in configuration order.
    #[must_use]
    pub fn into_sources(self) -> Vec<T> {
        self.sources
    }

    /// Returns the [`Item`] at order position `pos`, or `None` when `pos >= len()`.
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
    /// let item = order.get(2).unwrap();
    /// assert_eq!(item.source_ordinal, 0);
    /// assert_eq!((*item.source, item.record_index), (3, 2));
    /// assert_eq!(*order.get(3).unwrap().source, 5);
    /// assert_eq!(order.get(order.len()), None);
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    #[must_use]
    pub fn get(&self, pos: usize) -> Option<Item<'_, T>> {
        if pos >= self.len() {
            return None;
        }
        let (s, i) = get(&self.root, pos as u64, self.ctx);
        Some(Item { source_ordinal: s as usize, source: &self.sources[s as usize], record_index: i as usize })
    }

    /// Returns a cursor over the positions in `range`.
    /// `iter(a..b)?` yields the element at each `p` in `a..b`; the end is exclusive.
    /// `iter(..)?` visits the whole order, as does `for item in &order`.
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
    /// let all: Vec<usize> = order.iter(..)?.map(|item| item.record_index).collect();
    /// assert_eq!(order.iter(4..7)?.map(|item| item.record_index).collect::<Vec<_>>(), all[4..7]);
    /// assert_eq!(order.iter(8..)?.count(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    /// A reversed, overflowing or out-of-bounds range; see [`BoundsError`].
    pub fn iter(&self, range: impl RangeBounds<usize>) -> Result<Cursor<'_, T>, BoundsError> {
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
    type Item = Item<'a, T>;
    type IntoIter = Cursor<'a, T>;

    /// The whole order, without fallible range validation.
    fn into_iter(self) -> Cursor<'a, T> {
        Cursor::new(self, 0..self.len())
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

struct Compiler<T> {
    sources: Vec<T>,
    /// Salt and length of every source, by index, for the salts of the shuffles above them.
    salts: Vec<(u64, u64)>,
    /// Child indices from the root to the node being compiled, for error reports.
    path: Vec<usize>,
}

impl<T: Source> Compiler<T> {
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
    fn child(&mut self, i: usize, seq: Seq<T>, repeats: u32, level: u32) -> Result<Node, Error> {
        self.path.push(i);
        let node = self.compile(seq, repeats, level + 1);
        self.path.pop();
        node
    }

    /// Compiles the children of a wide node in order.
    fn children(&mut self, parts: impl IntoIterator<Item = Seq<T>>, repeats: u32, level: u32) -> Result<Vec<Node>, Error> {
        parts.into_iter().enumerate().map(|(i, seq)| self.child(i, seq, repeats, level)).collect()
    }

    /// `repeats` is the number of repeats above `seq`; `level` counts all nodes
    /// from the root. One method per variant keeps recursive frames small.
    fn compile(&mut self, seq: Seq<T>, repeats: u32, level: u32) -> Result<Node, Error> {
        if level > MAX_DEPTH {
            return Err(self.err(ErrorKind::TooDeep));
        }
        match seq {
            Seq::Source(source) => self.source(source),
            Seq::Concat(parts) => self.concat(parts, repeats, level),
            Seq::Mix(parts) => self.mix_parts(parts, repeats, level),
            Seq::Shuffle { seed, inner } => self.shuffle(seed, *inner, repeats, level),
            Seq::Repeat { times, inner } => self.repeat(times, *inner, repeats, level),
            Seq::Cycle { len, inner } => self.cycled(len, *inner, repeats, level),
            Seq::Skip { n, inner } => self.skip(n, *inner, repeats, level),
            Seq::Take { n, inner } => self.take(n, *inner, repeats, level),
            Seq::StepBy { step, inner } => self.strided(step, 0, *inner, repeats, level),
            Seq::Shard { count, index, inner } => self.sharded(count, index, *inner, repeats, level),
        }
    }

    fn source(&mut self, source: T) -> Result<Node, Error> {
        let len = source.len() as u64;
        let src = u32::try_from(self.sources.len()).map_err(|_| self.err(ErrorKind::TooManySources))?;
        self.salts.push((source.salt(), len));
        self.sources.push(source);
        Ok(if len == 0 { Node::Empty } else { Node::Source { src, offset: 0, len } })
    }

    /// Nested concatenations are flattened and empty parts dropped: both keep the order and
    /// the context of every element.
    fn concat(&mut self, parts: Vec<Seq<T>>, repeats: u32, level: u32) -> Result<Node, Error> {
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

    fn mix_parts(&mut self, parts: Vec<MixPart<T>>, repeats: u32, level: u32) -> Result<Node, Error> {
        let sampling: Vec<Sampling> = parts.iter().map(|p| p.sampling).collect();
        let children = self.children(parts.into_iter().map(|p| p.seq), repeats, level)?;
        self.mix(children, &sampling)
    }

    fn shuffle(&mut self, seed: u64, inner: Seq<T>, repeats: u32, level: u32) -> Result<Node, Error> {
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
    fn repeat(&mut self, times: usize, inner: Seq<T>, repeats: u32, level: u32) -> Result<Node, Error> {
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

    fn cycled(&mut self, len: usize, inner: Seq<T>, repeats: u32, level: u32) -> Result<Node, Error> {
        let child = self.child(0, inner, repeats, level)?;
        if len > 0 && child.len() == 0 {
            return Err(self.err(ErrorKind::EmptyCycle));
        }
        Ok(cycle(child, len as u64, repeats))
    }

    fn skip(&mut self, n: usize, inner: Seq<T>, repeats: u32, level: u32) -> Result<Node, Error> {
        let child = self.child(0, inner, repeats, level)?;
        let len = child.len();
        if n as u64 > len {
            return Err(self.err(ErrorKind::SkipOutOfRange { n, len }));
        }
        Ok(slice(child, n as u64, len - n as u64))
    }

    fn take(&mut self, n: usize, inner: Seq<T>, repeats: u32, level: u32) -> Result<Node, Error> {
        let child = self.child(0, inner, repeats, level)?;
        let len = child.len();
        if n as u64 > len {
            return Err(self.err(ErrorKind::TakeOutOfRange { n, len }));
        }
        Ok(slice(child, 0, n as u64))
    }

    fn sharded(&mut self, count: usize, index: usize, inner: Seq<T>, repeats: u32, level: u32) -> Result<Node, Error> {
        if index >= count {
            return Err(self.err(ErrorKind::InvalidShard { count, index }));
        }
        self.strided(count, index, inner, repeats, level)
    }

    fn strided(&mut self, step: usize, offset: usize, inner: Seq<T>, repeats: u32, level: u32) -> Result<Node, Error> {
        if step == 0 {
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
        let mut il = Interleave::with_sampling(&lens, sampling).map_err(|e| {
            let detail = e.detail();
            let (kind, part) = e.into_kind();
            self.err_at(kind, part).with_sampling_detail(detail)
        })?;
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
}

/// `child` repeated as often as `len` positions need and cut there (a [`Seq::Cycle`]):
/// within one repetition it is a slice; beyond, a repeat whose
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
