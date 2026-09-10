//! Validation, compilation and random access.
//!
//! The compiler separates source handles from the configuration, then builds a tree
//! with lengths, concat offsets, interleave profiles and shuffle shapes. [`get`]
//! follows that tree to resolve a position without keeping iteration state.

use crate::cursor::{Cursor, resolve_range};
use crate::interleave::{Interleave, Schedule};
use crate::perm::{self, Shape};
use crate::seq::MixPart;
use crate::{BoundsError, Error, ErrorKind, MAX_DEPTH, Seq, Source};
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
        offset: usize,
        len: usize,
    },
    /// `offsets[i]` is the position of child `i`'s first element; `offsets[k]` the length.
    Concat {
        offsets: Vec<usize>,
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
    /// short when `len` is not a multiple (a cycle). `level` is one more than the maximum
    /// repeat level retained in `child`, or 1 when there are none; it salts epoch contexts.
    Repeat {
        child_len: usize,
        len: usize,
        level: u8,
        child: Box<Self>,
    },
    Slice {
        start: usize,
        len: usize,
        child: Box<Self>,
    },
    Stride {
        step: usize,
        offset: usize,
        len: usize,
        child: Box<Self>,
    },
}

impl Node {
    pub(crate) fn len(&self) -> usize {
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
///
/// Equality compares `source_ordinal` and `record_index` first, then compares source
/// values only when both indices match. Source comparison uses `T`'s equality
/// implementation, so its cost depends on the source type.
#[derive(Debug, PartialEq, Eq)]
pub struct Item<'a, T> {
    /// Index into [`Order::sources`], including sources removed during compilation.
    pub source_ordinal: usize,
    /// Index of the record within this source, not its position in the order.
    pub record_index: usize,
    /// The source handle owned by the order.
    pub source: &'a T,
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
/// returns [`Item`] values through [`get`](Order::get), [`iter`](Order::iter), and
/// [`cursor`](Order::cursor). It stores the compiled structure, not the output elements.
///
/// Lengths and positions use `usize`. Every sequence node must fit in `usize`,
/// even when a parent would truncate it.
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
    /// Skips and takes past the end, a zero step,
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
    /// assert!(a.iter().ne(b.iter()));
    /// assert_eq!(b.seed(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    /// As for [`Order::new`].
    pub fn with_seed(seq: Seq<T>, seed: u64) -> Result<Self, Error> {
        let mut c = Compiler { sources: Vec::new(), salts: Vec::new(), path: Vec::new() };
        let (root, _, _) = c.compile(seq, 1)?;
        Ok(Self { root, ctx: seed, sources: c.sources })
    }
}

impl<T> Order<T> {
    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.root.len()
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
    /// assert!(order.iter().eq(reseeded.iter()));
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
    /// Use [`Order::iter`] or [`Order::cursor`] for consecutive positions. For many scattered positions,
    /// reuse a cursor with [`Cursor::reset`] (`reset(pos..)`) to reuse its allocations.
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
        let (s, i) = get(&self.root, pos, self.ctx);
        Some(Item { source_ordinal: s as usize, source: &self.sources[s as usize], record_index: i })
    }

    /// Returns a cursor over the whole order, starting at position 0.
    /// This is also the iterator used by `for item in &order`.
    /// Use [`Order::cursor`] to select a range.
    ///
    /// Construction positions the cursor immediately and can allocate; see
    /// [`Order::cursor`] for allocation and buffer reuse behavior.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::source(3))?;
    /// assert_eq!(order.iter().map(|item| item.record_index).collect::<Vec<_>>(), [0, 1, 2]);
    /// let mut cursor = order.iter();
    /// cursor.reset(2..)?;
    /// assert_eq!(cursor.next(), order.get(2));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn iter(&self) -> Cursor<'_, T> {
        Cursor::new(self, 0..self.len())
    }

    /// Returns a cursor over the positions in `range`.
    /// `cursor(a..b)?` yields the element at each `p` in `a..b`; the end is exclusive.
    /// `cursor(..)?` visits the whole order, as does [`Order::iter`].
    ///
    /// Construction positions the cursor immediately and can allocate, even for
    /// an empty range. Each entered mix reserves space for its parts, then initializes
    /// child cursors as it draws from them. Prefer reusing a cursor with
    /// [`reset`](Cursor::reset) when visiting many ranges, especially over large mixes.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::source(10).shuffle(3))?;
    /// let all: Vec<usize> = order.iter().map(|item| item.record_index).collect();
    /// assert_eq!(order.cursor(4..7)?.map(|item| item.record_index).collect::<Vec<_>>(), all[4..7]);
    /// assert_eq!(order.cursor(8..)?.count(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    /// A reversed, overflowing or out-of-bounds range; see [`BoundsError`].
    pub fn cursor(&self, range: impl RangeBounds<usize>) -> Result<Cursor<'_, T>, BoundsError> {
        Ok(Cursor::new(self, resolve_range(range, self.len())?))
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

    /// The whole order.
    fn into_iter(self) -> Cursor<'a, T> {
        self.iter()
    }
}

/// The element at `pos` of `node` in context `ctx`, as `(source index, index in it)`.
pub(crate) fn get(node: &Node, pos: usize, ctx: u64) -> (u32, usize) {
    get_with(node, pos, ctx, |il, pos| il.iter(pos..pos + 1).next().expect("dataorder: interleave yielded nothing"))
}

/// Random traversal with a caller-supplied mix seeker, allowing shuffles to retain
/// buffers for every mix they reach, including mixes in different concat children.
pub(crate) fn get_with<'a>(
    mut node: &'a Node,
    mut pos: usize,
    mut ctx: u64,
    mut mix: impl FnMut(&'a Interleave, usize) -> (usize, usize),
) -> (u32, usize) {
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
            Node::Repeat { child_len, level, child, .. } => {
                let epoch = pos / child_len;
                pos -= epoch * child_len;
                ctx = perm::epoch_ctx(ctx, epoch, *level);
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

/// Temporary result of a compiler or folding visit: node, length and maximum retained
/// repeat level. Summaries travel up the recursion and are not stored on every node.
type Compiled = (Node, usize, u8);

// Repetition nesting is bounded by the checked configuration depth.
const _: () = assert!(MAX_DEPTH <= u8::MAX as u32);

struct Compiler<T> {
    sources: Vec<T>,
    /// Salt and length of every source, by index, for the salts of the shuffles above them.
    salts: Vec<(u64, usize)>,
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

    /// Compiles child `i` one configuration level below the node being compiled.
    fn child(&mut self, i: usize, seq: Seq<T>, depth: u32) -> Result<Compiled, Error> {
        self.path.push(i);
        let result = self.compile(seq, depth + 1);
        self.path.pop();
        result
    }

    /// Compile every child before combining their summaries, preserving validation order.
    fn children(&mut self, parts: impl IntoIterator<Item = Seq<T>>, depth: u32) -> Result<Vec<Compiled>, Error> {
        parts.into_iter().enumerate().map(|(i, seq)| self.child(i, seq, depth)).collect()
    }

    /// `depth` counts configuration nodes from the root. Each visit returns its node,
    /// length and inside-out repeat level to the parent. Repetitions use those values directly.
    /// Every node must fit in `usize` before its parent can fold or truncate it.
    fn compile(&mut self, seq: Seq<T>, depth: u32) -> Result<Compiled, Error> {
        if depth > MAX_DEPTH {
            return Err(self.err(ErrorKind::TooDeep));
        }
        match seq {
            Seq::Source(source) => self.source(source),
            Seq::Concat(parts) => self.concat(parts, depth),
            Seq::Mix(parts) => self.mix_parts(parts, depth),
            Seq::Shuffle { seed, inner } => self.shuffle(seed, *inner, depth),
            Seq::Repeat { times, inner } => self.repeat(times, *inner, depth),
            Seq::Cycle { len, inner } => self.cycled(len, *inner, depth),
            Seq::Skip { n, inner } => self.skip(n, *inner, depth),
            Seq::Take { n, inner } => self.take(n, *inner, depth),
            Seq::StepBy { step, inner } => self.stepped(step, *inner, depth),
        }
    }

    fn source(&mut self, source: T) -> Result<Compiled, Error> {
        let len = source.len();
        let src = u32::try_from(self.sources.len()).map_err(|_| self.err(ErrorKind::TooManySources))?;
        self.salts.push((source.salt(), len));
        self.sources.push(source);
        Ok((if len == 0 { Node::Empty } else { Node::Source { src, offset: 0, len } }, len, 0))
    }

    /// Flatten concats using their existing offsets and the summaries returned by visits.
    fn concat(&mut self, parts: Vec<Seq<T>>, depth: u32) -> Result<Compiled, Error> {
        let mut children = Vec::new();
        let mut offsets = Vec::new();
        let mut len = 0usize;
        let mut level = 0;
        for (node, child_len, child_level) in self.children(parts, depth)? {
            let end = len.checked_add(child_len).ok_or_else(|| self.err(ErrorKind::LengthOverflow))?;
            match node {
                Node::Empty => {}
                Node::Concat { offsets: mut inner_offsets, children: inner } => {
                    inner_offsets.pop();
                    offsets.extend(inner_offsets.into_iter().map(|offset| len + offset));
                    children.extend(inner);
                }
                node => {
                    offsets.push(len);
                    children.push(node);
                }
            }
            len = end;
            level = level.max(child_level);
        }
        offsets.push(len);
        let node = match children.len() {
            0 => Node::Empty,
            1 => children.pop().unwrap(),
            _ => Node::Concat { offsets, children },
        };
        Ok((node, len, level))
    }

    fn mix_parts(&mut self, parts: Vec<MixPart<T>>, depth: u32) -> Result<Compiled, Error> {
        let schedule: Vec<Schedule> = parts.iter().map(|p| p.schedule).collect();
        let children = self.children(parts.into_iter().map(|p| p.seq), depth)?;
        self.mix(children, &schedule)
    }

    fn shuffle(&mut self, seed: u64, inner: Seq<T>, depth: u32) -> Result<Compiled, Error> {
        let (child, len, level) = self.child(0, inner, depth)?;
        if len <= 1 {
            return Ok((child, len, level));
        }
        let mut under = Vec::new();
        sources(&child, &mut under);
        let salt = perm::shuffle_salt(under.into_iter().map(|src| self.salts[src as usize]));
        Ok((Node::Shuffle { seed, salt, shape: Shape::new(len), child: Box::new(child) }, len, level))
    }

    /// A single repetition is the sequence itself and introduces no repeat level.
    fn repeat(&mut self, times: usize, inner: Seq<T>, depth: u32) -> Result<Compiled, Error> {
        let (child, child_len, child_level) = self.child(0, inner, depth)?;
        let len = times.checked_mul(child_len).ok_or_else(|| self.err(ErrorKind::LengthOverflow))?;
        Ok(if len == 0 {
            (Node::Empty, 0, 0)
        } else if times == 1 {
            (child, child_len, child_level)
        } else {
            let level = child_level + 1;
            (Node::Repeat { child_len, len, level, child: Box::new(child) }, len, level)
        })
    }

    fn cycled(&mut self, len: usize, inner: Seq<T>, depth: u32) -> Result<Compiled, Error> {
        let child = self.child(0, inner, depth)?;
        if len > 0 && child.1 == 0 {
            return Err(self.err(ErrorKind::EmptyCycle));
        }
        Ok(cycle(child, len))
    }

    fn skip(&mut self, n: usize, inner: Seq<T>, depth: u32) -> Result<Compiled, Error> {
        let (child, len, level) = self.child(0, inner, depth)?;
        if n > len {
            return Err(self.err(ErrorKind::SkipOutOfRange { n, len }));
        }
        Ok(slice(child, n, len - n, Some(level)))
    }

    fn take(&mut self, n: usize, inner: Seq<T>, depth: u32) -> Result<Compiled, Error> {
        let (child, len, level) = self.child(0, inner, depth)?;
        if n > len {
            return Err(self.err(ErrorKind::TakeOutOfRange { n, len }));
        }
        Ok(slice(child, 0, n, Some(level)))
    }

    fn stepped(&mut self, step: usize, inner: Seq<T>, depth: u32) -> Result<Compiled, Error> {
        if step == 0 {
            return Err(self.err(ErrorKind::ZeroStep));
        }
        let child = self.child(0, inner, depth)?;
        Ok(stride(child, step))
    }

    /// Validate schedules even when the mix folds away; empty parts return level zero.
    fn mix(&mut self, parts: Vec<Compiled>, schedule: &[Schedule]) -> Result<Compiled, Error> {
        // The tournament tree indexes parts with u32 and needs a spare bit.
        if parts.len() >= u32::MAX as usize / 2 {
            return Err(self.err(ErrorKind::TooManyMixParts));
        }
        let mut level = 0;
        let lens: Vec<usize> = parts
            .iter()
            .map(|&(_, len, child_level)| {
                level = level.max(child_level);
                len
            })
            .collect();
        let mut il = Interleave::with_schedule(&lens, schedule).map_err(|e| {
            let (kind, part) = e.into_kind();
            self.err_at(kind, part)
        })?;
        let len = il.len();
        let mut children: Vec<Node> = parts.into_iter().filter_map(|(node, len, _)| (len > 0).then_some(node)).collect();
        children.shrink_to_fit();
        il.remove_empty();
        let node = match children.len() {
            0 => Node::Empty,
            // A single sequence is interleaved with nothing: it stays in order.
            1 => children.pop().unwrap(),
            _ => Node::Mix { il, children },
        };
        Ok((node, len, level))
    }
}

/// A cycle uses the child's returned length and level. A short cycle folds as a slice;
/// an extended cycle introduces one repeat level without changing its child's levels.
fn cycle((child, child_len, child_level): Compiled, len: usize) -> Compiled {
    if len <= child_len {
        return slice(child, 0, len, Some(child_level));
    }
    let level = child_level + 1;
    (Node::Repeat { child_len, len, level, child: Box::new(child) }, len, level)
}

/// The retained sources of `node`, in traversal order, for shuffles above it.
/// Empty subtrees and excluded concat parts have already folded away.
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

/// Wraps an opaque node in a slice when needed, returning the supplied level unchanged.
fn sliced(node: Node, start: usize, len: usize, level: u8) -> Compiled {
    let node = if start == 0 && len == node.len() { node } else { Node::Slice { start, len, child: Box::new(node) } };
    (node, len, level)
}

/// Fold a selection and return its length and maximum retained repeat level.
/// `level` is the incoming subtree's summary when it came directly from a compiler visit.
/// Trimming a concat exposes children whose individual summaries have been discarded;
/// the fold visits those retained children and returns their summaries up this recursion.
/// A repeat supplies its stored level without visiting any of its descendants.
fn slice(node: Node, start: usize, len: usize, level: Option<u8>) -> Compiled {
    if len == 0 {
        return (Node::Empty, 0, 0);
    }
    if start == 0
        && len == node.len()
        && let Some(level) = level
    {
        return (node, len, level);
    }
    match node {
        Node::Empty => unreachable!("dataorder: nonempty slice of an empty node"),
        Node::Source { src, offset, .. } => (Node::Source { src, offset: offset + start, len }, len, 0),
        Node::Repeat { child_len, level, child, .. } if start == 0 => {
            if len <= child_len {
                // Removing the outer epoch-zero scope exposes the child's level exactly.
                slice(*child, 0, len, Some(level - 1))
            } else {
                (Node::Repeat { child_len, len, level, child }, len, level)
            }
        }
        node @ Node::Repeat { level, .. } => sliced(node, start, len, level),
        Node::Slice { start: inner, child, .. } => slice(*child, inner + start, len, level),
        Node::Stride { step, offset, mut child, .. } => {
            let offset = offset + start * step;
            if len == 1 {
                return slice(*child, offset, 1, level);
            }
            let level = match level {
                Some(level) => level,
                None => {
                    let child_len = child.len();
                    let node = std::mem::replace(&mut *child, Node::Empty);
                    let (node, _, level) = slice(node, 0, child_len, None);
                    *child = node;
                    level
                }
            };
            (Node::Stride { step, offset, len, child }, len, level)
        }
        Node::Concat { offsets: mut at, mut children } => {
            let first = at.partition_point(|&o| o <= start) - 1;
            let last = at.partition_point(|&o| o < start + len) - 1;
            let last_len = start + len - at[last];
            let start = start - at[first];
            children.truncate(last + 1);
            children.drain(..first);
            at.clear();
            let end = children.len() - 1;
            let mut total = 0;
            let mut retained_level = 0;
            for (i, slot) in children.iter_mut().enumerate() {
                let child = std::mem::replace(slot, Node::Empty);
                let a = if i == 0 { start } else { 0 };
                let b = if i == end { last_len } else { child.len() };
                // A zero maximum also proves that each child has level zero.
                let (child, child_len, child_level) = slice(child, a, b - a, level.filter(|&level| level == 0));
                at.push(total);
                total += child_len;
                retained_level = retained_level.max(child_level);
                *slot = child;
            }
            at.push(total);
            let node = if children.len() == 1 { children.pop().unwrap() } else { Node::Concat { offsets: at, children } };
            (node, total, retained_level)
        }
        Node::Shuffle { seed, salt, shape, mut child } => {
            let level = match level {
                Some(level) => level,
                None => {
                    let node = std::mem::replace(&mut *child, Node::Empty);
                    let (node, _, level) = slice(node, 0, shape.n, None);
                    *child = node;
                    level
                }
            };
            sliced(Node::Shuffle { seed, salt, shape, child }, start, len, level)
        }
        Node::Mix { il, mut children } => {
            let level = match level {
                Some(level) => level,
                None => {
                    let mut level = 0;
                    for slot in &mut children {
                        let child = std::mem::replace(slot, Node::Empty);
                        let len = child.len();
                        let (child, _, child_level) = slice(child, 0, len, None);
                        level = level.max(child_level);
                        *slot = child;
                    }
                    level
                }
            };
            sliced(Node::Mix { il, children }, start, len, level)
        }
    }
}

/// Fold a stride, carrying the incoming level unless it becomes a one-position slice.
fn stride((child, child_len, level): Compiled, step: usize) -> Compiled {
    let len = child_len.div_ceil(step);
    if len == 0 {
        return (Node::Empty, 0, 0);
    }
    if step == 1 || len == 1 {
        return slice(child, 0, len, Some(level));
    }
    let node = match child {
        Node::Slice { start, child, .. } => Node::Stride { step, offset: start, len, child },
        Node::Stride { step: inner, offset: base, child, .. } => Node::Stride { step: step * inner, offset: base, len, child },
        child => Node::Stride { step, offset: 0, len, child: Box::new(child) },
    };
    (node, len, level)
}
