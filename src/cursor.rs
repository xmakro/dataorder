//! Sequential iteration with a cursor tree that mirrors the compiled order.
//! Each node supports seeking, advancing and skipping. A mix keeps a tournament
//! tree to avoid seeking for every element. A shuffle uses random access for its
//! scattered child positions.
//!
//! Mix parts and the current concat child are initialized only when entered.
//! Mix and shuffle steps have separate structs so their inlining can be controlled:
//! the small mix step is inlined into dispatch, while the larger shuffle step stays
//! out of line to avoid adding overhead to other node kinds.

use crate::bounds::{BoundsError, resolve};
use crate::interleave::{Interleave, Iter};
use crate::order::{Node, Order, get_with};
use crate::perm::{self, Key, Shape};
use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Range, RangeBounds};

/// A seekable iterator over a range of an [`Order`].
///
/// Created by [`Order::iter`], it yields `(&source, index_within_source)` pairs.
/// Iteration moves forward; [`seek`](Cursor::seek) can move to an earlier or later
/// position, and [`set_range`](Cursor::set_range) selects a new range.
///
/// [`nth`](Iterator::nth) skips without returning intermediate elements.
/// [`count`](Iterator::count) uses the remaining length; [`last`](Iterator::last)
/// seeks through existing state, or uses random access for an undrawn cursor.
/// Neither walks the range. Allocation is deferred until an
/// element is requested, so creating, repositioning or counting an undrawn cursor
/// allocates nothing. `Debug` displays the current position and range end.
#[must_use = "a cursor is lazy: it yields nothing until iterated"]
pub struct Cursor<'a, T> {
    order: &'a Order<T>,
    /// Positioned at `pos` when in bounds; allocation-heavy roots defer construction
    /// until the first element is drawn.
    root: NodeCursor<'a>,
    pos: u64,
    end: u64,
    /// An exhausted cursor may leave its tree behind. Reposition it only when a
    /// later seek/range can produce elements, retaining all existing buffers.
    deferred_from: Option<u64>,
}

impl<'a, T> Cursor<'a, T> {
    pub(crate) fn new(order: &'a Order<T>, range: Range<usize>) -> Self {
        let (start, end) = (range.start as u64, range.end as u64);
        // Prepare allocation-free roots immediately. Composite roots defer their
        // buffers and seeks until the first draw.
        let root = match &order.root {
            Node::Empty => NodeCursor::Empty,
            Node::Source { src, offset, .. } => NodeCursor::Source { src: *src, offset: *offset, next: *offset + start },
            Node::Shuffle { seed, salt, shape, child } => NodeCursor::Shuffle(ShuffleCursor {
                seed: *seed,
                salt: *salt,
                shape: *shape,
                child,
                key: perm::key(*seed, order.ctx, *salt),
                pos: start,
                ctx: order.ctx,
                mixes: None,
            }),
            node => NodeCursor::Uninitialized { node, pos: start, ctx: order.ctx },
        };
        Cursor { order, root, pos: start, end, deferred_from: None }
    }

    /// Absolute order position of the next element, or the range end if exhausted.
    /// Unlike [`Iterator::position`], this does not consume any elements.
    #[must_use]
    pub fn offset(&self) -> usize {
        self.pos as usize
    }

    /// Elements left until the end of the range.
    #[must_use]
    pub fn remaining(&self) -> usize {
        (self.end - self.pos) as usize
    }

    /// Moves to absolute order position `pos`, keeping the current range end.
    /// `pos` may be before the range's original start. Seeking to the end exhausts
    /// the cursor; seeking backward lets iteration resume.
    ///
    /// Forward seeks skip; backward seeks reposition the cursor tree. Both reuse
    /// existing buffers. Entering another concat child or a previously unvisited mix
    /// part can allocate a child cursor. Use this method for repeated random access.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::mix([Seq::source(100).shuffle(1), Seq::source(50)]))?;
    /// let mut cursor = order.iter(100..);
    /// cursor.seek(120);
    /// assert_eq!(cursor.offset(), 120);
    /// assert_eq!(cursor.next(), Some(order.get(120)));
    /// cursor.seek(7); // Absolute position, even before the original range start.
    /// assert_eq!(cursor.len(), 150 - 7);
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    ///
    /// # Panics
    /// If `pos` is beyond the end of the range.
    pub fn seek(&mut self, pos: usize) {
        self.try_seek(pos).unwrap_or_else(|e| panic!("dataorder: {e}"));
    }

    /// Checked [`seek`](Self::seek). On error, the cursor is unchanged.
    ///
    /// # Errors
    /// [`BoundsError::SeekOutOfBounds`] when `pos` exceeds the current range end.
    pub fn try_seek(&mut self, pos: usize) -> Result<(), BoundsError> {
        if pos as u64 > self.end {
            return Err(BoundsError::SeekOutOfBounds { pos, end: self.end as usize });
        }
        let pos = pos as u64;
        let at = self.deferred_from.unwrap_or(self.pos);
        if pos == self.end {
            self.deferred_from = Some(at);
        } else {
            if pos > at {
                self.root.skip(pos - at);
            } else if pos < at {
                self.root.seek(pos, self.order.ctx);
            }
            self.deferred_from = None;
        }
        self.pos = pos;
        Ok(())
    }

    /// Selects a new range of the order and moves to its start.
    /// Reuses existing buffers, like [`seek`](Cursor::seek), so one cursor can serve
    /// multiple ranges. The range may extend beyond the previous range's end.
    /// Selecting an empty range defers tree repositioning and allocates nothing.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::source(10).shuffle(1))?;
    /// let all: Vec<usize> = order.iter(..).map(|(_, i)| i).collect();
    /// let mut cursor = order.iter(2..4);
    /// assert_eq!(cursor.by_ref().map(|(_, i)| i).collect::<Vec<_>>(), all[2..4]);
    /// cursor.set_range(7..);
    /// assert_eq!(cursor.len(), 3);
    /// assert_eq!(cursor.by_ref().map(|(_, i)| i).collect::<Vec<_>>(), all[7..]);
    /// cursor.set_range(..=0);
    /// assert_eq!(cursor.map(|(_, i)| i).collect::<Vec<_>>(), all[..1]);
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    ///
    /// # Panics
    /// As [`Order::iter`] does for the range.
    pub fn set_range(&mut self, range: impl RangeBounds<usize>) {
        self.try_set_range(range).unwrap_or_else(|e| panic!("dataorder: {e}"));
    }

    /// Checked [`set_range`](Self::set_range). On error, the cursor is unchanged.
    ///
    /// # Errors
    /// A reversed, overflowing or out-of-bounds range; see [`BoundsError`].
    pub fn try_set_range(&mut self, range: impl RangeBounds<usize>) -> Result<(), BoundsError> {
        let range = resolve(range, self.order.len())?;
        self.end = range.end as u64;
        self.try_seek(range.start)
    }

    /// Includes the source ordinal in each result, even for zero-sized source types.
    /// Existing position, range and allocated buffers are retained.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::mix([Seq::source(2), Seq::source(2)]))?;
    /// let ordinals: Vec<_> = order.iter(..).indexed().map(|(ordinal, _, _)| ordinal).collect();
    /// assert_eq!(ordinals, [0, 1, 0, 1]);
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    pub fn indexed(self) -> IndexedCursor<'a, T> {
        IndexedCursor { inner: self }
    }

    /// The next element with its ordinal, without recovering identity from a reference.
    #[inline]
    fn next_indexed(&mut self) -> Option<(usize, &'a T, usize)> {
        if self.pos == self.end {
            return None;
        }
        self.pos += 1;
        let (s, i) = self.root.next();
        Some((s as usize, &self.order.sources[s as usize], i as usize))
    }

    fn last_indexed(mut self) -> Option<(usize, &'a T, usize)> {
        if self.pos == self.end {
            None
        } else if matches!(self.root, NodeCursor::Uninitialized { .. }) {
            Some(self.order.get_indexed(self.end as usize - 1))
        } else {
            self.seek(self.end as usize - 1);
            self.next_indexed()
        }
    }

    /// Skip to the nth element, returning false when the cursor is exhausted.
    fn skip_n(&mut self, n: usize) -> bool {
        let n = n as u64;
        let left = self.end - self.pos;
        if n >= left {
            if left > 0 {
                self.deferred_from = Some(self.pos);
            }
            self.pos = self.end;
            return false;
        }
        if n > 0 {
            self.root.skip(n);
            self.pos += n;
        }
        true
    }
}

impl<T> fmt::Debug for Cursor<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cursor").field("position", &self.offset()).field("end", &(self.end as usize)).finish()
    }
}

/// Clones the current position and cursor state for independent iteration.
/// Copies initialized child cursors and mix buffers, so cloning an active cursor
/// can allocate. The source handles remain borrowed from the same order.
impl<T> Clone for Cursor<'_, T> {
    fn clone(&self) -> Self {
        Cursor { order: self.order, root: self.root.clone(), pos: self.pos, end: self.end, deferred_from: self.deferred_from }
    }
}

impl<'a, T> Iterator for Cursor<'a, T> {
    type Item = (&'a T, usize);

    #[inline]
    fn next(&mut self) -> Option<(&'a T, usize)> {
        self.next_indexed().map(|(_, source, index)| (source, index))
    }

    /// Skips `n` elements without visiting them, then yields the next.
    fn nth(&mut self, n: usize) -> Option<(&'a T, usize)> {
        if self.skip_n(n) { self.next() } else { None }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining(), Some(self.remaining()))
    }

    /// The elements left, without walking them.
    fn count(self) -> usize {
        self.remaining()
    }

    /// The last element of the range, reusing initialized state without walking there.
    fn last(self) -> Option<(&'a T, usize)> {
        self.last_indexed().map(|(_, source, index)| (source, index))
    }
}

impl<T> ExactSizeIterator for Cursor<'_, T> {}

impl<T> std::iter::FusedIterator for Cursor<'_, T> {}

/// A [`Cursor`] yielding `(source_ordinal, source, index_within_source)`.
/// Created by [`Cursor::indexed`]. Ordinals index [`Order::sources`] and remain
/// distinct for equal or zero-sized sources. Skipping, counting and cloning have
/// the same costs as on the underlying cursor.
#[must_use = "a cursor is lazy: it yields nothing until iterated"]
pub struct IndexedCursor<'a, T> {
    inner: Cursor<'a, T>,
}

impl<'a, T> IndexedCursor<'a, T> {
    /// Absolute position of the next element; see [`Cursor::offset`].
    #[must_use]
    pub fn offset(&self) -> usize {
        self.inner.offset()
    }

    /// Number of elements remaining in the range.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.inner.remaining()
    }

    /// Moves to an absolute position. Panics beyond the range end; see [`Cursor::seek`].
    pub fn seek(&mut self, pos: usize) {
        self.inner.seek(pos);
    }

    /// Checked seek; leaves the cursor unchanged on error. See [`Cursor::try_seek`].
    pub fn try_seek(&mut self, pos: usize) -> Result<(), BoundsError> {
        self.inner.try_seek(pos)
    }

    /// Selects a new range. Panics on invalid bounds; see [`Cursor::set_range`].
    pub fn set_range(&mut self, range: impl RangeBounds<usize>) {
        self.inner.set_range(range);
    }

    /// Checked range change; leaves the cursor unchanged on error. See [`Cursor::try_set_range`].
    pub fn try_set_range(&mut self, range: impl RangeBounds<usize>) -> Result<(), BoundsError> {
        self.inner.try_set_range(range)
    }

    /// Removes the ordinal adapter, preserving the cursor's position and buffers.
    pub fn into_cursor(self) -> Cursor<'a, T> {
        self.inner
    }
}

impl<T> Clone for IndexedCursor<'_, T> {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

impl<T> fmt::Debug for IndexedCursor<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IndexedCursor").field("position", &self.offset()).field("remaining", &self.remaining()).finish()
    }
}

impl<'a, T> Iterator for IndexedCursor<'a, T> {
    type Item = (usize, &'a T, usize);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next_indexed()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if self.inner.skip_n(n) { self.next() } else { None }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    fn count(self) -> usize {
        self.remaining()
    }

    fn last(self) -> Option<Self::Item> {
        self.inner.last_indexed()
    }
}

impl<T> ExactSizeIterator for IndexedCursor<'_, T> {}
impl<T> std::iter::FusedIterator for IndexedCursor<'_, T> {}

/// Marks a child whose position is unknown after a mix seek or before its first draw.
const UNSEEKED: u64 = u64::MAX;

/// Per-node iteration state. An explicit tag avoids decoding a tag stored in a
/// field's unused bit patterns on every dispatch.
/// `Uninitialized` defers the root's construction until an element is requested, including
/// across seeks and range changes. `Empty` doubles as "not built yet" for the children of
/// a `Concat` or `Mix`, whose real children are never empty (a concat drops them, a mix
/// never draws from them).
#[derive(Clone, Debug)]
#[repr(u8)]
pub(crate) enum NodeCursor<'a> {
    Empty,
    Source {
        src: u32,
        offset: u64,
        next: u64,
    },
    /// Only the current child has a cursor. Compatible buffers are recycled when
    /// entering another child; no cache grows with the number of visited children.
    Concat {
        children: &'a [Node],
        offsets: &'a [u64],
        idx: usize,
        left: u64,
        ctx: u64,
        child: Box<Self>,
    },
    // Keep the largest state out of every inline child slot. This adds one allocation
    // per active mix, but substantially shrinks wide mixes and worker-local cursors.
    Mix(Box<MixCursor<'a>>),
    Shuffle(ShuffleCursor<'a>),
    /// `depth` counts the repeats above this one; it salts the epoch contexts.
    Repeat {
        child_len: u64,
        depth: u32,
        epoch: u64,
        left: u64,
        ctx: u64,
        child: Box<Self>,
    },
    Slice {
        start: u64,
        child: Box<Self>,
    },
    Stride {
        step: u64,
        offset: u64,
        len: u64,
        left: u64,
        child: Box<Self>,
    },
    Uninitialized {
        node: &'a Node,
        pos: u64,
        ctx: u64,
    },
}

impl<'a> NodeCursor<'a> {
    /// Retarget storage to a new compiled node, then let the caller seek it. This
    /// is used at concat boundaries and lazily for children of a retargeted mix.
    /// Reuse only matching variants; all positional state is reset by `seek`.
    fn rebind(&mut self, node: &'a Node) {
        match (self, node) {
            (Self::Source { src, offset, .. }, Node::Source { src: s, offset: o, .. }) => {
                (*src, *offset) = (*s, *o);
            }
            (Self::Concat { children, offsets, .. }, Node::Concat { children: cs, offsets: os }) => {
                *children = cs;
                *offsets = os;
            }
            (Self::Mix(mix), Node::Mix { il, children }) => {
                if !std::ptr::eq(mix.il, il) {
                    mix.il = il;
                    mix.children = children;
                    mix.iter.rebind(il);
                    mix.next_j.resize(children.len(), UNSEEKED);
                    mix.next_j.fill(UNSEEKED);
                    mix.cursors.resize_with(children.len(), || Self::Empty);
                }
            }
            (Self::Shuffle(sh), Node::Shuffle { seed, salt, shape, child }) => {
                if !std::ptr::eq(sh.child, &**child) {
                    // Pointer-keyed seek states belong to the previous subtree.
                    // Keep the header, but do not accumulate a cache of old subtrees.
                    if let Some(mixes) = &mut sh.mixes {
                        mixes.clear();
                    }
                }
                (sh.seed, sh.salt, sh.shape, sh.child) = (*seed, *salt, *shape, child);
            }
            (Self::Repeat { child_len, depth, child, .. }, Node::Repeat { child_len: n, depth: d, child: c, .. }) => {
                (*child_len, *depth) = (*n, *d);
                child.rebind(c);
            }
            (Self::Slice { start, child }, Node::Slice { start: s, child: c, .. }) => {
                *start = *s;
                child.rebind(c);
            }
            (Self::Stride { step, offset, len, child, .. }, Node::Stride { step: s, offset: o, len: n, child: c }) => {
                (*step, *offset, *len) = (*s, *o, *n);
                child.rebind(c);
            }
            (cursor, node) => *cursor = Self::new(node),
        }
    }

    /// Creates a cursor over `node`; it must be positioned before drawing an element.
    fn new(node: &'a Node) -> Self {
        match node {
            Node::Empty => NodeCursor::Empty,
            Node::Source { src, offset, .. } => NodeCursor::Source { src: *src, offset: *offset, next: 0 },
            Node::Concat { offsets, children } => {
                NodeCursor::Concat { children, offsets, idx: 0, left: 0, ctx: 0, child: Box::new(NodeCursor::Empty) }
            }
            Node::Mix { il, children } => NodeCursor::Mix(Box::new(MixCursor::new(il, children))),
            Node::Shuffle { seed, salt, shape, child } => NodeCursor::Shuffle(ShuffleCursor {
                seed: *seed,
                salt: *salt,
                shape: *shape,
                child,
                key: Key::UNSET,
                pos: 0,
                ctx: 0,
                mixes: None,
            }),
            Node::Repeat { child_len, depth, child, .. } => NodeCursor::Repeat {
                child_len: *child_len,
                depth: *depth,
                epoch: 0,
                left: 0,
                ctx: 0,
                child: Box::new(NodeCursor::new(child)),
            },
            Node::Slice { start, child, .. } => NodeCursor::Slice { start: *start, child: Box::new(NodeCursor::new(child)) },
            Node::Stride { step, offset, len, child } => {
                NodeCursor::Stride { step: *step, offset: *offset, len: *len, left: 0, child: Box::new(NodeCursor::new(child)) }
            }
        }
    }

    /// Positions the cursor so that `next` yields element `pos` (`pos < len`) in context `ctx`.
    fn seek(&mut self, pos: u64, ctx: u64) {
        match self {
            NodeCursor::Empty => unreachable!("dataorder: seek in an empty sequence"),
            NodeCursor::Uninitialized { pos: p, ctx: c, .. } => {
                *p = pos;
                *c = ctx;
            }
            NodeCursor::Source { offset, next, .. } => *next = *offset + pos,
            NodeCursor::Concat { children, offsets, idx, left, ctx: c, child } => {
                let i = offsets.partition_point(|&o| o <= pos) - 1;
                // A boundary skip may retain buffers bound to a different child.
                *idx = i;
                child.rebind(&children[i]);
                *left = offsets[i + 1] - pos;
                *c = ctx;
                child.seek(pos - offsets[i], ctx);
            }
            NodeCursor::Mix(mix) => mix.seek(pos, ctx),
            NodeCursor::Shuffle(sh) => {
                sh.key = perm::key(sh.seed, ctx, sh.salt);
                sh.pos = pos;
                sh.ctx = ctx;
            }
            NodeCursor::Repeat { child_len, depth, epoch, left, ctx: c, child } => {
                let e = pos / *child_len;
                let r = pos - e * *child_len;
                *epoch = e;
                *left = *child_len - r;
                *c = ctx;
                child.seek(r, perm::epoch_ctx(ctx, e, *depth));
            }
            NodeCursor::Slice { start, child } => child.seek(*start + pos, ctx),
            NodeCursor::Stride { step, offset, len, left, child } => {
                *left = *len - pos;
                child.seek(*offset + pos * *step, ctx);
            }
        }
    }

    /// The next element. Must not be called past the end.
    #[inline]
    fn next(&mut self) -> (u32, u64) {
        match self {
            NodeCursor::Empty => unreachable!("dataorder: next in an empty sequence"),
            NodeCursor::Uninitialized { node, pos, ctx } => {
                let (node, pos, ctx) = (*node, *pos, *ctx);
                self.enter(node, pos, ctx)
            }
            NodeCursor::Source { src, next, .. } => {
                let i = *next;
                *next += 1;
                (*src, i)
            }
            NodeCursor::Concat { children, offsets, idx, left, ctx, child } => {
                if *left == 0 {
                    *idx += 1;
                    *left = offsets[*idx + 1] - offsets[*idx];
                    child.rebind(&children[*idx]);
                    child.seek(0, *ctx);
                }
                *left -= 1;
                child.next()
            }
            NodeCursor::Mix(mix) => mix.next(),
            NodeCursor::Shuffle(sh) => sh.next(),
            NodeCursor::Repeat { child_len, depth, epoch, left, ctx, child } => {
                if *left == 0 {
                    *epoch += 1;
                    *left = *child_len;
                    child.seek(0, perm::epoch_ctx(*ctx, *epoch, *depth));
                }
                *left -= 1;
                child.next()
            }
            NodeCursor::Slice { child, .. } => child.next(),
            NodeCursor::Stride { step, left, child, .. } => {
                *left -= 1;
                let r = child.next();
                if *left > 0 {
                    child.skip(*step - 1);
                }
                r
            }
        }
    }

    /// Only the first draw takes this path; keep initialization out of the walk's code.
    #[cold]
    #[inline(never)]
    fn enter(&mut self, node: &'a Node, pos: u64, ctx: u64) -> (u32, u64) {
        *self = Self::new(node);
        self.seek(pos, ctx);
        self.next()
    }

    /// Advances by `m` elements, which must exist. Within the current part or repetition the
    /// child skips; beyond it the cursor lands in the target one directly, or, exactly on a
    /// boundary, stays there and lets the next [`next`](NodeCursor::next) enter the following
    /// one, as the walk does.
    fn skip(&mut self, m: u64) {
        if m == 0 {
            return;
        }
        match self {
            NodeCursor::Empty => unreachable!("dataorder: skip in an empty sequence"),
            NodeCursor::Uninitialized { pos, .. } => *pos += m,
            NodeCursor::Source { next, .. } => *next += m,
            NodeCursor::Concat { children, offsets, idx, left, ctx, child } => {
                if m <= *left {
                    child.skip(m);
                    *left -= m;
                } else {
                    let pos = offsets[*idx + 1] - *left + m;
                    // The part containing `pos`, or the one ending there.
                    let i = offsets.partition_point(|&o| o < pos) - 1;
                    *idx = i;
                    if offsets[i + 1] == pos {
                        *left = 0;
                        // Keep the previous buffers; next/seek will retarget them.
                    } else {
                        *left = offsets[i + 1] - pos;
                        child.rebind(&children[i]);
                        child.seek(pos - offsets[i], *ctx);
                    }
                }
            }
            NodeCursor::Mix(mix) => mix.skip(m),
            NodeCursor::Shuffle(sh) => sh.pos += m,
            NodeCursor::Repeat { child_len, depth, epoch, left, ctx, child } => {
                if m <= *left {
                    child.skip(m);
                    *left -= m;
                } else {
                    let past = m - *left;
                    let e = *epoch + 1 + past / *child_len;
                    let r = past % *child_len;
                    if r == 0 {
                        *epoch = e - 1;
                        *left = 0;
                    } else {
                        *epoch = e;
                        *left = *child_len - r;
                        child.seek(r, perm::epoch_ctx(*ctx, e, *depth));
                    }
                }
            }
            NodeCursor::Slice { child, .. } => child.skip(m),
            NodeCursor::Stride { step, left, child, .. } => {
                *left -= m;
                // Land on the next element of the stride, or just past the last skipped one
                // when the stride is exhausted (the child may not extend a full step further).
                let steps = if *left > 0 { m * *step } else { (m - 1) * *step + 1 };
                child.skip(steps);
            }
        }
    }
}

/// A mix cursor with lazily positioned children.
///
/// `next_j[s]` records part `s`'s cursor position, or [`UNSEEKED`] when that position
/// is unknown. Skipping the mix leaves child cursors behind; the next draw from a
/// child catches it up. Children are stored inline to avoid an extra pointer load
/// per tree level, at the cost of reserving a full cursor slot for every part.
#[derive(Debug)]
pub(crate) struct MixCursor<'a> {
    il: &'a Interleave,
    children: &'a [Node],
    iter: Iter<'a>,
    pos: u64,
    next_j: Vec<u64>,
    cursors: Vec<NodeCursor<'a>>,
    ctx: u64,
}

impl Clone for MixCursor<'_> {
    fn clone(&self) -> Self {
        // Retain spare slots from previously visited larger concat children.
        // Removed child states stay removed; only their vector capacity survives.
        let mut next_j = Vec::with_capacity(self.next_j.capacity());
        next_j.extend_from_slice(&self.next_j);
        let mut cursors = Vec::with_capacity(self.cursors.capacity());
        cursors.extend_from_slice(&self.cursors);
        Self { il: self.il, children: self.children, iter: self.iter.clone(), pos: self.pos, next_j, cursors, ctx: self.ctx }
    }
}

impl<'a> MixCursor<'a> {
    fn new(il: &'a Interleave, children: &'a [Node]) -> Self {
        MixCursor {
            il,
            children,
            iter: il.iter(0..0),
            pos: 0,
            next_j: vec![UNSEEKED; children.len()],
            cursors: children.iter().map(|_| NodeCursor::Empty).collect(),
            ctx: 0,
        }
    }

    fn seek(&mut self, pos: u64, ctx: u64) {
        self.iter.seek(pos..self.il.len());
        self.pos = pos;
        self.next_j.fill(UNSEEKED);
        self.ctx = ctx;
    }

    /// Inlined into dispatch to avoid a function call for each element.
    #[inline(always)]
    fn next(&mut self) -> (u32, u64) {
        let (s, j) = self.iter.step();
        self.pos += 1;
        if self.next_j[s] != j {
            self.seek_child(s, j);
        }
        self.next_j[s] = j + 1;
        self.cursors[s].next()
    }

    /// Positions part `s` at `j`, initializing its cursor if necessary.
    /// A known child position can only be behind `j`, so it can skip forward. After
    /// a mix seek the position is unknown and needs a full seek. Keeping this slow
    /// path out of line reduces overhead in the ordinary mix step.
    #[cold]
    #[inline(never)]
    fn seek_child(&mut self, s: usize, j: u64) {
        let at = self.next_j[s];
        if at != UNSEEKED && at < j {
            self.cursors[s].skip(j - at);
            return;
        }
        // After recycling a mix, a slot may still refer to the previous child.
        self.cursors[s].rebind(&self.children[s]);
        self.cursors[s].seek(j, self.ctx);
    }

    /// Skips by walking short distances and seeking longer ones.
    /// The crossover is twice the part count for uniform mixes, four times for
    /// scheduled mixes. These thresholds were chosen from Ryzen 9 9950X3D timings;
    /// scheduled seeks cost more because their profiles have more segments.
    /// Child cursors stay in place until their next draw; see [`Self::seek_child`].
    fn skip(&mut self, m: u64) {
        self.pos += m;
        let hop = self.cursors.len() as u64 * if self.il.is_scheduled() { 4 } else { 2 };
        if m >= hop {
            self.iter.seek(self.pos..self.il.len());
        } else {
            for _ in 0..m {
                self.iter.step();
            }
        }
    }
}

/// The cursor of a `Shuffle`: a position counter; the child is read by random access.
#[derive(Clone, Debug)]
pub(crate) struct ShuffleCursor<'a> {
    seed: u64,
    salt: u64,
    shape: Shape,
    child: &'a Node,
    key: Key,
    pos: u64,
    ctx: u64,
    /// Sparse buffers for mixes reached by random traversal. Pointer keys identify
    /// immutable interleaves borrowed for this cursor's lifetime; they are never dereferenced.
    /// Box the map header too: shuffles that reach no mixes carry and drop only one
    /// optional pointer. An inline map measurably slows fresh shuffled-source cursors.
    #[allow(clippy::box_collection)]
    mixes: Option<Box<BTreeMap<usize, Iter<'a>>>>,
}

impl ShuffleCursor<'_> {
    /// Not inlined into the dispatcher: the permutation and the descent are the bulk of
    /// the code, and every other node kind would pay their prologue at each level.
    #[inline(never)]
    fn next(&mut self) -> (u32, u64) {
        let p = perm::permute(self.shape, self.key, self.pos);
        self.pos += 1;
        match self.child {
            Node::Source { src, offset, .. } => (*src, offset + p),
            _ => self.get_composite(p),
        }
    }

    /// Keep the map traversal's stack frame out of the common shuffled-source path.
    #[inline(never)]
    fn get_composite(&mut self, p: u64) -> (u32, u64) {
        get_with(self.child, p, self.ctx, |il, pos| {
            let key = std::ptr::from_ref(il) as usize;
            let mixes = self.mixes.get_or_insert_with(Box::default);
            let iter = mixes.entry(key).or_insert_with(|| il.iter(0..0));
            iter.seek(pos..pos + 1);
            iter.step()
        })
    }
}
