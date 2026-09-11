//! Sequential iteration with a cursor tree that mirrors the compiled order.
//! Each node supports seeking, advancing and skipping. A mix keeps a tournament
//! tree to avoid seeking for every element. A shuffle uses random access for its
//! scattered child positions.
//!
//! Mix parts and the current concat child are initialized only when entered.
//! Mix and shuffle steps have separate structs so their inlining can be controlled:
//! the small mix step is inlined into dispatch, while the larger shuffle step stays
//! out of line to avoid adding overhead to other node kinds.

use crate::error::BoundsError;
use crate::interleave::{Interleave, Iter};
use crate::order::{Item, Node, Order, get};
use crate::perm::{self, Context, Key, Shape};
use std::fmt;
use std::ops::{Bound, Range, RangeBounds};

/// A seekable iterator over a range of an [`Order`].
///
/// Created by [`Order::iter`] or [`Order::cursor`], it yields [`Item`] values with explicit source ordinals.
/// Iteration moves forward; [`reset`](Cursor::reset) replaces the remaining range
/// using absolute order positions and moves to its start.
/// [`len`](ExactSizeIterator::len) reports how many elements remain.
///
/// [`nth`](Iterator::nth) skips without returning intermediate elements.
/// [`count`](Iterator::count) uses the remaining length; [`last`](Iterator::last)
/// uses random access. Neither walks the range. Construction positions the cursor
/// immediately and can allocate, even for an empty range. `Debug` displays the
/// current position and range end.
#[must_use = "a cursor yields nothing until iterated"]
pub struct Cursor<'a, T> {
    order: &'a Order<T>,
    /// Positioned at `pos` whenever it is before the end of the order.
    root: NodeCursor<'a>,
    pos: usize,
    end: usize,
}

impl<'a, T> Cursor<'a, T> {
    /// Constructs a cursor over a validated range within the order.
    pub(crate) fn new(order: &'a Order<T>, range: Range<usize>) -> Self {
        let (start, end) = (range.start, range.end);
        let mut root = NodeCursor::new(&order.root);
        if start < order.root.len() {
            root.seek(start, Context::new(order.seed));
        }
        Self { order, root, pos: start, end }
    }

    /// Absolute order position of the next element, or the range end if exhausted.
    /// Unlike [`Iterator::position`], this does not consume any elements.
    #[must_use]
    pub fn offset(&self) -> usize {
        self.pos
    }

    /// Replaces the remaining range and moves to its start.
    /// Bounds use absolute positions in the original order, as in [`Order::cursor`].
    /// An unbounded start means 0; an unbounded end means the order's length.
    /// `reset(..)` restarts the whole order, and `reset(pos..)` reads from `pos`
    /// to the order's end. Use `reset(pos..end)` to keep a chosen endpoint.
    /// The previous range does not constrain the new one.
    ///
    /// Forward moves skip; backward moves reposition the cursor tree. Both reuse
    /// existing buffers within the current child. Entering another concat child
    /// creates fresh state; entering a previously unvisited mix part can also allocate.
    /// Use this method for repeated random access or to visit multiple ranges.
    /// Empty ranges are positioned like other ranges and can allocate.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::source(10).shuffle(1))?;
    /// let all: Vec<usize> = order.iter().map(|item| item.record_index).collect();
    /// let mut cursor = order.cursor(2..6)?;
    /// cursor.reset(4..6)?; // Keep the chosen endpoint explicitly.
    /// assert_eq!(cursor.by_ref().map(|item| item.record_index).collect::<Vec<_>>(), all[4..6]);
    /// cursor.reset(7..)?; // Continue through the order's end, beyond the old range.
    /// assert_eq!(cursor.len(), 3);
    /// assert_eq!(cursor.by_ref().map(|item| item.record_index).collect::<Vec<_>>(), all[7..]);
    /// cursor.reset(..=0)?;
    /// assert_eq!(cursor.by_ref().map(|item| item.record_index).collect::<Vec<_>>(), all[..1]);
    /// cursor.reset(..)?; // Restart the whole order.
    /// assert_eq!(cursor.map(|item| item.record_index).collect::<Vec<_>>(), all);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// On error, the cursor is unchanged.
    ///
    /// # Errors
    /// A reversed, overflowing or out-of-bounds range; see [`BoundsError`].
    pub fn reset(&mut self, range: impl RangeBounds<usize>) -> Result<(), BoundsError> {
        let range = resolve_range(range, self.order.len())?;
        self.end = range.end;
        self.reposition(range.start);
        Ok(())
    }

    /// Moves to a validated position, including positions in an empty range.
    fn reposition(&mut self, pos: usize) {
        // At the order's end there is no element to position the tree at. Any
        // subsequent move into the order is backward and seeks the tree anew.
        if pos < self.order.root.len() {
            if pos > self.pos {
                self.root.skip(pos - self.pos);
            } else if pos < self.pos {
                self.root.seek(pos, Context::new(self.order.seed));
            }
        }
        self.pos = pos;
    }
}

impl<T> fmt::Debug for Cursor<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cursor").field("position", &self.offset()).field("end", &self.end).finish()
    }
}

/// Clones the current position and cursor state for independent iteration.
/// Copies initialized child cursors and mix buffers, so cloning an active cursor
/// can allocate. Spare buffer capacity is not preserved, so later seeks may allocate
/// too. The source handles remain borrowed from the same order.
impl<T> Clone for Cursor<'_, T> {
    fn clone(&self) -> Self {
        Cursor { order: self.order, root: self.root.clone(), pos: self.pos, end: self.end }
    }
}

impl<'a, T> Iterator for Cursor<'a, T> {
    type Item = Item<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos == self.end {
            return None;
        }
        self.pos += 1;
        let (s, i, epoch) = self.root.next();
        Some(Item { source_ordinal: s as usize, source: &self.order.sources[s as usize], record_index: i, epoch })
    }

    /// Skips `n` elements without visiting them, then yields the next.
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let skip = n.min(self.end - self.pos);
        self.reposition(self.pos + skip);
        self.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    /// The elements left, without walking them.
    fn count(self) -> usize {
        self.len()
    }

    /// The last element of the range, by random access without walking there.
    fn last(self) -> Option<Self::Item> {
        if self.pos == self.end { None } else { self.order.get(self.end - 1) }
    }
}

impl<T> ExactSizeIterator for Cursor<'_, T> {
    /// Elements left until the end of the range.
    fn len(&self) -> usize {
        self.end - self.pos
    }
}

impl<T> std::iter::FusedIterator for Cursor<'_, T> {}

/// Normalize and validate against the order length.
pub(crate) fn resolve_range(range: impl RangeBounds<usize>, len: usize) -> Result<Range<usize>, BoundsError> {
    let start = match range.start_bound() {
        Bound::Included(&s) => s,
        Bound::Excluded(&s) => s.checked_add(1).ok_or(BoundsError::StartOverflow)?,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&e) => e.checked_add(1).ok_or(BoundsError::EndOverflow)?,
        Bound::Excluded(&e) => e,
        Bound::Unbounded => len,
    };
    if start > end {
        return Err(BoundsError::Reversed { start, end });
    }
    if end > len {
        return Err(BoundsError::OutOfBounds { end, len });
    }
    Ok(start..end)
}

/// Marks a child whose position is unknown after a mix seek or before its first draw.
const UNSEEKED: usize = usize::MAX;

/// Per-node iteration state. An explicit tag avoids decoding a tag stored in a
/// field's unused bit patterns on every dispatch.
/// `Empty` doubles as "not built yet" for the children of
/// a `Concat` or `Mix`, whose real children are never empty (a concat drops them, a mix
/// never draws from them).
#[derive(Clone, Debug)]
#[repr(u8)]
pub(crate) enum NodeCursor<'a> {
    Empty,
    Source {
        src: u32,
        offset: usize,
        next: usize,
        epoch: usize,
    },
    /// Only the current child has a cursor. Changing children replaces its state.
    Concat {
        children: &'a [Node],
        offsets: &'a [usize],
        idx: usize,
        left: usize,
        ctx: Context,
        child: Box<Self>,
    },
    // Keep the largest state out of every inline child slot. This adds one allocation
    // per active mix, but substantially shrinks wide mixes and worker-local cursors.
    Mix(Box<MixCursor<'a>>),
    Shuffle(ShuffleCursor<'a>),
    /// The original count determines the child's accumulated epoch number.
    Repeat {
        child_len: usize,
        times: usize,
        epoch: usize,
        left: usize,
        ctx: Context,
        child: Box<Self>,
    },
    Slice {
        start: usize,
        child: Box<Self>,
    },
    Stride {
        step: usize,
        offset: usize,
        len: usize,
        left: usize,
        child: Box<Self>,
    },
}

impl<'a> NodeCursor<'a> {
    /// Creates a cursor over `node`; it must be positioned before drawing an element.
    fn new(node: &'a Node) -> Self {
        match node {
            Node::Empty => NodeCursor::Empty,
            Node::Source { src, offset, .. } => NodeCursor::Source { src: *src, offset: *offset, next: 0, epoch: 0 },
            Node::Concat { offsets, children } => {
                NodeCursor::Concat { children, offsets, idx: 0, left: 0, ctx: Context::new(0), child: Box::new(NodeCursor::Empty) }
            }
            Node::Mix { il, children } => NodeCursor::Mix(Box::new(MixCursor::new(il, children))),
            Node::Shuffle { seed, salt, shape, child } => NodeCursor::Shuffle(ShuffleCursor {
                seed: *seed,
                salt: *salt,
                shape: *shape,
                child,
                key: Key::UNSET,
                pos: 0,
                ctx: Context::new(0),
            }),
            Node::Repeat { child_len, times, child, .. } => NodeCursor::Repeat {
                child_len: *child_len,
                times: *times,
                epoch: 0,
                left: 0,
                ctx: Context::new(0),
                child: Box::new(NodeCursor::new(child)),
            },
            Node::Slice { start, child, .. } => NodeCursor::Slice { start: *start, child: Box::new(NodeCursor::new(child)) },
            Node::Stride { step, offset, len, child } => {
                NodeCursor::Stride { step: *step, offset: *offset, len: *len, left: 0, child: Box::new(NodeCursor::new(child)) }
            }
        }
    }

    /// Positions the cursor so that `next` yields element `pos` (`pos < len`) in context `ctx`.
    fn seek(&mut self, pos: usize, ctx: Context) {
        match self {
            NodeCursor::Empty => unreachable!("dataorder: seek in an empty sequence"),
            NodeCursor::Source { offset, next, epoch, .. } => {
                *next = *offset + pos;
                *epoch = ctx.epoch;
            }
            NodeCursor::Concat { children, offsets, idx, left, ctx: c, child } => {
                let i = offsets.partition_point(|&o| o <= pos) - 1;
                if i != *idx || matches!(**child, Self::Empty) {
                    **child = Self::new(&children[i]);
                }
                *idx = i;
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
            NodeCursor::Repeat { child_len, times, epoch, left, ctx: c, child } => {
                let e = pos / *child_len;
                let r = pos - e * *child_len;
                *epoch = e;
                *left = *child_len - r;
                *c = ctx;
                child.seek(r, ctx.repeat(*times, e));
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
    fn next(&mut self) -> (u32, usize, usize) {
        match self {
            NodeCursor::Empty => unreachable!("dataorder: next in an empty sequence"),
            NodeCursor::Source { src, next, epoch, .. } => {
                let i = *next;
                *next += 1;
                (*src, i, *epoch)
            }
            NodeCursor::Concat { children, offsets, idx, left, ctx, child } => {
                if *left == 0 {
                    *idx += 1;
                    *left = offsets[*idx + 1] - offsets[*idx];
                    **child = Self::new(&children[*idx]);
                    child.seek(0, *ctx);
                }
                *left -= 1;
                child.next()
            }
            NodeCursor::Mix(mix) => mix.next(),
            NodeCursor::Shuffle(sh) => sh.next(),
            NodeCursor::Repeat { child_len, times, epoch, left, ctx, child } => {
                if *left == 0 {
                    *epoch += 1;
                    *left = *child_len;
                    child.seek(0, ctx.repeat(*times, *epoch));
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

    /// Advances by `m` elements, which must exist. Within the current part or repetition the
    /// child skips; beyond it the cursor lands in the target one directly, or, exactly on a
    /// boundary, stays there and lets the next [`next`](NodeCursor::next) enter the following
    /// one, as the walk does.
    fn skip(&mut self, m: usize) {
        if m == 0 {
            return;
        }
        match self {
            NodeCursor::Empty => unreachable!("dataorder: skip in an empty sequence"),
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
                    *left = offsets[i + 1] - pos;
                    // At a boundary, defer building a child until it is entered.
                    **child = if *left == 0 { Self::Empty } else { Self::new(&children[i]) };
                    if *left > 0 {
                        child.seek(pos - offsets[i], *ctx);
                    }
                }
            }
            NodeCursor::Mix(mix) => mix.skip(m),
            NodeCursor::Shuffle(sh) => sh.pos += m,
            NodeCursor::Repeat { child_len, times, epoch, left, ctx, child } => {
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
                        child.seek(r, ctx.repeat(*times, e));
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
#[derive(Clone, Debug)]
pub(crate) struct MixCursor<'a> {
    il: &'a Interleave,
    children: &'a [Node],
    iter: Iter<'a>,
    pos: usize,
    next_j: Vec<usize>,
    cursors: Vec<NodeCursor<'a>>,
    ctx: Context,
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
            ctx: Context::new(0),
        }
    }

    fn seek(&mut self, pos: usize, ctx: Context) {
        self.iter.seek(pos..self.il.len());
        self.pos = pos;
        self.next_j.fill(UNSEEKED);
        self.ctx = ctx;
    }

    /// Inlined into dispatch to avoid a function call for each element.
    #[inline(always)]
    fn next(&mut self) -> (u32, usize, usize) {
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
    fn seek_child(&mut self, s: usize, j: usize) {
        let at = self.next_j[s];
        if at != UNSEEKED && at < j {
            self.cursors[s].skip(j - at);
            return;
        }
        if matches!(self.cursors[s], NodeCursor::Empty) {
            self.cursors[s] = NodeCursor::new(&self.children[s]);
        }
        self.cursors[s].seek(j, self.ctx);
    }

    /// Skips by walking short distances and seeking longer ones.
    /// The crossover is twice the part count for uniform mixes, four times for
    /// scheduled mixes. These thresholds were chosen from Ryzen 9 9950X3D timings;
    /// scheduled seeks cost more because their profiles have more segments.
    /// Child cursors stay in place until their next draw; see [`Self::seek_child`].
    fn skip(&mut self, m: usize) {
        self.pos += m;
        let hop = self.cursors.len().saturating_mul(if self.il.is_scheduled() { 4 } else { 2 });
        if m >= hop {
            self.iter.seek(self.pos..self.il.len());
        } else {
            for _ in 0..m {
                self.iter.step();
            }
        }
    }
}

/// The cursor of a `Shuffle`: a position counter; its mix-free child is read by random access.
#[derive(Clone, Debug)]
pub(crate) struct ShuffleCursor<'a> {
    seed: u64,
    salt: u64,
    shape: Shape,
    child: &'a Node,
    key: Key,
    pos: usize,
    ctx: Context,
}

impl ShuffleCursor<'_> {
    /// Not inlined into the dispatcher: the permutation and the descent are the bulk of
    /// the code, and every other node kind would pay their prologue at each level.
    #[inline(never)]
    fn next(&mut self) -> (u32, usize, usize) {
        let p = perm::permute(self.shape, self.key, self.pos);
        self.pos += 1;
        match self.child {
            Node::Source { src, offset, .. } => (*src, offset + p, self.ctx.epoch),
            _ => get(self.child, p, self.ctx),
        }
    }
}
