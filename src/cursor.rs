//! Sequential iteration with a cursor tree that mirrors the compiled order.
//! Each node supports seeking, advancing and skipping. A mix keeps a tournament
//! tree to avoid seeking for every element. A shuffle uses random access for its
//! scattered child positions.
//!
//! The order seed stays in the immutably borrowed order and is passed unchanged
//! through traversal. Nodes cache derived shuffle keys, not copies of the order seed.
//!
//! Mix parts and the current concat child are initialized only when entered.
//! Mix and shuffle steps have separate structs so their inlining can be controlled:
//! the small mix step is inlined into dispatch, while the larger shuffle step stays
//! out of line to avoid adding overhead to other node kinds.

use crate::error::BoundsError;
use crate::interleave::{Interleave, Iter};
use crate::order::{Item, Node, Order, get};
use crate::perm::{self, Key, Shape};
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
            root.seek(start, order.seed);
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
    /// let order = Order::new(Seq::source(10).shuffle())?;
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
                self.root.skip(pos - self.pos, self.order.seed);
            } else if pos < self.pos {
                self.root.seek(pos, self.order.seed);
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
        let (src, index) = self.root.next(self.order.seed);
        Some(Item { source_ordinal: src, source: &self.order.sources[src], record_index: index })
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
        Bound::Excluded(&s) => s.checked_add(1).ok_or(BoundsError::Overflow)?,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&e) => e.checked_add(1).ok_or(BoundsError::Overflow)?,
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

/// Marks an unknown position: a mix child's after a mix seek or before its first draw,
/// or a shuffle cursor's pass before it is positioned.
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
        src: usize,
        offset: usize,
        next: usize,
    },
    Concat(ConcatCursor<'a>),
    // Keep the largest state out of every inline child slot. This adds one allocation
    // per active mix, but substantially shrinks wide mixes and worker-local cursors.
    Mix(Box<MixCursor<'a>>),
    /// A permutation per pass; a plain shuffle has one pass.
    Shuffle(ShuffleCursor<'a>),
    Repeat(RepeatCursor<'a>),
    /// A unit stride: a plain selection of the child, which needs no count of the
    /// elements left.
    Slice {
        start: usize,
        child: Box<Self>,
    },
    /// Every `step`-th child position from `offset`, for a step above one.
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
            Node::Source { src, offset, .. } => NodeCursor::Source { src: *src, offset: *offset, next: 0 },
            Node::Concat { offsets, children } => NodeCursor::Concat(ConcatCursor::new(children, offsets)),
            Node::Mix { il, children } => NodeCursor::Mix(Box::new(MixCursor::new(il, children))),
            Node::Repeat { child_len, shuffle: Some(salt), child, .. } => NodeCursor::Shuffle(ShuffleCursor::new(child, *child_len, *salt)),
            Node::Repeat { child_len, shuffle: None, child, .. } => NodeCursor::Repeat(RepeatCursor::new(child, *child_len)),
            Node::Stride { step: 1, offset, child, .. } => NodeCursor::Slice { start: *offset, child: Box::new(NodeCursor::new(child)) },
            Node::Stride { step, offset, len, child } => {
                NodeCursor::Stride { step: *step, offset: *offset, len: *len, left: 0, child: Box::new(NodeCursor::new(child)) }
            }
        }
    }

    /// Positions the cursor so that `next` yields element `pos` (`pos < len`).
    /// Children receive the same order seed, with no enclosing repetition state.
    fn seek(&mut self, pos: usize, order_seed: u64) {
        match self {
            NodeCursor::Empty => unreachable!("dataorder: seek in an empty sequence"),
            NodeCursor::Source { offset, next, .. } => {
                *next = *offset + pos;
            }
            NodeCursor::Concat(concat) => concat.seek(pos, order_seed),
            NodeCursor::Mix(mix) => mix.seek(pos),
            NodeCursor::Shuffle(sh) => sh.seek(pos, order_seed),
            NodeCursor::Repeat(repeat) => repeat.seek(pos, order_seed),
            NodeCursor::Slice { start, child } => child.seek(*start + pos, order_seed),
            NodeCursor::Stride { step, offset, len, left, child } => {
                *left = *len - pos;
                child.seek(*offset + pos * *step, order_seed);
            }
        }
    }

    /// The next element. Must not be called past the end.
    #[inline]
    fn next(&mut self, order_seed: u64) -> (usize, usize) {
        match self {
            NodeCursor::Empty => unreachable!("dataorder: next in an empty sequence"),
            NodeCursor::Source { src, next, .. } => {
                let i = *next;
                *next += 1;
                (*src, i)
            }
            NodeCursor::Concat(concat) => concat.next(order_seed),
            NodeCursor::Mix(mix) => mix.next(order_seed),
            NodeCursor::Shuffle(sh) => sh.next(order_seed),
            NodeCursor::Repeat(repeat) => repeat.next(order_seed),
            NodeCursor::Slice { child, .. } => child.next(order_seed),
            NodeCursor::Stride { step, left, child, .. } => {
                *left -= 1;
                let r = child.next(order_seed);
                if *left > 0 {
                    child.skip(*step - 1, order_seed);
                }
                r
            }
        }
    }

    /// Advances by `m` elements, which must exist. Concat and repetition cursors
    /// handle their own boundaries, deferring entry into a following child or pass.
    fn skip(&mut self, m: usize, order_seed: u64) {
        if m == 0 {
            return;
        }
        match self {
            NodeCursor::Empty => unreachable!("dataorder: skip in an empty sequence"),
            NodeCursor::Source { next, .. } => *next += m,
            NodeCursor::Concat(concat) => concat.skip(m, order_seed),
            NodeCursor::Mix(mix) => mix.skip(m),
            NodeCursor::Shuffle(sh) => sh.skip(m, order_seed),
            NodeCursor::Repeat(repeat) => repeat.skip(m, order_seed),
            NodeCursor::Slice { child, .. } => child.skip(m, order_seed),
            NodeCursor::Stride { step, left, child, .. } => {
                *left -= m;
                // Land on the next element of the stride, or just past the last skipped one
                // when the stride is exhausted (the child may not extend a full step further).
                let steps = if *left > 0 { m * *step } else { (m - 1) * *step + 1 };
                child.skip(steps, order_seed);
            }
        }
    }
}

/// Keeps state for only the current concat child. Changing children replaces it.
///
/// Once positioned, `left` counts the elements remaining in child `idx`. A seek
/// enters its target child immediately. Walking or skipping to a boundary leaves
/// `left == 0`; the child may be exhausted or unbuilt, and `next` enters the next
/// one. Skipping across children builds only the child containing the target.
#[derive(Clone, Debug)]
pub(crate) struct ConcatCursor<'a> {
    children: &'a [Node],
    offsets: &'a [usize],
    idx: usize,
    left: usize,
    child: Box<NodeCursor<'a>>,
}

impl<'a> ConcatCursor<'a> {
    fn new(children: &'a [Node], offsets: &'a [usize]) -> Self {
        Self { children, offsets, idx: 0, left: 0, child: Box::new(NodeCursor::Empty) }
    }

    fn seek(&mut self, pos: usize, order_seed: u64) {
        let i = self.offsets.partition_point(|&o| o <= pos) - 1;
        if i != self.idx || matches!(*self.child, NodeCursor::Empty) {
            *self.child = NodeCursor::new(&self.children[i]);
        }
        self.idx = i;
        self.left = self.offsets[i + 1] - pos;
        self.child.seek(pos - self.offsets[i], order_seed);
    }

    /// Keep the ordinary child step inlined into node dispatch.
    #[inline(always)]
    fn next(&mut self, order_seed: u64) -> (usize, usize) {
        if self.left == 0 {
            self.idx += 1;
            self.left = self.offsets[self.idx + 1] - self.offsets[self.idx];
            *self.child = NodeCursor::new(&self.children[self.idx]);
            self.child.seek(0, order_seed);
        }
        self.left -= 1;
        self.child.next(order_seed)
    }

    fn skip(&mut self, m: usize, order_seed: u64) {
        if m <= self.left {
            self.child.skip(m, order_seed);
            self.left -= m;
        } else {
            let pos = self.offsets[self.idx + 1] - self.left + m;
            // The part containing `pos`, or the one ending there.
            let i = self.offsets.partition_point(|&o| o < pos) - 1;
            self.idx = i;
            self.left = self.offsets[i + 1] - pos;
            // At a boundary, defer building a child until it is entered.
            *self.child = if self.left == 0 { NodeCursor::Empty } else { NodeCursor::new(&self.children[i]) };
            if self.left > 0 {
                self.child.seek(pos - self.offsets[i], order_seed);
            }
        }
    }
}

/// Reuses the child cursor across plain repetition passes.
///
/// Once positioned, `left` counts the elements remaining in the current pass.
/// At a boundary, `left == 0` and `next` restarts the child. Skips jump directly
/// to the target pass; the child can stay behind when the target is a boundary.
#[derive(Clone, Debug)]
pub(crate) struct RepeatCursor<'a> {
    child_len: usize,
    left: usize,
    child: Box<NodeCursor<'a>>,
}

impl<'a> RepeatCursor<'a> {
    fn new(child: &'a Node, child_len: usize) -> Self {
        Self { child_len, left: 0, child: Box::new(NodeCursor::new(child)) }
    }

    fn seek(&mut self, pos: usize, order_seed: u64) {
        let r = pos % self.child_len;
        self.left = self.child_len - r;
        self.child.seek(r, order_seed);
    }

    /// Keep the ordinary child step inlined into node dispatch.
    #[inline(always)]
    fn next(&mut self, order_seed: u64) -> (usize, usize) {
        if self.left == 0 {
            self.left = self.child_len;
            self.child.seek(0, order_seed);
        }
        self.left -= 1;
        self.child.next(order_seed)
    }

    fn skip(&mut self, m: usize, order_seed: u64) {
        if m <= self.left {
            self.child.skip(m, order_seed);
            self.left -= m;
        } else {
            let r = (m - self.left) % self.child_len;
            if r == 0 {
                self.left = 0;
            } else {
                self.left = self.child_len - r;
                self.child.seek(r, order_seed);
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
        }
    }

    fn seek(&mut self, pos: usize) {
        self.iter.seek(pos..self.il.len());
        self.pos = pos;
        self.next_j.fill(UNSEEKED);
    }

    /// Inlined into dispatch to avoid a function call for each element.
    #[inline(always)]
    fn next(&mut self, order_seed: u64) -> (usize, usize) {
        let (s, j) = self.iter.step();
        self.pos += 1;
        if self.next_j[s] != j {
            self.seek_child(s, j, order_seed);
        }
        self.next_j[s] = j + 1;
        self.cursors[s].next(order_seed)
    }

    /// Positions part `s` at `j`, initializing its cursor if necessary.
    /// A known child position can only be behind `j`, so it can skip forward. After
    /// a mix seek the position is unknown and needs a full seek. Keeping this slow
    /// path out of line reduces overhead in the ordinary mix step.
    #[cold]
    #[inline(never)]
    fn seek_child(&mut self, s: usize, j: usize, order_seed: u64) {
        let at = self.next_j[s];
        if at != UNSEEKED && at < j {
            self.cursors[s].skip(j - at, order_seed);
            return;
        }
        if matches!(self.cursors[s], NodeCursor::Empty) {
            self.cursors[s] = NodeCursor::new(&self.children[s]);
        }
        self.cursors[s].seek(j, order_seed);
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

/// The cursor of a shuffled repetition, including a shuffle with one pass. Each pass
/// permutes the child's positions with its own derived key; the mix-free child is read
/// by random access. Walking or skipping to a pass boundary leaves `pos == shape.n`;
/// `next` enters the following pass and derives its key only then.
#[derive(Clone, Debug)]
pub(crate) struct ShuffleCursor<'a> {
    salt: u64,
    shape: Shape,
    child: &'a Node,
    /// The key of `pass`, once positioned.
    key: Key,
    /// The current pass, or [`UNSEEKED`] until the cursor is positioned.
    pass: usize,
    pos: usize,
}

impl<'a> ShuffleCursor<'a> {
    fn new(child: &'a Node, child_len: usize, salt: u64) -> Self {
        Self { salt, shape: Shape::new(child_len), child, key: Key::UNSET, pass: UNSEEKED, pos: 0 }
    }

    /// Moves to `pos` within `pass`. The order seed cannot change while a cursor
    /// borrows its order, so a pass keeps its derived key across moves within it.
    fn position(&mut self, pass: usize, pos: usize, order_seed: u64) {
        if pass != self.pass {
            self.rekey(pass, order_seed);
        }
        self.pos = pos;
    }

    /// Derives the key of a new pass. Kept out of line so that the per-element step
    /// and the seeks within a pass stay small.
    #[cold]
    #[inline(never)]
    fn rekey(&mut self, pass: usize, order_seed: u64) {
        self.pass = pass;
        self.key = perm::key(order_seed, pass, self.salt);
    }

    fn seek(&mut self, pos: usize, order_seed: u64) {
        let n = self.shape.n;
        self.position(pos / n, pos % n, order_seed);
    }

    /// Not inlined into the dispatcher: the permutation and the descent are the bulk of
    /// the code, and every other node kind would pay their prologue at each level.
    /// Entering a pass is a tail call, so the ordinary step needs no frame of its own.
    #[inline(never)]
    fn next(&mut self, order_seed: u64) -> (usize, usize) {
        debug_assert!(self.pass != UNSEEKED, "dataorder: next before positioning a shuffle");
        if self.pos == self.shape.n {
            return self.next_pass(order_seed);
        }
        self.step(order_seed)
    }

    /// Enters the following pass and takes its first element.
    #[cold]
    #[inline(never)]
    fn next_pass(&mut self, order_seed: u64) -> (usize, usize) {
        self.position(self.pass + 1, 0, order_seed);
        self.step(order_seed)
    }

    /// The element at `pos` of the current pass.
    #[inline(always)]
    fn step(&mut self, order_seed: u64) -> (usize, usize) {
        let p = perm::permute(self.shape, self.key, self.pos);
        self.pos += 1;
        match self.child {
            Node::Source { src, offset, .. } => (*src, offset + p),
            _ => get(self.child, p, order_seed),
        }
    }

    /// Advances by a positive number of elements, which must exist. Landing on a pass
    /// boundary defers entering the next pass until `next` needs it.
    fn skip(&mut self, m: usize, order_seed: u64) {
        let n = self.shape.n;
        if m <= n - self.pos {
            self.pos += m;
        } else {
            // The absolute target fits even when the total length is usize::MAX.
            let pos = self.pass * n + self.pos + m;
            self.position((pos - 1) / n, (pos - 1) % n + 1, order_seed);
        }
    }
}
