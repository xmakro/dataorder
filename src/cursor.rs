//! Sequential iteration: a tree of per-node cursors mirroring the order. Every cursor can
//! [`seek`](NodeCursor::seek) to a position, yield the [`next`](NodeCursor::next) element
//! and [`skip`](NodeCursor::skip) elements. Only a `Mix` gains from sequential access (the
//! interleave's tournament tree instead of a seek per element); a `Shuffle` reads its child
//! by random access ([`get`]) because its positions are scattered.
//!
//! Children of wide nodes (the parts of a `Mix`, the current part of a `Concat`) are built
//! when they are first entered, so a cursor costs what it visits. The mix and shuffle steps
//! live in their own structs. The mix step is inlined into the per-node dispatcher (a call
//! per element costs more than its code); the shuffle step is not, since inlined it would
//! make every other node kind pay its prologue at each level.

use crate::interleave::{Interleave, Iter};
use crate::order::{Node, Order, get, resolve};
use crate::perm::{self, Key, Shape};
use std::fmt;
use std::ops::{Range, RangeBounds};

/// Iterator over a range of an [`Order`], returned by [`Order::iter`]; yields
/// `(&source, index in the source)`. [`seek`](Cursor::seek) repositions it,
/// [`set_range`](Cursor::set_range) gives it another range, [`nth`](Iterator::nth) skips
/// without visiting, and [`count`](Iterator::count) and [`last`](Iterator::last) answer
/// from the range without walking it. It walks forward only (there is no
/// `DoubleEndedIterator`); [`Order::get`] serves random access. `Debug` prints the
/// position and the end of the range.
#[must_use = "a cursor is lazy: it yields nothing until iterated"]
pub struct Cursor<'a, T> {
    order: &'a Order<T>,
    /// Positioned at `pos` whenever `pos < len` (a range starting at the end leaves it
    /// untouched until a seek).
    root: NodeCursor<'a>,
    pos: u64,
    end: u64,
}

impl<'a, T> Cursor<'a, T> {
    pub(crate) fn new(order: &'a Order<T>, range: Range<usize>) -> Self {
        let (start, end) = (range.start as u64, range.end as u64);
        let mut root = NodeCursor::new(&order.root);
        if start < order.root.len() {
            root.seek(start, order.ctx);
        }
        Cursor { order, root, pos: start, end }
    }

    /// Position of the next element.
    #[must_use]
    pub fn position(&self) -> usize {
        self.pos as usize
    }

    /// Elements left until the end of the range.
    #[must_use]
    pub fn remaining(&self) -> usize {
        (self.end - self.pos) as usize
    }

    /// Continues at `pos`, anywhere up to the end of the range. Moving forward skips: a mix
    /// steps through its interleave, or re-seeks it for a long hop, and a hop into another
    /// repetition or concat part lands there directly; moving backward seeks afresh. Either
    /// way the cursor's allocations are reused (except that a concat builds the cursor of
    /// the part it lands in, and a mix that of a part it draws from for the first time), so
    /// seeking is the way to visit many scattered positions.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let order = Order::new(Seq::mix([Seq::source(100).shuffle(1), Seq::source(50)]))?;
    /// let mut cursor = order.iter(..);
    /// cursor.seek(120);
    /// assert_eq!(cursor.position(), 120);
    /// assert_eq!(cursor.next().map(|(&s, i)| (s, i)), Some(order.get(120)).map(|(&s, i)| (s, i)));
    /// cursor.seek(7);
    /// assert_eq!(cursor.len(), 150 - 7);
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    ///
    /// # Panics
    /// If `pos` is beyond the end of the range.
    pub fn seek(&mut self, pos: usize) {
        let pos = pos as u64;
        assert!(pos <= self.end, "dataorder: seek to {pos} beyond the end {}", self.end);
        if pos > self.pos {
            self.root.skip(pos - self.pos);
        } else if pos < self.pos {
            self.root.seek(pos, self.order.ctx);
        }
        self.pos = pos;
    }

    /// Continues over `range` of the order: seeks to its start, forward or backward as
    /// [`seek`](Cursor::seek) does, and ends at its end. One cursor thus serves any number
    /// of ranges with its buffers.
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
        let range = resolve(range, self.order.len());
        self.end = range.end as u64;
        self.seek(range.start);
    }
}

impl<T> fmt::Debug for Cursor<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cursor").field("position", &self.position()).field("end", &(self.end as usize)).finish()
    }
}

/// A clone continues from the same position, independently (for a look-ahead, say). It
/// copies the cursor tree: the cursors of every part entered so far, and the interleave's
/// tree of every mix, so it costs about what building and seeking the cursor cost.
impl<T> Clone for Cursor<'_, T> {
    fn clone(&self) -> Self {
        Cursor { order: self.order, root: self.root.clone(), pos: self.pos, end: self.end }
    }
}

impl<'a, T> Iterator for Cursor<'a, T> {
    type Item = (&'a T, usize);

    #[inline]
    fn next(&mut self) -> Option<(&'a T, usize)> {
        if self.pos == self.end {
            return None;
        }
        self.pos += 1;
        let (s, i) = self.root.next();
        Some((&self.order.sources[s as usize], i as usize))
    }

    /// Skips `n` elements without visiting them, then yields the next.
    fn nth(&mut self, n: usize) -> Option<(&'a T, usize)> {
        let n = n as u64;
        let left = self.end - self.pos;
        if n >= left {
            // To the end, so that the root stays where `position` says it is.
            self.root.skip(left);
            self.pos = self.end;
            return None;
        }
        if n > 0 {
            self.root.skip(n);
            self.pos += n;
        }
        self.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining(), Some(self.remaining()))
    }

    /// The elements left, without walking them.
    fn count(self) -> usize {
        self.remaining()
    }

    /// The last element of the range, by random access, without walking there.
    fn last(self) -> Option<(&'a T, usize)> {
        if self.pos == self.end { None } else { Some(self.order.get(self.end as usize - 1)) }
    }
}

impl<T> ExactSizeIterator for Cursor<'_, T> {}

impl<T> std::iter::FusedIterator for Cursor<'_, T> {}

/// Not yet seeked: the value of `MixCursor::next_j` for an untouched child.
const UNSEEKED: u64 = u64::MAX;

/// An explicit tag: with one hidden in a field's niche, every dispatch would decode it.
/// `Empty` doubles as "not built yet" for the lazily built children of a `Concat` or `Mix`,
/// whose real children are never empty (a concat drops them, a mix never draws from them).
#[derive(Clone, Debug)]
#[repr(u8)]
pub(crate) enum NodeCursor<'a> {
    Empty,
    Source {
        src: u32,
        offset: u64,
        next: u64,
    },
    /// Only the current child has a cursor, built when the concat is entered or seeked.
    Concat {
        children: &'a [Node],
        offsets: &'a [u64],
        idx: usize,
        left: u64,
        ctx: u64,
        child: Box<Self>,
    },
    Mix(MixCursor<'a>),
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
}

impl<'a> NodeCursor<'a> {
    /// A cursor over `node`, not yet seeked.
    fn new(node: &'a Node) -> Self {
        match node {
            Node::Empty => NodeCursor::Empty,
            Node::Source { src, offset, .. } => NodeCursor::Source { src: *src, offset: *offset, next: 0 },
            Node::Concat { offsets, children } => {
                NodeCursor::Concat { children, offsets, idx: 0, left: 0, ctx: 0, child: Box::new(NodeCursor::Empty) }
            }
            Node::Mix { il, children } => NodeCursor::Mix(MixCursor::new(il, children)),
            Node::Shuffle { seed, salt, shape, child } => {
                NodeCursor::Shuffle(ShuffleCursor { seed: *seed, salt: *salt, shape: *shape, child, key: Key::UNSET, pos: 0, ctx: 0 })
            }
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
            NodeCursor::Source { offset, next, .. } => *next = *offset + pos,
            NodeCursor::Concat { children, offsets, idx, left, ctx: c, child } => {
                let i = offsets.partition_point(|&o| o <= pos) - 1;
                if i != *idx || matches!(**child, NodeCursor::Empty) {
                    *idx = i;
                    **child = NodeCursor::new(&children[i]);
                }
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
            NodeCursor::Source { src, next, .. } => {
                let i = *next;
                *next += 1;
                (*src, i)
            }
            NodeCursor::Concat { children, offsets, idx, left, ctx, child } => {
                if *left == 0 {
                    *idx += 1;
                    *left = offsets[*idx + 1] - offsets[*idx];
                    **child = NodeCursor::new(&children[*idx]);
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
                        **child = NodeCursor::Empty;
                    } else {
                        *left = offsets[i + 1] - pos;
                        **child = NodeCursor::new(&children[i]);
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

/// The cursor of a `Mix`. Children are built and seeked lazily: `next_j[s]` is the index
/// the cursor of part `s` stands at, or [`UNSEEKED`] after a seek of the mix; skipping
/// leaves them behind and the mismatch skips them forward when they are drawn from again.
/// The slots are cursors in place, about 220 bytes per part whether entered or not: boxing
/// them (16 bytes per part) was measured at 0.5 to 1.5 ns per element more on every mix and
/// 2.9 ns on a mix of mixes, one dependent load per level, and rejected.
#[derive(Clone, Debug)]
pub(crate) struct MixCursor<'a> {
    il: &'a Interleave,
    children: &'a [Node],
    iter: Iter<'a>,
    pos: u64,
    next_j: Vec<u64>,
    cursors: Vec<NodeCursor<'a>>,
    ctx: u64,
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

    /// Inlined into the dispatcher: measured 1.5 ns per element cheaper than a call.
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

    /// Brings the cursor of part `s` to `j`: builds it if the part has not been entered
    /// yet, skips it forward if it stands before `j` (the mix only ever moves its parts
    /// forward by skipping, so a part left behind is behind, never ahead; a skip of a part
    /// that is itself a mix steps or re-seeks its interleave, where a seek would count
    /// every element of every part again), and seeks it after a seek of the mix.
    /// Out of the walk's hot loop: inlined, it cost 0.3 to 0.6 ns per element on mixes.
    #[cold]
    #[inline(never)]
    fn seek_child(&mut self, s: usize, j: u64) {
        let at = self.next_j[s];
        if matches!(self.cursors[s], NodeCursor::Empty) {
            self.cursors[s] = NodeCursor::new(&self.children[s]);
        } else if at != UNSEEKED && at < j {
            self.cursors[s].skip(j - at);
            return;
        }
        self.cursors[s].seek(j, self.ctx);
    }

    /// A long skip re-seeks the interleave instead of stepping through it: from twice the
    /// number of parts, four times when some are scheduled. Measured on mixes of 100 parts
    /// of a million elements (Ryzen 9 9950X3D): a step costs 10 to 13 ns, a seek 1.8 µs
    /// uniform and 4.5 µs with a fifth of the parts scheduled (their share functions have
    /// more segments to search), so the break-even hops are about 1.7 and 3.6 parts. Either
    /// way the parts' cursors stay where they are: each is skipped up to its next index when
    /// it is next drawn from (see [`seek_child`](MixCursor::seek_child)).
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
}

impl ShuffleCursor<'_> {
    /// Not inlined into the dispatcher: the permutation and the descent are the bulk of
    /// the code, and every other node kind would pay their prologue at each level.
    #[inline(never)]
    fn next(&mut self) -> (u32, u64) {
        let p = perm::permute(self.shape, self.key, self.pos);
        self.pos += 1;
        get(self.child, p, self.ctx)
    }
}
