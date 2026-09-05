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
use crate::order::{Node, Order, get};
use crate::perm::{self, Key, Shape};
use std::ops::Range;

/// Iterator over a range of an [`Order`], returned by [`Order::iter`]; yields
/// `(&source, index in the source)`. [`seek`](Cursor::seek) repositions it, and
/// [`nth`](Iterator::nth) skips without visiting.
#[derive(Debug)]
#[must_use = "a cursor is lazy: it yields nothing until iterated"]
pub struct Cursor<'a, T> {
    sources: &'a [T],
    root: NodeCursor<'a>,
    pos: u64,
    end: u64,
    ctx: u64,
}

impl<'a, T> Cursor<'a, T> {
    pub(crate) fn new(order: &'a Order<T>, range: Range<usize>) -> Self {
        let (start, end) = (range.start as u64, range.end as u64);
        let mut root = NodeCursor::new(&order.root);
        if start < end {
            root.seek(start, order.ctx);
        }
        Cursor { sources: &order.sources, root, pos: start, end, ctx: order.ctx }
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
    /// way the cursor's allocations are reused, so seeking is the way to visit many
    /// scattered positions.
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
        if pos > self.pos && pos < self.end {
            self.root.skip(pos - self.pos);
        } else if pos < self.pos {
            self.root.seek(pos, self.ctx);
        }
        self.pos = pos;
    }
}

/// A clone continues from the same position, independently (for a look-ahead, say).
impl<T> Clone for Cursor<'_, T> {
    fn clone(&self) -> Self {
        Cursor { sources: self.sources, root: self.root.clone(), pos: self.pos, end: self.end, ctx: self.ctx }
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
        Some((&self.sources[s as usize], i as usize))
    }

    /// Skips `n` elements without visiting them, then yields the next.
    fn nth(&mut self, n: usize) -> Option<(&'a T, usize)> {
        let n = n as u64;
        if n >= self.end - self.pos {
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
            Node::Shuffle { seed, shape, child } => {
                NodeCursor::Shuffle(ShuffleCursor { seed: *seed, shape: *shape, child, key: Key::UNSET, pos: 0, ctx: 0 })
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
                sh.key = perm::key(sh.seed, ctx);
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
/// the cursor of part `s` stands at, or [`UNSEEKED`]; skipping leaves them behind and the
/// mismatch re-seeks them.
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

    /// Builds the cursor of part `s` if it has not been entered yet and seeks it to `j`.
    /// Out of the walk's hot loop: inlined, it cost 0.3 to 0.6 ns per element on mixes.
    #[cold]
    #[inline(never)]
    fn seek_child(&mut self, s: usize, j: u64) {
        if matches!(self.cursors[s], NodeCursor::Empty) {
            self.cursors[s] = NodeCursor::new(&self.children[s]);
        }
        self.cursors[s].seek(j, self.ctx);
    }

    /// A long skip re-seeks the interleave instead of stepping through it.
    fn skip(&mut self, m: u64) {
        self.pos += m;
        if m >= 4 * self.cursors.len() as u64 {
            self.iter.seek(self.pos..self.il.len());
            self.next_j.fill(UNSEEKED);
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
