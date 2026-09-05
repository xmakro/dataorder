//! Sequential iteration: a tree of per-node cursors mirroring the order. Every cursor can
//! [`seek`](NodeCursor::seek) to a position, yield the [`next`](NodeCursor::next) element
//! and [`skip`](NodeCursor::skip) elements. Only a `Mix` gains from sequential access (the
//! interleave's tournament tree instead of a seek per element); a `Shuffle` reads its child
//! by random access ([`get`]) because its positions are scattered.
//!
//! The mix and shuffle steps live in their own structs. The mix step is inlined into the
//! per-node dispatcher (a call per element costs more than its code); the shuffle step is
//! not, since inlined it would make every other node kind pay its prologue at each level.

use crate::interleave::{Interleave, Iter};
use crate::order::{get, Node, Order};
use crate::perm::{self, Key, Shape};
use std::ops::Range;

/// Iterator over a range of an [`Order`], returned by [`Order::iter`]; yields
/// `(&source, index in the source)`.
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

    /// Continues at `pos`, which may lie anywhere before the end of the range.
    ///
    /// # Panics
    /// If `pos` is beyond the end of the range.
    pub fn seek(&mut self, pos: usize) {
        let pos = pos as u64;
        assert!(pos <= self.end, "dataorder: seek to {pos} beyond the end {}", self.end);
        self.pos = pos;
        if pos < self.end {
            self.root.seek(pos, self.ctx);
        }
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

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining(), Some(self.remaining()))
    }
}

impl<T> ExactSizeIterator for Cursor<'_, T> {}

impl<T> std::iter::FusedIterator for Cursor<'_, T> {}

/// Not yet seeked: the value of `MixCursor::next_j` for an untouched child.
const UNSEEKED: u64 = u64::MAX;

/// An explicit tag: with one hidden in a field's niche, every dispatch would decode it.
#[derive(Clone, Debug)]
#[repr(u8)]
pub(crate) enum NodeCursor<'a> {
    Empty,
    Source { src: u32, offset: u64, next: u64 },
    /// Only the current child has a cursor; the others are built on entry.
    Concat { children: &'a [Node], offsets: &'a [u64], idx: usize, left: u64, ctx: u64, child: Box<Self> },
    Mix(MixCursor<'a>),
    Shuffle(ShuffleCursor<'a>),
    Repeat { child_len: u64, depth: u32, epoch: u64, left: u64, ctx: u64, child: Box<Self> },
    Slice { start: u64, child: Box<Self> },
    Stride { step: u64, offset: u64, len: u64, left: u64, child: Box<Self> },
}

impl<'a> NodeCursor<'a> {
    /// A cursor over `node`, not yet seeked.
    fn new(node: &'a Node) -> Self {
        match node {
            Node::Empty => NodeCursor::Empty,
            Node::Source { src, offset, .. } => NodeCursor::Source { src: *src, offset: *offset, next: 0 },
            Node::Concat { offsets, children } => NodeCursor::Concat {
                children,
                offsets,
                idx: 0,
                left: 0,
                ctx: 0,
                child: Box::new(NodeCursor::new(&children[0])),
            },
            Node::Mix { il, children } => NodeCursor::Mix(MixCursor::new(il, children)),
            Node::Shuffle { seed, shape, child } => NodeCursor::Shuffle(ShuffleCursor { seed: *seed, shape: *shape, child, key: Key::UNSET, pos: 0, ctx: 0 }),
            Node::Repeat { child_len, depth, child, .. } => NodeCursor::Repeat {
                child_len: *child_len,
                depth: *depth,
                epoch: 0,
                left: 0,
                ctx: 0,
                child: Box::new(NodeCursor::new(child)),
            },
            Node::Slice { start, child, .. } => NodeCursor::Slice { start: *start, child: Box::new(NodeCursor::new(child)) },
            Node::Stride { step, offset, len, child } => NodeCursor::Stride {
                step: *step,
                offset: *offset,
                len: *len,
                left: 0,
                child: Box::new(NodeCursor::new(child)),
            },
        }
    }

    /// Positions the cursor so that `next` yields element `pos` (`pos < len`) in context `ctx`.
    fn seek(&mut self, pos: u64, ctx: u64) {
        match self {
            NodeCursor::Empty => unreachable!("dataorder: seek in an empty sequence"),
            NodeCursor::Source { offset, next, .. } => *next = *offset + pos,
            NodeCursor::Concat { children, offsets, idx, left, ctx: c, child } => {
                let i = offsets.partition_point(|&o| o <= pos) - 1;
                if i != *idx {
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

    /// Advances by `m` elements, which must exist.
    fn skip(&mut self, mut m: u64) {
        match self {
            NodeCursor::Empty => unreachable!("dataorder: skip in an empty sequence"),
            NodeCursor::Source { next, .. } => *next += m,
            NodeCursor::Concat { children, offsets, idx, left, ctx, child } => {
                while m > 0 {
                    if *left == 0 {
                        *idx += 1;
                        *left = offsets[*idx + 1] - offsets[*idx];
                        **child = NodeCursor::new(&children[*idx]);
                        child.seek(0, *ctx);
                    }
                    let t = m.min(*left);
                    child.skip(t);
                    *left -= t;
                    m -= t;
                }
            }
            NodeCursor::Mix(mix) => mix.skip(m),
            NodeCursor::Shuffle(sh) => sh.pos += m,
            NodeCursor::Repeat { child_len, depth, epoch, left, ctx, child } => {
                while m > 0 {
                    if *left == 0 {
                        *epoch += 1;
                        *left = *child_len;
                        child.seek(0, perm::epoch_ctx(*ctx, *epoch, *depth));
                    }
                    let t = m.min(*left);
                    child.skip(t);
                    *left -= t;
                    m -= t;
                }
            }
            NodeCursor::Slice { child, .. } => child.skip(m),
            NodeCursor::Stride { step, left, child, .. } => {
                if m == 0 {
                    return;
                }
                *left -= m;
                // Land on the next element of the stride, or just past the last skipped one
                // when the stride is exhausted (the child may not extend a full step further).
                let steps = if *left > 0 { m * *step } else { (m - 1) * *step + 1 };
                child.skip(steps);
            }
        }
    }
}

/// The cursor of a `Mix`. Children are seeked lazily: `next_j[s]` is the index the cursor
/// of part `s` stands at, or [`UNSEEKED`]; skipping leaves them behind and the mismatch
/// re-seeks them.
#[derive(Clone, Debug)]
pub(crate) struct MixCursor<'a> {
    il: &'a Interleave,
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
            iter: il.iter(0..0),
            pos: 0,
            next_j: vec![UNSEEKED; children.len()],
            cursors: children.iter().map(NodeCursor::new).collect(),
            ctx: 0,
        }
    }

    fn seek(&mut self, pos: u64, ctx: u64) {
        self.iter = self.il.iter(pos..self.il.len());
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
            self.cursors[s].seek(j, self.ctx);
        }
        self.next_j[s] = j + 1;
        self.cursors[s].next()
    }

    /// A long skip re-seeks the interleave instead of stepping through it.
    fn skip(&mut self, m: u64) {
        self.pos += m;
        if m >= 4 * self.cursors.len() as u64 {
            self.iter = self.il.iter(self.pos..self.il.len());
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
