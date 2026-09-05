//! The compiled order: a tree of nodes with precomputed lengths, prefix sums, interleave
//! indices and shuffle shapes, and random access by a stateless descent ([`get`]).

use crate::cursor::Cursor;
use crate::interleave::{Interleave, Sampling};
use crate::perm::{self, Shape};
use crate::{Dataset, Error, Seq};
use std::ops::Range;

/// A compiled node. Empty subtrees are folded to [`Node::Empty`], so every child of a
/// `Concat` or `Mix` and every child of a transform has elements (except that `Mix` keeps
/// empty children to preserve the interleave's sequence indices).
#[derive(Clone, Debug)]
pub(crate) enum Node {
    Empty,
    /// Elements `offset..offset + len` of source `src` (an index into `Order::sources`).
    Source { src: u32, offset: u64, len: u64 },
    /// `offsets[i]` is the position of child `i`'s first element; `offsets[k]` the length.
    Concat { offsets: Vec<u64>, children: Vec<Node> },
    Mix { il: Interleave, children: Vec<Node> },
    Shuffle { seed: u64, shape: Shape, child: Box<Node> },
    /// `depth` counts the repeats above this one; it salts the epoch contexts.
    Repeat { times: u64, child_len: u64, depth: u32, child: Box<Node> },
    Slice { start: u64, len: u64, child: Box<Node> },
    Stride { step: u64, offset: u64, len: u64, child: Box<Node> },
}

impl Node {
    pub(crate) fn len(&self) -> u64 {
        match self {
            Node::Empty => 0,
            Node::Source { len, .. } => *len,
            Node::Concat { offsets, .. } => *offsets.last().unwrap(),
            Node::Mix { il, .. } => il.len(),
            Node::Shuffle { shape, .. } => shape.n,
            Node::Repeat { times, child_len, .. } => times * child_len,
            Node::Slice { len, .. } => *len,
            Node::Stride { len, .. } => *len,
        }
    }
}

/// A compiled [`Seq`]: its length, random access by [`Order::get`] and seekable iteration
/// by [`Order::iter`]. Owns the sources; elements are `(&source, index)`. Lengths and
/// positions are `usize` at the interface and 64-bit inside, so an intermediate node may
/// be longer than the address space as long as the order itself is not.
#[derive(Clone, Debug)]
pub struct Order<T> {
    pub(crate) root: Node,
    /// Context of the root (the order's seed); repetitions derive their own from it.
    pub(crate) ctx: u64,
    pub(crate) sources: Vec<T>,
}

impl<T: Dataset> Order<T> {
    /// Compiles `seq` with seed 0. Consumes it; clone it first to keep it.
    pub fn compile(seq: Seq<T>) -> Result<Order<T>, Error> {
        Self::compile_seeded(seq, 0)
    }

    /// Compiles `seq`. The `seed` reseeds every shuffle in the order at once; shuffles
    /// keep their relative distinctness from their own seeds.
    pub fn compile_seeded(seq: Seq<T>, seed: u64) -> Result<Order<T>, Error> {
        let mut c = Compiler { sources: Vec::new() };
        let root = c.compile(seq, 0)?;
        if usize::try_from(root.len()).is_err() {
            return Err(Error::Overflow);
        }
        Ok(Order { root, ctx: seed, sources: c.sources })
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.root.len() as usize
    }

    /// `true` when there are no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The sources, in order of appearance in the configuration.
    pub fn sources(&self) -> &[T] {
        &self.sources
    }

    /// The element at `pos`: the source and the index in it.
    ///
    /// Constant work per node on the path, except that a [`Seq::Mix`] on the path costs a
    /// seek of the interleave (`O(k log s)` for `k` parts, `s` scheduled). Use [`Order::iter`]
    /// for consecutive positions.
    ///
    /// # Panics
    /// If `pos >= len()`.
    pub fn get(&self, pos: usize) -> (&T, usize) {
        assert!(pos < self.len(), "dataorder: position {pos} out of range");
        let (s, i) = get(&self.root, pos as u64, self.ctx);
        (&self.sources[s as usize], i as usize)
    }

    /// Iterates positions `range` in order; `iter(a..b)` yields exactly the elements
    /// `get(a)..get(b)`.
    ///
    /// # Panics
    /// If `range.end > len()` or `range.start > range.end`.
    pub fn iter(&self, range: Range<usize>) -> Cursor<'_, T> {
        assert!(range.start <= range.end, "dataorder: invalid range");
        assert!(range.end <= self.len(), "dataorder: range end {} out of range", range.end);
        Cursor::new(self, range)
    }

    /// Iterates from `start` to the end.
    pub fn iter_from(&self, start: usize) -> Cursor<'_, T> {
        self.iter(start..self.len())
    }
}

/// The element at `pos` of `node` in context `ctx`, as `(source index, index in it)`.
pub(crate) fn get(mut node: &Node, mut pos: u64, mut ctx: u64) -> (u32, u64) {
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
                let (s, j) = il.iter(pos..pos + 1).next().expect("dataorder: interleave yielded nothing");
                pos = j;
                node = &children[s];
            }
            Node::Shuffle { seed, shape, child } => {
                pos = perm::permute(*shape, perm::key(*seed, ctx), pos);
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
}

impl<T: Dataset> Compiler<T> {
    /// `depth` is the number of repeats above `seq`.
    fn compile(&mut self, seq: Seq<T>, depth: u32) -> Result<Node, Error> {
        Ok(match seq {
            Seq::Source(source) => {
                let len = source.len() as u64;
                let src = u32::try_from(self.sources.len()).map_err(|_| Error::Overflow)?;
                self.sources.push(source);
                if len == 0 {
                    Node::Empty
                } else {
                    Node::Source { src, offset: 0, len }
                }
            }
            Seq::Concat(parts) => {
                // Flatten nested concatenations and drop empty parts: both keep the order
                // and the context of every element.
                let mut children = Vec::new();
                for part in parts {
                    match self.compile(part, depth)? {
                        Node::Empty => {}
                        Node::Concat { children: inner, .. } => children.extend(inner),
                        node => children.push(node),
                    }
                }
                match children.len() {
                    0 => Node::Empty,
                    1 => children.pop().unwrap(),
                    _ => {
                        let mut offsets = Vec::with_capacity(children.len() + 1);
                        let mut total = 0u64;
                        for child in &children {
                            offsets.push(total);
                            total = total.checked_add(child.len()).ok_or(Error::Overflow)?;
                        }
                        offsets.push(total);
                        Node::Concat { offsets, children }
                    }
                }
            }
            Seq::Mix(parts) => {
                let sampling: Vec<Sampling> = parts.iter().map(|(_, s)| *s).collect();
                let children = parts.into_iter().map(|(p, _)| self.compile(p, depth)).collect::<Result<Vec<_>, _>>()?;
                let lens: Vec<u64> = children.iter().map(Node::len).collect();
                // Validates the schedules even when the mix folds away.
                let il = Interleave::with_sampling(&lens, &sampling)?;
                let mut nonempty = children.iter().filter(|c| c.len() > 0);
                match (nonempty.next(), nonempty.next()) {
                    (None, _) => Node::Empty,
                    // A single sequence is interleaved with nothing: it stays in order.
                    (Some(_), None) => children.into_iter().find(|c| c.len() > 0).unwrap(),
                    _ => Node::Mix { il, children },
                }
            }
            Seq::Shuffle { seed, inner } => {
                let child = self.compile(*inner, depth)?;
                if child.len() <= 1 {
                    child
                } else {
                    Node::Shuffle { seed, shape: Shape::new(child.len()), child: Box::new(child) }
                }
            }
            Seq::Repeat { times, inner } => {
                let child = self.compile(*inner, depth + 1)?;
                let child_len = child.len();
                let times = times as u64;
                if times.checked_mul(child_len).ok_or(Error::Overflow)? == 0 {
                    Node::Empty
                } else if times == 1 {
                    // The first repetition keeps its context: a single one is the sequence.
                    child
                } else {
                    Node::Repeat { times, child_len, depth, child: Box::new(child) }
                }
            }
            Seq::Slice { start, end, inner } => {
                let child = self.compile(*inner, depth)?;
                let len = child.len();
                let (start, end) = (start as u64, end.map_or(len, |e| e as u64));
                if start > end || end > len {
                    let clamp = |x: u64| usize::try_from(x).unwrap_or(usize::MAX);
                    return Err(Error::SliceOutOfRange { start: clamp(start), end: clamp(end), len: clamp(len) });
                }
                slice(child, start, end - start)
            }
            Seq::Stride { step, offset, inner } => {
                if step == 0 {
                    return Err(Error::ZeroStep);
                }
                let child = self.compile(*inner, depth)?;
                let (step, offset) = (step as u64, offset as u64);
                let n = child.len();
                let len = if offset >= n { 0 } else { (n - offset - 1) / step + 1 };
                if len == 0 {
                    Node::Empty
                } else if step == 1 {
                    slice(child, offset, len)
                } else {
                    Node::Stride { step, offset, len, child: Box::new(child) }
                }
            }
        })
    }
}

/// Positions `start..start + len` of `child`, folded into the child where that is exact.
fn slice(child: Node, start: u64, len: u64) -> Node {
    if len == 0 {
        return Node::Empty;
    }
    if start == 0 && len == child.len() {
        return child;
    }
    match child {
        Node::Source { src, offset, .. } => Node::Source { src, offset: offset + start, len },
        Node::Slice { start: inner, child, .. } => Node::Slice { start: inner + start, len, child },
        Node::Stride { step, offset, child, .. } => Node::Stride { step, offset: offset + start * step, len, child },
        child => Node::Slice { start, len, child: Box::new(child) },
    }
}
