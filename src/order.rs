//! The compiled order: a tree of nodes with precomputed lengths, prefix sums, interleave
//! indices and shuffle shapes, and random access by a stateless descent ([`get`]).

use crate::cursor::Cursor;
use crate::interleave::{Interleave, MAX_TOTAL_LEN, Sampling};
use crate::perm::{self, Shape};
use crate::seq::WeightedPart;
use crate::{Error, ErrorKind, MAX_DEPTH, Seq, Source};
use std::ops::{Bound, RangeBounds};

/// A compiled node. Empty subtrees are folded to [`Node::Empty`], so every child of a
/// `Concat` and every child of a transform has elements. A `Mix` keeps its empty children
/// so that part indices stay those of the configuration; they take no part in the
/// interleave.
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
    /// `salt` folds the salts and lengths of the sources under it (see [`perm::shuffle_salt`]).
    Shuffle {
        seed: u64,
        salt: u64,
        shape: Shape,
        child: Box<Self>,
    },
    /// `depth` counts the repeats above this one; it salts the epoch contexts.
    Repeat {
        times: u64,
        child_len: u64,
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
            Self::Source { len, .. } | Self::Slice { len, .. } | Self::Stride { len, .. } => *len,
            Self::Concat { offsets, .. } => *offsets.last().unwrap(),
            Self::Mix { il, .. } => il.len(),
            Self::Shuffle { shape, .. } => shape.n,
            Self::Repeat { times, child_len, .. } => times * child_len,
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

impl<T: Source> Order<T> {
    /// The order of `seq` with seed 0: validates it and precomputes what iteration needs.
    /// Consumes it; clone it first to keep it, or validate it with [`Seq::check`] first.
    ///
    /// # Errors
    /// Skips and takes past the end, a zero stride, lengths that overflow, nesting deeper
    /// than [`MAX_DEPTH`], and schedules or weights the mix rejects; see [`ErrorKind`]. The
    /// error names the node it was found at.
    pub fn new(seq: Seq<T>) -> Result<Self, Error> {
        Self::with_seed(seq, 0)
    }

    /// The order of `seq` with the given `seed`, which reseeds every shuffle in it at once;
    /// shuffles keep their relative distinctness from their own seeds.
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

    /// The seed given to [`Order::with_seed`] (0 for [`Order::new`]).
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.ctx
    }

    /// The sources, in order of appearance in the configuration.
    #[must_use]
    pub fn sources(&self) -> &[T] {
        &self.sources
    }

    /// The sources, in order of appearance in the configuration, consuming the order.
    #[must_use]
    pub fn into_sources(self) -> Vec<T> {
        self.sources
    }

    /// The element at `pos`: the source and the index in it.
    ///
    /// Constant work per node on the path, except that a [`Seq::Mix`] on the path costs a
    /// seek of the interleave (`O(k log s)` for `k` parts, `s` scheduled), which allocates.
    /// For consecutive positions use [`Order::iter`]; for many scattered positions, a cursor
    /// and [`Cursor::seek`], which reuses the cursor's allocations.
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
    /// If `pos >= len()`.
    #[must_use]
    pub fn get(&self, pos: usize) -> (&T, usize) {
        assert!(pos < self.len(), "dataorder: position {pos} out of range");
        let (s, i) = get(&self.root, pos as u64, self.ctx);
        (&self.sources[s as usize], i as usize)
    }

    /// Iterates the positions in `range` in order; `iter(a..b)` yields exactly the elements
    /// `get(a)..get(b)`, and `iter(..)` the whole order (as does `&order` in a `for` loop).
    ///
    /// Building the cursor allocates one cursor per node on the active path (a mix builds
    /// its parts' cursors as they are first drawn from) and then seeks, which for a mix
    /// counts the elements before the start in each part. Walking is then a few nanoseconds
    /// per element, so make cursors for long ranges rather than many short ones, and
    /// [`seek`](Cursor::seek) a cursor rather than making a new one.
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
        let len = self.len();
        let start = match range.start_bound() {
            Bound::Included(&s) => s,
            Bound::Excluded(&s) => s.checked_add(1).expect("dataorder: range start overflows usize"),
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&e) => e.checked_add(1).expect("dataorder: range end overflows usize"),
            Bound::Excluded(&e) => e,
            Bound::Unbounded => len,
        };
        assert!(start <= end, "dataorder: range {start}..{end} ends before it starts");
        assert!(end <= len, "dataorder: range end {end} out of range for {len} positions");
        Cursor::new(self, start..end)
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
    /// Salt and length of every source, in order, for the salts of the shuffles above them.
    salts: Vec<(u64, u64)>,
    /// Child indices from the root to the node being compiled, for error reports.
    path: Vec<usize>,
}

impl<T: Source> Compiler<T> {
    fn err(&self, kind: ErrorKind) -> Error {
        Error::new(kind, self.path.clone())
    }

    /// Compiles child `i` of the node being compiled, `repeats` repeats deep, one nesting
    /// level below it.
    fn child(&mut self, i: usize, seq: Seq<T>, repeats: u32, level: u32) -> Result<Node, Error> {
        self.path.push(i);
        let node = self.compile(seq, repeats, level + 1);
        self.path.pop();
        node
    }

    /// Compiles the children of a wide node in order. On an error, the parts not yet
    /// compiled are taken apart without recursion, so that a rejected configuration of any
    /// depth is dropped on the heap rather than the stack.
    fn children(&mut self, parts: impl IntoIterator<Item = Seq<T>>, repeats: u32, level: u32) -> Result<Vec<Node>, Error> {
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
    /// recursion (see [`Seq::dismantle`]).
    fn compile(&mut self, seq: Seq<T>, repeats: u32, level: u32) -> Result<Node, Error> {
        if level > MAX_DEPTH {
            seq.dismantle();
            return Err(self.err(ErrorKind::TooDeep));
        }
        Ok(match seq {
            Seq::Source(source) => {
                let len = source.len() as u64;
                let src = u32::try_from(self.sources.len()).map_err(|_| self.err(ErrorKind::TooManySources))?;
                self.salts.push((source.salt(), len));
                self.sources.push(source);
                if len == 0 { Node::Empty } else { Node::Source { src, offset: 0, len } }
            }
            Seq::Concat(parts) => {
                // Flatten nested concatenations and drop empty parts: both keep the order
                // and the context of every element.
                let mut children = Vec::new();
                for node in self.children(parts, repeats, level)? {
                    match node {
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
                            total = total.checked_add(child.len()).ok_or_else(|| self.err(ErrorKind::LengthOverflow))?;
                        }
                        offsets.push(total);
                        Node::Concat { offsets, children }
                    }
                }
            }
            Seq::Mix(parts) => {
                let sampling: Vec<Sampling> = parts.iter().map(|p| p.sampling).collect();
                let children = self.children(parts.into_iter().map(|p| p.seq), repeats, level)?;
                self.mix(children, &sampling)?
            }
            Seq::Weighted { total, parts } => self.weighted(total, parts, repeats, level)?,
            Seq::Shuffle { seed, inner } => {
                let first = self.salts.len();
                let child = self.child(0, *inner, repeats, level)?;
                if child.len() <= 1 {
                    child
                } else {
                    let salt = perm::shuffle_salt(self.salts[first..].iter().copied());
                    Node::Shuffle { seed, salt, shape: Shape::new(child.len()), child: Box::new(child) }
                }
            }
            Seq::Repeat { times, inner } => {
                // A single repetition is the sequence itself, so it does not count as a repeat
                // above its child either.
                let child = self.child(0, *inner, if times > 1 { repeats + 1 } else { repeats }, level)?;
                let child_len = child.len();
                let times = times as u64;
                if times.checked_mul(child_len).ok_or_else(|| self.err(ErrorKind::LengthOverflow))? == 0 {
                    Node::Empty
                } else if times == 1 {
                    child
                } else {
                    Node::Repeat { times, child_len, depth: repeats, child: Box::new(child) }
                }
            }
            Seq::Skip { n, inner } => {
                let child = self.child(0, *inner, repeats, level)?;
                let len = child.len();
                if n as u64 > len {
                    return Err(self.err(ErrorKind::SkipOutOfRange { n, len }));
                }
                slice(child, n as u64, len - n as u64)
            }
            Seq::Take { n, inner } => {
                let child = self.child(0, *inner, repeats, level)?;
                let len = child.len();
                if n as u64 > len {
                    return Err(self.err(ErrorKind::TakeOutOfRange { n, len }));
                }
                slice(child, 0, n as u64)
            }
            Seq::Stride { step, offset, inner } => {
                if step == 0 {
                    inner.dismantle();
                    return Err(self.err(ErrorKind::ZeroStep));
                }
                let child = self.child(0, *inner, repeats, level)?;
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

    /// The mix of compiled `children` with their schedules; validates the schedules even
    /// when the mix folds away.
    fn mix(&self, children: Vec<Node>, sampling: &[Sampling]) -> Result<Node, Error> {
        // The tournament tree indexes parts with u32 and needs a spare bit.
        if children.len() >= u32::MAX as usize / 2 {
            return Err(self.err(ErrorKind::TooManyMixParts));
        }
        let lens: Vec<u64> = children.iter().map(Node::len).collect();
        let il = Interleave::with_sampling(&lens, sampling).map_err(|e| self.err(e.into()))?;
        let mut nonempty = children.iter().filter(|c| c.len() > 0);
        Ok(match (nonempty.next(), nonempty.next()) {
            (None, _) => Node::Empty,
            // A single sequence is interleaved with nothing: it stays in order.
            (Some(_), None) => children.into_iter().find(|c| c.len() > 0).unwrap(),
            _ => Node::Mix { il, children },
        })
    }

    /// A weighted mix: every part repeated as often as its share needs and cut to it, then
    /// mixed. A part is compiled once; when it turns out to need repeating, the repeats
    /// inside it move one level deeper after the fact.
    fn weighted(&mut self, total: usize, parts: Vec<WeightedPart<T>>, repeats: u32, level: u32) -> Result<Node, Error> {
        let weights: Vec<f64> = parts.iter().map(|p| p.weight).collect();
        let sampling: Vec<Sampling> = parts.iter().map(|p| p.sampling).collect();
        let shares = match weighted_shares(total as u64, &weights) {
            Ok(shares) => shares,
            Err(kind) => {
                parts.into_iter().for_each(|p| p.seq.dismantle());
                return Err(self.err(kind));
            }
        };
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

    /// Part `i` of a weighted mix, repeated as often as its `share` needs and cut to it.
    fn weighted_part(&mut self, i: usize, seq: Seq<T>, share: u64, repeats: u32, level: u32) -> Result<Node, Error> {
        let mut child = self.child(i, seq, repeats, level)?;
        let len = child.len();
        if len == 0 && share > 0 {
            return Err(self.err(ErrorKind::EmptyWeightedPart { part: i }));
        }
        if share == 0 {
            return Ok(Node::Empty);
        }
        let times = share.div_ceil(len);
        times.checked_mul(len).ok_or_else(|| self.err(ErrorKind::LengthOverflow))?;
        let repeated = if times > 1 {
            deepen(&mut child);
            Node::Repeat { times, child_len: len, depth: repeats, child: Box::new(child) }
        } else {
            child
        };
        Ok(slice(repeated, 0, share))
    }
}

/// The parts' element counts for the given weights, summing to `total`: the floors of the
/// exact shares `wᵢ / Σw · total`, the remainder going one each to the parts with the largest
/// fractional shares (lowest index first on ties). Rounding could make the floors sum to
/// one more than `total` in contrived cases; then the parts with the smallest fractions
/// give one back. A total beyond [`MAX_TOTAL_LEN`] is rejected first, so that `total` and
/// every share are exact in `f64` and the fix-ups above suffice.
pub(crate) fn weighted_shares(total: u64, weights: &[f64]) -> Result<Vec<u64>, ErrorKind> {
    for (i, &w) in weights.iter().enumerate() {
        if !(w.is_finite() && w >= 0.0) {
            return Err(ErrorKind::InvalidWeight { part: i, weight: w });
        }
    }
    if total > MAX_TOTAL_LEN {
        return Err(ErrorKind::MixTooLong);
    }
    if total == 0 {
        return Ok(vec![0; weights.len()]);
    }
    // Finite weights can still sum to infinity. Then they are scaled by a power of two first,
    // which is exact for every weight large enough to get a share and leaves every quotient
    // as it would be without overflow.
    let scale = if weights.iter().sum::<f64>().is_finite() { 1.0 } else { f64::from_bits((1023 - 600) << 52) };
    let sum: f64 = weights.iter().map(|w| w * scale).sum();
    // The scaled weights are finite and nonnegative, so the sum is too (no NaN).
    if sum <= 0.0 {
        return Err(ErrorKind::ZeroWeights);
    }
    let mut shares = Vec::with_capacity(weights.len());
    let mut fractions = Vec::with_capacity(weights.len());
    let mut given = 0u64;
    for (i, &w) in weights.iter().enumerate() {
        let exact = w * scale / sum * total as f64;
        let floor = exact.floor();
        let share = (floor as u64).min(total);
        shares.push(share);
        given += share;
        fractions.push((exact - floor, i));
    }
    // Descending fraction, ascending index; `partial_cmp` is total here (no NaN).
    fractions.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    for &(_, i) in &fractions {
        if given >= total {
            break;
        }
        shares[i] += 1;
        given += 1;
    }
    for &(_, i) in fractions.iter().rev() {
        if given <= total {
            break;
        }
        if shares[i] > 0 {
            shares[i] -= 1;
            given -= 1;
        }
    }
    debug_assert_eq!(given, total);
    Ok(shares)
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
