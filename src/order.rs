//! Validation, compilation and random access.
//!
//! The compiler separates source handles from the configuration, then builds a tree
//! with lengths, concat offsets, interleave profiles and shuffle salts. [`get`]
//! follows that tree to resolve a position without keeping iteration state.

use crate::cursor::{Cursor, resolve_range};
use crate::interleave::{Interleave, Schedule};
use crate::perm::{self, Shape};
use crate::seq::MixPart;
use crate::{BoundsError, Error, ErrorKind, MAX_MIX_LEN, Seq, Source};
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
        src: usize,
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
    /// `child` repeated: positions `0..len`, `child_len` per repetition, the last one cut
    /// short when `len` is not a multiple (a cycle). A shuffle is one shuffled pass.
    Repeat {
        child_len: usize,
        len: usize,
        /// The input configuration's salt, computed before pruning, when this node
        /// shuffles each pass; nested shuffles stay fixed.
        shuffle: Option<u64>,
        child: Box<Self>,
    },
    /// Child positions `offset + i × step` for `i < len`. A unit step is a plain
    /// selection: a skip and a take.
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
            Self::Source { len, .. } | Self::Repeat { len, .. } | Self::Stride { len, .. } => *len,
            Self::Concat { offsets, .. } => *offsets.last().unwrap(),
            Self::Mix { il, .. } => il.len(),
        }
    }
}

/// A record selected by [`Order::get`] or [`Cursor`].
///
/// The ordinal identifies the source within this order's [`sources`](Order::sources),
/// even when source values are equal or zero-sized. It is local to the order, not a
/// persistent dataset ID. The item borrows its source and is cheap to copy.
///
/// Equality compares `source_ordinal` and `record_index` first, then compares
/// source values only when both match. Source comparison uses `T`'s equality
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
    pub(crate) seed: u64,
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
    /// let order = Order::new(Seq::concat([Seq::source(3), Seq::source(2).shuffle()]))?;
    /// assert_eq!(order.len(), 5);
    /// let err = Order::new(Seq::source(3).skip(4)).unwrap_err();
    /// assert_eq!(err.kind(), &ErrorKind::SkipOutOfRange { n: 4, len: 3 });
    /// # Ok::<(), dataorder::Error>(())
    /// ```
    ///
    /// # Errors
    /// Skips and takes past the end, a zero step, shuffles containing mixes,
    /// lengths that overflow, and schedules the mix rejects; see [`ErrorKind`].
    /// The error identifies the invalid node.
    pub fn new(seq: Seq<T>) -> Result<Self, Error> {
        Self::with_seed(seq, 0)
    }

    /// Validates and compiles `seq` with the given order seed.
    /// This seed is combined with each shuffled input's configuration salt; it does not add shuffling
    /// to unshuffled sequences. Use [`Order::set_seed`] to reseed an existing order
    /// without rebuilding it.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let seq = Seq::source(100).shuffle();
    /// let (a, b) = (Order::with_seed(seq.clone(), 1)?, Order::with_seed(seq, 2)?);
    /// assert!(a.iter().ne(b.iter()));
    /// assert_eq!(b.seed(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    /// As for [`Order::new`].
    pub fn with_seed(seq: Seq<T>, seed: u64) -> Result<Self, Error> {
        let mut c = Compiler { sources: Vec::new(), path: Vec::new(), shuffle_path_len: None };
        let Compiled { node: root, .. } = c.compile(seq)?;
        Ok(Self { root, seed, sources: c.sources })
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
        self.seed
    }

    /// Changes the seed used by all shuffles, without rebuilding the order.
    /// Takes constant time: shuffle keys are derived during access and iteration.
    /// Previously cloned orders keep their own seeds.
    ///
    /// ```
    /// use dataorder::{Order, Seq};
    /// let seq = Seq::source(100).shuffle();
    /// let mut order = Order::new(seq.clone())?;
    /// order.set_seed(7);
    /// let reseeded = Order::with_seed(seq, 7)?;
    /// assert!(order.iter().eq(reseeded.iter()));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
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
    /// let order = Order::new(Seq::concat([Seq::source(3), Seq::source(5).shuffle()]))?;
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
        let (src, index) = get(&self.root, pos, self.seed);
        Some(Item { source_ordinal: src, source: &self.sources[src], record_index: index })
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
    /// let order = Order::new(Seq::source(10).shuffle())?;
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
        f.debug_struct("Order").field("len", &self.len()).field("seed", &self.seed).field("sources", &self.sources).finish()
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

/// Resolve a position to `(source index, record index)` with an unchanged order seed.
pub(crate) fn get(mut node: &Node, mut pos: usize, order_seed: u64) -> (usize, usize) {
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
            Node::Repeat { child_len, len, shuffle, child } => {
                // A single pass, such as a shuffle, never divides. Comparing the lengths
                // rather than the position keeps the compiler from folding this branch
                // back into an unconditional division.
                let pass = if len <= child_len { 0 } else { pos / child_len };
                pos -= pass * child_len;
                if let Some(salt) = shuffle {
                    pos = perm::permute(Shape::new(*child_len), perm::key(order_seed, pass, *salt), pos);
                }
                node = child;
            }
            Node::Stride { step, offset, child, .. } => {
                pos = offset + pos * step;
                node = child;
            }
        }
    }
}

/// Summaries returned by compilation. Salt follows the original
/// configuration; the node and length describe the folded result.
struct Compiled {
    node: Node,
    len: usize,
    salt: perm::ConfigSalt,
}

impl Compiled {
    /// Build a repetition after validating its input and target length.
    /// A positive target length requires a nonempty input. A single plain pass
    /// needs no node; a single shuffled pass over more than one element is a shuffle.
    fn repeat_to(self, len: usize, shuffled: bool) -> Self {
        let Self { node: child, len: child_len, salt } = self;
        let shuffle = (shuffled && child_len > 1).then_some(salt.value);
        let node = if len == 0 || (len <= child_len && shuffle.is_none()) {
            slice(child, 0, len)
        } else {
            Node::Repeat { child_len, len, shuffle, child: Box::new(child) }
        };
        Self { node, len, salt: if shuffled { perm::shuffled_salt(salt) } else { salt } }
    }
}

struct Compiler<T> {
    sources: Vec<T>,
    /// Child indices from the root to the node being compiled, for error reports.
    path: Vec<usize>,
    /// Path length of the nearest enclosing shuffle, for rejecting mix descendants.
    shuffle_path_len: Option<usize>,
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

    /// Compiles child `i` of the node being compiled.
    fn child(&mut self, i: usize, seq: Seq<T>) -> Result<Compiled, Error> {
        self.path.push(i);
        let result = self.compile(seq);
        self.path.pop();
        result
    }

    /// Compile every child before combining their summaries, preserving validation order.
    fn children(&mut self, parts: impl IntoIterator<Item = Seq<T>>) -> Result<Vec<Compiled>, Error> {
        parts.into_iter().enumerate().map(|(i, seq)| self.child(i, seq)).collect()
    }

    /// Each visit returns its node, length and configuration salt to the parent.
    /// Lengths must fit before a parent can truncate them.
    fn compile(&mut self, seq: Seq<T>) -> Result<Compiled, Error> {
        match seq {
            Seq::Source(source) => self.source(source),
            Seq::Concat(parts) => self.concat(parts),
            Seq::Mix(parts) => self.mix(parts),
            Seq::Shuffle { inner } => self.repeat(1, *inner, true),
            Seq::Repeat { times, inner } => self.repeat(times, *inner, false),
            Seq::Cycle { len, inner } => self.cycled(len, *inner, false),
            Seq::ShuffledRepeat { times, inner } => self.repeat(times, *inner, true),
            Seq::ShuffledCycle { len, inner } => self.cycled(len, *inner, true),
            Seq::Skip { n, inner } => self.skip(n, *inner),
            Seq::Take { n, inner } => self.take(n, *inner),
            Seq::StepBy { step, inner } => self.stepped(step, *inner),
        }
    }

    fn source(&mut self, source: T) -> Result<Compiled, Error> {
        let len = source.len();
        let src = self.sources.len();
        let salt = perm::source_salt(source.salt(), len);
        self.sources.push(source);
        let node = if len == 0 { Node::Empty } else { Node::Source { src, offset: 0, len } };
        Ok(Compiled { node, len, salt })
    }

    /// Flatten concats using their existing offsets and the summaries returned by visits.
    fn concat(&mut self, parts: Vec<Seq<T>>) -> Result<Compiled, Error> {
        let parts = self.children(parts)?;
        let salt = perm::combine_salts(parts.iter().map(|part| part.salt));
        let mut children = Vec::new();
        let mut offsets = Vec::new();
        let mut len = 0usize;
        for Compiled { node, len: child_len, .. } in parts {
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
        }
        offsets.push(len);
        let node = match children.len() {
            0 => Node::Empty,
            1 => children.pop().unwrap(),
            _ => Node::Concat { offsets, children },
        };
        Ok(Compiled { node, len, salt })
    }

    /// Compiles a repetition's input. A shuffled input must contain no mix, and is
    /// validated as such even if the repetition or its output folds away.
    fn repeat_child(&mut self, inner: Seq<T>, shuffled: bool) -> Result<Compiled, Error> {
        if !shuffled {
            return self.child(0, inner);
        }
        let enclosing = self.shuffle_path_len.replace(self.path.len());
        let child = self.child(0, inner);
        self.shuffle_path_len = enclosing;
        child
    }

    fn repeat(&mut self, times: usize, inner: Seq<T>, shuffled: bool) -> Result<Compiled, Error> {
        let child = self.repeat_child(inner, shuffled)?;
        let len = times.checked_mul(child.len).ok_or_else(|| self.err(ErrorKind::LengthOverflow))?;
        Ok(child.repeat_to(len, shuffled))
    }

    fn cycled(&mut self, len: usize, inner: Seq<T>, shuffled: bool) -> Result<Compiled, Error> {
        let child = self.repeat_child(inner, shuffled)?;
        if len > 0 && child.len == 0 {
            return Err(self.err(ErrorKind::EmptyCycle));
        }
        Ok(child.repeat_to(len, shuffled))
    }

    fn skip(&mut self, n: usize, inner: Seq<T>) -> Result<Compiled, Error> {
        let Compiled { node, len, salt } = self.child(0, inner)?;
        if n > len {
            return Err(self.err(ErrorKind::SkipOutOfRange { n, len }));
        }
        Ok(Compiled { node: slice(node, n, len - n), len: len - n, salt })
    }

    fn take(&mut self, n: usize, inner: Seq<T>) -> Result<Compiled, Error> {
        let Compiled { node, len, salt } = self.child(0, inner)?;
        if n > len {
            return Err(self.err(ErrorKind::TakeOutOfRange { n, len }));
        }
        Ok(Compiled { node: slice(node, 0, n), len: n, salt })
    }

    fn stepped(&mut self, step: usize, inner: Seq<T>) -> Result<Compiled, Error> {
        if step == 0 {
            return Err(self.err(ErrorKind::ZeroStep));
        }
        let Compiled { node, len, salt } = self.child(0, inner)?;
        Ok(Compiled { node: stride(node, step), len: len.div_ceil(step), salt })
    }

    /// Validates every part's length and schedule, then compiles the mix over its
    /// non-empty parts. Empty parts and a mix that folds away are still validated.
    fn mix(&mut self, parts: Vec<MixPart<T>>) -> Result<Compiled, Error> {
        if let Some(len) = self.shuffle_path_len {
            return Err(Error::new(ErrorKind::ShuffleContainsMix, self.path[..len].to_vec()));
        }
        let schedules: Vec<Schedule> = parts.iter().map(|part| part.schedule).collect();
        let parts = self.children(parts.into_iter().map(|part| part.seq))?;
        // The tournament tree indexes parts with u32 and needs a spare bit.
        if parts.len() >= u32::MAX as usize / 2 {
            return Err(self.err(ErrorKind::TooManyMixParts));
        }
        let salt = perm::combine_salts(parts.iter().map(|part| part.salt));
        let mut len = 0usize;
        for part in &parts {
            len = len.checked_add(part.len).ok_or_else(|| self.err(ErrorKind::LengthOverflow))?;
            if len as u64 > MAX_MIX_LEN {
                return Err(self.err(ErrorKind::MixTooLong));
            }
        }
        let mut live = Vec::new();
        let mut children = Vec::new();
        for (i, (part, schedule)) in parts.into_iter().zip(schedules).enumerate() {
            let profile = schedule.profile(part.len).map_err(|kind| self.err_at(kind, Some(i)))?;
            if part.len > 0 {
                live.push((part.len, profile));
                children.push(part.node);
            }
        }
        children.shrink_to_fit();
        let node = match children.len() {
            0 => Node::Empty,
            // A single sequence is interleaved with nothing: it stays in order.
            1 => children.pop().unwrap(),
            _ => Node::Mix { il: Interleave::new(live), children },
        };
        Ok(Compiled { node, len, salt })
    }
}

/// Fold a selection without changing configuration salts or local shuffle passes.
fn slice(node: Node, start: usize, len: usize) -> Node {
    if len == 0 {
        return Node::Empty;
    }
    if start == 0 && len == node.len() {
        return node;
    }
    match node {
        Node::Empty => unreachable!("dataorder: nonempty slice of an empty sequence"),
        Node::Source { src, offset, .. } => Node::Source { src, offset: offset + start, len },
        Node::Repeat { child_len, shuffle, child, .. } if start == 0 => Node::Repeat { child_len, len, shuffle, child },
        Node::Stride { step, offset, child, .. } => {
            let offset = offset + start * step;
            // A single element or a unit stride is a plain selection of the child, which
            // may fold further.
            if len == 1 || step == 1 { slice(*child, offset, len) } else { Node::Stride { step, offset, len, child } }
        }
        Node::Concat { offsets: mut at, mut children } => {
            at.clear();
            let mut offset = 0;
            children.retain_mut(|child| {
                let base = offset;
                offset += child.len();
                let a = start.max(base);
                let b = (start + len).min(offset);
                if a >= b {
                    return false;
                }
                at.push(a - start);
                let node = std::mem::replace(child, Node::Empty);
                *child = slice(node, a - base, b - a);
                true
            });
            at.push(len);
            if children.len() == 1 { children.pop().unwrap() } else { Node::Concat { offsets: at, children } }
        }
        node => Node::Stride { step: 1, offset: start, len, child: Box::new(node) },
    }
}

/// Fold a stride without changing the selected child positions.
fn stride(child: Node, step: usize) -> Node {
    let len = child.len().div_ceil(step);
    if step == 1 || len <= 1 {
        return slice(child, 0, len);
    }
    match child {
        Node::Stride { step: inner, offset: base, child, .. } => Node::Stride { step: step * inner, offset: base, len, child },
        child => Node::Stride { step, offset: 0, len, child: Box::new(child) },
    }
}
