//! Iteration over a range of the joint sequence. [`Iter::new`] seeks to the range's start;
//! [`advance`] then takes the next element from a tournament tree over the sequences' heads.
//! The `#[inline(always)]` attributes are measured: they keep the whole step inside the
//! cursor's mix step, which is worth about 3 ns per element.

use super::tournament::TournamentTree;
use super::Interleave;
use std::ops::Range;

/// Iterator returned by [`Interleave::iter`]; yields `(sequence, index_in_sequence)`.
#[derive(Clone, Debug)]
pub struct Iter<'a> {
    il: &'a Interleave,
    tree: TournamentTree<Slot>,
    remaining: u64,
}

impl<'a> Iter<'a> {
    /// Seeks to `range.start` (the range is already validated) and builds the tree of heads.
    pub(crate) fn new(il: &'a Interleave, range: Range<u64>) -> Self {
        let (a, remaining) = (range.start, range.end - range.start);
        let mut tree = TournamentTree::new(Vec::new());
        if remaining > 0 {
            let (counts, base) = counts_below(il, a);
            // Leaves in sequence order, so that ties go to the lower sequence index.
            tree = TournamentTree::new(counts.iter().enumerate().filter(|&(s, &c)| c < il.seqs[s].n).map(|(s, &c)| {
                let mut seg = 0;
                let key = il.key(s, c, &mut seg);
                (key, slot(il, s, c, key, seg))
            }));
            for _ in base..a {
                advance(il, &mut tree);
            }
        }
        Iter { il, tree, remaining }
    }
}

impl Iter<'_> {
    /// The next element of an iterator that is not exhausted (checked in debug builds
    /// only): the walk without the `Option`.
    #[inline(always)]
    pub(crate) fn step(&mut self) -> (usize, u64) {
        debug_assert!(self.remaining > 0, "interleave: iterator exhausted");
        self.remaining -= 1;
        advance(self.il, &mut self.tree)
    }
}

impl Iterator for Iter<'_> {
    type Item = (usize, u64);

    #[inline(always)]
    fn next(&mut self) -> Option<(usize, u64)> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        Some(advance(self.il, &mut self.tree))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match usize::try_from(self.remaining) {
            Ok(n) => (n, Some(n)),
            Err(_) => (usize::MAX, None),
        }
    }
}

impl std::iter::FusedIterator for Iter<'_> {}

/// A sequence's next element, as held by the tree alongside its key.
#[derive(Clone, Copy, Debug)]
struct Slot {
    /// The sequence.
    seq: u32,
    /// Index of the element within the sequence.
    j: u64,
    /// Segment of the sequence's rate profile that this element's key came from, the
    /// starting point for locating the next key (see [`Profile::quantile`](super::profile::Profile::quantile)).
    seg: u32,
    /// Key of the element after this one, computed ahead of time so that replacing this
    /// element in the tree does not wait for the arithmetic; NaN for the last element
    /// (keys are finite), which keeps the slot at 24 bytes.
    next_key: f64,
}

/// Per sequence, how many of its elements lie among the first `a` of the joint sequence,
/// except that the counts may fall short by a few elements in total (never overshoot).
/// Returns the counts and their sum.
///
/// The elements below progress `a/N` are within about `k` of `a` in number; if they
/// overshoot, the progress is lowered and the count repeated.
fn counts_below(il: &Interleave, a: u64) -> (Vec<u64>, u64) {
    let k = il.seqs.len() as u64;
    let mut t = a as f64 / il.total as f64;
    loop {
        let counts: Vec<u64> = (0..il.seqs.len()).map(|s| count_below(il, s, t)).collect();
        let base: u64 = counts.iter().sum();
        if base <= a {
            return (counts, base);
        }
        // Overshot: lower the progress by the excess plus a margin of k and count again.
        t = ((a as f64 - (base - a + k) as f64) / il.total as f64).max(0.0);
    }
}

/// Number of elements of `seq` whose key is below progress `t`.
fn count_below(il: &Interleave, seq: usize, t: f64) -> u64 {
    let s = &il.seqs[seq];
    // Guess from the share function, then make it exact against the real keys.
    let guess = s.n as f64 * il.profile(seq).share(t) - s.phi;
    let mut c = (guess.ceil().max(0.0) as u64).min(s.n);
    let mut seg = 0;
    while c < s.n && il.key(seq, c, &mut seg) < t {
        c += 1;
    }
    while c > 0 && il.key(seq, c - 1, &mut seg) >= t {
        c -= 1;
    }
    c
}

/// The slot for element `j` of `seq` (whose key is `key`), with the following element's
/// key already computed.
#[inline(always)]
fn slot(il: &Interleave, seq: usize, j: u64, key: f64, mut seg: usize) -> Slot {
    let next_key = if j + 1 < il.seqs[seq].n {
        let next = il.key(seq, j + 1, &mut seg);
        debug_assert!(next >= key, "interleave: keys of sequence {seq} not monotone at {j}");
        next
    } else {
        f64::NAN
    };
    Slot { seq: seq as u32, j, seg: seg as u32, next_key }
}

/// Takes the tree's minimum and replaces it with the sequence's next element.
///
/// The next element's key was computed one step ahead, so the replay of the tree does not
/// wait for that arithmetic; the key after it is computed off the critical path, in the
/// shadow of the tree walk. Computing it here instead costs about 12 ns per element
/// (27 vs 14 ns at k = 100), because the division and square root then sit on the serial
/// chain of every element.
#[inline(always)]
fn advance(il: &Interleave, tree: &mut TournamentTree<Slot>) -> (usize, u64) {
    let (_, &Slot { seq, j, seg, next_key }) = tree.min().expect("interleave: iterator exhausted");
    if next_key.is_nan() {
        tree.remove_min();
    } else {
        tree.set_min(next_key, slot(il, seq as usize, j + 1, next_key, seg as usize));
    }
    (seq as usize, j)
}
