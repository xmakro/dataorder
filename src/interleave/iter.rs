//! Iteration over a range of the joint sequence. [`Iter::seek`] positions at the range's
//! start; [`advance`] then takes the next element from a tournament tree over the sequences'
//! heads. The `#[inline(always)]` attributes are measured: they keep the whole step inside
//! the cursor's mix step, which is worth about 3 ns per element.

use super::Interleave;
use super::tournament::TournamentTree;
use std::ops::Range;

/// Iterator returned by [`Interleave::iter`]; yields `(sequence, index_in_sequence)`. It
/// can be [seeked](Iter::seek) again and reuses its buffers when it is.
#[derive(Clone, Debug)]
pub(crate) struct Iter<'a> {
    il: &'a Interleave,
    tree: TournamentTree<Slot>,
    /// Scratch for the seek's per-sequence counts, kept so that a seek allocates nothing.
    counts: Vec<u64>,
    remaining: u64,
}

impl<'a> Iter<'a> {
    /// Seeks to `range.start` (the range is already validated) and builds the tree of heads.
    pub(crate) fn new(il: &'a Interleave, range: Range<u64>) -> Self {
        let mut iter = Iter { il, tree: TournamentTree::empty(), counts: Vec::new(), remaining: 0 };
        iter.seek(range);
        iter
    }

    /// Repositions at `range` (already validated), reusing every allocation: the seek
    /// counts, per sequence, the elements before the start, rebuilds the tree over the heads
    /// and steps through the few positions the counts fell short by.
    pub(crate) fn seek(&mut self, range: Range<u64>) {
        let (a, remaining) = (range.start, range.end - range.start);
        self.remaining = remaining;
        if remaining == 0 {
            return;
        }
        let il = self.il;
        let base = counts_below(il, a, &mut self.counts);
        let counts = &self.counts;
        // Leaves in sequence order, so that ties go to the lower sequence index.
        self.tree.rebuild(counts.iter().enumerate().filter(|&(s, &c)| c < il.seqs[s].n).map(|(s, &c)| {
            let mut seg = 0;
            let key = il.key(s, c, &mut seg);
            (key, slot(il, s, c, key, seg))
        }));
        for _ in base..a {
            advance(il, &mut self.tree);
        }
    }

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
        usize::try_from(self.remaining).map_or((usize::MAX, None), |n| (n, Some(n)))
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
    /// A `u32` here keeps the slot at 24 bytes; the hint is widened to `usize` for the key
    /// computation and narrowed again, which is free, whereas a `u32` hint throughout was
    /// measured 0.6 ns per element slower on mixes of 100 parts (2.3 ns on nested ones).
    seg: u32,
    /// Key of the element after this one, computed ahead of time so that replacing this
    /// element in the tree does not wait for the arithmetic; NaN for the last element
    /// (keys are finite), which keeps the slot at 24 bytes.
    next_key: f64,
}

/// Per sequence, how many of its elements lie among the first `a` of the joint sequence,
/// except that the counts may fall short by a few elements in total (never overshoot).
/// Writes the counts into `counts` and returns their sum.
///
/// Try the analytic progress and one correction first. If rounding of a profile makes
/// those guesses poor, bisect progress instead; its nonnegative float bits are ordered,
/// so at most 63 passes suffice. Never replay more than twice the number of parts.
fn counts_below(il: &Interleave, a: u64, counts: &mut Vec<u64>) -> u64 {
    let k = il.seqs.len() as u64;
    let count = |t, counts: &mut Vec<u64>| {
        counts.clear();
        counts.extend((0..il.seqs.len()).map(|s| count_below(il, s, t)));
        counts.iter().sum::<u64>()
    };
    let mut t = a as f64 / il.total as f64;
    let (mut lo, mut hi) = (0.0f64, 1.0f64.next_up());
    for attempt in 0..2 {
        let base = count(t, counts);
        if base <= a && a - base <= 2 * k {
            return base;
        }
        if base > a {
            hi = t
        } else {
            lo = t
        }
        if attempt == 0 {
            t = ((a as f64 - (base as f64 - a as f64) - k as f64) / il.total as f64).clamp(lo, hi);
        }
    }
    loop {
        if hi.to_bits() - lo.to_bits() <= 1 {
            // The remaining rank lies among equal keys at lo. Consume that tie in part
            // order, by counts, even if clamping produced a long run of equal keys.
            let mut left = a - count(lo, counts);
            for (s, c) in counts.iter_mut().enumerate() {
                let take = left.min(count_below(il, s, hi) - *c);
                *c += take;
                left -= take;
                if left == 0 {
                    return a;
                }
            }
            unreachable!("interleave: rank not bracketed");
        }
        t = f64::from_bits(lo.to_bits() + (hi.to_bits() - lo.to_bits()) / 2);
        let base = count(t, counts);
        if base <= a && a - base <= 2 * k {
            return base;
        }
        if base > a { hi = t } else { lo = t }
    }
}

/// Number of elements of `seq` whose key is below progress `t`.
fn count_below(il: &Interleave, seq: usize, t: f64) -> u64 {
    let s = &il.seqs[seq];
    // Guess from the share function, then make it exact against the real keys.
    let guess = s.n as f64 * il.profile(seq).share(t) - s.phi;
    let mut c = (guess.ceil().max(0.0) as u64).min(s.n);
    let mut seg = 0;
    for _ in 0..4 {
        if c < s.n && il.key(seq, c, &mut seg) < t {
            c += 1;
        } else if c > 0 && il.key(seq, c - 1, &mut seg) >= t {
            c -= 1;
        } else {
            return c;
        }
    }
    // A poorly conditioned inverse or accumulated rounding must not cost O(n).
    let (mut lo, mut hi) = (0, s.n);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if il.key(seq, mid, &mut seg) < t { lo = mid + 1 } else { hi = mid }
    }
    lo
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
