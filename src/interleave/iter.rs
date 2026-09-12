//! Exact seeks and sequential iteration over a mix.
//! [`Iter::seek`] counts elements before the target, builds the remaining heads and
//! replays a bounded number of steps. [`advance`] emits the next head. [`slot`] and
//! [`advance`] are inlined for the same reason as [`Interleave::key`].

use super::Interleave;
use super::tournament::TournamentTree;
use std::ops::Range;

/// Iterator returned by [`Interleave::iter`]; yields `(part, index_within_part)`.
/// Further calls to [`seek`](Iter::seek) reuse its buffers.
#[derive(Clone, Debug)]
pub(crate) struct Iter<'a> {
    il: &'a Interleave,
    tree: TournamentTree<Slot>,
    /// Scratch for the seek's per-part counts, kept so that a seek allocates nothing.
    counts: Vec<usize>,
    remaining: usize,
}

impl<'a> Iter<'a> {
    /// Seeks to `range.start` (the range is already validated) and builds the tree of heads.
    pub(crate) fn new(il: &'a Interleave, range: Range<usize>) -> Self {
        let mut iter = Iter { il, tree: TournamentTree::empty(), counts: Vec::new(), remaining: 0 };
        iter.seek(range);
        iter
    }

    /// Repositions at an already validated range, reusing allocated buffers.
    /// Counts elements before the start, rebuilds the tournament over the remaining
    /// heads, then replays the small gap left by the counts.
    pub(crate) fn seek(&mut self, range: Range<usize>) {
        let (a, remaining) = (range.start, range.end - range.start);
        self.remaining = remaining;
        if remaining == 0 {
            return;
        }
        let il = self.il;
        let base = counts_below(il, a, &mut self.counts);
        let counts = &self.counts;
        // Leaves in part order, so that ties go to the lower part index.
        self.tree.rebuild(counts.iter().enumerate().filter(|&(s, &c)| c < il.parts[s].n).map(|(s, &c)| {
            let key = il.key(s, c);
            (key, slot(il, s, c, key))
        }));
        for _ in base..a {
            advance(il, &mut self.tree);
        }
    }

    /// The next element of an iterator that is not exhausted (checked in debug builds
    /// only): the walk without the `Option`.
    pub(crate) fn step(&mut self) -> (usize, usize) {
        debug_assert!(self.remaining > 0, "interleave: iterator exhausted");
        self.remaining -= 1;
        advance(self.il, &mut self.tree)
    }
}

impl Iterator for Iter<'_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        Some(advance(self.il, &mut self.tree))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl std::iter::FusedIterator for Iter<'_> {}

/// A part's next element, as held by the tree alongside its key.
#[derive(Clone, Copy, Debug)]
struct Slot {
    /// The part.
    part: usize,
    /// Index of the element within the part.
    j: usize,
    /// Key of the element after this one, computed ahead of time so that replacing this
    /// element in the tree does not wait for the arithmetic; NaN for the last element
    /// (keys are finite), avoiding a separate exhaustion flag.
    next_key: f64,
}

/// Writes per-part prefix counts whose sum is at most `a`, and returns that sum.
/// The counts describe a prefix of the merged order and leave at most `2k`
/// elements to replay, where `k` is the number of parts.
///
/// Interpolate between observed integer ranks for up to eight probes, starting at
/// `a / N`. Then bisect virtual-time bits, which takes at most 63 further probes.
/// Neither path assumes virtual time equals output progress. All rank sums are
/// integer counts; never replay more than twice the number of parts.
fn counts_below(il: &Interleave, a: usize, counts: &mut Vec<usize>) -> usize {
    if a == 0 {
        counts.clear();
        counts.resize(il.parts.len(), 0);
        return 0;
    }
    let k = il.parts.len();
    let count = |t, counts: &mut Vec<usize>| {
        counts.clear();
        counts.extend((0..il.parts.len()).map(|s| count_below(il, s, t)));
        counts.iter().sum::<usize>()
    };
    let mut t = a as f64 / il.total as f64;
    let (mut lo, mut hi) = (0.0f64, 1.0f64.next_up());
    let (mut previous_t, mut previous_count) = (0.0, 0);
    let mut probes = 0;
    loop {
        let base = count(t, counts);
        if base <= a && a - base <= 2 * k {
            return base;
        }
        if base > a {
            hi = t;
        } else {
            lo = t;
        }
        if hi.to_bits() - lo.to_bits() <= 1 {
            // The remaining rank lies among equal keys at lo. Consume that tie in part
            // order, by counts, even if rounding produced a long run of equal keys.
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
        probes += 1;
        let next = if probes < 8 && base != previous_count {
            // A secant through the last two integer ranks estimates the clock
            // position of a short prefix before a. Clamp inside the proven bracket;
            // flat ranks and unsuccessful probes fall back to bounded bisection.
            let fraction = (a.saturating_sub(k) as f64 - base as f64) / (base as f64 - previous_count as f64);
            (t + (t - previous_t) * fraction).clamp(lo.next_up(), hi.next_down())
        } else {
            f64::from_bits(lo.to_bits() + (hi.to_bits() - lo.to_bits()) / 2)
        };
        (previous_t, previous_count) = (t, base);
        t = next;
    }
}

/// Number of elements of `part` whose key is below virtual time `t`.
fn count_below(il: &Interleave, part: usize, t: f64) -> usize {
    let s = &il.parts[part];
    // Guess from the share function, then make it exact against the real keys.
    let guess = s.n as f64 * il.profile(part).share(t) - s.phi;
    let mut c = (guess.ceil().max(0.0) as usize).min(s.n);
    for _ in 0..4 {
        if c < s.n && il.key(part, c) < t {
            c += 1;
        } else if c > 0 && il.key(part, c - 1) >= t {
            c -= 1;
        } else {
            return c;
        }
    }
    // A poorly conditioned inverse or accumulated rounding must not cost O(n).
    let (mut lo, mut hi) = (0, s.n);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if il.key(part, mid) < t { lo = mid + 1 } else { hi = mid }
    }
    lo
}

/// The slot for element `j` of `part` (whose key is `key`), with the following element's
/// key already computed.
#[inline(always)]
fn slot(il: &Interleave, part: usize, j: usize, key: f64) -> Slot {
    let next_key = if j + 1 < il.parts[part].n {
        let next = il.key(part, j + 1);
        debug_assert!(next >= key, "interleave: keys of sequence {part} not monotone at {j}");
        next
    } else {
        f64::NAN
    };
    Slot { part, j, next_key }
}

/// Takes the tree's minimum and replaces it with the part's next element.
///
/// The replacement key was computed one step ahead. The tree can compare it
/// without waiting for division or square root; computing the following key can
/// overlap with the tree update.
#[inline(always)]
fn advance(il: &Interleave, tree: &mut TournamentTree<Slot>) -> (usize, usize) {
    let (_, &Slot { part, j, next_key }) = tree.min().expect("interleave: iterator exhausted");
    if next_key.is_nan() {
        tree.remove_min();
    } else {
        tree.set_min(next_key, slot(il, part, j + 1, next_key));
    }
    (part, j)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Schedule;

    #[test]
    fn returning_to_zero_clears_the_previous_counts() {
        let il = Interleave::new(vec![(10_000, None), (1000, Schedule::until(0.5).profile(1000).unwrap())]);
        let mut iter = il.iter(9000..il.len());
        iter.next();
        iter.seek(0..il.len());
        assert_eq!(iter.take(100).collect::<Vec<_>>(), il.iter(0..100).collect::<Vec<_>>());
    }
}
