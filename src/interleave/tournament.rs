//! A tournament ("loser") tree for k-way merging.
//!
//! Each of `k` leaves holds a key and a value. The root identifies the smallest key
//! in O(1). When that winner changes, it competes against the stored loser at each
//! node on its path to the root. The new winner continues upward; each loser stays
//! behind. This takes `⌈log2 k⌉` comparisons. Once only one live leaf remains, the
//! tree skips comparisons entirely.
//!
//! Ties are broken by leaf index (lower wins). Keys are `f64` and must be finite; a removed
//! leaf is given `+∞` internally so that it loses every match.
//!
//! Explicit inlining keeps the update inside the outer cursor's mix step, avoiding
//! a function call for every element.

use std::hint::select_unpredictable;

/// Key of a removed or padding leaf: it loses every match, so such a leaf never wins while
/// a live one remains.
const REMOVED: f64 = f64::INFINITY;

/// Maps a float to an integer with the same order (sign-magnitude to two's complement), so
/// that matches are integer comparisons.
fn sortable(key: f64) -> u64 {
    let b = key.to_bits();
    if b >> 63 == 1 { !b } else { b | 1 << 63 }
}

/// Inverse of [`sortable`].
fn float(bits: u64) -> f64 {
    f64::from_bits(if bits >> 63 == 1 { bits & !(1 << 63) } else { !bits })
}

/// A tournament tree over `k` leaves with `f64` keys and values of type `V`.
#[derive(Debug)]
pub(crate) struct TournamentTree<V> {
    /// Losers in heap layout, with the overall winner in `nodes[0]`.
    /// For `n` leaves (rounded up to a power of two), internal node `m` has children
    /// `2m` and `2m+1`; leaf `i` starts at `n+i`. Each leaf's key and index end up
    /// in exactly one node. Padding leaves use `+∞` and cannot beat a live leaf.
    /// Equal leaf depths give updates a fixed `log2 n` loop length, making the
    /// exit branch predictable.
    nodes: Vec<Entry>,
    values: Vec<V>,
    live: usize,
    /// The winners of the last rebuild, kept so that a rebuild allocates nothing.
    scratch: Vec<Entry>,
}

impl<V: Clone> Clone for TournamentTree<V> {
    fn clone(&self) -> Self {
        // A tree positioned near the end still reserves for every part a backward seek
        // can revive. Vec::clone keeps only the length, losing that allocation guarantee.
        let mut nodes = Vec::with_capacity(self.nodes.capacity());
        nodes.extend_from_slice(&self.nodes);
        let mut values = Vec::with_capacity(self.values.capacity());
        values.extend_from_slice(&self.values);
        Self { nodes, values, live: self.live, scratch: Vec::with_capacity(self.scratch.capacity()) }
    }
}

/// A leaf's key (in [`sortable`] form) and index, as stored in the nodes.
#[derive(Clone, Copy, Debug)]
struct Entry {
    key: u64,
    leaf: u32,
}

impl Entry {
    #[inline]
    fn new(key: f64, leaf: u32) -> Self {
        Self { key: sortable(key), leaf }
    }

    #[inline]
    fn key(self) -> f64 {
        float(self.key)
    }

    /// Does this leaf win against `o`? Smaller key, then lower leaf index. Written with
    /// bitwise operators: a short-circuiting `||` would compile to a branch on the key
    /// comparison, which is unpredictable in a merge.
    #[inline(always)]
    fn beats(self, o: Self) -> bool {
        (self.key < o.key) | ((self.key == o.key) & (self.leaf < o.leaf))
    }
}

impl<V> TournamentTree<V> {
    /// A tree without leaves, allocating nothing; [`rebuild`](TournamentTree::rebuild) fills it.
    pub(crate) fn empty() -> Self {
        Self { nodes: Vec::new(), values: Vec::new(), live: 0, scratch: Vec::new() }
    }

    /// Builds the tree from the leaves' initial keys and values (leaf `i` = `leaves[i]`).
    #[cfg(test)]
    pub(crate) fn new(leaves: impl IntoIterator<Item = (f64, V)>) -> Self {
        let mut tree = Self::empty();
        tree.rebuild(leaves);
        tree
    }

    /// Replaces the tree by one over new leaves, reusing every allocation.
    ///
    /// # Panics
    /// If there are `u32::MAX / 2` leaves or more.
    pub(crate) fn rebuild(&mut self, leaves: impl IntoIterator<Item = (f64, V)>) {
        // `winner[m]` is the winner of the subtree at node `m`: leaves `n..n+k` hold the
        // keys, `n+k..2n` are the padding, and the internal nodes are filled bottom-up.
        let leaves = leaves.into_iter();
        let at_most = leaves.size_hint().1.unwrap_or(0);
        let winner = &mut self.scratch;
        winner.clear();
        winner.reserve(2 * at_most.next_power_of_two());
        self.values.clear();
        self.values.reserve(at_most);
        for (key, value) in leaves {
            winner.push(Entry::new(key, winner.len() as u32));
            self.values.push(value);
        }
        let k = winner.len();
        assert!(k < (u32::MAX / 2) as usize, "tournament tree: too many leaves");
        let n = k.next_power_of_two();
        winner.resize(2 * n, Entry::new(REMOVED, 0));
        // Move the leaves to their nodes, from the back so that nothing is overwritten
        // before it is read (`n + i ≥ k > j` for every leaf `j` still to be moved).
        for i in (0..k).rev() {
            winner[n + i] = winner[i];
        }
        for i in k..n {
            winner[n + i] = Entry::new(REMOVED, i as u32);
        }
        self.nodes.clear();
        // A cursor first positioned near the end may have only one live leaf. Reserve
        // for the iterator's full upper bound so seeking backward does not grow this
        // buffer when the earlier parts become live again, just as for values/scratch.
        self.nodes.reserve(at_most.next_power_of_two());
        self.nodes.resize(n, Entry::new(REMOVED, 0));
        for m in (1..n).rev() {
            let (a, b) = (winner[2 * m], winner[2 * m + 1]);
            let (w, l) = if b.beats(a) { (b, a) } else { (a, b) };
            winner[m] = w;
            self.nodes[m] = l;
        }
        self.nodes[0] = winner[1];
        self.live = k;
    }

    /// Number of leaves not yet removed.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.live
    }

    /// `true` when every leaf has been removed.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Key and value of the leaf with the smallest key.
    #[inline(always)]
    pub(crate) fn min(&self) -> Option<(f64, &V)> {
        if self.live == 0 {
            return None;
        }
        let w = self.nodes[0];
        Some((w.key(), &self.values[w.leaf as usize]))
    }

    /// Gives the winning leaf a new key and value and replays its path.
    ///
    /// # Panics
    /// If the tree is empty.
    #[inline(always)]
    pub(crate) fn set_min(&mut self, key: f64, value: V) {
        assert!(self.live > 0, "tournament tree: empty");
        let leaf = self.nodes[0].leaf;
        self.values[leaf as usize] = value;
        self.replay(Entry::new(key, leaf));
    }

    /// Removes the winning leaf.
    ///
    /// # Panics
    /// If the tree is empty.
    #[inline(always)]
    pub(crate) fn remove_min(&mut self) {
        assert!(self.live > 0, "tournament tree: empty");
        let leaf = self.nodes[0].leaf;
        self.live -= 1;
        self.replay(Entry::new(REMOVED, leaf));
        if self.live == 1 {
            self.keep_survivor();
        }
    }

    /// After the penultimate removal the remaining leaf needs no comparisons. Compact
    /// once here, instead of testing the live count on every ordinary replacement.
    #[cold]
    fn keep_survivor(&mut self) {
        let leaf = self.nodes[0].leaf as usize;
        self.values.swap(0, leaf);
        self.values.truncate(1);
        self.nodes.truncate(1);
        self.nodes[0].leaf = 0;
    }

    /// Plays the winner's leaf, now carrying `cand`, up to the root against the stored
    /// losers: the winner of each match moves on, the loser stays in the node.
    #[inline(always)]
    fn replay(&mut self, mut cand: Entry) {
        let n = self.nodes.len();
        let mut m = (n + cand.leaf as usize) / 2;
        while m >= 1 {
            let stored = self.nodes[m];
            let (loser, winner) = trade(stored.beats(cand), cand, stored);
            self.nodes[m] = loser;
            cand = winner;
            m /= 2;
        }
        self.nodes[0] = cand;
    }
}

/// Returns `(loser, winner)` for the candidate and the stored loser.
/// Written to encourage conditional moves: the outcome is unpredictable in a merge,
/// so a branch is expensive, while masked swaps add a longer dependency chain.
#[inline(always)]
fn trade(stored_wins: bool, cand: Entry, stored: Entry) -> (Entry, Entry) {
    (select_unpredictable(stored_wins, cand, stored), select_unpredictable(stored_wins, stored, cand))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::Rng;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    /// Few distinct values, so that ties are common.
    fn random_key(rng: &mut Rng) -> f64 {
        (rng.next() % 50) as f64 / 7.0
    }

    /// `(key, leaf)` with a total order matching the tree's tie-break.
    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    struct K(u64, u32);
    fn k(key: f64, leaf: u32) -> K {
        K(key.to_bits(), leaf) // keys are non-negative, so bit order = numeric order
    }

    #[test]
    fn matches_binary_heap() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        let mut tree = TournamentTree::new(Vec::new());
        for &n in &[1usize, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 100, 1000] {
            for _ in 0..(if n < 20 { 50 } else { 3 }) {
                let keys: Vec<f64> = (0..n).map(|_| random_key(&mut rng)).collect();
                // Alternately a fresh tree and a rebuilt one.
                if rng.next().is_multiple_of(2) {
                    tree = TournamentTree::new(keys.iter().enumerate().map(|(i, &key)| (key, i as u32)));
                } else {
                    tree.rebuild(keys.iter().enumerate().map(|(i, &key)| (key, i as u32)));
                }
                let mut heap: BinaryHeap<Reverse<K>> = keys.iter().enumerate().map(|(i, &key)| Reverse(k(key, i as u32))).collect();
                let mut steps = 0;
                while let Some(Reverse(K(bits, leaf))) = heap.pop() {
                    let (key, &value) = tree.min().expect("tree empty too early");
                    assert_eq!((key.to_bits(), value), (bits, leaf), "n {n} step {steps}");
                    assert_eq!(tree.len(), heap.len() + 1);
                    if rng.next().is_multiple_of(3) {
                        tree.remove_min();
                    } else {
                        let new = random_key(&mut rng);
                        tree.set_min(new, value);
                        heap.push(Reverse(k(new, leaf)));
                    }
                    steps += 1;
                }
                assert!(tree.is_empty());
                assert!(tree.min().is_none());
            }
        }
    }

    #[test]
    fn empty_tree() {
        let tree: TournamentTree<u32> = TournamentTree::new(Vec::new());
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert!(tree.min().is_none());
    }

    #[test]
    fn last_survivor_has_no_tournament_path() {
        let mut tree = TournamentTree::new((0..100).map(|i| (f64::from(i), i)));
        for i in 0..99 {
            assert_eq!(tree.min(), Some((f64::from(i), &i)));
            tree.remove_min();
        }
        // A one-leaf tree bounds all subsequent replacements to constant work, even
        // when this leaf originally sat at the end of a much deeper tournament.
        assert_eq!(tree.nodes.len(), 1);
        let mut cloned = tree.clone();
        for key in [1000.0, -1000.0, 7.0] {
            cloned.set_min(key, 99);
            assert_eq!(cloned.min(), Some((key, &99)));
        }
        cloned.remove_min();
        assert!(cloned.is_empty());
        assert_eq!(tree.min(), Some((99.0, &99)));
        tree.rebuild((0..100).map(|i| (f64::from(i), i)));
        assert_eq!(tree.len(), 100);
        assert_eq!(tree.min(), Some((0.0, &0)));
    }

    /// Data-structure cost alone: the tournament tree vs `BinaryHeap` on a merge-like
    /// workload (take the minimum, put it back with a slightly larger key), for realistic
    /// increments (the element sinks to a random depth) and tiny ones (it stays on top).
    /// `cargo test --release -- --ignored bench_vs_binary_heap --nocapture`
    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_vs_binary_heap() {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        use std::hint::black_box;
        use std::time::Instant;
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 11) as f64 / (1u64 << 53) as f64
        };
        let ops = 5_000_000u64;
        for &(k, scale) in &[(100usize, 2.0f64), (1000, 2.0), (10_000, 2.0), (1000, 1e-6)] {
            let keys: Vec<f64> = (0..k).map(|_| rnd() * 0.5).collect();
            let bumps: Vec<f64> = (0..4096).map(|_| rnd() * scale / k as f64).collect();

            let mut tree = TournamentTree::new(keys.iter().enumerate().map(|(i, &key)| (key, i as u32)));
            let t0 = Instant::now();
            let mut acc = 0u64;
            for i in 0..ops {
                let (key, &v) = tree.min().unwrap();
                acc = acc.wrapping_add(v as u64);
                tree.set_min(key + bumps[i as usize & 4095], v);
            }
            let tree_ns = t0.elapsed().as_nanos() as f64 / ops as f64;

            let mut heap: BinaryHeap<Reverse<(u64, u32)>> =
                keys.iter().enumerate().map(|(i, &key)| Reverse((key.to_bits(), i as u32))).collect();
            let t0 = Instant::now();
            let mut acc2 = 0u64;
            for i in 0..ops {
                let mut top = heap.peek_mut().unwrap();
                let Reverse((bits, v)) = *top;
                acc2 = acc2.wrapping_add(v as u64);
                *top = Reverse(((f64::from_bits(bits) + bumps[i as usize & 4095]).to_bits(), v));
            }
            let heap_ns = t0.elapsed().as_nanos() as f64 / ops as f64;
            black_box((acc, acc2));
            println!("k = {k:>6}, increment ~{scale:>5.0e}/k: tournament tree {tree_ns:>5.1} ns/op   BinaryHeap {heap_ns:>5.1} ns/op");
        }
    }
}
