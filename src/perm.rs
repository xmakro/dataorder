//! Seeded permutations of `0..n` in O(1) per element, without materializing them.
//!
//! A permutation of `0..n` is a keyed bijection on the `k`-bit numbers, `2^(k-1) < n ≤ 2^k`,
//! restricted to `0..n` by cycle walking: a value that lands at or above `n` is mapped again
//! until one below `n` comes out. That is a bijection on `0..n` because the underlying map
//! is one on the superset, and it takes `2^k / n < 2` steps on average.
//!
//! The bijection is a six-round Feistel network on the two halves of the `k` bits (the
//! right half one bit wider when `k` is odd; the halves trade widths every round). The round
//! function adds the round key to one half, multiplies by the round's odd multiplier and
//! takes the top bits of the product, which every input bit influences. Six such rounds
//! pass the statistics in this module's tests (joint distribution of position and image on
//! a grid and in the low bits, serial correlation, fixed points) at every size tried, from 2
//! to 10⁶ and beyond; there is no security claim.
//!
//! Alternatives measured and rejected: a masked multiply–xorshift mixer (MurmurHash3's
//! finalizer cut to `k` bits) is twice as fast but maps consecutive inputs to outputs with
//! a nearly constant difference (serial correlation over 100σ); four rounds with a
//! two-multiply round function (multiply, xorshift, multiply) have the same quality and
//! cost about 0.8 ns more per element; three of those rounds fail the grid test.
//!
//! The six round keys are rotations of one 64-bit word derived from the seed and the
//! context, the six multipliers rotations of another derived from the first, so a
//! permutation is selected by 64 bits in effect, although a [`Key`] holds 768; that is
//! plenty for reproducible shuffles, which is all seeds are for.
//!
//! `permute` and its rounds are `#[inline(always)]`: the shuffle step is one small function
//! and the permutation is most of it.

/// Shape of the domain: `n` and the widths and masks of the two Feistel halves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Shape {
    pub(crate) n: u64,
    /// Width of the left half; the right half is `lb` or `lb + 1` bits wide.
    lb: u32,
    rb: u32,
    lmask: u64,
    rmask: u64,
}

impl Shape {
    pub(crate) fn new(n: u64) -> Self {
        let bits = 64 - n.saturating_sub(1).leading_zeros();
        let lb = bits / 2;
        let rb = bits - lb;
        Self { n, lb, rb, lmask: (1u64 << lb) - 1, rmask: (1u64 << rb) - 1 }
    }
}

/// Round keys and multipliers (odd) of one keyed bijection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Key {
    rk: [u64; 6],
    mul: [u64; 6],
}

impl Key {
    /// Placeholder for cursors that have not been seeked yet.
    pub(crate) const UNSET: Self = Self { rk: [0; 6], mul: [1; 6] };
}

/// SplitMix64's finalizer: a fixed 64-bit bijection with good avalanche (maps 0 to 0).
#[inline]
pub(crate) fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

const PHI: u64 = 0x9E37_79B9_7F4A_7C15;
const RC: [u64; 12] = [
    0x243F_6A88_85A3_08D3,
    0x1319_8A2E_0370_7344,
    0xA409_3822_299F_31D0,
    0x082E_FA98_EC4E_6C89,
    0x4528_21E6_38D0_1377,
    0xBE54_66CF_34E9_0C6C,
    0xC0AC_29B7_C97C_50DD,
    0x3F84_D5B5_B547_0917,
    0x9216_D5D9_8979_FB1B,
    0xD131_0BA6_98DF_B5AC,
    0x2FFD_72DB_D01A_DFB7,
    0xB8E1_AFED_6A26_7E96,
];

/// The key of a shuffle with `seed` inside context `ctx` (see [`epoch_ctx`]): the two are
/// hashed together, not merely xored, so no simple relation between a seed and a context
/// reproduces another pair's key. Not a security boundary: seeds are for reproducibility.
#[inline]
pub(crate) fn key(seed: u64, ctx: u64) -> Key {
    let a = mix64(mix64(seed ^ 0x2545_F491_4F6C_DD1D).wrapping_add(ctx.wrapping_mul(PHI)) ^ 0x1F83_D9AB_FB41_BD6B);
    let b = mix64(a ^ PHI);
    let mut k = Key::UNSET;
    for i in 0..6 {
        k.rk[i] = a.rotate_left(i as u32 * 11 + 5) ^ RC[i];
        k.mul[i] = (b.rotate_left(i as u32 * 11 + 9) ^ RC[i + 6]) | 1;
    }
    k
}

/// Context of repetition `epoch` of a repeat nested `depth` repeats deep, inside `ctx`. The
/// first repetition keeps its context, so a sequence repeated once is itself and a repeat
/// starts with the unrepeated sequence; every other repetition gets its own, so that the
/// shuffles inside reshuffle. The depth keeps the contexts of nested repeats apart (the
/// second repetition of an inner repeat inside the first of the outer, against the first of
/// the inner inside the second of the outer).
#[inline]
pub(crate) fn epoch_ctx(ctx: u64, epoch: u64, depth: u32) -> u64 {
    if epoch == 0 {
        return ctx;
    }
    mix64(mix64(ctx ^ 0x3C6E_F372_FE94_F82B).wrapping_add(epoch.wrapping_mul(PHI)) ^ (depth as u64 + 1).wrapping_mul(PHI))
}

/// One Feistel round: `(l, r)` becomes `(r, l ^ F(r))`, where `l` is `wl` bits wide
/// (mask `ml`) and `F` yields the top `wl` bits of `(r + rk) · mul` (a width of 0, when
/// n = 2, shifts by 63 and the mask discards everything).
#[inline(always)]
fn round(l: u64, r: u64, wl: u32, ml: u64, rk: u64, mul: u64) -> (u64, u64) {
    let f = r.wrapping_add(rk).wrapping_mul(mul);
    (r, (l ^ (f >> (64 - wl.max(1)))) & ml)
}

/// One application of the keyed bijection on `0..2^k`.
#[inline(always)]
fn mix(s: Shape, k: Key, x: u64) -> u64 {
    let (l, r) = (x >> s.rb, x & s.rmask);
    let (l, r) = round(l, r, s.lb, s.lmask, k.rk[0], k.mul[0]);
    let (l, r) = round(l, r, s.rb, s.rmask, k.rk[1], k.mul[1]);
    let (l, r) = round(l, r, s.lb, s.lmask, k.rk[2], k.mul[2]);
    let (l, r) = round(l, r, s.rb, s.rmask, k.rk[3], k.mul[3]);
    let (l, r) = round(l, r, s.lb, s.lmask, k.rk[4], k.mul[4]);
    let (l, r) = round(l, r, s.rb, s.rmask, k.rk[5], k.mul[5]);
    (l << s.rb) | r
}

/// Image of `i` (`i < shape.n`) under the permutation of `0..n` selected by `key`.
#[inline(always)]
pub(crate) fn permute(shape: Shape, key: Key, i: u64) -> u64 {
    debug_assert!(i < shape.n);
    if shape.n <= 1 {
        return i;
    }
    let mut x = i;
    loop {
        x = mix(shape, key, x);
        if x < shape.n {
            return x;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perm(n: u64, seed: u64) -> Vec<u64> {
        let (shape, key) = (Shape::new(n), key(seed, 0));
        (0..n).map(|i| permute(shape, key, i)).collect()
    }

    #[test]
    fn bijection() {
        for n in [0u64, 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 100, 255, 256, 257, 1000, 4097, 65_536, 100_003] {
            for seed in 0..4 {
                let p = perm(n, seed);
                let mut seen = vec![false; n as usize];
                for &v in &p {
                    assert!(!seen[v as usize], "n={n} seed={seed}: {v} twice");
                    seen[v as usize] = true;
                }
            }
        }
    }

    #[test]
    fn huge_domain_is_a_bijection_locally() {
        // n near 2^64: the forward map must still be invertible; check distinct images of a
        // window and that the walk terminates.
        let (shape, key) = (Shape::new(u64::MAX - 5), key(9, 0));
        let mut images: Vec<u64> = (0..10_000).map(|i| permute(shape, key, i)).collect();
        images.sort_unstable();
        images.dedup();
        assert_eq!(images.len(), 10_000);
    }

    #[test]
    fn seeds_and_contexts_differ() {
        let n = 1000;
        let a = perm(n, 1);
        let b = perm(n, 2);
        assert!(a.iter().zip(&b).filter(|(x, y)| x == y).count() < 10);
        let (shape, k) = (Shape::new(n), key(1, epoch_ctx(0, 1, 0)));
        let c: Vec<u64> = (0..n).map(|i| permute(shape, k, i)).collect();
        assert!(a.iter().zip(&c).filter(|(x, y)| x == y).count() < 10);
        // The first repetition keeps its context; contexts of nested repetitions do not collide.
        assert_eq!(epoch_ctx(5, 0, 0), 5);
        assert_ne!(epoch_ctx(0, 1, 0), 0);
        assert_ne!(epoch_ctx(epoch_ctx(0, 0, 0), 1, 1), epoch_ctx(epoch_ctx(0, 1, 0), 0, 1));
    }

    /// Chi-square of `values` against a uniform expectation over `cells` cells.
    fn chi2(values: impl Iterator<Item = usize>, cells: usize, total: u64) -> f64 {
        let mut counts = vec![0u64; cells];
        for v in values {
            counts[v] += 1;
        }
        let expected = total as f64 / cells as f64;
        counts.iter().map(|&c| (c as f64 - expected).powi(2) / expected).sum()
    }

    fn pearson(xs: impl Iterator<Item = f64> + Clone, ys: impl Iterator<Item = f64> + Clone) -> f64 {
        let n = xs.clone().count() as f64;
        let (mx, my) = (xs.clone().sum::<f64>() / n, ys.clone().sum::<f64>() / n);
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for (x, y) in xs.zip(ys) {
            sxy += (x - mx) * (y - my);
            sxx += (x - mx).powi(2);
            syy += (y - my).powi(2);
        }
        sxy / (sxx * syy).sqrt()
    }

    /// Statistical plausibility on large domains: joint distribution of `(i, p(i))` on a
    /// 32×32 grid and of their low 4 bits, serial and positional correlation, fixed points.
    /// Thresholds are 5σ (chi-square) or 5/√n (correlations) and hold for every seed tried.
    #[test]
    fn statistics() {
        for n in [100_000u64, 1 << 17, (1 << 17) + 1, 1_000_003] {
            for seed in 0..8u64 {
                let p = perm(n, seed);
                let grid = chi2(p.iter().enumerate().map(|(i, &v)| (i as u64 * 32 / n) as usize * 32 + (v * 32 / n) as usize), 1024, n);
                assert!(grid < 1023.0 + 5.0 * (2.0 * 1023.0f64).sqrt(), "n={n} seed={seed}: grid chi2 {grid}");
                let low = chi2(p.iter().enumerate().map(|(i, &v)| (i & 15) * 16 + (v & 15) as usize), 256, n);
                assert!(low < 255.0 + 5.0 * (2.0 * 255.0f64).sqrt(), "n={n} seed={seed}: low-bit chi2 {low}");
                let fixed = p.iter().enumerate().filter(|&(ref i, &v)| *i as u64 == v).count();
                assert!(fixed < 10, "n={n} seed={seed}: {fixed} fixed points");
                let bound = 5.0 / (n as f64).sqrt();
                let f = |v: &u64| *v as f64;
                let positional = pearson((0..n).map(|i| i as f64), p.iter().map(f));
                assert!(positional.abs() < bound, "n={n} seed={seed}: positional correlation {positional}");
                let serial = pearson(p[..p.len() - 1].iter().map(f), p[1..].iter().map(f));
                assert!(serial.abs() < bound, "n={n} seed={seed}: serial correlation {serial}");
                let lag7 = pearson(p[..p.len() - 7].iter().map(f), p[7..].iter().map(f));
                assert!(lag7.abs() < bound, "n={n} seed={seed}: lag-7 correlation {lag7}");
            }
        }
    }

    /// Small domains: over many seeds, every element lands on every position about equally
    /// often (each cell of the seed-averaged permutation matrix within ±35% of uniform,
    /// which is about 5σ for the smallest expectations here).
    #[test]
    fn small_domains_mix_over_seeds() {
        for n in [2u64, 3, 5, 8, 13, 32, 100] {
            let seeds = 20_000u64;
            let mut counts = vec![0u64; (n * n) as usize];
            for seed in 0..seeds {
                for (i, v) in perm(n, seed).into_iter().enumerate() {
                    counts[i * n as usize + v as usize] += 1;
                }
            }
            let expected = seeds as f64 / n as f64;
            for (cell, &c) in counts.iter().enumerate() {
                let ratio = c as f64 / expected;
                assert!((0.65..1.35).contains(&ratio), "n={n}: cell {cell} at {ratio:.2} of uniform");
            }
        }
    }
}
