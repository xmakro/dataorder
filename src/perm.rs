//! Seeded permutations of `0..n`, computed without storing an index array.
//!
//! First permute a power-of-two domain containing `0..n`. If the result is outside
//! `0..n`, apply the permutation again until it falls inside. This is cycle walking:
//! it preserves a bijection because each cycle is restricted to its in-range values.
//! For `n > 1`, choose `k` so `2^(k-1) < n ≤ 2^k`. Across all inputs, cycle walking
//! takes at most `2^k / n < 2` steps on average. A single input can take as many as
//! `2^k - n + 1` steps.
//!
//! The bijection is a six-round Feistel network on the two halves of the `k` bits (the
//! right half one bit wider when `k` is odd; the halves trade widths every round). Each
//! round adds its independently derived key to one half, applies the full `SplitMix64`
//! finalizer and takes the low bits needed by the other half. Mixing before truncation
//! avoids patterns retained by simpler multiply-and-truncate rounds. Six rounds give
//! margin beyond the four that passed the statistical probes used during development.
//!
//! The tests cover bijectivity, small-domain coverage over seeds, position/image joint
//! distributions, serial and positional correlations, fixed points and consecutive-image
//! differences. They include public-API regressions and a sweep of independently chosen
//! seeds and lengths, including powers of two and lengths on either side. Passing these
//! tests is evidence of statistical quality, not a guarantee for every seed and length.
//!
//! A seed, repetition context and source salt are combined into one 64-bit word,
//! which determines all six round keys. A [`Key`] stores 384 bits, but has only
//! 64 bits of independent input. This is for reproducible ordering, not cryptography.
//!
//! `permute` and its rounds are `#[inline(always)]`: the shuffle step is one small function
//! and the permutation is most of it.

/// Shape of the domain: `n` and the widths and masks of the two Feistel halves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Shape {
    pub(crate) n: usize,
    /// Width of the right half, which is at least as wide as the left half.
    rb: u32,
    lmask: u64,
    rmask: u64,
}

impl Shape {
    pub(crate) fn new(n: usize) -> Self {
        let bits = usize::BITS - n.saturating_sub(1).leading_zeros();
        let lb = bits / 2;
        let rb = bits - lb;
        Self { n, rb, lmask: (1u64 << lb) - 1, rmask: (1u64 << rb) - 1 }
    }
}

/// Round keys of one keyed bijection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Key {
    rk: [u64; 6],
}

impl Key {
    /// Placeholder for cursors that have not been positioned yet.
    pub(crate) const UNSET: Self = Self { rk: [0; 6] };
}

/// `SplitMix64`'s finalizer: a fixed 64-bit bijection with good avalanche (maps 0 to 0).
#[inline]
pub(crate) fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

const PHI: u64 = 0x9E37_79B9_7F4A_7C15;

/// Runtime shuffle inputs: a fixed order seed and one accumulated epoch number.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Context {
    pub(crate) seed: u64,
    pub(crate) epoch: u64,
}

impl Context {
    pub(crate) const fn new(seed: u64) -> Self {
        Self { seed, epoch: 0 }
    }

    /// Flatten nested repetitions. Counts belong to the original configuration;
    /// mixtures and selections do not change them. Arithmetic is modulo 2^64.
    #[inline]
    pub(crate) fn repeat(self, times: usize, epoch: usize) -> Self {
        Self { epoch: self.epoch.wrapping_mul(times as u64).wrapping_add(epoch as u64), ..self }
    }
}

/// The key of a shuffle with `seed` inside context `ctx` over sources
/// with the given configuration `salt`: the three are hashed together, not merely
/// xored, so no simple relation between them reproduces another triple's key. Not a
/// security boundary: seeds are for reproducibility.
#[inline]
pub(crate) fn key(seed: u64, ctx: Context, salt: u64) -> Key {
    // Preserve the original single-repeat seed arithmetic, now applied once to the
    // accumulated epoch instead of repeatedly hashing at each enclosing repeat.
    let ctx = if ctx.epoch == 0 {
        ctx.seed
    } else {
        mix64(mix64(ctx.seed ^ 0x3C6E_F372_FE94_F82B).wrapping_add(ctx.epoch.wrapping_mul(PHI)) ^ PHI)
    };
    let a =
        mix64(mix64(seed ^ 0x2545_F491_4F6C_DD1D).wrapping_add(ctx.wrapping_mul(PHI)).wrapping_add(mix64(salt)) ^ 0x1F83_D9AB_FB41_BD6B);
    let mut k = Key::UNSET;
    for (i, rk) in k.rk.iter_mut().enumerate() {
        *rk = mix64(a.wrapping_add((i as u64).wrapping_mul(PHI)));
    }
    k
}

/// A source's configuration salt, including its original length even when empty.
pub(crate) fn source_salt(salt: u64, len: usize) -> u64 {
    mix64(mix64(salt ^ 0x6A09_E667_F3BC_C908) ^ (len as u64).wrapping_mul(PHI))
}

/// Combines child configuration salts in order, before flattening or dropping children.
/// Empty lists use zero; a single child passes through unchanged. Grouping matters.
pub(crate) fn combine_salts(salts: impl IntoIterator<Item = u64>) -> u64 {
    salts.into_iter().reduce(|left, right| mix64(left.wrapping_add(PHI) ^ right)).unwrap_or(0)
}

/// One Feistel round: `(l, r)` becomes `(r, l ^ F(r))`, restricted to the left half's
/// mask. A zero-width left half (when n = 2) simply discards the round's output.
#[inline(always)]
fn round(l: u64, r: u64, mask: u64, rk: u64) -> (u64, u64) {
    (r, (l ^ mix64(r.wrapping_add(rk))) & mask)
}

/// One application of the keyed bijection on `0..2^k`.
#[inline(always)]
fn mix(s: Shape, k: Key, x: u64) -> u64 {
    let (l, r) = (x >> s.rb, x & s.rmask);
    let (l, r) = round(l, r, s.lmask, k.rk[0]);
    let (l, r) = round(l, r, s.rmask, k.rk[1]);
    let (l, r) = round(l, r, s.lmask, k.rk[2]);
    let (l, r) = round(l, r, s.rmask, k.rk[3]);
    let (l, r) = round(l, r, s.lmask, k.rk[4]);
    let (l, r) = round(l, r, s.rmask, k.rk[5]);
    // After an even number of rounds the halves have their original widths.
    (l << s.rb) | r
}

/// Image of `i` (`i < shape.n`) under the permutation of `0..n` selected by `key`.
#[inline(always)]
pub(crate) fn permute(shape: Shape, key: Key, i: usize) -> usize {
    debug_assert!(i < shape.n);
    if shape.n <= 1 {
        return i;
    }
    let mut x = i as u64;
    loop {
        x = mix(shape, key, x);
        if x < shape.n as u64 {
            return x as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Order, Seq};

    fn perm(n: usize, seed: u64) -> Vec<usize> {
        let (shape, key) = (Shape::new(n), key(seed, Context::new(0), 0));
        (0..n).map(|i| permute(shape, key, i)).collect()
    }

    #[test]
    fn bijection() {
        for n in [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 100, 255, 256, 257, 1000, 4097, 65_536, 100_003] {
            for seed in 0..4 {
                let p = perm(n, seed);
                let mut seen = vec![false; n];
                for &v in &p {
                    assert!(!seen[v], "n={n} seed={seed}: {v} twice");
                    seen[v] = true;
                }
            }
        }
    }

    #[test]
    fn huge_domain_is_a_bijection_locally() {
        // n near usize::MAX: the forward map must still be invertible; check distinct images of a
        // window and that the walk terminates.
        let (shape, key) = (Shape::new(usize::MAX - 5), key(9, Context::new(0), 0));
        let mut images: Vec<usize> = (0..10_000).map(|i| permute(shape, key, i)).collect();
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
        let (shape, k) = (Shape::new(n), key(1, Context::new(0).repeat(2, 1), 0));
        let c: Vec<usize> = (0..n).map(|i| permute(shape, k, i)).collect();
        assert!(a.iter().zip(&c).filter(|(x, y)| x == y).count() < 10);
        // Different salts, and sources of different lengths or in another order, decorrelate.
        let k = key(1, Context::new(0), source_salt(7, n));
        let d: Vec<usize> = (0..n).map(|i| permute(shape, k, i)).collect();
        assert!(a.iter().zip(&d).filter(|(x, y)| x == y).count() < 10);
        assert_ne!(source_salt(0, 1000), source_salt(0, 999));
        let (a, b) = (source_salt(1, 10), source_salt(2, 10));
        assert_ne!(combine_salts([a, b]), combine_salts([b, a]));
        // Nested repeats count the same epochs as one flat repeat.
        for outer in 0..2 {
            for inner in 0..3 {
                let nested = Context::new(5).repeat(2, outer).repeat(3, inner);
                let flat = Context::new(5).repeat(6, outer * 3 + inner);
                assert_eq!(nested.epoch, (outer * 3 + inner) as u64);
                assert_eq!(key(1, nested, 7), key(1, flat, 7));
            }
        }
    }

    /// Chi-square of `values` against a uniform expectation over `cells` cells.
    fn chi2(values: impl Iterator<Item = usize>, cells: usize, total: usize) -> f64 {
        let mut counts = vec![0usize; cells];
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

    /// A generous threshold above the chi-square mean. Its tail is not exactly normal;
    /// use a wider margin for large sweeps so an ordinary random outlier is not a failure.
    fn chi2_bound(dof: usize, sigmas: f64) -> f64 {
        dof as f64 + sigmas * (2.0 * dof as f64).sqrt()
    }

    fn assert_statistics(p: &[usize], seed: u64, sigmas: f64) {
        let n = p.len();
        let grid = chi2(p.iter().enumerate().map(|(i, &v)| (i * 32 / n) * 32 + (v * 32 / n)), 1024, n);
        // A permutation fixes both marginals of each joint table.
        assert!(grid < chi2_bound(31 * 31, sigmas), "n={n} seed={seed}: grid chi2 {grid}");
        let low = chi2(p.iter().enumerate().map(|(i, &v)| (i & 15) * 16 + (v & 15)), 256, n);
        assert!(low < chi2_bound(15 * 15, sigmas), "n={n} seed={seed}: low-bit chi2 {low}");
        let fixed = p.iter().enumerate().filter(|&(ref i, &v)| *i == v).count();
        assert!(fixed < 16, "n={n} seed={seed}: {fixed} fixed points");
        let bound = sigmas / (n as f64).sqrt();
        let f = |v: &usize| *v as f64;
        let positional = pearson((0..n).map(|i| i as f64), p.iter().map(f));
        assert!(positional.abs() < bound, "n={n} seed={seed}: positional correlation {positional}");
        let serial = pearson(p[..p.len() - 1].iter().map(f), p[1..].iter().map(f));
        assert!(serial.abs() < bound, "n={n} seed={seed}: serial correlation {serial}");
        let lag7 = pearson(p[..p.len() - 7].iter().map(f), p[7..].iter().map(f));
        assert!(lag7.abs() < bound, "n={n} seed={seed}: lag-7 correlation {lag7}");
        let difference = chi2(p.windows(2).map(|pair| (pair[1] + n - pair[0]) % n * 64 / n), 64, n - 1);
        // With only 63 degrees of freedom, a normal 5σ approximation still has a
        // roughly 1-in-40,000 upper tail. This bound reduces it to about 1-in-500-million.
        assert!(difference < chi2_bound(63, 8.0), "n={n} seed={seed}: difference chi2 {difference}");
    }

    #[test]
    fn statistics() {
        for n in [100_000usize, 1 << 17, (1 << 17) + 1, 1_000_003] {
            for seed in 0..8u64 {
                assert_statistics(&perm(n, seed), seed, 5.0);
            }
        }
    }

    fn public_perm(n: usize, seed: u64) -> Vec<usize> {
        let order = Order::new(Seq::source(n).shuffle(seed)).unwrap();
        order.iter().map(|item| item.record_index).collect()
    }

    #[test]
    fn public_seeds_with_patterned_consecutive_images() {
        // Seven multiply-and-truncate rounds produced serial correlation 0.0512 on the
        // first case and consecutive-difference chi-square 13,622 on the second. Use the
        // public API: a bare length contributes a source salt, unlike the private helper.
        for (n, seed) in [(56_444, 1), (65_536, 18_437)] {
            assert_statistics(&public_perm(n, seed), seed, 5.0);
        }
    }

    /// Exercise the public key derivation over unrelated seeds and lengths, rather than
    /// only successive seeds at a few hand-picked lengths. Include unbalanced Feistel
    /// halves and cycle walks just above and below powers of two. The wider thresholds
    /// allow ordinary random outliers across the thousands of statistics checked here.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "runs in release builds (cargo test --release)")]
    fn statistics_over_many_seeds_and_lengths() {
        for case in 0..512u64 {
            let random = mix64(case.wrapping_add(0x741A_D87B_E299_1437));
            let seed = mix64(random ^ 0xD30A_64E4_FB76_91E3);
            let power = 1usize << (14 + random % 6);
            let n = match case % 5 {
                0 => power,
                1 => power - 1,
                2 => power + 1,
                3 => power * 3 / 4,
                _ => power / 2 + (random as usize % (power / 2)),
            };
            assert_statistics(&public_perm(n, seed), seed, 8.0);
        }
    }

    /// Small domains: over many seeds, every element lands on every position about equally
    /// often (each cell of the seed-averaged permutation matrix within ±35% of uniform,
    /// which is about 5σ for the smallest expectations here).
    #[test]
    fn small_domains_mix_over_seeds() {
        for n in [2usize, 3, 5, 8, 13, 32, 100] {
            let seeds = 20_000u64;
            let mut counts = vec![0usize; n * n];
            for seed in 0..seeds {
                for (i, v) in perm(n, seed).into_iter().enumerate() {
                    counts[i * n + v] += 1;
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
