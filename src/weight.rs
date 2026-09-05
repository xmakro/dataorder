//! Exact largest-remainder apportionment of finite, nonnegative binary64 weights.
//!
//! Each weight is an integer significand times a power of two. Most configurations fit
//! in `u128` after a common power of two is removed. The fallback retains the full exponent
//! range in a small fixed-width integer, without a dependency or a wide integer per part.

use std::cmp::Ordering;

/// Allocate a positive `total` over validated weights with a positive sum.
pub(crate) fn shares(total: u64, weights: &[f64]) -> Vec<u64> {
    let base = weights.iter().filter(|&&w| w > 0.0).map(|&w| parts(w).1).min().unwrap();
    // Keep common weights entirely in machine integers, including construction of the
    // denominator. A significand has at most 53 bits, so shifts up to 75 fit in u128.
    let small_sum = weights.iter().try_fold(0u128, |sum, &w| {
        let (significand, shift) = parts(w);
        if significand == 0 {
            return Some(sum);
        }
        let shift = shift - base;
        if shift > 75 { None } else { sum.checked_add(u128::from(significand) << shift) }
    });
    if let Some(denominator) = small_sum.filter(|s| s.checked_mul(u128::from(total)).is_some()) {
        let mut result = Vec::with_capacity(weights.len());
        let mut remainders = Vec::with_capacity(weights.len());
        for (i, &w) in weights.iter().enumerate() {
            let (significand, shift) = parts(w);
            let n = if significand == 0 { 0 } else { (u128::from(significand) << (shift - base)) * u128::from(total) };
            result.push((n / denominator) as u64);
            remainders.push((n % denominator, i));
        }
        remainders.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        distribute(total, &mut result, remainders.into_iter().map(|(_, i)| i));
        return result;
    }

    let mut sum = Wide::default();
    for &w in weights {
        sum.add(&numerator(w, 1, base));
    }
    let mut result: Vec<u64> = weights.iter().map(|&w| numerator(w, total, base).quotient(&sum, total)).collect();
    let mut ranked: Vec<usize> = (0..weights.len()).collect();
    ranked.sort_unstable_by(|&a, &b| {
        // Equal quotients leave the original weights in remainder order. Otherwise
        // compare total*(w_a-w_b) with (q_a-q_b)*sum, keeping every low bit of the sum.
        let cmp = if result[a] == result[b] {
            weights[a].partial_cmp(&weights[b]).unwrap()
        } else if result[a] > result[b] {
            let mut difference = numerator(weights[a], total, base);
            difference.subtract(&numerator(weights[b], total, base));
            difference.cmp(&sum.multiply(result[a] - result[b]))
        } else {
            let mut difference = numerator(weights[b], total, base);
            difference.subtract(&numerator(weights[a], total, base));
            sum.multiply(result[b] - result[a]).cmp(&difference)
        };
        cmp.reverse().then(a.cmp(&b))
    });
    distribute(total, &mut result, ranked);
    result
}

fn distribute(total: u64, shares: &mut [u64], ranked: impl IntoIterator<Item = usize>) {
    let left = total - shares.iter().sum::<u64>();
    // The sum of the fractional remainders is an integer smaller than the part count.
    for i in ranked.into_iter().take(usize::try_from(left).unwrap()) {
        shares[i] += 1;
    }
    debug_assert_eq!(shares.iter().sum::<u64>(), total);
}

/// `w = significand * 2^(shift-1074)`, including subnormals and either signed zero.
fn parts(w: f64) -> (u64, u32) {
    let bits = w.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as u32;
    let significand = (bits & ((1 << 52) - 1)) | if exponent == 0 { 0 } else { 1 << 52 };
    (significand, exponent.saturating_sub(1))
}

fn numerator(weight: f64, total: u64, base: u32) -> Wide {
    let (significand, shift) = parts(weight);
    let mut out = Wide::default();
    if significand == 0 {
        return out;
    }
    let value = u128::from(significand) * u128::from(total);
    let shift = (shift - base) as usize;
    let (word, bits) = (shift / 64, shift % 64);
    out.words[word] = (value as u64) << bits;
    out.words[word + 1] = if bits == 0 { (value >> 64) as u64 } else { (value >> (64 - bits)) as u64 };
    if bits != 0 {
        out.words[word + 2] = (value >> (128 - bits)) as u64;
    }
    out.len = word + 3;
    while out.len > 0 && out.words[out.len - 1] == 0 {
        out.len -= 1;
    }
    out
}

/// A binary64 spans at most 2098 bits in units of the smallest subnormal. Even a sum
/// of `usize::MAX` weights multiplied by the largest mix length needs at most 2208 bits.
const WORDS: usize = 35;

struct Wide {
    words: [u64; WORDS],
    len: usize,
}

impl Default for Wide {
    fn default() -> Self {
        Self { words: [0; WORDS], len: 0 }
    }
}

impl Wide {
    fn cmp(&self, other: &Self) -> Ordering {
        self.len.cmp(&other.len).then_with(|| self.words[..self.len].iter().rev().cmp(other.words[..other.len].iter().rev()))
    }

    fn add(&mut self, other: &Self) {
        let mut carry = 0u128;
        self.len = self.len.max(other.len);
        for i in 0..self.len {
            let sum = u128::from(self.words[i]) + u128::from(other.words[i]) + carry;
            self.words[i] = sum as u64;
            carry = sum >> 64;
        }
        if carry != 0 {
            self.words[self.len] = carry as u64;
            self.len += 1;
        }
    }

    fn multiply(&self, factor: u64) -> Self {
        let mut out = Self::default();
        let mut carry = 0u128;
        for i in 0..self.len {
            let product = u128::from(self.words[i]) * u128::from(factor) + carry;
            out.words[i] = product as u64;
            carry = product >> 64;
        }
        out.len = self.len;
        if carry != 0 {
            out.words[out.len] = carry as u64;
            out.len += 1;
        }
        while out.len > 0 && out.words[out.len - 1] == 0 {
            out.len -= 1;
        }
        out
    }

    fn subtract(&mut self, other: &Self) {
        debug_assert!(self.cmp(other) != Ordering::Less);
        let mut borrow = false;
        for i in 0..self.len {
            let (difference, a) = self.words[i].overflowing_sub(other.words[i]);
            let (difference, b) = difference.overflowing_sub(u64::from(borrow));
            self.words[i] = difference;
            borrow = a || b;
        }
        debug_assert!(!borrow);
        while self.len > 0 && self.words[self.len - 1] == 0 {
            self.len -= 1;
        }
    }

    /// A guess using the leading 64 bits, followed by exact comparisons. Usually the
    /// guess or an adjacent integer is correct; bisection makes accuracy independent of
    /// the quality of the floating-point guess, with at most 46 additional comparisons.
    fn quotient(&self, denominator: &Self, total: u64) -> u64 {
        if self.len == 0 {
            return 0;
        }
        let leading = |v: &Self| {
            let high = v.words[v.len - 1];
            let zeros = high.leading_zeros();
            let mut mantissa = high << zeros;
            if zeros > 0 && v.len > 1 {
                mantissa |= v.words[v.len - 2] >> (64 - zeros);
            }
            (mantissa as f64, v.len as i32 * 64 - zeros as i32)
        };
        let (a, ae) = leading(self);
        let (b, be) = leading(denominator);
        let exponent = ae - be;
        let guess = if exponent < -1022 { 0.0 } else { a / b * f64::from_bits(((1023 + exponent) as u64) << 52) };
        let q = (guess as u64).min(total);
        let below = |q| denominator.multiply(q).cmp(self) != Ordering::Greater;
        let (mut low, mut high) = if !below(q) {
            if q > 0 && below(q - 1) {
                return q - 1;
            }
            (0, q)
        } else {
            if q == total || !below(q + 1) {
                return q;
            }
            if q + 1 == total || !below(q + 2) {
                return q + 1;
            }
            (q + 2, total + 1)
        };
        while high - low > 1 {
            let mid = low + (high - low) / 2;
            if below(mid) { low = mid } else { high = mid }
        }
        low
    }
}
