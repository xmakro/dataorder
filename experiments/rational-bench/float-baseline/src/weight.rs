//! Exact integer element counts from relative weights.
//!
//! Round each part's quota down, then assign the remaining elements to the largest
//! fractional remainders, breaking ties by part index. Floating-point division can
//! misorder close remainders, so quotas are computed from the exact binary weights.
//!
//! Each weight is an integer significand times a power of two. Most configurations fit
//! in `u128` after a common power of two is removed. Larger exponent ranges use
//! library integers for the same quotient and remainder calculation. The fallback
//! stores wide remainders during construction; iteration uses only the final counts.

use num_bigint::BigUint;
use num_integer::Integer;

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

    let sum: BigUint = weights.iter().map(|&w| numerator(w, base)).sum();
    let mut result = Vec::with_capacity(weights.len());
    let mut remainders = Vec::with_capacity(weights.len());
    for (i, &w) in weights.iter().enumerate() {
        let (quotient, remainder) = (numerator(w, base) * total).div_rem(&sum);
        result.push(u64::try_from(quotient).expect("a quota cannot exceed total"));
        remainders.push((remainder, i));
    }
    remainders.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    distribute(total, &mut result, remainders.into_iter().map(|(_, i)| i));
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

fn numerator(weight: f64, base: u32) -> BigUint {
    let (significand, shift) = parts(weight);
    if significand == 0 { BigUint::from(0u8) } else { BigUint::from(significand) << (shift - base) }
}

#[cfg(test)]
mod oracle_tests {
    use super::shares;

    #[test]
    fn matches_independent_arbitrary_precision_fixtures() {
        // Includes totals above usize::MAX on 32-bit targets: test the same exact
        // arithmetic there without relying on the public API's final length limit.
        for (line, fixture) in include_str!("../tests/fixtures/weight_oracle.txt").lines().enumerate() {
            if fixture.starts_with('#') {
                continue;
            }
            let fields: Vec<_> = fixture.split('|').collect();
            let total = fields[0].parse().unwrap();
            let weights: Vec<_> = fields[1].split(',').map(|s| f64::from_bits(u64::from_str_radix(s, 16).unwrap())).collect();
            let expected: Vec<u64> = fields[2].split(',').map(|s| s.parse().unwrap()).collect();
            assert_eq!(shares(total, &weights), expected, "independent fixture line {}", line + 1);
        }
    }
}
