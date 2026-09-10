//! Independent virtual-clock schedules merged into one seekable order.
//!
//! Each part has a nonnegative draw rate over virtual time `t` in `[0, 1]`.
//! Its normalized integral `F_i(t)` describes how much of that part has been drawn.
//! Uniform parts have `F_i(t) = t`; every other profile is also built independently.
//! No profile fills a remainder or adjusts another profile's rate.
//!
//! # Defining the order
//!
//! Each element receives a virtual-time key:
//!
//! ```text
//! key(i, j) = inverse_F_i((j + phi_i) / n_i)
//! phi_i    = (2 * r + 1) / (2 * k)
//! ```
//!
//! Here `j` is the index within part `i`, `n_i` is its length, `k` is the number of
//! non-empty parts, and `r` is its rank among them. The stagger makes equal uniform
//! parts round-robin. Empty parts do not affect it. The merge sorts by
//! `(key, part index, element index)` and preserves every part's local order.
//!
//! Virtual time is not output progress. In the continuous model, output progress
//! at time `t` is `sum(n_i * F_i(t)) / N`. A part's local output fraction is
//! `n_i * rate_i(t) / sum(n_j * rate_j(t))` wherever the combined rate is positive.
//! All curves undergo the same time transformation; linear virtual-time ramps
//! need not remain linear against output positions. Intervals where all rates are
//! zero produce no elements. Any combination of individually valid schedules works.
//!
//! # Seeking and iteration
//!
//! Count each part's keys below a trial virtual time and sum those integer counts.
//! A seek tries `position / N`, then interpolates between observed integer ranks
//! before falling back to bounded bisection. It locates a prefix at
//! most `2k` elements before its target. Per-part CDF estimates are checked against
//! the actual keys; bounded index searches correct rounding differences. Equal-key
//! runs are consumed by counts in part order, without walking through the run.
//!
//! A tournament tree merges the remaining heads and replays to the exact target.
//! Each subsequent step uses `ceil(log2 k)` comparisons, dropping to none when one
//! part remains. Monotone keys and deterministic ties make seeks agree with walks,
//! independent of cursor history or how virtual time maps to output progress.

mod iter;
mod profile;
mod sampling;
#[cfg(test)]
mod tests;
mod tournament;

pub(crate) use iter::Iter;
pub use sampling::Sampling;
pub(crate) use sampling::SamplingError;

use profile::Profile;
use std::ops::Range;

/// Largest supported total length. Counts convert exactly to binary64. The additional
/// per-part limit `length × max_rate ≤ MAX_TOTAL_LEN` keeps nominal consecutive-key
/// spacing at least `1/MAX_TOTAL_LEN`, leaving room for virtual-time rounding.
pub(crate) const MAX_TOTAL_LEN: u64 = 1 << 46;

#[derive(Clone, Copy, Debug)]
struct Seq {
    n: u64,
    /// `1/n` (0 for an empty sequence): keys multiply by it instead of dividing (monotone
    /// in `j` all the same).
    inv_n: f64,
    /// Stagger offset `(2r+1)/(2k')` among the `k'` non-empty sequences (0 for an empty one).
    phi: f64,
    /// Index into `Interleave::profiles`; 0 is the shared uniform profile.
    profile: u32,
}

/// A balanced, order-preserving interleaving of `k` sequences with sampling schedules,
/// given only their lengths. Build it with [`Interleave::with_sampling`] and walk any
/// merged range with [`Interleave::iter`].
#[derive(Clone, Debug)]
pub(crate) struct Interleave {
    seqs: Vec<Seq>,
    /// Rate profiles: `profiles[0]` for the uniform sequences, one more per scheduled one.
    profiles: Vec<Profile>,
    total: u64,
}

impl Interleave {
    /// All sequences [`Sampling::Uniform`].
    ///
    /// # Panics
    /// If the total length exceeds [`MAX_TOTAL_LEN`].
    #[cfg(test)]
    pub(crate) fn new(lens: &[u64]) -> Self {
        Self::with_sampling(lens, &vec![Sampling::Uniform; lens.len()]).expect("interleave: total length exceeds MAX_TOTAL_LEN")
    }

    /// Builds a mix from lengths and schedules, one schedule per part.
    /// Empty parts still have their parameters validated but do not affect the
    /// resulting order. Costs `O(k)` for `k` parts, independent of their lengths.
    ///
    /// # Panics
    /// If `lens` and `sampling` differ in length.
    pub(crate) fn with_sampling(lens: &[u64], sampling: &[Sampling]) -> Result<Self, SamplingError> {
        assert_eq!(lens.len(), sampling.len(), "interleave: one schedule per sequence");
        let k = lens.len();
        let mut total: u64 = 0;
        for &n in lens {
            total = total.checked_add(n).filter(|&t| t <= MAX_TOTAL_LEN).ok_or(SamplingError::TooLong)?;
        }
        let live = lens.iter().filter(|&&n| n > 0).count();
        let mut rank = 0usize;
        let mut seqs = Vec::with_capacity(k);
        let mut profiles = vec![Profile::trapezoid(0.0, 0.0, 1.0, 1.0)];
        for (i, (&n, &s)) in lens.iter().zip(sampling).enumerate() {
            let profile = match s {
                Sampling::Uniform => None,
                Sampling::Trapezoid { start: d0, full: d1, fade: d2, off: d3 } => {
                    let ordered = 0.0 <= d0 && d0 <= d1 && d1 <= d2 && d2 <= d3 && d3 <= 1.0 && d0 < d3;
                    if !([d0, d1, d2, d3].iter().all(|d| d.is_finite()) && ordered) {
                        return Err(SamplingError::InvalidParameter {
                            seq: i,
                            sampling: s,
                            reason: if [d0, d1, d2, d3].iter().all(|d| d.is_finite()) {
                                crate::SamplingReason::InvalidBreakpoints
                            } else {
                                crate::SamplingReason::NonFiniteParameter
                            },
                        });
                    }
                    Some(Profile::trapezoid(d0, d1, d2, d3))
                }
            };
            let profile = match profile {
                // Profile 0 is the shared uniform one; an empty scheduled sequence uses it too.
                None => 0,
                Some(p) => {
                    if n > 0 && !p.is_finite() {
                        return Err(SamplingError::InvalidParameter {
                            seq: i,
                            sampling: s,
                            reason: crate::SamplingReason::CoefficientOverflow,
                        });
                    }
                    if n as f64 * p.max_rate() > MAX_TOTAL_LEN as f64 {
                        return Err(SamplingError::TooSteep { seq: i, len: n, peak_rate: p.max_rate() });
                    }
                    if n == 0 {
                        0
                    } else {
                        profiles.push(p);
                        (profiles.len() - 1) as u32
                    }
                }
            };
            let (inv_n, phi) = if n > 0 {
                rank += 1;
                (1.0 / n as f64, (2 * rank - 1) as f64 / (2 * live) as f64)
            } else {
                (0.0, 0.0)
            };
            seqs.push(Seq { n, inv_n, phi, profile });
        }
        Ok(Self { seqs, profiles, total })
    }

    /// Length of the merged sequence (sum of all sequence lengths).
    pub(crate) fn len(&self) -> u64 {
        self.total
    }

    /// After validation, compile a mix against only its live children. The stagger and
    /// profiles already ignored empty parts, so compaction preserves every key and tie.
    pub(crate) fn remove_empty(&mut self) {
        self.seqs.retain(|s| s.n > 0);
        self.seqs.shrink_to_fit();
    }

    /// `true` when some non-empty sequence has a schedule.
    pub(crate) fn is_scheduled(&self) -> bool {
        self.profiles.len() > 1
    }

    /// Rate profile of `seq`.
    #[inline(always)]
    fn profile(&self, seq: usize) -> &Profile {
        &self.profiles[self.seqs[seq].profile as usize]
    }

    /// Virtual time of element `j` of `seq`. `seg` caches the profile segment.
    #[inline(always)]
    fn key(&self, seq: usize, j: u64, seg: &mut usize) -> f64 {
        let s = &self.seqs[seq];
        self.profile(seq).quantile((j as f64 + s.phi) * s.inv_n, seg)
    }

    /// Iterates the merged range in merged order, yielding `(sequence, index_in_sequence)`.
    ///
    /// # Panics
    /// If `range.end > len()` or `range.start > range.end`.
    pub(crate) fn iter(&self, range: Range<u64>) -> Iter<'_> {
        assert!(range.start <= range.end, "interleave: invalid range");
        assert!(range.end <= self.total, "interleave: range end {} out of range", range.end);
        Iter::new(self, range)
    }
}
