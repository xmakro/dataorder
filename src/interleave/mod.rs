//! Balanced interleaving from source lengths and sampling schedules.
//!
//! An [`Interleave`] merges ordered parts without storing their elements. It defines
//! one order for the whole mix; both sequential iteration and random seeks recover
//! positions in that same order.
//!
//! # Rate profiles
//!
//! A schedule describes a part's draw rate over progress `t` from 0 to 1. Integrating
//! that rate gives a share function `F(t)`: the fraction of the part drawn by `t`.
//! Scheduled profiles are normalized so `F(1) = 1`. A constant rate produces a linear
//! share function; a linear ramp produces a quadratic one.
//!
//! Uniform parts fill the remaining capacity. Let `N` be the total length, `rho_i` the
//! fraction `n_i / N` for scheduled part `i`, and `u` the fraction of uniform elements.
//! All uniform parts use the same share function:
//!
//! ```text
//! F_uniform(t) = (t - sum(rho_i * F_i(t))) / u
//! ```
//!
//! This is nondecreasing when the scheduled rates never exceed the total draw rate.
//! Rates are piecewise linear, so checking their segment boundaries suffices. If the
//! peak exceeds the permitted rounding tolerance, construction rejects the mix.
//! With no uniform elements, the uniform profile is an unused placeholder.
//!
//! # Defining the order
//!
//! Each element receives an ideal progress key:
//!
//! ```text
//! key(i, j) = inverse_F_i((j + phi_i) / n_i)
//! phi_i    = (2 * r + 1) / (2 * k)
//! ```
//!
//! Here `j` is the index within part `i`, `n_i` is its length, `k` is the number of
//! non-empty parts, and `r` is the part's zero-based rank among them. The stagger
//! `phi_i` makes equal uniform parts round-robin. Empty parts do not affect it.
//!
//! The merged order sorts by `(key, part index, element index)`. Keys are nondecreasing
//! within each part, including under floating-point rounding, so a merge can preserve
//! each part's order without materializing the sort.
//!
//! In exact arithmetic, rounding each part's count contributes less than one element
//! of error. For a feasible schedule, an element's actual rank therefore differs from
//! `key * N` by at most `k`. Accepted overcommitment and clamping of negative uniform
//! rates can add drift proportional to `N` times the tolerance; the bound does not
//! hold for every accepted configuration.
//!
//! # Seeking and iteration
//!
//! A seek first estimates the target progress as `position / N`. For each part, it
//! counts elements below that progress, then checks the count against the actual keys.
//! If the estimates are poor, bounded binary searches find counts that do not pass
//! the target and leave at most `2k` elements to replay. Long runs of equal keys are
//! handled by counts, in part order.
//!
//! The cursor builds a [tournament tree](tournament::TournamentTree) over the remaining
//! heads and replays to the exact position. Each subsequent step emits the smallest
//! head and replaces it with that part's next element. This takes `ceil(log2 k)`
//! comparisons per element, dropping to none when one part remains.
//!
//! The sorted keys define the order, not the cursor's history. Reconstructing the same
//! heads at a seek therefore gives the same elements as walking from the beginning.
//! Monotone keys ensure rounding cannot duplicate or drop an element.

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

/// Largest supported total length. Keeps the gap between consecutive keys of one sequence
/// (at least `1/N`) far above floating-point rounding, and keeps every length and count
/// exact when converted to `f64` (which holds integers up to 2⁵³). A scheduled sequence
/// must likewise satisfy `length × max_rate ≤ MAX_TOTAL_LEN`.
pub(crate) const MAX_TOTAL_LEN: u64 = 1 << 46;

/// Slack on the overcommitment check: the summed rates are rounded, and a mix whose
/// scheduled parts need exactly the whole draw rate somewhere is valid.
const OVERCOMMIT_TOLERANCE: f64 = 1e-9;

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
    /// resulting order. Costs `O(k + s log s)` for `k` parts and `s` scheduled parts,
    /// independent of the number of elements.
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
        let mut scheduled: Vec<(f64, Profile)> = Vec::new();
        let mut uniform_len = 0u64;
        for (i, (&n, &s)) in lens.iter().zip(sampling).enumerate() {
            let profile = match s {
                Sampling::Uniform => None,
                Sampling::DelayedLinear { start: d0, full: d1 } => {
                    let ordered = 0.0 <= d0 && d0 <= d1 && d1 <= 1.0 && d0 < 1.0;
                    if !(d0.is_finite() && d1.is_finite() && ordered) {
                        return Err(SamplingError::InvalidParameter { seq: i, sampling: s });
                    }
                    Some(Profile::delayed_linear(d0, d1))
                }
                Sampling::Trapezoid { start: d0, full: d1, fade: d2, off: d3 } => {
                    let ordered = 0.0 <= d0 && d0 <= d1 && d1 <= d2 && d2 <= d3 && d3 <= 1.0 && d0 + d1 < d2 + d3;
                    if !([d0, d1, d2, d3].iter().all(|d| d.is_finite()) && ordered) {
                        return Err(SamplingError::InvalidParameter { seq: i, sampling: s });
                    }
                    Some(Profile::trapezoid(d0, d1, d2, d3))
                }
            };
            let profile = match profile {
                // Profile 0 is the shared uniform one; an empty scheduled sequence uses it too.
                None => 0,
                Some(p) => {
                    if n > 0 && !p.is_finite() {
                        return Err(SamplingError::InvalidParameter { seq: i, sampling: s });
                    }
                    if n as f64 * p.max_rate() > MAX_TOTAL_LEN as f64 {
                        return Err(SamplingError::TooSteep { seq: i });
                    }
                    if n == 0 {
                        0
                    } else {
                        scheduled.push((n as f64, p));
                        scheduled.len() as u32
                    }
                }
            };
            if profile == 0 {
                uniform_len += n;
            }
            let (inv_n, phi) = if n > 0 {
                rank += 1;
                (1.0 / n as f64, (2 * rank - 1) as f64 / (2 * live) as f64)
            } else {
                (0.0, 0.0)
            };
            seqs.push(Seq { n, inv_n, phi, profile });
        }
        // Without uniform elements the shared profile is a placeholder that nothing reads.
        let (uniform, demand, (start, end)) = Profile::uniform(&scheduled, uniform_len as f64, total.max(1) as f64);
        if !uniform.is_finite() || !demand.is_finite() {
            return Err(SamplingError::Overflow);
        }
        if demand > 1.0 + OVERCOMMIT_TOLERANCE {
            return Err(SamplingError::Overcommitted { demand, start, end });
        }
        let mut profiles = vec![uniform];
        profiles.extend(scheduled.into_iter().map(|(_, p)| p));
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

    /// Ideal progress of element `j` of `seq`. `seg` caches the profile segment.
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
