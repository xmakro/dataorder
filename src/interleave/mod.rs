//! Balanced interleaving of many ordered sequences with per-sequence sampling schedules,
//! iterated over any sub-range without materializing the merged sequence.
//!
//! Given `k` sequences of lengths `n_0 … n_{k-1}`, an [`Interleave`] describes one merged
//! ("joint") sequence of length `N = Σ n_i` in which every sequence keeps its own order
//! and is drawn according to its [`Sampling`], a profile of its draw rate over the joint
//! sequence:
//!
//! ```text
//!   ramp(d0, d1)                       delayed(d)                      Uniform
//!             ________________                 ________________     ________________
//!            /                                 |
//!   ________/                         ________|
//!           d0     d1                          d
//! ```
//!
//! # Model
//!
//! Progress `τ ∈ [0, 1]` is the joint position divided by `N`. Every sequence has a share
//! function `F(τ)`: the fraction of it drawn by progress `τ`. For a scheduled sequence it is
//! the integral of its rate profile normalized to `F(1) = 1`: zero until `d0`, a parabola on
//! the ramp, a straight line after `d1`. Every joint position holds exactly one element, so
//! the uniform sequences absorb the slack: by progress `τ` they have jointly drawn the
//! `τ·N − Σ_scheduled n_i·F_i(τ)` elements the scheduled ones did not, shared in proportion
//! to their lengths, which gives them the common share function
//! `F_U(τ) = (τ − Σ ρ_i·F_i(τ)) / u` with `ρ_i = n_i/N` and `u` the uniform fraction of all
//! elements. Rates never fall, so `F_U` is steepest at the start and flattest at the end; it
//! stays nondecreasing exactly when the scheduled sequences' final rates sum to at most the
//! whole draw rate, otherwise the configuration is rejected.
//!
//! Every rate profile is piecewise linear and is stored as such; share functions are their
//! integrals.
//! Element `j` of sequence `i` gets the ideal progress `F_i⁻¹((j + φ_i)/n_i)`, where
//! `φ_i = (2i+1)/(2k)` staggers the sequences so that equal ones round-robin instead of
//! bunching, and the joint sequence is the sort of all elements by ideal progress (ties by
//! sequence index). Every sequence follows its schedule to within about one element at any
//! joint position; the joint position of an element is within `k` (typically `√k`) of
//! `progress·N`, the same warp for all sequences.
//!
//! # Iteration
//!
//! [`Interleave::iter`] seeks by counting, per sequence, the elements below the progress
//! `a/N` (inverse formula, then made exact against real keys) and then repeatedly takes the
//! minimum of a [tournament tree](tournament::TournamentTree) over the next element of
//! every sequence (`⌈log2 k⌉` comparisons per element).
//!
//! `iter(a..b)` yields exactly the elements at positions `a..b` of `iter(0..N)`, whatever
//! the seek history. The order is defined as the sort by `(key, sequence, index)` and the
//! tree only ever returns the true minimum of the remaining heads, so its internal state is
//! irrelevant; the seek reproduces the heads exactly because keys are nondecreasing within
//! a sequence *by construction* (every step of their evaluation is a correctly rounded
//! monotone operation), which also means rounding can never duplicate or drop an element.

mod iter;
mod profile;
mod sampling;
#[cfg(test)]
mod tests;
mod tournament;

pub use iter::Iter;
pub use sampling::{Sampling, SamplingError};

use profile::Profile;
use std::ops::Range;

/// Largest supported total length. Keeps the gap between consecutive keys of one sequence
/// (at least `1/N`) far above floating-point rounding, and keeps every length and count
/// exact when converted to `f64` (which holds integers up to 2⁵³). A scheduled sequence
/// must likewise satisfy `length × final_rate ≤ MAX_TOTAL_LEN`.
pub const MAX_TOTAL_LEN: u64 = 1 << 46;

/// Slack on the overcommitment check: the final rates are rounded sums, and a mix whose
/// scheduled parts need exactly the whole draw rate at the end is valid.
const OVERCOMMIT_TOLERANCE: f64 = 1e-9;

#[derive(Clone, Copy, Debug)]
struct Seq {
    n: u64,
    /// `1/n`: keys multiply by it instead of dividing (monotone in `j` all the same).
    inv_n: f64,
    /// Stagger offset `(2i+1)/(2k)`.
    phi: f64,
    /// Index into `Interleave::profiles`; 0 is the shared uniform profile.
    profile: u32,
}

/// A balanced, order-preserving interleaving of `k` sequences with sampling schedules,
/// given only their lengths. Build it with [`Interleave::new`] or
/// [`Interleave::with_sampling`] and walk any merged range with [`Interleave::iter`].
#[derive(Clone, Debug)]
pub struct Interleave {
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
    pub fn new(lens: &[u64]) -> Self {
        Self::with_sampling(lens, &vec![Sampling::Uniform; lens.len()]).expect("interleave: total length exceeds MAX_TOTAL_LEN")
    }

    /// Sequences of the given lengths and schedules (one per sequence). Zero lengths are
    /// allowed and their schedule is ignored. Cost `O(k + s²)` for `s` scheduled sequences,
    /// independent of the lengths.
    ///
    /// # Panics
    /// If `lens` and `sampling` differ in length.
    pub fn with_sampling(lens: &[u64], sampling: &[Sampling]) -> Result<Self, SamplingError> {
        assert_eq!(lens.len(), sampling.len(), "interleave: one schedule per sequence");
        let k = lens.len();
        let mut total: u64 = 0;
        for &n in lens {
            total = total.checked_add(n).filter(|&t| t <= MAX_TOTAL_LEN).ok_or(SamplingError::TooLong)?;
        }
        let mut seqs = Vec::with_capacity(k);
        let mut scheduled: Vec<(f64, Profile)> = Vec::new();
        let mut uniform_len = 0u64;
        for (i, (&n, &s)) in lens.iter().zip(sampling).enumerate() {
            // Profile 0 is the shared uniform one; an empty scheduled sequence uses it too.
            let profile = match s {
                Sampling::Uniform => 0,
                Sampling::DelayedLinear { start: d0, full: d1 } => {
                    if !(d0.is_finite() && d1.is_finite() && 0.0 <= d0 && d0 <= d1 && d1 <= 1.0 && d0 < 1.0) {
                        return Err(SamplingError::InvalidParameter { seq: i, sampling: s });
                    }
                    let p = Profile::delayed_linear(d0, d1);
                    if n as f64 * p.final_rate() > MAX_TOTAL_LEN as f64 {
                        return Err(SamplingError::TooSteep { seq: i });
                    }
                    if n == 0 {
                        0
                    } else {
                        scheduled.push((n as f64 / total as f64, p));
                        scheduled.len() as u32
                    }
                }
            };
            if profile == 0 {
                uniform_len += n;
            }
            seqs.push(Seq { n, inv_n: 1.0 / n as f64, phi: (2 * i + 1) as f64 / (2 * k) as f64, profile });
        }
        let demand: f64 = scheduled.iter().map(|(rho, p)| rho * p.final_rate()).sum();
        if demand > 1.0 + OVERCOMMIT_TOLERANCE {
            return Err(SamplingError::Overcommitted { demand });
        }
        let u = if total == 0 { 1.0 } else { uniform_len as f64 / total as f64 };
        let mut profiles = vec![Profile::uniform(&scheduled, u.max(f64::MIN_POSITIVE))];
        profiles.extend(scheduled.into_iter().map(|(_, p)| p));
        Ok(Self { seqs, profiles, total })
    }

    /// Length of the merged sequence (sum of all sequence lengths).
    pub fn len(&self) -> u64 {
        self.total
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
    pub fn iter(&self, range: Range<u64>) -> Iter<'_> {
        assert!(range.start <= range.end, "interleave: invalid range");
        assert!(range.end <= self.total, "interleave: range end {} out of range", range.end);
        Iter::new(self, range)
    }
}
