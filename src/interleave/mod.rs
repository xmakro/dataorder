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
//! parts and `r` is the part's rank. The compiler builds an interleave over a mix's
//! non-empty parts only, so empty parts do not affect it. The stagger makes equal
//! uniform parts round-robin. The merge sorts by
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
//! A seek tries `position / N`, then up to eight secant interpolations between
//! observed integer ranks, then at most 63 virtual-time bisections, each counting all
//! `k` parts. It locates a prefix at most `2k` elements before its target. Each
//! per-part count starts from a constant-time CDF estimate and corrects it against
//! the actual keys with at most four steps, then a bounded index bisection of at most
//! 46 steps; profiles have at most five segments. Equal-key runs are consumed by
//! counts in part order, without walking through the run.
//!
//! A tournament tree merges the remaining heads and replays to the exact target.
//! Each subsequent step uses `ceil(log2 k)` comparisons, dropping to none when one
//! part remains. Monotone keys and deterministic ties make seeks agree with walks,
//! independent of cursor history or how virtual time maps to output progress.

mod iter;
mod profile;
mod schedule;
#[cfg(test)]
mod tests;
mod tournament;

pub(crate) use iter::Iter;
pub use schedule::Schedule;

use profile::Profile;
use std::ops::Range;

/// Largest supported total length. Counts convert exactly to binary64. The additional
/// per-part limit `length × max_rate ≤ MAX_TOTAL_LEN` keeps nominal consecutive-key
/// spacing at least `1/MAX_TOTAL_LEN`, leaving room for virtual-time rounding.
pub(crate) const MAX_TOTAL_LEN: u64 = 1 << 46;

#[derive(Clone, Copy, Debug)]
struct Part {
    n: usize,
    /// `1/n`: keys multiply by it instead of dividing (monotone in `j` all the same).
    inv_n: f64,
    /// Stagger offset `(2r+1)/(2k)` of the part with rank `r` among the `k` parts.
    phi: f64,
    /// Index into `Interleave::profiles`; 0 is the shared uniform profile.
    profile: u32,
}

/// A balanced, order-preserving interleaving of `k` non-empty parts with schedules,
/// given only their lengths. Build it with [`Interleave::new`] and walk any merged range
/// with [`Interleave::iter`].
#[derive(Clone, Debug)]
pub(crate) struct Interleave {
    parts: Vec<Part>,
    /// Rate profiles: `profiles[0]` for the uniform parts, one more per scheduled one.
    profiles: Vec<Profile>,
    total: usize,
}

impl Interleave {
    /// Builds a mix from its non-empty parts' lengths and profiles, `None` meaning
    /// uniform, in part order. The caller has validated each profile against its part's
    /// length and the total against [`MAX_TOTAL_LEN`]; see [`Schedule::profile`].
    /// Costs `O(k)` for `k` parts, independent of their lengths.
    pub(crate) fn new(parts: Vec<(usize, Option<Profile>)>) -> Self {
        let k = parts.len();
        let mut profiles = vec![Profile::trapezoid(0.0, 0.0, 1.0, 1.0)];
        let mut total = 0;
        let parts = parts
            .into_iter()
            .enumerate()
            .map(|(rank, (n, profile))| {
                debug_assert!(n > 0, "interleave: empty part");
                let profile = match profile {
                    None => 0,
                    Some(profile) => {
                        profiles.push(profile);
                        (profiles.len() - 1) as u32
                    }
                };
                total += n;
                Part { n, inv_n: 1.0 / n as f64, phi: (2 * rank + 1) as f64 / (2 * k) as f64, profile }
            })
            .collect();
        Self { parts, profiles, total }
    }

    /// Length of the merged sequence (sum of all part lengths).
    pub(crate) fn len(&self) -> usize {
        self.total
    }

    /// `true` when some part has a schedule.
    pub(crate) fn is_scheduled(&self) -> bool {
        self.profiles.len() > 1
    }

    /// Rate profile of `part`.
    #[inline(always)]
    fn profile(&self, part: usize) -> &Profile {
        &self.profiles[self.parts[part].profile as usize]
    }

    /// Virtual time of element `j` of `part`. `seg` caches the profile segment.
    #[inline(always)]
    fn key(&self, part: usize, j: usize, seg: &mut usize) -> f64 {
        let s = &self.parts[part];
        self.profile(part).quantile((j as f64 + s.phi) * s.inv_n, seg)
    }

    /// Iterates the merged range in merged order, yielding `(part, index within the part)`.
    ///
    /// # Panics
    /// If `range.end > len()` or `range.start > range.end`.
    pub(crate) fn iter(&self, range: Range<usize>) -> Iter<'_> {
        assert!(range.start <= range.end, "interleave: invalid range");
        assert!(range.end <= self.total, "interleave: range end {} out of range", range.end);
        Iter::new(self, range)
    }
}
