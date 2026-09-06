//! Draw-rate profiles, cumulative shares and inverse lookup.
//!
//! A profile consists of linear rate segments over `[0, 1]`, with jumps allowed
//! between segments. Integrating the rate gives the fraction of a part drawn by
//! each progress value. `quantile` inverts that share to compute an element's key.
//! Its result must stay monotone under rounding for exact seeks to work.
//! It is explicitly inlined because every iteration step computes a key.
//!
//! Keep multiply and add separate: `mul_add` changes rounding and would change
//! the reproducible orders promised by the crate.

use crate::sum::{Compensated, Expansion};

/// A nonnegative, piecewise-linear draw rate over joint progress, with its running integral.
#[derive(Clone, Debug)]
pub(crate) struct Profile {
    segs: Vec<Segment>,
    /// Contiguous copies of segment starts and shares for binary search.
    /// A uniform profile can have thousands of segments; searching these arrays
    /// touches fewer cache lines than searching full segment records.
    starts: Vec<f64>,
    shares: Vec<f64>,
}

/// A linear rate from `r0` to `r1` over `[start, end]`.
/// `share` is the integral up to `start`. Within the segment, at offset `x`, the
/// additional share is `r0*x + c*x²`, where `c = (r1-r0) / (2*(end-start))`.
/// Cached coefficients and reciprocals reduce work when inverting that expression.
#[derive(Clone, Copy, Debug)]
struct Segment {
    start: f64,
    end: f64,
    r0: f64,
    r1: f64,
    share: f64,
    c: f64,
    inv_r0: f64,
    /// Zero-start ramps have a monotone inverse without any root subtraction.
    inv_2c: f64,
    c4: f64,
}

impl Profile {
    /// Builds consecutive `(start, end, r0, r1)` segments covering `[0, 1]`.
    /// Empty segments are dropped. Shares use the trapezoid rule, which integrates
    /// linear rates exactly apart from floating-point rounding.
    fn from_rates(rates: impl IntoIterator<Item = (f64, f64, f64, f64)>) -> Self {
        let mut share = 0.0;
        let mut segs = Vec::new();
        for (start, end, r0, r1) in rates {
            if end > start {
                let inv_r0 = if r1 == r0 && r0 > 0.0 { 1.0 / r0 } else { 0.0 };
                let c = (r1 - r0) / (2.0 * (end - start));
                let inv_2c = if c > 0.0 && r0 == 0.0 { 1.0 / (2.0 * c) } else { 0.0 };
                segs.push(Segment { start, end, r0, r1, share, c, inv_r0, inv_2c, c4: 4.0 * c });
                share += (r0 + r1) / 2.0 * (end - start);
            }
        }
        let starts = segs.iter().map(|s| s.start).collect();
        let shares = segs.iter().map(|s| s.share).collect();
        Self { segs, starts, shares }
    }

    /// `DelayedLinear { start: d0, full: d1 }`: zero until `d0`, rising linearly to the
    /// final rate at `d1`, then constant. The final rate `r = 2/(2 − d0 − d1)` makes the
    /// total share one.
    pub(crate) fn delayed_linear(d0: f64, d1: f64) -> Self {
        Self::trapezoid(d0, d1, 1.0, 1.0)
    }

    /// `Trapezoid { start: d0, full: d1, fade: d2, off: d3 }`: zero until `d0`, rising
    /// linearly to the full rate at `d1`, constant until `d2`, falling linearly to zero at
    /// `d3`, zero afterwards. The full rate `r = 2/((d2 − d1) + (d3 − d0))` makes the total
    /// share one. Subtract endpoints before adding widths to avoid cancellation.
    pub(crate) fn trapezoid(d0: f64, d1: f64, d2: f64, d3: f64) -> Self {
        let r = 2.0 / ((d2 - d1) + (d3 - d0));
        Self::from_rates([(0.0, d0, 0.0, 0.0), (d0, d1, 0.0, r), (d1, d2, r, r), (d2, d3, r, 0.0), (d3, 1.0, 0.0, 0.0)])
    }

    /// Returns the uniform profile, peak combined demand and a segment attaining that peak.
    /// For scheduled fractions `rho_i` and uniform fraction `u`, the uniform rate is
    /// `(1 - sum(rho_i * rate_i)) / u`, clamped to zero. With `u = 0`, no elements
    /// use the returned profile. A peak demand above 1 indicates overcommitment.
    ///
    /// Sweep the scheduled segment boundaries in order. Between boundaries, the
    /// total rate is linear, so only its current value and slope need updating.
    /// Adding and removing slopes uses an expansion to retain contributions across
    /// widely different magnitudes; the rate value uses a compensated sum.
    /// Sorting the boundaries costs `O(s log s)` for `s` scheduled profiles, compared
    /// with `O(s²)` when evaluating every profile at every boundary.
    pub(crate) fn uniform(scheduled: &[(f64, Self)], u: f64) -> (Self, f64, (f64, f64)) {
        // Remove the old contribution and add the new one separately: forming their
        // difference first can round away a small new slope before compensation sees it.
        let mut events = Vec::new();
        for (rho, p) in scheduled {
            let (mut prev_r1, mut prev_m) = (0.0, 0.0);
            for seg in &p.segs {
                let m = (seg.r1 - seg.r0) / (seg.end - seg.start);
                events.push((seg.start, -rho * prev_r1, -rho * prev_m));
                events.push((seg.start, rho * seg.r0, rho * m));
                (prev_r1, prev_m) = (seg.r1, m);
            }
        }
        events.sort_by(|a, b| a.0.total_cmp(&b.0));
        let rate = |total: f64| if u > 0.0 { (1.0 - total).max(0.0) / u } else { 0.0 };
        let (mut total, mut slope) = (Compensated::default(), Expansion::default());
        let (mut windows, mut peak, mut next, mut t) = (Vec::new(), 0.0f64, 0, 0.0);
        let mut peak_interval = (0.0, 1.0);
        while t < 1.0 {
            while next < events.len() && events[next].0 <= t {
                total.add(events[next].1);
                slope.add(events[next].2);
                next += 1;
            }
            let t1 = if next < events.len() { events[next].0.min(1.0) } else { 1.0 };
            let r0 = rate(total.value());
            if total.value() > peak {
                peak_interval = (t, t1);
            }
            peak = peak.max(total.value());
            total.add(slope.value() * (t1 - t));
            if total.value() > peak {
                peak_interval = (t, t1);
            }
            peak = peak.max(total.value());
            windows.push((t, t1, r0, rate(total.value())));
            t = t1;
        }
        (if u > 0.0 { Self::from_rates(windows) } else { Self::delayed_linear(0.0, 0.0) }, peak, peak_interval)
    }

    /// The rate just after `t` (`before == false`) or just before it.
    #[cfg(test)]
    fn rate_at(&self, t: f64, before: bool) -> f64 {
        let s = &self.segs[self.segs.partition_point(|s| if before { s.start < t } else { s.start <= t }).saturating_sub(1)];
        s.r0 + (s.r1 - s.r0) * ((t - s.start) / (s.end - s.start)).clamp(0.0, 1.0)
    }

    /// `f'(1⁻)`, the rate at the end.
    #[cfg(test)]
    pub(crate) fn final_rate(&self) -> f64 {
        self.segs.last().map_or(0.0, |s| s.r1)
    }

    /// The largest rate anywhere.
    pub(crate) fn max_rate(&self) -> f64 {
        self.segs.iter().fold(0.0, |m, s| m.max(s.r0).max(s.r1))
    }

    /// Finite parameters need not give finite slopes or inverse coefficients.
    pub(crate) fn is_finite(&self) -> bool {
        self.segs.iter().all(|s| [s.r0, s.r1, s.share, s.c, s.c4, s.inv_r0, s.inv_2c].iter().all(|x| x.is_finite()))
    }

    /// The share drawn by progress `t`: the integral of the rate up to `t`.
    pub(crate) fn share(&self, t: f64) -> f64 {
        let s = &self.segs[self.starts.partition_point(|&start| start <= t).saturating_sub(1)];
        let x = (t - s.start).clamp(0.0, s.end - s.start);
        s.share + x * (s.r0 + s.c * x)
    }

    /// Inverts share `y` to progress `t`, using the left endpoint of a flat interval.
    /// Updates `hint` to the selected segment. Consecutive elements usually stay in
    /// the same segment or enter the next, so lookup first walks a few nearby
    /// segments. A binary search handles larger jumps in `O(log S)` for `S` segments.
    ///
    /// Nondecreasing in `y` even under rounding: the segment index is monotone because
    /// shares are and the result is clamped to its segment. Constant and falling segments
    /// use monotone operations, as do zero-start ramps; other rising segments canonicalize
    /// their inverse against a monotone polynomial.
    #[inline(always)]
    pub(crate) fn quantile(&self, y: f64, hint: &mut usize) -> f64 {
        // The last segment whose starting share is strictly below `y` (or the first
        // segment at zero). Equality belongs to the preceding segment, so a flat share
        // interval is inverted at its left endpoint, including with a hint beyond it.
        let mut m = *hint;
        // Bound the local walk so a stale hint cannot make a seek linear in segment count.
        let mut budget = 16;
        loop {
            if m + 1 < self.segs.len() && y > self.segs[m + 1].share {
                m += 1;
            } else if m > 0 && y <= self.segs[m].share {
                m -= 1;
            } else {
                break;
            }
            budget -= 1;
            if budget == 0 {
                m = self.shares.partition_point(|&share| share < y).saturating_sub(1);
                break;
            }
        }
        *hint = m;
        let s = &self.segs[m];
        // Solve c·x² + r0·x = z for x ≥ 0; a constant rate is the common case.
        let z = (y - s.share).max(0.0);
        let x = if s.c == 0.0 {
            z * s.inv_r0
        } else {
            let root = (s.r0 * s.r0 + s.c4 * z).max(0.0).sqrt();
            if s.c > 0.0 {
                if s.r0 == 0.0 { root * s.inv_2c } else { s.rising_inverse(z, root) }
            } else if s.r0 + root > 0.0 {
                2.0 * z / (s.r0 + root)
            } else {
                0.0
            }
        };
        (s.start + x).clamp(s.start, s.end)
    }
}

impl Segment {
    /// Invert the increasing polynomial without subtracting nearly equal roots. The
    /// quotient alone can round nonmonotonically at adjacent floats, so canonicalize to
    /// the first representable x whose (monotone) polynomial reaches z. Usually this
    /// takes one or two neighboring floats; a bitwise bisection bounds even a bad guess.
    #[inline]
    fn rising_inverse(&self, z: f64, root: f64) -> f64 {
        if z == 0.0 {
            return 0.0;
        }
        let polynomial = |x: f64| x * (self.r0 + self.c * x);
        let (mut lo, mut hi) = (0.0f64, self.end - self.start);
        let mut x = (2.0 * z / (self.r0 + root)).min(hi);
        for _ in 0..4 {
            if polynomial(x) < z {
                lo = x;
                if x == hi {
                    return hi;
                }
                x = x.next_up().min(hi);
            } else {
                hi = x;
                let prev = x.next_down().max(0.0);
                if polynomial(prev) < z {
                    return x;
                }
                x = prev;
            }
        }
        while hi.to_bits() - lo.to_bits() > 1 {
            let mid = f64::from_bits(lo.to_bits() + (hi.to_bits() - lo.to_bits()) / 2);
            if polynomial(mid) < z { lo = mid } else { hi = mid }
        }
        hi
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> impl Iterator<Item = f64> {
        (0..=1000).map(|i| i as f64 / 1000.0)
    }

    /// The uniform profile by direct evaluation of every scheduled profile at every
    /// breakpoint, the `O(s²)` way the sweep replaces.
    fn uniform_direct(scheduled: &[(f64, Profile)], u: f64) -> Profile {
        let mut points: Vec<f64> = scheduled.iter().flat_map(|(_, p)| p.segs.iter().map(|s| s.start)).collect();
        points.extend([0.0, 1.0]);
        points.sort_by(f64::total_cmp);
        points.dedup();
        let rate = |t: f64, before: bool| (1.0 - scheduled.iter().map(|(rho, p)| rho * p.rate_at(t, before)).sum::<f64>()).max(0.0) / u;
        Profile::from_rates(points.windows(2).map(|w| (w[0], w[1], rate(w[0], false), rate(w[1], true))))
    }

    #[test]
    fn delayed_linear_shape() {
        for &(d0, d1) in &[(0.0, 0.0), (0.2, 0.2), (0.2, 0.6), (0.0, 1.0), (0.5, 1.0), (0.999, 0.9995)] {
            let p = Profile::delayed_linear(d0, d1);
            assert_eq!(p.share(0.0), 0.0);
            assert_eq!(p.share(d0), 0.0);
            assert!((p.share(1.0) - 1.0).abs() < 1e-12, "F(1) = {}", p.share(1.0));
            assert!((p.final_rate() - 2.0 / ((1.0 - d0) + (1.0 - d1))).abs() < 1e-12);
            // Nothing before d0, the final rate from d1 on, a jump exactly at a step.
            if d0 > 0.0 {
                assert_eq!(p.rate_at(d0, true), 0.0);
            }
            assert!((p.rate_at(d1, false) - p.final_rate()).abs() < 1e-12);
            if d1 > d0 {
                // The rate rises linearly, so the share on the ramp is a parabola.
                let mid = (d0 + d1) / 2.0;
                let tol = 1e-9 * p.final_rate();
                assert!((p.rate_at(mid, false) - p.final_rate() / 2.0).abs() < tol);
                assert!((p.share(mid) - p.share(d1) / 4.0).abs() < tol);
            } else {
                assert!((p.rate_at(d0, false) - p.final_rate()).abs() < 1e-12);
            }
            let mut prev = -1.0;
            for t in grid() {
                let v = p.share(t);
                assert!(v >= prev);
                prev = v;
            }
        }
    }

    #[test]
    fn trapezoid_shape() {
        for &(d0, d1, d2, d3) in
            &[(0.0, 0.0, 0.5, 0.5), (0.1, 0.3, 0.6, 0.9), (0.0, 0.5, 0.5, 1.0), (0.2, 0.2, 0.2, 0.7), (0.0, 0.0, 1.0, 1.0)]
        {
            let p = Profile::trapezoid(d0, d1, d2, d3);
            let r = 2.0 / (d2 + d3 - d0 - d1);
            assert!((p.max_rate() - r).abs() < 1e-12);
            assert_eq!(p.share(d0), 0.0);
            assert!((p.share(d3) - 1.0).abs() < 1e-12, "F(d3) = {}", p.share(d3));
            assert!((p.share(1.0) - 1.0).abs() < 1e-12);
            assert_eq!(p.final_rate(), if d2 < d3 || d3 < 1.0 { 0.0 } else { r });
            if d3 > d2 {
                let mid = (d2 + d3) / 2.0;
                assert!((p.rate_at(mid, false) - r / 2.0).abs() < 1e-9 * r);
            }
            if d3 < 1.0 {
                assert_eq!(p.rate_at(d3, false), 0.0);
            }
            let mut prev = -1.0;
            for t in grid() {
                let v = p.share(t);
                assert!(v >= prev);
                prev = v;
            }
            let mut hint = 0;
            let mut prev = -1.0;
            for i in 0..=10_000 {
                let t = p.quantile(i as f64 / 10_000.0, &mut hint);
                assert!(t >= prev && t <= 1.0);
                prev = t;
            }
        }
        // A delayed-linear profile is the trapezoid with the fall at the end, bit for bit.
        let (a, b) = (Profile::delayed_linear(0.2, 0.6), Profile::trapezoid(0.2, 0.6, 1.0, 1.0));
        assert_eq!(a.segs.len(), b.segs.len());
        for (x, y) in a.segs.iter().zip(&b.segs) {
            assert_eq!((x.r0, x.r1, x.share, x.c), (y.r0, y.r1, y.share, y.c));
        }
        assert_eq!(a.max_rate(), 2.0 / (2.0 - 0.2 - 0.6));
    }

    #[test]
    fn uniform_absorbs_exactly() {
        // u·F_U(τ) + Σ ρ_i·F_i(τ) = τ for every τ, with rising and falling rates.
        let scheduled = vec![
            (0.2, Profile::delayed_linear(0.3, 0.3)),
            (0.25, Profile::delayed_linear(0.1, 0.7)),
            (0.05, Profile::delayed_linear(0.0, 0.4)),
            (0.1, Profile::trapezoid(0.0, 0.0, 0.4, 0.8)),
            (0.05, Profile::trapezoid(0.5, 0.6, 0.8, 0.9)),
        ];
        let u = 0.35;
        let (fu, peak, _) = Profile::uniform(&scheduled, u);
        assert!((0.97..1.0).contains(&peak), "peak {peak}");
        for t in grid() {
            let total = u * fu.share(t) + scheduled.iter().map(|(rho, p)| rho * p.share(t)).sum::<f64>();
            assert!((total - t).abs() < 1e-12, "τ = {t}: {total}");
        }
        // The uniform rate drops at the step and is lowest at the end.
        assert!(fu.rate_at(0.3, true) > fu.rate_at(0.3, false));
        let end = (1.0 - 0.2 / 0.7 - 0.25 * 2.0 / 1.2 - 0.05 * 2.0 / 1.6) / u;
        assert!((fu.rate_at(1.0, true) - end).abs() < 1e-12);
        // The peak is where the summed rate is highest, not at the end.
        let (_, peak, _) =
            Profile::uniform(&[(0.6, Profile::trapezoid(0.0, 0.0, 0.5, 0.5)), (0.2, Profile::delayed_linear(0.5, 0.5))], 0.2);
        assert!((peak - 1.2).abs() < 1e-12, "peak {peak}");
    }

    /// The sweep agrees with evaluating every profile at every breakpoint, also when the
    /// profiles are many, distinct and steep.
    #[test]
    fn sweep_matches_direct_evaluation() {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 11) as f64 / (1u64 << 53) as f64
        };
        for &(s, steep) in &[(1usize, false), (3, false), (20, false), (300, true), (2000, true)] {
            let mut scheduled = Vec::new();
            for _ in 0..s {
                let d0 = rnd() * 0.8;
                let d1 = if steep && rnd() < 0.5 { d0 + 1e-9 * rnd() } else { d0 + rnd() * (1.0 - d0) };
                let (d2, d3) = if rnd() < 0.5 {
                    (1.0, 1.0)
                } else {
                    let d2 = d1 + rnd() * (1.0 - d1);
                    (d2, d2 + rnd() * (1.0 - d2))
                };
                let p = Profile::trapezoid(d0, d1, d2, d3);
                scheduled.push((0.3 / s as f64 / p.max_rate(), p));
            }
            let u = 0.5;
            let (fu, peak, _) = Profile::uniform(&scheduled, u);
            let direct = uniform_direct(&scheduled, u);
            assert!(peak <= 0.31, "peak {peak}");
            assert_eq!(fu.segs.len(), direct.segs.len());
            for (a, b) in fu.segs.iter().zip(&direct.segs) {
                assert_eq!((a.start, a.end), (b.start, b.end));
                let tol = 1e-12 * (1.0 + a.r0.abs().max(a.r1.abs()));
                assert!(
                    (a.r0 - b.r0).abs() <= tol && (a.r1 - b.r1).abs() <= tol,
                    "s = {s}: [{}, {}] rates {} {} vs {} {}",
                    a.start,
                    a.end,
                    a.r0,
                    a.r1,
                    b.r0,
                    b.r1
                );
            }
            for t in grid() {
                let total = u * fu.share(t) + scheduled.iter().map(|(rho, p)| rho * p.share(t)).sum::<f64>();
                assert!((total - t).abs() < 1e-11, "s = {s}, τ = {t}: {total}");
            }
        }
    }

    #[test]
    fn narrow_triangles_preserve_mass_and_adjacent_quantiles() {
        for d in [1e-8, 1e-12, 1e-16, 1e-20, 1e-300] {
            let p = Profile::trapezoid(0.0, d, d, 1.0);
            let (uniform, peak, _) = Profile::uniform(&[(0.25, p.clone())], 0.75);
            assert!(p.is_finite() && uniform.is_finite());
            assert!((peak - 0.5).abs() < 1e-14);
            for t in grid() {
                let mass = 0.25 * p.share(t) + 0.75 * uniform.share(t);
                assert!((mass - t).abs() < 2e-15, "d={d}, t={t}, mass={mass}");
            }
        }
        // Three simultaneous slope scales require more than a two-float accumulator.
        let profiles =
            [Profile::trapezoid(0.0, 1e-300, 1e-300, 1.0), Profile::delayed_linear(0.0, 1e-200), Profile::delayed_linear(0.0, 1.0)];
        let scheduled: Vec<_> = profiles.into_iter().map(|p| (0.1, p)).collect();
        let (uniform, _, _) = Profile::uniform(&scheduled, 0.7);
        for t in grid() {
            let mass = 0.7 * uniform.share(t) + scheduled.iter().map(|(rho, p)| rho * p.share(t)).sum::<f64>();
            assert!((mass - t).abs() < 2e-15, "t={t}, mass={mass}");
        }
        for n in [1e6, 1e9, 1e12, (1u64 << 46) as f64] {
            let (p, _, _) = Profile::uniform(&[(1.0 / (n + 1.0), Profile::trapezoid(0.0, 0.0, 0.0, 1.0))], n / (n + 1.0));
            for center in grid() {
                let mut y = center;
                let mut previous = p.quantile(y, &mut 0);
                for _ in 0..32 {
                    y = y.next_up().min(1.0);
                    let t = p.quantile(y, &mut 0);
                    assert!(t >= previous, "n={n}, y={y}: {t} < {previous}");
                    assert!((p.share(t) - y).abs() < 8.0 * f64::EPSILON);
                    previous = t;
                }
            }
        }
    }

    #[test]
    fn quantile_uses_the_left_endpoint_of_flat_shares() {
        // F(t) = 1/2 from 1/4 through 3/4. Split the plateau into enough segments to
        // exercise both the short hint walk and its binary-search fallback.
        let p = Profile::from_rates((0..128).map(|i| {
            let r = if (32..96).contains(&i) { 0.0 } else { 2.0 };
            (i as f64 / 128.0, (i + 1) as f64 / 128.0, r, r)
        }));
        assert_eq!(p.share(0.25), 0.5);
        assert_eq!(p.share(0.75), 0.5);
        for start in 0..p.segs.len() {
            let mut hint = start;
            assert_eq!(p.quantile(0.5, &mut hint), 0.25, "hint {start}");
            let left = p.quantile(0.5f64.next_down(), &mut hint);
            let right = p.quantile(0.5f64.next_up(), &mut hint);
            assert!(left <= 0.25 && right >= 0.75);
        }
        // Initial and final plateaus have the same left-endpoint convention.
        let p = Profile::trapezoid(0.25, 0.25, 0.75, 0.75);
        for start in 0..p.segs.len() {
            let mut hint = start;
            assert_eq!(p.quantile(0.0, &mut hint), 0.0);
            assert_eq!(p.quantile(1.0, &mut hint), 0.75);
        }
    }

    #[test]
    fn quantile_inverts_share_and_is_monotone() {
        let scheduled = vec![
            (0.3, Profile::delayed_linear(0.2, 0.6)),
            (0.1, Profile::delayed_linear(0.5, 0.5)),
            (0.1, Profile::trapezoid(0.1, 0.2, 0.3, 0.7)),
        ];
        let profiles = [
            Profile::uniform(&scheduled, 0.5).0,
            Profile::delayed_linear(0.2, 0.6),
            Profile::delayed_linear(0.5, 0.5),
            Profile::trapezoid(0.1, 0.2, 0.3, 0.7),
            Profile::trapezoid(0.0, 0.0, 0.5, 0.5),
        ];
        for p in &profiles {
            let mut hint = 0;
            let mut prev = -1.0;
            for i in 0..=100_000 {
                let y = i as f64 / 100_000.0;
                let t = p.quantile(y, &mut hint);
                assert!(t >= prev, "y = {y}: {t} < {prev}");
                prev = t;
                if y > 0.0 && y < 1.0 {
                    let back = p.share(t);
                    assert!((back - y).abs() < 1e-9 || back < y && p.share((t + 1e-9).min(1.0)) >= y, "y = {y}: F(F⁻¹(y)) = {back}");
                }
            }
            // A hint far off still gives the same answer.
            let mut far = p.segs.len() - 1;
            let mut zero = 0;
            for i in 0..=1000 {
                let y = i as f64 / 1000.0;
                assert_eq!(p.quantile(y, &mut far).to_bits(), p.quantile(y, &mut zero).to_bits());
            }
        }
    }
}
