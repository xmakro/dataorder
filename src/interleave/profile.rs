//! Draw-rate profiles, cumulative shares and inverse lookup.
//!
//! A profile consists of linear rate segments over `[0, 1]`, with jumps allowed
//! between segments. Integrating the rate gives the fraction of a part drawn by
//! each virtual time. `quantile` inverts that share to compute an element's key.
//! Its result must stay monotone under rounding for exact seeks to work.
//! It is explicitly inlined because every iteration step computes a key.
//!
//! Keep lookup's multiply and add operations separate to preserve reproducible
//! rounding. Each independently normalized profile has at most five segments.

/// A nonnegative, piecewise-linear draw rate over virtual time, with its running integral.
#[derive(Clone, Debug)]
pub(crate) struct Profile {
    segs: Vec<Segment>,
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
        Self { segs }
    }

    /// `Trapezoid { start: d0, full: d1, fade: d2, off: d3 }`: zero until `d0`, rising
    /// linearly to the full rate at `d1`, constant until `d2`, falling linearly to zero at
    /// `d3`, zero afterwards. The full rate `r = 2/((d2 − d1) + (d3 − d0))` makes the total
    /// share one. Subtract endpoints before adding widths to avoid cancellation.
    pub(crate) fn trapezoid(d0: f64, d1: f64, d2: f64, d3: f64) -> Self {
        let r = 2.0 / ((d2 - d1) + (d3 - d0));
        Self::from_rates([(0.0, d0, 0.0, 0.0), (d0, d1, 0.0, r), (d1, d2, r, r), (d2, d3, r, 0.0), (d3, 1.0, 0.0, 0.0)])
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

    /// The share drawn by virtual time `t`: the integral of the rate up to `t`.
    pub(crate) fn share(&self, t: f64) -> f64 {
        let s = &self.segs[self.segs.partition_point(|s| s.start <= t).saturating_sub(1)];
        let x = (t - s.start).clamp(0.0, s.end - s.start);
        s.share + x * (s.r0 + s.c * x)
    }

    /// Inverts share `y` to virtual time, using the left endpoint of a flat interval.
    /// The hint walks within at most five segments. Each segment's inverse uses
    /// monotone operations; clamping to its endpoints preserves monotonicity across
    /// segments too. Rising segments always start at rate zero.
    #[inline(always)]
    pub(crate) fn quantile(&self, y: f64, hint: &mut usize) -> f64 {
        // The last segment whose starting share is strictly below `y` (or the first
        // segment at zero). Equality belongs to the preceding segment, so a flat share
        // interval is inverted at its left endpoint, including with a hint beyond it.
        let mut m = *hint;
        loop {
            if m + 1 < self.segs.len() && y > self.segs[m + 1].share {
                m += 1;
            } else if m > 0 && y <= self.segs[m].share {
                m -= 1;
            } else {
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
                root * s.inv_2c
            } else if s.r0 + root > 0.0 {
                2.0 * z / (s.r0 + root)
            } else {
                0.0
            }
        };
        (s.start + x).clamp(s.start, s.end)
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> impl Iterator<Item = f64> {
        (0..=1000).map(|i| i as f64 / 1000.0)
    }

    #[test]
    fn ramp_and_constant_shapes() {
        for &(d0, d1) in &[(0.0, 0.0), (0.2, 0.2), (0.2, 0.6), (0.0, 1.0), (0.5, 1.0), (0.999, 0.9995)] {
            let p = Profile::trapezoid(d0, d1, 1.0, 1.0);
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
        assert!((Profile::trapezoid(0.2, 0.6, 1.0, 1.0).max_rate() - 5.0 / 3.0).abs() <= f64::EPSILON);
    }

    #[test]
    fn inverse_is_monotone_and_independent_of_hint() {
        for p in [
            Profile::trapezoid(0.0, 0.0, 1.0, 1.0),
            Profile::trapezoid(0.2, 0.6, 1.0, 1.0),
            Profile::trapezoid(0.1, 0.3, 0.5, 0.9),
            Profile::trapezoid(0.25, 0.25, 0.75, 0.75),
            Profile::trapezoid(0.0, 1e-300, 1e-300, 1.0),
        ] {
            for center in grid() {
                let mut y = center;
                let mut previous = p.quantile(y, &mut 0);
                for _ in 0..32 {
                    y = y.next_up().min(1.0);
                    let t = p.quantile(y, &mut (p.segs.len() - 1));
                    assert_eq!(t, p.quantile(y, &mut 0));
                    assert!(t >= previous, "y={y}: {t} < {previous}");
                    assert!((p.share(t) - y).abs() < 8.0 * f64::EPSILON);
                    previous = t;
                }
            }
        }
    }

    #[test]
    fn inverse_uses_left_endpoint_of_flat_shares() {
        let p = Profile::trapezoid(0.25, 0.25, 0.75, 0.75);
        for start in 0..p.segs.len() {
            assert_eq!(p.quantile(0.0, &mut { start }), 0.0);
            assert_eq!(p.quantile(1.0, &mut { start }), 0.75);
        }
    }
}
