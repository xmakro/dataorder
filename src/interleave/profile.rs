//! Draw-rate profiles and their integrals. A profile is a sequence of segments covering
//! `[0, 1]` on each of which the rate changes linearly; consecutive segments may meet at
//! different rates (a jump). That is exactly what a [`Sampling`](super::Sampling) describes,
//! and it is also the shape of the uniform sequences' rate, one minus the scheduled rates.
//! The integral of a profile is the share function, the fraction of a sequence drawn by a
//! given progress; `quantile` inverts it and is what every element's key comes from. It is
//! monotone even under floating-point rounding, which the exact-seek guarantee relies on.
//! `quantile` is `#[inline(always)]` for the same measured reason as the tournament tree:
//! it is the key computation of every element of a walk.

/// A nonnegative, piecewise-linear draw rate over joint progress, with its running integral.
#[derive(Clone, Debug)]
pub(crate) struct Profile {
    segs: Vec<Segment>,
}

/// On `[start, end]` the rate goes linearly from `r0` to `r1`; `share` is the integral of
/// the profile up to `start`, and `c = (r1 − r0) / (2·(end − start))` the coefficient of
/// `x²` in the share, kept so that keys need no division. A constant rate (`c = 0`) also
/// keeps `inv_r0 = 1/r0` so that its keys need neither division nor square root; a rising
/// one keeps `inv_2c = 1/(2c)` and every segment `c4 = 4c`.
#[derive(Clone, Copy, Debug)]
struct Segment {
    start: f64,
    end: f64,
    r0: f64,
    r1: f64,
    share: f64,
    c: f64,
    inv_r0: f64,
    inv_2c: f64,
    c4: f64,
}

impl Profile {
    /// From consecutive `(start, end, r0, r1)` segments covering `[0, 1]` (empty ones are
    /// dropped); the shares are accumulated by the trapezoid rule, which is exact here.
    fn from_rates(rates: impl IntoIterator<Item = (f64, f64, f64, f64)>) -> Self {
        let mut share = 0.0;
        let mut segs = Vec::new();
        for (start, end, r0, r1) in rates {
            if end > start {
                let inv_r0 = if r1 == r0 && r0 > 0.0 { 1.0 / r0 } else { 0.0 };
                let c = (r1 - r0) / (2.0 * (end - start));
                let inv_2c = if c > 0.0 { 1.0 / (2.0 * c) } else { 0.0 };
                segs.push(Segment { start, end, r0, r1, share, c, inv_r0, inv_2c, c4: 4.0 * c });
                share += (r0 + r1) / 2.0 * (end - start);
            }
        }
        Self { segs }
    }

    /// `DelayedLinear { start: d0, full: d1 }`: zero until `d0`, rising linearly to the final rate at `d1`,
    /// then constant. The final rate `r = 2/(2 − d0 − d1)` makes the total share one.
    pub(crate) fn delayed_linear(d0: f64, d1: f64) -> Self {
        let r = 2.0 / (2.0 - d0 - d1);
        Self::from_rates([(0.0, d0, 0.0, 0.0), (d0, d1, 0.0, r), (d1, 1.0, r, r)])
    }

    /// The uniform sequences' profile `(1 − Σ ρ_i·rate_i) / u` for the scheduled profiles
    /// with their element fractions `ρ_i`, `u` being the uniform fraction of all elements.
    pub(crate) fn uniform(scheduled: &[(f64, Self)], u: f64) -> Self {
        let mut points: Vec<f64> = scheduled.iter().flat_map(|(_, p)| p.segs.iter().map(|s| s.start)).collect();
        points.extend([0.0, 1.0]);
        points.sort_by(f64::total_cmp);
        points.dedup();
        let rate = |t: f64, before: bool| (1.0 - scheduled.iter().map(|(rho, p)| rho * p.rate_at(t, before)).sum::<f64>()).max(0.0) / u;
        Self::from_rates(points.windows(2).map(|w| (w[0], w[1], rate(w[0], false), rate(w[1], true))))
    }

    /// The rate just after `t` (`before == false`) or just before it.
    fn rate_at(&self, t: f64, before: bool) -> f64 {
        let s = &self.segs[self.segs.partition_point(|s| if before { s.start < t } else { s.start <= t }).saturating_sub(1)];
        s.r0 + (s.r1 - s.r0) * ((t - s.start) / (s.end - s.start)).clamp(0.0, 1.0)
    }

    /// `f'(1⁻)`, the rate at the end.
    pub(crate) fn final_rate(&self) -> f64 {
        self.segs.last().map_or(0.0, |s| s.r1)
    }

    /// The share drawn by progress `t`: the integral of the rate up to `t`.
    pub(crate) fn share(&self, t: f64) -> f64 {
        let s = &self.segs[self.segs.partition_point(|s| s.start <= t).saturating_sub(1)];
        let x = (t - s.start).clamp(0.0, s.end - s.start);
        s.share + x * (s.r0 + s.c * x)
    }

    /// The smallest `t` whose share is at least `y`. `hint` is the segment to try first and
    /// is updated; consecutive keys of a sequence mostly stay on one segment or move on.
    ///
    /// Nondecreasing in `y` even under rounding: the segment index is monotone because
    /// shares are, the result is clamped to the segment, and inside a segment every operation
    /// is a correctly rounded monotone function of the previous one.
    #[inline(always)]
    pub(crate) fn quantile(&self, y: f64, hint: &mut usize) -> f64 {
        let mut m = *hint;
        while m + 1 < self.segs.len() && y >= self.segs[m + 1].share {
            m += 1;
        }
        while m > 0 && y < self.segs[m].share {
            m -= 1;
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
                // Monotone in `z`: the root is, and the product with a positive constant is.
                (root - s.r0) * s.inv_2c
            } else if s.r0 + root > 0.0 {
                2.0 * z / (s.r0 + root)
            } else {
                0.0
            }
        };
        (s.start + x).clamp(s.start, s.end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> impl Iterator<Item = f64> {
        (0..=1000).map(|i| i as f64 / 1000.0)
    }

    #[test]
    fn delayed_linear_shape() {
        for &(d0, d1) in &[(0.0, 0.0), (0.2, 0.2), (0.2, 0.6), (0.0, 1.0), (0.5, 1.0), (0.999, 0.9995)] {
            let p = Profile::delayed_linear(d0, d1);
            assert_eq!(p.share(0.0), 0.0);
            assert_eq!(p.share(d0), 0.0);
            assert!((p.share(1.0) - 1.0).abs() < 1e-12, "F(1) = {}", p.share(1.0));
            assert!((p.final_rate() - 2.0 / (2.0 - d0 - d1)).abs() < 1e-12);
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
    fn uniform_absorbs_exactly() {
        // u·F_U(τ) + Σ ρ_i·F_i(τ) = τ for every τ.
        let scheduled = vec![
            (0.2, Profile::delayed_linear(0.3, 0.3)),
            (0.25, Profile::delayed_linear(0.1, 0.7)),
            (0.05, Profile::delayed_linear(0.0, 0.4)),
        ];
        let u = 0.5;
        let fu = Profile::uniform(&scheduled, u);
        for t in grid() {
            let total = u * fu.share(t) + scheduled.iter().map(|(rho, p)| rho * p.share(t)).sum::<f64>();
            assert!((total - t).abs() < 1e-12, "τ = {t}: {total}");
        }
        // The uniform rate drops at the step and is lowest at the end.
        assert!(fu.rate_at(0.3, true) > fu.rate_at(0.3, false));
        assert!((fu.rate_at(1.0, true) - (1.0 - 0.2 / 0.7 - 0.25 * 2.0 / 1.2 - 0.05 * 2.0 / 1.6) / u).abs() < 1e-12);
    }

    #[test]
    fn quantile_inverts_share_and_is_monotone() {
        let scheduled = vec![(0.3, Profile::delayed_linear(0.2, 0.6)), (0.1, Profile::delayed_linear(0.5, 0.5))];
        let profiles = [Profile::uniform(&scheduled, 0.6), Profile::delayed_linear(0.2, 0.6), Profile::delayed_linear(0.5, 0.5)];
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
