//! Public schedules and the validated draw-rate profiles behind them.

use super::MAX_TOTAL_LEN;
use super::profile::Profile;
use crate::{ErrorKind, ScheduleReason};

/// Spreads a part's elements along a shared virtual clock from 0 to 1.
///
/// Part lengths determine how many elements each part contributes. Each
/// schedule independently assigns those elements virtual-time keys; the mix emits
/// them in increasing key order. `Uniform` is constant on this clock, with the
/// same meaning as `delayed(0.0)` or `until(1.0)`.
///
/// **Virtual time is not the fraction of the output already consumed.** A delay
/// of 0.6 does not promise a start 60% through the output. Changing another part's
/// count or schedule can change that position. Linear ramps are linear in virtual
/// time; the final mixture generally transforms both ramps and constant rates.
///
/// | Schedule | Rate in virtual time |
/// | --- | --- |
/// | [`Uniform`](Self::Uniform) (default) | Constant over `[0, 1]` |
/// | [`delayed(at)`](Self::delayed) | Starts at `at`, then stays constant |
/// | [`ramp(start, full)`](Self::ramp) | Rises from zero, then stays constant |
/// | [`until(at)`](Self::until) | Starts constant, then stops at `at` |
/// | [`fade(fade, off)`](Self::fade) | Starts constant, then falls to zero |
/// | [`trapezoid(start, full, fade, off)`](Self::trapezoid) | Rises, stays constant, then falls |
///
/// Each curve is normalized to an integral of one. If `F_i(t)` is its cumulative
/// share, `r_i(t)` its rate, `n_i` its count and `N` the total count, the continuous
/// model gives output progress `sum(n_i * F_i(t)) / N`. Its local mixture fraction
/// is `n_i * r_i(t) / sum(n_j * r_j(t))` wherever the denominator is positive.
/// Discrete items approximate these curves; exact counts and source-local order
/// are preserved. Gaps with no active parts produce no output positions.
///
/// ```
/// use dataorder::{Order, Schedule, Seq};
/// let order = Order::new(Seq::mix([
///     (Seq::source(100), Schedule::Uniform),
///     (Seq::source(100), Schedule::delayed(0.6)),
/// ]))?;
/// // At virtual time 0.6, about 60 of the first part's 100 items have appeared.
/// // The second part therefore starts around 30% through the 200-item output.
/// let first_delayed = order.iter().position(|item| item.source_ordinal == 1).unwrap();
/// assert!((59..=61).contains(&first_delayed));
/// assert_eq!(order.iter().filter(|item| item.source_ordinal == 1).count(), 100);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// A schedule belongs to its mix: repeating the mix restarts its virtual clock.
/// To span multiple epochs, repeat its parts before mixing them.
///
/// # Validation and rounding
///
/// Constructors store parameters; [`Order::new`](crate::Order::new) validates them.
/// Parameters must be finite and satisfy the ranges documented on each variant.
/// Schedules can overlap or leave gaps; they do not compete for fixed output-time
/// capacity, and no uniform filler is required.
///
/// For numerical resolution, a part must satisfy
/// `length × peak normalized rate ≤ MAX_MIX_LEN`. Very narrow transitions can
/// overflow derived coefficients. Use equal adjacent breakpoints for an abrupt
/// change. These individual numerical limits are separate from schedule overlap.
///
/// Equality compares variants and parameters using ordinary `f64` equality:
/// `-0.0` equals `0.0`, and NaN equals nothing, including itself.
/// Constructors such as [`delayed`](Self::delayed) and [`ramp`](Self::ramp)
/// return [`Trapezoid`](Self::Trapezoid), so equal breakpoints compare equal.
/// [`Uniform`](Self::Uniform) remains a distinct variant.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(deny_unknown_fields))]
#[non_exhaustive]
pub enum Schedule {
    /// A constant rate over the full virtual clock `[0, 1]`.
    /// Its fraction of the actual output changes as other parts start, ramp or stop.
    #[default]
    Uniform,
    /// A rate that rises, stays constant, then falls back to zero.
    ///
    /// It is zero before `start`, rises linearly until `full`, stays constant until
    /// `fade`, falls linearly until `off`, then stays zero. Equal neighboring
    /// breakpoints make a transition abrupt.
    /// Setting `fade` and `off` to 1 keeps the full rate through the end of the clock.
    ///
    /// Requires finite parameters with `0 ≤ start ≤ full ≤ fade ≤ off ≤ 1` and
    /// `start < off`, ensuring some time at a positive rate.
    /// Build it with [`Schedule::delayed`], [`Schedule::ramp`], [`Schedule::until`],
    /// [`Schedule::fade`] or [`Schedule::trapezoid`].
    Trapezoid {
        /// Virtual time at which the rate starts rising from zero.
        start: f64,
        /// Virtual time at which it reaches its full value.
        full: f64,
        /// Virtual time at which it starts falling.
        fade: f64,
        /// Virtual time at which it reaches zero.
        off: f64,
    },
}

impl Schedule {
    /// Creates a schedule whose rate is zero before virtual time `at`, then constant.
    /// Requires `0 ≤ at < 1`; validated when the order is built.
    ///
    /// ```
    /// use dataorder::Schedule;
    /// assert_eq!(Schedule::delayed(0.5), Schedule::Trapezoid { start: 0.5, full: 0.5, fade: 1.0, off: 1.0 });
    /// ```
    #[must_use]
    pub const fn delayed(at: f64) -> Self {
        Self::Trapezoid { start: at, full: at, fade: 1.0, off: 1.0 }
    }

    /// Creates a rate that rises linearly from zero at virtual time `start` to `full`.
    /// The rate stays constant afterward. Requires `0 ≤ start ≤ full ≤ 1` and
    /// `start < 1`; validated when the order is built.
    ///
    /// ```
    /// use dataorder::Schedule;
    /// assert_eq!(Schedule::ramp(0.2, 0.6), Schedule::Trapezoid { start: 0.2, full: 0.6, fade: 1.0, off: 1.0 });
    /// ```
    #[must_use]
    pub const fn ramp(start: f64, full: f64) -> Self {
        Self::Trapezoid { start, full, fade: 1.0, off: 1.0 }
    }

    /// Creates a schedule with a constant rate until virtual time `at`, then zero.
    /// Requires `0 < at ≤ 1`; validated when the order is built.
    ///
    /// ```
    /// use dataorder::Schedule;
    /// assert_eq!(Schedule::until(0.5), Schedule::Trapezoid { start: 0.0, full: 0.0, fade: 0.5, off: 0.5 });
    /// ```
    #[must_use]
    pub const fn until(at: f64) -> Self {
        Self::Trapezoid { start: 0.0, full: 0.0, fade: at, off: at }
    }

    /// Creates a rate that is constant, then falls to zero from virtual time `fade` to `off`.
    /// Requires `0 ≤ fade ≤ off ≤ 1` and `off > 0`; validated when the order is built.
    ///
    /// ```
    /// use dataorder::Schedule;
    /// assert_eq!(Schedule::fade(0.4, 0.8), Schedule::Trapezoid { start: 0.0, full: 0.0, fade: 0.4, off: 0.8 });
    /// ```
    #[must_use]
    pub const fn fade(fade: f64, off: f64) -> Self {
        Self::Trapezoid { start: 0.0, full: 0.0, fade, off }
    }

    /// Creates a schedule that rises, stays constant, then falls to zero.
    /// See [`Trapezoid`](Schedule::Trapezoid) for the breakpoint constraints.
    ///
    /// ```
    /// use dataorder::Schedule;
    /// assert_eq!(Schedule::trapezoid(0.1, 0.3, 0.6, 0.9), Schedule::Trapezoid { start: 0.1, full: 0.3, fade: 0.6, off: 0.9 });
    /// ```
    #[must_use]
    pub const fn trapezoid(start: f64, full: f64, fade: f64, off: f64) -> Self {
        Self::Trapezoid { start, full, fade, off }
    }

    /// The draw-rate profile of a part with `len` elements, or `None` for the shared
    /// uniform profile. Breakpoints are validated for every part; an empty part then
    /// draws from no profile. A non-empty part must also have finite derived
    /// coefficients and satisfy `len × peak rate ≤ MAX_TOTAL_LEN`.
    pub(crate) fn profile(self, len: usize) -> Result<Option<Profile>, ErrorKind> {
        let Self::Trapezoid { start, full, fade, off } = self else { return Ok(None) };
        let invalid = |reason| ErrorKind::InvalidSchedule { schedule: self, reason };
        if ![start, full, fade, off].iter().all(|d| d.is_finite()) {
            return Err(invalid(ScheduleReason::NonFiniteParameter));
        }
        if !(0.0 <= start && start <= full && full <= fade && fade <= off && off <= 1.0 && start < off) {
            return Err(invalid(ScheduleReason::InvalidBreakpoints));
        }
        if len == 0 {
            return Ok(None);
        }
        let profile = Profile::trapezoid(start, full, fade, off);
        if !profile.is_finite() {
            return Err(invalid(ScheduleReason::CoefficientOverflow));
        }
        let peak_rate = profile.max_rate();
        if len as f64 * peak_rate > MAX_TOTAL_LEN as f64 {
            return Err(ErrorKind::ScheduleTooSteep { len, peak_rate, limit: MAX_TOTAL_LEN });
        }
        Ok(Some(profile))
    }
}
