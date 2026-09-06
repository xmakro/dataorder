//! Public sampling schedules and internal schedule validation errors.

use super::MAX_TOTAL_LEN;
use crate::float_bits;
use std::fmt;
use std::hash::{Hash, Hasher};

/// Controls when a part's elements appear in a mix.
///
/// A part's length determines how many elements it contributes; its schedule
/// spreads those elements over the mix. Progress runs from 0 at the start to 1 at
/// the end. For example, `delayed(0.5)` places a part around the second half.
/// Schedules describe ideal progress: rounding to individual positions can move
/// elements slightly across a breakpoint.
///
/// | Schedule | Draw rate |
/// | --- | --- |
/// | [`Uniform`](Self::Uniform) (default) | Fills the space left by scheduled parts |
/// | [`delayed(at)`](Self::delayed) | Starts at `at`, then stays constant |
/// | [`ramp(start, full)`](Self::ramp) | Rises from zero, then stays constant |
/// | [`until(at)`](Self::until) | Starts constant, then stops at `at` |
/// | [`fading(fade, off)`](Self::fading) | Starts constant, then falls to zero |
/// | [`trapezoid(start, full, fade, off)`](Self::trapezoid) | Rises, stays constant, then falls |
///
/// Uniform parts keep a constant rate relative to each other, in proportion to
/// their lengths. Their combined rate changes to fill the space left by scheduled
/// parts. A schedule belongs to its mix: repeating that mix restarts the schedule.
///
/// ```
/// use dataorder::{Order, Sampling, Seq};
/// let seq = Seq::mix_with([
///     (Seq::source(600), Sampling::Uniform),
///     (Seq::source(200), Sampling::until(0.5)),   // Phase out around halfway.
///     (Seq::source(100), Sampling::delayed(0.5)), // Introduce around halfway.
/// ]);
/// let order = Order::new(seq)?;
/// let positions = |source: usize| {
///     order.iter(..)
///         .enumerate()
///         .filter(|&(_, (&s, _))| s == source)
///         .map(|(p, _)| p)
///         .collect::<Vec<_>>()
/// };
/// // Halfway is position 450. Allow a few positions for discrete rounding.
/// assert!(positions(200).iter().all(|&p| p < 455));
/// assert!(positions(100).iter().all(|&p| p >= 445));
/// assert_eq!(positions(600).len(), 600);
/// # Ok::<(), dataorder::Error>(())
/// ```
///
/// # Validation and rounding
///
/// Constructors store parameters; [`Order::new`](crate::Order::new) validates them.
/// Parameters must be finite and satisfy the ranges documented on each variant.
/// The combined scheduled rate must fit the mix's capacity. For example, placing
/// 75% of the elements in the last half would require 150% of its available rate
/// and returns [`Overcommitted`](crate::ErrorKind::Overcommitted).
///
/// Excess demand up to 10⁻⁹ is accepted for numerical rounding. In exact arithmetic,
/// a feasible schedule places each element within `k` positions of its ideal rank,
/// where `k` is the number of non-empty parts. Accepted excess demand can add drift
/// proportional to the mix length, so that bound is not a guarantee for every
/// accepted configuration, especially near [`MAX_MIX_LEN`](crate::MAX_MIX_LEN).
///
/// A scheduled part must satisfy `length × peak rate ≤ MAX_MIX_LEN`, where the rate
/// is normalized so the part's total share is 1. Very narrow transitions can also
/// overflow derived coefficients. Use equal adjacent breakpoints for an abrupt change.
///
/// Equality and hashing compare parameter bits, treating `-0.0` as `0.0`.
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(deny_unknown_fields))]
#[non_exhaustive]
pub enum Sampling {
    /// Fills the space left by scheduled parts, in proportion to this part's length
    /// relative to the other uniform parts.
    #[default]
    Uniform,
    /// A rate that rises from zero at `start` to its final value at `full`, then
    /// stays constant. The rate is zero before `start`.
    ///
    /// `start == full` switches the rate on abruptly. Requires finite parameters
    /// with `0 ≤ start ≤ full ≤ 1` and `start < 1`. Build it with
    /// [`Sampling::delayed`] or [`Sampling::ramp`].
    DelayedLinear {
        /// Progress at which the rate starts rising from zero.
        start: f64,
        /// Progress at which it reaches its final value.
        full: f64,
    },
    /// A rate that rises, stays constant, then falls back to zero.
    ///
    /// It is zero before `start`, rises linearly until `full`, stays constant until
    /// `fade`, falls linearly until `off`, then stays zero. Equal neighboring
    /// breakpoints make a transition abrupt.
    ///
    /// Requires finite parameters with `0 ≤ start ≤ full ≤ fade ≤ off ≤ 1` and
    /// `start + full < fade + off`, ensuring some time at a positive rate.
    /// Build it with [`Sampling::until`], [`Sampling::fading`] or [`Sampling::trapezoid`].
    Trapezoid {
        /// Progress at which the rate starts rising from zero.
        start: f64,
        /// Progress at which it reaches its full value.
        full: f64,
        /// Progress at which it starts falling.
        fade: f64,
        /// Progress at which it reaches zero.
        off: f64,
    },
}

impl Sampling {
    /// Creates a schedule whose rate is zero before `at`, then constant.
    /// Requires `0 ≤ at < 1`; validated when the order is built.
    ///
    /// ```
    /// use dataorder::Sampling;
    /// assert_eq!(Sampling::delayed(0.5), Sampling::DelayedLinear { start: 0.5, full: 0.5 });
    /// ```
    #[must_use]
    pub const fn delayed(at: f64) -> Self {
        Self::DelayedLinear { start: at, full: at }
    }

    /// Creates a schedule whose rate rises from zero at `start` to full at `full`.
    /// The rate stays constant afterward. Requires `0 ≤ start ≤ full ≤ 1` and
    /// `start < 1`; validated when the order is built.
    ///
    /// ```
    /// use dataorder::Sampling;
    /// assert_eq!(Sampling::ramp(0.2, 0.6), Sampling::DelayedLinear { start: 0.2, full: 0.6 });
    /// ```
    #[must_use]
    pub const fn ramp(start: f64, full: f64) -> Self {
        Self::DelayedLinear { start, full }
    }

    /// Creates a schedule with a constant rate until `at`, then zero.
    /// Requires `0 < at ≤ 1`; validated when the order is built.
    ///
    /// ```
    /// use dataorder::Sampling;
    /// assert_eq!(Sampling::until(0.5), Sampling::Trapezoid { start: 0.0, full: 0.0, fade: 0.5, off: 0.5 });
    /// ```
    #[must_use]
    pub const fn until(at: f64) -> Self {
        Self::Trapezoid { start: 0.0, full: 0.0, fade: at, off: at }
    }

    /// Creates a schedule with a constant rate that falls to zero from `fade` to `off`.
    /// Requires `0 ≤ fade ≤ off ≤ 1` and `off > 0`; validated when the order is built.
    ///
    /// ```
    /// use dataorder::Sampling;
    /// assert_eq!(Sampling::fading(0.4, 0.8), Sampling::Trapezoid { start: 0.0, full: 0.0, fade: 0.4, off: 0.8 });
    /// ```
    #[must_use]
    pub const fn fading(fade: f64, off: f64) -> Self {
        Self::Trapezoid { start: 0.0, full: 0.0, fade, off }
    }

    /// Creates a schedule that rises, stays constant, then falls to zero.
    /// See [`Trapezoid`](Sampling::Trapezoid) for the breakpoint constraints.
    ///
    /// ```
    /// use dataorder::Sampling;
    /// assert_eq!(Sampling::trapezoid(0.1, 0.3, 0.6, 0.9), Sampling::Trapezoid { start: 0.1, full: 0.3, fade: 0.6, off: 0.9 });
    /// ```
    #[must_use]
    pub const fn trapezoid(start: f64, full: f64, fade: f64, off: f64) -> Self {
        Self::Trapezoid { start, full, fade, off }
    }

    /// The parameters, as the bits equality and hashing compare.
    fn bits(&self) -> [u64; 4] {
        match *self {
            Self::Uniform => [0; 4],
            Self::DelayedLinear { start, full } => [float_bits(start), float_bits(full), 0, 0],
            Self::Trapezoid { start, full, fade, off } => [float_bits(start), float_bits(full), float_bits(fade), float_bits(off)],
        }
    }
}

impl PartialEq for Sampling {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other) && self.bits() == other.bits()
    }
}

impl Eq for Sampling {}

impl Hash for Sampling {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        self.bits().hash(state);
    }
}

/// Why a sampling configuration was rejected.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SamplingError {
    /// The total length exceeds [`MAX_TOTAL_LEN`].
    TooLong,
    /// A breakpoint or derived profile coefficient is invalid.
    InvalidParameter { seq: usize, sampling: Sampling, detail: crate::SamplingDetail },
    /// The combined profile cannot be represented by finite coefficients.
    Overflow,
    /// `length × peak rate` of a scheduled sequence exceeds [`MAX_TOTAL_LEN`].
    TooSteep { seq: usize, len: u64, peak_rate: f64 },
    /// The scheduled sequences' rates sum to `demand` (> 1) times the total draw rate at
    /// some progress, leaving nothing for the uniform sequences there.
    Overcommitted { demand: f64, start: f64, end: f64 },
}

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLong => write!(f, "total length exceeds {MAX_TOTAL_LEN}"),
            Self::InvalidParameter { seq, sampling, .. } => write!(f, "sequence {seq}: invalid {sampling:?}"),
            Self::Overflow => write!(f, "combined sampling profile exceeds floating-point range"),
            Self::TooSteep { seq, .. } => write!(f, "sequence {seq}: too long for the steepness of its schedule"),
            Self::Overcommitted { demand, start, end } => {
                write!(f, "scheduled sequences need {}% of the draw rate at their peak (progress {start}..{end})", demand * 100.0)
            }
        }
    }
}

impl std::error::Error for SamplingError {}
