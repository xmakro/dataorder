//! The public schedule type and the errors of `Interleave::with_sampling`.

use super::MAX_TOTAL_LEN;
use std::fmt;
use std::hash::{Hash, Hasher};

/// The bits of a float with `-0.0` taken as `0.0`: what equality and hashing of schedules
/// and weights compare.
pub(crate) fn float_bits(x: f64) -> u64 {
    (x + 0.0).to_bits()
}

/// How a sequence's elements are spread over the joint sequence. Equality and hashing
/// compare the parameters bit for bit (with `-0.0` taken as `0.0`).
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(deny_unknown_fields))]
#[non_exhaustive]
pub enum Sampling {
    /// A constant rate relative to the other uniform sequences; uniform sequences absorb
    /// whatever share of the joint sequence the scheduled sequences leave free.
    Uniform,
    /// Nothing before the joint sequence is a fraction `start` in, a rate rising linearly
    /// from zero at `start` to its final value at `full`, then constant until the end.
    /// `start == full` switches the rate on abruptly. Requires `0 ≤ start ≤ full ≤ 1` and
    /// `start < 1`. Build it with [`Sampling::delayed`] or [`Sampling::ramp`].
    DelayedLinear {
        /// Progress at which the rate starts rising from zero.
        start: f64,
        /// Progress at which it reaches its final value.
        full: f64,
    },
    /// A rate that is zero until `start`, rises linearly to its full value at `full`, stays
    /// there until `fade`, falls linearly to zero at `off` and stays zero: a curriculum
    /// source that is phased out, or [`DelayedLinear`](Sampling::DelayedLinear) with an
    /// end. Equal neighbours make the change abrupt. Requires
    /// `0 ≤ start ≤ full ≤ fade ≤ off ≤ 1` and `start + full < fade + off` (some time at a
    /// positive rate). Build it with [`Sampling::until`], [`Sampling::fading`] or
    /// [`Sampling::trapezoid`].
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
    /// Nothing before progress `at`, then a constant rate.
    ///
    /// ```
    /// use dataorder::Sampling;
    /// assert_eq!(Sampling::delayed(0.5), Sampling::DelayedLinear { start: 0.5, full: 0.5 });
    /// ```
    #[must_use]
    pub const fn delayed(at: f64) -> Self {
        Self::DelayedLinear { start: at, full: at }
    }

    /// Nothing before `start`, a rate rising linearly until `full`, then constant.
    ///
    /// ```
    /// use dataorder::Sampling;
    /// assert_eq!(Sampling::ramp(0.2, 0.6), Sampling::DelayedLinear { start: 0.2, full: 0.6 });
    /// ```
    #[must_use]
    pub const fn ramp(start: f64, full: f64) -> Self {
        Self::DelayedLinear { start, full }
    }

    /// A constant rate from the start, switched off at progress `at`: the mirror image of
    /// [`delayed`](Sampling::delayed).
    ///
    /// ```
    /// use dataorder::Sampling;
    /// assert_eq!(Sampling::until(0.5), Sampling::Trapezoid { start: 0.0, full: 0.0, fade: 0.5, off: 0.5 });
    /// ```
    #[must_use]
    pub const fn until(at: f64) -> Self {
        Self::Trapezoid { start: 0.0, full: 0.0, fade: at, off: at }
    }

    /// A constant rate from the start, falling linearly to zero between `fade` and `off`:
    /// the mirror image of [`ramp`](Sampling::ramp).
    ///
    /// ```
    /// use dataorder::Sampling;
    /// assert_eq!(Sampling::fading(0.4, 0.8), Sampling::Trapezoid { start: 0.0, full: 0.0, fade: 0.4, off: 0.8 });
    /// ```
    #[must_use]
    pub const fn fading(fade: f64, off: f64) -> Self {
        Self::Trapezoid { start: 0.0, full: 0.0, fade, off }
    }

    /// Zero until `start`, rising until `full`, constant until `fade`, falling to zero at
    /// `off`; see [`Trapezoid`](Sampling::Trapezoid).
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
    /// A schedule parameter is out of range or not finite.
    InvalidParameter { seq: usize, sampling: Sampling },
    /// `length × final_rate` of a scheduled sequence exceeds [`MAX_TOTAL_LEN`].
    TooSteep { seq: usize },
    /// The scheduled sequences' rates sum to `demand` (> 1) times the total draw rate at
    /// some progress, leaving nothing for the uniform sequences there.
    Overcommitted { demand: f64 },
}

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLong => write!(f, "total length exceeds {MAX_TOTAL_LEN}"),
            Self::InvalidParameter { seq, sampling } => write!(f, "sequence {seq}: invalid {sampling:?}"),
            Self::TooSteep { seq } => write!(f, "sequence {seq}: too long for the steepness of its schedule"),
            Self::Overcommitted { demand } => write!(f, "scheduled sequences need {:.1}% of the draw rate at their peak", demand * 100.0),
        }
    }
}

impl std::error::Error for SamplingError {}
