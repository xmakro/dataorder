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
}

impl PartialEq for Sampling {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Uniform, Self::Uniform) => true,
            (Self::DelayedLinear { start: a, full: b }, Self::DelayedLinear { start: c, full: d }) => {
                float_bits(*a) == float_bits(*c) && float_bits(*b) == float_bits(*d)
            }
            _ => false,
        }
    }
}

impl Eq for Sampling {}

impl Hash for Sampling {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        if let Self::DelayedLinear { start, full } = self {
            float_bits(*start).hash(state);
            float_bits(*full).hash(state);
        }
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
    /// The scheduled sequences' final rates sum to `demand` (> 1) times the total draw
    /// rate, leaving nothing for the uniform sequences at the end.
    Overcommitted { demand: f64 },
}

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLong => write!(f, "total length exceeds {MAX_TOTAL_LEN}"),
            Self::InvalidParameter { seq, sampling } => write!(f, "sequence {seq}: invalid {sampling:?}"),
            Self::TooSteep { seq } => write!(f, "sequence {seq}: too long for the steepness of its schedule"),
            Self::Overcommitted { demand } => write!(f, "scheduled sequences need {:.1}% of the draw rate at the end", demand * 100.0),
        }
    }
}

impl std::error::Error for SamplingError {}
