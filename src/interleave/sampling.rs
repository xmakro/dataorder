//! The public schedule type and the errors of [`Interleave::with_sampling`](super::Interleave::with_sampling).

use super::MAX_TOTAL_LEN;
use std::fmt;

/// How a sequence's elements are spread over the joint sequence.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    #[must_use]
    pub const fn delayed(at: f64) -> Self {
        Self::DelayedLinear { start: at, full: at }
    }

    /// Nothing before `start`, a rate rising linearly until `full`, then constant.
    #[must_use]
    pub const fn ramp(start: f64, full: f64) -> Self {
        Self::DelayedLinear { start, full }
    }
}

/// Why a sampling configuration was rejected.
#[derive(Clone, Debug, PartialEq)]
pub enum SamplingError {
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
