//! The public schedule type and the errors of [`Interleave::with_sampling`](super::Interleave::with_sampling).

use super::MAX_TOTAL_LEN;
use std::fmt;

/// How a sequence's elements are spread over the joint sequence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sampling {
    /// A constant rate relative to the other uniform sequences; uniform sequences absorb
    /// whatever share of the joint sequence the scheduled sequences leave free.
    Uniform,
    /// Nothing before the joint sequence is a fraction `d0` in, a rate rising linearly
    /// from zero at `d0` to its final value at `d1`, then constant until the end.
    /// `DelayedLinear(d, d)` switches the rate on abruptly at `d`.
    /// Requires `0 ≤ d0 ≤ d1 ≤ 1` and `d0 < 1`.
    DelayedLinear(f64, f64),
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
    /// Lengths and sampling slices differ in length.
    LengthMismatch,
}

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLong => write!(f, "total length exceeds {MAX_TOTAL_LEN}"),
            Self::InvalidParameter { seq, sampling } => write!(f, "sequence {seq}: invalid {sampling:?}"),
            Self::TooSteep { seq } => write!(f, "sequence {seq}: too long for the steepness of its schedule"),
            Self::Overcommitted { demand } => write!(f, "scheduled sequences need {:.1}% of the draw rate at the end", demand * 100.0),
            Self::LengthMismatch => write!(f, "lengths and sampling have different lengths"),
        }
    }
}

impl std::error::Error for SamplingError {}
