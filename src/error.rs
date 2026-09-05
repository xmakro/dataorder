//! The errors of [`Order::compile`](crate::Order::compile).

use crate::interleave::{SamplingError, MAX_TOTAL_LEN};
use crate::Sampling;
use std::fmt;

/// Why a configuration was rejected by [`Order::compile`](crate::Order::compile).
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// A slice's `start..end` does not fit in the `len` of its inner sequence.
    SliceOutOfRange {
        /// First position of the slice.
        start: usize,
        /// One past its last position.
        end: usize,
        /// Length of the sliced sequence.
        len: usize,
    },
    /// A stride with `step == 0`.
    ZeroStep,
    /// The order is longer than `usize::MAX`, an intermediate length does not fit in 64 bits,
    /// or there are more than 2³² sources.
    Overflow,
    /// The total length of a mix exceeds 2⁴⁶.
    MixTooLong,
    /// A schedule parameter of mix part `part` is out of range or not finite.
    InvalidSampling {
        /// Index of the part in the mix.
        part: usize,
        /// Its schedule.
        sampling: Sampling,
    },
    /// Mix part `part` is too long for the steepness of its schedule
    /// (`length × final_rate` exceeds 2⁴⁶).
    TooSteep {
        /// Index of the part in the mix.
        part: usize,
    },
    /// The scheduled parts of a mix need `demand` (> 1) times the whole draw rate at the
    /// end, leaving nothing for the uniform parts.
    Overcommitted {
        /// The scheduled parts' final rates, summed, as a fraction of the whole draw rate.
        demand: f64,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SliceOutOfRange { start, end, len } => write!(f, "slice {start}..{end} out of range for length {len}"),
            Self::ZeroStep => write!(f, "stride step is zero"),
            Self::Overflow => write!(f, "order longer than usize::MAX, or a length beyond 64 bits"),
            Self::MixTooLong => write!(f, "mix longer than {MAX_TOTAL_LEN}"),
            Self::InvalidSampling { part, sampling } => write!(f, "mix part {part}: invalid {sampling:?}"),
            Self::TooSteep { part } => write!(f, "mix part {part}: too long for the steepness of its schedule"),
            Self::Overcommitted { demand } => write!(f, "scheduled mix parts need {:.1}% of the draw rate at the end", demand * 100.0),
        }
    }
}

impl std::error::Error for Error {}

impl From<SamplingError> for Error {
    fn from(e: SamplingError) -> Self {
        match e {
            SamplingError::TooLong => Error::MixTooLong,
            SamplingError::InvalidParameter { seq, sampling } => Error::InvalidSampling { part: seq, sampling },
            SamplingError::TooSteep { seq } => Error::TooSteep { part: seq },
            SamplingError::Overcommitted { demand } => Error::Overcommitted { demand },
            SamplingError::LengthMismatch => unreachable!("dataorder: a mix passes one schedule per part"),
        }
    }
}
