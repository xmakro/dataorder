//! The errors of [`Order::new`](crate::Order::new).

use crate::interleave::{SamplingError, MAX_TOTAL_LEN};
use crate::Sampling;
use std::fmt;

/// Why a configuration was rejected by [`Order::new`](crate::Order::new).
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// A `Skip` of `n` positions from a sequence of `len < n`.
    SkipOutOfRange {
        /// Positions to skip.
        n: usize,
        /// Length of the sequence.
        len: usize,
    },
    /// A `Take` of `n` positions from a sequence of `len < n`.
    TakeOutOfRange {
        /// Positions to take.
        n: usize,
        /// Length of the sequence.
        len: usize,
    },
    /// A stride with `step == 0`.
    ZeroStep,
    /// The order is longer than `usize::MAX`, an intermediate length does not fit in 64 bits,
    /// there are more than 2³² sources, or a mix has 2³¹ parts or more.
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
    /// A weight of a weighted mix is negative or not finite.
    InvalidWeight {
        /// Index of the part in the mix.
        part: usize,
        /// Its weight.
        weight: f64,
    },
    /// The weights of a weighted mix sum to zero.
    ZeroWeights,
    /// A part of a weighted mix has a positive share but no elements.
    EmptyWeightedPart {
        /// Index of the part in the mix.
        part: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SkipOutOfRange { n, len } => write!(f, "cannot skip {n} of {len} positions"),
            Self::TakeOutOfRange { n, len } => write!(f, "cannot take {n} of {len} positions"),
            Self::ZeroStep => write!(f, "stride step is zero"),
            Self::Overflow => write!(f, "order longer than usize::MAX, or a length beyond 64 bits"),
            Self::MixTooLong => write!(f, "mix longer than {MAX_TOTAL_LEN}"),
            Self::InvalidSampling { part, sampling } => write!(f, "mix part {part}: invalid {sampling:?}"),
            Self::TooSteep { part } => write!(f, "mix part {part}: too long for the steepness of its schedule"),
            Self::Overcommitted { demand } => write!(f, "scheduled mix parts need {:.1}% of the draw rate at the end", demand * 100.0),
            Self::InvalidWeight { part, weight } => write!(f, "weighted mix part {part}: invalid weight {weight}"),
            Self::ZeroWeights => write!(f, "weighted mix: the weights sum to zero"),
            Self::EmptyWeightedPart { part } => write!(f, "weighted mix part {part} has a share but no elements"),
        }
    }
}

impl std::error::Error for Error {}

impl From<SamplingError> for Error {
    fn from(e: SamplingError) -> Self {
        match e {
            SamplingError::TooLong => Self::MixTooLong,
            SamplingError::InvalidParameter { seq, sampling } => Self::InvalidSampling { part: seq, sampling },
            SamplingError::TooSteep { seq } => Self::TooSteep { part: seq },
            SamplingError::Overcommitted { demand } => Self::Overcommitted { demand },
        }
    }
}
