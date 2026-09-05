//! The errors of [`Order::new`](crate::Order::new).

use crate::Sampling;
use crate::interleave::{MAX_TOTAL_LEN, SamplingError};
use std::fmt;

/// Why and where [`Order::new`](crate::Order::new) rejected a configuration: the
/// [`kind`](Error::kind) of the problem and the [`path`](Error::path) of the node it was
/// found at.
#[derive(Clone, Debug, PartialEq)]
pub struct Error {
    kind: ErrorKind,
    path: Vec<usize>,
}

impl Error {
    pub(crate) fn new(kind: ErrorKind, path: Vec<usize>) -> Self {
        Self { kind, path }
    }

    /// What went wrong.
    #[must_use]
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }

    /// Where: the indices of the children followed from the root of the
    /// [`Seq`](crate::Seq) to the node the problem was found at (the part index under a
    /// `Concat`, `Mix` or `Weighted`, `0` under a node with one child). Empty for the root.
    #[must_use]
    pub fn path(&self) -> &[usize] {
        &self.path
    }

    /// The kind, discarding the path.
    #[must_use]
    pub fn into_kind(self) -> ErrorKind {
        self.kind
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.kind.fmt(f)?;
        if self.path.is_empty() {
            write!(f, " (at the root)")
        } else {
            write!(f, " (at node")?;
            for i in &self.path {
                write!(f, "/{i}")?;
            }
            write!(f, ")")
        }
    }
}

impl std::error::Error for Error {}

/// What [`Order::new`](crate::Order::new) found wrong with a configuration.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ErrorKind {
    /// A `Skip` of `n` positions from a sequence of `len < n`. The length is that of an
    /// intermediate node, which may exceed `usize` on a 32-bit target.
    SkipOutOfRange {
        /// Positions to skip.
        n: usize,
        /// Length of the sequence.
        len: u64,
    },
    /// A `Take` of `n` positions from a sequence of `len < n`. The length is that of an
    /// intermediate node, which may exceed `usize` on a 32-bit target.
    TakeOutOfRange {
        /// Positions to take.
        n: usize,
        /// Length of the sequence.
        len: u64,
    },
    /// A stride with `step == 0`.
    ZeroStep,
    /// The order is longer than `usize::MAX`. An intermediate node may be, the order itself
    /// may not.
    OrderTooLong {
        /// Length of the order.
        len: u64,
    },
    /// A length does not fit in 64 bits.
    LengthOverflow,
    /// More than 2³² sources.
    TooManySources,
    /// A mix has 2³¹ parts or more.
    TooManyMixParts,
    /// The configuration nests deeper than [`MAX_DEPTH`](crate::MAX_DEPTH).
    TooDeep,
    /// The total length of a mix exceeds [`MAX_MIX_LEN`](crate::MAX_MIX_LEN).
    MixTooLong,
    /// A schedule parameter of mix part `part` is out of range or not finite.
    InvalidSampling {
        /// Index of the part in the mix.
        part: usize,
        /// Its schedule.
        sampling: Sampling,
    },
    /// Mix part `part` is too long for the steepness of its schedule (`length × final_rate`
    /// exceeds [`MAX_MIX_LEN`](crate::MAX_MIX_LEN)).
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
    /// A weighted mix with a positive total has no weight to distribute it over: no parts,
    /// or weights that sum to zero.
    ZeroWeights,
    /// A part of a weighted mix has a positive share but no elements.
    EmptyWeightedPart {
        /// Index of the part in the mix.
        part: usize,
    },
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SkipOutOfRange { n, len } => write!(f, "cannot skip {n} of {len} positions"),
            Self::TakeOutOfRange { n, len } => write!(f, "cannot take {n} of {len} positions"),
            Self::ZeroStep => write!(f, "stride step is zero"),
            Self::OrderTooLong { len } => write!(f, "order of {len} positions is longer than usize::MAX"),
            Self::LengthOverflow => write!(f, "a length does not fit in 64 bits"),
            Self::TooManySources => write!(f, "more than 2^32 sources"),
            Self::TooManyMixParts => write!(f, "mix with 2^31 parts or more"),
            Self::TooDeep => write!(f, "configuration nests deeper than {} levels", crate::MAX_DEPTH),
            Self::MixTooLong => write!(f, "mix longer than {MAX_TOTAL_LEN}"),
            Self::InvalidSampling { part, sampling } => write!(f, "mix part {part}: invalid {sampling:?}"),
            Self::TooSteep { part } => write!(f, "mix part {part}: too long for the steepness of its schedule"),
            Self::Overcommitted { demand } => write!(f, "scheduled mix parts need {:.1}% of the draw rate at the end", demand * 100.0),
            Self::InvalidWeight { part, weight } => write!(f, "weighted mix part {part}: invalid weight {weight}"),
            Self::ZeroWeights => write!(f, "weighted mix: no parts, or weights that sum to zero"),
            Self::EmptyWeightedPart { part } => write!(f, "weighted mix part {part} has a share but no elements"),
        }
    }
}

impl From<SamplingError> for ErrorKind {
    fn from(e: SamplingError) -> Self {
        match e {
            SamplingError::TooLong => Self::MixTooLong,
            SamplingError::InvalidParameter { seq, sampling } => Self::InvalidSampling { part: seq, sampling },
            SamplingError::TooSteep { seq } => Self::TooSteep { part: seq },
            SamplingError::Overcommitted { demand } => Self::Overcommitted { demand },
        }
    }
}
