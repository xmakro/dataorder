//! The errors of [`Order::new`](crate::Order::new).

use crate::Sampling;
use crate::interleave::{MAX_TOTAL_LEN, SamplingError};
use std::fmt;

/// Why and where [`Order::new`](crate::Order::new) rejected a configuration: the
/// [`kind`](Error::kind) of the problem and the [`path`](Error::path) of the node it was
/// found at.
///
/// ```
/// use dataorder::{ErrorKind, Seq};
/// let err = Seq::concat([Seq::source(10), Seq::source(5).take(6)]).check().unwrap_err();
/// assert_eq!(err.kind(), &ErrorKind::TakeOutOfRange { n: 6, len: 5 });
/// assert_eq!(err.path(), [1]);
/// assert_eq!(err.to_string(), "cannot take 6 of 5 positions (at node 1)");
/// ```
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
    /// A schedule or weight problem is found at the part it belongs to.
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
            write!(f, " (at node ")?;
            for (k, i) in self.path.iter().enumerate() {
                if k > 0 {
                    write!(f, "/")?;
                }
                write!(f, "{i}")?;
            }
            write!(f, ")")
        }
    }
}

impl std::error::Error for Error {}

/// What [`Order::new`](crate::Order::new) found wrong with a configuration.
///
/// ```
/// use dataorder::{ErrorKind, Sampling, Seq};
/// let err = Seq::source(10).stride(0, 0).check().unwrap_err();
/// assert!(matches!(err.kind(), ErrorKind::ZeroStep));
/// let err = Seq::mix_with([(Seq::source(10), Sampling::delayed(1.5))]).check().unwrap_err();
/// assert!(matches!(err.kind(), ErrorKind::InvalidSampling { .. }));
/// ```
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
    /// A mix has 2³¹ − 1 parts or more.
    TooManyMixParts,
    /// The configuration nests deeper than [`MAX_DEPTH`](crate::MAX_DEPTH).
    TooDeep,
    /// The total length of a mix exceeds [`MAX_MIX_LEN`](crate::MAX_MIX_LEN).
    MixTooLong,
    /// A schedule parameter is out of range or non-finite, or its derived profile
    /// coefficients overflow floating-point arithmetic.
    InvalidSampling {
        /// The schedule.
        sampling: Sampling,
    },
    /// The combined uniform profile's coefficients exceed floating-point range, even
    /// though the individual schedules can be represented.
    SamplingOverflow,
    /// A mix part is too long for the steepness of its schedule (`length × its highest
    /// rate` exceeds [`MAX_MIX_LEN`](crate::MAX_MIX_LEN)).
    TooSteep,
    /// The scheduled parts of a mix need `demand` (> 1) times the whole draw rate at some
    /// progress, leaving nothing for the uniform parts there. A demand within 10⁻⁹ of 1 is
    /// accepted, for rounding: a hand-over that needs exactly the whole rate is valid.
    Overcommitted {
        /// The scheduled parts' rates, summed, at their peak, as a fraction of the whole
        /// draw rate.
        demand: f64,
    },
    /// The weight of a weighted mix part is negative or not finite.
    InvalidWeight {
        /// The weight.
        weight: f64,
    },
    /// A weighted mix with a positive total has no weight to distribute it over: no parts,
    /// or weights that sum to zero.
    ZeroWeights,
    /// A part of a weighted mix has a positive share but no elements.
    EmptyWeightedPart,
    /// A `Cycle` of positive length over a sequence without elements.
    EmptyCycle,
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
            Self::TooManyMixParts => write!(f, "mix with 2^31 - 1 parts or more"),
            Self::TooDeep => write!(f, "configuration nests deeper than {} levels", crate::MAX_DEPTH),
            Self::MixTooLong => write!(f, "mix longer than {MAX_TOTAL_LEN}"),
            Self::InvalidSampling { sampling } => write!(f, "invalid schedule {sampling:?}"),
            Self::SamplingOverflow => write!(f, "combined sampling profile exceeds floating-point range"),
            Self::TooSteep => write!(f, "mix part too long for the steepness of its schedule"),
            Self::Overcommitted { demand } => write!(f, "scheduled mix parts need {:.1}% of the draw rate at their peak", demand * 100.0),
            Self::InvalidWeight { weight } => write!(f, "invalid weight {weight}"),
            Self::ZeroWeights => write!(f, "weighted mix: no parts, or weights that sum to zero"),
            Self::EmptyWeightedPart => write!(f, "weighted mix part has a share but no elements"),
            Self::EmptyCycle => write!(f, "cannot cycle a sequence without elements"),
        }
    }
}

impl SamplingError {
    /// The kind and, for a problem with one part, the part's index.
    pub(crate) fn into_kind(self) -> (ErrorKind, Option<usize>) {
        match self {
            Self::TooLong => (ErrorKind::MixTooLong, None),
            Self::InvalidParameter { seq, sampling } => (ErrorKind::InvalidSampling { sampling }, Some(seq)),
            Self::Overflow => (ErrorKind::SamplingOverflow, None),
            Self::TooSteep { seq } => (ErrorKind::TooSteep, Some(seq)),
            Self::Overcommitted { demand } => (ErrorKind::Overcommitted { demand }, None),
        }
    }
}
