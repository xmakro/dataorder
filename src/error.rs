//! Configuration errors, including their location in the sequence tree.

use crate::Sampling;
use crate::interleave::{MAX_TOTAL_LEN, SamplingError};
use std::fmt;

/// A configuration error from [`Order::new`](crate::Order::new).
/// [`kind`](Error::kind) describes the problem; [`path`](Error::path) identifies
/// the node where it was found.
///
/// ```
/// use dataorder::{ErrorKind, Order, Seq};
/// let err = Order::new(Seq::concat([Seq::source(10), Seq::source(5).take(6)])).unwrap_err();
/// assert_eq!(err.kind(), &ErrorKind::TakeOutOfRange { n: 6, len: 5 });
/// assert_eq!(err.path(), [1]);
/// assert_eq!(err.to_string(), "cannot take 6 of 5 positions (at node 1)");
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Error {
    kind: ErrorKind,
    path: Vec<usize>,
    sampling_detail: Option<SamplingDetail>,
}

impl Error {
    pub(crate) fn new(kind: ErrorKind, path: Vec<usize>) -> Self {
        Self { kind, path, sampling_detail: None }
    }

    pub(crate) fn with_sampling_detail(mut self, detail: Option<SamplingDetail>) -> Self {
        self.sampling_detail = detail;
        self
    }

    /// Additional numerical context for an invalid schedule or excessive steepness.
    /// The stable classification remains available through [`Error::kind`].
    #[must_use]
    pub fn sampling_detail(&self) -> Option<&SamplingDetail> {
        self.sampling_detail.as_ref()
    }

    /// What went wrong.
    #[must_use]
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }

    /// Child indices leading from the root to the invalid node.
    ///
    /// Each index selects a part of a `Concat` or `Mix` node, or is 0
    /// for a node with one child. An empty path means the root. Invalid schedule
    /// parameters point to their part; errors for the mix as a whole,
    /// such as excessive total length, point to the mix.
    #[must_use]
    pub fn path(&self) -> &[usize] {
        &self.path
    }

    /// Consumes the error and returns its kind, discarding the path and sampling details.
    #[must_use]
    pub fn into_kind(self) -> ErrorKind {
        self.kind
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.kind.fmt(f)?;
        if let Some(detail) = &self.sampling_detail {
            write!(f, ": {detail}")?;
        }
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

/// Additional context for [`ErrorKind::InvalidSampling`] and [`ErrorKind::TooSteep`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum SamplingDetail {
    /// At least one breakpoint is NaN or infinite.
    NonFiniteParameter,
    /// Finite breakpoints violate their ordering, range or positive-area constraints.
    InvalidBreakpoints,
    /// Valid breakpoints produced non-finite profile coefficients.
    CoefficientOverflow,
    /// The part's assigned length multiplied by its peak rate exceeds the limit.
    TooSteep {
        /// Compiled part length.
        len: u64,
        /// Highest normalized rate in the part's profile.
        peak_rate: f64,
        /// Maximum supported length times rate.
        limit: u64,
    },
}

impl fmt::Display for SamplingDetail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteParameter => write!(f, "a breakpoint is not finite"),
            Self::InvalidBreakpoints => write!(f, "breakpoints are out of range, out of order, or have zero area"),
            Self::CoefficientOverflow => write!(f, "derived profile coefficients exceed floating-point range"),
            Self::TooSteep { len, peak_rate, limit } => write!(f, "length {len} × peak rate {peak_rate} exceeds {limit}"),
        }
    }
}

/// The reason a configuration failed validation.
///
/// ```
/// use dataorder::{ErrorKind, Order, Sampling, Seq};
/// let err = Order::new(Seq::source(10).step_by(0)).unwrap_err();
/// assert!(matches!(err.kind(), ErrorKind::ZeroStep));
/// let err = Order::new(Seq::mix_with([(Seq::source(10), Sampling::delayed(1.5))])).unwrap_err();
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
    /// A `StepBy` with `step == 0`.
    ZeroStep,
    /// The final order is longer than `usize::MAX`. Only intermediate nodes may
    /// exceed that limit.
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
    /// A mix part is too long for the steepness of its schedule (`length × its highest
    /// rate` exceeds [`MAX_MIX_LEN`](crate::MAX_MIX_LEN)).
    TooSteep,
    /// A `Cycle` of positive length over a sequence without elements.
    EmptyCycle,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SkipOutOfRange { n, len } => write!(f, "cannot skip {n} of {len} positions"),
            Self::TakeOutOfRange { n, len } => write!(f, "cannot take {n} of {len} positions"),
            Self::ZeroStep => write!(f, "step is zero"),
            Self::OrderTooLong { len } => write!(f, "order of {len} positions is longer than usize::MAX"),
            Self::LengthOverflow => write!(f, "a length does not fit in 64 bits"),
            Self::TooManySources => write!(f, "more than 2^32 sources"),
            Self::TooManyMixParts => write!(f, "mix with 2^31 - 1 parts or more"),
            Self::TooDeep => write!(f, "configuration nests deeper than {} levels", crate::MAX_DEPTH),
            Self::MixTooLong => write!(f, "mix longer than {MAX_TOTAL_LEN}"),
            Self::InvalidSampling { sampling } => write!(f, "invalid schedule {sampling:?}"),
            Self::TooSteep => write!(f, "mix part too long for the steepness of its schedule"),
            Self::EmptyCycle => write!(f, "cannot cycle a sequence without elements"),
        }
    }
}

impl SamplingError {
    pub(crate) fn detail(&self) -> Option<SamplingDetail> {
        match self {
            Self::InvalidParameter { detail, .. } => Some(*detail),
            Self::TooSteep { len, peak_rate, .. } => {
                Some(SamplingDetail::TooSteep { len: *len, peak_rate: *peak_rate, limit: MAX_TOTAL_LEN })
            }
            _ => None,
        }
    }

    /// The kind and, for a problem with one part, the part's index.
    pub(crate) fn into_kind(self) -> (ErrorKind, Option<usize>) {
        match self {
            Self::TooLong => (ErrorKind::MixTooLong, None),
            Self::InvalidParameter { seq, sampling, .. } => (ErrorKind::InvalidSampling { sampling }, Some(seq)),
            Self::TooSteep { seq, .. } => (ErrorKind::TooSteep, Some(seq)),
        }
    }
}
