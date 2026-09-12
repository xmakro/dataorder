//! Configuration errors with sequence-tree locations, and order/cursor bounds errors.

use crate::Schedule;
use crate::interleave::MAX_TOTAL_LEN;
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
}

impl Error {
    pub(crate) fn new(kind: ErrorKind, path: Vec<usize>) -> Self {
        Self { kind, path }
    }

    /// What went wrong, including schedule diagnostics.
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

    /// Consumes the error and returns its kind, including schedule diagnostics,
    /// discarding only the path.
    #[must_use]
    pub fn into_kind(self) -> ErrorKind {
        self.kind
    }

    /// A schedule or length rejection of a mix.
    #[cfg(test)]
    pub(crate) fn is_schedule(&self) -> bool {
        matches!(self.kind(), ErrorKind::MixTooLong | ErrorKind::InvalidSchedule { .. } | ErrorKind::ScheduleTooSteep { .. })
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

/// Why the schedule in [`ErrorKind::InvalidSchedule`] is invalid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ScheduleReason {
    /// At least one breakpoint is NaN or infinite.
    NonFiniteParameter,
    /// Finite breakpoints violate their ordering, range or positive-area constraints.
    InvalidBreakpoints,
    /// Valid breakpoints produced non-finite profile coefficients.
    CoefficientOverflow,
}

impl fmt::Display for ScheduleReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteParameter => write!(f, "a breakpoint is not finite"),
            Self::InvalidBreakpoints => write!(f, "breakpoints are out of range, out of order, or have zero area"),
            Self::CoefficientOverflow => write!(f, "derived profile coefficients exceed floating-point range"),
        }
    }
}

/// The reason a configuration failed validation.
///
/// ```
/// use dataorder::{ErrorKind, Order, Schedule, ScheduleReason, Seq};
/// let err = Order::new(Seq::source(10).step_by(0)).unwrap_err();
/// assert!(matches!(err.kind(), ErrorKind::ZeroStep));
/// let err = Order::new(Seq::mix([(Seq::source(10), Schedule::delayed(1.5))])).unwrap_err();
/// assert!(matches!(err.into_kind(), ErrorKind::InvalidSchedule {
///     reason: ScheduleReason::InvalidBreakpoints, ..
/// }));
/// ```
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ErrorKind {
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
    /// A `StepBy` with `step == 0`.
    ZeroStep,
    /// A shuffle contains a mix anywhere in its input configuration, even if the mix
    /// would fold away. The error path identifies the nearest enclosing shuffle.
    ShuffleContainsMix,
    /// A sequence length exceeds `usize::MAX`, including at an intermediate node.
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
    InvalidSchedule {
        /// The schedule.
        schedule: Schedule,
        /// Why the schedule is invalid.
        reason: ScheduleReason,
    },
    /// A mix part is too long for the steepness of its schedule (`length × its highest
    /// rate` exceeds [`MAX_MIX_LEN`](crate::MAX_MIX_LEN)).
    ScheduleTooSteep {
        /// Compiled part length.
        len: usize,
        /// Highest normalized rate in the part's profile.
        peak_rate: f64,
        /// Maximum supported length times rate.
        limit: u64,
    },
    /// A `Cycle` of positive length over a sequence without elements.
    EmptyCycle,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SkipOutOfRange { n, len } => write!(f, "cannot skip {n} of {len} positions"),
            Self::TakeOutOfRange { n, len } => write!(f, "cannot take {n} of {len} positions"),
            Self::ZeroStep => write!(f, "step is zero"),
            Self::ShuffleContainsMix => write!(f, "cannot shuffle a sequence containing a mix; shuffle its inputs before mixing"),
            Self::LengthOverflow => write!(f, "sequence length exceeds usize::MAX"),
            Self::TooManySources => write!(f, "more than 2^32 sources"),
            Self::TooManyMixParts => write!(f, "mix with 2^31 - 1 parts or more"),
            Self::TooDeep => write!(f, "configuration nests deeper than {} levels", crate::MAX_DEPTH),
            Self::MixTooLong => write!(f, "mix longer than {MAX_TOTAL_LEN}"),
            Self::InvalidSchedule { schedule, reason } => write!(f, "invalid schedule {schedule:?}: {reason}"),
            Self::ScheduleTooSteep { len, peak_rate, limit } => {
                write!(f, "mix part too long for the steepness of its schedule: length {len} × peak rate {peak_rate} exceeds {limit}")
            }
            Self::EmptyCycle => write!(f, "cannot cycle a sequence without elements"),
        }
    }
}

/// An invalid order range. Failed cursor operations leave the
/// cursor unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BoundsError {
    /// An exclusive start at `usize::MAX` cannot be advanced by one.
    StartOverflow,
    /// An inclusive end at `usize::MAX` cannot be advanced by one.
    EndOverflow,
    /// The exclusive end precedes the start.
    Reversed {
        /// Inclusive start.
        start: usize,
        /// Exclusive end.
        end: usize,
    },
    /// A range extends beyond the order.
    OutOfBounds {
        /// Requested exclusive end.
        end: usize,
        /// Order length.
        len: usize,
    },
}

impl fmt::Display for BoundsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StartOverflow => write!(f, "range start overflows usize"),
            Self::EndOverflow => write!(f, "range end overflows usize"),
            Self::Reversed { start, end } => write!(f, "range {start}..{end} ends before it starts"),
            Self::OutOfBounds { end, len } => write!(f, "range end {end} out of range for {len} positions"),
        }
    }
}

impl std::error::Error for BoundsError {}
