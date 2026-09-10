//! Fallible position, range and shard validation used by orders and cursors.

use std::fmt;
use std::ops::{Bound, Range, RangeBounds};

/// An invalid range, cursor position or shard. Failed cursor operations leave the
/// cursor unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BoundsError {
    /// A worker index is outside `0..count`, including a zero worker count.
    InvalidShard {
        /// Number of workers.
        count: usize,
        /// Requested worker index.
        index: usize,
    },
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
    /// A seek extends beyond the cursor's current range end.
    SeekOutOfBounds {
        /// Requested absolute position.
        pos: usize,
        /// Cursor's exclusive range end.
        end: usize,
    },
}

impl fmt::Display for BoundsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShard { count, index } => write!(f, "shard index {index} out of range for {count} shards"),
            Self::StartOverflow => write!(f, "range start overflows usize"),
            Self::EndOverflow => write!(f, "range end overflows usize"),
            Self::Reversed { start, end } => write!(f, "range {start}..{end} ends before it starts"),
            Self::OutOfBounds { end, len } => write!(f, "range end {end} out of range for {len} positions"),
            Self::SeekOutOfBounds { pos, end } => write!(f, "seek to {pos} beyond the end {end}"),
        }
    }
}

impl std::error::Error for BoundsError {}

/// Normalize bounds without requiring a sequence length.
pub(crate) fn boundaries(range: impl RangeBounds<usize>) -> Result<(usize, Option<usize>), BoundsError> {
    let start = match range.start_bound() {
        Bound::Included(&s) => s,
        Bound::Excluded(&s) => s.checked_add(1).ok_or(BoundsError::StartOverflow)?,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&e) => Some(e.checked_add(1).ok_or(BoundsError::EndOverflow)?),
        Bound::Excluded(&e) => Some(e),
        Bound::Unbounded => None,
    };
    if let Some(end) = end.filter(|&end| start > end) {
        return Err(BoundsError::Reversed { start, end });
    }
    Ok((start, end))
}

/// Normalize and validate against the order length.
pub(crate) fn resolve(range: impl RangeBounds<usize>, len: usize) -> Result<Range<usize>, BoundsError> {
    let (start, end) = boundaries(range)?;
    let end = end.unwrap_or(len);
    if start > end {
        return Err(BoundsError::Reversed { start, end });
    }
    if end > len {
        return Err(BoundsError::OutOfBounds { end, len });
    }
    Ok(start..end)
}
