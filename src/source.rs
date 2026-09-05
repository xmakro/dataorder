//! The trait a source must implement: a length. The order never reads an element; it
//! yields the source and an index, and the caller reads however it likes.

use std::rc::Rc;
use std::sync::Arc;

/// A source of elements: a length, and a [`salt`](Source::salt) that tells it apart from
/// other sources of the same length when it is shuffled. Implement it for your dataset
/// handle and put the handle into [`Seq::Source`](crate::Seq::Source); the order yields
/// `(&handle, index)`. A bare `usize` is a source too (salt 0), handy when only the order
/// matters, as are slices, arrays and vectors of anything.
///
/// The order reads the length and the salt once, when it is built; a source whose length
/// changes afterwards yields indices past its new end.
///
/// ```
/// use dataorder::Source;
///
/// struct Shard { path: String, records: usize }
///
/// impl Source for Shard {
///     fn len(&self) -> usize { self.records }
///     fn salt(&self) -> u64 { dataorder::salt(&self.path) }
/// }
/// ```
pub trait Source {
    /// Number of elements.
    fn len(&self) -> usize;

    /// `true` when there are no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Distinguishes this source from others of its length. A shuffle's permutation depends
    /// on its seed, on the order's seed and repetition, and on the salts and lengths of the
    /// sources under it: two sources of one length and salt shuffled with one seed get the
    /// same permutation, different salts give unrelated ones. Derive it from the dataset's
    /// identity, its path say, with [`salt`](crate::salt). The default is 0.
    fn salt(&self) -> u64 {
        0
    }
}

/// A salt for [`Source::salt`] from any bytes, a path for instance: FNV-1a, which is part
/// of the orders and therefore never changes.
///
/// ```
/// assert_eq!(dataorder::salt("web.bin"), dataorder::salt(b"web.bin"));
/// assert_ne!(dataorder::salt("web.bin"), dataorder::salt("code.bin"));
/// ```
#[must_use]
pub fn salt(bytes: impl AsRef<[u8]>) -> u64 {
    bytes.as_ref().iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
}

impl Source for usize {
    fn len(&self) -> usize {
        *self
    }
}

impl<T> Source for [T] {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
}

impl<T, const N: usize> Source for [T; N] {
    fn len(&self) -> usize {
        N
    }
}

impl<T> Source for Vec<T> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
}

macro_rules! forward {
    ($($t:ty),*) => {$(
        impl<T: Source + ?Sized> Source for $t {
            fn len(&self) -> usize {
                (**self).len()
            }

            fn salt(&self) -> u64 {
                (**self).salt()
            }
        }
    )*};
}

forward!(&T, &mut T, Box<T>, Rc<T>, Arc<T>);
