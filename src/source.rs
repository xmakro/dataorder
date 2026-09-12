//! Dataset lengths and stable identities. An order returns a source and an index;
//! the caller decides how to load the corresponding record.

use std::rc::Rc;
use std::sync::Arc;

/// A dataset described by its length and an optional shuffle salt.
///
/// Implement this trait for your dataset handle when compiling a sequence with
/// [`Order::new`](crate::Order::new). [`Seq`](crate::Seq) itself accepts any type.
/// The order yields an [`Item`](crate::Item) containing a source ordinal, a reference
/// to the handle and a record index, without reading any records.
/// [`salt`](Source::salt) lets datasets of the same length have distinct shuffle inputs.
///
/// ```
/// use dataorder::Source;
///
/// struct Shard { name: String, records: usize }
///
/// impl Source for Shard {
///     fn len(&self) -> usize { self.records }
///     fn salt(&self) -> u64 { dataorder::salt(&self.name) }
/// }
/// ```
///
/// Slices, arrays and vectors implement this trait with salt 0. A `usize` also
/// represents a source of that length, which is useful when only the order matters.
/// References, `Box`, `Rc` and `Arc` forward to the underlying source.
///
/// The order reads each source's length and salt at construction and does not
/// refresh them. Keep lengths stable: shrinking a source can make returned indices
/// invalid, and growing it does not add positions to the order.
pub trait Source {
    /// Number of elements.
    fn len(&self) -> usize;

    /// `true` when there are no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A stable dataset identity used when deriving shuffle keys. Defaults to 0.
    ///
    /// Sources with the same length and salt shuffle alike under the same order seed
    /// and, for shuffled repetitions, the local pass number. Derive a salt from a dataset name with [`crate::salt`]
    /// to keep ordering independent of storage location.
    /// The compiler includes this salt and the original length in configuration
    /// salts, even when the source is empty or a selection discards its records.
    fn salt(&self) -> u64 {
        0
    }
}

/// Computes a stable [`Source::salt`] from bytes, such as a dataset name.
/// Uses FNV-1a; changing the hash would change orders and is covered by the crate's
/// stability policy.
///
/// ```
/// assert_eq!(dataorder::salt("web"), dataorder::salt(b"web"));
/// assert_ne!(dataorder::salt("web"), dataorder::salt("code"));
/// ```
#[must_use]
pub fn salt(bytes: impl AsRef<[u8]>) -> u64 {
    bytes.as_ref().iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
}

// Keep usize as the only integer implementation so Seq::source(10) infers usize.
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
        <[T]>::len(self)
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

#[cfg(test)]
mod tests {
    use super::salt;

    #[test]
    fn stable_salt_vectors() {
        // Fixed FNV-1a answers, including bytes that string-only tests would miss.
        for (bytes, expected) in [
            (b"".as_slice(), 0xcbf2_9ce4_8422_2325),
            (b"a".as_slice(), 0xaf63_dc4c_8601_ec8c),
            (b"foobar".as_slice(), 0x8594_4171_f739_67e8),
            (b"a\0b".as_slice(), 0xe5d2_9919_0426_66b2),
            (b"web.bin".as_slice(), 0x3e74_c77b_571a_dd96),
            ("web/训练.bin".as_bytes(), 0xc2ed_ac2d_2053_c259),
        ] {
            assert_eq!(salt(bytes), expected, "bytes: {bytes:?}");
        }
    }
}
