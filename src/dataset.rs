//! The trait a source must implement: a length. The order never reads an element; it
//! yields the source and an index, and the caller reads however it likes.

use std::rc::Rc;
use std::sync::Arc;

/// A source of elements, identified by nothing but its length. Implement it for your
/// dataset handle and put the handle into [`Seq::Source`](crate::Seq::Source); the order
/// yields `(&handle, index)`. A bare `usize` is a source too, handy when only the order
/// matters.
pub trait Dataset {
    /// Number of elements.
    fn len(&self) -> usize;

    /// `true` when there are no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Dataset for usize {
    fn len(&self) -> usize {
        *self
    }
}

impl<T: Dataset + ?Sized> Dataset for &T {
    fn len(&self) -> usize {
        (**self).len()
    }
}

impl<T: Dataset + ?Sized> Dataset for Box<T> {
    fn len(&self) -> usize {
        (**self).len()
    }
}

impl<T: Dataset + ?Sized> Dataset for Rc<T> {
    fn len(&self) -> usize {
        (**self).len()
    }
}

impl<T: Dataset + ?Sized> Dataset for Arc<T> {
    fn len(&self) -> usize {
        (**self).len()
    }
}
