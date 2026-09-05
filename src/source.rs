//! The trait a source must implement: a length. The order never reads an element; it
//! yields the source and an index, and the caller reads however it likes.

use std::rc::Rc;
use std::sync::Arc;

/// A source of elements, identified by nothing but its length. Implement it for your
/// dataset handle and put the handle into [`Seq::Source`](crate::Seq::Source); the order
/// yields `(&handle, index)`. A bare `usize` is a source too, handy when only the order
/// matters.
///
/// The order reads the length once, when it is built; a source whose length changes
/// afterwards yields indices past its new end.
pub trait Source {
    /// Number of elements.
    fn len(&self) -> usize;

    /// `true` when there are no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Source for usize {
    fn len(&self) -> usize {
        *self
    }
}

impl<T: Source + ?Sized> Source for &T {
    fn len(&self) -> usize {
        (**self).len()
    }
}

impl<T: Source + ?Sized> Source for &mut T {
    fn len(&self) -> usize {
        (**self).len()
    }
}

impl<T: Source + ?Sized> Source for Box<T> {
    fn len(&self) -> usize {
        (**self).len()
    }
}

impl<T: Source + ?Sized> Source for Rc<T> {
    fn len(&self) -> usize {
        (**self).len()
    }
}

impl<T: Source + ?Sized> Source for Arc<T> {
    fn len(&self) -> usize {
        (**self).len()
    }
}
