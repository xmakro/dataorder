//! Helpers shared by the integration tests. Each test binary compiles its own copy and
//! uses a subset of it.
#![allow(dead_code)]

use dataorder::{Item, Order, Seq, Source};
use std::fmt::Debug;

/// A test source: an id to compare orders by, which also salts its shuffles, and a length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Src {
    pub id: u64,
    pub len: usize,
}

impl Source for Src {
    fn len(&self) -> usize {
        self.len
    }

    fn salt(&self) -> u64 {
        self.id
    }
}

pub fn src(id: u64, len: usize) -> Seq<Src> {
    Seq::source(Src { id, len })
}

/// Elements as `(id, index)`.
pub fn ids<'a>(it: impl Iterator<Item = Item<'a, Src>>) -> Vec<(u64, usize)> {
    it.map(|item| (item.source.id, item.record_index)).collect()
}

/// Random access, seeks, clones and `nth` agree with a full walk: `get` at every
/// position, a short window from every start, and the whole remainder from a few.
pub fn check_access<T: PartialEq + Debug>(order: &Order<T>) {
    let expected: Vec<_> = order.iter().collect();
    assert_eq!(expected.len(), order.len());
    let mut cursor = order.cursor(0..0).unwrap();
    for start in (0..=order.len()).rev() {
        assert_eq!(order.get(start), expected.get(start).copied());
        let end = (start + 11).min(order.len());
        cursor.reset(start..end).unwrap();
        assert_eq!(cursor.by_ref().collect::<Vec<_>>(), expected[start..end], "window from {start}");
    }
    for pos in [order.len(), 0, order.len() / 2, 1, order.len().saturating_sub(1)] {
        let pos = pos.min(order.len());
        cursor.reset(pos..).unwrap();
        assert_eq!(cursor.clone().collect::<Vec<_>>(), expected[pos..], "remainder from {pos}");
        assert_eq!(cursor.nth(3), expected.get(pos + 3).copied(), "nth from {pos}");
    }
}
