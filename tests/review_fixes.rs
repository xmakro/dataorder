//! Public regressions for review findings and access APIs.
use dataorder::{BoundsError, ErrorKind, Item, Order, Schedule, Seq, Source};
use std::ops::Bound;

#[test]
fn standard_iterator_position_is_available() {
    let order = Order::new(Seq::source(10)).unwrap();
    let mut cursor = order.iter(..).unwrap();
    assert_eq!(cursor.position(|item| item.record_index == 3), Some(3));
    assert_eq!(cursor.offset(), 4);
    assert_eq!(cursor.position(|item| item.record_index == 6), Some(2));
    assert_eq!(cursor.offset(), 7);
}

#[test]
#[allow(clippy::reversed_empty_ranges)]
fn checked_access_preserves_cursor_on_errors() {
    let order = Order::new(Seq::mix([Seq::source(10).shuffle(1), Seq::source(7)])).unwrap();
    assert_eq!(order.get(16), (&order).into_iter().nth(16));
    assert_eq!(order.get(17), None);
    assert_eq!(order.get(usize::MAX), None);
    let empty = Order::new(Seq::source(0)).unwrap();
    assert_eq!(empty.get(0), None);
    assert_eq!((&empty).into_iter().next(), None);
    assert_eq!(order.iter(..=usize::MAX).unwrap_err(), BoundsError::EndOverflow);
    assert_eq!(order.iter((Bound::Excluded(usize::MAX), Bound::Unbounded)).unwrap_err(), BoundsError::StartOverflow);
    assert_eq!(order.iter(10..9).unwrap_err(), BoundsError::Reversed { start: 10, end: 9 });
    assert_eq!(order.iter(..18).unwrap_err(), BoundsError::OutOfBounds { end: 18, len: 17 });
    let mut c = order.iter(3..10).unwrap();
    c.next(); // Initialize the cursor before testing rollback.
    let expected = c.clone().collect::<Vec<_>>();
    assert_eq!(c.seek(11), Err(BoundsError::SeekOutOfBounds { pos: 11, end: 10 }));
    assert!(c.set_range(2..18).is_err());
    assert!(c.set_range(12..4).is_err());
    assert_eq!(c.offset(), 4);
    assert_eq!(c.clone().collect::<Vec<_>>(), expected);
    c.set_range(17..17).unwrap();
    assert_eq!(c.next(), None);
    c.set_range(..).unwrap();
    assert_eq!(c.next(), order.get(0));
    let huge = Order::new(Seq::source(usize::MAX)).unwrap();
    assert_eq!(huge.iter(usize::MAX - 1..).unwrap().next().unwrap().record_index, usize::MAX - 1);
    assert_eq!(huge.iter(usize::MAX..).unwrap().count(), 0);
}

#[test]
fn items_distinguish_zero_sized_sources() {
    #[derive(Debug, PartialEq)]
    struct Zero;
    impl Source for Zero {
        fn len(&self) -> usize {
            4
        }
    }
    let order = Order::new(Seq::mix([Seq::source(Zero), Seq::source(Zero)])).unwrap();
    let expected: Vec<_> = (0..4).flat_map(|i| [(0, i), (1, i)]).collect();
    let mut cursor = order.iter(..).unwrap();
    assert_eq!(cursor.clone().map(|item| (item.source_ordinal, item.record_index)).collect::<Vec<_>>(), expected);
    for (pos, &(s, i)) in expected.iter().enumerate() {
        assert_eq!(order.get(pos), Some(Item { source_ordinal: s, source: &Zero, record_index: i }));
    }
    assert_eq!(cursor.nth(3), Some(Item { source_ordinal: 1, source: &Zero, record_index: 1 }));
    assert_eq!(cursor.offset(), 4);
    assert!(cursor.seek(9).is_err());
    assert!(cursor.set_range(..9).is_err());
    assert_eq!(cursor.offset(), 4);
    cursor.seek(0).unwrap();
    assert_eq!(cursor.next(), Some(Item { source_ordinal: 0, source: &Zero, record_index: 0 }));
    assert_eq!(cursor.clone().last(), Some(Item { source_ordinal: 1, source: &Zero, record_index: 3 }));
    assert_eq!(cursor.clone().count(), 7);
    cursor.set_range(2..4).unwrap();
    assert_eq!(cursor.nth(usize::MAX), None);
    assert_eq!(cursor.offset(), 4);
    assert_eq!(cursor.next(), None);
    cursor.set_range(..).unwrap();
    cursor.seek(2).unwrap();
    assert_eq!(cursor.offset(), 2);
}

#[test]
fn items_copy_without_cloning_source_handles() {
    struct Handle(usize);
    impl Source for Handle {
        fn len(&self) -> usize {
            self.0
        }
    }
    let order = Order::new(Seq::concat([Seq::source(Handle(0)), Seq::source(Handle(4)), Seq::source(Handle(4))])).unwrap();
    for (pos, item) in (&order).into_iter().enumerate() {
        // Handle has neither Copy nor Clone; copying an item only copies its reference.
        for copy in [item, item, order.get(pos).unwrap()] {
            assert_eq!(copy.source_ordinal, 1 + pos / 4);
            assert_eq!(copy.record_index, pos % 4);
            assert!(std::ptr::eq(copy.source, &order.sources()[copy.source_ordinal]));
        }
    }
}

#[test]
fn sharding_preserves_global_partition_not_worker_mixture() {
    let seq = Seq::mix([Seq::source(4).shuffle(1), Seq::source(4).shuffle(2)]);
    for worker in 0..2 {
        let order = Order::new(seq.clone().skip(worker).step_by(2)).unwrap();
        assert!(order.iter(..).unwrap().all(|item| item.source_ordinal == worker));
    }
}

#[test]
fn equality_preserves_nan_payload_sign_and_signaling_bits() {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let hash = |x: &Schedule| {
        let mut h = DefaultHasher::new();
        x.hash(&mut h);
        h.finish()
    };
    let bits = [0x7ff0000000000001, 0x7ff8000000000001, 0x7ff8000000000002, 0xfff8000000000001];
    for (i, a) in bits.iter().enumerate() {
        for (j, b) in bits.iter().enumerate() {
            let (a, b) = (f64::from_bits(*a), f64::from_bits(*b));
            assert_eq!(Schedule::delayed(a) == Schedule::delayed(b), i == j);
        }
    }
    assert_eq!(Schedule::ramp(-0.0, 0.5), Schedule::ramp(0.0, 0.5));
    assert_eq!(hash(&Schedule::ramp(-0.0, 0.5)), hash(&Schedule::ramp(0.0, 0.5)));
}

#[test]
fn schedule_errors_describe_independent_profiles() {
    use dataorder::{MAX_MIX_LEN, ScheduleReason};
    for (schedule, reason) in [
        (Schedule::delayed(f64::INFINITY), ScheduleReason::NonFiniteParameter),
        (Schedule::ramp(0.5, 0.25), ScheduleReason::InvalidBreakpoints),
        (Schedule::ramp(0.0, f64::from_bits(1)), ScheduleReason::CoefficientOverflow),
    ] {
        let err = Order::new(Seq::mix([(Seq::source(1), schedule)])).unwrap_err();
        let expected = ErrorKind::InvalidSchedule { schedule, reason };
        assert_eq!(err.kind(), &expected);
        assert_eq!(err.path(), [0]);
        assert_eq!(err.to_string(), format!("invalid schedule {schedule:?}: {reason} (at node 0)"));
        assert_eq!(err.into_kind(), expected);
    }
    let seq = Seq::mix([(Seq::source(1usize << 30), Schedule::until(1e-6))]);
    let err = Order::new(seq).unwrap_err();
    let expected = ErrorKind::TooSteep { len: 1 << 30, peak_rate: 1e6, limit: MAX_MIX_LEN };
    assert_eq!(err.kind(), &expected);
    assert_eq!(err.path(), [0]);
    assert_eq!(
        err.to_string(),
        "mix part too long for the steepness of its schedule: length 1073741824 × peak rate 1000000 exceeds 70368744177664 (at node 0)"
    );
    assert_eq!(err.into_kind(), expected);
    let mixed = Seq::mix([(Seq::source(10).cycle(3 << 28), Schedule::until(1e-6)), (Seq::source(10).cycle(1 << 28), Schedule::Uniform)]);
    let err = Order::new(mixed).unwrap_err();
    let expected = ErrorKind::TooSteep { len: 3 << 28, peak_rate: 1e6, limit: MAX_MIX_LEN };
    assert_eq!(err.kind(), &expected);
    assert_eq!(err.path(), [0]);
    assert_eq!(err.into_kind(), expected);
}
