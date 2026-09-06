//! Public regressions for review findings and the added inspection/access APIs.
use dataorder::{BoundsError, ErrorKind, Order, PreparedKind, Sampling, Seq, Source, WeightedPart};
use std::ops::Bound;

#[test]
fn standard_iterator_position_is_available() {
    let order = Order::new(Seq::source(10)).unwrap();
    let mut cursor = order.iter(..);
    assert_eq!(cursor.position(|(_, i)| i == 3), Some(3));
    assert_eq!(cursor.offset(), 4);
    let mut indexed = cursor.indexed();
    assert_eq!(indexed.position(|(_, _, i)| i == 6), Some(2));
    assert_eq!(indexed.offset(), 7);
}

#[test]
#[allow(clippy::reversed_empty_ranges)]
fn checked_access_preserves_cursor_on_errors() {
    for (count, index) in [(0, 0), (2, 2), (1, usize::MAX)] {
        let error = Seq::source(10).try_shard(count, index).unwrap_err();
        assert_eq!(error, BoundsError::InvalidShard { count, index });
        assert_eq!(error.to_string(), format!("shard index {index} out of range for {count} shards"));
    }
    assert_eq!(Seq::source(10).try_shard(3, 1).unwrap(), Seq::source(10).shard(3, 1));
    let order = Order::new(Seq::mix([Seq::source(10).shuffle(1), Seq::source(7)])).unwrap();
    assert_eq!(order.try_get(16), Some(order.get(16)));
    assert_eq!(order.try_get(17), None);
    assert_eq!(order.try_get_indexed(usize::MAX), None);
    assert_eq!(order.try_iter(..=usize::MAX).unwrap_err(), BoundsError::EndOverflow);
    assert_eq!(order.try_iter((Bound::Excluded(usize::MAX), Bound::Unbounded)).unwrap_err(), BoundsError::StartOverflow);
    assert_eq!(order.try_iter(10..9).unwrap_err(), BoundsError::Reversed { start: 10, end: 9 });
    assert_eq!(order.try_iter(..18).unwrap_err(), BoundsError::OutOfBounds { end: 18, len: 17 });
    assert_eq!(Seq::source(10).try_slice(..=usize::MAX).unwrap_err(), BoundsError::EndOverflow);
    assert_eq!(Seq::source(10).try_slice(3..=5).unwrap().check(), Ok(3));
    assert!(Seq::source(10).try_slice(..11).unwrap().check().is_err());
    let mut c = order.iter(3..10);
    c.next(); // Initialize the cursor before testing rollback.
    let expected = c.clone().collect::<Vec<_>>();
    assert_eq!(c.try_seek(11), Err(BoundsError::SeekOutOfBounds { pos: 11, end: 10 }));
    assert!(c.try_set_range(2..18).is_err());
    assert!(c.try_set_range(12..4).is_err());
    assert_eq!(c.offset(), 4);
    assert_eq!(c.clone().collect::<Vec<_>>(), expected);
    c.try_set_range(17..17).unwrap();
    assert_eq!(c.next(), None);
    c.try_set_range(..).unwrap();
    assert_eq!(c.next(), Some(order.get(0)));
    let huge = Order::new(Seq::source(usize::MAX)).unwrap();
    assert_eq!(huge.try_iter(usize::MAX - 1..).unwrap().next().unwrap().1, usize::MAX - 1);
    assert_eq!(huge.try_iter(usize::MAX..).unwrap().count(), 0);
}

#[test]
fn indexed_results_distinguish_zero_sized_sources() {
    #[derive(Debug, PartialEq)]
    struct Zero;
    impl Source for Zero {
        fn len(&self) -> usize {
            4
        }
    }
    let order = Order::new(Seq::mix([Seq::source(Zero), Seq::source(Zero)])).unwrap();
    let expected: Vec<_> = (0..4).flat_map(|i| [(0, i), (1, i)]).collect();
    let mut cursor = order.iter(..).indexed();
    assert_eq!(cursor.clone().map(|(s, _, i)| (s, i)).collect::<Vec<_>>(), expected);
    for (pos, &(s, i)) in expected.iter().enumerate() {
        assert_eq!(order.get_indexed(pos), (s, &Zero, i));
        assert_eq!(order.try_get_indexed(pos), Some((s, &Zero, i)));
    }
    assert_eq!(cursor.nth(3), Some((1, &Zero, 1)));
    assert_eq!(cursor.offset(), 4);
    assert!(cursor.try_seek(9).is_err());
    assert!(cursor.try_set_range(..9).is_err());
    assert_eq!(cursor.offset(), 4);
    cursor.seek(0);
    assert_eq!(cursor.next(), Some((0, &Zero, 0)));
    assert_eq!(cursor.clone().last(), Some((1, &Zero, 3)));
    assert_eq!(cursor.clone().count(), 7);
    cursor.set_range(2..4);
    assert_eq!(cursor.nth(usize::MAX), None);
    assert_eq!(cursor.offset(), 4);
    assert_eq!(cursor.next(), None);
    cursor.try_set_range(..).unwrap();
    cursor.try_seek(2).unwrap();
    assert_eq!(cursor.into_cursor().offset(), 2);
}

#[test]
fn preparation_reports_original_quotas_and_compiled_lengths() {
    let seq = Seq::concat([
        Seq::weighted(4, [(Seq::source(10), 5.0), (Seq::source(10), 3.0), (Seq::source(10), 1.0)]),
        Seq::weighted(100, [(Seq::source(3), 1.0)]).take(0),
    ]);
    let ordinary = Order::with_seed(seq.clone(), 7).unwrap();
    let (order, report) = Order::prepare(seq, 7).unwrap();
    assert!(order.iter(..).eq(ordinary.iter(..)));
    assert_eq!(report.weighted[0].path, [0]);
    assert_eq!(report.weighted[0].counts, [2, 1, 1]);
    assert_eq!(report.weighted[1].path, [1, 0]);
    assert_eq!(report.weighted[1].counts, [100]);
    assert_eq!(report.nodes.len(), 4); // Concat and discarded tail have folded away.
    assert!(report.nodes[0].path.is_empty());
    assert_eq!(report.nodes[0].kind, PreparedKind::Mix);
    assert_eq!(report.nodes[0].len, 4);
    for (i, node) in report.nodes[1..].iter().enumerate() {
        assert_eq!(node.path, [i]);
        assert_eq!(node.source_ordinal, Some(i));
        assert_eq!(node.len, [2, 1, 1][i]);
    }
    let nested = Seq::weighted(10, [(Seq::weighted(4, [(Seq::source(2), 1.0)]), 1.0)]);
    let (_, report) = Order::prepare(nested, 0).unwrap();
    assert!(report.weighted[0].path.is_empty());
    assert_eq!(report.weighted[1].path, [0]);
    assert!(Order::prepare(Seq::source(1).take(2), 0).is_err());
    let (_, report) = Order::prepare(Seq::source(1usize << 30).repeat(1 << 20).stride(1 << 30, 0), 0).unwrap();
    assert_eq!(report.nodes[0].len, 1 << 20);
    assert!(report.nodes.iter().any(|node| node.len == 1u64 << 50));
}

#[test]
fn sharding_preserves_global_partition_not_worker_mixture() {
    let seq = Seq::mix([Seq::source(4).shuffle(1), Seq::source(4).shuffle(2)]);
    for worker in 0..2 {
        let order = Order::new(seq.clone().shard(2, worker)).unwrap();
        assert!(order.iter(..).indexed().all(|(ordinal, _, _)| ordinal == worker));
    }
    let (small, large) = (4, 5);
    for (total, expected) in [(small, [2, 1, 1]), (large, [3, 2, 0])] {
        let (_, report) =
            Order::prepare(Seq::weighted(total, [(Seq::source(10), 5.0), (Seq::source(10), 3.0), (Seq::source(10), 1.0)]), 0).unwrap();
        assert_eq!(report.weighted[0].counts, expected);
    }
}

#[test]
fn equality_preserves_nan_payload_sign_and_signaling_bits() {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let hash = |x: &Sampling| {
        let mut h = DefaultHasher::new();
        x.hash(&mut h);
        h.finish()
    };
    let bits = [0x7ff0000000000001, 0x7ff8000000000001, 0x7ff8000000000002, 0xfff8000000000001];
    for (i, a) in bits.iter().enumerate() {
        for (j, b) in bits.iter().enumerate() {
            let (a, b) = (f64::from_bits(*a), f64::from_bits(*b));
            assert_eq!(Sampling::delayed(a) == Sampling::delayed(b), i == j);
            let part = |weight| WeightedPart::from((Seq::source(1), weight));
            assert_eq!(part(a) == part(b), i == j);
        }
    }
    assert_eq!(Sampling::ramp(-0.0, 0.5), Sampling::ramp(0.0, 0.5));
    assert_eq!(hash(&Sampling::ramp(-0.0, 0.5)), hash(&Sampling::ramp(0.0, 0.5)));
}

#[test]
fn overcommit_diagnostic_explains_small_excess_and_location() {
    let off = 0.999999998;
    let error = Order::new(Seq::mix_with([(Seq::source(1000), Sampling::until(off))])).unwrap_err();
    let ErrorKind::Overcommitted { demand, start, end } = error.kind() else { panic!("{error}") };
    assert!(*demand > 1.0 + 1e-9);
    assert_eq!((*start, *end), (0.0, off));
    let text = error.to_string();
    assert!(text.contains("100.0000002"), "{text}");
    assert!(text.contains("progress 0..0.999999998"), "{text}");
}

#[test]
fn preparation_explains_folded_ranges_and_original_sources() {
    use dataorder::PreparedParameters as P;
    let seq = Seq::concat([Seq::source(4), Seq::concat([Seq::source(10), Seq::source(6)])]).skip(7).take(5);
    let (_, report) = Order::prepare(seq, 0).unwrap();
    assert_eq!(report.nodes.len(), 1);
    let node = &report.nodes[0];
    assert_eq!(node.parameters, P::Source { offset: 3 });
    assert_eq!(node.len, 5);
    assert_eq!(node.source_ordinal, Some(1));
    assert_eq!(report.sources[1].path, [0, 0, 1, 0]);
    assert_eq!(report.sources.iter().map(|s| s.len).collect::<Vec<_>>(), [4, 10, 6]);
    let (_, report) = Order::prepare(Seq::source(100).skip(5).stride(3, 1).skip(2).stride(2, 1), 0).unwrap();
    assert_eq!(report.nodes[0].parameters, P::Stride { step: 6, offset: 10 });
    assert_eq!(report.nodes[1].parameters, P::Source { offset: 5 });
    assert_eq!(report.sources[0].path, [0, 0, 0, 0]);
    let seq = Seq::source(10).shuffle(7).repeat(3).skip(2);
    let (_, report) = Order::prepare(seq, 0).unwrap();
    assert_eq!(report.nodes[0].parameters, P::Slice { start: 2 });
    assert_eq!(report.nodes[1].parameters, P::Repeat { child_len: 10, depth: 0 });
    assert!(matches!(report.nodes[2].parameters, P::Shuffle { seed: 7, .. }));
}

#[test]
fn sampling_diagnostics_distinguish_reasons_and_report_successful_tolerance() {
    use dataorder::{MAX_MIX_LEN, SamplingDetail as D};
    for (sampling, expected) in [
        (Sampling::delayed(f64::INFINITY), D::NonFiniteParameter),
        (Sampling::ramp(0.5, 0.25), D::InvalidBreakpoints),
        (Sampling::ramp(0.0, f64::from_bits(1)), D::CoefficientOverflow),
    ] {
        let err = Order::new(Seq::mix_with([(Seq::source(1), sampling)])).unwrap_err();
        assert!(matches!(err.kind(), ErrorKind::InvalidSampling { .. }));
        assert_eq!(err.sampling_detail(), Some(&expected));
        assert_eq!(err.path(), [0]);
        assert!(err.to_string().contains(&expected.to_string()));
    }
    let seq = Seq::mix_with([(Seq::source(1usize << 30), Sampling::until(1e-6))]);
    let err = Order::new(seq).unwrap_err();
    assert_eq!(err.sampling_detail(), Some(&D::TooSteep { len: 1 << 30, peak_rate: 1e6, limit: MAX_MIX_LEN }));
    let weighted = Seq::weighted_with(1 << 30, [(Seq::source(10), 3.0, Sampling::until(1e-6)), (Seq::source(10), 1.0, Sampling::Uniform)]);
    let err = Order::prepare(weighted, 0).unwrap_err();
    assert_eq!(err.path(), [0]);
    assert_eq!(err.sampling_detail(), Some(&D::TooSteep { len: 3 << 28, peak_rate: 1e6, limit: MAX_MIX_LEN }));
    let (_, report) = Order::prepare(Seq::mix_with([(Seq::source(1000), Sampling::until(1.0 - 5e-10))]).take(0), 0).unwrap();
    let mix = &report.mixes[0];
    assert_eq!(mix.path, [0]);
    assert_eq!(mix.counts, [1000]);
    assert!(mix.diagnostics.demand > 1.0);
    assert!(mix.diagnostics.used_tolerance);
    assert!(!mix.diagnostics.clamped_uniform); // No uniform records to clamp.
    assert_eq!((mix.diagnostics.start, mix.diagnostics.end), (0.0, 1.0 - 5e-10));
    #[cfg(target_pointer_width = "64")]
    {
        let (_, report) = Order::prepare(
            Seq::mix_with([(Seq::source((1usize << 44) - 1), Sampling::until(1.0 - 5e-10)), (Seq::source(1), Sampling::Uniform)]),
            0,
        )
        .unwrap();
        assert!(report.mixes[0].diagnostics.used_tolerance);
        assert!(report.mixes[0].diagnostics.clamped_uniform);
    }
    let (_, report) = Order::prepare(Seq::mix([Seq::source(10), Seq::source(20)]), 0).unwrap();
    assert_eq!(report.mixes[0].diagnostics.demand, 0.0);
    assert!(!report.mixes[0].diagnostics.used_tolerance);
    assert!(!report.mixes[0].diagnostics.clamped_uniform);
}
