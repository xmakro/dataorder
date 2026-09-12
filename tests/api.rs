//! The public surface, used as a downstream crate would: building configurations by hand,
//! matching on non-exhaustive enums, errors with paths, cursors, sources.

use dataorder::{Cursor, Error, ErrorKind, MixPart, Order, Schedule, Seq, Source};

#[derive(Clone, Debug, PartialEq)]
struct Shard {
    name: &'static str,
    len: usize,
}

impl Source for Shard {
    fn len(&self) -> usize {
        self.len
    }

    fn salt(&self) -> u64 {
        dataorder::salt(self.name)
    }
}

fn shard(name: &'static str, len: usize) -> Seq<Shard> {
    Seq::source(Shard { name, len })
}

fn names(cursor: Cursor<'_, Shard>) -> Vec<(&'static str, usize)> {
    cursor.map(|item| (item.source.name, item.record_index)).collect()
}

#[test]
fn builders_accept_unresolved_sources() {
    // No trait bounds, including Source or Clone, are needed to build the tree.
    fn configuration<T>(first: T, second: T) -> Seq<T> {
        Seq::concat([Seq::mix([Seq::source(first).shuffle()]), Seq::mix([(Seq::source(second), Schedule::Uniform)])])
            .repeat(2)
            .cycle_to(50)
            .skip(2)
            .take(40)
            .skip(1)
            .step_by(2)
            .skip(1)
            .take(8)
            .skip(1)
            .step_by(3)
    }
    struct Unresolved(&'static str);
    let seq = configuration(Unresolved("10"), Unresolved("20"));
    let mut visited = Vec::new();
    let seq = seq.map_sources(|source| {
        visited.push(source.0);
        source.0
    });
    assert_eq!(visited, ["10", "20"]);
    let order = Order::new(seq.try_map_sources(str::parse::<usize>).unwrap()).unwrap();
    let expected = Order::new(configuration(10, 20)).unwrap();
    assert_eq!(order.len(), 3);
    assert_eq!(order.sources(), [10, 20]);
    assert!(order.iter().eq(expected.iter()));
}

#[test]
fn shuffles_reject_mix_descendants_before_folding() {
    let mix = || Seq::mix([Seq::source(10), Seq::source(20)]);
    let cases = [
        mix(),
        Seq::mix([(Seq::source(10), Schedule::ramp(0.2, 0.6)), (Seq::source(20), Schedule::Uniform)]),
        Seq::concat([Seq::source(5), mix()]),
        mix().repeat(2),
        mix().cycle_to(60),
        mix().skip(1),
        mix().take(5),
        mix().step_by(2),
        Seq::concat([mix().repeat(2), Seq::source(5)]).skip(1).take(50).step_by(3),
        Seq::mix(std::iter::empty::<Seq<usize>>()),
        Seq::mix([Seq::source(0), Seq::source(0)]),
        Seq::mix([Seq::source(1)]),
        Seq::mix([Seq::source(10)]),
        mix().repeat(0),
        mix().cycle_to(0),
        mix().take(0),
        mix().skip(30),
        Seq::concat([Seq::source(5), mix()]).take(5),
        Seq::concat([mix(), Seq::source(5)]).skip(30),
    ];
    for inner in cases {
        // Build the variant directly and discard its output: validation must still
        // reject the mix and report the shuffle's original configuration path.
        let shuffled = Seq::Shuffle { inner: Box::new(inner) };
        let error = Order::new(Seq::concat([Seq::source(1), shuffled.repeat(0)])).unwrap_err();
        assert_eq!(error.kind(), &ErrorKind::ShuffleContainsMix);
        assert_eq!(error.path(), [1, 0]);
        assert_eq!(error.to_string(), "cannot shuffle a sequence containing a mix; shuffle its inputs before mixing (at node 1/0)");
    }
}

#[test]
fn shuffle_restrictions_follow_ancestors_and_allow_mixed_siblings() {
    let mix = || Seq::mix([Seq::source(10), Seq::source(20)]);
    let error = Order::with_seed(mix().shuffle().shuffle(), 3).unwrap_err();
    assert_eq!(error.kind(), &ErrorKind::ShuffleContainsMix);
    assert_eq!(error.path(), [0]); // The nearest enclosing shuffle.

    // Finishing an inner shuffle must restore the enclosing restriction.
    let error = Order::new(Seq::concat([Seq::source(10).shuffle(), mix()]).shuffle()).unwrap_err();
    assert_eq!(error.kind(), &ErrorKind::ShuffleContainsMix);
    assert!(error.path().is_empty());

    // After leaving a shuffle, later siblings may contain mixes. Nested scheduled
    // mixes, shuffled concatenations, repeats and selections remain supported.
    let seq = Seq::mix([
        (Seq::concat([Seq::source(10), Seq::source(20).shuffle()]).repeat(2).shuffle().cycle_to(100), Schedule::Uniform),
        (Seq::concat([Seq::source(10).shuffle().shuffle(), mix()]).skip(2), Schedule::delayed(0.3)),
    ])
    .repeat(2)
    .skip(1)
    .step_by(3);
    let order = Order::new(seq).unwrap();
    assert!(order.iter().eq((0..order.len()).map(|pos| order.get(pos).unwrap())));
}

#[test]
#[cfg(feature = "serde")]
fn deserialized_shuffles_cannot_contain_mixes() {
    let seq: Seq<usize> = serde_json::from_str(r#"{"Shuffle":{"inner":{"Mix":[]}}}"#).unwrap();
    let error = Order::new(seq).unwrap_err();
    assert_eq!(error.kind(), &ErrorKind::ShuffleContainsMix);
    assert!(error.path().is_empty());
}

#[test]
fn configuration_errors_are_validated_only_when_compiling() {
    let invalid = [
        (Seq::source("data").step_by(0), ErrorKind::ZeroStep),
        (Seq::source("data").skip(11), ErrorKind::SkipOutOfRange { n: 11, len: 10 }),
        (Seq::source("data").take(11), ErrorKind::TakeOutOfRange { n: 11, len: 10 }),
    ];
    for (seq, expected) in invalid {
        // Mapping must preserve invalid nodes, including in an empty subtree.
        let seq = Seq::concat([Seq::source("other"), seq.repeat(0)]).map_sources(|_| 10usize);
        let error = Order::new(seq.clone()).unwrap_err();
        assert_eq!(error.kind(), &expected);
        assert_eq!(error.path(), [1, 0]);
        assert_eq!(Order::with_seed(seq.clone(), 7).unwrap_err(), error);
        assert_eq!(Order::try_from(seq).unwrap_err(), error);
    }
}

#[test]
fn workers_partition_short_sequences_with_explicit_offsets() {
    let workers = 4;
    for len in [0, 1, 3, 10] {
        let mut combined = Vec::new();
        for worker in 0..workers {
            let seq = Seq::source(len).skip(worker.min(len)).step_by(workers);
            let order = Order::new(seq).unwrap();
            let positions = order.iter().map(|item| item.record_index).collect::<Vec<_>>();
            assert_eq!(positions, (worker..len).step_by(workers).collect::<Vec<_>>());
            combined.extend(positions);
        }
        combined.sort_unstable();
        assert_eq!(combined, (0..len).collect::<Vec<_>>());
    }
    // Offsets are sequence positions; they need not be smaller than the step.
    let order = Order::new(Seq::source(10).skip(5).step_by(2)).unwrap();
    assert_eq!(order.iter().map(|item| item.record_index).collect::<Vec<_>>(), [5, 7, 9]);
    assert_eq!(Order::new(Seq::source(3).skip(4).step_by(workers)).unwrap_err().kind(), &ErrorKind::SkipOutOfRange { n: 4, len: 3 });
}

#[cfg(feature = "serde")]
#[test]
fn unresolved_position_operations_round_trip() {
    let seq = Seq::source("data").skip(2).take(7).step_by(3).skip(1).step_by(2);
    let json = serde_json::to_string(&seq).unwrap();
    assert_eq!(
        json,
        r#"{"StepBy":{"step":2,"inner":{"Skip":{"n":1,"inner":{"StepBy":{"step":3,"inner":{"Take":{"n":7,"inner":{"Skip":{"n":2,"inner":{"Source":"data"}}}}}}}}}}}"#
    );
    let back: Seq<String> = serde_json::from_str(&json).unwrap();
    let order = Order::new(back.map_sources(|_| 10usize)).unwrap();
    assert_eq!(order.iter().map(|item| item.record_index).collect::<Vec<_>>(), [5]);
    let seq = Seq::source("data").skip(11).step_by(0);
    let json = serde_json::to_string(&seq).unwrap();
    let back: Seq<String> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, seq.map_sources(str::to_owned));
    // The invalid skip is found first: a stride validates its input before its step.
    let error = Order::new(back.map_sources(|_| 10usize)).unwrap_err();
    assert_eq!(error.kind(), &ErrorKind::SkipOutOfRange { n: 11, len: 10 });
    assert_eq!(error.path(), [0]);
}

#[test]
fn hand_built_configuration() {
    let seq = Seq::Mix(vec![
        MixPart { seq: shard("a", 10).shuffle().cycle_to(75), schedule: Schedule::Uniform },
        MixPart::from(
            Seq::Mix(vec![MixPart::from(shard("b", 40)), MixPart { seq: shard("c", 5), schedule: Schedule::delayed(0.5) }]).cycle_to(25),
        ),
    ]);
    // The mix builder takes parts, pairs or bare sequences alike.
    let Seq::Mix(parts) = seq.clone() else { unreachable!() };
    assert_eq!(Seq::mix(parts), seq);
    let mixed = Seq::mix([MixPart::from(shard("b", 40)), MixPart { seq: shard("c", 5), schedule: Schedule::default() }]);
    assert_eq!(mixed, Seq::mix([(shard("b", 40), Schedule::Uniform), (shard("c", 5), Schedule::Uniform)]));
    assert_eq!(mixed, Seq::mix([shard("b", 40), shard("c", 5)]));
    let empty = Seq::mix(std::iter::empty::<Seq<Shard>>());
    assert!(Order::new(empty).unwrap().is_empty());
    let order: Order<Shard> = seq.try_into().unwrap();
    assert_eq!(order.len(), 100);
    let all = names(order.iter());
    assert_eq!(all.iter().filter(|e| e.0 == "a").count(), 75);
    assert_eq!(all.iter().filter(|e| e.0 == "b" || e.0 == "c").count(), 25);
    assert_eq!(order.sources().iter().map(|s| s.name).collect::<Vec<_>>(), ["a", "b", "c"]);
    let sources = order.into_sources();
    assert_eq!(sources.len(), 3);
}

#[test]
fn errors_name_kind_and_path() {
    let seq = Seq::concat([shard("a", 10), Seq::mix([shard("b", 3), shard("c", 4).skip(5)])]);
    let err: Error = Order::new(seq).unwrap_err();
    // Downstream matches need a wildcard: the enums are non-exhaustive.
    let described = match err.kind() {
        ErrorKind::SkipOutOfRange { n, len } => format!("skip {n} of {len}"),
        ErrorKind::TakeOutOfRange { .. } | ErrorKind::ZeroStep => "other".to_string(),
        _ => "unknown".to_string(),
    };
    assert_eq!(described, "skip 5 of 4");
    assert_eq!(err.path(), [1, 1]);
    assert_eq!(err.to_string(), "cannot skip 5 of 4 positions (at node 1/1)");
    let _: &dyn std::error::Error = &err;
    let kind = err.into_kind();
    assert_eq!(kind, ErrorKind::SkipOutOfRange { n: 5, len: 4 });
    // The numerical mix limit is reachable on 64-bit targets and named in the message.
    #[cfg(target_pointer_width = "64")]
    {
        let long = Seq::mix([Seq::source(1usize << 30).repeat(1 << 17), Seq::source(1)]);
        let err = Order::new(long).unwrap_err();
        assert_eq!(err.kind(), &ErrorKind::MixTooLong);
        assert!(err.to_string().contains(&dataorder::MAX_MIX_LEN.to_string()));
    }
}

#[test]
fn full_iteration_covers_empty_and_maximum_lengths() {
    for len in [0, 1, 17, usize::MAX] {
        let order = Order::new(Seq::source(len)).unwrap();
        for mut cursor in [order.iter(), order.cursor(..).unwrap(), (&order).into_iter()] {
            assert_eq!(cursor.offset(), 0);
            assert_eq!(cursor.len(), len);
            assert_eq!(cursor.next(), order.get(0));
            cursor.reset(len..).unwrap();
            assert_eq!(cursor.len(), 0);
            assert_eq!(cursor.next(), None);
            cursor.reset(len.saturating_sub(1)..).unwrap();
            assert_eq!(cursor.next(), order.get(len.saturating_sub(1)));
            cursor.reset(..).unwrap();
            assert_eq!(cursor.count(), len);
        }
    }
}

#[test]
fn resets_replace_remaining_ranges_in_order_coordinates() {
    use std::ops::Bound::{Excluded, Included, Unbounded};

    let mix = Seq::mix([shard("a", 250).shuffle(), shard("b", 150)]);
    for seq in [shard("a", 400), mix, Seq::concat([shard("a", 250), shard("b", 150)]).shuffle()] {
        let order = Order::new(seq).unwrap();
        let mut cursor = order.cursor(100..200).unwrap();
        assert_eq!(cursor.next(), order.get(100));
        for (bounds, expected) in [
            ((Included(150), Excluded(200)), 150..200),
            ((Included(300), Excluded(400)), 300..400),
            ((Included(150), Unbounded), 150..400),
            ((Included(100), Excluded(100)), 100..100),
            ((Included(100), Included(100)), 100..101),
            ((Unbounded, Included(2)), 0..3),
            ((Excluded(1), Included(3)), 2..4),
            ((Unbounded, Unbounded), 0..400),
            ((Included(400), Unbounded), 400..400),
            ((Unbounded, Unbounded), 0..400),
        ] {
            cursor.reset(bounds).unwrap();
            assert_eq!(cursor.offset(), expected.start);
            assert_eq!(cursor.len(), expected.len());
            assert_eq!(cursor.clone().count(), expected.len());
            assert_eq!(cursor.clone().last(), expected.clone().last().and_then(|pos| order.get(pos)));
            let items: Vec<_> = expected.clone().map(|pos| order.get(pos).unwrap()).collect();
            assert_eq!(cursor.by_ref().collect::<Vec<_>>(), items);
            assert_eq!(cursor.offset(), expected.end);
            assert_eq!(cursor.len(), 0);
            assert_eq!(cursor.next(), None);
        }
    }
}

#[test]
fn cursors_reset_skip_and_clone() {
    let order = Order::new(Seq::mix([shard("a", 300).shuffle().repeat(2), shard("b", 100).shuffle()]).skip(2).step_by(3)).unwrap();
    let all = names(order.iter());
    assert_eq!(all.len(), order.len());
    let mut cursor = order.iter();
    assert_eq!(cursor.nth(10).map(|item| (item.source.name, item.record_index)), Some(all[10]));
    cursor.reset(100..).unwrap();
    let ahead = cursor.clone();
    assert_eq!(names(cursor), all[100..]);
    assert_eq!(names(ahead), all[100..]);
    let mut back = order.cursor(50..60).unwrap();
    back.reset(55..60).unwrap();
    assert_eq!(back.len(), 5);
    back.reset(52..60).unwrap();
    assert_eq!(names(back), all[52..60]);
    for (i, e) in (&order).into_iter().enumerate() {
        assert_eq!((e.source.name, e.record_index), all[i]);
        assert_eq!(order.get(i).unwrap(), e);
    }
}

#[test]
fn sources_through_pointers_and_lengths() {
    use std::rc::Rc;
    use std::sync::Arc;
    let shared = Arc::new(Shard { name: "s", len: 7 });
    let seq: Seq<Box<dyn Source>> = Seq::concat([
        Seq::source(Box::new(5usize) as Box<dyn Source>),
        Seq::source(Box::new(Rc::new(3usize)) as Box<dyn Source>),
        Seq::source(Box::new(shared.clone()) as Box<dyn Source>),
        Seq::source(Box::new(&*shared) as Box<dyn Source>),
    ]);
    assert_eq!(Order::new(seq).unwrap().len(), 22);
    let mut n = 3usize;
    assert_eq!(Order::new(Seq::source(&mut n)).unwrap().len(), 3);
    let lens = Seq::mix([Seq::source(4), Seq::source(6)]).map_sources(|n| n * 2);
    assert_eq!(Order::new(lens.clone()).unwrap().len(), 20);
    let opened = lens.clone().try_map_sources(|n| if n < 10 { Ok(shard("x", n).map_sources(|s| s.len)) } else { Err(n) });
    assert_eq!(opened, Err(12));
    assert_eq!(Order::new(lens.try_map_sources(|n| Ok::<_, ()>(n / 2)).unwrap()).unwrap().len(), 10);
    // Salts pass through pointers; slices, arrays and vectors are sources of their elements.
    let boxed: Box<&Shard> = Box::new(&shared);
    assert_eq!(boxed.salt(), dataorder::salt("s"));
    assert_eq!((&&shared).salt(), shared.salt());
    let order = Order::new(Seq::concat([Seq::source(vec!['a', 'b', 'c']), Seq::source(['d', 'e'].to_vec())]).shuffle()).unwrap();
    let letters: String = order.iter().map(|item| item.source[item.record_index]).collect();
    assert_eq!(letters.len(), 5);
    assert_eq!(Order::new(Seq::source(&[1u8, 2, 3][..])).unwrap().len(), 3);
    assert_eq!(Order::new(Seq::source([0u8; 4])).unwrap().len(), 4);
}

#[cfg(feature = "serde")]
#[test]
fn serde_round_trip() {
    let seq = Seq::mix([(Seq::source(10).shuffle(), Schedule::Uniform), (Seq::source(5), Schedule::ramp(0.2, 0.6))]).skip(1).step_by(2);
    let json = serde_json::to_string(&seq).unwrap();
    let back: Seq<usize> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, seq);
    let (a, b) = (Order::new(seq).unwrap(), Order::new(back).unwrap());
    assert!(a.iter().eq(b.iter()));
    // The wire format is part of the API.
    let seq: Seq<usize> = Seq::mix([(Seq::source(4).shuffle().cycle_to(9), Schedule::ramp(0.1, 0.2))]);
    assert_eq!(
        serde_json::to_string(&seq).unwrap(),
        r#"{"Mix":[{"seq":{"Cycle":{"len":9,"shuffled":false,"inner":{"Shuffle":{"inner":{"Source":4}}}}},"schedule":{"Trapezoid":{"start":0.1,"full":0.2,"fade":1.0,"off":1.0}}}]}"#
    );
    assert_eq!(serde_json::to_string(&Schedule::delayed(0.5)).unwrap(), r#"{"Trapezoid":{"start":0.5,"full":0.5,"fade":1.0,"off":1.0}}"#);
    assert_eq!(serde_json::to_string(&Schedule::until(0.5)).unwrap(), r#"{"Trapezoid":{"start":0.0,"full":0.0,"fade":0.5,"off":0.5}}"#);
    assert_eq!(
        serde_json::to_string(&Seq::source(4usize).cycle_to(9)).unwrap(),
        r#"{"Cycle":{"len":9,"shuffled":false,"inner":{"Source":4}}}"#
    );
    // Repetitions written before the `shuffled` field existed load as plain repetitions.
    assert_eq!(serde_json::from_str::<Seq<usize>>(r#"{"Repeat":{"times":2,"inner":{"Source":4}}}"#).unwrap(), Seq::source(4).repeat(2));
    assert_eq!(serde_json::from_str::<Seq<usize>>(r#"{"Cycle":{"len":9,"inner":{"Source":4}}}"#).unwrap(), Seq::source(4).cycle_to(9));
    // Unknown fields are rejected in every variant.
    for json in [
        r#"{"Skip":{"n":1,"inner":{"Source":5},"bogus":1}}"#,
        r#"{"Shuffle":{"inner":{"Source":5},"extra":true}}"#,
        r#"{"Mix":[{"seq":{"Source":4},"schedule":"Uniform","extra":1}]}"#,
        r#"{"Mix":[{"seq":{"Source":4},"schedule":{"Trapezoid":{"start":0.1,"full":0.2,"fade":1.0,"off":1.0,"end":0.3}}}]}"#,
    ] {
        assert!(serde_json::from_str::<Seq<usize>>(json).is_err(), "{json}");
    }
    // Removed fields and variants are rejected rather than silently reinterpreted.
    for json in [
        r#"{"Shuffle":{"seed":0,"inner":{"Source":5}}}"#,
        r#"{"Shuffle":{"seed":7,"inner":{"Source":5}}}"#,
        r#"{"Mix":[{"seq":{"Source":4},"sampling":"Uniform"}]}"#,
        r#"{"Mix":[{"seq":{"Source":4},"schedule":"Uniform","sampling":"Uniform"}]}"#,
        r#"{"Mix":[{"seq":{"Source":4},"schedule":{"DelayedLinear":{"start":0.1,"full":0.2}}}]}"#,
        r#"{"Weighted":{"total":9,"parts":[]}}"#,
        r#"{"Stride":{"step":2,"offset":1,"inner":{"Source":4}}}"#,
        r#"{"Shard":{"count":2,"index":1,"inner":{"Source":4}}}"#,
        r#"{"Slice":{"start":"Unbounded","end":"Unbounded","inner":{"Source":4}}}"#,
        r#"{"StepBy":{"step":2,"offset":1,"inner":{"Source":4}}}"#,
        r#"{"ShuffledRepeat":{"times":3,"inner":{"Source":4}}}"#,
        r#"{"ShuffledCycle":{"len":9,"inner":{"Source":4}}}"#,
    ] {
        assert!(serde_json::from_str::<Seq<usize>>(json).is_err(), "{json}");
    }
}

#[cfg(feature = "serde")]
#[test]
fn serde_round_trips_nested_configurations() {
    for variant in ["Take", "Mix"] {
        let seq = (1..16).fold(Seq::source(10usize), |seq, _| match variant {
            "Take" => seq.take(10),
            _ => Seq::mix([seq]),
        });
        let json = serde_json::to_string(&seq).unwrap();
        let back: Seq<usize> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, seq, "{variant}");
        assert_eq!(Order::new(back).unwrap().len(), 10);
    }
}
