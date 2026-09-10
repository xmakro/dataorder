//! The public surface, used as a downstream crate would: building configurations by hand,
//! matching on non-exhaustive enums, errors with paths, cursors, sources.

use dataorder::{Cursor, Error, ErrorKind, MAX_MIX_LEN, MixPart, Order, Sampling, Seq, Source};

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
        Seq::concat([Seq::mix([Seq::source(first)]), Seq::mix_with([(Seq::source(second), Sampling::Uniform)])])
            .shuffle(7)
            .repeat(2)
            .cycle(50)
            .skip(2)
            .take(40)
            .skip(1)
            .step_by(2)
            .skip(1)
            .take(8)
            .shard(3, 1)
    }
    struct Unresolved(&'static str);
    let seq = configuration(Unresolved("10"), Unresolved("20"));
    let mut visited = Vec::new();
    let seq = seq.map(|source| {
        visited.push(source.0);
        source.0
    });
    assert_eq!(visited, ["10", "20"]);
    let order = Order::new(seq.try_map(str::parse::<usize>).unwrap()).unwrap();
    let expected = Order::new(configuration(10, 20)).unwrap();
    assert_eq!(order.len(), 3);
    assert_eq!(order.sources(), [10, 20]);
    assert!(order.iter(..).unwrap().eq(expected.iter(..).unwrap()));
}

#[test]
fn configuration_errors_are_validated_only_when_compiling() {
    let invalid = [
        (Seq::source("data").step_by(0), ErrorKind::ZeroStep),
        (Seq::source("data").skip(11), ErrorKind::SkipOutOfRange { n: 11, len: 10 }),
        (Seq::source("data").take(11), ErrorKind::TakeOutOfRange { n: 11, len: 10 }),
        (Seq::source("data").shard(0, 0), ErrorKind::InvalidShard { count: 0, index: 0 }),
        (Seq::source("data").shard(2, 2), ErrorKind::InvalidShard { count: 2, index: 2 }),
    ];
    for (seq, expected) in invalid {
        // Mapping must preserve invalid nodes, including in an empty subtree.
        let seq = Seq::concat([Seq::source("other"), seq.repeat(0)]).map(|_| 10usize);
        let error = Order::new(seq.clone()).unwrap_err();
        assert_eq!(error.kind(), &expected);
        assert_eq!(error.path(), [1, 0]);
        assert_eq!(Order::with_seed(seq.clone(), 7).unwrap_err(), error);
        assert_eq!(Order::try_from(seq).unwrap_err(), error);
    }
}

#[test]
fn step_by_and_shard_count_as_configuration_nodes() {
    for step_by in [false, true] {
        let chain = |levels| (1..levels).fold(Seq::source(10), |seq, _| if step_by { seq.step_by(1) } else { seq.shard(1, 0) });
        assert_eq!(Order::new(chain(dataorder::MAX_DEPTH)).unwrap().len(), 10);
        let error = Order::new(chain(dataorder::MAX_DEPTH + 1)).unwrap_err();
        assert_eq!(error.kind(), &ErrorKind::TooDeep);
        assert_eq!(error.path().len(), dataorder::MAX_DEPTH as usize);
    }
}

#[cfg(feature = "serde")]
#[test]
fn unresolved_position_operations_round_trip() {
    let seq = Seq::source("data").skip(2).take(7).step_by(3).shard(2, 1);
    let json = serde_json::to_string(&seq).unwrap();
    assert_eq!(
        json,
        r#"{"Shard":{"count":2,"index":1,"inner":{"StepBy":{"step":3,"inner":{"Take":{"n":7,"inner":{"Skip":{"n":2,"inner":{"Source":"data"}}}}}}}}}"#
    );
    let back: Seq<String> = serde_json::from_str(&json).unwrap();
    let order = Order::new(back.map(|_| 10usize)).unwrap();
    assert_eq!(order.iter(..).unwrap().map(|item| item.record_index).collect::<Vec<_>>(), [5]);
    let seq = Seq::source("data").step_by(0).shard(0, 0);
    let json = serde_json::to_string(&seq).unwrap();
    let back: Seq<String> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, seq.map(str::to_owned));
    assert_eq!(Order::new(back.map(|_| 10usize)).unwrap_err().kind(), &ErrorKind::InvalidShard { count: 0, index: 0 });
}

#[test]
fn hand_built_configuration() {
    let seq = Seq::Mix(vec![
        MixPart { seq: shard("a", 10).shuffle(1).cycle(75), sampling: Sampling::Uniform },
        MixPart::from(
            Seq::Mix(vec![MixPart::from(shard("b", 40)), MixPart { seq: shard("c", 5), sampling: Sampling::delayed(0.5) }]).cycle(25),
        ),
    ]);
    // The builders take parts, pairs or bare sequences alike.
    let Seq::Mix(parts) = seq.clone() else { unreachable!() };
    assert_eq!(Seq::mix_with(parts), seq);
    let mixed = Seq::mix_with([MixPart::from(shard("b", 40)), MixPart { seq: shard("c", 5), sampling: Sampling::default() }]);
    assert_eq!(mixed, Seq::mix_with([(shard("b", 40), Sampling::Uniform), (shard("c", 5), Sampling::Uniform)]));
    assert_eq!(mixed, Seq::mix_with([shard("b", 40), shard("c", 5)]));
    let order: Order<Shard> = seq.try_into().unwrap();
    assert_eq!(order.len(), 100);
    let all = names(order.iter(..).unwrap());
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
    // The mix limit is public, and named in the message.
    let long = Seq::mix([Seq::source(1usize << 30).repeat(1 << 17), Seq::source(1)]);
    let err = Order::new(long).unwrap_err();
    assert_eq!(err.kind(), &ErrorKind::MixTooLong);
    assert!(err.to_string().contains(&MAX_MIX_LEN.to_string()));
}

#[test]
fn cursors_seek_skip_and_clone() {
    let order = Order::new(Seq::mix([shard("a", 300).shuffle(1).repeat(2), shard("b", 100).shuffle(2)]).shard(3, 2)).unwrap();
    let all = names(order.iter(..).unwrap());
    assert_eq!(all.len(), order.len());
    let mut cursor = order.iter(..).unwrap();
    assert_eq!(cursor.nth(10).map(|item| (item.source.name, item.record_index)), Some(all[10]));
    cursor.seek(100).unwrap();
    let ahead = cursor.clone();
    assert_eq!(names(cursor), all[100..]);
    assert_eq!(names(ahead), all[100..]);
    let mut back = order.iter(50..60).unwrap();
    back.seek(55).unwrap();
    assert_eq!(back.len(), 5);
    back.seek(52).unwrap();
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
    let lens = Seq::mix([Seq::source(4), Seq::source(6)]).map(|n| n * 2);
    assert_eq!(Order::new(lens.clone()).unwrap().len(), 20);
    let opened = lens.clone().try_map(|n| if n < 10 { Ok(shard("x", n).map(|s| s.len)) } else { Err(n) });
    assert_eq!(opened, Err(12));
    assert_eq!(Order::new(lens.try_map(|n| Ok::<_, ()>(n / 2)).unwrap()).unwrap().len(), 10);
    // Salts pass through pointers; slices, arrays and vectors are sources of their elements.
    let boxed: Box<&Shard> = Box::new(&shared);
    assert_eq!(boxed.salt(), dataorder::salt("s"));
    assert_eq!((&&shared).salt(), shared.salt());
    let order = Order::new(Seq::concat([Seq::source(vec!['a', 'b', 'c']), Seq::source(['d', 'e'].to_vec())]).shuffle(1)).unwrap();
    let letters: String = order.iter(..).unwrap().map(|item| item.source[item.record_index]).collect();
    assert_eq!(letters.len(), 5);
    assert_eq!(Order::new(Seq::source(&[1u8, 2, 3][..])).unwrap().len(), 3);
    assert_eq!(Order::new(Seq::source([0u8; 4])).unwrap().len(), 4);
}

#[cfg(feature = "serde")]
#[test]
fn serde_round_trip() {
    let seq = Seq::mix_with([(Seq::source(10).shuffle(1), Sampling::Uniform), (Seq::source(5), Sampling::ramp(0.2, 0.6))]).shard(2, 1);
    let json = serde_json::to_string(&seq).unwrap();
    let back: Seq<usize> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, seq);
    let (a, b) = (Order::new(seq).unwrap(), Order::new(back).unwrap());
    assert!(a.iter(..).unwrap().eq(b.iter(..).unwrap()));
    // The wire format is part of the API.
    let seq: Seq<usize> = Seq::mix_with([(Seq::source(4).shuffle(1).cycle(9), Sampling::ramp(0.1, 0.2))]);
    assert_eq!(
        serde_json::to_string(&seq).unwrap(),
        r#"{"Mix":[{"seq":{"Cycle":{"len":9,"inner":{"Shuffle":{"seed":1,"inner":{"Source":4}}}}},"sampling":{"DelayedLinear":{"start":0.1,"full":0.2}}}]}"#
    );
    assert_eq!(serde_json::to_string(&Sampling::until(0.5)).unwrap(), r#"{"Trapezoid":{"start":0.0,"full":0.0,"fade":0.5,"off":0.5}}"#);
    assert_eq!(serde_json::to_string(&Seq::source(4usize).cycle(9)).unwrap(), r#"{"Cycle":{"len":9,"inner":{"Source":4}}}"#);
    // Unknown fields are rejected in every variant.
    for json in [
        r#"{"Skip":{"n":1,"inner":{"Source":5},"bogus":1}}"#,
        r#"{"Shuffle":{"seed":1,"inner":{"Source":5},"extra":true}}"#,
        r#"{"Mix":[{"seq":{"Source":4},"sampling":"Uniform","extra":1}]}"#,
        r#"{"Mix":[{"seq":{"Source":4},"sampling":{"DelayedLinear":{"start":0.1,"full":0.2,"end":0.3}}}]}"#,
    ] {
        assert!(serde_json::from_str::<Seq<usize>>(json).is_err(), "{json}");
    }
    // Removed variants are rejected rather than silently reinterpreted.
    for json in [
        r#"{"Weighted":{"total":9,"parts":[]}}"#,
        r#"{"Stride":{"step":2,"offset":1,"inner":{"Source":4}}}"#,
        r#"{"Slice":{"start":"Unbounded","end":"Unbounded","inner":{"Source":4}}}"#,
        r#"{"StepBy":{"step":2,"offset":1,"inner":{"Source":4}}}"#,
    ] {
        assert!(serde_json::from_str::<Seq<usize>>(json).is_err(), "{json}");
    }
}

#[cfg(feature = "serde")]
#[test]
fn serde_round_trips_at_the_supported_depth_limit() {
    for variant in ["Take", "Mix"] {
        let seq = (1..dataorder::MAX_DEPTH).fold(Seq::source(10usize), |seq, _| match variant {
            "Take" => seq.take(10),
            _ => Seq::mix([seq]),
        });
        let json = serde_json::to_string(&seq).unwrap();
        let back: Seq<usize> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, seq, "{variant}");
        assert_eq!(Order::new(back).unwrap().len(), 10);
    }
}
