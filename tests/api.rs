//! The public surface, used as a downstream crate would: building configurations by hand,
//! matching on non-exhaustive enums, errors with paths, cursors, sources.

use dataorder::{Cursor, Error, ErrorKind, MAX_MIX_LEN, MixPart, Order, Sampling, Seq, Source, WeightedPart};

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
    cursor.map(|(s, i)| (s.name, i)).collect()
}

#[test]
fn hand_built_configuration() {
    let seq = Seq::Weighted {
        total: 100,
        parts: vec![
            WeightedPart { seq: shard("a", 10).shuffle(1), weight: 3.0, sampling: Sampling::Uniform },
            WeightedPart::from((
                Seq::Mix(vec![MixPart::from(shard("b", 40)), MixPart { seq: shard("c", 5), sampling: Sampling::delayed(0.5) }]),
                1.0,
            )),
        ],
    };
    assert_eq!(seq.check(), Ok(100));
    // The builders take parts, pairs or bare sequences alike.
    let Seq::Weighted { parts, .. } = seq.clone() else { unreachable!() };
    assert_eq!(Seq::weighted_with(100, parts), seq);
    let mixed = Seq::mix_with([MixPart::from(shard("b", 40)), MixPart { seq: shard("c", 5), sampling: Sampling::default() }]);
    assert_eq!(mixed, Seq::mix_with([(shard("b", 40), Sampling::Uniform), (shard("c", 5), Sampling::Uniform)]));
    assert_eq!(mixed, Seq::mix_with([shard("b", 40), shard("c", 5)]));
    let mut order: Order<Shard> = seq.try_into().unwrap();
    order.sources_mut()[0].name = "A";
    let all = names(order.iter(..));
    assert_eq!(all.iter().filter(|e| e.0 == "A").count(), 75);
    assert_eq!(all.iter().filter(|e| e.0 == "b" || e.0 == "c").count(), 25);
    assert_eq!(order.sources().iter().map(|s| s.name).collect::<Vec<_>>(), ["A", "b", "c"]);
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
    let all = names(order.iter(..));
    assert_eq!(all.len(), order.len());
    let mut cursor = order.iter(..);
    assert_eq!(cursor.nth(10).map(|(s, i)| (s.name, i)), Some(all[10]));
    cursor.seek(100);
    let ahead = cursor.clone();
    assert_eq!(names(cursor), all[100..]);
    assert_eq!(names(ahead), all[100..]);
    let mut back = order.iter(50..60);
    back.seek(55);
    assert_eq!(back.len(), 5);
    back.seek(52);
    assert_eq!(names(back), all[52..60]);
    for (i, e) in (&order).into_iter().enumerate() {
        assert_eq!((e.0.name, e.1), all[i]);
        assert_eq!(order.get(i), (e.0, e.1));
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
    assert_eq!(seq.check(), Ok(22));
    let mut n = 3usize;
    assert_eq!(Seq::source(&mut n).check(), Ok(3));
    let lens = Seq::mix([Seq::source(4), Seq::source(6)]).map(|n| n * 2);
    assert_eq!(lens.check(), Ok(20));
    let opened = lens.clone().try_map(|n| if n < 10 { Ok(shard("x", n).map(|s| s.len)) } else { Err(n) });
    assert_eq!(opened, Err(12));
    assert_eq!(lens.try_map(|n| Ok::<_, ()>(n / 2)).unwrap().check(), Ok(10));
    // Salts pass through pointers; slices, arrays and vectors are sources of their elements.
    let boxed: Box<&Shard> = Box::new(&shared);
    assert_eq!(boxed.salt(), dataorder::salt("s"));
    assert_eq!((&&shared).salt(), shared.salt());
    let order = Order::new(Seq::concat([Seq::source(vec!['a', 'b', 'c']), Seq::source(['d', 'e'].to_vec())]).shuffle(1)).unwrap();
    let letters: String = order.iter(..).map(|(v, i)| v[i]).collect();
    assert_eq!(letters.len(), 5);
    assert_eq!(Seq::source(&[1u8, 2, 3][..]).check(), Ok(3));
    assert_eq!(Seq::source([0u8; 4]).check(), Ok(4));
}

#[cfg(feature = "serde")]
#[test]
fn serde_round_trip() {
    let seq = Seq::mix_with([(Seq::source(10).shuffle(1), Sampling::Uniform), (Seq::source(5), Sampling::ramp(0.2, 0.6))]).shard(2, 1);
    let json = serde_json::to_string(&seq).unwrap();
    let back: Seq<usize> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, seq);
    let (a, b) = (Order::new(seq).unwrap(), Order::new(back).unwrap());
    assert!(a.iter(..).eq(b.iter(..)));
    // The wire format is part of the API.
    let seq: Seq<usize> = Seq::weighted_with(9, [(Seq::source(4).shuffle(1), 0.5, Sampling::ramp(0.1, 0.2))]);
    assert_eq!(
        serde_json::to_string(&seq).unwrap(),
        r#"{"Weighted":{"total":9,"parts":[{"seq":{"Shuffle":{"seed":1,"inner":{"Source":4}}},"weight":0.5,"sampling":{"DelayedLinear":{"start":0.1,"full":0.2}}}]}}"#
    );
    assert_eq!(serde_json::to_string(&Sampling::until(0.5)).unwrap(), r#"{"Trapezoid":{"start":0.0,"full":0.0,"fade":0.5,"off":0.5}}"#);
    assert_eq!(serde_json::to_string(&Seq::source(4usize).cycle(9)).unwrap(), r#"{"Cycle":{"len":9,"inner":{"Source":4}}}"#);
    // Unknown fields are rejected in every variant.
    for json in [
        r#"{"Skip":{"n":1,"inner":{"Source":5},"bogus":1}}"#,
        r#"{"Shuffle":{"seed":1,"inner":{"Source":5},"extra":true}}"#,
        r#"{"Weighted":{"total":9,"parts":[],"x":0}}"#,
        r#"{"Mix":[{"seq":{"Source":4},"sampling":"Uniform","extra":1}]}"#,
        r#"{"Mix":[{"seq":{"Source":4},"sampling":{"DelayedLinear":{"start":0.1,"full":0.2,"end":0.3}}}]}"#,
    ] {
        assert!(serde_json::from_str::<Seq<usize>>(json).is_err(), "{json}");
    }
}

#[cfg(feature = "serde")]
#[test]
fn serde_round_trips_at_the_supported_depth_limit() {
    for variant in ["Take", "Mix", "Weighted"] {
        let seq = (1..dataorder::MAX_DEPTH).fold(Seq::source(10usize), |seq, _| match variant {
            "Take" => seq.take(10),
            "Mix" => Seq::mix([seq]),
            _ => Seq::weighted(10, [(seq, 1.0)]),
        });
        assert_eq!(seq.check(), Ok(10));
        let json = serde_json::to_string(&seq).unwrap();
        let back: Seq<usize> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, seq, "{variant}");
        assert_eq!(back.check(), Ok(10));
    }
}
