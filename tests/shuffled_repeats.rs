//! Repetition owns its permutation; it never changes permutations in its input.
use dataorder::{ErrorKind, Order, Seq};

fn indices(order: &Order<usize>) -> Vec<usize> {
    order.iter().map(|item| item.record_index).collect()
}

fn check_access(order: &Order<usize>) {
    let expected: Vec<_> = order.iter().collect();
    for (pos, item) in expected.iter().enumerate() {
        assert_eq!(order.get(pos), Some(*item));
    }
    let mut cursor = order.iter();
    for pos in [order.len(), 0, order.len() / 2, 1, order.len() - 1, 0] {
        cursor.reset(pos..).unwrap();
        assert_eq!(cursor.clone().collect::<Vec<_>>(), expected[pos..]);
        assert_eq!(cursor.nth(3), expected.get(pos + 3).copied());
    }
}

#[test]
fn nested_shuffles_do_not_reuse_the_inner_permutation() {
    // Two applications of the same two-element permutation always cancel. Fresh
    // layers should produce both orders, independently of the first layer's order.
    let mut pairs = [[0; 2]; 2];
    for seed in 0..256 {
        let once = Order::with_seed(Seq::source(2).shuffle(), seed).unwrap();
        let twice = Order::with_seed(Seq::source(2).shuffle().shuffle(), seed).unwrap();
        pairs[once.get(0).unwrap().record_index][twice.get(0).unwrap().record_index] += 1;
        check_access(&twice);
    }
    for count in pairs.into_iter().flatten() {
        assert!((32..=96).contains(&count), "correlated shuffle layers: {pairs:?}");
    }

    for seed in [0, 42, u64::MAX] {
        let once = Order::with_seed(Seq::source(37).shuffle(), seed).unwrap();
        let p = indices(&once);
        let squared: Vec<_> = p.iter().map(|&i| p[i]).collect();
        let twice = Order::with_seed(Seq::source(37).shuffle().shuffle(), seed).unwrap();
        let actual = indices(&twice);
        assert_ne!(actual, squared);
        assert_ne!(actual, p);
        let mut sorted = actual;
        sorted.sort_unstable();
        assert_eq!(sorted, (0..37).collect::<Vec<_>>());
        check_access(&twice);
    }
}

#[test]
fn one_pass_shuffles_are_interchangeable_inside_outer_shuffles() {
    for n in [2, 37] {
        for seed in [0, 42, u64::MAX] {
            let expected = Order::with_seed(Seq::source(n).shuffle().shuffle().repeat_shuffled(2), seed).unwrap();
            for inner in [Seq::source(n).shuffle(), Seq::source(n).repeat_shuffled(1), Seq::source(n).cycle_to_shuffled(n)] {
                // These wrappers can fold away; the shuffled layer must survive them.
                let inner = Seq::concat([inner.skip(0).take(n).step_by(1).repeat(1).cycle_to(n)]);
                for outer in [inner.clone().shuffle(), inner.clone().repeat_shuffled(1), inner.cycle_to_shuffled(n)] {
                    let order = Order::with_seed(outer.repeat_shuffled(2), seed).unwrap();
                    assert!(order.iter().eq(expected.iter()));
                    check_access(&order);
                }
            }
        }
    }
}

#[test]
fn repeating_a_selected_shuffle_keeps_the_selected_records() {
    for seed in [0, 1, 42, u64::MAX] {
        for times in [1, 3] {
            let selected = Seq::source(37).repeat_shuffled(times).take(2);
            let input = Order::with_seed(selected.clone(), seed).unwrap();
            let mut expected = indices(&input);
            expected.sort_unstable();
            let order = Order::with_seed(selected.repeat_shuffled(2), seed).unwrap();
            assert_eq!(order.len(), 4);
            for pass in indices(&order).as_chunks::<2>().0 {
                let mut actual = pass.to_vec();
                actual.sort_unstable();
                assert_eq!(actual, expected);
            }
            check_access(&order);
        }
    }
}

#[test]
fn plain_repeats_preserve_even_nested_shuffled_inputs() {
    for base in [
        Seq::source(37).shuffle(),
        Seq::source(37).repeat_shuffled(3),
        Seq::source(37).cycle_to_shuffled(83),
        Seq::source(37).repeat_shuffled(3).skip(17).take(70).shuffle(),
        Seq::mix([Seq::source(37).repeat_shuffled(3), Seq::source(19).cycle_to_shuffled(45)]),
    ] {
        for seed in [0, 42, u64::MAX] {
            let once = Order::with_seed(base.clone(), seed).unwrap();
            let records: Vec<_> = once.iter().map(|item| (item.source_ordinal, item.record_index)).collect();
            for repeated in [base.clone().repeat(3), base.clone().cycle_to(once.len() * 2 + 1)] {
                let order = Order::with_seed(repeated, seed).unwrap();
                assert!(
                    order.iter().map(|item| (item.source_ordinal, item.record_index)).eq(records.iter().copied().cycle().take(order.len()))
                );
                check_access(&order);
            }
        }
    }
}

#[test]
fn shuffled_passes_are_permutations_and_cycles_preserve_their_prefix() {
    let n = 37;
    for seed in [0, 42, u64::MAX] {
        let base = Seq::source(n);
        let order = Order::with_seed(base.clone().repeat_shuffled(3), seed).unwrap();
        let all = indices(&order);
        for pass in all.chunks_exact(n) {
            let mut sorted = pass.to_vec();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..n).collect::<Vec<_>>());
            assert_ne!(pass, sorted);
        }
        assert_ne!(all[..n], all[n..2 * n]);
        assert_ne!(all[n..2 * n], all[2 * n..]);
        check_access(&order);
        for len in [1, n - 1, n, n + 1, 2 * n, 2 * n + 7, 3 * n] {
            let cycle = Order::with_seed(base.clone().cycle_to_shuffled(len), seed).unwrap();
            assert!(cycle.iter().eq(order.cursor(..len).unwrap()));
            check_access(&cycle);
        }
        let once = Order::with_seed(base.clone().repeat_shuffled(1), seed).unwrap();
        assert!(once.iter().eq(order.cursor(..n).unwrap()));
        let mut reseeded = Order::new(base.repeat_shuffled(3)).unwrap();
        reseeded.set_seed(seed);
        assert!(reseeded.iter().eq(order.iter()));
    }
    assert_ne!(
        indices(&Order::new(Seq::source(n).repeat_shuffled(3)).unwrap()),
        indices(&Order::with_seed(Seq::source(n).repeat_shuffled(3), 42).unwrap())
    );
}

#[test]
fn shuffled_repetition_preserves_the_immediate_input_multiset() {
    for base in [Seq::source(37).shuffle().take(19), Seq::source(37).shuffle().repeat_shuffled(2), Seq::source(37).cycle_to_shuffled(51)] {
        let input = Order::with_seed(base.clone(), 42).unwrap();
        let n = input.len();
        let actual = Order::with_seed(base.repeat_shuffled(3), 42).unwrap();
        let mut expected = indices(&input);
        expected.sort_unstable();
        for pass in 0..3 {
            let mut got: Vec<_> = actual.cursor(pass * n..(pass + 1) * n).unwrap().map(|item| item.record_index).collect();
            got.sort_unstable();
            assert_eq!(got, expected);
        }
        check_access(&actual);
    }
}

#[test]
fn shuffled_repetition_validates_original_inputs_and_limits() {
    for seq in [Seq::source(0).repeat_shuffled(3), Seq::source(10).repeat_shuffled(0), Seq::source(0).cycle_to_shuffled(0)] {
        assert!(Order::new(seq).unwrap().is_empty());
    }
    for seq in [Seq::source(0).cycle_to_shuffled(1), Seq::source(0).cycle_to_shuffled(usize::MAX)] {
        assert_eq!(Order::new(seq).unwrap_err().kind(), &ErrorKind::EmptyCycle);
    }
    let error = Order::new(Seq::source(2).repeat_shuffled(usize::MAX)).unwrap_err();
    assert_eq!(error.kind(), &ErrorKind::LengthOverflow);
    assert!(error.path().is_empty());
    for mix in [Seq::mix([Seq::source(10)]), Seq::mix([] as [Seq<usize>; 0]), Seq::mix([Seq::source(10)]).take(0)] {
        for seq in
            [mix.clone().repeat_shuffled(0), mix.clone().repeat_shuffled(1), mix.clone().cycle_to_shuffled(0), mix.cycle_to_shuffled(3)]
        {
            let err = Order::new(Seq::concat([Seq::source(1), seq])).unwrap_err();
            assert_eq!(err.kind(), &ErrorKind::ShuffleContainsMix);
            assert_eq!(err.path(), [1]);
        }
    }
    for seq in [Seq::source(3).take(4).repeat_shuffled(0), Seq::source(3).take(4).cycle_to_shuffled(0)] {
        let err = Order::new(seq).unwrap_err();
        assert_eq!(err.kind(), &ErrorKind::TakeOutOfRange { n: 4, len: 3 });
        assert_eq!(err.path(), [0]);
    }
    for seq in [Seq::source(1).repeat_shuffled(usize::MAX), Seq::source(3).cycle_to_shuffled(usize::MAX)] {
        let order = Order::new(seq).unwrap();
        let mut cursor = order.cursor(usize::MAX - 8..).unwrap();
        assert_eq!(cursor.nth(6), order.get(usize::MAX - 2));
        assert_eq!(cursor.next(), order.get(usize::MAX - 1));
        assert_eq!(cursor.next(), None);
        cursor.reset(..).unwrap();
        assert_eq!(cursor.nth(usize::MAX - 1), order.get(usize::MAX - 1));
    }
}

#[test]
fn mapping_and_serialization_preserve_shuffled_variants() {
    let seq = Seq::concat([Seq::source("37").repeat_shuffled(3), Seq::source("19").cycle_to_shuffled(45)]);
    let parsed = seq.clone().try_map_sources(str::parse::<usize>).unwrap();
    assert_eq!(parsed, Seq::concat([Seq::source(37).repeat_shuffled(3), Seq::source(19).cycle_to_shuffled(45)]));
    assert_eq!(seq.map_sources(str::to_owned).map_sources(|s| s.parse::<usize>().unwrap()), parsed);
    #[cfg(feature = "serde")]
    for (seq, json) in [
        (Seq::source(37).repeat_shuffled(3), r#"{"Repeat":{"times":3,"shuffled":true,"inner":{"Source":37}}}"#),
        (Seq::source(19).cycle_to_shuffled(45), r#"{"Cycle":{"len":45,"shuffled":true,"inner":{"Source":19}}}"#),
    ] {
        assert_eq!(serde_json::to_string(&seq).unwrap(), json);
        let back: Seq<usize> = serde_json::from_str(json).unwrap();
        assert_eq!(seq, back);
        assert!(Order::new(seq).unwrap().iter().eq(Order::new(back).unwrap().iter()));
    }
}
