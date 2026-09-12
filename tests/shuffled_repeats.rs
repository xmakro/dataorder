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
fn repeating_a_selected_shuffle_keeps_the_selected_records() {
    for seed in [0, 1, 42, u64::MAX] {
        for times in [1, 3] {
            let selected = Seq::source(37).shuffled_repeat(times).take(2);
            let input = Order::with_seed(selected.clone(), seed).unwrap();
            let mut expected = indices(&input);
            expected.sort_unstable();
            let order = Order::with_seed(selected.shuffled_repeat(2), seed).unwrap();
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
        Seq::source(37).shuffle(7),
        Seq::source(37).shuffled_repeat(3),
        Seq::source(37).shuffled_cycle_to(83),
        Seq::source(37).shuffled_repeat(3).skip(17).take(70).shuffle(11),
        Seq::mix([Seq::source(37).shuffled_repeat(3), Seq::source(19).shuffled_cycle_to(45)]),
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
        let order = Order::with_seed(base.clone().shuffled_repeat(3), seed).unwrap();
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
            let cycle = Order::with_seed(base.clone().shuffled_cycle_to(len), seed).unwrap();
            assert!(cycle.iter().eq(order.cursor(..len).unwrap()));
            check_access(&cycle);
        }
        let once = Order::with_seed(base.clone().shuffled_repeat(1), seed).unwrap();
        assert!(once.iter().eq(order.cursor(..n).unwrap()));
        let mut reseeded = Order::new(base.shuffled_repeat(3)).unwrap();
        reseeded.set_seed(seed);
        assert!(reseeded.iter().eq(order.iter()));
    }
    assert_ne!(
        indices(&Order::new(Seq::source(n).shuffled_repeat(3)).unwrap()),
        indices(&Order::with_seed(Seq::source(n).shuffled_repeat(3), 42).unwrap())
    );
}

#[test]
fn shuffled_repetition_preserves_the_immediate_input_multiset() {
    for base in [Seq::source(37).shuffle(7).take(19), Seq::source(37).shuffle(7).shuffled_repeat(2), Seq::source(37).shuffled_cycle_to(51)]
    {
        let input = Order::with_seed(base.clone(), 42).unwrap();
        let n = input.len();
        let actual = Order::with_seed(base.shuffled_repeat(3), 42).unwrap();
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
    for seq in [Seq::source(0).shuffled_repeat(3), Seq::source(10).shuffled_repeat(0), Seq::source(0).shuffled_cycle_to(0)] {
        assert!(Order::new(seq).unwrap().is_empty());
    }
    for seq in [Seq::source(0).shuffled_cycle_to(1), Seq::source(0).shuffled_cycle_to(usize::MAX)] {
        assert_eq!(Order::new(seq).unwrap_err().kind(), &ErrorKind::EmptyCycle);
    }
    let error = Order::new(Seq::source(2).shuffled_repeat(usize::MAX)).unwrap_err();
    assert_eq!(error.kind(), &ErrorKind::LengthOverflow);
    assert!(error.path().is_empty());
    for mix in [Seq::mix([Seq::source(10)]), Seq::mix([] as [Seq<usize>; 0]), Seq::mix([Seq::source(10)]).take(0)] {
        for seq in
            [mix.clone().shuffled_repeat(0), mix.clone().shuffled_repeat(1), mix.clone().shuffled_cycle_to(0), mix.shuffled_cycle_to(3)]
        {
            let err = Order::new(Seq::concat([Seq::source(1), seq])).unwrap_err();
            assert_eq!(err.kind(), &ErrorKind::ShuffleContainsMix);
            assert_eq!(err.path(), [1]);
        }
    }
    for seq in [Seq::source(3).take(4).shuffled_repeat(0), Seq::source(3).take(4).shuffled_cycle_to(0)] {
        let err = Order::new(seq).unwrap_err();
        assert_eq!(err.kind(), &ErrorKind::TakeOutOfRange { n: 4, len: 3 });
        assert_eq!(err.path(), [0]);
    }
    for seq in [Seq::source(1).shuffled_repeat(usize::MAX), Seq::source(3).shuffled_cycle_to(usize::MAX)] {
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
    let seq = Seq::concat([Seq::source("37").shuffled_repeat(3), Seq::source("19").shuffled_cycle_to(45)]);
    let parsed = seq.clone().try_map_sources(str::parse::<usize>).unwrap();
    assert_eq!(parsed, Seq::concat([Seq::source(37).shuffled_repeat(3), Seq::source(19).shuffled_cycle_to(45)]));
    assert_eq!(seq.map_sources(str::to_owned).map_sources(|s| s.parse::<usize>().unwrap()), parsed);
    #[cfg(feature = "serde")]
    for (seq, json) in [
        (Seq::source(37).shuffled_repeat(3), r#"{"ShuffledRepeat":{"times":3,"inner":{"Source":37}}}"#),
        (Seq::source(19).shuffled_cycle_to(45), r#"{"ShuffledCycle":{"len":45,"inner":{"Source":19}}}"#),
    ] {
        assert_eq!(serde_json::to_string(&seq).unwrap(), json);
        let back: Seq<usize> = serde_json::from_str(json).unwrap();
        assert_eq!(seq, back);
        assert!(Order::new(seq).unwrap().iter().eq(Order::new(back).unwrap().iter()));
    }
}
