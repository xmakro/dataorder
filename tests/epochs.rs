//! Epoch numbering belongs to each input stream, independently of its surrounding mix.

use dataorder::{ErrorKind, Order, Schedule, Seq, Source};

#[derive(Clone, Debug, PartialEq)]
struct Dataset {
    id: u64,
    len: usize,
}

impl Source for Dataset {
    fn len(&self) -> usize {
        self.len
    }

    fn salt(&self) -> u64 {
        self.id
    }
}

fn source(id: u64, len: usize) -> Seq<Dataset> {
    Seq::source(Dataset { id, len }).shuffle(7)
}

fn records(order: &Order<Dataset>) -> Vec<(u64, usize, usize)> {
    order.iter().map(|item| (item.source.id, item.record_index, item.epoch)).collect()
}

fn stream(records: &[(u64, usize, usize)], id: u64) -> Vec<(usize, usize)> {
    records.iter().filter_map(|&(s, i, epoch)| (s == id).then_some((i, epoch))).collect()
}

fn check_access(order: &Order<Dataset>, expected: &[(u64, usize, usize)]) {
    assert_eq!(records(order), expected);
    let mut cursor = order.iter();
    for pos in [expected.len() - 1, 0, expected.len() / 2, 1] {
        cursor.reset(pos..).unwrap();
        assert_eq!(cursor.clone().next(), order.get(pos));
        assert_eq!(cursor.nth(3), order.get(pos + 3));
        for (offset, item) in cursor.by_ref().take(5).enumerate() {
            assert_eq!((item.source.id, item.record_index, item.epoch), expected[pos + 4 + offset]);
        }
    }
    for (pos, &record) in expected.iter().enumerate() {
        let item = order.get(pos).unwrap();
        assert_eq!((item.source.id, item.record_index, item.epoch), record);
    }
}

#[test]
fn nested_repeats_match_flat_repeats() {
    for base in [
        source(0, 37),
        Seq::concat([source(0, 37), source(1, 19).repeat(2)]).shuffle(11),
        Seq::mix([(source(0, 37).repeat(2), Schedule::Uniform), (source(1, 19), Schedule::delayed(0.3))]),
    ] {
        for seed in [0, 51, u64::MAX] {
            let flat = Order::with_seed(base.clone().repeat(24), seed).unwrap();
            let expected = records(&flat);
            for nested in [base.clone().repeat(3).repeat(8), base.clone().repeat(3).repeat(2).repeat(4)] {
                check_access(&Order::with_seed(nested, seed).unwrap(), &expected);
            }
        }
    }
}

#[test]
fn mixture_edits_preserve_existing_streams_in_every_epoch() {
    let a = source(0, 17).repeat(2).repeat(3);
    let b = source(1, 11).cycle_to(29);
    let deep = source(2, 7).repeat(2).repeat(3).repeat(2).repeat(2);
    let variants = [
        Seq::mix([a.clone(), b.clone()]),
        Seq::mix([deep.clone(), b.clone(), a.clone()]),
        Seq::mix([Seq::mix([a.clone(), deep.clone()]), b.clone()]),
        Seq::mix([Seq::mix([a.clone(), b.clone()]), deep.clone()]),
        Seq::mix([(b.clone(), Schedule::ramp(0.1, 0.8)), (a.clone(), Schedule::delayed(0.2)), (deep, Schedule::Uniform)]),
        Seq::mix([source(2, 7).repeat(2).repeat(3).take(0), a.clone(), b.clone()]),
    ];
    for seed in [0, 51, u64::MAX] {
        let expected_a = stream(&records(&Order::with_seed(a.clone().repeat(3), seed).unwrap()), 0);
        let expected_b = stream(&records(&Order::with_seed(b.clone().repeat(3), seed).unwrap()), 1);
        for seq in &variants {
            let order = Order::with_seed(seq.clone().repeat(3), seed).unwrap();
            let actual = records(&order);
            assert_eq!(stream(&actual, 0), expected_a);
            assert_eq!(stream(&actual, 1), expected_b);
            check_access(&order, &actual);
        }
    }
}

#[test]
fn changing_the_mix_mid_training_continues_each_stream() {
    let a = source(0, 17).repeat(3).repeat(2);
    let b = source(1, 11).repeat(2).cycle_to(83);
    let old_mix = Seq::mix([a.clone(), b.clone()]);
    for seed in [0, 51] {
        let old = Order::with_seed(old_mix.clone(), seed).unwrap();
        let original = records(&old);
        let consumed = old.len() / 2;
        let a_used = stream(&original[..consumed], 0).len();
        let b_used = stream(&original[..consumed], 1).len();
        let new_data = source(2, 13).repeat(2).repeat(2);

        let changed = Seq::mix([
            (a.clone().skip(a_used), Schedule::Uniform),
            (Seq::mix([new_data.clone(), b.clone().skip(b_used)]), Schedule::delayed(0.2)),
        ]);
        let order = Order::with_seed(changed, seed).unwrap();
        let remaining = records(&order);
        assert_eq!(stream(&remaining, 0), stream(&original[consumed..], 0));
        assert_eq!(stream(&remaining, 1), stream(&original[consumed..], 1));
        check_access(&order, &remaining);

        // Keep the entire old continuation as one input to a new mixture.
        let changed = Seq::mix([old_mix.clone().skip(consumed), new_data]);
        let order = Order::with_seed(changed, seed).unwrap();
        let remaining = records(&order);
        let old_remaining: Vec<_> = remaining.iter().copied().filter(|&(id, _, _)| id != 2).collect();
        assert_eq!(old_remaining, original[consumed..]);
        check_access(&order, &remaining);
    }
}

#[test]
fn selections_preserve_original_repeat_counts() {
    let n = 17;
    let base = source(0, n);
    for seed in [0, 51] {
        let flat = records(&Order::with_seed(base.clone().repeat(9), seed).unwrap());
        let selections: Vec<(Seq<Dataset>, Vec<usize>)> = vec![
            (base.clone().repeat(3).take(1), vec![0]),
            (base.clone().repeat(3).take(n), (0..n).collect()),
            (base.clone().repeat(3).take(n + 1), (0..n + 1).collect()),
            (base.clone().repeat(3).skip(n - 1).take(n + 2), (n - 1..2 * n + 1).collect()),
            (base.clone().repeat(3).step_by(3 * n), vec![0]),
            (base.clone().repeat(3).step_by(4).skip(2).take(7), (2..9).map(|i| i * 4).collect()),
            (base.clone().cycle_to(2 * n + 7), (0..2 * n + 7).collect()),
            (Seq::concat([base.clone().repeat(3), source(1, 13)]).take(n), (0..n).collect()),
        ];
        for (selected, positions) in selections {
            let expected: Vec<_> = flat.chunks_exact(3 * n).flat_map(|epoch| positions.iter().map(move |&p| epoch[p])).collect();
            let order = Order::with_seed(selected.repeat(3), seed).unwrap();
            check_access(&order, &expected);
        }
    }
}

#[test]
fn drawn_items_report_source_epochs_with_or_without_shuffling() {
    for shuffled in [false, true] {
        let base = Seq::source(Dataset { id: 0, len: 17 });
        let base = if shuffled { base.shuffle(7) } else { base };
        let order = Order::with_seed(base.repeat(3).repeat(2), 51).unwrap();
        for (pos, item) in order.iter().enumerate() {
            assert_eq!(item.epoch, pos / 17);
            assert_eq!(order.get(pos), Some(item));
        }
    }

    // Without repeats, including shuffled and selected sources, the epoch is zero.
    let order = Order::with_seed(source(0, 17).skip(2).step_by(3), 51).unwrap();
    assert!(order.iter().all(|item| item.epoch == 0));

    // Shuffling the repeated sequence reorders its epochs along with its records.
    let order = Order::new(Seq::source(Dataset { id: 0, len: 17 }).repeat(3).shuffle(7)).unwrap();
    let drawn = records(&order);
    assert!(drawn.windows(2).any(|pair| pair[0].2 > pair[1].2));
    let mut actual = drawn.clone();
    actual.sort_unstable();
    let expected: Vec<_> = (0..17).flat_map(|i| (0..3).map(move |epoch| (0, i, epoch))).collect();
    assert_eq!(actual, expected);
    check_access(&order, &drawn);

    // Identical records from different epochs are distinct items.
    let repeated = Order::new(Seq::source(1).repeat(2)).unwrap();
    let first = repeated.get(0).unwrap();
    let second = repeated.get(1).unwrap();
    assert_eq!(first.record_index, second.record_index);
    assert_ne!(first, second);
}

#[test]
fn sparse_epochs_use_usize_and_overflow_is_rejected_at_compilation() {
    let sparse = Seq::source(1).repeat(100).take(1).repeat(2);
    let order = Order::new(sparse).unwrap();
    let epochs: Vec<usize> = order.iter().map(|item| item.epoch).collect();
    assert_eq!(epochs, [0, 100]);

    // A full usize-sized sequence still supports the last representable position.
    let huge = Order::new(Seq::source(1).repeat(usize::MAX)).unwrap();
    let last = huge.get(usize::MAX - 1).unwrap();
    assert_eq!(last.epoch, usize::MAX - 1);
    assert_eq!(huge.cursor(usize::MAX - 1..).unwrap().next(), Some(last));
    let times = usize::MAX / 2;
    let sparse = Order::new(Seq::source(1).repeat(times).take(1).repeat(2)).unwrap();
    assert_eq!(sparse.get(1).unwrap().epoch, times);

    let huge = || Seq::source(1).repeat(usize::MAX);
    for selected in [huge().take(1), huge().skip(usize::MAX - 1), huge().step_by(usize::MAX)] {
        for invalid in [selected.clone().repeat(2), selected.cycle_to(2)] {
            let err = Order::new(invalid.clone()).unwrap_err();
            assert_eq!(err.kind(), &ErrorKind::EpochOverflow);
            assert!(err.path().is_empty());
            assert_eq!(err.to_string(), "nested repeat counts exceed usize::MAX (at the root)");
            let err = Order::new(Seq::concat([Seq::source(1), invalid.take(0)])).unwrap_err();
            assert_eq!(err.kind(), &ErrorKind::EpochOverflow);
            assert_eq!(err.path(), [1, 0]);
        }
    }
    // Empty outputs have no epochs, while sibling inputs do not multiply each other.
    assert!(Order::new(huge().take(0).repeat(usize::MAX)).unwrap().is_empty());
    let parts = [huge().take(1), huge().take(1)];
    assert!(Order::new(Seq::mix(parts.clone())).unwrap().iter().all(|item| item.epoch == 0));
    assert!(Order::new(Seq::concat(parts)).unwrap().iter().all(|item| item.epoch == 0));
}
