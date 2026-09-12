//! Cursor behavior: a stateful property test against random access, then targeted cases
//! for resets, empty ranges, boundaries, clones and failed operations. Reproduce a property
//! failure with DATAORDER_STATE_SEED=<decimal seed>; failure histories are minimized by
//! deleting operations before being reported.
use dataorder::{BoundsError, Order, Schedule, Seq};
use std::ops::Bound;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        self.next() as usize % n
    }
}

fn configuration(r: &mut Rng, depth: usize, allow_mix: bool) -> (Seq<usize>, usize) {
    if depth == 0 || r.below(5) == 0 {
        let n = match r.below(5) {
            0 => usize::MAX,
            1 => usize::MAX - r.below(10),
            2 => 1usize << r.below(usize::BITS as usize),
            _ => r.below(100),
        };
        return (Seq::source(n), n);
    }
    let kind = r.below(11);
    let (s, n) = configuration(r, depth - 1, allow_mix && !matches!(kind, 0 | 9 | 10));
    match kind {
        7 | 8 if !allow_mix => (s, n),
        0 => (s.shuffle(), n),
        9 => {
            let k = r.below(5);
            if let Some(len) = n.checked_mul(k) { (s.repeat_shuffled(k), len) } else { (s, n) }
        }
        10 => {
            let k = if r.below(3) == 0 { usize::MAX } else { r.below(100) };
            if n > 0 { (s.cycle_to_shuffled(k), k) } else { (s, n) }
        }
        1 => {
            let k = r.below(5);
            if let Some(len) = n.checked_mul(k) { (s.repeat(k), len) } else { (s, n) }
        }
        2 => {
            let k = if r.below(3) == 0 { usize::MAX } else { r.below(100) };
            if n > 0 { (s.cycle_to(k), k) } else { (s, n) }
        }
        3 => {
            let k = if n == 0 { 0 } else { r.below(n) };
            (s.skip(k), n - k)
        }
        4 => {
            let k = if n == 0 { 0 } else { r.below(n) };
            (s.take(k), k)
        }
        5 => {
            let step = if r.below(2) == 0 { r.below(16) + 1 } else { (r.next() as usize).max(1) };
            let offset = if n == 0 { 0 } else { r.below(n) };
            let len = if offset >= n { 0 } else { (n - offset - 1) / step + 1 };
            (s.skip(offset).step_by(step), len)
        }
        6 => {
            let (t, m) = configuration(r, depth - 1, allow_mix);
            if let Some(len) = n.checked_add(m) { (Seq::concat([s, t]), len) } else { (s, n) }
        }
        7 => {
            let (t, m) = configuration(r, depth - 1, allow_mix);
            let a = n.min(100);
            let schedule = match r.below(5) {
                0 => Schedule::Uniform,
                1 => Schedule::until(0.25),
                2 => Schedule::ramp(0.25, 0.75),
                3 => Schedule::fade(0.25, 0.75),
                _ => Schedule::trapezoid(0.125, 0.25, 0.75, 0.875),
            };
            let b = m.min(if schedule == Schedule::Uniform { 100 } else { a / 3 });
            (Seq::mix([(s.take(a), Schedule::Uniform), (t.take(b), schedule)]), a + b)
        }
        _ => {
            if n == 0 {
                return (s, n);
            }
            let total = r.below(300);
            let first = total / 4;
            (Seq::mix([s.cycle_to(first), Seq::source(7).shuffle().cycle_to(total - first)]), total)
        }
    }
}

#[derive(Clone, Debug)]
enum Op {
    Next,
    Nth(usize),
    ResetFrom(usize),
    ResetRange(usize, usize),
    Clone,
    Last,
    Count,
    PastEnd,
    BadRange,
}

// Include the endpoint without overflowing when it is usize::MAX.
fn position(raw: usize, end: usize) -> usize {
    if raw == usize::MAX {
        end
    } else if end == usize::MAX {
        raw
    } else {
        raw % (end + 1)
    }
}

fn replay(order: &Order<usize>, ops: &[Op]) -> Result<(), String> {
    let mut cursor = order.iter();
    let (mut pos, mut end) = (0, order.len());
    for (step, op) in ops.iter().enumerate() {
        let fail = || format!("operation {step}: {op:?}, position={pos}, end={end}");
        match *op {
            Op::Next | Op::Nth(_) => {
                let n = if let Op::Nth(n) = *op { n } else { 0 };
                let expected = (n < end - pos).then(|| order.get(pos + n).unwrap());
                let actual = if matches!(op, Op::Next) { cursor.next() } else { cursor.nth(n) };
                if actual != expected {
                    return Err(format!("{}: {actual:?} != {expected:?}", fail()));
                }
                pos = if n < end - pos { pos + n + 1 } else { end };
            }
            Op::ResetFrom(raw) => {
                (pos, end) = (position(raw, order.len()), order.len());
                cursor.reset(pos..).unwrap();
            }
            Op::ResetRange(a, b) => {
                let (a, b) = (position(a, order.len()), position(b, order.len()));
                (pos, end) = (a.min(b), a.max(b));
                cursor.reset(pos..end).unwrap();
            }
            Op::Clone => cursor = cursor.clone(),
            Op::Last => {
                let expected = (pos < end).then(|| order.get(end - 1).unwrap());
                if cursor.clone().last() != expected {
                    return Err(fail());
                }
            }
            Op::Count => {
                if cursor.clone().count() != end - pos {
                    return Err(fail());
                }
            }
            Op::PastEnd => {
                if let Some(bad) = order.len().checked_add(1)
                    && cursor.reset(..bad).is_ok()
                {
                    return Err(fail());
                }
            }
            Op::BadRange => {
                use std::ops::Bound::{Excluded, Unbounded};
                if cursor.reset((Excluded(usize::MAX), Unbounded)).is_ok() {
                    return Err(fail());
                }
            }
        }
        if (cursor.offset(), cursor.len()) != (pos, end - pos) {
            return Err(format!("{op:?}: cursor state differs from model"));
        }
    }
    Ok(())
}

fn checked_replay(order: &Order<usize>, ops: &[Op]) -> Result<(), String> {
    std::panic::catch_unwind(|| replay(order, ops)).unwrap_or_else(|_| Err("cursor operation panicked".into()))
}

fn minimize(mut ops: Vec<Op>, fails: impl Fn(&[Op]) -> bool) -> Vec<Op> {
    let mut width = ops.len().div_ceil(2);
    while width > 0 {
        let mut start = 0;
        while start < ops.len() {
            let mut candidate = ops.clone();
            candidate.drain(start..(start + width).min(candidate.len()));
            if fails(&candidate) {
                ops = candidate;
            } else {
                start += width;
            }
        }
        width /= 2;
    }
    ops
}

#[test]
fn operation_sequences_match_random_access() {
    let explicit = std::env::var("DATAORDER_STATE_SEED").ok().map(|s| s.parse::<u64>().expect("decimal seed"));
    let mut seeds = Rng(0xcafe_ba5e_dead_beef);
    for _ in 0..if explicit.is_some() { 1 } else { 2000 } {
        let seed = explicit.unwrap_or_else(|| seeds.next());
        let mut r = Rng(seed);
        let (seq, len) = configuration(&mut r, 7, true);
        let order = Order::with_seed(seq.clone(), seed).unwrap_or_else(|err| panic!("{err}: {seq:?}"));
        assert_eq!(order.len(), len, "seed={seed}, seq={seq:?}");
        let ops: Vec<_> = (0..100)
            .map(|_| {
                let raw = match r.below(4) {
                    0 => 0,
                    1 => usize::MAX,
                    2 => r.below(16),
                    _ => r.next() as usize,
                };
                match r.below(10) {
                    0 | 1 => Op::Next,
                    2 => Op::Nth(raw),
                    3 => Op::ResetFrom(raw),
                    4 => Op::ResetRange(raw, r.next() as usize),
                    5 => Op::Clone,
                    6 => Op::Last,
                    7 => Op::Count,
                    8 => Op::PastEnd,
                    _ => Op::BadRange,
                }
            })
            .collect();
        if let Err(error) = checked_replay(&order, &ops) {
            let reduced = minimize(ops, |ops| checked_replay(&order, ops).is_err());
            panic!("DATAORDER_STATE_SEED={seed}: {error}\nconfiguration={seq:?}\nminimized operations={reduced:?}");
        }
    }
}

#[test]
fn failure_history_shrinking_keeps_only_relevant_operations() {
    let result =
        minimize(vec![Op::Next, Op::Clone, Op::Next, Op::PastEnd, Op::Count], |ops| ops.iter().any(|op| matches!(op, Op::PastEnd)));
    assert!(matches!(result.as_slice(), [Op::PastEnd]));
}

#[test]
fn empty_ranges_resume_after_resets_skips_and_clones() {
    let mix = || Seq::mix([Seq::source(100).shuffle(), Seq::source(50).shuffle()]);
    let sequences = [
        Seq::source(0),
        Seq::source(100),
        Seq::source(100).shuffle(),
        mix(),
        mix().repeat(3).skip(2).step_by(7),
        Seq::concat([mix(), Seq::concat([Seq::source(100), Seq::source(50)]).shuffle()]).skip(20),
    ];
    for seq in sequences {
        let order = Order::with_seed(seq, 19).unwrap();
        for start in [0, order.len() / 2, order.len()] {
            let mut cursor = order.cursor(start..start).unwrap();
            assert_eq!(cursor.clone().count(), 0);
            assert_eq!(cursor.clone().last(), None);
            assert_eq!(cursor.next(), None);
            assert_eq!(cursor.nth(usize::MAX), None);
            cursor.reset(0..0).unwrap();
            cursor.reset(..).unwrap();
            assert_eq!(cursor.clone().count(), order.len());
            assert_eq!(cursor.clone().last(), order.len().checked_sub(1).and_then(|pos| order.get(pos)));
            let pos = order.len() / 3;
            cursor.reset(pos..).unwrap();
            let mut cloned = cursor.clone();
            assert_eq!(cursor.nth(1), order.get(pos + 1));
            assert_eq!(cloned.nth(1), order.get(pos + 1));
            cursor.reset(order.len()..).unwrap();
            assert_eq!(cursor.next(), None);
            let mut cloned = cursor.clone();
            for c in [&mut cursor, &mut cloned] {
                c.reset(..).unwrap();
                assert_eq!(c.next(), order.get(0));
            }
        }
    }
}

#[test]
fn exhausted_subranges_resume_at_their_end() {
    let seq = Seq::concat([
        Seq::mix([Seq::source(5), Seq::source(5)]),
        Seq::mix([Seq::source(7), Seq::source(7), Seq::source(7)]),
        Seq::source(3),
    ]);
    for seq in [seq.clone(), seq.clone().repeat(3), seq.skip(2).step_by(3)] {
        let order = Order::new(seq).unwrap();
        for end in 0..=order.len() {
            for exhaust in 0..3 {
                let mut cursor = order.cursor(..end).unwrap();
                assert_eq!(cursor.clone().last(), end.checked_sub(1).and_then(|pos| order.get(pos)));
                match exhaust {
                    0 => cursor.by_ref().for_each(drop),
                    1 => assert_eq!(cursor.nth(usize::MAX), None),
                    _ => cursor.reset(end..end).unwrap(),
                }
                assert_eq!((cursor.offset(), cursor.len()), (end, 0));
                assert_eq!(cursor.clone().last(), None);
                // Extending the range at the same position must find the next element,
                // including at concat/repeat boundaries and after the order's end.
                cursor.reset(end..).unwrap();
                assert_eq!(cursor.next(), order.get(end));
                cursor.reset(0..).unwrap();
                assert_eq!(cursor.next(), order.get(0));
            }
        }
    }
}

#[test]
fn boundary_skip_initializes_target_child_before_backward_seek() {
    let order = Order::new(Seq::concat([
        Seq::mix([Seq::source(5), Seq::source(5)]),
        Seq::mix([Seq::source(7), Seq::source(7), Seq::source(7)]),
        Seq::source(3),
    ]))
    .unwrap();
    let mut cursor = order.iter();
    cursor.next(); // Initialize buffers for the first child.
    cursor.reset(31..).unwrap(); // End of the second child; its state has not been built.
    cursor.reset(10..).unwrap(); // Same child index, but its state must now be initialized.
    assert_eq!(cursor.next(), Some(order.get(10).unwrap()));
}

#[test]
fn concat_children_keep_their_own_transform_parameters() {
    let source = |n| Seq::source(n).shuffle();
    let nested = |n| {
        Seq::mix([
            source(n).repeat(3).skip(2).skip(1).step_by(3),
            source(n + 2).repeat(2).take(n + 3),
            Seq::concat([source(n), Seq::source(n + 3)]).skip(1).step_by(2),
            Seq::concat([source(n), source(n + 1)]).shuffle(),
        ])
    };
    let seq = Seq::concat([nested(11), nested(19), Seq::source(3), nested(7)]).repeat(3);
    let order = Order::with_seed(seq, 42).unwrap();
    let expected: Vec<_> = (0..order.len()).map(|pos| order.get(pos).unwrap()).collect();
    assert_eq!(order.iter().collect::<Vec<_>>(), expected);
    let mut cursor = order.iter();
    for pos in (0..order.len()).rev().step_by(3).chain((0..order.len()).step_by(7)) {
        cursor.reset(pos..).unwrap();
        let mut copy = cursor.clone();
        for expected in &expected[pos..(pos + 5).min(order.len())] {
            assert_eq!(cursor.next(), Some(*expected));
            assert_eq!(copy.next(), Some(*expected));
        }
    }
}

#[test]
fn selections_resume_after_skips_and_exhaustion() {
    // Keep the slice around a repeat so the contiguous selection's cursor is exercised.
    let seq = Seq::source(7).shuffle().repeat(5);
    let base = Order::new(seq.clone()).unwrap();
    let all: Vec<_> = base.iter().map(|item| item.record_index).collect();
    for step in [1, 2, 8, usize::MAX] {
        let order = Order::new(seq.clone().skip(5).take(23).step_by(step)).unwrap();
        let expected: Vec<_> = all[5..28].iter().copied().step_by(step).collect();
        for n in 0..=expected.len() {
            let mut cursor = order.iter();
            assert_eq!(cursor.nth(n).map(|item| item.record_index), expected.get(n).copied());
            cursor.reset(0..).unwrap();
            assert_eq!(cursor.by_ref().map(|item| item.record_index).collect::<Vec<_>>(), expected);
            cursor.reset(order.len()..).unwrap();
            assert_eq!(cursor.next(), None);
            cursor.reset(..).unwrap();
            let mut copy = cursor.clone();
            assert_eq!(copy.by_ref().map(|item| item.record_index).collect::<Vec<_>>(), expected);
            assert_eq!(cursor.by_ref().map(|item| item.record_index).collect::<Vec<_>>(), expected);
        }
    }
}

#[test]
fn standard_iterator_position_is_available() {
    let order = Order::new(Seq::source(10)).unwrap();
    let mut cursor = order.iter();
    assert_eq!(cursor.position(|item| item.record_index == 3), Some(3));
    assert_eq!(cursor.offset(), 4);
    assert_eq!(cursor.position(|item| item.record_index == 6), Some(2));
    assert_eq!(cursor.offset(), 7);
}

#[test]
#[allow(clippy::reversed_empty_ranges)]
fn checked_access_preserves_cursor_on_errors() {
    let order = Order::new(Seq::mix([Seq::source(10).shuffle(), Seq::source(7)])).unwrap();
    assert_eq!(order.get(16), (&order).into_iter().nth(16));
    assert_eq!(order.get(17), None);
    assert_eq!(order.get(usize::MAX), None);
    let empty = Order::new(Seq::source(0)).unwrap();
    assert_eq!(empty.get(0), None);
    assert_eq!((&empty).into_iter().next(), None);
    assert_eq!(order.cursor(..=usize::MAX).unwrap_err(), BoundsError::Overflow);
    assert_eq!(order.cursor((Bound::Excluded(usize::MAX), Bound::Unbounded)).unwrap_err(), BoundsError::Overflow);
    assert_eq!(order.cursor(10..9).unwrap_err(), BoundsError::Reversed { start: 10, end: 9 });
    assert_eq!(order.cursor(..18).unwrap_err(), BoundsError::OutOfBounds { end: 18, len: 17 });
    // A start past the end is out of bounds, not reversed, even with an unbounded end.
    assert_eq!(order.cursor(18..).unwrap_err(), BoundsError::StartOutOfBounds { start: 18, len: 17 });
    assert_eq!(order.cursor(18..20).unwrap_err(), BoundsError::StartOutOfBounds { start: 18, len: 17 });
    assert_eq!(order.cursor(18..).unwrap_err().to_string(), "range start 18 out of range for 17 positions");
    assert_eq!(order.cursor(17..).unwrap().count(), 0);
    let mut c = order.cursor(3..10).unwrap();
    c.next(); // Initialize the cursor before testing rollback.
    let expected = c.clone().collect::<Vec<_>>();
    assert_eq!(c.reset(..=usize::MAX), Err(BoundsError::Overflow));
    assert_eq!(c.offset(), 4);
    assert_eq!(c.clone().collect::<Vec<_>>(), expected);
    assert_eq!(c.reset((Bound::Excluded(usize::MAX), Bound::Unbounded)), Err(BoundsError::Overflow));
    assert_eq!(c.offset(), 4);
    assert_eq!(c.clone().collect::<Vec<_>>(), expected);
    assert_eq!(c.reset(2..18), Err(BoundsError::OutOfBounds { end: 18, len: 17 }));
    assert_eq!(c.offset(), 4);
    assert_eq!(c.clone().collect::<Vec<_>>(), expected);
    assert_eq!(c.reset(12..4), Err(BoundsError::Reversed { start: 12, end: 4 }));
    assert_eq!(c.offset(), 4);
    assert_eq!(c.clone().collect::<Vec<_>>(), expected);
    assert_eq!(c.reset(18..), Err(BoundsError::StartOutOfBounds { start: 18, len: 17 }));
    assert_eq!(c.offset(), 4);
    assert_eq!(c.clone().collect::<Vec<_>>(), expected);
    c.reset(17..17).unwrap();
    assert_eq!(c.next(), None);
    c.reset(..).unwrap();
    assert_eq!(c.next(), order.get(0));
    let huge = Order::new(Seq::source(usize::MAX)).unwrap();
    assert_eq!(huge.cursor(usize::MAX - 1..).unwrap().next().unwrap().record_index, usize::MAX - 1);
    assert_eq!(huge.cursor(usize::MAX..).unwrap().count(), 0);
}
