//! Stateful cursor properties: reproduce with DATAORDER_STATE_SEED=<decimal seed>.
//! Failure histories are minimized by deleting operations before being reported.
use dataorder::{Order, Sampling, Seq};

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

fn configuration(r: &mut Rng, depth: usize) -> (Seq<usize>, usize) {
    if depth == 0 || r.below(5) == 0 {
        let n = match r.below(5) {
            0 => usize::MAX,
            1 => usize::MAX - r.below(10),
            2 => 1usize << r.below(usize::BITS as usize),
            _ => r.below(100),
        };
        return (Seq::source(n), n);
    }
    let (s, n) = configuration(r, depth - 1);
    match r.below(9) {
        0 => (s.shuffle(r.next()), n),
        1 => {
            let k = r.below(5);
            if let Some(len) = n.checked_mul(k) { (s.repeat(k), len) } else { (s, n) }
        }
        2 => {
            let k = if r.below(3) == 0 { usize::MAX } else { r.below(100) };
            if n > 0 { (s.cycle(k), k) } else { (s, n) }
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
            let (t, m) = configuration(r, depth - 1);
            if let Some(len) = n.checked_add(m) { (Seq::concat([s, t]), len) } else { (s, n) }
        }
        7 => {
            let (t, m) = configuration(r, depth - 1);
            let a = n.min(100);
            let sampling = match r.below(5) {
                0 => Sampling::Uniform,
                1 => Sampling::until(0.25),
                2 => Sampling::ramp(0.25, 0.75),
                3 => Sampling::fading(0.25, 0.75),
                _ => Sampling::trapezoid(0.125, 0.25, 0.75, 0.875),
            };
            let b = m.min(if sampling == Sampling::Uniform { 100 } else { a / 3 });
            (Seq::mix_with([(s.take(a), Sampling::Uniform), (t.take(b), sampling)]), a + b)
        }
        _ => {
            if n == 0 {
                return (s, n);
            }
            let total = r.below(300);
            let first = total / 4;
            (Seq::mix([s.cycle(first), Seq::source(7).shuffle(r.next()).cycle(total - first)]), total)
        }
    }
}

#[derive(Clone, Debug)]
enum Op {
    Next,
    Nth(usize),
    Seek(usize),
    Range(usize, usize),
    Clone,
    Last,
    Count,
    BadSeek,
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
    let mut cursor = order.iter(..).unwrap();
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
            Op::Seek(raw) => {
                pos = position(raw, end);
                cursor.seek(pos).unwrap();
            }
            Op::Range(a, b) => {
                let (a, b) = (position(a, order.len()), position(b, order.len()));
                (pos, end) = (a.min(b), a.max(b));
                cursor.set_range(pos..end).unwrap();
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
            Op::BadSeek => {
                if let Some(bad) = end.checked_add(1)
                    && cursor.seek(bad).is_ok()
                {
                    return Err(fail());
                }
            }
            Op::BadRange => {
                use std::ops::Bound::{Excluded, Unbounded};
                if cursor.set_range((Excluded(usize::MAX), Unbounded)).is_ok() {
                    return Err(fail());
                }
            }
        }
        if (cursor.offset(), cursor.remaining()) != (pos, end - pos) {
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
        let (seq, len) = configuration(&mut r, 7);
        let order = Order::new(seq.clone()).unwrap();
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
                    3 => Op::Seek(raw),
                    4 => Op::Range(raw, r.next() as usize),
                    5 => Op::Clone,
                    6 => Op::Last,
                    7 => Op::Count,
                    8 => Op::BadSeek,
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
        minimize(vec![Op::Next, Op::Clone, Op::Next, Op::BadSeek, Op::Count], |ops| ops.iter().any(|op| matches!(op, Op::BadSeek)));
    assert!(matches!(result.as_slice(), [Op::BadSeek]));
}

#[test]
fn empty_ranges_resume_after_seeks_skips_and_clones() {
    let mix = || Seq::mix([Seq::source(100).shuffle(1), Seq::source(50).shuffle(2)]);
    let sequences = [
        Seq::source(0),
        Seq::source(100),
        Seq::source(100).shuffle(11),
        mix(),
        mix().repeat(3).skip(2).step_by(7),
        Seq::concat([mix(), mix().shuffle(7)]).skip(20),
    ];
    for seq in sequences {
        let order = Order::with_seed(seq, 19).unwrap();
        for start in [0, order.len() / 2, order.len()] {
            let mut cursor = order.iter(start..start).unwrap();
            assert_eq!(cursor.clone().count(), 0);
            assert_eq!(cursor.clone().last(), None);
            assert_eq!(cursor.next(), None);
            assert_eq!(cursor.nth(usize::MAX), None);
            cursor.set_range(0..0).unwrap();
            cursor.set_range(..).unwrap();
            assert_eq!(cursor.clone().count(), order.len());
            assert_eq!(cursor.clone().last(), order.len().checked_sub(1).and_then(|pos| order.get(pos)));
            let pos = order.len() / 3;
            cursor.seek(pos).unwrap();
            let mut cloned = cursor.clone();
            assert_eq!(cursor.nth(1), order.get(pos + 1));
            assert_eq!(cloned.nth(1), order.get(pos + 1));
            cursor.set_range(order.len()..).unwrap();
            assert_eq!(cursor.next(), None);
            let mut cloned = cursor.clone();
            for c in [&mut cursor, &mut cloned] {
                c.set_range(..).unwrap();
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
                let mut cursor = order.iter(..end).unwrap();
                assert_eq!(cursor.clone().last(), end.checked_sub(1).and_then(|pos| order.get(pos)));
                match exhaust {
                    0 => cursor.by_ref().for_each(drop),
                    1 => assert_eq!(cursor.nth(usize::MAX), None),
                    _ => cursor.seek(end).unwrap(),
                }
                assert_eq!((cursor.offset(), cursor.len()), (end, 0));
                assert_eq!(cursor.clone().last(), None);
                // Extending the range at the same position must find the next element,
                // including at concat/repeat boundaries and after the order's end.
                cursor.set_range(end..).unwrap();
                assert_eq!(cursor.next(), order.get(end));
                cursor.seek(0).unwrap();
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
    let mut cursor = order.iter(..).unwrap();
    cursor.next(); // Initialize buffers for the first child.
    cursor.seek(31).unwrap(); // End of the second child; its state has not been built.
    cursor.seek(10).unwrap(); // Same child index, but its state must now be initialized.
    assert_eq!(cursor.next(), Some(order.get(10).unwrap()));
}

#[test]
fn concat_children_keep_their_own_transform_parameters() {
    let source = |n, seed| Seq::source(n).shuffle(seed);
    let nested = |seed, n| {
        Seq::mix([
            source(n, seed).repeat(3).skip(2).skip(1).step_by(3),
            source(n + 2, seed + 1).repeat(2).take(n + 3),
            Seq::concat([source(n, seed + 2), Seq::source(n + 3)]).skip(1).step_by(2),
            Seq::mix([source(n, seed), source(n + 1, seed + 3)]).shuffle(seed + 4),
        ])
    };
    let seq = Seq::concat([nested(1, 11), nested(55, 19), Seq::source(3), nested(99, 7)]).repeat(3);
    let order = Order::with_seed(seq, 42).unwrap();
    let expected: Vec<_> = (0..order.len()).map(|pos| order.get(pos).unwrap()).collect();
    assert_eq!(order.iter(..).unwrap().collect::<Vec<_>>(), expected);
    let mut cursor = order.iter(..).unwrap();
    for pos in (0..order.len()).rev().step_by(3).chain((0..order.len()).step_by(7)) {
        cursor.seek(pos).unwrap();
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
    let seq = Seq::source(7).shuffle(13).repeat(5);
    let base = Order::new(seq.clone()).unwrap();
    let all: Vec<_> = base.iter(..).unwrap().map(|item| item.record_index).collect();
    for step in [1, 2, 8, usize::MAX] {
        let order = Order::new(seq.clone().skip(5).take(23).step_by(step)).unwrap();
        let expected: Vec<_> = all[5..28].iter().copied().step_by(step).collect();
        for n in 0..=expected.len() {
            let mut cursor = order.iter(..).unwrap();
            assert_eq!(cursor.nth(n).map(|item| item.record_index), expected.get(n).copied());
            cursor.seek(0).unwrap();
            assert_eq!(cursor.by_ref().map(|item| item.record_index).collect::<Vec<_>>(), expected);
            cursor.set_range(order.len()..).unwrap();
            assert_eq!(cursor.next(), None);
            cursor.set_range(..).unwrap();
            let mut copy = cursor.clone();
            assert_eq!(copy.by_ref().map(|item| item.record_index).collect::<Vec<_>>(), expected);
            assert_eq!(cursor.by_ref().map(|item| item.record_index).collect::<Vec<_>>(), expected);
        }
    }
}
