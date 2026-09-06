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
            (s.stride(step, offset), len)
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
            (Seq::weighted(total, [(s, 1.0), (Seq::source(7).shuffle(r.next()), 3.0)]), total)
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
    let mut cursor = order.iter(..).indexed();
    let (mut pos, mut end) = (0, order.len());
    for (step, op) in ops.iter().enumerate() {
        let fail = || format!("operation {step}: {op:?}, position={pos}, end={end}");
        match *op {
            Op::Next | Op::Nth(_) => {
                let n = if let Op::Nth(n) = *op { n } else { 0 };
                let expected = (n < end - pos).then(|| order.get_indexed(pos + n));
                let actual = if matches!(op, Op::Next) { cursor.next() } else { cursor.nth(n) };
                if actual != expected {
                    return Err(format!("{}: {actual:?} != {expected:?}", fail()));
                }
                pos = if n < end - pos { pos + n + 1 } else { end };
            }
            Op::Seek(raw) => {
                pos = position(raw, end);
                cursor.seek(pos);
            }
            Op::Range(a, b) => {
                let (a, b) = (position(a, order.len()), position(b, order.len()));
                (pos, end) = (a.min(b), a.max(b));
                cursor.set_range(pos..end);
            }
            Op::Clone => cursor = cursor.clone(),
            Op::Last => {
                let expected = (pos < end).then(|| order.get_indexed(end - 1));
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
                    && cursor.try_seek(bad).is_ok()
                {
                    return Err(fail());
                }
            }
            Op::BadRange => {
                use std::ops::Bound::{Excluded, Unbounded};
                if cursor.try_set_range((Excluded(usize::MAX), Unbounded)).is_ok() {
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
fn boundary_skip_rebinds_retained_child_before_backward_seek() {
    let order = Order::new(Seq::concat([
        Seq::mix([Seq::source(5), Seq::source(5)]),
        Seq::mix([Seq::source(7), Seq::source(7), Seq::source(7)]),
        Seq::source(3),
    ]))
    .unwrap();
    let mut cursor = order.iter(..).indexed();
    cursor.next(); // Initialize buffers for the first child.
    cursor.seek(31); // End of the second child; retain the first child's buffers.
    cursor.seek(10); // Same child index, but the retained buffers must be rebound.
    assert_eq!(cursor.next(), Some(order.get_indexed(10)));
}

#[test]
fn recycled_children_replace_every_transform_parameter() {
    let source = |n, seed| Seq::source(n).shuffle(seed);
    let nested = |seed, n| {
        Seq::mix([
            source(n, seed).repeat(3).skip(2).stride(3, 1),
            source(n + 2, seed + 1).repeat(2).take(n + 3),
            Seq::concat([source(n, seed + 2), Seq::source(n + 3)]).stride(2, 1),
            Seq::mix([source(n, seed), source(n + 1, seed + 3)]).shuffle(seed + 4),
        ])
    };
    let seq = Seq::concat([nested(1, 11), nested(55, 19), Seq::source(3), nested(99, 7)]).repeat(3);
    let order = Order::with_seed(seq, 42).unwrap();
    let expected: Vec<_> = (0..order.len()).map(|pos| order.get_indexed(pos)).collect();
    assert_eq!(order.iter(..).indexed().collect::<Vec<_>>(), expected);
    let mut cursor = order.iter(..).indexed();
    for pos in (0..order.len()).rev().step_by(3).chain((0..order.len()).step_by(7)) {
        cursor.seek(pos);
        let mut copy = cursor.clone();
        for expected in &expected[pos..(pos + 5).min(order.len())] {
            assert_eq!(cursor.next(), Some(*expected));
            assert_eq!(copy.next(), Some(*expected));
        }
    }
}
