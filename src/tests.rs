//! Whole-crate tests: random configurations against a reference evaluator that
//! materializes every node, plus the errors, the folds and the reshuffling semantics.
//!
//! The reference evaluator shares the interleave and the permutation with the crate, so it
//! checks that the node tree, the folds, the cursors and random access compose those two
//! correctly, not the two themselves: the interleave is checked against a brute-force sort
//! and analytic schedule bounds in `interleave/tests.rs`, the permutation against
//! bijection and distribution statistics in `perm.rs`. The golden orders in
//! `tests/golden.rs` pin the result of all of it.

use crate::interleave::Interleave;
use crate::perm::{self, Shape};
use crate::*;

/// A test source: an id to compare orders by (and to salt shuffles with), and a length.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Src {
    id: u32,
    len: usize,
}

impl Source for Src {
    fn len(&self) -> usize {
        self.len
    }

    fn salt(&self) -> u64 {
        self.id as u64
    }
}

/// Salts and lengths of the sources under `seq` that can contribute elements, in order of
/// appearance: a subtree without elements counts for nothing, whatever is under it, nor
/// does a part of a concatenation that skips and takes above it cut away entirely. `seq` is valid.
fn salts(seq: &Seq<Src>, out: &mut Vec<(u64, usize)>) {
    let n = eval(seq, 0).unwrap().len();
    reachable(seq, 0..n, out);
}

/// [`salts`] of the sources that positions `range` of `seq` can reach. Skips and takes narrow
/// the range, concatenations hand each part its share of it, and a mix with one nonempty
/// part passes it through. Other nodes keep their children's whole range.
fn reachable(seq: &Seq<Src>, range: std::ops::Range<usize>, out: &mut Vec<(u64, usize)>) {
    if range.is_empty() {
        return;
    }
    let whole = |s: &Seq<Src>, out: &mut Vec<(u64, usize)>| salts(s, out);
    match seq {
        Seq::Source(s) => out.push((s.salt(), s.len)),
        Seq::Concat(parts) => {
            let mut offset = 0;
            for p in parts {
                let n = eval(p, 0).unwrap().len();
                let (a, b) = (range.start.max(offset), range.end.min(offset + n));
                if a < b {
                    reachable(p, a - offset..b - offset, out);
                }
                offset += n;
            }
        }
        Seq::Skip { n, inner } => reachable(inner, range.start + n..range.end + n, out),
        Seq::Take { inner, .. } => reachable(inner, range, out),
        Seq::Mix(parts) => {
            let mut nonempty = parts.iter().filter(|p| !eval(&p.seq, 0).unwrap().is_empty());
            if let (Some(part), None) = (nonempty.next(), nonempty.next()) {
                reachable(&part.seq, range, out);
            } else {
                parts.iter().for_each(|p| whole(&p.seq, out));
            }
        }
        Seq::StepBy { step: 1, inner } => reachable(inner, range, out),
        Seq::Cycle { len, inner } => cycled(inner, *len, out),
        Seq::Shuffle { inner, .. } | Seq::Repeat { inner, .. } | Seq::StepBy { inner, .. } => whole(inner, out),
    }
}

/// [`salts`] of `seq` cycled to `len` positions: within one repetition a take, beyond it the
/// whole sequence.
fn cycled(seq: &Seq<Src>, len: usize, out: &mut Vec<(u64, usize)>) {
    let n = eval(seq, 0).unwrap().len();
    if len <= n { reachable(seq, 0..len, out) } else { salts(seq, out) }
}

fn src(id: u32, len: usize) -> Seq<Src> {
    Seq::source(Src { id, len })
}

/// Elements as `(id, index)`.
fn ids<'a>(it: impl Iterator<Item = crate::Item<'a, Src>>) -> Vec<(u32, usize)> {
    it.map(|item| (item.source.id, item.record_index)).collect()
}

impl Error {
    /// A schedule or length rejection of a mix.
    fn is_schedule(&self) -> bool {
        matches!(self.kind(), ErrorKind::MixTooLong | ErrorKind::InvalidSchedule { .. } | ErrorKind::TooSteep { .. })
    }
}

/// An error at the root, for comparisons.
fn root(kind: ErrorKind) -> Error {
    Error::new(kind, Vec::new())
}

/// An error at the given path, for comparisons.
fn at(kind: ErrorKind, path: &[usize]) -> Error {
    Error::new(kind, path.to_vec())
}

/// A xorshift generator for the random configurations of every test module.
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform below `n`.
    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// A reference model of the repeat scopes retained by selections. Shuffle and multi-part
/// mix boundaries are opaque; concat boundaries and epoch-zero prefixes can be narrowed.
/// Unlike the compiler, all selections use one affine map over this length-only view.
/// This keeps the eager evaluator independent of compiled nodes and their cached levels.
enum LevelView {
    Opaque { len: usize, level: u8 },
    Concat(Vec<Self>),
    Repeat { len: usize, child: Box<Self> },
    Select { len: usize, start: usize, step: usize, child: Box<Self> },
}

impl LevelView {
    fn of(seq: &Seq<Src>) -> Self {
        let opaque = |len, level| Self::Opaque { len, level };
        match seq {
            Seq::Source(s) => opaque(s.len, 0),
            Seq::Concat(parts) => Self::Concat(parts.iter().map(Self::of).collect()),
            Seq::Mix(parts) => {
                let mut parts: Vec<_> = parts.iter().map(|p| Self::of(&p.seq)).filter(|p| p.len() > 0).collect();
                if parts.len() == 1 {
                    parts.pop().unwrap()
                } else {
                    opaque(parts.iter().map(Self::len).sum(), parts.iter().map(Self::level).max().unwrap_or(0))
                }
            }
            Seq::Shuffle { inner, .. } => {
                let child = Self::of(inner);
                if child.len() <= 1 { child } else { opaque(child.len(), child.level()) }
            }
            Seq::Repeat { times, inner } => {
                let child = Self::of(inner);
                let len = times * child.len();
                if len == 0 {
                    opaque(0, 0)
                } else if *times == 1 {
                    child
                } else {
                    Self::Repeat { len, child: Box::new(child) }
                }
            }
            Seq::Cycle { len, inner } => {
                let child = Self::of(inner);
                if *len <= child.len() { child.select(0, 1, *len) } else { Self::Repeat { len: *len, child: Box::new(child) } }
            }
            Seq::Skip { n, inner } => {
                let child = Self::of(inner);
                let len = child.len() - n;
                child.select(*n, 1, len)
            }
            Seq::Take { n, inner } => Self::of(inner).select(0, 1, *n),
            Seq::StepBy { step, inner } => {
                let child = Self::of(inner);
                let len = child.len().div_ceil(*step);
                child.select(0, *step, len)
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Opaque { len, .. } | Self::Repeat { len, .. } | Self::Select { len, .. } => *len,
            Self::Concat(parts) => parts.iter().map(Self::len).sum(),
        }
    }

    fn level(&self) -> u8 {
        match self {
            Self::Opaque { level, .. } => *level,
            Self::Concat(parts) => parts.iter().map(Self::level).max().unwrap_or(0),
            Self::Repeat { child, .. } => child.level() + 1,
            Self::Select { child, .. } => child.level(),
        }
    }

    fn select(self, start: usize, step: usize, len: usize) -> Self {
        if len == 0 {
            return Self::Opaque { len: 0, level: 0 };
        }
        if start == 0 && step == 1 && len == self.len() {
            return self;
        }
        match self {
            Self::Select { start: base, step: stride, child, .. } => child.select(base + start * stride, stride * step, len),
            Self::Repeat { child, .. } if start == 0 && (step == 1 || len == 1) => {
                if len <= child.len() {
                    child.select(0, 1, len)
                } else {
                    Self::Repeat { len, child }
                }
            }
            Self::Concat(parts) if step == 1 || len == 1 => {
                let mut at = 0;
                Self::Concat(
                    parts
                        .into_iter()
                        .filter_map(|p| {
                            let end = at + p.len();
                            let a = start.max(at);
                            let b = (start + len).min(end);
                            let offset = a.saturating_sub(at);
                            at = end;
                            (a < b).then(|| p.select(offset, 1, b - a))
                        })
                        .collect(),
                )
            }
            child => Self::Select { len, start, step, child: Box::new(child) },
        }
    }
}

/// Materializes `seq` in context `ctx` by the definitions in the crate docs.
fn eval(seq: &Seq<Src>, ctx: u64) -> Result<Vec<(u32, usize)>, Error> {
    Ok(match seq {
        Seq::Source(s) => (0..s.len).map(|i| (s.id, i)).collect(),
        Seq::Concat(parts) => {
            let mut out = Vec::new();
            for p in parts {
                out.extend(eval(p, ctx)?);
            }
            out
        }
        Seq::Mix(parts) => {
            let evs = parts.iter().map(|p| eval(&p.seq, ctx)).collect::<Result<Vec<_>, _>>()?;
            let lens: Vec<usize> = evs.iter().map(|v| v.len()).collect();
            let schedule: Vec<Schedule> = parts.iter().map(|p| p.schedule).collect();
            let il = Interleave::with_schedule(&lens, &schedule).map_err(|e| {
                let (kind, part) = e.into_kind();
                at(kind, part.as_slice())
            })?;
            il.iter(0..il.len()).map(|(s, j)| evs[s][j]).collect()
        }
        Seq::Cycle { len, inner } => {
            let n = eval(inner, ctx)?.len();
            if n == 0 && *len > 0 {
                return Err(root(ErrorKind::EmptyCycle));
            }
            eval(&cycle_of(inner, *len, n), ctx)?
        }
        Seq::Shuffle { seed, inner } => {
            let v = eval(inner, ctx)?;
            let mut under = Vec::new();
            salts(inner, &mut under);
            let (shape, key) = (Shape::new(v.len()), perm::key(*seed, ctx, perm::shuffle_salt(under)));
            (0..v.len()).map(|i| v[perm::permute(shape, key, i)]).collect()
        }
        Seq::Repeat { times, inner } => {
            // Validate even when empty; derive levels independently of the compiler.
            let mut out = eval(inner, ctx)?;
            let level = LevelView::of(inner).level() + 1;
            out.clear();
            for e in 0..*times {
                out.extend(eval(inner, perm::epoch_ctx(ctx, e, level))?);
            }
            out
        }
        Seq::Skip { n, inner } => {
            let v = eval(inner, ctx)?;
            if *n > v.len() {
                return Err(root(ErrorKind::SkipOutOfRange { n: *n, len: v.len() }));
            }
            v[*n..].to_vec()
        }
        Seq::Take { n, inner } => {
            let v = eval(inner, ctx)?;
            if *n > v.len() {
                return Err(root(ErrorKind::TakeOutOfRange { n: *n, len: v.len() }));
            }
            v[..*n].to_vec()
        }
        Seq::StepBy { step, inner } => {
            if *step == 0 {
                return Err(root(ErrorKind::ZeroStep));
            }
            eval(inner, ctx)?.into_iter().step_by(*step).collect()
        }
    })
}

/// A cycle in terms of repeat and take: `seq` (of length `n`) repeated as often as `len`
/// positions need, then cut to `len`. A single repetition introduces no level.
fn cycle_of(seq: &Seq<Src>, len: usize, n: usize) -> Seq<Src> {
    let times = if n == 0 { 0 } else { len.div_ceil(n) };
    let inner = if times > 1 { Seq::Repeat { times, inner: Box::new(seq.clone()) } } else { seq.clone() };
    Seq::Take { n: len, inner: Box::new(inner) }
}

fn random_seq(rng: &mut Rng, depth: u32, lens: &[usize]) -> Seq<Src> {
    if depth == 0 || rng.below(5) == 0 {
        let id = rng.below(lens.len());
        return src(id as u32, lens[id]);
    }
    let parts = |rng: &mut Rng, depth| (0..1 + rng.below(3)).map(|_| random_seq(rng, depth, lens)).collect::<Vec<_>>();
    match rng.below(9) {
        8 => {
            let inner = random_seq(rng, depth - 1, lens);
            let n = eval(&inner, 0).map(|v| v.len()).unwrap_or(0);
            inner.cycle(if n == 0 { 0 } else { rng.below(70) })
        }
        7 => {
            // Empty parts in a mix, which must not affect the order (checked separately) and
            // must not break the walk.
            let mut ps = parts(rng, depth - 1);
            ps.insert(rng.below(ps.len() + 1), src(0, 0));
            Seq::mix(ps)
        }
        0 => Seq::concat(parts(rng, depth - 1)),
        1 => Seq::mix(parts(rng, depth - 1)),
        2 => Seq::mix(parts(rng, depth - 1).into_iter().map(|p| {
            let schedule = match rng.below(6) {
                0 => Schedule::ramp(0.3, 0.6),
                1 => Schedule::delayed(0.5),
                2 => Schedule::until(0.5),
                3 => Schedule::trapezoid(0.1, 0.3, 0.6, 0.9),
                _ => Schedule::Uniform,
            };
            (p, schedule)
        })),
        3 => random_seq(rng, depth - 1, lens).shuffle(rng.next()),
        4 => random_seq(rng, depth - 1, lens).repeat(rng.below(4)),
        5 => {
            let inner = random_seq(rng, depth - 1, lens);
            let n = eval(&inner, 0).map(|v| v.len()).unwrap_or(0);
            let start = rng.below(n + 1);
            match rng.below(3) {
                0 => inner.skip(start),
                1 => inner.take(rng.below(n + 1)),
                _ => inner.skip(start).take(rng.below(n - start + 1)),
            }
        }
        _ => {
            let inner = random_seq(rng, depth - 1, lens);
            let n = eval(&inner, 0).map(|v| v.len()).unwrap_or(0);
            inner.skip(rng.below(n + 1)).step_by(1 + rng.below(4))
        }
    }
}

/// Every random configuration: `get` at every position, the full iteration, random ranges
/// and seeks all agree with the reference.
#[test]
fn random_configurations_match_reference() {
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
    let lens = [0usize, 1, 2, 3, 7, 13, 40];
    let (mut checked, mut skipped) = (0, 0);
    for round in 0..800 {
        let seq = random_seq(&mut rng, 4, &lens);
        let seed = rng.next();
        let order = match Order::with_seed(seq.clone(), seed) {
            Ok(o) => o,
            Err(e) if e.is_schedule() => {
                assert!(eval(&seq, seed).is_err_and(|e| e.is_schedule()), "round {round}: {seq:?}");
                assert!(!e.path().is_empty() || matches!(seq, Seq::Mix(_)), "round {round}: {e}");
                skipped += 1;
                continue;
            }
            Err(e) => panic!("round {round}: {e} for {seq:?}"),
        };
        let reference = eval(&seq, seed).unwrap();
        let n = reference.len();
        assert_eq!(order.len(), n, "round {round}: {seq:?}");
        for (i, &r) in reference.iter().enumerate() {
            let crate::Item { source: s, record_index: idx, .. } = order.get(i).unwrap();
            assert_eq!((s.id, idx), r, "round {round}: get({i}) of {seq:?}");
        }
        assert_eq!(ids(order.cursor(0..n).unwrap()), reference, "round {round}: {seq:?}");
        for _ in 0..4 {
            let a = rng.below(n + 1);
            let b = a + rng.below(n - a + 1);
            assert_eq!(ids(order.cursor(a..b).unwrap()), reference[a..b], "round {round}: {a}..{b} of {seq:?}");
        }
        let mut cursor = order.cursor(0..n).unwrap();
        for _ in 0..6 {
            let a = rng.below(n + 1);
            cursor.reset(a..).unwrap();
            assert_eq!(cursor.offset(), a);
            let m = rng.below(n - a + 1);
            assert_eq!(cursor.len(), n - a);
            let got = ids(cursor.by_ref().take(m));
            assert_eq!(got, reference[a..a + m], "round {round}: seek {a} of {seq:?}");
            // A forward seek and `nth` skip; both must land where a fresh cursor would.
            let b = a + m + rng.below(n - a - m + 1);
            cursor.reset(b..).unwrap();
            assert_eq!(ids(cursor.by_ref().take(3)), reference[b..(b + 3).min(n)], "round {round}: forward seek {b} of {seq:?}");
            let p = cursor.offset();
            let k = rng.below(5);
            assert_eq!(
                cursor.nth(k).map(|item| (item.source.id, item.record_index)),
                reference.get(p + k).copied(),
                "round {round}: nth({k}) at {p} of {seq:?}"
            );
            assert_eq!(cursor.offset(), (p + k + 1).min(n));
        }
        // `reset` re-ranges the same cursor, forward or backward, from wherever it stands.
        for _ in 0..4 {
            let a = rng.below(n + 1);
            let b = a + rng.below(n - a + 1);
            cursor.reset(a..b).unwrap();
            assert_eq!(cursor.len(), b - a);
            assert_eq!(ids(cursor.by_ref()), reference[a..b], "round {round}: reset {a}..{b} of {seq:?}");
            assert_eq!(cursor.offset(), b);
        }
        assert_eq!(ids((&order).into_iter()), reference);
        checked += 1;
    }
    assert!(checked > 400, "only {checked} configurations checked ({skipped} invalid)");
}

/// Deeper, larger configurations exercise long strides over mixes (interleave re-seeks),
/// epoch boundaries inside strides and nested repeats.
#[test]
fn large_configurations_match_reference() {
    let mut rng = Rng(0xFEED_FACE_CAFE_BEEF);
    let lens = [50usize, 333, 1000, 2000];
    for round in 0..40 {
        let seq = random_seq(&mut rng, 5, &lens);
        let order = match Order::new(seq.clone()) {
            Ok(o) => o,
            Err(e) if e.is_schedule() => continue,
            Err(e) => panic!("round {round}: {e}"),
        };
        let reference = eval(&seq, 0).unwrap();
        let n = reference.len();
        if n == 0 || n > 200_000 {
            continue;
        }
        assert_eq!(ids(order.cursor(0..n).unwrap()), reference, "round {round}: {seq:?}");
        for _ in 0..8 {
            let a = rng.below(n + 1);
            let b = (a + rng.below(500)).min(n);
            assert_eq!(ids(order.cursor(a..b).unwrap()), reference[a..b], "round {round}: {a}..{b}");
            let p = a.min(n - 1);
            let crate::Item { source: s, record_index: i, .. } = order.get(p).unwrap();
            assert_eq!((s.id, i), reference[p]);
        }
    }
}

#[test]
fn shuffle_is_a_permutation_and_reshuffles_per_epoch() {
    let seq = src(7, 1000).shuffle(3).repeat(3);
    let order = Order::new(seq.clone()).unwrap();
    assert_eq!(order.len(), 3000);
    let epochs: Vec<Vec<usize>> = (0..3)
        .map(|e| {
            order
                .cursor(e * 1000..(e + 1) * 1000)
                .unwrap()
                .map(|crate::Item { source: s, record_index: i, .. }| {
                    assert_eq!(s.id, 7);
                    i
                })
                .collect()
        })
        .collect();
    for epoch in &epochs {
        let mut sorted = epoch.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..1000).collect::<Vec<_>>());
    }
    assert_ne!(epochs[0], epochs[1]);
    assert_ne!(epochs[1], epochs[2]);
    assert_ne!(epochs[0], epochs[2]);
    // The first repetition is the sequence itself, and repeating once changes nothing,
    // even around a repeat (which must not count one level deeper for it).
    let once = Order::new(src(7, 1000).shuffle(3)).unwrap();
    assert_eq!(once.cursor(0..1000).unwrap().map(|item| item.record_index).collect::<Vec<_>>(), epochs[0]);
    assert_eq!(ids(Order::new(src(7, 1000).shuffle(3).repeat(1)).unwrap().cursor(0..1000).unwrap()), ids(once.cursor(0..1000).unwrap()));
    assert_eq!(ids(Order::new(seq.clone().repeat(1)).unwrap().iter()), ids(order.iter()));
    assert_eq!(ids(Order::new(Seq::concat([seq.clone().repeat(1)]).repeat(1)).unwrap().iter()), ids(order.iter()));
    // A cycle that fits within its part is not repeated, so it is the part itself;
    // one that repeats also preserves the entire first pass, including nested epochs.
    let part = || src(7, 100).shuffle(3).repeat(2);
    let fits = Order::new(Seq::mix([part().cycle(200), src(8, 1000).cycle(200)])).unwrap();
    let repeats = Order::new(Seq::mix([part().cycle(250), src(8, 1000).cycle(250)])).unwrap();
    let sevens = |o: &Order<Src>| ids(o.iter()).into_iter().filter(|e| e.0 == 7).collect::<Vec<_>>();
    let alone = ids(Order::new(part()).unwrap().iter());
    assert_eq!(sevens(&fits), alone);
    assert_eq!(sevens(&repeats)[..200], alone);
    // Adding a repeat preserves the nested prefix, including when a take removes it.
    for extended in [part().repeat(2), part().cycle(201), part().repeat(2).take(200)] {
        let extended = Order::new(extended).unwrap();
        assert_eq!(ids(extended.cursor(..200).unwrap()), alone);
    }
    // Nested repeats: (outer 0, inner 1) and (outer 1, inner 0) are different orders.
    let nested = Order::new(src(7, 100).shuffle(3).repeat(2).repeat(2)).unwrap();
    let block = |b: usize| ids(nested.cursor(b * 100..(b + 1) * 100).unwrap());
    assert_ne!(block(1), block(2));
    assert_eq!(block(0), ids(Order::new(src(7, 100).shuffle(3)).unwrap().cursor(0..100).unwrap()));
    // Same seed twice under a concat: the same order twice.
    let twice = Order::new(Seq::concat([src(7, 1000).shuffle(3), src(7, 1000).shuffle(3)])).unwrap();
    let v = ids(twice.cursor(0..2000).unwrap());
    assert_eq!(v[..1000], v[1000..]);
    // Another salt or another length with the same seed: unrelated orders.
    let salted = Order::new(Seq::concat([src(7, 1000).shuffle(3), src(8, 1000).shuffle(3), src(7, 999).shuffle(3)])).unwrap();
    let w = ids(salted.iter());
    let alike = |a: &[(u32, usize)], b: &[(u32, usize)]| a.iter().zip(b).filter(|(x, y)| x.1 == y.1).count();
    assert!(alike(&w[..1000], &w[1000..2000]) < 10);
    assert!(alike(&w[..999], &w[2000..]) < 10);
    assert!(alike(&w[1000..1999], &w[2000..]) < 10);
    // The order's seed changes every shuffle, whether given at construction or set later.
    let reseeded = Order::with_seed(seq, 99).unwrap();
    assert_ne!(ids(reseeded.cursor(0..1000).unwrap()), ids(order.cursor(0..1000).unwrap()));
    let mut later = order.clone();
    later.set_seed(99);
    assert_eq!(later.seed(), 99);
    assert_eq!(ids(later.iter()), ids(reseeded.iter()));
    later.set_seed(0);
    assert_eq!(ids(later.iter()), ids(order.iter()));
}

/// Levels follow retained inner scopes, including unequal branches and selections that
/// erase an epoch-zero scope. Check each outer epoch against an independently seeded child.
#[test]
fn repeat_levels_are_assigned_from_the_inside_out() {
    let x = || src(0, 17).shuffle(3);
    let deep = || src(1, 11).shuffle(7).repeat(2).repeat(3);
    let cases = [
        ("plain", x(), 0),
        ("repeat", x().repeat(2), 1),
        ("nested", x().repeat(2).repeat(3), 2),
        ("concat max", Seq::concat([deep(), x().repeat(2)]), 2),
        ("mix max", Seq::mix([x().repeat(2), deep()]), 2),
        ("single repeat", deep().repeat(1), 2),
        ("empty repeat", Seq::concat([deep().repeat(0), x()]), 0),
        ("empty take", Seq::mix([deep().take(0), x()]), 0),
        ("prefix", x().repeat(3).take(17).skip(1), 0),
        ("short cycle", x().repeat(3).cycle(17), 0),
        ("one strided position", x().repeat(3).step_by(51), 0),
        ("discard concat tail", Seq::concat([x(), deep()]).take(17), 0),
        ("discard concat head", Seq::concat([deep(), x()]).skip(66), 0),
        ("retain multiple concat children", Seq::concat([deep(), x(), x()]).skip(66), 0),
        ("trim a concat boundary", Seq::concat([x(), deep()]).take(39), 1),
        ("recover inner repeat level", deep().take(22), 1),
        ("retained shuffled child", Seq::concat([x(), deep().shuffle(23), x()]).take(83), 2),
        ("retained mix child", Seq::concat([x(), Seq::mix([deep(), x()]), x()]).skip(17).take(83), 2),
        ("retained strided child", Seq::concat([x(), deep().step_by(2), x()]).take(50), 2),
        ("retained sliced child", Seq::concat([x(), deep().skip(1), x()]).take(82), 2),
        ("retained slice", x().repeat(3).skip(1).take(16), 1),
        ("retained stride", x().repeat(3).step_by(3), 1),
        ("partial extra epoch", x().cycle(18), 1),
        ("cycle of repeat", x().repeat(2).cycle(35), 2),
    ];
    for (name, seq, inner_level) in cases {
        // A shuffle above the child makes the outer context observable even when a
        // selection has kept only one position of an inner shuffled epoch.
        let seq = seq.shuffle(19);
        assert_eq!(LevelView::of(&seq).level(), inner_level, "reference level: {name}");
        for seed in [0, 51] {
            let repeated = Order::with_seed(seq.clone().repeat(3), seed).unwrap();
            assert!(matches!(repeated.root, crate::order::Node::Repeat { level, .. } if level == inner_level + 1), "{name}");
            let n = repeated.len() / 3;
            for epoch in 0..3 {
                let context = perm::epoch_ctx(seed, epoch, inner_level + 1);
                let expected = ids(Order::with_seed(seq.clone(), context).unwrap().iter());
                let start = epoch * n;
                assert_eq!(ids(repeated.cursor(start..start + n).unwrap()), expected, "{name}: epoch {epoch}");
                for (pos, &item) in expected.iter().enumerate() {
                    let actual = repeated.get(start + pos).unwrap();
                    assert_eq!((actual.source.id, actual.record_index), item, "{name}: get({})", start + pos);
                }
            }
        }
    }
}

#[test]
fn extending_nested_repetitions_preserves_every_existing_position() {
    let seq = Seq::mix([src(0, 17).shuffle(3).repeat(2), Seq::concat([src(1, 11).shuffle(7).repeat(2).repeat(3), src(2, 5)])]);
    for seed in [0, 51] {
        let once = Order::with_seed(seq.clone(), seed).unwrap();
        let n = once.len();
        let long = Order::with_seed(seq.clone().repeat(3), seed).unwrap();
        let all = ids(long.iter());
        assert_eq!(ids(once.iter()), all[..n]);
        for len in [0, 1, 17, n - 1, n, n + 1, 2 * n, 2 * n + 1, 3 * n] {
            for selected in [seq.clone().cycle(len), seq.clone().repeat(3).take(len)] {
                let order = Order::with_seed(selected, seed).unwrap();
                assert_eq!(ids(order.iter()), all[..len], "prefix length {len}");
            }
        }
        assert_ne!(all[..n], all[n..2 * n]);
        assert_ne!(all[n..2 * n], all[2 * n..]);
    }
}

#[test]
fn shards_partition_the_sequence() {
    let base = Seq::mix([src(0, 1000).shuffle(1), src(1, 300).shuffle(2)]).repeat(2);
    let order = Order::new(base.clone()).unwrap();
    let all = ids(order.cursor(0..order.len()).unwrap());
    let mut from_shards = Vec::new();
    for w in 0..8 {
        let shard = Order::new(base.clone().skip(w).step_by(8)).unwrap();
        let elems = ids(shard.cursor(0..shard.len()).unwrap());
        for (i, &e) in elems.iter().enumerate() {
            assert_eq!(e, all[w + 8 * i]);
        }
        from_shards.extend(elems);
    }
    assert_eq!(from_shards.len(), all.len());
}

/// `map` keeps the structure: over sources of the same lengths and salts the same indices
/// come out; over bare lengths (salt 0) only the shuffles differ.
#[test]
fn map_keeps_the_order() {
    struct Loaded {
        salt: u64,
        len: usize,
    }
    impl Source for Loaded {
        fn len(&self) -> usize {
            self.len
        }
        fn salt(&self) -> u64 {
            self.salt
        }
    }
    let seq = Seq::mix([src(0, 700).shuffle(1).repeat(2), Seq::concat([src(1, 50), src(2, 120).shuffle(2)])]).skip(1).step_by(3);
    let order = Order::new(seq.clone()).unwrap();
    let loaded = Order::new(seq.clone().map(|s| Loaded { salt: s.salt(), len: s.len })).unwrap();
    assert_eq!(loaded.sources().iter().map(|l| l.len).collect::<Vec<_>>(), [700, 50, 120]);
    let a: Vec<(usize, usize)> = order.iter().map(|item| (item.source.len, item.record_index)).collect();
    let b: Vec<(usize, usize)> = loaded.iter().map(|item| (item.source.len, item.record_index)).collect();
    assert_eq!(a, b);
    let lens = Order::new(seq.map(|s| s.len)).unwrap();
    assert_eq!(lens.sources(), &[700, 50, 120]);
    let c: Vec<(usize, usize)> = lens.iter().map(|item| (*item.source, item.record_index)).collect();
    assert_ne!(a, c);
    let unshuffled = |v: &[(usize, usize)]| v.iter().enumerate().filter(|(_, e)| e.0 == 50).map(|(p, e)| (p, e.1)).collect::<Vec<_>>();
    assert_eq!(unshuffled(&a), unshuffled(&c));
}

#[test]
fn errors() {
    let a = src(0, 10);
    assert_eq!(Order::new(a.clone().skip(3).take(9)).unwrap_err(), root(ErrorKind::TakeOutOfRange { n: 9, len: 7 }));
    assert_eq!(Order::new(a.clone().skip(11)).unwrap_err(), root(ErrorKind::SkipOutOfRange { n: 11, len: 10 }));
    assert_eq!(Order::new(a.clone().take(11)).unwrap_err(), root(ErrorKind::TakeOutOfRange { n: 11, len: 10 }));
    assert_eq!(Order::new(a.clone().skip(2).take(8)).unwrap().len(), 8);
    assert_eq!(Order::new(a.clone().skip(10)).unwrap().len(), 0);
    assert_eq!(Order::new(a.clone().step_by(0)).unwrap_err(), root(ErrorKind::ZeroStep));
    // Beyond usize::MAX on every target: a repeat of a repeat, and a concat of two halves.
    assert_eq!(Order::new(a.clone().repeat(usize::MAX).repeat(usize::MAX)).unwrap_err().kind(), &ErrorKind::LengthOverflow);
    let half = || src(0, usize::MAX / 2 + 1);
    assert_eq!(Order::new(Seq::concat([half(), half()])).unwrap_err(), root(ErrorKind::LengthOverflow));
    // A mix that folds away is still validated; a schedule problem is found at the part.
    let over1 = Seq::mix([(src(0, 10), Schedule::delayed(2.0))]);
    assert_eq!(
        Order::new(over1).unwrap_err(),
        at(ErrorKind::InvalidSchedule { schedule: Schedule::delayed(2.0), reason: crate::ScheduleReason::InvalidBreakpoints }, &[0])
    );
    let steep = Seq::concat([a.clone(), Seq::mix([(a.clone(), Schedule::Uniform), (src(1, 1 << 30), Schedule::until(1e-6))])]);
    let err = Order::new(steep).unwrap_err();
    assert_eq!(err.kind(), &ErrorKind::TooSteep { len: 1 << 30, peak_rate: 1e6, limit: crate::MAX_MIX_LEN });
    assert_eq!(err.path(), &[1, 1]);
    // The path leads to the node: part 1 of the mix, then the single child of the shuffle.
    let nested = Seq::mix([a.clone(), Seq::concat([a.clone(), a.take(11).shuffle(1)])]).repeat(2);
    let err = Order::new(nested).unwrap_err();
    assert_eq!(err.kind(), &ErrorKind::TakeOutOfRange { n: 11, len: 10 });
    assert_eq!(err.path(), [0, 1, 1, 0]);
    assert_eq!(err.to_string(), "cannot take 11 of 10 positions (at node 0/1/1/0)");
    assert_eq!(root(ErrorKind::ZeroStep).to_string(), "step is zero (at the root)");
    assert_eq!(err.into_kind(), ErrorKind::TakeOutOfRange { n: 11, len: 10 });
    assert_eq!(Order::new(src(0, 0).cycle(5)).unwrap_err(), root(ErrorKind::EmptyCycle));
    assert_eq!(Order::new(Seq::concat([src(0, 3), src(1, 0).cycle(5)])).unwrap_err(), at(ErrorKind::EmptyCycle, &[1]));
    assert_eq!(Order::new(src(0, 5).take(6).cycle(5)).unwrap_err(), at(ErrorKind::TakeOutOfRange { n: 6, len: 5 }, &[0]));
    assert_eq!(Order::new(src(0, 0).cycle(0)).unwrap().len(), 0);
    assert_eq!(root(ErrorKind::EmptyCycle).to_string(), "cannot cycle a sequence without elements (at the root)");
}

/// A cycle is a repeat cut to length and preserves its existing prefix as it grows.
#[test]
fn cycles() {
    use crate::order::Node;
    let x = || src(0, 100).shuffle(3);
    let all = ids(Order::new(x().repeat(4)).unwrap().iter());
    for len in [0, 1, 99, 100, 101, 250, 400] {
        let order = Order::new(x().cycle(len)).unwrap();
        assert_eq!(order.len(), len);
        assert_eq!(ids(order.iter()), all[..len], "cycle({len})");
        assert_eq!(ids(Order::new(x().repeat(4).take(len)).unwrap().iter()), all[..len]);
    }
    assert_eq!(ids(Order::new(x().cycle(99)).unwrap().iter()), ids(Order::new(x().take(99)).unwrap().iter()));
    // Inner repeat levels stay unchanged when a cycle extends to another pass.
    let y = || src(0, 10).shuffle(3).repeat(2);
    assert_eq!(ids(Order::new(y().cycle(15)).unwrap().iter()), ids(Order::new(y()).unwrap().cursor(..15).unwrap()));
    assert_eq!(ids(Order::new(y().cycle(45)).unwrap().iter()), ids(Order::new(y().repeat(3)).unwrap().cursor(..45).unwrap()));
    assert_eq!(ids(Order::new(y().cycle(45)).unwrap().iter())[..20], ids(Order::new(y()).unwrap().iter())[..]);
    // Node shapes: no slice above a repeat, a short cycle is a slice or the child.
    let root = |seq: Seq<Src>| Order::new(seq).unwrap().root;
    assert!(matches!(root(x().cycle(250)), Node::Repeat { child_len: 100, len: 250, .. }));
    assert!(matches!(root(x().repeat(4).take(250)), Node::Repeat { child_len: 100, len: 250, .. }));
    assert!(matches!(root(x().repeat(4).take(100)), Node::Shuffle { .. }));
    assert!(matches!(root(x().cycle(7)), Node::Slice { start: 0, len: 7, .. }));
    assert!(matches!(root(src(0, 100).cycle(7)), Node::Source { offset: 0, len: 7, .. }));
    assert!(matches!(root(x().repeat(4).skip(1).take(250)), Node::Slice { start: 1, len: 250, .. }));
    let Node::Mix { children, .. } = root(Seq::mix([x().cycle(200), src(1, 1000).shuffle(4).cycle(100)])) else { panic!() };
    assert!(matches!(children[0], Node::Repeat { child_len: 100, len: 200, .. }));
    assert!(matches!(children[1], Node::Slice { start: 0, len: 100, .. }));
    // The longest order representable by the public API; still finite.
    let endless = Order::new(x().cycle(usize::MAX)).unwrap();
    assert_eq!(endless.len(), usize::MAX);
    let last = usize::MAX - 1;
    assert_eq!(ids(endless.cursor(last..).unwrap()), vec![(0, endless.get(last).unwrap().record_index)]);
    assert_eq!(ids(endless.cursor(..300).unwrap()), all[..300]);
    // A cycle over a concatenation narrows it like a take, beyond one repetition it counts
    // every part.
    let base = ids(Order::new(src(0, 100).shuffle(1)).unwrap().iter());
    assert_eq!(ids(Order::new(Seq::concat([src(0, 100), src(1, 50)]).cycle(100).shuffle(1)).unwrap().iter()), base);
    assert_ne!(ids(Order::new(Seq::concat([src(0, 100), src(1, 50)]).cycle(200).shuffle(1)).unwrap().iter())[..100], base[..]);
    // A cycled concat part is narrowed to the requested count before mixing.
    let part = || Seq::concat([src(0, 100), src(1, 50)]);
    let narrowed = Order::new(Seq::mix([part().cycle(75), src(2, 75)]).shuffle(1)).unwrap();
    let plain = Order::new(Seq::mix([src(0, 100).take(75), src(2, 75)]).shuffle(1)).unwrap();
    assert_eq!(ids(narrowed.iter()), ids(plain.iter()));
}

#[test]
fn depth_limit() {
    let chain = |levels: u32| (1..levels).fold(src(0, 10), |s, _| s.take(10));
    assert_eq!(Order::new(chain(MAX_DEPTH)).unwrap().len(), 10);
    let err = Order::new(chain(MAX_DEPTH + 1)).unwrap_err();
    assert_eq!(err.kind(), &ErrorKind::TooDeep);
    assert_eq!(err.path().len(), MAX_DEPTH as usize);
}

/// Configurations just beyond the supported limit report the first invalid node.
#[test]
fn configurations_over_the_depth_limit_are_rejected() {
    let run = || {
        let deep = || (0..MAX_DEPTH).fold(src(0, 10), |s, _| s.take(10));
        assert_eq!(Order::new(deep()).unwrap_err().kind(), &ErrorKind::TooDeep);
        let bad = || src(0, 10).take(99);
        let out_of_range = ErrorKind::TakeOutOfRange { n: 99, len: 10 };
        assert_eq!(Order::new(Seq::concat([bad(), deep()])).unwrap_err().kind(), &out_of_range);
        assert_eq!(Order::new(Seq::mix([bad(), deep()])).unwrap_err().kind(), &out_of_range);
        assert_eq!(Order::new(deep().step_by(0)).unwrap_err().kind(), &ErrorKind::ZeroStep);
        let first_too_deep = Seq::concat([deep(), bad()]);
        assert_eq!(Order::new(first_too_deep).unwrap_err().kind(), &ErrorKind::TooDeep);
    };
    std::thread::Builder::new().stack_size(2 << 20).spawn(run).unwrap().join().unwrap();
}

/// An empty source, or a part that folds away as empty, does not change the shuffle above
/// it: only sources that contribute elements salt it.
#[test]
fn empty_parts_do_not_affect_shuffles_above() {
    let x = || src(0, 100);
    let base = ids(Order::new(x().shuffle(1)).unwrap().iter());
    let same = |seq: Seq<Src>| assert_eq!(ids(Order::new(seq).unwrap().iter()), base);
    same(Seq::concat([x(), src(1, 0)]).shuffle(1));
    same(Seq::concat([src(0, 0), x()]).shuffle(1));
    same(Seq::concat([src(1, 0), x(), src(2, 7).repeat(0), src(3, 7).take(0), src(4, 3).skip(3), src(5, 2).skip(2).step_by(1)]).shuffle(1));
    same(Seq::mix([x(), src(1, 0), Seq::concat([src(2, 5).skip(5), src(3, 0)])]).shuffle(1));
    same(Seq::mix([x()]).shuffle(1));
    same(x().step_by(1).shuffle(1));
    // A part of a concatenation that a skip or take cuts away entirely does not count
    // either, however the concatenation nests; a part it touches counts.
    same(Seq::concat([x(), src(1, 50)]).take(100).shuffle(1));
    same(Seq::concat([src(1, 50), x()]).skip(50).shuffle(1));
    same(Seq::concat([src(1, 50), x(), src(2, 7)]).skip(50).take(100).shuffle(1));
    same(Seq::concat([Seq::concat([src(1, 5), x()]), src(2, 3)]).skip(5).take(100).shuffle(1));
    same(Seq::concat([src(1, 5), Seq::concat([x(), src(2, 3)])]).take(105).skip(5).shuffle(1));
    let with = |extra: Seq<Src>| ids(Order::new(Seq::concat([x(), extra]).take(101).shuffle(1)).unwrap().iter());
    assert_ne!(with(src(1, 1)), base);
    assert_ne!(with(src(0, 1)), base);
    // An explicit skip removes preceding concat parts before deriving a later
    // shuffle salt.
    let strided = |seq: Seq<Src>| ids(Order::new(seq).unwrap().iter());
    assert_eq!(strided(Seq::concat([src(1, 1), x()]).skip(1).step_by(2)), strided(x().step_by(2)));
    assert_eq!(strided(Seq::concat([src(1, 1), x()]).skip(1).step_by(2).shuffle(1)), strided(x().step_by(2).shuffle(1)));
}

/// The folds: what compiles to a single node.
#[test]
fn folds() {
    use crate::order::Node;
    let root = |seq: Seq<Src>| Order::new(seq).unwrap().root;
    let a = || src(0, 100);
    assert!(
        matches!(root(a().skip(1).step_by(8).skip(1).step_by(4)), Node::Stride { step: 32, offset: 8, len: 3, ref child } if matches!(**child, Node::Source { offset: 1, .. }))
    );
    assert!(matches!(root(a().skip(1).step_by(8).skip(1).step_by(4).skip(1).step_by(2)), Node::Source { offset: 41, len: 1, .. }));
    assert!(
        matches!(root(a().shuffle(1).skip(10).skip(2).step_by(3)), Node::Stride { step: 3, offset: 12, len: 30, ref child } if matches!(**child, Node::Shuffle { .. }))
    );
    assert!(matches!(root(a().shuffle(1).skip(10).take(50).skip(5)), Node::Slice { start: 15, len: 45, .. }));
    assert!(
        matches!(root(a().skip(1).step_by(4).skip(1).step_by(2)), Node::Stride { step: 8, offset: 4, len: 12, ref child } if matches!(**child, Node::Source { offset: 1, .. }))
    );
    let cat = || Seq::concat([src(1, 10), src(2, 20), src(3, 30)]);
    assert!(matches!(root(cat().take(10)), Node::Source { src: 0, offset: 0, len: 10 }));
    assert!(matches!(root(cat().skip(15)), Node::Concat { ref offsets, ref children } if offsets == &[0, 15, 45] && children.len() == 2));
    assert!(matches!(root(cat().skip(10).take(20)), Node::Source { src: 1, offset: 0, len: 20 }));
    assert!(matches!(root(cat().skip(10)), Node::Concat { ref children, .. } if children.len() == 2));
    assert!(
        matches!(root(cat().skip(5).take(30)), Node::Concat { ref offsets, ref children } if offsets == &[0, 5, 25, 30] && children.len() == 3)
    );
    assert_eq!(
        ids(Order::new(cat().skip(15).skip(3).step_by(7)).unwrap().iter()),
        ids(Order::new(cat()).unwrap().iter()).into_iter().skip(18).step_by(7).collect::<Vec<_>>()
    );
}

/// A mix's order does not depend on empty parts, wherever they sit.
#[test]
fn empty_mix_parts_do_not_affect_the_order() {
    let mut rng = Rng(0x0E0E_0E0E_1234_5678);
    for round in 0..200 {
        let k = 2 + rng.below(6);
        let parts: Vec<(Seq<Src>, Schedule)> = (0..k)
            .map(|i| {
                let s = match rng.below(4) {
                    0 => Schedule::delayed(0.3 + 0.1 * rng.below(5) as f64),
                    1 => Schedule::ramp(0.1, 0.6),
                    _ => Schedule::Uniform,
                };
                (src(i as u32, 1 + rng.below(80)), s)
            })
            .collect();
        let mut with = parts.clone();
        for _ in 0..1 + rng.below(3) {
            let at = rng.below(with.len() + 1);
            with.insert(at, (src(99, 0), Schedule::delayed(0.9)));
        }
        let (Ok(a), Ok(b)) = (Order::new(Seq::mix(with)), Order::new(Seq::mix(parts))) else { continue };
        assert_eq!(ids(a.iter()), ids(b.iter()), "round {round}");
        assert_eq!(a.sources().len(), b.sources().len() + a.sources().iter().filter(|s| s.id == 99).count());
    }
}

/// `Seq` is `Eq` and `Hash` by comparing floats bitwise, with the two zeros equal.
#[test]
fn seq_eq_and_hash() {
    use std::collections::HashSet;
    let a = Seq::mix([(src(0, 5), Schedule::ramp(0.0, 0.5))]);
    let b = Seq::mix([(src(0, 5), Schedule::ramp(-0.0, 0.5))]);
    let c = Seq::mix([(src(0, 5), Schedule::ramp(0.1, 0.5))]);
    let trapezoid = Seq::mix([(src(0, 5), Schedule::trapezoid(0.0, 0.5, 1.0, 1.0))]);
    assert_eq!(a, b);
    assert_eq!(a, trapezoid);
    assert_ne!(a, c);
    let set: HashSet<Seq<Src>> = [a.clone(), b, c.clone(), trapezoid].into_iter().collect();
    assert_eq!(set.len(), 2);
    assert!(set.contains(&a) && set.contains(&c));
    let nan = Seq::mix([(src(0, 5), Schedule::delayed(f64::NAN))]);
    assert_eq!(nan, nan.clone());
    assert_eq!(Schedule::delayed(0.5), Schedule::ramp(0.5, 0.5));
    assert_eq!(Schedule::until(0.5), Schedule::fading(0.5, 0.5));
    assert_ne!(Schedule::until(0.5), Schedule::delayed(0.5));
    assert_eq!(Schedule::trapezoid(0.0, 0.0, 1.0, 1.0), Schedule::ramp(0.0, 0.0));
    assert_ne!(Schedule::Uniform, Schedule::ramp(0.0, 0.0));
    assert_eq!(Schedule::trapezoid(-0.0, 0.1, 0.5, 0.9), Schedule::trapezoid(0.0, 0.1, 0.5, 0.9));
    assert_eq!(MixPart::from(src(1, 2)), MixPart { seq: src(1, 2), schedule: Schedule::Uniform });
}

#[test]
fn edge_cases() {
    let empty = Order::new(Seq::<Src>::concat([])).unwrap();
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.cursor(0..0).unwrap().count(), 0);
    let empty = Order::new(src(1, 5).shuffle(1).take(0).repeat(4)).unwrap();
    assert!(empty.is_empty());
    assert_eq!(Order::new(src(1, 5).skip(5).step_by(3)).unwrap().len(), 0);
    assert_eq!(Order::new(src(1, 5).skip(9).step_by(3)).unwrap_err().kind(), &ErrorKind::SkipOutOfRange { n: 9, len: 5 });
    let one = Order::new(src(2, 1).shuffle(5).repeat(3)).unwrap();
    assert_eq!(ids(one.cursor(0..3).unwrap()), vec![(2, 0); 3]);
    let order = Order::with_seed(src(0, 5), 77).unwrap();
    assert_eq!(order.sources(), &[Src { id: 0, len: 5 }]);
    assert_eq!(order.seed(), 77);
    assert_eq!(format!("{order:?}"), "Order { len: 5, seed: 77, sources: [Src { id: 0, len: 5 }] }");
    let mut c = order.cursor(1..4).unwrap();
    assert_eq!(format!("{c:?}"), "Cursor { position: 1, end: 4 }");
    assert_eq!(c.len(), 3);
    assert_eq!(c.next().map(|item| (item.source.id, item.record_index)), Some((0, 1)));
    c.reset(4..4).unwrap();
    assert!(c.next().is_none());
    assert_eq!(c.nth(3), None);
    c.reset(0..4).unwrap();
    assert_eq!(c.nth(2).map(|item| (item.source.id, item.record_index)), Some((0, 2)));
    assert_eq!(c.offset(), 3);
    assert_eq!(c.nth(1), None);
    assert_eq!(c.offset(), 4);
    c.reset(0..4).unwrap();
    assert_eq!(ids(c), vec![(0, 0), (0, 1), (0, 2), (0, 3)]);
    // `count` and `last` answer without walking, `last` by random access.
    let shuffled = Order::new(src(0, 1000).shuffle(3)).unwrap();
    assert_eq!(shuffled.cursor(10..).unwrap().count(), 990);
    assert_eq!(
        shuffled.cursor(10..900).unwrap().last().map(|item| (item.source.id, item.record_index)),
        Some((0, shuffled.get(899).unwrap().record_index))
    );
    assert_eq!(shuffled.cursor(7..7).unwrap().last(), None);
    let mut c = shuffled.iter();
    c.nth(4);
    assert_eq!(c.count(), 995);
    // Range forms of `cursor`.
    assert_eq!(ids(order.iter()), ids(order.cursor(0..5).unwrap()));
    assert_eq!(ids(order.cursor(3..).unwrap()), vec![(0, 3), (0, 4)]);
    assert_eq!(ids(order.cursor(..=1).unwrap()), vec![(0, 0), (0, 1)]);
    assert_eq!(ids(order.cursor(2..=2).unwrap()), vec![(0, 2)]);
    assert_eq!(order.cursor(5..).unwrap().count(), 0);
    assert_eq!(ids(order.cursor((std::ops::Bound::Excluded(3), std::ops::Bound::Unbounded)).unwrap()), vec![(0, 4)]);
    assert_eq!(ids((&order).into_iter()), ids(order.iter()));
    // A cursor run past its end, or ranged at the end, still moves forward correctly.
    let mut c = order.cursor(1..3).unwrap();
    assert_eq!(c.nth(10), None);
    c.reset(2..).unwrap();
    assert_eq!(ids(c.by_ref()), vec![(0, 2), (0, 3), (0, 4)]);
    c.reset(..1).unwrap();
    assert_eq!(ids(c.by_ref()), vec![(0, 0)]);
    c.reset(1..1).unwrap();
    c.reset(3..4).unwrap();
    assert_eq!(ids(c.by_ref()), vec![(0, 3)]);
    let mut at_end = order.cursor(5..).unwrap();
    assert_eq!(at_end.next(), None);
    at_end.reset(4..).unwrap();
    assert_eq!(ids(at_end), vec![(0, 4)]);
    assert_eq!(order.into_sources(), vec![Src { id: 0, len: 5 }]);
    // Bare lengths are sources; a source may be shared through a reference.
    let lens = [5usize, 3];
    let shared = Order::new(Seq::concat([Seq::source(&lens[0]), Seq::source(&lens[1]), Seq::source(&lens[0]).shuffle(1)])).unwrap();
    assert_eq!(shared.len(), 13);
    assert_eq!(shared.sources().len(), 3);
    // Which source an element came from, also when sources compare equal.
    let which: Vec<usize> = shared.iter().map(|item| item.source_ordinal).collect();
    assert_eq!(which, [vec![0; 5], vec![1; 3], vec![2; 5]].concat());
    let equal = Order::new(Seq::mix([src(1, 4), src(1, 4), src(1, 2)])).unwrap();
    let which: Vec<usize> = equal.iter().map(|item| item.source_ordinal).collect();
    assert_eq!(which, [0, 1, 0, 1, 2, 0, 1, 0, 1, 2]);
    struct Unit;
    impl Source for Unit {
        fn len(&self) -> usize {
            2
        }
    }
    let units = Order::new(Seq::concat([Seq::source(Unit), Seq::source(Unit)])).unwrap();
    assert_eq!(units.iter().map(|item| item.source_ordinal).collect::<Vec<_>>(), [0, 0, 1, 1]);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn huge_lengths() {
    // Lengths far beyond anything materializable: positions must still resolve.
    let seq = src(0, 1 << 40).shuffle(1).repeat(1 << 20).skip(12345);
    let order = Order::new(seq).unwrap();
    assert_eq!(order.len(), (1usize << 60) - 12345);
    let last = order.len() - 1;
    let crate::Item { source: s, record_index: i, .. } = order.get(last).unwrap();
    assert_eq!(s.id, 0);
    assert!(i < 1 << 40);
    assert_eq!(ids(order.cursor(last..).unwrap()), vec![(0, i)]);
    let v = ids(order.cursor(1 << 59..(1 << 59) + 5).unwrap());
    for (k, &e) in v.iter().enumerate() {
        let crate::Item { source: s, record_index: i, .. } = order.get((1 << 59) + k).unwrap();
        assert_eq!((s.id, i), e);
    }
}

/// Reject overflow at its node, before any parent can shorten or discard it.
#[test]
fn sequence_lengths_must_fit_usize() {
    let oversized = [Seq::concat([src(0, usize::MAX), src(1, 1)]), src(0, usize::MAX / 2 + 1).repeat(2)];
    // On 64-bit targets the smaller numerical mix limit is reached first.
    #[cfg(target_pointer_width = "32")]
    let oversized = oversized.into_iter().chain([Seq::mix([src(0, usize::MAX / 2 + 1), src(1, usize::MAX / 2 + 1)])]);
    for seq in oversized {
        assert_eq!(Order::new(seq.clone()).unwrap_err(), root(ErrorKind::LengthOverflow));
        for shortened in [
            seq.clone().take(10),
            seq.clone().take(0),
            seq.clone().skip(usize::MAX),
            seq.clone().step_by(2),
            seq.clone().cycle(10),
            seq.clone().cycle(0),
            seq.repeat(0),
        ] {
            assert_eq!(Order::new(shortened.clone()).unwrap_err(), at(ErrorKind::LengthOverflow, &[0]));
            let nested = Seq::concat([src(2, 1), shortened]);
            assert_eq!(Order::new(nested).unwrap_err(), at(ErrorKind::LengthOverflow, &[1, 0]));
        }
    }
    assert_eq!(root(ErrorKind::LengthOverflow).to_string(), "sequence length exceeds usize::MAX (at the root)");

    // The inclusive length limit remains usable without materializing any elements.
    let at_limit = [
        src(0, usize::MAX),
        src(0, usize::MAX).shuffle(7),
        Seq::concat([src(0, usize::MAX - 1), src(1, 1)]),
        src(0, 1).repeat(usize::MAX),
        src(0, 3).shuffle(7).cycle(usize::MAX),
    ];
    #[cfg(target_pointer_width = "32")]
    let at_limit = at_limit.into_iter().chain([Seq::mix([src(0, usize::MAX - 1), src(1, 1)])]);
    for seq in at_limit {
        let order = Order::new(seq).unwrap();
        assert_eq!(order.len(), usize::MAX);
        let last = usize::MAX - 1;
        assert_eq!(order.cursor(last..).unwrap().next(), order.get(last));
    }
}

#[test]
fn cloned_cursor_continues_independently() {
    let order = Order::new(Seq::mix([src(0, 500).shuffle(1).repeat(2), src(1, 300).shuffle(2)]).skip(1).step_by(3)).unwrap();
    let n = order.len();
    let mut c = order.cursor(0..n).unwrap();
    let head = ids(c.by_ref().take(100));
    let d = c.clone();
    let rest_c = ids(c);
    let rest_d = ids(d);
    assert_eq!(rest_c, rest_d);
    assert_eq!([head, rest_c].concat(), ids(order.cursor(0..n).unwrap()));
}

#[test]
fn types_are_send_and_sync() {
    fn assert_send_sync<X: Send + Sync>() {}
    assert_send_sync::<Seq<usize>>();
    assert_send_sync::<Order<usize>>();
    assert_send_sync::<Cursor<'static, usize>>();
    assert_send_sync::<Error>();
    assert_send_sync::<ErrorKind>();
    assert_send_sync::<Schedule>();
    assert_send_sync::<MixPart<usize>>();
}

/// Prefixes, suffixes and finite ranges composed from skip and take.
#[test]
fn skip_and_take_ranges() {
    let a = || src(0, 10);
    assert_eq!(ids(Order::new(a().take(3)).unwrap().cursor(0..3).unwrap()), vec![(0, 0), (0, 1), (0, 2)]);
    assert_eq!(ids(Order::new(a().skip(8)).unwrap().cursor(0..2).unwrap()), vec![(0, 8), (0, 9)]);
    assert_eq!(ids(Order::new(a().skip(3).take(1)).unwrap().cursor(0..1).unwrap()), vec![(0, 3)]);
    assert_eq!(Order::new(a()).unwrap().len(), 10);
    assert_eq!(Order::new(a().skip(4).take(0)).unwrap().len(), 0);
}

/// Schedules at the steep end of what a mix accepts, at lengths near its limit: seeks and
/// walks agree, so the interleave's count-and-fix loops settle there too.
#[test]
#[cfg(target_pointer_width = "64")]
fn steep_schedule_at_scale() {
    let seq = Seq::mix([
        (src(0, 1 << 45), Schedule::Uniform),
        (src(1, 1 << 44), Schedule::delayed(0.5)), // final rate 2: length × rate = 2⁴⁵, within 2⁴⁶
        (src(2, 1 << 40), Schedule::ramp(0.0, 1.0)),
    ]);
    let order = Order::new(seq).unwrap();
    let n = order.len();
    assert_eq!(n, (1 << 45) + (1 << 44) + (1 << 40));
    for start in [0, n / 2 - 777, n - 1500, 12_345_678_901] {
        let walked = ids(order.cursor(start..start + 1500).unwrap());
        for (k, &e) in walked.iter().enumerate() {
            let crate::Item { source: s, record_index: i, .. } = order.get(start + k).unwrap();
            assert_eq!((s.id, i), e, "position {}", start + k);
        }
    }
    // Too steep is rejected, not looped over.
    let steep = Seq::mix([(src(0, 1 << 45), Schedule::Uniform), (src(1, 1 << 46), Schedule::delayed(0.5))]);
    assert!(matches!(Order::new(steep).unwrap_err().kind(), ErrorKind::TooSteep { .. } | ErrorKind::MixTooLong));
    assert_eq!(MAX_MIX_LEN, 1 << 46);
}

/// Explicit counts repeat short parts (reshuffled) and truncate long ones before mixing.
#[test]
fn mix_with_explicit_counts() {
    let seq = Seq::mix([src(0, 100).shuffle(1).cycle(1800), src(1, 5000).shuffle(2).cycle(1200)]);
    let order = Order::new(seq).unwrap();
    assert_eq!(order.len(), 3000);
    let all = ids(order.cursor(0..3000).unwrap());
    assert_eq!(all.iter().filter(|e| e.0 == 0).count(), 1800);
    assert_eq!(all.iter().filter(|e| e.0 == 1).count(), 1200);
    // Source 0 (100 elements) is repeated 18 times, each repetition a permutation, the
    // first two different; source 1 contributes 1200 distinct elements.
    let zeros: Vec<usize> = all.iter().filter(|e| e.0 == 0).map(|e| e.1).collect();
    for epoch in zeros.chunks(100) {
        let mut sorted = epoch.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..100).collect::<Vec<_>>());
    }
    assert_ne!(zeros[..100], zeros[100..200]);
    let mut ones: Vec<usize> = all.iter().filter(|e| e.0 == 1).map(|e| e.1).collect();
    ones.sort_unstable();
    ones.dedup();
    assert_eq!(ones.len(), 1200);
    // A part is compiled once: its sources appear once.
    let once = Order::new(Seq::mix([Seq::concat([src(0, 4), src(1, 4)]).cycle(10), src(2, 10).cycle(20)])).unwrap();
    assert_eq!(once.sources().len(), 3);
}
