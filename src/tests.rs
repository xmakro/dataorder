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
fn salts(seq: &Seq<Src>, out: &mut Vec<(u64, u64)>) {
    let n = eval(seq, 0).unwrap().len();
    reachable(seq, 0..n, out);
}

/// [`salts`] of the sources that positions `range` of `seq` can reach. Skips and takes narrow
/// the range, concatenations hand each part its share of it, and a mix with one nonempty
/// part passes it through. Other nodes keep their children's whole range.
fn reachable(seq: &Seq<Src>, range: std::ops::Range<usize>, out: &mut Vec<(u64, u64)>) {
    if range.is_empty() {
        return;
    }
    let whole = |s: &Seq<Src>, out: &mut Vec<(u64, u64)>| salts(s, out);
    match seq {
        Seq::Source(s) => out.push((s.salt(), s.len as u64)),
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
        Seq::Stride { step: 1, offset, inner } => reachable(inner, range.start + offset..range.end + offset, out),
        Seq::Cycle { len, inner } => cycled(inner, *len, out),
        Seq::Shuffle { inner, .. } | Seq::Repeat { inner, .. } | Seq::Stride { inner, .. } => whole(inner, out),
    }
}

/// [`salts`] of `seq` cycled to `len` positions: within one repetition a take, beyond it the
/// whole sequence.
fn cycled(seq: &Seq<Src>, len: usize, out: &mut Vec<(u64, u64)>) {
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
    fn is_sampling(&self) -> bool {
        matches!(self.kind(), ErrorKind::MixTooLong | ErrorKind::InvalidSampling { .. } | ErrorKind::TooSteep)
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

    /// Uniform below `n`.
    pub(crate) fn below64(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Materializes `seq` in context `ctx` by the definitions in the crate docs.
fn eval(seq: &Seq<Src>, ctx: u64) -> Result<Vec<(u32, usize)>, Error> {
    eval_at(seq, ctx, 0)
}

/// `depth` repeats enclose `seq`.
fn eval_at(seq: &Seq<Src>, ctx: u64, depth: u32) -> Result<Vec<(u32, usize)>, Error> {
    Ok(match seq {
        Seq::Source(s) => (0..s.len).map(|i| (s.id, i)).collect(),
        Seq::Concat(parts) => {
            let mut out = Vec::new();
            for p in parts {
                out.extend(eval_at(p, ctx, depth)?);
            }
            out
        }
        Seq::Mix(parts) => {
            let evs = parts.iter().map(|p| eval_at(&p.seq, ctx, depth)).collect::<Result<Vec<_>, _>>()?;
            let lens: Vec<u64> = evs.iter().map(|v| v.len() as u64).collect();
            let sampling: Vec<Sampling> = parts.iter().map(|p| p.sampling).collect();
            let il = Interleave::with_sampling(&lens, &sampling).map_err(|e| {
                let detail = e.detail();
                let (kind, part) = e.into_kind();
                at(kind, part.as_slice()).with_sampling_detail(detail)
            })?;
            il.iter(0..il.len()).map(|(s, j)| evs[s][j as usize]).collect()
        }
        Seq::Cycle { len, inner } => {
            let n = eval_at(inner, ctx, depth)?.len();
            if n == 0 && *len > 0 {
                return Err(root(ErrorKind::EmptyCycle));
            }
            eval_at(&cycle_of(inner, *len, n), ctx, depth)?
        }
        Seq::Shuffle { seed, inner } => {
            let v = eval_at(inner, ctx, depth)?;
            let mut under = Vec::new();
            salts(inner, &mut under);
            let (shape, key) = (Shape::new(v.len() as u64), perm::key(*seed, ctx, perm::shuffle_salt(under)));
            (0..v.len() as u64).map(|i| v[perm::permute(shape, key, i) as usize]).collect()
        }
        Seq::Repeat { times, inner } => {
            // A single repetition is no repeat at all: the inner sequence is not one level
            // deeper. Validated even when repeated zero times, like the compiler does.
            let inner_depth = if *times > 1 { depth + 1 } else { depth };
            let mut out = eval_at(inner, ctx, inner_depth)?;
            out.clear();
            for e in 0..*times {
                out.extend(eval_at(inner, perm::epoch_ctx(ctx, e as u64, depth), inner_depth)?);
            }
            out
        }
        Seq::Skip { n, inner } => {
            let v = eval_at(inner, ctx, depth)?;
            if *n > v.len() {
                return Err(root(ErrorKind::SkipOutOfRange { n: *n, len: v.len() as u64 }));
            }
            v[*n..].to_vec()
        }
        Seq::Take { n, inner } => {
            let v = eval_at(inner, ctx, depth)?;
            if *n > v.len() {
                return Err(root(ErrorKind::TakeOutOfRange { n: *n, len: v.len() as u64 }));
            }
            v[..*n].to_vec()
        }
        Seq::Stride { step, offset, inner } => {
            if *step == 0 {
                return Err(root(ErrorKind::ZeroStep));
            }
            eval_at(inner, ctx, depth)?.into_iter().skip(*offset).step_by(*step).collect()
        }
    })
}

/// A cycle in terms of repeat and take: `seq` (of length `n`) repeated as often as `len`
/// positions need, then cut to `len`; a repetition only when there is more than one, so
/// that the depth of the repeats inside is what the compiler gives them.
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
        2 => Seq::mix_with(parts(rng, depth - 1).into_iter().map(|p| {
            let sampling = match rng.below(6) {
                0 => Sampling::DelayedLinear { start: 0.3, full: 0.6 },
                1 => Sampling::DelayedLinear { start: 0.5, full: 0.5 },
                2 => Sampling::until(0.5),
                3 => Sampling::trapezoid(0.1, 0.3, 0.6, 0.9),
                _ => Sampling::Uniform,
            };
            (p, sampling)
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
                _ => inner.slice(start..start + rng.below(n - start + 1)).unwrap(),
            }
        }
        _ => {
            let inner = random_seq(rng, depth - 1, lens);
            let n = eval(&inner, 0).map(|v| v.len()).unwrap_or(0);
            inner.stride(1 + rng.below(4), rng.below(n + 2))
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
        assert_eq!(seq.check(), Order::new(seq.clone()).map(|o| o.len()), "round {round}: check");
        let order = match Order::with_seed(seq.clone(), seed) {
            Ok(o) => o,
            Err(e) if e.is_sampling() => {
                assert!(eval(&seq, seed).is_err_and(|e| e.is_sampling()), "round {round}: {seq:?}");
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
        assert_eq!(ids(order.iter(0..n).unwrap()), reference, "round {round}: {seq:?}");
        for _ in 0..4 {
            let a = rng.below(n + 1);
            let b = a + rng.below(n - a + 1);
            assert_eq!(ids(order.iter(a..b).unwrap()), reference[a..b], "round {round}: {a}..{b} of {seq:?}");
        }
        let mut cursor = order.iter(0..n).unwrap();
        for _ in 0..6 {
            let a = rng.below(n + 1);
            cursor.seek(a).unwrap();
            assert_eq!(cursor.offset(), a);
            let m = rng.below(n - a + 1);
            assert_eq!(cursor.len(), n - a);
            let got = ids(cursor.by_ref().take(m));
            assert_eq!(got, reference[a..a + m], "round {round}: seek {a} of {seq:?}");
            // A forward seek and `nth` skip; both must land where a fresh cursor would.
            let b = a + m + rng.below(n - a - m + 1);
            cursor.seek(b).unwrap();
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
        // `set_range` re-ranges the same cursor, forward or backward, from wherever it stands.
        for _ in 0..4 {
            let a = rng.below(n + 1);
            let b = a + rng.below(n - a + 1);
            cursor.set_range(a..b).unwrap();
            assert_eq!(cursor.len(), b - a);
            assert_eq!(ids(cursor.by_ref()), reference[a..b], "round {round}: set_range {a}..{b} of {seq:?}");
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
            Err(e) if e.is_sampling() => continue,
            Err(e) => panic!("round {round}: {e}"),
        };
        let reference = eval(&seq, 0).unwrap();
        let n = reference.len();
        if n == 0 || n > 200_000 {
            continue;
        }
        assert_eq!(ids(order.iter(0..n).unwrap()), reference, "round {round}: {seq:?}");
        for _ in 0..8 {
            let a = rng.below(n + 1);
            let b = (a + rng.below(500)).min(n);
            assert_eq!(ids(order.iter(a..b).unwrap()), reference[a..b], "round {round}: {a}..{b}");
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
                .iter(e * 1000..(e + 1) * 1000)
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
    assert_eq!(once.iter(0..1000).unwrap().map(|item| item.record_index).collect::<Vec<_>>(), epochs[0]);
    assert_eq!(ids(Order::new(src(7, 1000).shuffle(3).repeat(1)).unwrap().iter(0..1000).unwrap()), ids(once.iter(0..1000).unwrap()));
    assert_eq!(ids(Order::new(seq.clone().repeat(1)).unwrap().iter(..).unwrap()), ids(order.iter(..).unwrap()));
    assert_eq!(ids(Order::new(Seq::concat([seq.clone().repeat(1)]).repeat(1)).unwrap().iter(..).unwrap()), ids(order.iter(..).unwrap()));
    // A cycle that fits within its part is not repeated, so it is the part itself;
    // one that is repeated preserves the part's first inner epoch, but being one repeat
    // deeper reshuffles the part's later epochs even during its first outer repetition.
    let part = || src(7, 100).shuffle(3).repeat(2);
    let fits = Order::new(Seq::mix([part().cycle(200), src(8, 1000).cycle(200)])).unwrap();
    let repeats = Order::new(Seq::mix([part().cycle(250), src(8, 1000).cycle(250)])).unwrap();
    let sevens = |o: &Order<Src>| ids(o.iter(..).unwrap()).into_iter().filter(|e| e.0 == 7).collect::<Vec<_>>();
    let alone = ids(Order::new(part()).unwrap().iter(..).unwrap());
    assert_eq!(sevens(&fits), alone);
    assert_eq!(sevens(&repeats)[..100], alone[..100]);
    assert_ne!(sevens(&repeats)[100..200], alone[100..200]);
    // Adding a repeat changes the nested prefix even if a take cuts away the new epoch.
    for extended in [part().repeat(2), part().cycle(201), part().repeat(2).take(200)] {
        let extended = Order::new(extended).unwrap();
        assert_eq!(ids(extended.iter(..100).unwrap()), alone[..100]);
        assert_ne!(ids(extended.iter(100..200).unwrap()), alone[100..200]);
    }
    // Nested repeats: (outer 0, inner 1) and (outer 1, inner 0) are different orders.
    let nested = Order::new(src(7, 100).shuffle(3).repeat(2).repeat(2)).unwrap();
    let block = |b: usize| ids(nested.iter(b * 100..(b + 1) * 100).unwrap());
    assert_ne!(block(1), block(2));
    assert_eq!(block(0), ids(Order::new(src(7, 100).shuffle(3)).unwrap().iter(0..100).unwrap()));
    // Same seed twice under a concat: the same order twice.
    let twice = Order::new(Seq::concat([src(7, 1000).shuffle(3), src(7, 1000).shuffle(3)])).unwrap();
    let v = ids(twice.iter(0..2000).unwrap());
    assert_eq!(v[..1000], v[1000..]);
    // Another salt or another length with the same seed: unrelated orders.
    let salted = Order::new(Seq::concat([src(7, 1000).shuffle(3), src(8, 1000).shuffle(3), src(7, 999).shuffle(3)])).unwrap();
    let w = ids(salted.iter(..).unwrap());
    let alike = |a: &[(u32, usize)], b: &[(u32, usize)]| a.iter().zip(b).filter(|(x, y)| x.1 == y.1).count();
    assert!(alike(&w[..1000], &w[1000..2000]) < 10);
    assert!(alike(&w[..999], &w[2000..]) < 10);
    assert!(alike(&w[1000..1999], &w[2000..]) < 10);
    // The order's seed changes every shuffle, whether given at construction or set later.
    let reseeded = Order::with_seed(seq, 99).unwrap();
    assert_ne!(ids(reseeded.iter(0..1000).unwrap()), ids(order.iter(0..1000).unwrap()));
    let mut later = order.clone();
    later.set_seed(99);
    assert_eq!(later.seed(), 99);
    assert_eq!(ids(later.iter(..).unwrap()), ids(reseeded.iter(..).unwrap()));
    later.set_seed(0);
    assert_eq!(ids(later.iter(..).unwrap()), ids(order.iter(..).unwrap()));
}

#[test]
fn shards_partition_the_sequence() {
    let base = Seq::mix([src(0, 1000).shuffle(1), src(1, 300).shuffle(2)]).repeat(2);
    let order = Order::new(base.clone()).unwrap();
    let all = ids(order.iter(0..order.len()).unwrap());
    let mut from_shards = Vec::new();
    for w in 0..8 {
        let shard = Order::new(base.clone().shard(8, w).unwrap()).unwrap();
        let elems = ids(shard.iter(0..shard.len()).unwrap());
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
    let seq = Seq::mix([src(0, 700).shuffle(1).repeat(2), Seq::concat([src(1, 50), src(2, 120).shuffle(2)])]).shard(3, 1).unwrap();
    let order = Order::new(seq.clone()).unwrap();
    let loaded = Order::new(seq.clone().map(|s| Loaded { salt: s.salt(), len: s.len })).unwrap();
    assert_eq!(loaded.sources().iter().map(|l| l.len).collect::<Vec<_>>(), [700, 50, 120]);
    let a: Vec<(usize, usize)> = order.iter(..).unwrap().map(|item| (item.source.len, item.record_index)).collect();
    let b: Vec<(usize, usize)> = loaded.iter(..).unwrap().map(|item| (item.source.len, item.record_index)).collect();
    assert_eq!(a, b);
    let lens = Order::new(seq.map(|s| s.len)).unwrap();
    assert_eq!(lens.sources(), &[700, 50, 120]);
    let c: Vec<(usize, usize)> = lens.iter(..).unwrap().map(|item| (*item.source, item.record_index)).collect();
    assert_ne!(a, c);
    let unshuffled = |v: &[(usize, usize)]| v.iter().enumerate().filter(|(_, e)| e.0 == 50).map(|(p, e)| (p, e.1)).collect::<Vec<_>>();
    assert_eq!(unshuffled(&a), unshuffled(&c));
}

#[test]
fn errors() {
    let a = src(0, 10);
    assert_eq!(Order::new(a.clone().slice(3..12).unwrap()).unwrap_err(), root(ErrorKind::TakeOutOfRange { n: 9, len: 7 }));
    assert_eq!(Order::new(a.clone().skip(11)).unwrap_err(), root(ErrorKind::SkipOutOfRange { n: 11, len: 10 }));
    assert_eq!(Order::new(a.clone().take(11)).unwrap_err(), root(ErrorKind::TakeOutOfRange { n: 11, len: 10 }));
    assert_eq!(Order::new(a.clone().slice(2..=9).unwrap()).unwrap().len(), 8);
    assert_eq!(Order::new(a.clone().slice(10..).unwrap()).unwrap().len(), 0);
    assert_eq!(Order::new(a.clone().stride(0, 0)).unwrap_err(), root(ErrorKind::ZeroStep));
    // Beyond 64 bits on every target: a repeat of a repeat, and a concat of two halves of 2⁶⁴.
    assert_eq!(Order::new(a.clone().repeat(usize::MAX).repeat(usize::MAX)).unwrap_err().kind(), &ErrorKind::LengthOverflow);
    let half = || src(0, 1 << 31).repeat(1 << 31).repeat(2);
    assert_eq!(Order::new(Seq::concat([half(), half()])).unwrap_err(), root(ErrorKind::LengthOverflow));
    // A mix that folds away is still validated; a schedule problem is found at the part.
    let over1 = Seq::mix_with([(src(0, 10), Sampling::DelayedLinear { start: 2.0, full: 2.0 })]);
    assert_eq!(
        Order::new(over1).unwrap_err(),
        at(ErrorKind::InvalidSampling { sampling: Sampling::DelayedLinear { start: 2.0, full: 2.0 } }, &[0])
            .with_sampling_detail(Some(crate::SamplingDetail::InvalidBreakpoints))
    );
    let steep = Seq::concat([
        a.clone(),
        Seq::mix_with([(a.clone(), Sampling::Uniform), (src(1, 1 << 30).repeat(1 << 16), Sampling::delayed(0.999))]),
    ]);
    let err = Order::new(steep).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::TooSteep | ErrorKind::MixTooLong), "{err}");
    assert_eq!(err.path(), if err.kind() == &ErrorKind::TooSteep { &[1, 1][..] } else { &[1][..] });
    // The path leads to the node: part 1 of the mix, then the single child of the shuffle.
    let nested = Seq::mix([a.clone(), Seq::concat([a.clone(), a.take(11).shuffle(1)])]).repeat(2);
    let err = Order::new(nested).unwrap_err();
    assert_eq!(err.kind(), &ErrorKind::TakeOutOfRange { n: 11, len: 10 });
    assert_eq!(err.path(), [0, 1, 1, 0]);
    assert_eq!(err.to_string(), "cannot take 11 of 10 positions (at node 0/1/1/0)");
    assert_eq!(root(ErrorKind::ZeroStep).to_string(), "stride step is zero (at the root)");
    assert_eq!(err.into_kind(), ErrorKind::TakeOutOfRange { n: 11, len: 10 });
    assert_eq!(Order::new(src(0, 0).cycle(5)).unwrap_err(), root(ErrorKind::EmptyCycle));
    assert_eq!(Order::new(Seq::concat([src(0, 3), src(1, 0).cycle(5)])).unwrap_err(), at(ErrorKind::EmptyCycle, &[1]));
    assert_eq!(Order::new(src(0, 5).take(6).cycle(5)).unwrap_err(), at(ErrorKind::TakeOutOfRange { n: 6, len: 5 }, &[0]));
    assert_eq!(Order::new(src(0, 0).cycle(0)).unwrap().len(), 0);
    assert_eq!(root(ErrorKind::EmptyCycle).to_string(), "cannot cycle a sequence without elements (at the root)");
}

/// A cycle is the repeat cut to length, with the repeats inside one level deeper only when
/// it does repeat; a prefix of a repeat folds into it.
#[test]
fn cycles() {
    use crate::order::Node;
    let x = || src(0, 100).shuffle(3);
    let all = ids(Order::new(x().repeat(4)).unwrap().iter(..).unwrap());
    for len in [0, 1, 99, 100, 101, 250, 400] {
        let order = Order::new(x().cycle(len)).unwrap();
        assert_eq!(order.len(), len);
        assert_eq!(ids(order.iter(..).unwrap()), all[..len], "cycle({len})");
        assert_eq!(x().cycle(len).check(), Ok(len));
        assert_eq!(ids(Order::new(x().repeat(4).take(len)).unwrap().iter(..).unwrap()), all[..len]);
    }
    assert_eq!(ids(Order::new(x().cycle(99)).unwrap().iter(..).unwrap()), ids(Order::new(x().take(99)).unwrap().iter(..).unwrap()));
    // The repeats inside move one level deeper exactly when the cycle repeats.
    let y = || src(0, 10).shuffle(3).repeat(2);
    assert_eq!(ids(Order::new(y().cycle(15)).unwrap().iter(..).unwrap()), ids(Order::new(y()).unwrap().iter(..15).unwrap()));
    assert_eq!(ids(Order::new(y().cycle(45)).unwrap().iter(..).unwrap()), ids(Order::new(y().repeat(3)).unwrap().iter(..45).unwrap()));
    assert_ne!(ids(Order::new(y().cycle(45)).unwrap().iter(..).unwrap())[..20], ids(Order::new(y()).unwrap().iter(..).unwrap())[..]);
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
    assert_eq!(ids(endless.iter(last..).unwrap()), vec![(0, endless.get(last).unwrap().record_index)]);
    assert_eq!(ids(endless.iter(..300).unwrap()), all[..300]);
    // A cycle over a concatenation narrows it like a take, beyond one repetition it counts
    // every part.
    let base = ids(Order::new(src(0, 100).shuffle(1)).unwrap().iter(..).unwrap());
    assert_eq!(ids(Order::new(Seq::concat([src(0, 100), src(1, 50)]).cycle(100).shuffle(1)).unwrap().iter(..).unwrap()), base);
    assert_ne!(ids(Order::new(Seq::concat([src(0, 100), src(1, 50)]).cycle(200).shuffle(1)).unwrap().iter(..).unwrap())[..100], base[..]);
    // A cycled concat part is narrowed to the requested count before mixing.
    let part = || Seq::concat([src(0, 100), src(1, 50)]);
    let narrowed = Order::new(Seq::mix([part().cycle(75), src(2, 75)]).shuffle(1)).unwrap();
    let plain = Order::new(Seq::mix([src(0, 100).take(75), src(2, 75)]).shuffle(1)).unwrap();
    assert_eq!(ids(narrowed.iter(..).unwrap()), ids(plain.iter(..).unwrap()));
}

#[test]
fn depth_limit() {
    let chain = |levels: u32| (1..levels).fold(src(0, 10), |s, _| s.take(10));
    assert_eq!(Order::new(chain(MAX_DEPTH)).unwrap().len(), 10);
    let err = Order::new(chain(MAX_DEPTH + 1)).unwrap_err();
    assert_eq!(err.kind(), &ErrorKind::TooDeep);
    assert_eq!(err.path().len(), MAX_DEPTH as usize);
    assert_eq!(chain(MAX_DEPTH + 1).check().unwrap_err().kind(), &ErrorKind::TooDeep);
}

/// Configurations just beyond the supported limit report the first invalid node.
#[test]
fn configurations_over_the_depth_limit_are_rejected() {
    let run = || {
        let deep = || (0..MAX_DEPTH).fold(src(0, 10), |s, _| s.take(10));
        assert_eq!(Order::new(deep()).unwrap_err().kind(), &ErrorKind::TooDeep);
        let d = deep();
        assert_eq!(d.check().unwrap_err().kind(), &ErrorKind::TooDeep);
        drop(d);
        let bad = || src(0, 10).take(99);
        let out_of_range = ErrorKind::TakeOutOfRange { n: 99, len: 10 };
        assert_eq!(Order::new(Seq::concat([bad(), deep()])).unwrap_err().kind(), &out_of_range);
        assert_eq!(Order::new(Seq::mix([bad(), deep()])).unwrap_err().kind(), &out_of_range);
        assert_eq!(Order::new(deep().stride(0, 0)).unwrap_err().kind(), &ErrorKind::ZeroStep);
        let first_too_deep = Seq::concat([deep(), bad()]);
        assert_eq!(first_too_deep.check().unwrap_err().kind(), &ErrorKind::TooDeep);
        drop(first_too_deep);
    };
    std::thread::Builder::new().stack_size(2 << 20).spawn(run).unwrap().join().unwrap();
}

/// An empty source, or a part that folds away as empty, does not change the shuffle above
/// it: only sources that contribute elements salt it.
#[test]
fn empty_parts_do_not_affect_shuffles_above() {
    let x = || src(0, 100);
    let base = ids(Order::new(x().shuffle(1)).unwrap().iter(..).unwrap());
    let same = |seq: Seq<Src>| assert_eq!(ids(Order::new(seq).unwrap().iter(..).unwrap()), base);
    same(Seq::concat([x(), src(1, 0)]).shuffle(1));
    same(Seq::concat([src(0, 0), x()]).shuffle(1));
    same(Seq::concat([src(1, 0), x(), src(2, 7).repeat(0), src(3, 7).take(0), src(4, 3).skip(3), src(5, 2).stride(1, 2)]).shuffle(1));
    same(Seq::mix([x(), src(1, 0), Seq::concat([src(2, 5).skip(5), src(3, 0)])]).shuffle(1));
    same(Seq::mix([x()]).shuffle(1));
    same(x().stride(1, 0).shuffle(1));
    // A part of a concatenation that a skip or take cuts away entirely does not count
    // either, however the concatenation nests; a part it touches counts.
    same(Seq::concat([x(), src(1, 50)]).take(100).shuffle(1));
    same(Seq::concat([src(1, 50), x()]).skip(50).shuffle(1));
    same(Seq::concat([src(1, 50), x(), src(2, 7)]).skip(50).take(100).shuffle(1));
    same(Seq::concat([src(1, 50), x(), src(2, 7)]).slice(50..150).unwrap().shuffle(1));
    same(Seq::concat([Seq::concat([src(1, 5), x()]), src(2, 3)]).skip(5).take(100).shuffle(1));
    same(Seq::concat([src(1, 5), Seq::concat([x(), src(2, 3)])]).take(105).skip(5).shuffle(1));
    let with = |extra: Seq<Src>| ids(Order::new(Seq::concat([x(), extra]).take(101).shuffle(1)).unwrap().iter(..).unwrap());
    assert_ne!(with(src(1, 1)), base);
    assert_ne!(with(src(0, 1)), base);
    // A stride does not cut parts away, even ones it never reaches.
    let strided = |seq: Seq<Src>| ids(Order::new(seq).unwrap().iter(..).unwrap());
    assert_eq!(strided(Seq::concat([src(1, 1), x()]).stride(2, 1)), strided(x().stride(2, 0)));
    assert_ne!(strided(Seq::concat([src(1, 1), x()]).stride(2, 1).shuffle(1)), strided(x().stride(2, 0).shuffle(1)));
}

/// The folds: what compiles to a single node.
#[test]
fn folds() {
    use crate::order::Node;
    let root = |seq: Seq<Src>| Order::new(seq).unwrap().root;
    let a = || src(0, 100);
    assert!(
        matches!(root(a().shard(8, 1).unwrap().shard(4, 1).unwrap()), Node::Stride { step: 32, offset: 9, len: 3, ref child } if matches!(**child, Node::Source { .. }))
    );
    assert!(matches!(root(a().shard(8, 1).unwrap().shard(4, 1).unwrap().shard(2, 1).unwrap()), Node::Source { offset: 41, len: 1, .. }));
    assert!(
        matches!(root(a().shuffle(1).skip(10).stride(3, 2)), Node::Stride { step: 3, offset: 12, len: 30, ref child } if matches!(**child, Node::Shuffle { .. }))
    );
    assert!(matches!(root(a().shuffle(1).skip(10).take(50).skip(5)), Node::Slice { start: 15, len: 45, .. }));
    assert!(
        matches!(root(a().stride(4, 1).stride(2, 1)), Node::Stride { step: 8, offset: 5, len: 12, ref child } if matches!(**child, Node::Source { .. }))
    );
    let cat = || Seq::concat([src(1, 10), src(2, 20), src(3, 30)]);
    assert!(matches!(root(cat().take(10)), Node::Source { src: 0, offset: 0, len: 10 }));
    assert!(matches!(root(cat().skip(15)), Node::Concat { ref offsets, ref children } if offsets == &[0, 15, 45] && children.len() == 2));
    assert!(matches!(root(cat().skip(10).take(20)), Node::Source { src: 1, offset: 0, len: 20 }));
    assert!(matches!(root(cat().skip(10)), Node::Concat { ref children, .. } if children.len() == 2));
    assert!(
        matches!(root(cat().slice(5..35).unwrap()), Node::Concat { ref offsets, ref children } if offsets == &[0, 5, 25, 30] && children.len() == 3)
    );
    assert_eq!(
        ids(Order::new(cat().skip(15).stride(7, 3)).unwrap().iter(..).unwrap()),
        ids(Order::new(cat()).unwrap().iter(..).unwrap()).into_iter().skip(18).step_by(7).collect::<Vec<_>>()
    );
}

/// A mix's order does not depend on empty parts, wherever they sit.
#[test]
fn empty_mix_parts_do_not_affect_the_order() {
    let mut rng = Rng(0x0E0E_0E0E_1234_5678);
    for round in 0..200 {
        let k = 2 + rng.below(6);
        let parts: Vec<(Seq<Src>, Sampling)> = (0..k)
            .map(|i| {
                let s = match rng.below(4) {
                    0 => Sampling::delayed(0.3 + 0.1 * rng.below(5) as f64),
                    1 => Sampling::ramp(0.1, 0.6),
                    _ => Sampling::Uniform,
                };
                (src(i as u32, 1 + rng.below(80)), s)
            })
            .collect();
        let mut with = parts.clone();
        for _ in 0..1 + rng.below(3) {
            let at = rng.below(with.len() + 1);
            with.insert(at, (src(99, 0), Sampling::delayed(0.9)));
        }
        let (Ok(a), Ok(b)) = (Order::new(Seq::mix_with(with)), Order::new(Seq::mix_with(parts))) else { continue };
        assert_eq!(ids(a.iter(..).unwrap()), ids(b.iter(..).unwrap()), "round {round}");
        assert_eq!(a.sources().len(), b.sources().len() + a.sources().iter().filter(|s| s.id == 99).count());
    }
}

/// `Seq` is `Eq` and `Hash` by comparing floats bitwise, with the two zeros equal.
#[test]
fn seq_eq_and_hash() {
    use std::collections::HashSet;
    let a = Seq::mix_with([(src(0, 5), Sampling::ramp(0.0, 0.5))]);
    let b = Seq::mix_with([(src(0, 5), Sampling::ramp(-0.0, 0.5))]);
    let c = Seq::mix_with([(src(0, 5), Sampling::ramp(0.1, 0.5))]);
    assert_eq!(a, b);
    assert_ne!(a, c);
    let set: HashSet<Seq<Src>> = [a.clone(), b, c.clone(), a.clone()].into_iter().collect();
    assert_eq!(set.len(), 2);
    assert!(set.contains(&a) && set.contains(&c));
    let nan = Seq::mix_with([(src(0, 5), Sampling::delayed(f64::NAN))]);
    assert_eq!(nan, nan.clone());
    assert_eq!(Sampling::delayed(0.5), Sampling::ramp(0.5, 0.5));
    assert_eq!(Sampling::until(0.5), Sampling::fading(0.5, 0.5));
    assert_ne!(Sampling::until(0.5), Sampling::delayed(0.5));
    assert_ne!(Sampling::trapezoid(0.0, 0.0, 1.0, 1.0), Sampling::ramp(0.0, 0.0));
    assert_eq!(Sampling::trapezoid(-0.0, 0.1, 0.5, 0.9), Sampling::trapezoid(0.0, 0.1, 0.5, 0.9));
    assert_eq!(MixPart::from(src(1, 2)), MixPart { seq: src(1, 2), sampling: Sampling::Uniform });
}

#[test]
fn edge_cases() {
    let empty = Order::new(Seq::<Src>::concat([])).unwrap();
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.iter(0..0).unwrap().count(), 0);
    let empty = Order::new(src(1, 5).shuffle(1).take(0).repeat(4)).unwrap();
    assert!(empty.is_empty());
    assert_eq!(Order::new(src(1, 5).stride(3, 9)).unwrap().len(), 0);
    let one = Order::new(src(2, 1).shuffle(5).repeat(3)).unwrap();
    assert_eq!(ids(one.iter(0..3).unwrap()), vec![(2, 0); 3]);
    let order = Order::with_seed(src(0, 5), 77).unwrap();
    assert_eq!(order.sources(), &[Src { id: 0, len: 5 }]);
    assert_eq!(order.seed(), 77);
    assert_eq!(format!("{order:?}"), "Order { len: 5, seed: 77, sources: [Src { id: 0, len: 5 }] }");
    let mut c = order.iter(1..4).unwrap();
    assert_eq!(format!("{c:?}"), "Cursor { position: 1, end: 4 }");
    assert_eq!(c.remaining(), 3);
    assert_eq!(c.next().map(|item| (item.source.id, item.record_index)), Some((0, 1)));
    c.seek(4).unwrap();
    assert!(c.next().is_none());
    assert_eq!(c.nth(3), None);
    c.seek(0).unwrap();
    assert_eq!(c.nth(2).map(|item| (item.source.id, item.record_index)), Some((0, 2)));
    assert_eq!(c.offset(), 3);
    assert_eq!(c.nth(1), None);
    assert_eq!(c.offset(), 4);
    c.seek(0).unwrap();
    assert_eq!(ids(c), vec![(0, 0), (0, 1), (0, 2), (0, 3)]);
    // `count` and `last` answer without walking, `last` by random access.
    let shuffled = Order::new(src(0, 1000).shuffle(3)).unwrap();
    assert_eq!(shuffled.iter(10..).unwrap().count(), 990);
    assert_eq!(
        shuffled.iter(10..900).unwrap().last().map(|item| (item.source.id, item.record_index)),
        Some((0, shuffled.get(899).unwrap().record_index))
    );
    assert_eq!(shuffled.iter(7..7).unwrap().last(), None);
    let mut c = shuffled.iter(..).unwrap();
    c.nth(4);
    assert_eq!(c.count(), 995);
    // Range forms of `iter`.
    assert_eq!(ids(order.iter(..).unwrap()), ids(order.iter(0..5).unwrap()));
    assert_eq!(ids(order.iter(3..).unwrap()), vec![(0, 3), (0, 4)]);
    assert_eq!(ids(order.iter(..=1).unwrap()), vec![(0, 0), (0, 1)]);
    assert_eq!(ids(order.iter(2..=2).unwrap()), vec![(0, 2)]);
    assert_eq!(order.iter(5..).unwrap().count(), 0);
    assert_eq!(ids(order.iter((std::ops::Bound::Excluded(3), std::ops::Bound::Unbounded)).unwrap()), vec![(0, 4)]);
    assert_eq!(ids((&order).into_iter()), ids(order.iter(..).unwrap()));
    // A cursor run past its end, or ranged at the end, still moves forward correctly.
    let mut c = order.iter(1..3).unwrap();
    assert_eq!(c.nth(10), None);
    c.set_range(2..).unwrap();
    assert_eq!(ids(c.by_ref()), vec![(0, 2), (0, 3), (0, 4)]);
    c.set_range(..1).unwrap();
    assert_eq!(ids(c.by_ref()), vec![(0, 0)]);
    c.seek(1).unwrap();
    c.set_range(3..4).unwrap();
    assert_eq!(ids(c.by_ref()), vec![(0, 3)]);
    let mut at_end = order.iter(5..).unwrap();
    assert_eq!(at_end.next(), None);
    at_end.set_range(4..).unwrap();
    assert_eq!(ids(at_end), vec![(0, 4)]);
    assert_eq!(order.into_sources(), vec![Src { id: 0, len: 5 }]);
    // Bare lengths are sources; a source may be shared through a reference.
    let lens = [5usize, 3];
    let shared = Order::new(Seq::concat([Seq::source(&lens[0]), Seq::source(&lens[1]), Seq::source(&lens[0]).shuffle(1)])).unwrap();
    assert_eq!(shared.len(), 13);
    assert_eq!(shared.sources().len(), 3);
    // Which source an element came from, also when sources compare equal.
    let which: Vec<usize> = shared.iter(..).unwrap().map(|item| item.source_ordinal).collect();
    assert_eq!(which, [vec![0; 5], vec![1; 3], vec![2; 5]].concat());
    let equal = Order::new(Seq::mix([src(1, 4), src(1, 4), src(1, 2)])).unwrap();
    let which: Vec<usize> = equal.iter(..).unwrap().map(|item| item.source_ordinal).collect();
    assert_eq!(which, [0, 1, 0, 1, 2, 0, 1, 0, 1, 2]);
    struct Unit;
    impl Source for Unit {
        fn len(&self) -> usize {
            2
        }
    }
    let units = Order::new(Seq::concat([Seq::source(Unit), Seq::source(Unit)])).unwrap();
    assert_eq!(units.iter(..).unwrap().map(|item| item.source_ordinal).collect::<Vec<_>>(), [0, 0, 1, 1]);
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
    assert_eq!(ids(order.iter(last..).unwrap()), vec![(0, i)]);
    let v = ids(order.iter(1 << 59..(1 << 59) + 5).unwrap());
    for (k, &e) in v.iter().enumerate() {
        let crate::Item { source: s, record_index: i, .. } = order.get((1 << 59) + k).unwrap();
        assert_eq!((s.id, i), e);
    }
}

/// An order longer than the address space is rejected, with the length; an intermediate node
/// may exceed it.
#[test]
fn orders_longer_than_usize_are_rejected() {
    let err = Order::new(Seq::concat([src(0, usize::MAX), src(1, 1)])).unwrap_err();
    if cfg!(target_pointer_width = "64") {
        assert_eq!(err, root(ErrorKind::LengthOverflow));
    } else {
        assert_eq!(err, root(ErrorKind::OrderTooLong { len: usize::MAX as u64 + 1 }));
        let intermediate = Seq::concat([src(0, usize::MAX), src(1, 1)]).take(10);
        assert_eq!(Order::new(intermediate).unwrap().len(), 10);
        let err = Order::new(Seq::concat([src(0, usize::MAX), src(1, 1)]).skip(usize::MAX).skip(3)).unwrap_err();
        assert_eq!(err.kind(), &ErrorKind::SkipOutOfRange { n: 3, len: 1 });
    }
}

#[test]
fn cloned_cursor_continues_independently() {
    let order = Order::new(Seq::mix([src(0, 500).shuffle(1).repeat(2), src(1, 300).shuffle(2)]).shard(3, 1).unwrap()).unwrap();
    let n = order.len();
    let mut c = order.iter(0..n).unwrap();
    let head = ids(c.by_ref().take(100));
    let d = c.clone();
    let rest_c = ids(c);
    let rest_d = ids(d);
    assert_eq!(rest_c, rest_d);
    assert_eq!([head, rest_c].concat(), ids(order.iter(0..n).unwrap()));
}

#[test]
fn types_are_send_and_sync() {
    fn assert_send_sync<X: Send + Sync>() {}
    assert_send_sync::<Seq<usize>>();
    assert_send_sync::<Order<usize>>();
    assert_send_sync::<Cursor<'static, usize>>();
    assert_send_sync::<Error>();
    assert_send_sync::<ErrorKind>();
    assert_send_sync::<Sampling>();
    assert_send_sync::<MixPart<usize>>();
}

/// Inclusive and open bounds in `slice`.
#[test]
fn slice_bounds() {
    let a = || src(0, 10);
    assert_eq!(ids(Order::new(a().slice(..=2).unwrap()).unwrap().iter(0..3).unwrap()), vec![(0, 0), (0, 1), (0, 2)]);
    assert_eq!(ids(Order::new(a().slice(8..).unwrap()).unwrap().iter(0..2).unwrap()), vec![(0, 8), (0, 9)]);
    assert_eq!(ids(Order::new(a().slice(3..=3).unwrap()).unwrap().iter(0..1).unwrap()), vec![(0, 3)]);
    assert_eq!(Order::new(a().slice(..).unwrap()).unwrap().len(), 10);
    assert_eq!(Order::new(a().slice(4..4).unwrap()).unwrap().len(), 0);
}

/// Schedules at the steep end of what a mix accepts, at lengths near its limit: seeks and
/// walks agree, so the interleave's count-and-fix loops settle there too.
#[test]
#[cfg(target_pointer_width = "64")]
fn steep_schedule_at_scale() {
    let seq = Seq::mix_with([
        (src(0, 1 << 45), Sampling::Uniform),
        (src(1, 1 << 44), Sampling::delayed(0.5)), // final rate 2: length × rate = 2⁴⁵, within 2⁴⁶
        (src(2, 1 << 40), Sampling::ramp(0.0, 1.0)),
    ]);
    let order = Order::new(seq).unwrap();
    let n = order.len();
    assert_eq!(n, (1 << 45) + (1 << 44) + (1 << 40));
    for start in [0, n / 2 - 777, n - 1500, 12_345_678_901] {
        let walked = ids(order.iter(start..start + 1500).unwrap());
        for (k, &e) in walked.iter().enumerate() {
            let crate::Item { source: s, record_index: i, .. } = order.get(start + k).unwrap();
            assert_eq!((s.id, i), e, "position {}", start + k);
        }
    }
    // Too steep is rejected, not looped over.
    let steep = Seq::mix_with([(src(0, 1 << 45), Sampling::Uniform), (src(1, 1 << 46), Sampling::delayed(0.5))]);
    assert!(matches!(Order::new(steep).unwrap_err().kind(), ErrorKind::TooSteep | ErrorKind::MixTooLong));
    assert_eq!(MAX_MIX_LEN, 1 << 46);
}

/// Explicit counts repeat short parts (reshuffled) and truncate long ones before mixing.
#[test]
fn mix_with_explicit_counts() {
    let seq = Seq::mix([src(0, 100).shuffle(1).cycle(1800), src(1, 5000).shuffle(2).cycle(1200)]);
    assert_eq!(seq.check(), Ok(3000));
    let order = Order::new(seq).unwrap();
    let all = ids(order.iter(0..3000).unwrap());
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
