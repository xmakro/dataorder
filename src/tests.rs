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

/// A test source: an id to compare orders by, and a length.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Src {
    id: u32,
    len: usize,
}

impl Source for Src {
    fn len(&self) -> usize {
        self.len
    }
}

fn src(id: u32, len: usize) -> Seq<Src> {
    Seq::source(Src { id, len })
}

/// Elements as `(id, index)`.
fn ids<'a>(it: impl Iterator<Item = (&'a Src, usize)>) -> Vec<(u32, usize)> {
    it.map(|(s, i)| (s.id, i)).collect()
}

impl Error {
    /// A schedule or length rejection of a mix.
    fn is_sampling(&self) -> bool {
        matches!(
            self.kind(),
            ErrorKind::MixTooLong
                | ErrorKind::InvalidSampling { .. }
                | ErrorKind::TooSteep { .. }
                | ErrorKind::Overcommitted { .. }
                | ErrorKind::ZeroWeights
                | ErrorKind::EmptyWeightedPart { .. }
        )
    }
}

/// An error at the root, for comparisons.
fn root(kind: ErrorKind) -> Error {
    Error::new(kind, Vec::new())
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
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
            let il = Interleave::with_sampling(&lens, &sampling).map_err(|e| root(e.into()))?;
            il.iter(0..il.len()).map(|(s, j)| evs[s][j as usize]).collect()
        }
        Seq::Weighted { total, parts } => {
            let weights: Vec<f64> = parts.iter().map(|p| p.weight).collect();
            let shares = crate::order::weighted_shares(*total as u64, &weights).map_err(root)?;
            let mut mixed = Vec::new();
            for (i, (part, share)) in parts.iter().zip(shares).enumerate() {
                let len = eval_at(&part.seq, ctx, depth)?.len();
                if len == 0 && share > 0 {
                    return Err(root(ErrorKind::EmptyWeightedPart { part: i }));
                }
                let times = if len == 0 { 1 } else { (share as usize).div_ceil(len).max(1) };
                let inner = if times > 1 { Seq::Repeat { times, inner: Box::new(part.seq.clone()) } } else { part.seq.clone() };
                mixed.push((Seq::Take { n: share as usize, inner: Box::new(inner) }, part.sampling));
            }
            eval_at(&Seq::mix_with(mixed), ctx, depth)?
        }
        Seq::Shuffle { seed, inner } => {
            let v = eval_at(inner, ctx, depth)?;
            let (shape, key) = (Shape::new(v.len() as u64), perm::key(*seed, ctx));
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

fn random_seq(rng: &mut Rng, depth: u32, lens: &[usize]) -> Seq<Src> {
    if depth == 0 || rng.below(5) == 0 {
        let id = rng.below(lens.len());
        return src(id as u32, lens[id]);
    }
    let parts = |rng: &mut Rng, depth| (0..1 + rng.below(3)).map(|_| random_seq(rng, depth, lens)).collect::<Vec<_>>();
    match rng.below(9) {
        8 => {
            // Empty parts in a mix, which must not affect the order (checked separately) and
            // must not break the walk.
            let mut ps = parts(rng, depth - 1);
            ps.insert(rng.below(ps.len() + 1), src(0, 0));
            Seq::mix(ps)
        }
        7 => {
            let total = rng.below(60);
            let weighted: Vec<(Seq<Src>, f64, Sampling)> = parts(rng, depth - 1)
                .into_iter()
                .map(|p| {
                    let w = (1 + rng.below(3)) as f64;
                    let s = if rng.below(3) == 0 { Sampling::delayed(0.5) } else { Sampling::Uniform };
                    (p, w, s)
                })
                .collect();
            Seq::weighted_with(total, weighted)
        }
        0 => Seq::concat(parts(rng, depth - 1)),
        1 => Seq::mix(parts(rng, depth - 1)),
        2 => Seq::mix_with(parts(rng, depth - 1).into_iter().map(|p| {
            let sampling = match rng.below(4) {
                0 => Sampling::DelayedLinear { start: 0.3, full: 0.6 },
                1 => Sampling::DelayedLinear { start: 0.5, full: 0.5 },
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
                _ => inner.slice(start..start + rng.below(n - start + 1)),
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
                assert!(!e.path().is_empty() || matches!(seq, Seq::Mix(_) | Seq::Weighted { .. }), "round {round}: {e}");
                skipped += 1;
                continue;
            }
            Err(e) => panic!("round {round}: {e} for {seq:?}"),
        };
        let reference = eval(&seq, seed).unwrap();
        let n = reference.len();
        assert_eq!(order.len(), n, "round {round}: {seq:?}");
        for (i, &r) in reference.iter().enumerate() {
            let (s, idx) = order.get(i);
            assert_eq!((s.id, idx), r, "round {round}: get({i}) of {seq:?}");
        }
        assert_eq!(ids(order.iter(0..n)), reference, "round {round}: {seq:?}");
        for _ in 0..4 {
            let a = rng.below(n + 1);
            let b = a + rng.below(n - a + 1);
            assert_eq!(ids(order.iter(a..b)), reference[a..b], "round {round}: {a}..{b} of {seq:?}");
        }
        let mut cursor = order.iter(0..n);
        for _ in 0..6 {
            let a = rng.below(n + 1);
            cursor.seek(a);
            assert_eq!(cursor.position(), a);
            let m = rng.below(n - a + 1);
            assert_eq!(cursor.len(), n - a);
            let got = ids(cursor.by_ref().take(m));
            assert_eq!(got, reference[a..a + m], "round {round}: seek {a} of {seq:?}");
            // A forward seek and `nth` skip; both must land where a fresh cursor would.
            let b = a + m + rng.below(n - a - m + 1);
            cursor.seek(b);
            assert_eq!(ids(cursor.by_ref().take(3)), reference[b..(b + 3).min(n)], "round {round}: forward seek {b} of {seq:?}");
            let p = cursor.position();
            let k = rng.below(5);
            assert_eq!(cursor.nth(k).map(|(s, i)| (s.id, i)), reference.get(p + k).copied(), "round {round}: nth({k}) at {p} of {seq:?}");
            assert_eq!(cursor.position(), (p + k + 1).min(n));
        }
        assert_eq!(ids((&order).into_iter()), reference);
        checked += 1;
    }
    assert!(checked > 400, "only {checked} configurations checked ({skipped} overcommitted)");
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
        assert_eq!(ids(order.iter(0..n)), reference, "round {round}: {seq:?}");
        for _ in 0..8 {
            let a = rng.below(n + 1);
            let b = (a + rng.below(500)).min(n);
            assert_eq!(ids(order.iter(a..b)), reference[a..b], "round {round}: {a}..{b}");
            let p = a.min(n - 1);
            let (s, i) = order.get(p);
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
                .map(|(s, i)| {
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
    assert_eq!(once.iter(0..1000).map(|(_, i)| i).collect::<Vec<_>>(), epochs[0]);
    assert_eq!(ids(Order::new(src(7, 1000).shuffle(3).repeat(1)).unwrap().iter(0..1000)), ids(once.iter(0..1000)));
    assert_eq!(ids(Order::new(seq.clone().repeat(1)).unwrap().iter(..)), ids(order.iter(..)));
    assert_eq!(ids(Order::new(Seq::concat([seq.clone().repeat(1)]).repeat(1)).unwrap().iter(..)), ids(order.iter(..)));
    // A weighted part that fits its share once is not repeated, so it is the part itself;
    // one that is repeated starts with the part (the first repetition keeps its context) and
    // then, being one repeat deeper, reshuffles the part's own epochs differently.
    let part = || src(7, 100).shuffle(3).repeat(2);
    let fits = Order::new(Seq::weighted(400, [(part(), 1.0), (src(8, 1000), 1.0)])).unwrap();
    let repeats = Order::new(Seq::weighted(500, [(part(), 1.0), (src(8, 1000), 1.0)])).unwrap();
    let sevens = |o: &Order<Src>| ids(o.iter(..)).into_iter().filter(|e| e.0 == 7).collect::<Vec<_>>();
    let alone = ids(Order::new(part()).unwrap().iter(..));
    assert_eq!(sevens(&fits), alone);
    assert_eq!(sevens(&repeats)[..100], alone[..100]);
    assert_ne!(sevens(&repeats)[100..200], alone[100..200]);
    // Nested repeats: (outer 0, inner 1) and (outer 1, inner 0) are different orders.
    let nested = Order::new(src(7, 100).shuffle(3).repeat(2).repeat(2)).unwrap();
    let block = |b: usize| ids(nested.iter(b * 100..(b + 1) * 100));
    assert_ne!(block(1), block(2));
    assert_eq!(block(0), ids(Order::new(src(7, 100).shuffle(3)).unwrap().iter(0..100)));
    // Same seed twice under a concat: the same order twice.
    let twice = Order::new(Seq::concat([src(7, 1000).shuffle(3), src(7, 1000).shuffle(3)])).unwrap();
    let v = ids(twice.iter(0..2000));
    assert_eq!(v[..1000], v[1000..]);
    // The order's seed changes every shuffle.
    let reseeded = Order::with_seed(seq, 99).unwrap();
    assert_ne!(ids(reseeded.iter(0..1000)), ids(order.iter(0..1000)));
}

#[test]
fn shards_partition_the_sequence() {
    let base = Seq::mix([src(0, 1000).shuffle(1), src(1, 300).shuffle(2)]).repeat(2);
    let order = Order::new(base.clone()).unwrap();
    let all = ids(order.iter(0..order.len()));
    let mut from_shards = Vec::new();
    for w in 0..8 {
        let shard = Order::new(base.clone().shard(8, w)).unwrap();
        let elems = ids(shard.iter(0..shard.len()));
        for (i, &e) in elems.iter().enumerate() {
            assert_eq!(e, all[w + 8 * i]);
        }
        from_shards.extend(elems);
    }
    assert_eq!(from_shards.len(), all.len());
}

/// `map` keeps the structure: the same indices come out over the mapped sources.
#[test]
fn map_keeps_the_order() {
    let seq = Seq::mix([src(0, 700).shuffle(1).repeat(2), Seq::concat([src(1, 50), src(2, 120).shuffle(2)])]).shard(3, 1);
    let order = Order::new(seq.clone()).unwrap();
    let mapped = Order::new(seq.map(|s| s.len)).unwrap();
    assert_eq!(mapped.sources(), &[700, 50, 120]);
    let a: Vec<(usize, usize)> = order.iter(0..order.len()).map(|(s, i)| (s.len, i)).collect();
    let b: Vec<(usize, usize)> = mapped.iter(0..mapped.len()).map(|(&l, i)| (l, i)).collect();
    assert_eq!(a, b);
}

#[test]
fn errors() {
    let a = src(0, 10);
    assert_eq!(Order::new(a.clone().slice(3..12)).unwrap_err(), root(ErrorKind::TakeOutOfRange { n: 9, len: 7 }));
    assert_eq!(Order::new(a.clone().skip(11)).unwrap_err(), root(ErrorKind::SkipOutOfRange { n: 11, len: 10 }));
    assert_eq!(Order::new(a.clone().take(11)).unwrap_err(), root(ErrorKind::TakeOutOfRange { n: 11, len: 10 }));
    assert_eq!(Order::new(a.clone().slice(2..=9)).unwrap().len(), 8);
    assert_eq!(Order::new(a.clone().slice(10..)).unwrap().len(), 0);
    assert_eq!(Order::new(a.clone().stride(0, 0)).unwrap_err(), root(ErrorKind::ZeroStep));
    // Beyond 64 bits on every target: a repeat of a repeat, and a concat of two halves of 2⁶⁴.
    assert_eq!(Order::new(a.clone().repeat(usize::MAX).repeat(usize::MAX)).unwrap_err().kind(), &ErrorKind::LengthOverflow);
    let half = || src(0, 1 << 31).repeat(1 << 31).repeat(2);
    assert_eq!(Order::new(Seq::concat([half(), half()])).unwrap_err(), root(ErrorKind::LengthOverflow));
    let over = Seq::mix_with([(src(0, 10), Sampling::DelayedLinear { start: 0.5, full: 0.5 }), (src(1, 1), Sampling::Uniform)]);
    assert!(matches!(Order::new(over).unwrap_err().kind(), ErrorKind::Overcommitted { .. }));
    // A mix that folds away is still validated.
    let over1 = Seq::mix_with([(src(0, 10), Sampling::DelayedLinear { start: 2.0, full: 2.0 })]);
    assert_eq!(
        Order::new(over1).unwrap_err(),
        root(ErrorKind::InvalidSampling { part: 0, sampling: Sampling::DelayedLinear { start: 2.0, full: 2.0 } })
    );
    // The path leads to the node: part 1 of the mix, then the single child of the shuffle.
    let nested = Seq::mix([a.clone(), Seq::concat([a.clone(), a.clone().take(11).shuffle(1)])]).repeat(2);
    let err = Order::new(nested).unwrap_err();
    assert_eq!(err.kind(), &ErrorKind::TakeOutOfRange { n: 11, len: 10 });
    assert_eq!(err.path(), [0, 1, 1, 0]);
    assert_eq!(err.to_string(), "cannot take 11 of 10 positions (at node/0/1/1/0)");
    assert_eq!(root(ErrorKind::ZeroStep).to_string(), "stride step is zero (at the root)");
    assert_eq!(err.clone().into_kind(), ErrorKind::TakeOutOfRange { n: 11, len: 10 });
    // Weights: reported at the weighted node with the part index, before the parts are compiled.
    let w = Seq::concat([a.clone(), Seq::weighted(10, [(a.clone(), 1.0), (a.clone().take(99), -1.0)])]);
    let err = Order::new(w).unwrap_err();
    assert_eq!((err.kind(), err.path()), (&ErrorKind::InvalidWeight { part: 1, weight: -1.0 }, &[1][..]));
    assert_eq!(Order::new(Seq::<Src>::weighted(5, [])).unwrap_err(), root(ErrorKind::ZeroWeights));
}

#[test]
fn depth_limit() {
    let chain = |levels: u32| (1..levels).fold(src(0, 10), |s, _| s.take(10));
    assert_eq!(Order::new(chain(Seq::<Src>::MAX_DEPTH)).unwrap().len(), 10);
    let err = Order::new(chain(Seq::<Src>::MAX_DEPTH + 1)).unwrap_err();
    assert_eq!(err.kind(), &ErrorKind::TooDeep);
    assert_eq!(err.path().len(), Seq::<Src>::MAX_DEPTH as usize);
    assert_eq!(chain(Seq::<Src>::MAX_DEPTH + 1).check().unwrap_err().kind(), &ErrorKind::TooDeep);
}

#[test]
#[should_panic(expected = "shard index 4 out of range for 4 shards")]
fn shard_index_out_of_range_panics() {
    let _ = src(0, 10).shard(4, 4);
}

#[test]
#[should_panic(expected = "position 5 out of range")]
fn get_out_of_range_panics() {
    let _ = Order::new(src(0, 5)).unwrap().get(5);
}

#[test]
#[should_panic(expected = "ends before it starts")]
#[allow(clippy::reversed_empty_ranges)]
fn reversed_iter_range_panics() {
    let _ = Order::new(src(0, 5)).unwrap().iter(3..2);
}

#[test]
#[should_panic(expected = "range end 6 out of range for 5 positions")]
fn iter_past_the_end_panics() {
    let _ = Order::new(src(0, 5)).unwrap().iter(..6);
}

#[test]
#[should_panic(expected = "seek to 4 beyond the end 3")]
fn seek_past_the_end_panics() {
    Order::new(src(0, 5)).unwrap().iter(..3).seek(4);
}

#[test]
#[should_panic(expected = "slice bound overflows usize")]
fn slice_bound_overflow_panics() {
    let _ = src(0, 5).slice(..=usize::MAX);
}

/// A mix's order does not depend on empty parts, wherever they sit, nor a weighted mix's on
/// parts without a share.
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
        assert_eq!(ids(a.iter(..)), ids(b.iter(..)), "round {round}");
        assert_eq!(a.sources().len(), b.sources().len() + a.sources().iter().filter(|s| s.id == 99).count());
    }
    let with = Order::new(Seq::weighted(80, [(src(0, 30), 1.0), (src(9, 5), 0.0), (src(1, 50), 2.0)])).unwrap();
    let without = Order::new(Seq::weighted(80, [(src(0, 30), 1.0), (src(1, 50), 2.0)])).unwrap();
    assert_eq!(ids(with.iter(..)), ids(without.iter(..)));
}

/// `Seq` is `Eq` and `Hash` by comparing floats bitwise, with the two zeros equal.
#[test]
fn seq_eq_and_hash() {
    use std::collections::HashSet;
    let a = Seq::weighted_with(10, [(src(0, 5), 0.0, Sampling::ramp(0.0, 0.5))]);
    let b = Seq::weighted_with(10, [(src(0, 5), -0.0, Sampling::ramp(-0.0, 0.5))]);
    let c = Seq::weighted_with(10, [(src(0, 5), 1.0, Sampling::ramp(0.0, 0.5))]);
    assert_eq!(a, b);
    assert_ne!(a, c);
    let set: HashSet<Seq<Src>> = [a.clone(), b, c.clone(), a.clone()].into_iter().collect();
    assert_eq!(set.len(), 2);
    assert!(set.contains(&a) && set.contains(&c));
    let nan = Seq::weighted(10, [(src(0, 5), f64::NAN)]);
    assert_eq!(nan, nan.clone());
    assert_eq!(Sampling::delayed(0.5), Sampling::ramp(0.5, 0.5));
    assert_eq!(MixPart::from(src(1, 2)), MixPart { seq: src(1, 2), sampling: Sampling::Uniform });
    assert_eq!(WeightedPart::from((src(1, 2), 2.0)), WeightedPart { seq: src(1, 2), weight: 2.0, sampling: Sampling::Uniform });
}

#[test]
#[should_panic(expected = "slice end before start")]
#[allow(clippy::reversed_empty_ranges)]
fn reversed_slice_panics() {
    let _ = src(0, 10).slice(5..3);
}

#[test]
fn edge_cases() {
    let empty = Order::new(Seq::<Src>::concat([])).unwrap();
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.iter(0..0).count(), 0);
    let empty = Order::new(src(1, 5).shuffle(1).take(0).repeat(4)).unwrap();
    assert!(empty.is_empty());
    assert_eq!(Order::new(src(1, 5).stride(3, 9)).unwrap().len(), 0);
    let one = Order::new(src(2, 1).shuffle(5).repeat(3)).unwrap();
    assert_eq!(ids(one.iter(0..3)), vec![(2, 0); 3]);
    let order = Order::with_seed(src(0, 5), 77).unwrap();
    assert_eq!(order.sources(), &[Src { id: 0, len: 5 }]);
    assert_eq!(order.seed(), 77);
    let mut c = order.iter(1..4);
    assert_eq!(c.remaining(), 3);
    assert_eq!(c.next().map(|(s, i)| (s.id, i)), Some((0, 1)));
    c.seek(4);
    assert!(c.next().is_none());
    assert_eq!(c.nth(3), None);
    c.seek(0);
    assert_eq!(c.nth(2).map(|(s, i)| (s.id, i)), Some((0, 2)));
    assert_eq!(c.position(), 3);
    assert_eq!(c.nth(1), None);
    assert_eq!(c.position(), 4);
    c.seek(0);
    assert_eq!(ids(c), vec![(0, 0), (0, 1), (0, 2), (0, 3)]);
    // Range forms of `iter`.
    assert_eq!(ids(order.iter(..)), ids(order.iter(0..5)));
    assert_eq!(ids(order.iter(3..)), vec![(0, 3), (0, 4)]);
    assert_eq!(ids(order.iter(..=1)), vec![(0, 0), (0, 1)]);
    assert_eq!(ids(order.iter(2..=2)), vec![(0, 2)]);
    assert_eq!(order.iter(5..).count(), 0);
    assert_eq!(ids(order.iter((std::ops::Bound::Excluded(3), std::ops::Bound::Unbounded))), vec![(0, 4)]);
    assert_eq!(ids((&order).into_iter()), ids(order.iter(..)));
    assert_eq!(order.into_sources(), vec![Src { id: 0, len: 5 }]);
    // Bare lengths are sources; a source may be shared through a reference.
    let lens = [5usize, 3];
    let shared = Order::new(Seq::concat([Seq::source(&lens[0]), Seq::source(&lens[1]), Seq::source(&lens[0]).shuffle(1)])).unwrap();
    assert_eq!(shared.len(), 13);
    assert_eq!(shared.sources().len(), 3);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn huge_lengths() {
    // Lengths far beyond anything materializable: positions must still resolve.
    let seq = src(0, 1 << 40).shuffle(1).repeat(1 << 20).skip(12345);
    let order = Order::new(seq).unwrap();
    assert_eq!(order.len(), (1usize << 60) - 12345);
    let last = order.len() - 1;
    let (s, i) = order.get(last);
    assert_eq!(s.id, 0);
    assert!(i < 1 << 40);
    assert_eq!(ids(order.iter(last..)), vec![(0, i)]);
    let v = ids(order.iter(1 << 59..(1 << 59) + 5));
    for (k, &e) in v.iter().enumerate() {
        let (s, i) = order.get((1 << 59) + k);
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

/// With the `serde` feature a configuration survives a round trip and compiles to the same order.
#[cfg(feature = "serde")]
#[test]
fn serde_round_trip() {
    let seq: Seq<usize> = Seq::mix_with([
        (Seq::source(1000).shuffle(1).repeat(2), Sampling::Uniform),
        (Seq::concat([Seq::source(300).skip(10), Seq::source(50).take(20)]).shuffle(2), Sampling::ramp(0.2, 0.6)),
    ])
    .shard(3, 1);
    let json = serde_json::to_string(&seq).unwrap();
    let back: Seq<usize> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, seq);
    let (a, b) = (Order::new(seq).unwrap(), Order::new(back).unwrap());
    assert!(a.iter(0..a.len()).map(|(s, i)| (*s, i)).eq(b.iter(0..b.len()).map(|(s, i)| (*s, i))));
    // The wire format is part of the API.
    let seq: Seq<usize> = Seq::weighted_with(9, [(Seq::source(4).shuffle(1), 0.5, Sampling::ramp(0.1, 0.2))]);
    assert_eq!(
        serde_json::to_string(&seq).unwrap(),
        r#"{"Weighted":{"total":9,"parts":[{"seq":{"Shuffle":{"seed":1,"inner":{"Source":4}}},"weight":0.5,"sampling":{"DelayedLinear":{"start":0.1,"full":0.2}}}]}}"#
    );
    assert!(serde_json::from_str::<Seq<usize>>(r#"{"Mix":[{"seq":{"Source":4},"sampling":"Uniform","extra":1}]}"#).is_err());
}

#[test]
fn cloned_cursor_continues_independently() {
    let order = Order::new(Seq::mix([src(0, 500).shuffle(1).repeat(2), src(1, 300).shuffle(2)]).shard(3, 1)).unwrap();
    let n = order.len();
    let mut c = order.iter(0..n);
    let head = ids(c.by_ref().take(100));
    let d = c.clone();
    let rest_c = ids(c);
    let rest_d = ids(d);
    assert_eq!(rest_c, rest_d);
    assert_eq!([head, rest_c].concat(), ids(order.iter(0..n)));
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
    assert_send_sync::<WeightedPart<usize>>();
}

/// Sources through references and smart pointers, trait objects included.
#[test]
fn source_impls() {
    use std::rc::Rc;
    use std::sync::Arc;
    let shared = Arc::new(Src { id: 3, len: 7 });
    let seq: Seq<Box<dyn Source>> = Seq::concat([
        Seq::source(Box::new(5usize) as Box<dyn Source>),
        Seq::source(Box::new(Rc::new(3usize)) as Box<dyn Source>),
        Seq::source(Box::new(shared.clone()) as Box<dyn Source>),
        Seq::source(Box::new(&*shared) as Box<dyn Source>),
    ]);
    assert_eq!(seq.check(), Ok(22));
    let order = Order::new(seq).unwrap();
    assert_eq!(order.sources().iter().map(|s| s.len()).collect::<Vec<_>>(), [5, 3, 7, 7]);
    assert_eq!(order.get(21).1, 6);
    assert!(Seq::source(Rc::new(4usize)).check() == Ok(4) && Seq::source(Box::new(4usize)).check() == Ok(4));
}

/// Inclusive and open bounds in `slice`.
#[test]
fn slice_bounds() {
    let a = || src(0, 10);
    assert_eq!(ids(Order::new(a().slice(..=2)).unwrap().iter(0..3)), vec![(0, 0), (0, 1), (0, 2)]);
    assert_eq!(ids(Order::new(a().slice(8..)).unwrap().iter(0..2)), vec![(0, 8), (0, 9)]);
    assert_eq!(ids(Order::new(a().slice(3..=3)).unwrap().iter(0..1)), vec![(0, 3)]);
    assert_eq!(Order::new(a().slice(..)).unwrap().len(), 10);
    assert_eq!(Order::new(a().slice(4..4)).unwrap().len(), 0);
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
        let walked = ids(order.iter(start..start + 1500));
        for (k, &e) in walked.iter().enumerate() {
            let (s, i) = order.get(start + k);
            assert_eq!((s.id, i), e, "position {}", start + k);
        }
    }
    // Too steep is rejected, not looped over.
    let steep = Seq::mix_with([(src(0, 1 << 45), Sampling::Uniform), (src(1, 1 << 46), Sampling::delayed(0.5))]);
    assert!(matches!(Order::new(steep).unwrap_err().kind(), ErrorKind::TooSteep { part: 1 } | ErrorKind::MixTooLong));
    assert_eq!(MAX_MIX_LEN, 1 << 46);
}

#[test]
fn weighted_shares_sum_and_round() {
    use crate::order::weighted_shares;
    assert_eq!(weighted_shares(1000, &[0.6, 0.4]).unwrap(), [600, 400]);
    assert_eq!(weighted_shares(10, &[1.0, 1.0, 1.0]).unwrap(), [4, 3, 3]);
    assert_eq!(weighted_shares(0, &[]).unwrap(), Vec::<u64>::new());
    assert_eq!(weighted_shares(7, &[0.0, 2.0]).unwrap(), [0, 7]);
    for total in [1u64, 17, 999, 123_456] {
        let w = [0.1, 0.25, 3.0, 0.65, 2.0];
        let shares = weighted_shares(total, &w).unwrap();
        assert_eq!(shares.iter().sum::<u64>(), total);
        let sum: f64 = w.iter().sum();
        for (share, w) in shares.iter().zip(w) {
            assert!((*share as f64 - w / sum * total as f64).abs() < 1.0);
        }
    }
    assert_eq!(weighted_shares(5, &[1.0, -1.0]).unwrap_err(), ErrorKind::InvalidWeight { part: 1, weight: -1.0 });
    assert!(matches!(weighted_shares(5, &[f64::NAN]).unwrap_err(), ErrorKind::InvalidWeight { part: 0, .. }));
    assert_eq!(weighted_shares(5, &[0.0, 0.0]).unwrap_err(), ErrorKind::ZeroWeights);
    assert_eq!(weighted_shares(5, &[]).unwrap_err(), ErrorKind::ZeroWeights);
    assert_eq!(weighted_shares(0, &[0.0, 0.0]).unwrap(), [0, 0]);
    // Many parts with near-integer shares at the mix limit: the sum still comes out exact.
    let w: Vec<f64> = (0..300).map(|i| 1.0 + 1e-9 * (i % 7) as f64).collect();
    for total in [(1u64 << 46) - 1, 1 << 46, 123_456_789_012_345] {
        assert_eq!(weighted_shares(total, &w).unwrap().iter().sum::<u64>(), total);
    }
}

/// A weighted mix has the composition of its weights, repeats short parts (reshuffled) and
/// cuts long ones.
#[test]
fn weighted_mix() {
    let seq = Seq::weighted(3000, [(src(0, 100).shuffle(1), 0.6), (src(1, 5000).shuffle(2), 0.4)]);
    assert_eq!(seq.check(), Ok(3000));
    let order = Order::new(seq).unwrap();
    let all = ids(order.iter(0..3000));
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
    // Errors and edges.
    assert_eq!(
        Order::new(Seq::weighted(10, [(src(0, 0), 1.0), (src(1, 5), 1.0)])).unwrap_err(),
        root(ErrorKind::EmptyWeightedPart { part: 0 })
    );
    assert_eq!(Order::new(Seq::weighted(10, [(src(0, 0), 0.0), (src(1, 5), 1.0)])).unwrap().len(), 10);
    assert_eq!(Order::new(Seq::weighted(0, [(src(0, 5), 1.0)])).unwrap().len(), 0);
    assert_eq!(Order::new(Seq::weighted(0, [(src(0, 5), 0.0)])).unwrap().len(), 0);
    assert_eq!(Order::new(Seq::<Src>::weighted(0, [])).unwrap().len(), 0);
    assert_eq!(Order::new(Seq::weighted(10, [(src(0, 5), 0.0)])).unwrap_err(), root(ErrorKind::ZeroWeights));
    // A part is compiled once: its sources appear once.
    let once = Order::new(Seq::weighted(30, [(Seq::concat([src(0, 4), src(1, 4)]), 1.0), (src(2, 10), 2.0)])).unwrap();
    assert_eq!(once.sources().len(), 3);
}
