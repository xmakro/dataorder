//! Whole-crate tests: random configurations against a reference evaluator that
//! materializes every node, plus the errors, the folds and the reshuffling semantics.

use crate::interleave::Interleave;
use crate::perm::{self, Shape};
use crate::*;

/// A test source: an id to compare orders by, and a length.
#[derive(Clone, Copy, Debug, PartialEq)]
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
        matches!(self, Self::MixTooLong | Self::InvalidSampling { .. } | Self::TooSteep { .. } | Self::Overcommitted { .. })
    }
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
            let evs = parts.iter().map(|(p, _)| eval_at(p, ctx, depth)).collect::<Result<Vec<_>, _>>()?;
            let lens: Vec<u64> = evs.iter().map(|v| v.len() as u64).collect();
            let sampling: Vec<Sampling> = parts.iter().map(|(_, s)| *s).collect();
            let il = Interleave::with_sampling(&lens, &sampling)?;
            il.iter(0..il.len()).map(|(s, j)| evs[s][j as usize]).collect()
        }
        Seq::Shuffle { seed, inner } => {
            let v = eval_at(inner, ctx, depth)?;
            let (shape, key) = (Shape::new(v.len() as u64), perm::key(*seed, ctx));
            (0..v.len() as u64).map(|i| v[perm::permute(shape, key, i) as usize]).collect()
        }
        Seq::Repeat { times, inner } => {
            // Validated even when repeated zero times, like the compiler does.
            let mut out = eval_at(inner, ctx, depth + 1)?;
            out.clear();
            for e in 0..*times {
                out.extend(eval_at(inner, perm::epoch_ctx(ctx, e as u64, depth), depth + 1)?);
            }
            out
        }
        Seq::Skip { n, inner } => {
            let v = eval_at(inner, ctx, depth)?;
            if *n > v.len() {
                return Err(Error::SkipOutOfRange { n: *n, len: v.len() });
            }
            v[*n..].to_vec()
        }
        Seq::Take { n, inner } => {
            let v = eval_at(inner, ctx, depth)?;
            if *n > v.len() {
                return Err(Error::TakeOutOfRange { n: *n, len: v.len() });
            }
            v[..*n].to_vec()
        }
        Seq::Stride { step, offset, inner } => {
            if *step == 0 {
                return Err(Error::ZeroStep);
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
    match rng.below(7) {
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
    for round in 0..600 {
        let seq = random_seq(&mut rng, 4, &lens);
        let seed = rng.next();
        assert_eq!(seq.check(), Order::compile(seq.clone()).map(|o| o.len()), "round {round}: check");
        let order = match Order::compile_seeded(seq.clone(), seed) {
            Ok(o) => o,
            Err(e) if e.is_sampling() => {
                assert!(eval(&seq, seed).is_err_and(|e| e.is_sampling()), "round {round}: {seq:?}");
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
        for _ in 0..4 {
            let a = rng.below(n + 1);
            cursor.seek(a);
            assert_eq!(cursor.position(), a);
            let m = rng.below(n - a + 1);
            assert_eq!(cursor.len(), n - a);
            let got = ids(cursor.by_ref().take(m));
            assert_eq!(got, reference[a..a + m], "round {round}: seek {a} of {seq:?}");
        }
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
        let order = match Order::compile(seq.clone()) {
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
    let order = Order::compile(seq.clone()).unwrap();
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
    // The first repetition is the sequence itself, and repeating once changes nothing.
    let once = Order::compile(src(7, 1000).shuffle(3)).unwrap();
    assert_eq!(once.iter(0..1000).map(|(_, i)| i).collect::<Vec<_>>(), epochs[0]);
    assert_eq!(ids(Order::compile(src(7, 1000).shuffle(3).repeat(1)).unwrap().iter(0..1000)), ids(once.iter(0..1000)));
    // Nested repeats: (outer 0, inner 1) and (outer 1, inner 0) are different orders.
    let nested = Order::compile(src(7, 100).shuffle(3).repeat(2).repeat(2)).unwrap();
    let block = |b: usize| ids(nested.iter(b * 100..(b + 1) * 100));
    assert_ne!(block(1), block(2));
    assert_eq!(block(0), ids(Order::compile(src(7, 100).shuffle(3)).unwrap().iter(0..100)));
    // Same seed twice under a concat: the same order twice.
    let twice = Order::compile(Seq::concat([src(7, 1000).shuffle(3), src(7, 1000).shuffle(3)])).unwrap();
    let v = ids(twice.iter(0..2000));
    assert_eq!(v[..1000], v[1000..]);
    // The order's seed changes every shuffle.
    let reseeded = Order::compile_seeded(seq, 99).unwrap();
    assert_ne!(ids(reseeded.iter(0..1000)), ids(order.iter(0..1000)));
}

#[test]
fn shards_partition_the_sequence() {
    let base = Seq::mix([src(0, 1000).shuffle(1), src(1, 300).shuffle(2)]).repeat(2);
    let order = Order::compile(base.clone()).unwrap();
    let all = ids(order.iter(0..order.len()));
    let mut from_shards = Vec::new();
    for w in 0..8 {
        let shard = Order::compile(base.clone().shard(w, 8)).unwrap();
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
    let seq = Seq::mix([src(0, 700).shuffle(1).repeat(2), Seq::concat([src(1, 50), src(2, 120).shuffle(2)])]).shard(1, 3);
    let order = Order::compile(seq.clone()).unwrap();
    let mapped = Order::compile(seq.map(|s| s.len)).unwrap();
    assert_eq!(mapped.sources(), &[700, 50, 120]);
    let a: Vec<(usize, usize)> = order.iter(0..order.len()).map(|(s, i)| (s.len, i)).collect();
    let b: Vec<(usize, usize)> = mapped.iter(0..mapped.len()).map(|(&l, i)| (l, i)).collect();
    assert_eq!(a, b);
}

#[test]
fn errors() {
    let a = src(0, 10);
    assert_eq!(Order::compile(a.clone().slice(3..12)).unwrap_err(), Error::TakeOutOfRange { n: 9, len: 7 });
    assert_eq!(Order::compile(a.clone().skip(11)).unwrap_err(), Error::SkipOutOfRange { n: 11, len: 10 });
    assert_eq!(Order::compile(a.clone().take(11)).unwrap_err(), Error::TakeOutOfRange { n: 11, len: 10 });
    assert_eq!(Order::compile(a.clone().slice(2..=9)).unwrap().len(), 8);
    assert_eq!(Order::compile(a.clone().slice(10..)).unwrap().len(), 0);
    assert_eq!(Order::compile(a.clone().stride(0, 0)).unwrap_err(), Error::ZeroStep);
    assert_eq!(Order::compile(a.clone().repeat(usize::MAX)).unwrap_err(), Error::Overflow);
    assert_eq!(Order::compile(Seq::concat([a.clone().repeat(usize::MAX / 10), a.clone()])).unwrap_err(), Error::Overflow);
    let over = Seq::mix_with([(src(0, 10), Sampling::DelayedLinear { start: 0.5, full: 0.5 }), (src(1, 1), Sampling::Uniform)]);
    assert!(matches!(Order::compile(over), Err(Error::Overcommitted { .. })));
    // A mix that folds away is still validated.
    let over1 = Seq::mix_with([(src(0, 10), Sampling::DelayedLinear { start: 2.0, full: 2.0 })]);
    assert_eq!(Order::compile(over1).unwrap_err(), Error::InvalidSampling { part: 0, sampling: Sampling::DelayedLinear { start: 2.0, full: 2.0 } });
}

#[test]
#[should_panic(expected = "slice end before start")]
#[allow(clippy::reversed_empty_ranges)]
fn reversed_slice_panics() {
    let _ = src(0, 10).slice(5..3);
}

#[test]
fn edge_cases() {
    let empty = Order::compile(Seq::<Src>::concat([])).unwrap();
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.iter(0..0).count(), 0);
    let empty = Order::compile(src(1, 5).shuffle(1).take(0).repeat(4)).unwrap();
    assert!(empty.is_empty());
    assert_eq!(Order::compile(src(1, 5).stride(3, 9)).unwrap().len(), 0);
    let one = Order::compile(src(2, 1).shuffle(5).repeat(3)).unwrap();
    assert_eq!(ids(one.iter(0..3)), vec![(2, 0); 3]);
    let order = Order::compile(src(0, 5)).unwrap();
    assert_eq!(order.sources(), &[Src { id: 0, len: 5 }]);
    let mut c = order.iter(1..4);
    assert_eq!(c.remaining(), 3);
    assert_eq!(c.next().map(|(s, i)| (s.id, i)), Some((0, 1)));
    c.seek(4);
    assert!(c.next().is_none());
    c.seek(0);
    assert_eq!(ids(c), vec![(0, 0), (0, 1), (0, 2), (0, 3)]);
    // Bare lengths are sources; a source may be shared through a reference.
    let lens = [5usize, 3];
    let shared = Order::compile(Seq::concat([Seq::source(&lens[0]), Seq::source(&lens[1]), Seq::source(&lens[0]).shuffle(1)])).unwrap();
    assert_eq!(shared.len(), 13);
    assert_eq!(shared.sources().len(), 3);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn huge_lengths() {
    // Lengths far beyond anything materializable: positions must still resolve.
    let seq = src(0, 1 << 40).shuffle(1).repeat(1 << 20).skip(12345);
    let order = Order::compile(seq).unwrap();
    assert_eq!(order.len(), (1usize << 60) - 12345);
    let last = order.len() - 1;
    let (s, i) = order.get(last);
    assert_eq!(s.id, 0);
    assert!(i < 1 << 40);
    assert_eq!(ids(order.iter_from(last)), vec![(0, i)]);
    let v = ids(order.iter(1 << 59..(1 << 59) + 5));
    for (k, &e) in v.iter().enumerate() {
        let (s, i) = order.get((1 << 59) + k);
        assert_eq!((s.id, i), e);
    }
}

/// FNV-1a over the elements: a stable fingerprint of an order.
fn fingerprint<'a>(it: impl Iterator<Item = (&'a Src, usize)>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for (s, i) in it {
        for b in (s.id as u64).to_le_bytes().into_iter().chain((i as u64).to_le_bytes()) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

/// The orders themselves, pinned. A mismatch means the crate's orders changed: that is a
/// breaking change (see the crate docs on stability), to be made deliberately, with a
/// version bump and new values here.
#[test]
fn golden_orders() {
    use Sampling::*;
    let cases: Vec<(&str, Seq<Src>, u64)> = vec![
        ("shuffle", src(0, 1000).shuffle(7), 0),
        ("shuffle, seeded order", src(0, 1000).shuffle(7), 42),
        ("shuffle.repeat", src(0, 777).shuffle(1).repeat(3), 0),
        ("shuffle(concat)", Seq::concat([src(0, 300), src(1, 500).shuffle(2)]).shuffle(3), 0),
        ("slice of shuffle", src(0, 5000).shuffle(9).skip(100).take(2000), 0),
        ("mix uniform", Seq::mix([src(0, 1000).shuffle(1), src(1, 300).shuffle(2), src(2, 50)]), 0),
        (
            "mix scheduled",
            Seq::mix_with([
                (src(0, 2000).shuffle(1), Uniform),
                (src(1, 400).shuffle(2), DelayedLinear { start: 0.5, full: 0.5 }),
                (src(2, 600), DelayedLinear { start: 0.2, full: 0.6 }),
            ]),
            0,
        ),
        (
            "nested mixes, epochs, shard",
            Seq::mix([
                Seq::mix([src(0, 500).shuffle(1).repeat(2), src(1, 300).shuffle(2).repeat(3)]),
                src(2, 900).shuffle(3),
            ])
            .shard(1, 4),
            0,
        ),
        ("stride over mix", Seq::mix([src(0, 1000), src(1, 999).shuffle(4)]).stride(7, 3), 0),
        ("repeat of mix", Seq::mix([src(0, 200).shuffle(1), src(1, 100).shuffle(2)]).repeat(4), 0),
    ];
    const EXPECTED: [u64; 10] = [
        8944480274337887517,
        4625008917299269681,
        10153334795136506768,
        2873731158959093005,
        11124861484752690026,
        11279912391434559340,
        10444670851558434453,
        9703835265997803807,
        10527708798688175491,
        15930632077147421093,
    ];
    let actual: Vec<u64> = cases
        .iter()
        .map(|(name, seq, seed)| {
            let order = Order::compile_seeded(seq.clone(), *seed).unwrap_or_else(|e| panic!("{name}: {e}"));
            fingerprint(order.iter(0..order.len()))
        })
        .collect();
    let names: Vec<&str> = cases.iter().map(|c| c.0).collect();
    assert_eq!(actual, EXPECTED, "orders changed for {names:?}");
    // A few elements in the clear, for the first case.
    let order = Order::compile(src(0, 1000).shuffle(7)).unwrap();
    const FIRST: [usize; 6] = [658, 809, 435, 971, 671, 326];
    assert_eq!(order.iter(0..6).map(|(_, i)| i).collect::<Vec<_>>(), FIRST);
    assert!((0..6).all(|k| order.get(k).1 == FIRST[k]));
}

/// With the `serde` feature a configuration survives a round trip and compiles to the same order.
#[cfg(feature = "serde")]
#[test]
fn serde_round_trip() {
    let seq: Seq<usize> = Seq::mix_with([
        (Seq::source(1000).shuffle(1).repeat(2), Sampling::Uniform),
        (Seq::concat([Seq::source(300).skip(10), Seq::source(50).take(20)]).shuffle(2), Sampling::ramp(0.2, 0.6)),
    ])
    .shard(1, 3);
    let json = serde_json::to_string(&seq).unwrap();
    let back: Seq<usize> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, seq);
    let (a, b) = (Order::compile(seq).unwrap(), Order::compile(back).unwrap());
    assert!(a.iter(0..a.len()).map(|(s, i)| (*s, i)).eq(b.iter(0..b.len()).map(|(s, i)| (*s, i))));
}

#[test]
fn cloned_cursor_continues_independently() {
    let order = Order::compile(Seq::mix([src(0, 500).shuffle(1).repeat(2), src(1, 300).shuffle(2)]).shard(1, 3)).unwrap();
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
    assert_send_sync::<Sampling>();
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
    let order = Order::compile(seq).unwrap();
    assert_eq!(order.sources().iter().map(|s| s.len()).collect::<Vec<_>>(), [5, 3, 7, 7]);
    assert_eq!(order.get(21).1, 6);
    assert!(Seq::source(Rc::new(4usize)).check() == Ok(4) && Seq::source(Box::new(4usize)).check() == Ok(4));
}

/// Inclusive and open bounds in `slice`.
#[test]
fn slice_bounds() {
    let a = || src(0, 10);
    assert_eq!(ids(Order::compile(a().slice(..=2)).unwrap().iter(0..3)), vec![(0, 0), (0, 1), (0, 2)]);
    assert_eq!(ids(Order::compile(a().slice(8..)).unwrap().iter(0..2)), vec![(0, 8), (0, 9)]);
    assert_eq!(ids(Order::compile(a().slice(3..=3)).unwrap().iter(0..1)), vec![(0, 3)]);
    assert_eq!(Order::compile(a().slice(..)).unwrap().len(), 10);
    assert_eq!(Order::compile(a().slice(4..4)).unwrap().len(), 0);
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
    let order = Order::compile(seq).unwrap();
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
    assert!(matches!(Order::compile(steep), Err(Error::TooSteep { part: 1 }) | Err(Error::MixTooLong)));
}
