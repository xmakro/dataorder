//! The orders themselves, pinned through the public API. A mismatch means the crate's orders
//! changed: that is a breaking change (see the crate docs on stability), to be made
//! deliberately, with a version bump, a changelog entry and new values here.

use Sampling::*;
use dataorder::{Order, Sampling, Seq, Source};

#[derive(Clone, Copy, Debug, PartialEq)]
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

fn src(id: u32, len: usize) -> Seq<Src> {
    Seq::source(Src { id, len })
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

#[test]
fn golden_orders() {
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
            Seq::mix([Seq::mix([src(0, 500).shuffle(1).repeat(2), src(1, 300).shuffle(2).repeat(3)]), src(2, 900).shuffle(3)]).shard(4, 1),
            0,
        ),
        ("stride over mix", Seq::mix([src(0, 1000), src(1, 999).shuffle(4)]).stride(7, 3), 0),
        ("repeat of mix", Seq::mix([src(0, 200).shuffle(1), src(1, 100).shuffle(2)]).repeat(4), 0),
        ("weighted", Seq::weighted(3000, [(src(0, 100).shuffle(1), 0.6), (src(1, 5000).shuffle(2), 0.4)]), 0),
        ("nested repeats", src(0, 100).shuffle(3).repeat(3).repeat(2), 0),
        ("cycle of a shuffled repeat", src(0, 300).shuffle(5).repeat(2).cycle(1000), 0),
        (
            "weighted with a concat part",
            Seq::weighted(500, [(Seq::concat([src(0, 200), src(1, 300)]), 1.0), (src(2, 50).shuffle(1), 1.0)]).shuffle(2),
            0,
        ),
        (
            "mix fading",
            Seq::mix_with([
                (src(0, 1500).shuffle(1), Uniform),
                (src(1, 300).shuffle(2), Sampling::until(0.4)),
                (src(2, 400), Sampling::trapezoid(0.2, 0.4, 0.6, 0.9)),
            ]),
            0,
        ),
    ];
    const EXPECTED: [u64; 15] = [
        9476750321503116817,
        3802312810472554945,
        2814580347210730272,
        15433798136956187369,
        690855147851393981,
        5722012605816943760,
        14380507312460122493,
        10653168844818408168,
        9595942154208374878,
        10043771473524833253,
        6532104605628653644,
        9662099335159339301,
        4277006377877170397,
        991967725648243685,
        11336253557503442625,
    ];
    let actual: Vec<u64> = cases
        .iter()
        .map(|(name, seq, seed)| {
            let order = Order::with_seed(seq.clone(), *seed).unwrap_or_else(|e| panic!("{name}: {e}"));
            fingerprint(order.iter(..))
        })
        .collect();
    let names: Vec<&str> = cases.iter().map(|c| c.0).collect();
    assert_eq!(actual, EXPECTED, "orders changed for {names:?}");
    // A few elements in the clear, for the first case.
    let order = Order::new(src(0, 1000).shuffle(7)).unwrap();
    const FIRST: [usize; 6] = [289, 680, 702, 764, 472, 996];
    assert_eq!(order.iter(0..6).map(|(_, i)| i).collect::<Vec<_>>(), FIRST);
    assert!((0..6).all(|k| order.get(k).1 == FIRST[k]));
}
