//! The orders themselves, pinned through the public API. A mismatch means the crate's orders
//! changed: that is a breaking change (see the crate docs on stability), to be made
//! deliberately, with a version bump and new values here.

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
        13510848840803686825,
        5734682759774056529,
        2152343650903757428,
        2341334849338153325,
        2420967535336859100,
        11125969817394475988,
        3696955324159907089,
        905848074036688221,
        12216569398622504889,
        4113493865489899877,
        16159320458550980399,
        3632482425850924517,
        5044342942726522595,
        14166974499237794981,
        16685696649537605013,
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
    const FIRST: [usize; 6] = [629, 114, 228, 812, 639, 604];
    assert_eq!(order.iter(0..6).map(|(_, i)| i).collect::<Vec<_>>(), FIRST);
    assert!((0..6).all(|k| order.get(k).1 == FIRST[k]));
}

#[test]
fn golden_name_and_path_salted_orders() {
    struct Named {
        name: &'static str,
        as_path: bool,
    }

    impl Source for Named {
        fn len(&self) -> usize {
            1000
        }

        fn salt(&self) -> u64 {
            if self.as_path { dataorder::salt_path(self.name) } else { dataorder::salt(self.name) }
        }
    }

    // Pin the public identity helpers as well as their effect on the whole permutation.
    // Both the prefix and fingerprint were independently calculated from the specified
    // FNV-1a and Feistel arithmetic, rather than obtained by blessing this test's output.
    const FIRST: [usize; 12] = [445, 928, 15, 77, 0, 293, 540, 797, 190, 407, 709, 652];
    const EXPECTED: u64 = 15_852_656_108_745_184_545;
    for as_path in [false, true] {
        let order = Order::with_seed(Seq::source(Named { name: "web/训练.bin", as_path }).shuffle(7), 42).unwrap();
        assert_eq!(order.iter(..12).map(|(_, i)| i).collect::<Vec<_>>(), FIRST);
        let actual = order
            .iter(..)
            .flat_map(|(_, i)| (i as u64).to_le_bytes())
            .fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3));
        assert_eq!(actual, EXPECTED, "as_path={as_path}");
    }
}
