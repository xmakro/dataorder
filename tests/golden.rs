//! The orders themselves, pinned through the public API. A mismatch means the crate's orders
//! changed: that is a breaking change (see the crate docs on stability), to be made
//! deliberately, with a version bump and new values here.

mod common;

use Schedule::*;
use common::{Src, src};
use dataorder::{Order, Schedule, Seq, Source};

/// FNV-1a over the elements: a stable fingerprint of an order.
fn fingerprint<'a>(it: impl Iterator<Item = dataorder::Item<'a, Src>>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for item in it {
        for b in item.source.id.to_le_bytes().into_iter().chain((item.record_index as u64).to_le_bytes()) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

#[test]
fn golden_orders() {
    let cases: Vec<(&str, Seq<Src>, u64)> = vec![
        ("shuffle", src(0, 1000).shuffle(), 0),
        ("shuffle, seeded order", src(0, 1000).shuffle(), 42),
        ("shuffle.repeat", src(0, 777).shuffle().repeat(3), 0),
        ("shuffle(concat)", Seq::concat([src(0, 300), src(1, 500).shuffle()]).shuffle(), 0),
        ("slice of shuffle", src(0, 5000).shuffle().skip(100).take(2000), 0),
        ("mix uniform", Seq::mix([src(0, 1000).shuffle(), src(1, 300).shuffle(), src(2, 50)]), 0),
        (
            "mix scheduled",
            Seq::mix([
                (src(0, 2000).shuffle(), Uniform),
                (src(1, 400).shuffle(), Schedule::delayed(0.5)),
                (src(2, 600), Schedule::ramp(0.2, 0.6)),
            ]),
            0,
        ),
        (
            "nested mixes, epochs, shard",
            Seq::mix([Seq::mix([src(0, 500).shuffle().repeat(2), src(1, 300).shuffle().repeat(3)]), src(2, 900).shuffle()])
                .skip(1)
                .step_by(4),
            0,
        ),
        ("stride over mix", Seq::mix([src(0, 1000), src(1, 999).shuffle()]).skip(3).step_by(7), 0),
        ("repeat of mix", Seq::mix([src(0, 200).shuffle(), src(1, 100).shuffle()]).repeat(4), 0),
        ("mix with explicit counts", Seq::mix([src(0, 100).shuffle().cycle_to(1800), src(1, 5000).shuffle().cycle_to(1200)]), 0),
        ("nested repeats", src(0, 100).shuffle().repeat(3).repeat(2), 0),
        ("cycle of a repeated shuffle", src(0, 300).shuffle().repeat(2).cycle_to(1000), 0),
        (
            "mix fading",
            Seq::mix([
                (src(0, 1500).shuffle(), Uniform),
                (src(1, 300).shuffle(), Schedule::until(0.4)),
                (src(2, 400), Schedule::trapezoid(0.2, 0.4, 0.6, 0.9)),
            ]),
            0,
        ),
        ("shuffled repeat", src(0, 777).repeat_shuffled(3), 0),
        ("shuffled repeat, seeded order", src(0, 777).repeat_shuffled(3), 42),
        ("shuffled cycle", src(0, 300).cycle_to_shuffled(1000), 0),
        ("double shuffle", src(0, 1000).shuffle().shuffle(), 42),
        ("shuffle, shuffled repeat", src(0, 37).shuffle().repeat_shuffled(3), 42),
        ("shuffled repeat, shuffled cycle", src(0, 37).repeat_shuffled(2).cycle_to_shuffled(100), 42),
    ];
    // Captured after the 0.4.0 key derivation change. Every case without a mix
    // (shuffles, repeats, cycles, selections and shuffled repetitions, nested or not)
    // was independently recomputed in Python from the documented Feistel arithmetic,
    // key derivation and configuration salts. The mix cases are this implementation's
    // output.
    const EXPECTED: [u64; 20] = [
        8775043117251707565,
        3994110308236394269,
        3557037918981979648,
        9224125550474297501,
        17133402622130432927,
        1083239877109596384,
        3718334892975712781,
        12369965706742603044,
        4378099840426581584,
        10161468430947224613,
        4352769584230633250,
        16917178287728660389,
        5795192745291637981,
        5529291456710688149,
        14913036641801498832,
        949497126177981280,
        2289177200024162075,
        15290745128168257965,
        13806615233696808257,
        12692409462087837018,
    ];
    let actual: Vec<u64> = cases
        .iter()
        .map(|(name, seq, seed)| {
            let order = Order::with_seed(seq.clone(), *seed).unwrap_or_else(|e| panic!("{name}: {e}"));
            fingerprint(order.iter())
        })
        .collect();
    let names: Vec<&str> = cases.iter().map(|c| c.0).collect();
    assert_eq!(actual, EXPECTED, "orders changed for {names:?}");
    // A few elements in the clear, for the first case.
    let order = Order::new(src(0, 1000).shuffle()).unwrap();
    const FIRST: [usize; 6] = [526, 829, 306, 360, 895, 764];
    assert_eq!(order.cursor(0..6).unwrap().map(|item| item.record_index).collect::<Vec<_>>(), FIRST);
    assert!((0..6).all(|k| order.get(k).unwrap().record_index == FIRST[k]));
}

#[test]
fn golden_name_salted_order() {
    struct Named {
        name: &'static str,
    }

    impl Source for Named {
        fn len(&self) -> usize {
            1000
        }

        fn salt(&self) -> u64 {
            dataorder::salt(self.name)
        }
    }

    // Pin the public salt helper as well as its effect on the whole permutation.
    // Both the prefix and fingerprint were independently calculated from the specified
    // FNV-1a and Feistel arithmetic, rather than obtained by blessing this test's output.
    const FIRST: [usize; 12] = [506, 411, 725, 884, 42, 543, 801, 386, 458, 650, 383, 516];
    const EXPECTED: u64 = 15_280_393_085_099_573_649;
    let order = Order::with_seed(Seq::source(Named { name: "web/训练.bin" }).shuffle(), 42).unwrap();
    assert_eq!(order.cursor(..12).unwrap().map(|item| item.record_index).collect::<Vec<_>>(), FIRST);
    let actual = order
        .iter()
        .flat_map(|item| (item.record_index as u64).to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3));
    assert_eq!(actual, EXPECTED);
}
