//! Whole-crate tests: the merge against a brute-force sort, permutation and order
//! properties, exact seeks, balance bounds and the schedule semantics.

use super::*;
use crate::tests::Rng;
use Sampling::*;

fn full(il: &Interleave) -> Vec<(usize, u64)> {
    il.iter(0..il.len()).collect()
}

/// Brute force: sort every element by (key, sequence, index).
fn reference(il: &Interleave) -> Vec<(usize, u64)> {
    let mut all = Vec::new();
    for (s, seq) in il.seqs.iter().enumerate() {
        for j in 0..seq.n {
            all.push((il.key(s, j, &mut 0), s, j));
        }
    }
    all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    all.into_iter().map(|(_, s, j)| (s, j)).collect()
}

/// Share of sequence `s` that its share function prescribes at `t`.
fn share(il: &Interleave, s: usize, t: f64) -> f64 {
    il.profile(s).share(t).clamp(0.0, 1.0)
}

/// Largest `|count_s(t) − n_s·F_s(t/N)|` over all `t`, per sequence (in elements of `s`).
fn worst_deviation(il: &Interleave, all: &[(usize, u64)]) -> Vec<f64> {
    let k = il.seqs.len();
    let n = il.len() as f64;
    let mut counts = vec![0u64; k];
    let mut worst = vec![0.0f64; k];
    for t in 0..=all.len() {
        for s in 0..k {
            if il.seqs[s].n == 0 {
                continue;
            }
            let ideal = il.seqs[s].n as f64 * share(il, s, t as f64 / n);
            worst[s] = worst[s].max((counts[s] as f64 - ideal).abs());
        }
        if t < all.len() {
            counts[all[t].0] += 1;
        }
    }
    worst
}

fn random_lens(rng: &mut Rng) -> Vec<u64> {
    let k = rng.below64(9) as usize;
    (0..k)
        .map(|_| match rng.below64(4) {
            0 => 0,
            1 => rng.below64(4),
            2 => rng.below64(30),
            _ => rng.below64(300),
        })
        .collect()
}

fn uniform_cases() -> Vec<Vec<u64>> {
    let mut v = vec![
        vec![],
        vec![0],
        vec![0, 0, 0],
        vec![1],
        vec![7],
        vec![1, 1],
        vec![1, 9],
        vec![9, 1],
        vec![3, 7],
        vec![14, 124, 43],
        vec![5, 5, 5],
        vec![100, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        vec![1, 2, 4, 8, 16, 32, 64],
        vec![0, 10, 0, 3, 0],
        vec![5000, 1, 3000, 2, 0, 7000],
    ];
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for _ in 0..200 {
        v.push(random_lens(&mut rng));
    }
    v
}

fn scheduled_cases() -> Vec<(Vec<u64>, Vec<Sampling>)> {
    vec![
        (vec![1000, 300], vec![Uniform, DelayedLinear { start: 0.4, full: 0.4 }]),
        (vec![2000, 500], vec![Uniform, DelayedLinear { start: 0.2, full: 0.6 }]),
        (vec![14, 124, 43], vec![DelayedLinear { start: 0.5, full: 0.5 }, Uniform, DelayedLinear { start: 0.2, full: 0.6 }]),
        (
            vec![500, 500, 500, 500],
            vec![
                Uniform,
                DelayedLinear { start: 0.1, full: 0.1 },
                DelayedLinear { start: 0.3, full: 0.3 },
                DelayedLinear { start: 0.0, full: 0.5 },
            ],
        ),
        (
            vec![300, 300, 0, 7],
            vec![
                DelayedLinear { start: 0.0, full: 0.0 },
                DelayedLinear { start: 0.0, full: 0.0 },
                DelayedLinear { start: 0.9, full: 0.9 },
                Uniform,
            ],
        ),
        (vec![1000, 50, 50], vec![Uniform, DelayedLinear { start: 0.9, full: 0.9 }, DelayedLinear { start: 0.8, full: 0.95 }]),
        (vec![100, 100, 800], vec![DelayedLinear { start: 0.0, full: 0.5 }, DelayedLinear { start: 0.5, full: 1.0 }, Uniform]),
        (vec![3, 1000, 1], vec![DelayedLinear { start: 0.7, full: 0.7 }, Uniform, DelayedLinear { start: 0.2, full: 0.9 }]),
        (vec![300, 700], vec![Sampling::until(0.5), Uniform]),
        (vec![200, 300, 500], vec![Sampling::fading(0.2, 0.6), Sampling::trapezoid(0.3, 0.5, 0.7, 0.9), Uniform]),
        (vec![500, 500], vec![Sampling::until(0.5), Sampling::delayed(0.5)]),
        (vec![40, 60, 900], vec![Sampling::trapezoid(0.1, 0.1, 0.1, 0.3), Sampling::trapezoid(0.6, 0.8, 0.8, 1.0), Uniform]),
    ]
}

fn all_cases() -> Vec<Interleave> {
    let mut v: Vec<Interleave> = uniform_cases().iter().map(|l| Interleave::new(l)).collect();
    v.extend(scheduled_cases().iter().map(|(l, s)| Interleave::with_sampling(l, s).unwrap()));
    v
}

#[test]
fn a_quantile_on_a_flat_share_is_drawn_before_the_plateau() {
    // The uniform singleton's target share is 1/4. Its profile first reaches that
    // share at 1/16, then stays flat while all three scheduled elements are drawn.
    let il = Interleave::with_sampling(&[1, 3], &[Uniform, Sampling::trapezoid(0.0625, 0.0625, 0.8125, 0.8125)]).unwrap();
    let expected = [(0, 0), (1, 0), (1, 1), (1, 2)];
    assert_eq!(full(&il), expected);
    for start in 0..=il.len() {
        assert_eq!(il.iter(start..il.len()).collect::<Vec<_>>(), expected[start as usize..]);
    }
}

#[test]
fn matches_brute_force_sort() {
    for il in all_cases() {
        let expect = reference(&il);
        assert_eq!(expect.len() as u64, il.len());
        assert_eq!(full(&il), expect, "{il:?}");
    }
}

#[test]
fn is_a_permutation_preserving_order() {
    for il in all_cases() {
        let mut next = vec![0u64; il.seqs.len()];
        for (t, (s, j)) in full(&il).into_iter().enumerate() {
            assert_eq!(j, next[s], "pos {t}");
            next[s] += 1;
        }
        let lens: Vec<u64> = il.seqs.iter().map(|s| s.n).collect();
        assert_eq!(next, lens);
    }
}

#[test]
fn subranges_and_seams() {
    let mut rng = Rng(12345);
    for il in all_cases() {
        let n = il.len();
        let all = full(&il);
        for _ in 0..30 {
            let a = rng.below64(n + 1);
            let b = a + rng.below64(n + 1 - a);
            let got: Vec<_> = il.iter(a..b).collect();
            assert_eq!(got, &all[a as usize..b as usize], "range {a}..{b}");
            assert_eq!(il.iter(a..b).size_hint(), ((b - a) as usize, Some((b - a) as usize)));
        }
        // Random partitions tile the full sequence exactly.
        let mut cuts: Vec<u64> = (0..5).map(|_| rng.below64(n + 1)).collect();
        cuts.push(0);
        cuts.push(n);
        cuts.sort_unstable();
        let tiled: Vec<_> = cuts.windows(2).flat_map(|w| il.iter(w[0]..w[1])).collect();
        assert_eq!(tiled, all);
    }
}

#[test]
fn every_seek_matches_the_slice() {
    // For every start position (small cases) or many random ones (large cases), the
    // seeded iterator must reproduce the slice of the full iteration exactly.
    let mut rng = Rng(4242);
    for il in all_cases() {
        let n = il.len();
        let all = full(&il);
        let starts: Vec<u64> = if n <= 3000 { (0..=n).collect() } else { (0..300).map(|_| rng.below64(n + 1)).collect() };
        for a in starts {
            let b = (a + 1 + rng.below64(40)).min(n);
            let got: Vec<_> = il.iter(a..b).collect();
            assert_eq!(got, &all[a as usize..b as usize], "seek to {a}");
        }
    }
}

#[test]
fn keys_are_monotone_within_sequences() {
    for il in all_cases() {
        for (s, seq) in il.seqs.iter().enumerate() {
            let mut seg = 0;
            let mut prev = f64::NEG_INFINITY;
            for j in 0..seq.n {
                let key = il.key(s, j, &mut seg);
                assert!(key >= prev, "seq {s} index {j}: {key} < {prev}");
                prev = key;
            }
        }
    }
}

#[test]
fn random_configurations() {
    // Random lengths and schedules: the merge is a permutation in order, equals the
    // brute-force sort, keys are monotone, every seek matches the slice, and nothing of a
    // scheduled sequence appears before its start.
    let mut rng = Rng(0xC0FF_EE00_1234_5678);
    let mut checked = 0;
    while checked < 2000 {
        let k = 1 + rng.below64(12) as usize;
        let lens: Vec<u64> = (0..k)
            .map(|_| match rng.below64(3) {
                0 => rng.below64(3),
                1 => rng.below64(50),
                _ => rng.below64(400),
            })
            .collect();
        let sampling: Vec<Sampling> = (0..k)
            .map(|_| {
                let d0 = rng.below64(999);
                let d1 = if rng.below64(2) == 0 { d0 } else { d0 + rng.below64(1001 - d0) };
                match rng.below64(4) {
                    0 => Uniform,
                    1 => {
                        let d2 = d1 + rng.below64(1001 - d1);
                        let d3 = if rng.below64(2) == 0 { d2 } else { d2 + rng.below64(1001 - d2) };
                        if d0 + d1 < d2 + d3 {
                            Sampling::trapezoid(d0 as f64 / 1000.0, d1 as f64 / 1000.0, d2 as f64 / 1000.0, d3 as f64 / 1000.0)
                        } else {
                            Uniform
                        }
                    }
                    _ => DelayedLinear { start: d0 as f64 / 1000.0, full: d1 as f64 / 1000.0 },
                }
            })
            .collect();
        let il = match Interleave::with_sampling(&lens, &sampling) {
            Ok(il) => il,
            Err(SamplingError::Overcommitted { .. }) => continue,
            Err(e) => panic!("{lens:?} {sampling:?}: {e}"),
        };
        checked += 1;
        let n = il.len();
        let all = full(&il);
        assert_eq!(all, reference(&il), "{lens:?} {sampling:?}");
        let mut next = vec![0u64; k];
        for &(s, j) in &all {
            assert_eq!(j, next[s]);
            next[s] += 1;
        }
        assert_eq!(next, lens);
        for (s, seq) in il.seqs.iter().enumerate() {
            let mut seg = 0;
            let mut prev = -1.0;
            for j in 0..seq.n {
                let key = il.key(s, j, &mut seg);
                assert!(key.is_finite() && (0.0..=1.0).contains(&key) && key >= prev, "{lens:?} {sampling:?} seq {s} j {j}");
                prev = key;
            }
        }
        for _ in 0..20 {
            let a = rng.below64(n + 1);
            let b = (a + rng.below64(60)).min(n);
            assert_eq!(il.iter(a..b).collect::<Vec<_>>(), &all[a as usize..b as usize], "{lens:?} {sampling:?} seek {a}");
        }
        for (s, samp) in sampling.iter().enumerate() {
            let (start, off) = match *samp {
                DelayedLinear { start, .. } => (start, 1.0),
                Trapezoid { start, off, .. } => (start, off),
                Uniform => continue,
            };
            if let Some(first) = all.iter().position(|&(x, _)| x == s) {
                assert!(first as f64 >= start * n as f64 - k as f64 - 1.0, "{lens:?} {sampling:?} seq {s} first at {first}");
            }
            if let Some(last) = all.iter().rposition(|&(x, _)| x == s) {
                assert!(last as f64 <= off * n as f64 + k as f64 + 1.0, "{lens:?} {sampling:?} seq {s} last at {last}");
            }
        }
    }
}

#[test]
fn uniform_balance() {
    for lens in uniform_cases() {
        let il = Interleave::new(&lens);
        let worst = worst_deviation(&il, &full(&il));
        for (s, w) in worst.iter().enumerate() {
            assert!(*w <= 2.0, "lens {lens:?} seq {s}: deviation {w}");
        }
    }
}

#[test]
fn equal_lengths_round_robin_in_input_order() {
    for k in 1..=17usize {
        let il = Interleave::new(&vec![6u64; k]);
        for (t, (s, j)) in full(&il).into_iter().enumerate() {
            assert_eq!(s, t % k);
            assert_eq!(j as usize, t / k);
        }
    }
}

#[test]
fn schedules_are_followed() {
    for (lens, sampling) in scheduled_cases() {
        let il = Interleave::with_sampling(&lens, &sampling).unwrap();
        let all = full(&il);
        let n = il.len() as f64;
        let k = lens.len() as f64;
        let worst = worst_deviation(&il, &all);
        for (s, w) in worst.iter().enumerate() {
            // one element of own rounding plus the joint time warp of at most k positions
            let bound = 1.5 + k * lens[s] as f64 / n;
            assert!(*w <= bound, "{lens:?} {sampling:?} seq {s}: deviation {w} > {bound}");
        }
        // Nothing from a delayed sequence before its start, nor from a fading one after its
        // end (up to the k-position warp).
        for (s, samp) in sampling.iter().enumerate() {
            let (start, off) = match *samp {
                DelayedLinear { start, .. } => (start, 1.0),
                Trapezoid { start, off, .. } => (start, off),
                Uniform => continue,
            };
            if lens[s] == 0 {
                continue;
            }
            let first = all.iter().position(|&(x, _)| x == s).unwrap() as f64;
            assert!(first >= start * n - k - 1.0, "{lens:?} {sampling:?} seq {s}: first at {first}, start {}", start * n);
            let last = all.iter().rposition(|&(x, _)| x == s).unwrap() as f64;
            assert!(last <= off * n + k + 1.0, "{lens:?} {sampling:?} seq {s}: last at {last}, off {}", off * n);
        }
    }
}

#[test]
fn fading_sequences_stop_and_free_the_rest() {
    // Seq 0 (30% of the elements) runs at a constant rate until 0.5 and stops: it is done
    // by the middle, and only seq 1 fills the second half.
    let il = Interleave::with_sampling(&[300, 700], &[Sampling::until(0.5), Uniform]).unwrap();
    let all = full(&il);
    assert!(all[500..].iter().all(|&(s, _)| s == 1));
    assert_eq!(all[..500].iter().filter(|&&(s, _)| s == 0).count(), 300);
    for w in all[..500].windows(50) {
        let c = w.iter().filter(|&&(s, _)| s == 0).count();
        assert!((28..=32).contains(&c), "window has {c} fading elements");
    }
    // A hand-over: one sequence until the middle, another from it; no uniform ones at all.
    let il = Interleave::with_sampling(&[500, 500], &[Sampling::until(0.5), Sampling::delayed(0.5)]).unwrap();
    let all = full(&il);
    assert!(all[..500].iter().all(|&(s, _)| s == 0) && all[500..].iter().all(|&(s, _)| s == 1));
}

#[test]
fn uniform_sequences_absorb_the_slack() {
    // Seq 1 (30% of the elements) is delayed to 0.5: in the first half only seq 0 appears,
    // and seq 0 must be 5/7 consumed by then.
    let il = Interleave::with_sampling(&[700, 300], &[Uniform, DelayedLinear { start: 0.5, full: 0.5 }]).unwrap();
    let all = full(&il);
    let first_half = &all[..500];
    assert!(first_half.iter().all(|&(s, _)| s == 0));
    let after = &all[500..];
    // In the second half both appear, seq 1 at 300/500 = 60% share.
    let ones = after.iter().filter(|&&(s, _)| s == 1).count();
    assert_eq!(ones, 300);
    for w in after.windows(50) {
        let c = w.iter().filter(|&&(s, _)| s == 1).count();
        assert!((28..=32).contains(&c), "window has {c} delayed elements");
    }
}

#[test]
fn ramp_rate_rises_linearly() {
    // Seq 1 ramps from 0.2 to 0.6 over a joint sequence of 10 000; rate in successive
    // windows of the ramp must increase roughly linearly.
    let il = Interleave::with_sampling(&[8000, 2000], &[Uniform, DelayedLinear { start: 0.2, full: 0.6 }]).unwrap();
    let all = full(&il);
    let counts: Vec<usize> = (0..10).map(|w| all[1000 * w..1000 * (w + 1)].iter().filter(|&&(s, _)| s == 1).count()).collect();
    assert_eq!(&counts[..2], &[0, 0], "{counts:?}");
    // Ramp windows 2..6 (0.2..0.6): rate r·(τ−d0)/(d1−d0) with r = 2/1.2, final windows constant.
    let r = 2.0 / (2.0 - 0.2 - 0.6);
    for w in 2..10 {
        let tau0 = w as f64 / 10.0;
        let expect: f64 = if w < 6 { 2000.0 * ((tau0 + 0.1 - 0.2).powi(2) - (tau0 - 0.2).powi(2)) / (0.4 * 1.2) } else { 2000.0 * r * 0.1 };
        assert!((counts[w] as f64 - expect).abs() <= 3.0, "window {w}: {} vs {expect:.1} ({counts:?})", counts[w]);
    }
}

#[test]
fn rejects_bad_configurations() {
    use SamplingError::*;
    assert!(matches!(
        Interleave::with_sampling(&[100, 900], &[Uniform, DelayedLinear { start: 0.5, full: 0.5 }]),
        Err(Overcommitted { .. })
    ));
    assert!(matches!(
        Interleave::with_sampling(&[100, 100], &[DelayedLinear { start: 0.5, full: 0.5 }, DelayedLinear { start: 0.5, full: 0.5 }]),
        Err(Overcommitted { .. })
    ));
    assert!(matches!(
        Interleave::with_sampling(&[100, 900], &[Uniform, DelayedLinear { start: 0.0, full: 0.9 }]),
        Err(Overcommitted { .. })
    ));
    for bad in [
        DelayedLinear { start: 1.0, full: 1.0 },
        DelayedLinear { start: -0.1, full: -0.1 },
        DelayedLinear { start: f64::NAN, full: 0.5 },
        DelayedLinear { start: 0.5, full: 0.4 },
        DelayedLinear { start: 1.0, full: 1.0 },
        DelayedLinear { start: 0.2, full: 1.5 },
        Sampling::trapezoid(0.5, 0.5, 0.5, 0.5),
        Sampling::trapezoid(0.0, 0.0, 0.0, 0.0),
        Sampling::trapezoid(0.2, 0.1, 0.5, 0.6),
        Sampling::trapezoid(0.1, 0.2, 0.6, 0.5),
        Sampling::trapezoid(0.1, 0.2, 0.5, 1.1),
        Sampling::trapezoid(-0.1, 0.2, 0.5, 0.9),
        Sampling::trapezoid(0.1, f64::INFINITY, 0.5, 0.9),
        Sampling::until(0.0),
    ] {
        assert!(matches!(Interleave::with_sampling(&[10, 10], &[Uniform, bad]), Err(InvalidParameter { seq: 1, .. })), "{bad:?}");
    }
    // Overcommitted in the middle, although nothing is scheduled at the end.
    assert!(matches!(Interleave::with_sampling(&[600, 400], &[Sampling::until(0.5), Uniform]), Err(Overcommitted { .. })));
    assert!(Interleave::with_sampling(&[500, 500], &[Sampling::until(0.5), Uniform]).is_ok());
    assert!(matches!(
        Interleave::with_sampling(&[1 << 40, 1 << 40], &[Uniform, Sampling::trapezoid(0.0, 0.0, 0.001, 0.001)]),
        Err(TooSteep { seq: 1 })
    ));
    assert_eq!(Interleave::with_sampling(&[MAX_TOTAL_LEN, 1], &[Uniform, Uniform]).err(), Some(TooLong));
    assert!(matches!(
        Interleave::with_sampling(&[1 << 40, 1 << 40], &[Uniform, DelayedLinear { start: 0.999, full: 0.999 }]),
        Err(TooSteep { seq: 1 })
    ));
    // Schedules on empty sequences are ignored, and consistent all-scheduled setups work.
    assert!(Interleave::with_sampling(&[10, 0], &[Uniform, DelayedLinear { start: 0.999, full: 0.999 }]).is_ok());
    let il = Interleave::with_sampling(&[100, 100], &[DelayedLinear { start: 0.0, full: 0.0 }, DelayedLinear { start: 0.0, full: 0.0 }])
        .unwrap();
    assert_eq!(full(&il).len(), 200);
    // Exactly at capacity is allowed: the delayed 50% fills the whole second half.
    let il = Interleave::with_sampling(&[500, 500], &[Uniform, DelayedLinear { start: 0.5, full: 0.5 }]).unwrap();
    assert!(full(&il)[..500].iter().all(|&(s, _)| s == 0));
}

/// Empty sequences take no part in the stagger: the order is that of the non-empty ones alone.
#[test]
fn empty_sequences_do_not_affect_the_order() {
    let mut rng = Rng(0xE0E0_1234);
    for lens in uniform_cases() {
        let live: Vec<u64> = lens.iter().copied().filter(|&n| n > 0).collect();
        let mut map = Vec::new(); // index in `lens` -> index in `live`
        let mut next = 0;
        for &n in &lens {
            map.push(next);
            if n > 0 {
                next += 1;
            }
        }
        let got: Vec<(usize, u64)> = full(&Interleave::new(&lens)).into_iter().map(|(s, j)| (map[s], j)).collect();
        assert_eq!(got, full(&Interleave::new(&live)), "{lens:?}");
    }
    for (lens, sampling) in scheduled_cases() {
        let il = Interleave::with_sampling(&lens, &sampling).unwrap();
        let mut lens2 = lens.clone();
        let mut sampling2 = sampling.clone();
        let at = rng.below64(lens.len() as u64 + 1) as usize;
        lens2.insert(at, 0);
        sampling2.insert(at, if at.is_multiple_of(2) { DelayedLinear { start: 0.99, full: 0.99 } } else { Sampling::until(0.01) });
        let il2 = Interleave::with_sampling(&lens2, &sampling2).unwrap();
        let got: Vec<(usize, u64)> = full(&il2).into_iter().map(|(s, j)| (if s > at { s - 1 } else { s }, j)).collect();
        assert_eq!(got, full(&il), "{lens:?} {sampling:?} with an empty part at {at}");
    }
}

/// A seeked-again iterator is the same as a fresh one.
#[test]
fn reseeking_matches_fresh_iterators() {
    let mut rng = Rng(0x5EEC);
    for il in all_cases() {
        let n = il.len();
        let all = full(&il);
        let mut it = il.iter(0..0);
        for _ in 0..20 {
            let a = rng.below64(n + 1);
            let b = (a + rng.below64(30)).min(n);
            it.seek(a..b);
            assert_eq!(it.by_ref().collect::<Vec<_>>(), &all[a as usize..b as usize], "reseek {a}..{b}");
        }
    }
}

#[test]
fn empty_and_trivial() {
    for lens in [vec![], vec![0], vec![0, 0]] {
        let il = Interleave::new(&lens);
        assert_eq!(il.len(), 0);
        assert_eq!(il.len(), 0);
        assert_eq!(il.iter(0..0).count(), 0);
    }
    let il = Interleave::new(&[5]);
    assert_eq!(full(&il), (0..5).map(|j| (0, j)).collect::<Vec<_>>());
    assert_eq!(il.iter(2..2).count(), 0);
    assert_eq!(il.iter(3..5).collect::<Vec<_>>(), vec![(0, 3), (0, 4)]);
}

#[test]
fn huge_lengths_seek_consistently() {
    let lens = [MAX_TOTAL_LEN / 2, MAX_TOTAL_LEN / 2 - (1 << 41), 7, 1, 1 << 40];
    let sampling = [Uniform, Uniform, DelayedLinear { start: 0.5, full: 0.5 }, Uniform, DelayedLinear { start: 0.3, full: 0.7 }];
    let il = Interleave::with_sampling(&lens, &sampling).unwrap();
    let n = il.len();
    let mut rng = Rng(777);
    for _ in 0..40 {
        let a = rng.below64(n - 30_000);
        let b = a + rng.below64(10_000);
        let c = b + rng.below64(10_000);
        let whole: Vec<_> = il.iter(a..c).collect();
        let mut parts: Vec<_> = il.iter(a..b).collect();
        parts.extend(il.iter(b..c));
        assert_eq!(whole, parts, "seam at {b}");
        let mut last = std::collections::HashMap::new();
        for &(s, j) in &whole {
            if let Some(&prev) = last.get(&s) {
                assert_eq!(j, prev + 1);
            }
            last.insert(s, j);
        }
    }
    // Nothing from the delayed sequences early on.
    assert!(il.iter(0..1000).all(|(s, _)| s != 2 && s != 4));
    // The singleton (sequence 3 of 5 non-empty ones, stagger offset 0.7) sits at
    // F_U⁻¹(0.7)·N, within the k-position warp.
    let expect = (il.key(3, 0, &mut 0) * n as f64) as u64;
    let found = il.iter(expect - 100..expect + 100).any(|(s, _)| s == 3);
    assert!(found, "singleton not near {expect}");
}

/// Seek and per-element cost of the interleave alone, uniform and with schedules, by `k`.
/// `cargo test --release -- --ignored bench_seek_and_walk --nocapture`
#[test]
#[ignore = "benchmark: run with --ignored --nocapture"]
fn bench_seek_and_walk() {
    use std::hint::black_box;
    use std::time::Instant;
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    let mut rnd = move |m: u64| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        1 + x % m
    };
    for &k in &[3usize, 100, 1000, 10_000] {
        let lens: Vec<u64> = (0..k).map(|_| rnd(if k == 3 { 200 } else { 2_000_000 })).collect();
        // Uniform; a fifth scheduled with a few distinct starts; a fifth with distinct starts
        // each (the uniform profile then has a segment per scheduled part).
        for schedules in ["uniform", "scheduled", "distinct"] {
            let sampling: Vec<Sampling> = (0..k)
                .map(|i| match (schedules, i % 10) {
                    ("scheduled", 3) => DelayedLinear { start: 0.1 + 0.05 * (i % 7) as f64, full: 0.1 + 0.05 * (i % 7) as f64 },
                    ("scheduled", 7) => DelayedLinear { start: 0.05 * (i % 5) as f64, full: 0.3 + 0.05 * (i % 9) as f64 },
                    ("distinct", 3 | 7) => Sampling::delayed(0.1 + 0.6 * i as f64 / k as f64),
                    _ => Uniform,
                })
                .collect();
            let t0 = Instant::now();
            let il = Interleave::with_sampling(&lens, &sampling).unwrap();
            let build = t0.elapsed();
            let n = il.len();

            let reps = 200u64;
            let t0 = Instant::now();
            let mut acc = 0;
            for r in 0..reps {
                let a = (r * 7_919_337_017) % n;
                acc += il.iter(a..a + 1).next().unwrap().1;
            }
            let seek = t0.elapsed().as_nanos() as f64 / reps as f64;

            let len = n.min(5_000_000);
            let start = (n - len) / 2;
            let t0 = Instant::now();
            let acc2 = il.iter(start..start + len).fold(0u64, |a, (s, j)| a.wrapping_add(s as u64 ^ j));
            let per = t0.elapsed().as_nanos() as f64 / len as f64;
            black_box((acc, acc2));
            println!(
                "k = {k:>6} {schedules:<9} N = {n:>12}: build {:>7.1} µs | seek {:>9.1} µs | {per:>5.1} ns per element",
                build.as_nanos() as f64 / 1e3,
                seek / 1e3
            );
        }
    }
}
