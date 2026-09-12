//! Whole-crate tests: the merge against a brute-force sort, permutation and order
//! properties, exact seeks, balance bounds and the schedule semantics.

use super::*;
use crate::ErrorKind;
use crate::tests::Rng;
use Schedule::*;

fn full(il: &Interleave) -> Vec<(usize, usize)> {
    il.iter(0..il.len()).collect()
}

/// Builds the interleave the compiler would: every part's schedule is validated
/// against its length, and only the non-empty parts take part.
fn build(lens: &[usize], schedules: &[Schedule]) -> Result<Interleave, ErrorKind> {
    let mut parts = Vec::new();
    for (&n, schedule) in lens.iter().zip(schedules) {
        let profile = schedule.profile(n)?;
        if n > 0 {
            parts.push((n, profile));
        }
    }
    Ok(Interleave::new(parts))
}

fn uniform(lens: &[usize]) -> Interleave {
    build(lens, &vec![Uniform; lens.len()]).unwrap()
}

/// Brute force: sort every element by (key, part, index).
fn reference(il: &Interleave) -> Vec<(usize, usize)> {
    let mut all = Vec::new();
    for (s, part) in il.parts.iter().enumerate() {
        for j in 0..part.n {
            all.push((il.key(s, j, &mut 0), s, j));
        }
    }
    all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    all.into_iter().map(|(_, s, j)| (s, j)).collect()
}

/// Share of part `s` that its share function prescribes at `t`.
fn share(il: &Interleave, s: usize, t: f64) -> f64 {
    il.profile(s).share(t).clamp(0.0, 1.0)
}

/// Largest `|count_s(t) − n_s·F_s(t/N)|` over all `t`, per part (in elements of `s`).
fn worst_deviation(il: &Interleave, all: &[(usize, usize)]) -> Vec<f64> {
    let k = il.parts.len();
    let n = il.len() as f64;
    let mut counts = vec![0usize; k];
    let mut worst = vec![0.0f64; k];
    for t in 0..=all.len() {
        for s in 0..k {
            let ideal = il.parts[s].n as f64 * share(il, s, t as f64 / n);
            worst[s] = worst[s].max((counts[s] as f64 - ideal).abs());
        }
        if t < all.len() {
            counts[all[t].0] += 1;
        }
    }
    worst
}

fn random_lens(rng: &mut Rng) -> Vec<usize> {
    let k = rng.below(9);
    (0..k)
        .map(|_| match rng.below(4) {
            0 => 0,
            1 => rng.below(4),
            2 => rng.below(30),
            _ => rng.below(300),
        })
        .collect()
}

fn uniform_cases() -> Vec<Vec<usize>> {
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

fn scheduled_cases() -> Vec<(Vec<usize>, Vec<Schedule>)> {
    vec![
        (vec![1000, 300], vec![Uniform, Schedule::delayed(0.4)]),
        (vec![2000, 500], vec![Uniform, Schedule::ramp(0.2, 0.6)]),
        (vec![14, 124, 43], vec![Schedule::delayed(0.5), Uniform, Schedule::ramp(0.2, 0.6)]),
        (vec![500, 500, 500, 500], vec![Uniform, Schedule::delayed(0.1), Schedule::delayed(0.3), Schedule::ramp(0.0, 0.5)]),
        (vec![300, 300, 7], vec![Schedule::delayed(0.0), Schedule::delayed(0.0), Uniform]),
        (vec![1000, 50, 50], vec![Uniform, Schedule::delayed(0.9), Schedule::ramp(0.8, 0.95)]),
        (vec![100, 100, 800], vec![Schedule::ramp(0.0, 0.5), Schedule::ramp(0.5, 1.0), Uniform]),
        (vec![3, 1000, 1], vec![Schedule::delayed(0.7), Uniform, Schedule::ramp(0.2, 0.9)]),
        (vec![300, 700], vec![Schedule::until(0.5), Uniform]),
        (vec![200, 300, 500], vec![Schedule::fade(0.2, 0.6), Schedule::trapezoid(0.3, 0.5, 0.7, 0.9), Uniform]),
        (vec![500, 500], vec![Schedule::until(0.5), Schedule::delayed(0.5)]),
        (vec![40, 60, 900], vec![Schedule::trapezoid(0.1, 0.1, 0.1, 0.3), Schedule::trapezoid(0.6, 0.8, 0.8, 1.0), Uniform]),
    ]
}

fn all_cases() -> Vec<Interleave> {
    let mut v: Vec<Interleave> = uniform_cases().iter().map(|l| uniform(l)).collect();
    v.extend(scheduled_cases().iter().map(|(l, s)| build(l, s).unwrap()));
    v
}

#[test]
fn independent_constant_profiles_resolve_an_exact_tie() {
    // The singleton key is 1/4. The other part's first key is also 1/4;
    // the lower part index wins, independently of the other part's support.
    let il = build(&[1, 3], &[Uniform, Schedule::trapezoid(0.0625, 0.0625, 0.8125, 0.8125)]).unwrap();
    let expected = [(0, 0), (1, 0), (1, 1), (1, 2)];
    assert_eq!(full(&il), expected);
    for start in 0..=il.len() {
        assert_eq!(il.iter(start..il.len()).collect::<Vec<_>>(), expected[start..]);
    }
}

#[test]
fn matches_brute_force_sort() {
    for il in all_cases() {
        let expect = reference(&il);
        assert_eq!(expect.len(), il.len());
        assert_eq!(full(&il), expect, "{il:?}");
    }
}

#[test]
fn is_a_permutation_preserving_order() {
    for il in all_cases() {
        let mut next = vec![0usize; il.parts.len()];
        for (t, (s, j)) in full(&il).into_iter().enumerate() {
            assert_eq!(j, next[s], "pos {t}");
            next[s] += 1;
        }
        let lens: Vec<usize> = il.parts.iter().map(|s| s.n).collect();
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
            let a = rng.below(n + 1);
            let b = a + rng.below(n + 1 - a);
            let got: Vec<_> = il.iter(a..b).collect();
            assert_eq!(got, &all[a..b], "range {a}..{b}");
            assert_eq!(il.iter(a..b).size_hint(), (b - a, Some(b - a)));
        }
        // Random partitions tile the full sequence exactly.
        let mut cuts: Vec<usize> = (0..5).map(|_| rng.below(n + 1)).collect();
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
        let starts: Vec<usize> = if n <= 3000 { (0..=n).collect() } else { (0..300).map(|_| rng.below(n + 1)).collect() };
        for a in starts {
            let b = (a + 1 + rng.below(40)).min(n);
            let got: Vec<_> = il.iter(a..b).collect();
            assert_eq!(got, &all[a..b], "seek to {a}");
        }
    }
}

#[test]
fn keys_are_monotone_within_sequences() {
    for il in all_cases() {
        for (s, part) in il.parts.iter().enumerate() {
            let mut seg = 0;
            let mut prev = f64::NEG_INFINITY;
            for j in 0..part.n {
                let key = il.key(s, j, &mut seg);
                assert!(key >= prev, "part {s} index {j}: {key} < {prev}");
                prev = key;
            }
        }
    }
}

#[test]
fn random_configurations() {
    // Random lengths and schedules: the merge is a permutation in order, equals the
    // brute-force sort, keys are monotone, every seek matches the slice, and nothing of a
    // scheduled part appears before its start.
    let mut rng = Rng(0xC0FF_EE00_1234_5678);
    let mut checked = 0;
    while checked < 2000 {
        let k = 1 + rng.below(12);
        let lens: Vec<usize> = (0..k)
            .map(|_| match rng.below(3) {
                0 => rng.below(3),
                1 => rng.below(50),
                _ => rng.below(400),
            })
            .collect();
        let schedule: Vec<Schedule> = (0..k)
            .map(|_| {
                let d0 = rng.below(999);
                let d1 = if rng.below(2) == 0 { d0 } else { d0 + rng.below(1001 - d0) };
                match rng.below(4) {
                    0 => Uniform,
                    1 => {
                        let d2 = d1 + rng.below(1001 - d1);
                        let d3 = if rng.below(2) == 0 { d2 } else { d2 + rng.below(1001 - d2) };
                        if d0 + d1 < d2 + d3 {
                            Schedule::trapezoid(d0 as f64 / 1000.0, d1 as f64 / 1000.0, d2 as f64 / 1000.0, d3 as f64 / 1000.0)
                        } else {
                            Uniform
                        }
                    }
                    _ => Schedule::ramp(d0 as f64 / 1000.0, d1 as f64 / 1000.0),
                }
            })
            .collect();
        // Empty parts never reach the interleave: the compiler drops them first.
        let (lens, schedule): (Vec<usize>, Vec<Schedule>) = lens.into_iter().zip(schedule).filter(|&(n, _)| n > 0).unzip();
        let k = lens.len();
        let il = build(&lens, &schedule).unwrap_or_else(|e| panic!("{lens:?} {schedule:?}: {e}"));
        checked += 1;
        let n = il.len();
        let all = full(&il);
        assert_eq!(all, reference(&il), "{lens:?} {schedule:?}");
        let mut next = vec![0usize; k];
        for &(s, j) in &all {
            assert_eq!(j, next[s]);
            next[s] += 1;
        }
        assert_eq!(next, lens);
        for (s, part) in il.parts.iter().enumerate() {
            let mut seg = 0;
            let mut prev = -1.0;
            for j in 0..part.n {
                let key = il.key(s, j, &mut seg);
                assert!(key.is_finite() && (0.0..=1.0).contains(&key) && key >= prev, "{lens:?} {schedule:?} part {s} j {j}");
                prev = key;
            }
        }
        for _ in 0..20 {
            let a = rng.below(n + 1);
            let b = (a + rng.below(60)).min(n);
            assert_eq!(il.iter(a..b).collect::<Vec<_>>(), &all[a..b], "{lens:?} {schedule:?} seek {a}");
        }
        for (s, samp) in schedule.iter().enumerate() {
            let (start, off) = match *samp {
                Trapezoid { start, off, .. } => (start, off),
                Uniform => continue,
            };
            if let Some(first) = all.iter().position(|&(x, _)| x == s) {
                assert!(first as f64 >= joint_count(&il, start) - k as f64 - 1.0, "{lens:?} {schedule:?} part {s} first at {first}");
            }
            if let Some(last) = all.iter().rposition(|&(x, _)| x == s) {
                assert!(last as f64 <= joint_count(&il, off) + k as f64 + 1.0, "{lens:?} {schedule:?} part {s} last at {last}");
            }
        }
    }
}

#[test]
fn uniform_balance() {
    for lens in uniform_cases() {
        let il = uniform(&lens);
        let worst = worst_deviation(&il, &full(&il));
        for (s, w) in worst.iter().enumerate() {
            assert!(*w <= 2.0, "lens {lens:?} part {s}: deviation {w}");
        }
    }
}

#[test]
fn equal_lengths_round_robin_in_input_order() {
    for k in 1..=17usize {
        let il = uniform(&vec![6usize; k]);
        for (t, (s, j)) in full(&il).into_iter().enumerate() {
            assert_eq!(s, t % k);
            assert_eq!(j, t / k);
        }
    }
}

/// Expected prefix size on the shared virtual clock, before discrete rounding.
fn joint_count(il: &Interleave, t: f64) -> f64 {
    il.parts.iter().enumerate().map(|(s, part)| part.n as f64 * share(il, s, t)).sum()
}

#[test]
fn schedules_are_followed_on_the_virtual_clock() {
    for (lens, schedule) in scheduled_cases() {
        let il = build(&lens, &schedule).unwrap();
        let all = full(&il);
        for step in 0..=100 {
            let t = step as f64 / 100.0;
            let prefix = all.partition_point(|&(s, j)| il.key(s, j, &mut 0) < t);
            let mut counts = vec![0; lens.len()];
            for &(s, _) in &all[..prefix] {
                counts[s] += 1;
            }
            for (s, &n) in lens.iter().enumerate() {
                let expected = n as f64 * share(&il, s, t);
                assert!((counts[s] as f64 - expected).abs() <= 1.0 + 1e-10, "{lens:?} {schedule:?} part {s} at virtual time {t}");
            }
        }
    }
}

#[test]
fn uniform_and_explicit_constant_schedules_are_interchangeable() {
    for constant in [Uniform, Schedule::delayed(0.0), Schedule::until(1.0)] {
        let lens = [300, 200, 100];
        let a = build(&lens, &[Uniform, Schedule::ramp(0.2, 0.6), Schedule::until(0.8)]).unwrap();
        let b = build(&lens, &[constant, Schedule::ramp(0.2, 0.6), Schedule::until(0.8)]).unwrap();
        assert_eq!(full(&a), full(&b));
        // Changing another source's schedule does not alter this source's keys.
        let c = build(&lens, &[constant, Schedule::until(0.1), Schedule::delayed(0.9)]).unwrap();
        for j in 0..lens[0] {
            assert_eq!(a.key(0, j, &mut 0), c.key(0, j, &mut 0));
        }
    }
}

#[test]
fn rejects_bad_configurations() {
    for bad in [
        Schedule::delayed(1.0),
        Schedule::delayed(-0.1),
        Schedule::ramp(f64::NAN, 0.5),
        Schedule::ramp(0.5, 0.4),
        Schedule::ramp(1.0, 1.0),
        Schedule::ramp(0.2, 1.5),
        Schedule::trapezoid(0.5, 0.5, 0.5, 0.5),
        Schedule::trapezoid(0.0, 0.0, 0.0, 0.0),
        Schedule::trapezoid(0.2, 0.1, 0.5, 0.6),
        Schedule::trapezoid(0.1, 0.2, 0.6, 0.5),
        Schedule::trapezoid(0.1, 0.2, 0.5, 1.1),
        Schedule::trapezoid(-0.1, 0.2, 0.5, 0.9),
        Schedule::trapezoid(0.1, f64::INFINITY, 0.5, 0.9),
        Schedule::until(0.0),
    ] {
        // Breakpoints are validated whether or not the part has elements.
        for len in [10, 0] {
            assert!(matches!(bad.profile(len), Err(ErrorKind::InvalidSchedule { .. })), "{bad:?} over {len} elements");
        }
    }
    // Overlapping schedules and gaps impose no shared capacity constraint.
    assert_eq!(build(&[600, 400], &[Schedule::until(0.5), Uniform]).unwrap().len(), 1000);
    assert_eq!(build(&[500, 500], &[Schedule::until(0.5), Uniform]).unwrap().len(), 1000);
    for steep in [Schedule::trapezoid(0.0, 0.0, 1e-6, 1e-6), Schedule::delayed(0.999999)] {
        assert!(matches!(steep.profile(1 << 30), Err(ErrorKind::ScheduleTooSteep { len, .. }) if len == 1 << 30), "{steep:?}");
        // Numerical limits concern elements, so an empty part is never too steep.
        assert!(matches!(steep.profile(0), Ok(None)), "{steep:?}");
    }
    // A schedule on an empty part is validated but draws from no profile; uniform parts
    // share one profile, and consistent all-scheduled setups work.
    assert!(matches!(Schedule::delayed(0.999).profile(0), Ok(None)));
    assert!(matches!(Uniform.profile(10), Ok(None)));
    assert!(matches!(Schedule::delayed(0.5).profile(10), Ok(Some(_))));
    let il = build(&[100, 100], &[Schedule::delayed(0.0), Schedule::delayed(0.0)]).unwrap();
    assert_eq!(full(&il).len(), 200);
    // A delay halfway along the virtual clock begins one quarter through this output.
    let il = build(&[500, 500], &[Uniform, Schedule::delayed(0.5)]).unwrap();
    assert_eq!(full(&il).iter().position(|&(s, _)| s == 1), Some(251));
}

/// Repositioning an iterator gives the same elements as creating a fresh one.
#[test]
fn reseeking_matches_fresh_iterators() {
    let mut rng = Rng(0x5EEC);
    for il in all_cases() {
        let n = il.len();
        let all = full(&il);
        let mut it = il.iter(0..0);
        for _ in 0..20 {
            let a = rng.below(n + 1);
            let b = (a + rng.below(30)).min(n);
            it.seek(a..b);
            assert_eq!(it.by_ref().collect::<Vec<_>>(), &all[a..b], "reseek {a}..{b}");
        }
    }
}

#[test]
fn empty_and_trivial() {
    for lens in [vec![], vec![0], vec![0, 0]] {
        let il = uniform(&lens);
        assert_eq!(il.len(), 0);
        assert_eq!(il.len(), 0);
        assert_eq!(il.iter(0..0).count(), 0);
    }
    let il = uniform(&[5]);
    assert_eq!(full(&il), (0..5).map(|j| (0, j)).collect::<Vec<_>>());
    assert_eq!(il.iter(2..2).count(), 0);
    assert_eq!(il.iter(3..5).collect::<Vec<_>>(), vec![(0, 3), (0, 4)]);
}

#[test]
fn huge_lengths_seek_consistently() {
    let max_len = usize::try_from(MAX_TOTAL_LEN).unwrap_or(usize::MAX);
    let lens = [max_len / 2, max_len / 2 - max_len / 32, 7, 1, max_len / 64];
    let schedule = [Uniform, Uniform, Schedule::delayed(0.5), Uniform, Schedule::ramp(0.3, 0.7)];
    let il = build(&lens, &schedule).unwrap();
    let n = il.len();
    let mut rng = Rng(777);
    for _ in 0..40 {
        let a = rng.below(n - 30_000);
        let b = a + rng.below(10_000);
        let c = b + rng.below(10_000);
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
    // Project the singleton's virtual key through the whole mixture's CDF.
    let expect = joint_count(&il, il.key(3, 0, &mut 0)) as usize;
    let found = il.iter(expect - 100..expect + 100).any(|(s, _)| s == 3);
    assert!(found, "singleton not near {expect}");
}
