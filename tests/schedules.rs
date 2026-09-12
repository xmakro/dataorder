//! Schedule semantics through the public API: hand-derived counts on the virtual clock,
//! the independent Python oracle's fixtures, numerical edge cases and error reporting.

mod common;

use common::check_access;
use dataorder::{ErrorKind, Order, Schedule, Seq};

fn order(parts: &[(usize, Schedule)]) -> Order<usize> {
    Order::new(Seq::mix(parts.iter().map(|&(n, schedule)| (Seq::source(n), schedule)))).unwrap()
}

fn entries(order: &Order<usize>) -> Vec<(usize, usize)> {
    order.iter().map(|item| (item.source_ordinal, item.record_index)).collect()
}

#[test]
fn delayed_source_begins_in_virtual_time_and_uniform_continues() {
    let order = order(&[(100, Schedule::Uniform), (100, Schedule::delayed(0.6))]);
    let all = entries(&order);
    // B's first key is .6 + (.75 / 100) * .4 = .603.
    // A's first 61 keys (j + .25) / 100 are below it.
    assert_eq!(all.iter().position(|&(s, _)| s == 1), Some(61));
    assert!(all[62..].iter().any(|&(s, _)| s == 0));
    assert_eq!(all.iter().filter(|&&(s, _)| s == 1).count(), 100);
    check_access(&order);
}

#[test]
fn both_constant_and_ramp_adapt_in_output_space() {
    let order = order(&[(100, Schedule::Uniform), (100, Schedule::ramp(0.0, 1.0))]);
    let all = entries(&order);
    // At virtual time .5: A = 100*t = 50, B = 100*t² = 25.
    // This is output position 75 (37.5%), not position 100 (50%).
    assert_eq!(all[..75].iter().filter(|&&(s, _)| s == 0).count(), 50);
    assert_eq!(all[..75].iter().filter(|&&(s, _)| s == 1).count(), 25);
    // At virtual time .8: A = 80, B = 64.
    assert_eq!(all[..144].iter().filter(|&&(s, _)| s == 1).count(), 64);
    check_access(&order);
}

#[test]
fn a_constant_source_can_stop_during_another_sources_ramp() {
    let order = order(&[(100, Schedule::until(0.8)), (100, Schedule::ramp(0.0, 1.0))]);
    let all = entries(&order);
    // At virtual time .8: A is done and B has supplied 100*.8² = 64 items.
    assert_eq!(all[..164].iter().filter(|&&(s, _)| s == 0).count(), 100);
    assert!(all[164..].iter().all(|&(s, _)| s == 1));
    check_access(&order);
}

#[test]
fn schedules_can_overlap_without_uniform_or_skip_empty_clock_intervals() {
    let delayed = order(&[(31, Schedule::delayed(0.8)); 3]);
    assert_eq!(entries(&delayed), entries(&order(&[(31, Schedule::Uniform); 3])));
    check_access(&delayed);

    let separated = order(&[(11, Schedule::until(0.2)), (29, Schedule::delayed(0.8))]);
    assert_eq!(entries(&separated), (0..11).map(|j| (0, j)).chain((0..29).map(|j| (1, j))).collect::<Vec<_>>());
    check_access(&separated);

    let overlapping =
        order(&[(43, Schedule::ramp(0.2, 0.9)), (37, Schedule::trapezoid(0.3, 0.4, 0.5, 0.8)), (29, Schedule::fade(0.4, 0.7))]);
    check_access(&overlapping);
    for (s, n) in [43, 37, 29].into_iter().enumerate() {
        assert_eq!(entries(&overlapping).iter().filter(|&&(i, _)| i == s).count(), n);
    }
}

#[test]
fn seeks_cover_narrow_late_support_and_nearly_coincident_boundaries() {
    let at = 0.9;
    let order = order(&[
        (21, Schedule::trapezoid(at, at, at + 1e-12, at + 1e-12)),
        (23, Schedule::trapezoid(at, at + 4e-13, at + 6e-13, at + 1e-12)),
        (17, Schedule::delayed(at + 1e-12)),
    ]);
    check_access(&order);
}

#[test]
fn sharding_parts_can_change_counts_without_schedule_capacity_errors() {
    let global = Seq::mix([(Seq::source(3), Schedule::Uniform), (Seq::source(1), Schedule::delayed(0.75))]);
    assert_eq!(Order::new(global.clone()).unwrap().len(), 4);
    let shard =
        Seq::mix([(Seq::source(3).skip(0).step_by(2), Schedule::Uniform), (Seq::source(1).skip(0).step_by(2), Schedule::delayed(0.75))]);
    assert_eq!(Order::new(shard).unwrap().len(), 3);
    for worker in 0..2 {
        assert_eq!(Order::new(global.clone().skip(worker).step_by(2)).unwrap().len(), 2);
    }
}

/// Expected positions come from rational CDFs in the independent Python oracle. Every
/// part is a bare length, so an item's source ordinal is its part index.
#[test]
fn schedules_match_independent_cdfs_and_minority_ranks() {
    let cases: serde_json::Value = serde_json::from_str(include_str!("fixtures/schedule_oracle.json")).unwrap();
    for (case, fixture) in cases.as_array().unwrap().iter().enumerate() {
        let lens = fixture["lens"].as_array().unwrap();
        if lens.iter().map(|n| n.as_u64().unwrap()).sum::<u64>() > usize::MAX as u64 {
            continue; // Large public lengths are checked on the 64-bit CI targets.
        }
        let order = Order::new(Seq::mix(lens.iter().enumerate().map(|(i, n)| {
            let schedule = match fixture["schedules"][i].as_array() {
                None => Schedule::Uniform,
                Some(p) => {
                    Schedule::trapezoid(p[0].as_f64().unwrap(), p[1].as_f64().unwrap(), p[2].as_f64().unwrap(), p[3].as_f64().unwrap())
                }
            };
            (Seq::source(n.as_u64().unwrap() as usize), schedule)
        })))
        .unwrap();
        let samples = fixture["samples"].as_array().unwrap();
        let mut walking = order.cursor(0..0).unwrap();
        let mut next_position = None;
        for sample in samples {
            let pos = sample[0].as_u64().unwrap() as usize;
            let expected = (sample[1].as_u64().unwrap() as usize, sample[2].as_u64().unwrap() as usize);
            let item = order.get(pos).unwrap();
            assert_eq!((item.source_ordinal, item.record_index), expected, "case {case}, position {pos}");
            let item = order.cursor(pos..).unwrap().next().unwrap();
            assert_eq!((item.source_ordinal, item.record_index), expected, "cursor: case {case}, position {pos}");
            // Every small fixture is a complete walk. Large fixtures contain short
            // independently generated windows: seek only across gaps, then exercise
            // the tournament and cached segment transitions with consecutive nexts.
            if next_position != Some(pos) {
                walking.reset(pos..).unwrap();
            }
            let item = walking.next().unwrap();
            assert_eq!((item.source_ordinal, item.record_index), expected, "walk: case {case}, position {pos}");
            next_position = pos.checked_add(1);
        }
        if samples.len() == order.len() {
            assert!(walking.next().is_none());
        }
    }
}

#[test]
fn finite_profiles_and_seeks_terminate() {
    for full in [f64::from_bits(1), 1e-309, 1e-308] {
        let seq = Seq::mix([(Seq::source(100), Schedule::Uniform), (Seq::source(100), Schedule::ramp(0.0, full))]);
        let error = Order::new(seq).unwrap_err();
        assert!(matches!(error.kind(), ErrorKind::InvalidSchedule { .. }));
        assert_eq!(error.path(), &[1]);
    }
    for d in [1e-8, 1e-12, 1e-16, 1e-20, 1e-300] {
        let order =
            Order::new(Seq::mix([(Seq::source(300), Schedule::Uniform), (Seq::source(100), Schedule::trapezoid(0.0, d, d, 1.0))])).unwrap();
        let all: Vec<_> = order.iter().map(|item| (*item.source, item.record_index)).collect();
        for p in 0..order.len() {
            assert_eq!(
                all[p],
                {
                    let dataorder::Item { source: &s, record_index: i, .. } = order.get(p).unwrap();
                    (s, i)
                },
                "d={d}, p={p}"
            );
            let drawn = all[..p].iter().filter(|&&(s, _)| s == 100).count();
            // In the limiting fading shape: p = 300*t + 100*(2*t-t*t).
            let t = 2.0 * p as f64 / (500.0 + (250_000.0 - 400.0 * p as f64).sqrt());
            assert!((drawn as f64 - 100.0 * (2.0 * t - t * t)).abs() < 2.0, "d={d}, p={p}, drawn={drawn}");
        }
    }
    #[cfg(target_pointer_width = "64")]
    {
        // These profiles formerly overflowed the shared remainder builder.
        let n = (1usize << 46) - 1;
        let order = Order::new(Seq::mix([(Seq::source(1), Schedule::Uniform), (Seq::source(n), Schedule::ramp(0.0, 1e-296))])).unwrap();
        let at = (n + 1) / 4;
        let window: Vec<_> = order.cursor(at - 8..at + 8).unwrap().map(|item| (*item.source, item.record_index)).collect();
        assert!(window.iter().any(|&(s, _)| s == 1));
        for (j, expected) in window.into_iter().enumerate() {
            let dataorder::Item { source: &s, record_index: i, .. } = order.get(at - 8 + j).unwrap();
            assert_eq!((s, i), expected);
        }
    }
    #[cfg(target_pointer_width = "64")]
    for n in [1_000_000_000_000usize, (1 << 46) - 1] {
        let order = Order::new(Seq::mix([(Seq::source(n), Schedule::Uniform), (Seq::source(1), Schedule::fade(0.0, 1.0))])).unwrap();
        for start in [0, n / 4, n / 2 - 16, 3 * (n / 4), n - 16] {
            for (p, dataorder::Item { source: &s, record_index: i, .. }) in (start..).zip(order.cursor(start..).unwrap().take(16)) {
                assert_eq!((s, i), {
                    let dataorder::Item { source: &s, record_index: i, .. } = order.get(p).unwrap();
                    (s, i)
                });
                // The singleton's stagger is 3/4, whose fading quantile is 1/2.
                if p < n / 2 - 2 {
                    assert_eq!((s, i), (n, p));
                }
                if p > n / 2 + 2 {
                    assert_eq!((s, i), (n, p - 1));
                }
            }
        }
    }
}

#[test]
fn schedule_errors_describe_independent_profiles() {
    use dataorder::MAX_MIX_LEN;
    // A non-finite breakpoint, breakpoints out of order, and breakpoints too close
    // together for finite profile coefficients.
    for schedule in [Schedule::delayed(f64::INFINITY), Schedule::ramp(0.5, 0.25), Schedule::ramp(0.0, f64::from_bits(1))] {
        let err = Order::new(Seq::mix([(Seq::source(1), schedule)])).unwrap_err();
        assert_eq!(err.kind(), &ErrorKind::InvalidSchedule { schedule });
        assert_eq!(err.path(), [0]);
        assert_eq!(
            err.to_string(),
            format!(
                "invalid schedule {schedule:?}: breakpoints must be finite, ordered within [0, 1], allow time at a positive rate and not lie too close together (at node 0)"
            )
        );
    }
    let seq = Seq::mix([(Seq::source(1usize << 30), Schedule::until(1e-6))]);
    let err = Order::new(seq).unwrap_err();
    let expected = ErrorKind::ScheduleTooSteep { len: 1 << 30, peak_rate: 1e6, limit: MAX_MIX_LEN };
    assert_eq!(err.kind(), &expected);
    assert_eq!(err.path(), [0]);
    assert_eq!(
        err.to_string(),
        "mix part too long for the steepness of its schedule: length 1073741824 × peak rate 1000000 exceeds 70368744177664 (at node 0)"
    );
    let mixed =
        Seq::mix([(Seq::source(10).cycle_to(3 << 28), Schedule::until(1e-6)), (Seq::source(10).cycle_to(1 << 28), Schedule::Uniform)]);
    let err = Order::new(mixed).unwrap_err();
    let expected = ErrorKind::ScheduleTooSteep { len: 3 << 28, peak_rate: 1e6, limit: MAX_MIX_LEN };
    assert_eq!(err.kind(), &expected);
    assert_eq!(err.path(), [0]);
}
