//! Public semantics of independent virtual-time curves, with hand-derived counts.
use dataorder::{Order, Sampling, Seq};

fn order(parts: &[(usize, Sampling)]) -> Order<usize> {
    Order::new(Seq::mix_with(parts.iter().map(|&(n, sampling)| (Seq::source(n), sampling)))).unwrap()
}

fn entries(order: &Order<usize>) -> Vec<(usize, usize)> {
    order.iter(..).map(|(s, j)| (order.source_index(s), j)).collect()
}

fn check_seeks(order: &Order<usize>) {
    let all = entries(order);
    let mut cursor = order.iter(0..0);
    for start in (0..=order.len()).rev() {
        let end = (start + 11).min(order.len());
        cursor.set_range(start..end);
        let window: Vec<_> = cursor.by_ref().map(|(s, j)| (order.source_index(s), j)).collect();
        assert_eq!(window, all[start..end], "seek {start}");
        if start < order.len() {
            let (s, j) = order.get(start);
            assert_eq!((order.source_index(s), j), all[start]);
        }
    }
}

#[test]
fn delayed_source_begins_in_virtual_time_and_uniform_continues() {
    let order = order(&[(100, Sampling::Uniform), (100, Sampling::delayed(0.6))]);
    let all = entries(&order);
    // B's first key is .6 + (.75 / 100) * .4 = .603.
    // A's first 61 keys (j + .25) / 100 are below it.
    assert_eq!(all.iter().position(|&(s, _)| s == 1), Some(61));
    assert!(all[62..].iter().any(|&(s, _)| s == 0));
    assert_eq!(all.iter().filter(|&&(s, _)| s == 1).count(), 100);
    check_seeks(&order);
}

#[test]
fn both_constant_and_ramp_adapt_in_output_space() {
    let order = order(&[(100, Sampling::Uniform), (100, Sampling::ramp(0.0, 1.0))]);
    let all = entries(&order);
    // At virtual time .5: A = 100*t = 50, B = 100*t² = 25.
    // This is output position 75 (37.5%), not position 100 (50%).
    assert_eq!(all[..75].iter().filter(|&&(s, _)| s == 0).count(), 50);
    assert_eq!(all[..75].iter().filter(|&&(s, _)| s == 1).count(), 25);
    // At virtual time .8: A = 80, B = 64.
    assert_eq!(all[..144].iter().filter(|&&(s, _)| s == 1).count(), 64);
    check_seeks(&order);
}

#[test]
fn a_constant_source_can_stop_during_another_sources_ramp() {
    let order = order(&[(100, Sampling::until(0.8)), (100, Sampling::ramp(0.0, 1.0))]);
    let all = entries(&order);
    // At virtual time .8: A is done and B has supplied 100*.8² = 64 items.
    assert_eq!(all[..164].iter().filter(|&&(s, _)| s == 0).count(), 100);
    assert!(all[164..].iter().all(|&(s, _)| s == 1));
    check_seeks(&order);
}

#[test]
fn schedules_can_overlap_without_uniform_or_skip_empty_clock_intervals() {
    let delayed = order(&[(31, Sampling::delayed(0.8)); 3]);
    assert_eq!(entries(&delayed), entries(&order(&[(31, Sampling::Uniform); 3])));
    check_seeks(&delayed);

    let separated = order(&[(11, Sampling::until(0.2)), (29, Sampling::delayed(0.8))]);
    assert_eq!(entries(&separated), (0..11).map(|j| (0, j)).chain((0..29).map(|j| (1, j))).collect::<Vec<_>>());
    check_seeks(&separated);

    let overlapping =
        order(&[(43, Sampling::ramp(0.2, 0.9)), (37, Sampling::trapezoid(0.3, 0.4, 0.5, 0.8)), (29, Sampling::fading(0.4, 0.7))]);
    check_seeks(&overlapping);
    for (s, n) in [43, 37, 29].into_iter().enumerate() {
        assert_eq!(entries(&overlapping).iter().filter(|&&(i, _)| i == s).count(), n);
    }
}

#[test]
fn seeks_cover_narrow_late_support_and_nearly_coincident_boundaries() {
    let at = 0.9;
    let order = order(&[
        (21, Sampling::trapezoid(at, at, at + 1e-12, at + 1e-12)),
        (23, Sampling::trapezoid(at, at + 4e-13, at + 6e-13, at + 1e-12)),
        (17, Sampling::delayed(at + 1e-12)),
    ]);
    check_seeks(&order);
}
