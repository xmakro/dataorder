//! Pin large-domain outputs across shuffles, epochs and virtual-clock schedules.
//! Fixture headers identify the ordering versions; these checks are independent of
//! current get/iteration agreement.
#![cfg(target_pointer_width = "64")]

mod common;

use common::src;
use dataorder::{Order, Schedule, Seq};

#[test]
fn large_orders_keep_baseline_outputs() {
    let cases = [
        Seq::mix([src(0, 1 << 30).shuffle().repeat(1 << 15), src(1, 1 << 30).shuffle().repeat(1 << 15)]),
        Seq::mix([
            (src(0, 3 * (1 << 28)).shuffle().repeat(1 << 16), Schedule::Uniform),
            (src(1, 1 << 27).shuffle().repeat(1 << 16), Schedule::trapezoid(0.1, 0.3, 0.6, 0.9)),
            (src(2, 1 << 27).shuffle().repeat(1 << 16), Schedule::delayed(0.5)),
        ]),
        src(3, 777).shuffle().repeat(1 << 30).repeat(1 << 20),
        src(4, 1 << 30).repeat(1 << 30).shuffle(),
    ];
    let seeds = [42, 19, 7, 5];
    for fixture in include_str!("fixtures/large_orders.txt").lines().filter(|line| !line.starts_with('#')) {
        let fields: Vec<u64> = fixture.split('|').map(|s| s.parse().unwrap()).collect();
        let (case, pos) = (fields[0] as usize, fields[1] as usize);
        let expected = (fields[2], fields[3] as usize);
        let order = Order::with_seed(cases[case].clone(), seeds[case]).unwrap();
        let item = order.get(pos).unwrap();
        assert_eq!((item.source.id, item.record_index), expected, "get: {fixture}");
        let item = order.cursor(pos..).unwrap().next().unwrap();
        assert_eq!((item.source.id, item.record_index), expected, "cursor: {fixture}");
    }
}
