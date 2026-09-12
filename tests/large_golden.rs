//! Pin large-domain outputs across shuffles, epochs and virtual-clock schedules.
//! Fixture headers identify the ordering versions; these checks are independent of
//! current get/iteration agreement.
#![cfg(target_pointer_width = "64")]

use dataorder::{Order, Schedule, Seq, Source};
#[derive(Clone)]
struct Src {
    id: usize,
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
fn source(id: usize, len: usize) -> Seq<Src> {
    Seq::source(Src { id, len })
}

#[test]
fn large_orders_keep_baseline_outputs() {
    let cases = [
        Seq::mix([source(0, 1 << 30).shuffle().repeat(1 << 15), source(1, 1 << 30).shuffle().repeat(1 << 15)]),
        Seq::mix([
            (source(0, 3 * (1 << 28)).shuffle().repeat(1 << 16), Schedule::Uniform),
            (source(1, 1 << 27).shuffle().repeat(1 << 16), Schedule::trapezoid(0.1, 0.3, 0.6, 0.9)),
            (source(2, 1 << 27).shuffle().repeat(1 << 16), Schedule::delayed(0.5)),
        ]),
        source(3, 777).shuffle().repeat(1 << 30).repeat(1 << 20),
        source(4, 1 << 30).repeat(1 << 30).shuffle(),
    ];
    let seeds = [42, 19, 7, 5];
    for fixture in include_str!("fixtures/large_orders.txt").lines().filter(|line| !line.starts_with('#')) {
        let fields: Vec<u64> = fixture.split('|').map(|s| s.parse().unwrap()).collect();
        let (case, pos) = (fields[0] as usize, fields[1] as usize);
        let expected = (fields[2] as usize, fields[3] as usize);
        let order = Order::with_seed(cases[case].clone(), seeds[case]).unwrap();
        let item = order.get(pos).unwrap();
        assert_eq!((item.source.id, item.record_index), expected, "get: {fixture}");
        let item = order.cursor(pos..).unwrap().next().unwrap();
        assert_eq!((item.source.id, item.record_index), expected, "cursor: {fixture}");
    }
}
