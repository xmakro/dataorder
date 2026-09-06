//! Pin large-domain outputs from commit 837f3b4, independent of current get/iteration agreement.
use dataorder::{Order, Sampling, Seq, Source};
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
        Seq::mix([source(0, 1 << 30).shuffle(11).repeat(1 << 15), source(1, 1 << 30).shuffle(22).repeat(1 << 15)]),
        Seq::mix_with([
            (source(0, 3 * (1 << 28)).shuffle(11).repeat(1 << 16), Sampling::Uniform),
            (source(1, 1 << 27).shuffle(22).repeat(1 << 16), Sampling::trapezoid(0.1, 0.3, 0.6, 0.9)),
            (source(2, 1 << 27).shuffle(33).repeat(1 << 16), Sampling::delayed(0.5)),
        ]),
        source(3, 777).shuffle(44).repeat(1 << 30).repeat(1 << 20),
        source(4, 1 << 30).repeat(1 << 30).shuffle(55),
    ];
    let seeds = [42, 19, 7, 5];
    for fixture in include_str!("fixtures/large_orders.txt").lines().filter(|line| !line.starts_with('#')) {
        let fields: Vec<u64> = fixture.split('|').map(|s| s.parse().unwrap()).collect();
        let (case, pos) = (fields[0] as usize, fields[1]);
        let expected = (fields[2] as usize, fields[3] as usize);
        // An outer stride addresses u64 intermediate positions on 32-bit hosts too.
        // It adds no repeat context and changes no shuffle or mix keys. Both the
        // resulting order length and each original source index fit in 32 bits.
        let step = 1usize << 30;
        let order = Order::with_seed(cases[case].clone().stride(step, (pos % step as u64) as usize), seeds[case]).unwrap();
        let at = (pos / step as u64) as usize;
        let (s, i) = order.get(at);
        assert_eq!((s.id, i), expected, "get: {fixture}");
        let (_, s, i) = order.iter(at..).indexed().next().unwrap();
        assert_eq!((s.id, i), expected, "cursor: {fixture}");
    }
}
