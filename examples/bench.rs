//! Cost of typical orders: seek = `order.iter_from(pos)` at a random position (positions a
//! cursor, allocating it), walk = one element of that cursor after the seek, get =
//! `order.get(pos)` at a random position.
//! `cargo run --release --example bench`
use dataorder::{Order, Sampling, Seq};
use std::hint::black_box;
use std::time::Instant;

fn measure(name: &str, seq: Seq<usize>, count: usize) {
    let order = Order::compile(seq).unwrap();
    let n = order.len();
    let seeks = 200;
    let t = Instant::now();
    for i in 0..seeks {
        black_box(order.iter_from(n / seeks * i + 1));
    }
    let seek_us = t.elapsed().as_nanos() as f64 / seeks as f64 / 1000.0;

    let mut acc = 0u64;
    let start = n / 3;
    let count = count.min(n - start);
    let t = Instant::now();
    for (s, i) in order.iter(start..start + count) {
        acc = acc.wrapping_add((*s ^ i) as u64);
    }
    let stream_ns = t.elapsed().as_nanos() as f64 / count as f64;

    let gets = count.min(200_000);
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let t = Instant::now();
    for _ in 0..gets {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let (s, i) = order.get(((x >> 11) % n as u64) as usize);
        acc = acc.wrapping_add((*s ^ i) as u64);
    }
    let get_ns = t.elapsed().as_nanos() as f64 / gets as f64;
    black_box(acc);
    println!("{name:<44} {seek_us:>8.2} µs {stream_ns:>11.1} ns {get_ns:>9.1} ns");
}

fn main() {
    // Sources are bare lengths here; the id only keeps the calls readable.
    let src = |_id: u32, len: usize| Seq::source(len);
    let shuffled = |k: u32, len: usize| (0..k).map(move |i| src(i, len).shuffle(i as u64 + 1));
    let scheduled = |k: u32, len: usize| {
        (0..k).map(move |i| {
            let s = if i % 5 == 0 { Sampling::DelayedLinear(0.2 + 0.05 * (i % 7) as f64, 0.7) } else { Sampling::Uniform };
            (src(i, len).shuffle(i as u64 + 1), s)
        })
    };
    let m = 1_000_000usize;
    // A realistic training order: two mixes of shuffled sources, 2–4 epochs each, mixed.
    let epochs = |k: u32, first_id: u32| {
        Seq::mix((0..k).map(move |i| {
            let h = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let len = 500_000 + ((h >> 40) % 1_500_000) as usize;
            src(first_id + i, len).shuffle(h).repeat(2 + ((h >> 20) % 3) as usize)
        }))
    };
    let realistic = Seq::mix([epochs(1000, 0), epochs(100, 1000)]);
    println!("{:<44} {:>11} {:>14} {:>12}", "order", "seek", "walk / elem", "get(pos)");
    measure("mix(mix(1000 × shuffled × 2–4 epochs), mix(100 × …))", realistic, 5 * m);
    measure("source 1e9", src(0, 1000 * m), 5 * m);
    measure("shuffle(source 1e9)", src(0, 1000 * m).shuffle(1), 5 * m);
    measure("shuffle(source 1e6).repeat(1000)", src(0, m).shuffle(1).repeat(1000), 5 * m);
    measure("concat(100 × shuffle(source 1e6))", Seq::concat(shuffled(100, m)), 5 * m);
    measure("shuffle(concat(100 × source 1e6))", Seq::concat((0..100).map(|i| src(i, m))).shuffle(7), 5 * m);
    measure("mix(5 × source 1e6)", Seq::mix((0..5).map(|i| src(i, m))), 5 * m);
    measure("mix(80% source + 4 × 5%)", Seq::mix([src(0, 80 * m), src(1, 5 * m), src(2, 5 * m), src(3, 5 * m), src(4, 5 * m)]), 5 * m);
    measure("mix(60% shuffled + 9 × 4.4% shuffled)", Seq::mix((0..10).map(|i| src(i, if i == 0 { 60 * m } else { 4_444_444 }).shuffle(i as u64 + 1))), 5 * m);
    measure("mix(100 × source 1e6)", Seq::mix((0..100).map(|i| src(i, m))), 5 * m);
    measure("mix(100 × shuffled)", Seq::mix(shuffled(100, m)), 5 * m);
    measure("mix(100 × shuffled, 20% scheduled)", Seq::mix_with(scheduled(100, m)), 5 * m);
    measure("mix(1000 × shuffled, 20% scheduled)", Seq::mix_with(scheduled(1000, 100_000)), 5 * m);
    measure("↑ .shard(0, 8)", Seq::mix_with(scheduled(100, m)).shard(0, 8), m);
    measure("↑ .shard(0, 512)", Seq::mix_with(scheduled(100, m)).shard(0, 512), 100_000);
    let nested = Seq::mix_with([
        (Seq::concat([src(0, m).shuffle(1), src(1, m).shuffle(2)]).shuffle(3), Sampling::Uniform),
        (src(2, m).shuffle(4).take(500_000), Sampling::DelayedLinear(0.5, 0.5)),
        (src(3, m).shuffle(5).repeat(2), Sampling::DelayedLinear(0.1, 0.4)),
    ])
    .repeat(3)
    .shard(1, 4);
    measure("repeat(3, mix(3 nested)).shard(1, 4)", nested, m);
    measure("shuffle(mix(100 × source 1e6))  [slow path]", Seq::mix((0..100).map(|i| src(i, m))).shuffle(9), 100_000);
}
