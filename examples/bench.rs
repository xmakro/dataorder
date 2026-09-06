//! Cost of typical orders: seek = `order.iter(pos..).next()` at a random position,
//! including cursor construction. Walk is the average per element over a long range,
//! including its initial seek; get = `order.get(pos)` at a random position. All
//! measurements use source lengths only and exclude record I/O.
//! `cargo run --release --example bench`
//! `-- --phases` measures early, rising, falling and exhausted schedule phases; `-- --lifecycle`
//! measures compilation, reusable seeks and requested cursor-allocation bytes. `-- --all`
//! includes both alongside the original table. Allocation bytes are cumulative requests,
//! not retained memory or RSS; compilation excludes cloning the input configuration.
use dataorder::{Order, Sampling, Seq};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::time::Instant;

struct Counting;
thread_local! {
    static COUNT_BYTES: Cell<bool> = const { Cell::new(false) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = COUNT_BYTES.try_with(|enabled| {
            if enabled.get() {
                BYTES.with(|bytes| bytes.set(bytes.get() + layout.size()));
            }
        });
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let _ = COUNT_BYTES.try_with(|enabled| {
            if enabled.get() {
                BYTES.with(|bytes| bytes.set(bytes.get() + layout.size()));
            }
        });
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = COUNT_BYTES.try_with(|enabled| {
            if enabled.get() {
                BYTES.with(|bytes| bytes.set(bytes.get() + new_size));
            }
        });
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn measure(name: &str, seq: Seq<usize>, count: usize) {
    measure_at(name, seq, count, None, false);
}

fn measure_at(name: &str, seq: Seq<usize>, count: usize, start: Option<usize>, continuous: bool) {
    let focused = start.is_some();
    let order = Order::new(seq).unwrap();
    let n = order.len();
    let seeks = 200;
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let t = Instant::now();
    for _ in 0..seeks {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let pos = ((x >> 11) % n as u64) as usize;
        let _ = black_box(order.iter(pos..).next());
    }
    let seek_us = t.elapsed().as_nanos() as f64 / seeks as f64 / 1000.0;

    let mut acc = 0u64;
    let start = start.unwrap_or(n / 3);
    let count = count.min(n - start);
    // Enter early parts by actually walking them: nth/seek would rebuild the tournament
    // and hide the cost of retaining exhausted leaves during an uninterrupted stream.
    let mut ongoing = continuous.then(|| order.iter(..));
    if let Some(cursor) = &mut ongoing {
        for _ in 0..start {
            black_box(cursor.next());
        }
    }
    let t = Instant::now();
    let cursor = ongoing.get_or_insert_with(|| order.iter(start..start + count));
    for (s, i) in cursor.take(count) {
        acc = acc.wrapping_add((*s ^ i) as u64);
    }
    let stream_ns = t.elapsed().as_nanos() as f64 / count as f64;

    let gets = count.min(if focused { 2000 } else { 200_000 });
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let t = Instant::now();
    for _ in 0..gets {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let (s, i) = order.get(((x >> 11) % n as u64) as usize);
        acc = acc.wrapping_add((*s ^ i) as u64);
    }
    let get_ns = t.elapsed().as_nanos() as f64 / gets as f64;
    black_box(acc);
    println!("{name:<44} {seek_us:>11.5} µs {stream_ns:>13.3} ns {get_ns:>11.3} ns");
}

fn phases() {
    let scheduled = |sampling| {
        Seq::mix_with((0..1000).map(|i| (Seq::source(10_000).shuffle(i + 1), if i % 5 == 0 { sampling } else { Sampling::Uniform })))
    };
    for (phase, start) in [("early", 0), ("rising", 3_000_000), ("late rise", 6_000_000), ("full", 8_000_000)] {
        measure_at(&format!("phase ramp: {phase}"), scheduled(Sampling::ramp(0.2, 0.7)), 200_000, Some(start), false);
    }
    for (phase, start) in [("full", 1_000_000), ("falling", 5_000_000), ("finished", 9_000_000)] {
        measure_at(&format!("phase fading: {phase}"), scheduled(Sampling::fading(0.3, 0.8)), 200_000, Some(start), false);
    }
    for k in [100, 1000, 10_000] {
        let seq = || {
            Seq::mix_with(
                std::iter::once((Seq::source(10_000_000), Sampling::Uniform))
                    .chain((0..k).map(|_| (Seq::source(1), Sampling::until(0.002)))),
            )
        };
        for (kind, continuous) in [("fresh", false), ("continuous", true)] {
            measure_at(&format!("exhausted {k} parts: {kind} tail"), seq(), 2_000_000, Some(50_000), continuous);
        }
    }
}

fn lifecycle(name: &str, seq: Seq<usize>, warmup: usize) {
    let mut elapsed = 0;
    for _ in 0..20 {
        let input = seq.clone();
        let t = Instant::now();
        let order = black_box(Order::new(input).unwrap());
        elapsed += t.elapsed().as_nanos();
        drop(order);
    }
    let build_us = elapsed as f64 / 20.0 / 1000.0;
    let order = Order::new(seq).unwrap();
    BYTES.set(0);
    COUNT_BYTES.set(true);
    let mut cursor = order.iter(..);
    cursor.by_ref().take(warmup).for_each(|item| {
        black_box(item);
    });
    COUNT_BYTES.set(false);
    let bytes = BYTES.get();
    // The warmup enters every part in these all-uniform configurations. Measure both
    // the known-zero rank and random ranks without allocating another cursor.
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let t = Instant::now();
    for i in 0..400 {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let pos = if i % 2 == 0 { 0 } else { ((x >> 11) % order.len() as u64) as usize };
        cursor.seek(pos);
        black_box(cursor.next());
    }
    let reuse_us = t.elapsed().as_nanos() as f64 / 400.0 / 1000.0;
    println!("lifecycle {name:<44} {build_us:>11.5} µs {reuse_us:>11.5} µs {bytes:>10} B");
}

fn lifecycles() {
    for k in [100, 1000, 10_000] {
        lifecycle(&format!("mix({k} sources)"), Seq::mix((0..k).map(|_| Seq::source(10_000))), 2 * k);
    }
    for k in [1000, 10_000] {
        lifecycle(
            &format!("weighted({k} sources)"),
            Seq::weighted(k * 10_000, (0..k).map(|i| (Seq::source(10_000), (i % 13 + 1) as f64))),
            20 * k,
        );
    }
    lifecycle("mix(100 × mix(10 sources))", Seq::mix((0..100).map(|_| Seq::mix((0..10).map(|_| Seq::source(10_000))))), 4000);
}

fn typical() {
    // Sources are bare lengths here; the id only keeps the calls readable.
    let src = |_id: u32, len: usize| Seq::source(len);
    let shuffled = |k: u32, len: usize| (0..k).map(move |i| src(i, len).shuffle(i as u64 + 1));
    let scheduled = |k: u32, len: usize| {
        (0..k).map(move |i| {
            let s = if i % 5 == 0 { Sampling::DelayedLinear { start: 0.2 + 0.05 * (i % 7) as f64, full: 0.7 } } else { Sampling::Uniform };
            (src(i, len).shuffle(i as u64 + 1), s)
        })
    };
    let m = 1_000_000usize;
    // Two nested mixes of shuffled sources, with each source repeated for 2–4 epochs.
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
    measure(
        "mix(60% shuffled + 9 × 4.4% shuffled)",
        Seq::mix((0..10).map(|i| src(i, if i == 0 { 60 * m } else { 4_444_444 }).shuffle(i as u64 + 1))),
        5 * m,
    );
    measure("mix(100 × source 1e6)", Seq::mix((0..100).map(|i| src(i, m))), 5 * m);
    measure("mix(100 × shuffled)", Seq::mix(shuffled(100, m)), 5 * m);
    measure("mix(100 × shuffled, 20% scheduled)", Seq::mix_with(scheduled(100, m)), 5 * m);
    measure("mix(1000 × shuffled, 20% scheduled)", Seq::mix_with(scheduled(1000, 100_000)), 5 * m);
    measure("↑ .shard(8, 0)", Seq::mix_with(scheduled(100, m)).shard(8, 0), m);
    measure("↑ .shard(512, 0)", Seq::mix_with(scheduled(100, m)).shard(512, 0), 100_000);
    let nested = Seq::mix_with([
        (Seq::concat([src(0, m).shuffle(1), src(1, m).shuffle(2)]).shuffle(3), Sampling::Uniform),
        (src(2, m).shuffle(4).take(500_000), Sampling::DelayedLinear { start: 0.5, full: 0.5 }),
        (src(3, m).shuffle(5).repeat(2), Sampling::DelayedLinear { start: 0.1, full: 0.4 }),
    ])
    .repeat(3)
    .shard(4, 1);
    measure("repeat(3, mix(3 nested)).shard(4, 1)", nested, m);
    let two_mixes = || Seq::mix([Seq::mix(shuffled(100, m)), Seq::mix((100..200).map(|i| src(i, m).shuffle(i as u64 + 1)))]);
    measure("mix(mix(100 × shuffled) × 2)", two_mixes(), 5 * m);
    measure("mix(mix(100 × shuffled) × 2).shard(8, 0)", two_mixes().shard(8, 0), m);
    measure("shuffle(mix(100 × source 1e6))  [slow path]", Seq::mix((0..100).map(|i| src(i, m))).shuffle(9), 100_000);
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    match mode.as_str() {
        "" => typical(),
        "--phases" => phases(),
        "--lifecycle" => lifecycles(),
        "--all" => {
            typical();
            phases();
            lifecycles();
        }
        _ => {
            eprintln!("usage: bench [--phases|--lifecycle|--all]");
            std::process::exit(2);
        }
    }
}
