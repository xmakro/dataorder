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
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::ops::Range;
use std::time::{Duration, Instant};
#[path = "support/measurements.rs"]
#[allow(dead_code)]
mod measurements;
use measurements::{Measurement, Report, fingerprint, summarize};
const SAMPLES: usize = 5;
const SAMPLE_TIME: Duration = Duration::from_millis(20);
thread_local! { static ROWS: RefCell<BTreeMap<String, Measurement>> = const { RefCell::new(BTreeMap::new()) }; }

/// Calibration also warms the code and data. Keep five timed samples after it.
fn samples(mut batch: impl FnMut(usize) -> Duration, units: usize) -> Vec<f64> {
    let mut repetitions = 1usize;
    while batch(repetitions) < SAMPLE_TIME {
        repetitions = repetitions.checked_mul(2).expect("benchmark batch overflow");
    }
    (0..SAMPLES).map(|_| batch(repetitions).as_secs_f64() * 1e9 / repetitions as f64 / units as f64).collect()
}

/// Generate positions outside the timed region, including phase-specific bounds.
fn positions(range: Range<usize>) -> Vec<usize> {
    assert!(!range.is_empty());
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    (0..2048)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            range.start + ((x >> 11) % range.len() as u64) as usize
        })
        .collect()
}

fn record(name: &str, workload: String, columns: [Option<Vec<f64>>; 6]) {
    let row = Measurement {
        workload: fingerprint(workload.as_bytes()),
        samples: (0..SAMPLES).map(|i| columns.iter().map(|col| col.as_ref().map(|values| values[i])).collect()).collect(),
    };
    let (median, low, high) = summarize(&row.samples);
    let display = |col: usize| format!("{:.3} [{:.3}..{:.3}]", median[col].unwrap(), low[col].unwrap(), high[col].unwrap());
    if name.starts_with("lifecycle: ") {
        println!("{name:<54} {} µs  {} µs  {:.0} B", display(3), display(4), median[5].unwrap());
    } else {
        println!("{name:<54} {} µs  {} ns  {} ns", display(0), display(1), display(2));
    }
    ROWS.with_borrow_mut(|rows| assert!(rows.insert(name.to_owned(), row).is_none(), "duplicate benchmark row"));
}

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
    let config = format!("{seq:?}; count={count}; start={start:?}; continuous={continuous}");
    let order = Order::new(seq).unwrap();
    let n = order.len();
    let walk_start = start.unwrap_or(n / 3);
    let count = count.min(n - walk_start);
    let region = start.map_or(0..n, |s| s..s + count);
    let positions = positions(region);
    let seek = samples(
        |reps| {
            let t = Instant::now();
            for &pos in positions.iter().cycle().take(reps) {
                black_box(order.iter(black_box(pos)..).next());
            }
            t.elapsed()
        },
        1,
    )
    .into_iter()
    .map(|ns| ns / 1000.0)
    .collect();
    let walk = samples(
        |reps| {
            let mut elapsed = Duration::ZERO;
            for _ in 0..reps {
                let mut ongoing = continuous.then(|| order.iter(..));
                if let Some(cursor) = &mut ongoing {
                    for _ in 0..walk_start {
                        black_box(cursor.next());
                    }
                }
                let t = Instant::now();
                let cursor = ongoing.get_or_insert_with(|| order.iter(walk_start..walk_start + count));
                let mut acc = 0u64;
                for (s, i) in cursor.take(count) {
                    acc = acc.wrapping_add((*s ^ i) as u64);
                }
                black_box(acc);
                elapsed += t.elapsed();
            }
            elapsed
        },
        count,
    );
    let get = samples(
        |reps| {
            let t = Instant::now();
            for &pos in positions.iter().cycle().take(reps) {
                black_box(order.get(black_box(pos)));
            }
            t.elapsed()
        },
        1,
    );
    record(name, config, [Some(seek), Some(walk), Some(get), None, None, None]);
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
    let config = format!("{seq:?}; warmup={warmup}");
    let build = samples(
        |reps| {
            let mut elapsed = Duration::ZERO;
            for _ in 0..reps {
                let input = seq.clone();
                let t = Instant::now();
                let order = black_box(Order::new(input).unwrap());
                elapsed += t.elapsed();
                drop(order);
            }
            elapsed
        },
        1,
    )
    .into_iter()
    .map(|ns| ns / 1000.0)
    .collect();
    let order = Order::new(seq).unwrap();
    BYTES.set(0);
    COUNT_BYTES.set(true);
    let mut cursor = order.iter(..);
    cursor.by_ref().take(warmup).for_each(|item| {
        black_box(item);
    });
    COUNT_BYTES.set(false);
    let bytes = BYTES.get();
    let mut positions = positions(0..order.len());
    for pos in positions.iter_mut().step_by(2) {
        *pos = 0;
    }
    let reuse = samples(
        |reps| {
            let t = Instant::now();
            for &pos in positions.iter().cycle().take(reps) {
                cursor.seek(black_box(pos));
                black_box(cursor.next());
            }
            t.elapsed()
        },
        1,
    )
    .into_iter()
    .map(|ns| ns / 1000.0)
    .collect();
    record(&format!("lifecycle: {name}"), config, [None, None, None, Some(build), Some(reuse), Some(vec![bytes as f64; SAMPLES])]);
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
    println!("Median [min..max] of {SAMPLES} samples, batches calibrated to at least 20 ms; random positions precomputed.");
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
    let report = Report {
        schema: 1,
        harness: fingerprint(concat!(include_str!("bench.rs"), include_str!("support/measurements.rs")).as_bytes()),
        mode: if mode.is_empty() { "default".into() } else { mode.trim_start_matches("--").into() },
        rows: ROWS.with_borrow_mut(std::mem::take),
    };
    report.validate().unwrap();
    println!("{}{}", measurements::PREFIX, serde_json::to_string(&report).unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn random_access_positions_stay_inside_the_named_phase() {
        let early = positions(0..200_000);
        let late = positions(9_000_000..9_200_000);
        assert!(early.iter().all(|p| (0..200_000).contains(p)));
        assert!(late.iter().all(|p| (9_000_000..9_200_000).contains(p)));
        assert_eq!(early.iter().map(|p| p + 9_000_000).collect::<Vec<_>>(), late);
    }
}
