//! Cost of typical orders: seek = `order.iter(pos..).unwrap().next()` at a random position,
//! including cursor construction. Walk is the average per element over a long range,
//! including its initial seek; get = `order.get(pos)` at a random position. All
//! measurements use source lengths only and exclude record I/O.
//! `cargo run --release --example bench`
//! `-- --phases` measures early, rising, falling and exhausted schedule phases; `-- --lifecycle`
//! measures compilation, reusable seeks and cursor memory, including worker scaling.
//! `-- --all` includes both alongside the original table. Memory is requested, retained
//! and peak live allocator bytes during cursor construction and warmup, including the
//! cursor vector but excluding the shared order and allocator overhead (not RSS).
//! Compilation excludes cloning the input configuration.
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

fn record(name: &str, workload: String, columns: [Option<Vec<f64>>; 8]) {
    let row = Measurement {
        workload: fingerprint(workload.as_bytes()),
        samples: (0..SAMPLES).map(|i| columns.iter().map(|col| col.as_ref().map(|values| values[i])).collect()).collect(),
    };
    let (median, low, high) = summarize(&row.samples);
    let display = |col: usize| format!("{:.3} [{:.3}..{:.3}]", median[col].unwrap(), low[col].unwrap(), high[col].unwrap());
    if name.starts_with("lifecycle: ") {
        println!(
            "{name:<54} {} µs  {} µs  {:.0} / {:.0} / {:.0} B (requested / retained / peak)",
            display(3),
            display(4),
            median[5].unwrap(),
            median[6].unwrap(),
            median[7].unwrap()
        );
    } else {
        println!("{name:<54} {} µs  {} ns  {} ns", display(0), display(1), display(2));
    }
    ROWS.with_borrow_mut(|rows| assert!(rows.insert(name.to_owned(), row).is_none(), "duplicate benchmark row"));
}

struct Counting;
thread_local! {
    static COUNT_BYTES: Cell<bool> = const { Cell::new(false) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
    static LIVE: Cell<usize> = const { Cell::new(0) };
    static PEAK: Cell<usize> = const { Cell::new(0) };
}

// Only cursor-owned allocations are created/freed in the measured window. Realloc
// counts the full successful request, and replaces the old live allocation. Peak
// measures live Rust layouts; it cannot observe a system allocator's internal copy.
fn allocation(old: usize, new: usize) {
    let _ = COUNT_BYTES.try_with(|enabled| {
        if enabled.get() {
            BYTES.set(BYTES.get() + new);
            LIVE.set(LIVE.get() - old + new);
            PEAK.set(PEAK.get().max(LIVE.get()));
        }
    });
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            allocation(0, layout.size());
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        allocation(layout.size(), 0);
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            allocation(0, layout.size());
        }
        ptr
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !ptr.is_null() {
            allocation(layout.size(), new_size);
        }
        ptr
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
                black_box(order.iter(black_box(pos)..).unwrap().next());
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
                let mut ongoing = continuous.then(|| order.iter(..).unwrap());
                if let Some(cursor) = &mut ongoing {
                    for _ in 0..walk_start {
                        black_box(cursor.next());
                    }
                }
                let t = Instant::now();
                let cursor = ongoing.get_or_insert_with(|| order.iter(walk_start..walk_start + count).unwrap());
                let mut acc = 0u64;
                for item in cursor.take(count) {
                    acc = acc.wrapping_add((*item.source ^ item.record_index) as u64);
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
                black_box(order.get(black_box(pos)).unwrap());
            }
            t.elapsed()
        },
        1,
    );
    record(name, config, [Some(seek), Some(walk), Some(get), None, None, None, None, None]);
}

fn phases() {
    let scheduled = |sampling| {
        Seq::mix_with((0..1000).map(|i| (Seq::source(10_000).shuffle(i + 1), if i % 5 == 0 { sampling } else { Sampling::Uniform })))
    };
    for (phase, start) in [("output 0%", 0), ("output 30%", 3_000_000), ("output 60%", 6_000_000), ("output 80%", 8_000_000)] {
        measure_at(&format!("phase ramp: {phase}"), scheduled(Sampling::ramp(0.2, 0.7)), 200_000, Some(start), false);
    }
    for (phase, start) in [("output 10%", 1_000_000), ("output 50%", 5_000_000), ("output 90%", 9_000_000)] {
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
    lifecycle_workers(name, seq, warmup, 1);
}

fn lifecycle_workers(name: &str, seq: Seq<usize>, warmup: usize, workers: usize) {
    lifecycle_at(name, seq, warmup, workers, None);
}

fn lifecycle_at(name: &str, seq: Seq<usize>, warmup: usize, workers: usize, seek_range: Option<Range<usize>>) {
    let config = format!("{seq:?}; warmup={warmup}; workers={workers}; seek_range={seek_range:?}");
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
    LIVE.set(0);
    PEAK.set(0);
    COUNT_BYTES.set(true);
    let mut cursors = Vec::with_capacity(workers);
    for _ in 0..workers {
        let mut cursor = order.iter(..).unwrap();
        cursor.by_ref().take(warmup).for_each(|item| {
            black_box(item);
        });
        cursors.push(cursor);
    }
    COUNT_BYTES.set(false);
    let (bytes, retained, peak) = (BYTES.get(), LIVE.get(), PEAK.get());
    let cursor = &mut cursors[0];
    let mut positions = positions(seek_range.unwrap_or(0..order.len()));
    for pos in positions.iter_mut().step_by(2) {
        *pos = 0;
    }
    let reuse = samples(
        |reps| {
            let t = Instant::now();
            for &pos in positions.iter().cycle().take(reps) {
                cursor.seek(black_box(pos)).unwrap();
                black_box(cursor.next());
            }
            t.elapsed()
        },
        1,
    )
    .into_iter()
    .map(|ns| ns / 1000.0)
    .collect();
    record(
        &format!("lifecycle: {name}"),
        config,
        [
            None,
            None,
            None,
            Some(build),
            Some(reuse),
            Some(vec![bytes as f64; SAMPLES]),
            Some(vec![retained as f64; SAMPLES]),
            Some(vec![peak as f64; SAMPLES]),
        ],
    );
}

fn lifecycles() {
    lifecycle(
        &format!("mix(256 sources, {} identity transforms each)", dataorder::MAX_DEPTH - 2),
        Seq::mix((0..256).map(|_| (0..dataorder::MAX_DEPTH - 2).fold(Seq::source(10), |seq, _| seq.take(10)))),
        512,
    );
    lifecycle(
        "concat(100 mixes of 100 sources), repeated epochs",
        Seq::concat((0..100).map(|_| Seq::mix((0..100).map(|_| Seq::source(10))))).repeat(3),
        200_000,
    );
    for k in [100, 1000, 10_000, 100_000] {
        lifecycle(&format!("mix({k} sources)"), Seq::mix((0..k).map(|_| Seq::source(10_000))), 2 * k);
    }
    for workers in [8, 32] {
        lifecycle_workers(
            &format!("mix(10000 sources), {workers} workers"),
            Seq::mix((0..10_000).map(|_| Seq::source(10_000))),
            20_000,
            workers,
        );
    }
    // A minority uniform source beside a large independent scheduled source.
    // On 64-bit hosts this exercises rank recovery close to MAX_MIX_LEN.
    let n = (dataorder::MAX_MIX_LEN / 2).min(usize::MAX as u64) as usize;
    lifecycle_at(
        "large virtual-clock schedule, tail seeks",
        Seq::mix_with([(Seq::source(1), Sampling::Uniform), (Seq::source(n - 1), Sampling::until(1.0 - 5e-10))]),
        100,
        1,
        Some(n - 20_000..n),
    );
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
    measure("↑ .skip(0).step_by(8)", Seq::mix_with(scheduled(100, m)).skip(0).step_by(8), m);
    measure("↑ .skip(0).step_by(512)", Seq::mix_with(scheduled(100, m)).skip(0).step_by(512), 100_000);
    let nested = Seq::mix_with([
        (Seq::concat([src(0, m).shuffle(1), src(1, m).shuffle(2)]).shuffle(3), Sampling::Uniform),
        (src(2, m).shuffle(4).take(500_000), Sampling::DelayedLinear { start: 0.5, full: 0.5 }),
        (src(3, m).shuffle(5).repeat(2), Sampling::DelayedLinear { start: 0.1, full: 0.4 }),
    ])
    .repeat(3)
    .skip(1)
    .step_by(4);
    measure("repeat(3, mix(3 nested)).skip(1).step_by(4)", nested, m);
    let two_mixes = || Seq::mix([Seq::mix(shuffled(100, m)), Seq::mix((100..200).map(|i| src(i, m).shuffle(i as u64 + 1)))]);
    measure("mix(mix(100 × shuffled) × 2)", two_mixes(), 5 * m);
    measure("mix(mix(100 × shuffled) × 2).skip(0).step_by(8)", two_mixes().skip(0).step_by(8), m);
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
        schema: 2,
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
    fn allocation_metrics_track_retained_and_peak_bytes() {
        BYTES.set(0);
        LIVE.set(0);
        PEAK.set(0);
        COUNT_BYTES.set(true);
        let metrics = || (BYTES.get(), LIVE.get(), PEAK.get());
        let (first, grown, second, freed, empty);
        unsafe {
            let layout = Layout::from_size_align(16, 8).unwrap();
            let a = ALLOCATOR.alloc(layout);
            if a.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            first = metrics();
            let a = ALLOCATOR.realloc(a, layout, 32);
            if a.is_null() {
                std::alloc::handle_alloc_error(Layout::from_size_align(32, 8).unwrap());
            }
            grown = metrics();
            let b = ALLOCATOR.alloc_zeroed(layout);
            if b.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            second = metrics();
            ALLOCATOR.dealloc(a, Layout::from_size_align(32, 8).unwrap());
            freed = metrics();
            ALLOCATOR.dealloc(b, layout);
            empty = metrics();
        }
        COUNT_BYTES.set(false);
        assert_eq!(first, (16, 16, 16));
        assert_eq!(grown, (48, 32, 32));
        assert_eq!(second, (64, 48, 48));
        assert_eq!(freed, (64, 16, 48));
        assert_eq!(empty, (64, 0, 48));
    }

    #[test]
    fn random_access_positions_stay_inside_the_named_phase() {
        let early = positions(0..200_000);
        let late = positions(9_000_000..9_200_000);
        assert!(early.iter().all(|p| (0..200_000).contains(p)));
        assert!(late.iter().all(|p| (9_000_000..9_200_000).contains(p)));
        assert_eq!(early.iter().map(|p| p + 9_000_000).collect::<Vec<_>>(), late);
    }
}
