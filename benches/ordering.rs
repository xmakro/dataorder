//! `cargo bench --bench ordering`; source lengths only, no record I/O.
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use dataorder::{Order, Sampling, Seq};
use std::hint::black_box;

const WALK_LEN: usize = 100_000;

fn ordering(c: &mut Criterion) {
    let workloads = [
        ("shuffle", Seq::source(1_000_000_000).shuffle(1)),
        ("mix_100", Seq::mix((0..100).map(|i| Seq::source(1_000_000).shuffle(i + 1)))),
        (
            "scheduled_1000",
            Seq::mix_with((0..1000).map(|i| {
                let sampling = if i % 5 == 0 { Sampling::ramp(0.2 + 0.05 * (i % 7) as f64, 0.7) } else { Sampling::Uniform };
                (Seq::source(100_000).shuffle(i + 1), sampling)
            })),
        ),
    ];

    for (name, seq) in workloads {
        let order = Order::new(seq.clone()).unwrap();
        // Deterministic random positions, generated outside the timed loops.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let positions: [usize; 2048] = std::array::from_fn(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) % order.len() as u64) as usize
        });
        let mut group = c.benchmark_group(name);

        // Clone the configuration and drop the resulting order outside timing.
        group.bench_function("build", |b| {
            b.iter_batched(|| seq.clone(), |seq| Order::new(seq).unwrap(), BatchSize::LargeInput);
        });
        group.bench_function("get", |b| {
            let mut positions = positions.iter().cycle();
            b.iter(|| order.get(black_box(*positions.next().unwrap())).unwrap());
        });
        // Fresh seek includes cursor construction, first item and destruction.
        group.bench_function("seek", |b| {
            let mut positions = positions.iter().cycle();
            b.iter(|| order.iter(black_box(*positions.next().unwrap())..).unwrap().next().unwrap());
        });
        group.bench_function("reused_seek", |b| {
            let mut positions = positions.iter().cycle();
            let mut cursor = order.iter(..).unwrap();
            b.iter(|| {
                cursor.seek(black_box(*positions.next().unwrap())).unwrap();
                cursor.next().unwrap()
            });
        });

        // Report throughput for a whole walk, including its initial seek.
        let start = order.len() / 3;
        group.throughput(Throughput::Elements(WALK_LEN as u64));
        group.bench_function("walk", |b| {
            b.iter(|| {
                for item in order.iter(black_box(start)..start + WALK_LEN).unwrap() {
                    black_box(item);
                }
            });
        });
        group.finish();
    }
}

criterion_group!(benches, ordering);
criterion_main!(benches);
