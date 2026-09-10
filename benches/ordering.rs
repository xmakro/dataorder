//! `cargo bench --bench ordering`; source lengths only, no record I/O.
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use dataorder::{Order, Schedule, Seq};
use std::hint::black_box;

const WALK_LEN: usize = 100_000;

fn ordering(c: &mut Criterion) {
    let workloads = [
        ("shuffle", Seq::source(1_000_000_000).shuffle(1)),
        ("mix_100", Seq::mix((0..100).map(|i| Seq::source(1_000_000).shuffle(i + 1)))),
        (
            "scheduled_1000",
            Seq::mix((0..1000).map(|i| {
                let schedule = if i % 5 == 0 { Schedule::ramp(0.2 + 0.05 * (i % 7) as f64, 0.7) } else { Schedule::Uniform };
                (Seq::source(100_000).shuffle(i + 1), schedule)
            })),
        ),
        ("selection/slice_shuffle", Seq::source(1_000_000_000).shuffle(1).skip(12_345).take(750_000_000)),
        ("selection/slice_repeat", Seq::source(1024).repeat(1_000_000).skip(17).take(900_000_000)),
        ("selection/slice_mix", Seq::mix((0..100).map(|i| Seq::source(1_000_000).shuffle(i + 1))).skip(12_345).take(75_000_000)),
        ("selection/stride_source", Seq::source(1_000_000_000).skip(11).step_by(8)),
        ("selection/stride_shuffle", Seq::source(1_000_000_000).shuffle(1).skip(11).step_by(8)),
        ("selection/stride_repeat", Seq::source(1024).repeat(1_000_000).skip(17).step_by(7)),
        ("selection/stride_mix", Seq::mix((0..100).map(|i| Seq::source(1_000_000).shuffle(i + 1))).skip(11).step_by(8)),
        (
            "selection/mix_selected_parts",
            Seq::mix((0..100).map(|i| Seq::source(1_000_000).shuffle(i + 1).skip(11).take(900_000).step_by(1 + i as usize % 4))),
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

fn cursor_state(c: &mut Criterion) {
    let part = |k| Seq::mix((0..k).map(|i| Seq::source(10).shuffle(i as u64)));
    let order = Order::new(Seq::concat([part(1000), part(2)]).repeat(2)).unwrap();
    let mut cursor = order.iter(..).unwrap();
    cursor.next();
    cursor.seek(10_000).unwrap();
    cursor.next();

    let mut group = c.benchmark_group("cursor_state");
    group.bench_function("clone_after_small_mix", |b| b.iter(|| black_box(cursor.clone())));
    group.bench_function("cloned_seek_back", |b| {
        b.iter_batched(
            || cursor.clone(),
            |mut copy| {
                copy.seek(0).unwrap();
                black_box(copy.next());
            },
            BatchSize::LargeInput,
        );
    });
    // Two seeks cross between different mixes on every iteration.
    group.bench_function("concat_seeks", |b| {
        b.iter(|| {
            for pos in [0, 10_000] {
                cursor.seek(black_box(pos)).unwrap();
                black_box(cursor.next());
            }
        });
    });
    group.throughput(Throughput::Elements(order.len() as u64));
    group.bench_function("concat_walk", |b| {
        b.iter(|| {
            for item in order.iter(..).unwrap() {
                black_box(item);
            }
        });
    });
    group.finish();
}

criterion_group!(benches, ordering, cursor_state);
criterion_main!(benches);
