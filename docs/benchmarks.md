# Benchmarks

Run the benchmarks on the hardware and configuration you plan to use. The crate's
[cost model](https://docs.rs/dataorder/latest/dataorder/#cost) explains how composition
affects performance; the timings below are historical measurements.

## Running the benchmarks

```sh
cargo run --release --example bench
cargo run --release --example bench -- --phases
cargo run --release --example bench -- --lifecycle
cargo run --release --example bench -- --all
```

The default table measures common order shapes. `--phases` measures early, rising,
falling and exhausted schedule phases, including uninterrupted iteration into the tail.
`--lifecycle` measures construction, seeking with an existing cursor and cursor allocation
requests. `--all` runs every group.

| Measurement | Operation |
| --- | --- |
| Walk | One `next()` after the cursor has been positioned |
| Seek | `order.iter(pos..).next()` at a random position, including cursor construction |
| Get | `order.get(pos)` at a random position |
| Build | `Order::new`, excluding the clone of the input configuration |
| Reused seek | Repositioning and drawing from an existing cursor |
| Cursor requested bytes | Cumulative allocation requests, not retained memory or process memory |

Run the ignored microbenchmarks for the interleave, tournament tree and permutation with:

```sh
cargo test --release -- --ignored --nocapture
```

## Comparing changes

The campaign harness runs each case twice and saves per-column minima in
`target/bench-campaign.json`. Give each run a label, then print a comparison table:

```sh
cargo run --release --example bench_campaign -- run baseline . all
# Make the change, then measure it with the same settings.
cargo run --release --example bench_campaign -- run changed . all
cargo run --release --example bench_campaign -- table 1
```

The `run` arguments are a label, a crate directory and an optional mode (`default`,
`phases`, `lifecycle` or `all`). `BENCH_MODE` supplies the default mode.
`BENCH_CORE` selects the CPU core (default `2`); set `BENCH_CORE=none` on systems
without `taskset`, including macOS. For example:

```sh
BENCH_CORE=none cargo run --release --example bench_campaign -- run baseline . all
```

Table columns are `0` seek, `1` walk (the default), `2` get, `3` build,
`4` reused seek and `5` cursor requested bytes. Existing campaign files can be moved
to `target/bench-campaign.json`.

## Historical measurements

These measurements were recorded in September 2026 on one core of a Ryzen 9 9950X3D.
They predate the numerical and shuffle fixes and do **not** describe the current
implementation's performance. They are retained as a reference for past comparisons.

### Mix size and schedules

Each row compares uniform parts, 20% scheduled parts with seven distinct starts,
and 20% scheduled parts with all starts distinct. `k` is the number of parts.

| Parts | Seek (uniform / shared starts / distinct starts) | Walk (same order) |
| --- | --- | --- |
| 100 | 2.8 µs / 4.8 µs / 6.0 µs | 10 ns / 13 ns / 12 ns |
| 1 000 | 29 µs / 51 µs / 91 µs | 15 ns / 19 ns / 18 ns |
| 10 000 | 0.44 ms / 0.62 ms / 1.24 ms | 22 ns / 29 ns / 26 ns |

### Composed orders

The first row combines two mixes of 1,000 and 100 shuffled sources. Each source
contains 0.5–2 million elements and repeats for 2–4 epochs.

| Order | Walk | Seek | Get |
| --- | --- | --- | --- |
| `mix(mix(1000 × shuffled, 2–4 epochs), mix(100 × same))` | 38.4 ns | 30 µs | 27.6 µs |
| `source` | 1.4 ns | 0.02 µs | 3.1 ns |
| `shuffle(source)` | 9.0 ns | 0.03 µs | 16.2 ns |
| `shuffle(source 10⁶).repeat(1000)` | 9.2 ns | 0.04 µs | 19.4 ns |
| `concat(100 × shuffle(source))` | 9.2 ns | 0.06 µs | 29.9 ns |
| `shuffle(concat(100 × source))` | 20.3 ns | 0.05 µs | 29.1 ns |
| `mix(5 × source)` | 6.9 ns | 0.19 µs | 142.7 ns |
| `mix(80% source + 4 × 5% source)` | 7.3 ns | 0.18 µs | 155.2 ns |
| `mix(60% shuffled + 9 × 4.4% shuffled)` | 22.3 ns | 0.36 µs | 284.0 ns |
| `mix(100 × source)` | 10.5 ns | 2.07 µs | 1.9 µs |
| `mix(100 × shuffled)` | 18.3 ns | 2.10 µs | 1.9 µs |
| `mix(100 × shuffled, 20% scheduled)` | 24.2 ns | 4.04 µs | 3.8 µs |
| `mix(1000 × shuffled, 20% scheduled)` | 29.5 ns | 37 µs | 36.4 µs |
| `mix(100 × shuffled, 20% scheduled).shard(8, 0)` | 114.8 ns | 3.76 µs | 3.8 µs |
| `mix(100 × shuffled, 20% scheduled).shard(512, 0)` | 4.3 µs | 7.58 µs | 3.8 µs |
| `repeat(3, mix(3 nested)).shard(4, 1)` | 56.7 ns | 0.26 µs | 170.6 ns |
| `mix(mix(100 × shuffled), mix(100 × shuffled))` | 23.6 ns | 2.23 µs | 2.0 µs |
| `mix(mix(100 × shuffled), mix(100 × shuffled)).shard(8, 0)` | 106.0 ns | 2.27 µs | 2.0 µs |
| `shuffle(mix(100 × source))` | 1.9 µs | 1.96 µs | 1.9 µs |
