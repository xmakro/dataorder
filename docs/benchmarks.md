# Benchmarks

Run the benchmarks on the hardware and configuration you plan to use. The crate's
[cost model](https://docs.rs/dataorder/latest/dataorder/#cost) explains how composition
affects performance. Current measurements are below; older Ryzen results are kept
in the [historical section](#historical-measurements).

## Current measurements

Recorded on 2026-09-05 from revision [`bc2908f`](https://github.com/xmakro/dataorder/commit/bc2908f7f717461af2c302c2a5b7b8e267912a18),
on an Apple M2 Pro with Rust 1.98.1, macOS arm64, release build. The benchmark is
single-threaded, with no CPU affinity. Each column is the minimum of two runs.
[Raw results](benchmarks/2026-09-05-m2-pro.json) include the exact command and toolchain.

The benchmark uses source lengths and generates source/index pairs; it does not load
records. Seek averages 200 randomly selected positions and includes cursor construction
and the first item. Get averages up to 200,000 random lookups. Walk starts one third
of the way into each order and averages up to five million items, including the
initial seek. The shorter shard and slow-path cases use fewer items; see
[`examples/bench.rs`](../examples/bench.rs) for the exact configurations.

The nested order combines mixes of 1,000 and 100 shuffled sources. Source
lengths range from 0.5 to 2 million, repeated for 2–4 epochs, giving 4,098,465,102
output positions. A fresh seek into it takes about 34 µs; walking from that point
averages 62.4 ns per item. The order is computed without storing those four billion
source/index pairs.

```sh
BENCH_CORE=none cargo run --locked --release --example bench_campaign -- run readme-2026-09-05 . default
```

| Order | Seek + first item | Walk / item | Get |
| --- | --- | --- | --- |
| `mix(mix(1000 × shuffled × 2–4 epochs), mix(100 × …))` | 33.86 µs | 62.4 ns | 30687.2 ns |
| `source 1e9` | 0.01 µs | 3.5 ns | 3.8 ns |
| `shuffle(source 1e9)` | 0.04 µs | 14.2 ns | 21.7 ns |
| `shuffle(source 1e6).repeat(1000)` | 0.10 µs | 14.2 ns | 24.9 ns |
| `concat(100 × shuffle(source 1e6))` | 0.10 µs | 14.1 ns | 35.2 ns |
| `shuffle(concat(100 × source 1e6))` | 0.05 µs | 31.3 ns | 40.1 ns |
| `mix(5 × source 1e6)` | 0.38 µs | 11.9 ns | 206.1 ns |
| `mix(80% source + 4 × 5%)` | 0.36 µs | 13.0 ns | 221.7 ns |
| `mix(60% shuffled + 9 × 4.4% shuffled)` | 0.54 µs | 37.7 ns | 368.9 ns |
| `mix(100 × source 1e6)` | 2.26 µs | 18.9 ns | 1842.6 ns |
| `mix(100 × shuffled)` | 2.20 µs | 26.7 ns | 1871.3 ns |
| `mix(100 × shuffled, 20% scheduled)` | 4.91 µs | 28.3 ns | 4494.9 ns |
| `mix(1000 × shuffled, 20% scheduled)` | 47.40 µs | 45.1 ns | 44184.3 ns |
| `mix(100 × shuffled, 20% scheduled).shard(8, 0)` | 5.41 µs | 180.2 ns | 4463.7 ns |
| `mix(100 × shuffled, 20% scheduled).shard(512, 0)` | 9.39 µs | 5040.1 ns | 4485.6 ns |
| `repeat(3, mix(3 nested)).shard(4, 1)` | 0.53 µs | 79.9 ns | 272.2 ns |
| `mix(mix(100 × shuffled) × 2)` | 2.57 µs | 36.2 ns | 2249.4 ns |
| `mix(mix(100 × shuffled) × 2).shard(8, 0)` | 2.72 µs | 152.0 ns | 2254.6 ns |
| `shuffle(mix(100 × source 1e6))  [slow path]` | 1.97 µs | 1954.4 ns | 1960.7 ns |

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
| Walk | Average per element over a long range, including the initial seek; continuous-tail cases resume an existing cursor |
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
