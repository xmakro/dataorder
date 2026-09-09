# Adopted float simplifications

The smaller inverse and two construction simplifications are now integrated into
the production float engine. The development version is **0.3.0**, since rounding
changes can reorder items near ties. No release has been published. Existing
checkpoints must keep their original ordering version.

The changes remove 40 lines from `profile.rs` and `sum.rs` before their unit-test
modules (including comments and whitespace). Focused numerical regressions were
added. There are no new runtime fields, dependencies, schedule restrictions, or
backend choices in this round. The earlier weight-only change is separate.

## What was simplified

1. Rising segments invert their remaining area from the right endpoint. This
   removes the neighboring-float repair loop and its binary-search fallback;
   [FLOAT.md](FLOAT.md) records the derivation and compatibility investigation.
2. Construction uses the precise scheduled peak directly. It no longer divides
   that peak by its rounded value to create a correction factor, then multiplies
   the correction and rounded value back together. This also deletes the general
   compensated-ratio and compensated-multiply helpers. Scheduled segment endpoints
   are already restricted to zero or their profile's peak by the ramp/trapezoid
   constructors. Flat segments have zero slope; rising and falling segments divide
   the weighted peak by their signed duration.
3. A fused multiply-add obtains the product residual, replacing the custom
   significand-splitting implementation. Rust specifies a correctly rounded fused
   result, so this is an explicit arithmetic choice rather than compiler-dependent
   contraction. Other multiply/add expressions remain separate.
   [Rust's `f64::mul_add` specification](https://doc.rust-lang.org/std/primitive.f64.html#method.mul_add)
   also notes that its performance depends on the target hardware.

Rates remain continuous wherever the requested ramps/trapezoids are continuous.
The change does not introduce phase approximations. Quotas, source-local order,
and seeks agreeing with walks remain structural guarantees; infinite-precision
event ordering is still not the float engine's contract.

## Final paired measurements

Times below are the primary run on an Apple M2 Pro with Rust 1.98.1. Both sides
include the same inputs and allocation/destruction costs for construction. Each
measurement covers the interleave engine, excluding source I/O, shuffle, and the
public `Seq`/`Order` wrapper. Each sample is calibrated to at least 20 ms of measured work; five samples alternate
the two implementations. A second build swaps the module assignments as a control.
No benchmark overlaps compilation, tests, or another benchmark. No CPU affinity
or external-load isolation is used.

| Sources / construction | Before | Adopted | Time reduction |
| --- | --- | --- | --- |
| 3, ramps/trapezoids | 1.29 µs | 1.11 µs | 14% |
| 100, ramps/trapezoids | 9.03 µs | 7.54 µs | 17% |
| 1,000, ramps/trapezoids | 83.77 µs | 68.31 µs | 18% |
| 1,000, distinct breakpoints | 68.23 µs | 53.66 µs | 21% |
| 10,000, ramps/trapezoids | 838.74 µs | 676.18 µs | 19% |
| 10,000, distinct breakpoints | 696.43 µs | 543.07 µs | 22% |

The swapped-module control gives similar 18-23% construction reductions for the
1,000/10,000-source non-uniform cases. All-uniform construction is slightly slower
in these runs: about 3-8% at 1,000/10,000 sources. Thus this is not a universal
speedup. Its runtime layouts are unchanged.

For the 1,000-source ramp/trapezoid mixes, the affected 80%-progress walking windows
take about **10-17% less time** across the two runs, and fresh seeks take **22-25%
less time**. Walking measures 1,024 outputs after positioning and excludes the
seek. Other windows and distinct-breakpoint seeks change little.

The three-source scheduled fixtures explicitly contain a ramp/trapezoid or two
distinct delays, with lengths `[2_000_000, 100_000, 100_000]`. Larger fixtures use
the earlier deterministic workload generator. In the first exploratory build
trial its three-source cases accidentally contained only uniform sources; the
final table and raw results fix that and do not use those exploratory rows.

## Rejected alternative: evaluate rates at each boundary

The direct-evaluation prototype reconstructs demand independently at every
distinct schedule boundary. It avoids accumulating cancelling slopes, so it
could remove the expansion accumulator. It passes the existing release library
and integration tests, including the independent schedule fixtures.

However, it takes O(s*S) work for `s` scheduled sources and `S` boundaries. With
many distinct boundaries it is much slower:

| Sources / distinct breakpoints | Existing sweep | Direct evaluation | Ratio |
| --- | --- | --- | --- |
| 100 | 7.02 µs | 24.14 µs | 3.4× |
| 1,000 | 66.55 µs | 2,023.90 µs | 30.4× |
| 10,000 | 663.83 µs | 197,536.46 µs | 297.6× |

Even the shared-breakpoint ramp fixtures are about 4-6% slower than the old sweep
at 100-10,000 sources. This method was not integrated, and there is no automatic
fallback to it. An explicit limit on distinct control points could make its cost
bounded while preserving smooth interpolation, but that would change the accepted
schedule contract; this round introduces no such restriction.

## Validation and limits

The final implementation passes release and debug tests with all features,
doctests, Clippy with warnings denied, and the complete library/integration suite
on Rust 1.89. Linux x86-64 and i686 compile checks pass; those targets were not
executed or benchmarked on this ARM machine. Package listing confirms experiments
are excluded from the published crate. The [pre-commit verification](float-precommit-checks.json)
also passes example tests, default-feature tests, fixture regeneration, documentation
with warnings denied, Windows x86-64 compilation, and package build verification.
It reruns all 22,528 adjacent-quantile checks and 704 Decimal comparisons on the
final sources; the largest sampled error is about 1.21 endpoint-resolution units.

New regressions check a mathematically derived midpoint tie, product residuals
across large exponent scales, one uniform item beside up to 2,000 complementary
ramp sources at near-maximum length, and large counts with `1e-300`/`1e-200`
transitions. Existing tests cover tiny remainders, slope cancellation, neighboring
quantiles, independent schedule oracles, and cursor histories. The original
inverse audit's 22,528 adjacent-quantile checks and 704 Decimal reference samples
are recorded in [FLOAT.md](FLOAT.md); they are historical checks of the unchanged
inverse formula, not new full-construction error bounds.

The sweep's compensated accumulator and slope expansion remain. They still solve
real cancellation problems under the supported numeric range. Removing them is
not justified by the experiments above. No model-training experiment was run.

## Reproduce

Run `python3 experiments/rational-bench/float_further_reproduce.py`. It verifies
the adopted source hashes, restores the old baseline into a temporary directory,
builds the comparison variants, and runs the tests and measurements sequentially.
`--prepare-only` checks and prepares the files without executing the tests/benchmarks.
Preparation and compilation of all five generated comparison programs were verified.
The source tree is not changed. Cargo dependencies must be available offline.

Artifacts: [adopted changes](float-further.patch), [walk/seek results](float-further-results.json),
[build results](float-further-build-results.json), [swapped walk/seek control](float-further-reversed-results.json),
[swapped build control](float-further-reversed-build-results.json), [direct-build results](float-direct-build-results.json),
[rejected prototype patch](float-direct.patch), and [metadata](float-further-metadata.json).
