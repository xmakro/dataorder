# A smaller float inverse

There is a useful simplification without changing the continuous rate model:
invert a rising segment's **remaining area from its right endpoint**. The
[candidate patch](float-inverse.patch) removes 26 net lines from `profile.rs`,
including the entire `Segment::rising_inverse` repair routine, and adds one short
regression test. It adds no fields, dependencies, numeric limits, or runtime backend.

The inverse is now integrated together with further construction simplifications
in the **0.3.0 development version**; see [FLOAT_FURTHER.md](FLOAT_FURTHER.md).
This report preserves the initial inverse-only measurements and patch. It changes
some orders and is not a compatible patch release even though existing golden
tests pass. No release has been published.

## Why the smaller formula works

For a segment of width `w` whose rate rises from `r0 > 0` to `r1`, let `z` be the
area already consumed within it and `M = (r0+r1)*w/2` its total area. The rate at
the desired point is `R = sqrt(r0*r0 + 2*(r1-r0)*z/w)`.

The old inverse starts with `2*z/(r0+R)`. Both numerator and denominator increase,
so rounding can reverse neighboring results. It repairs the result by checking
neighboring floats, with a bitwise binary search as a fallback.

The replacement is:

```text
time = end - 2*(M-z)/(r1+R)
```

As `z` increases, the numerator decreases and the positive denominator increases.
Their quotient is nonincreasing, and subtracting it from the fixed endpoint gives
a nondecreasing time. The implementation clamps the remaining area and final time
and returns the exact start at zero area. Constants, falling segments, and
zero-start rising segments retain their existing formulas.

This avoids a repair algorithm by choosing monotone operations. It retains the
same continuous piecewise-linear rate profiles and the existing seek checks.
Endpoint subtraction trades relative accuracy very near the start for small
absolute error; it does not make the rates piecewise constant.

## Measurements

The two 1,000-source schedule fixtures mix ramps, trapezoids, and a uniform
remainder. At 80% progress that remainder has a rising rate, exercising the changed
formula. Each positioned window emits 1,024 items and excludes seeking.

| Fixture at 80% progress | Before | Candidate | Walk time reduction | Fresh seek time reduction |
| --- | --- | --- | --- | --- |
| Dyadic breakpoints | 37.0 ns/item | 32.0 ns/item | 13.5% | 23.4% |
| Decimal/binary breakpoints | 36.7 ns/item | 33.2 ns/item | 9.6% | 25.6% |

A second build swaps the before/after module assignments as a control. Its walk
reductions are 13.4% and 10.1%, and seek reductions are 22.6% and 24.8%. Construction
is essentially unchanged. Other windows show much smaller differences; this is
not a claim of a uniform speedup across all schedules.

Both runs use five alternating paired samples, calibrated to at least 20 ms of
measured work per sample. CPU: Apple M2 Pro; Rust 1.98.1; no CPU affinity or external
load isolation. Benchmarks did not overlap compilation or verification. The
retained runtime layouts and construction algorithm are unchanged.

## Correctness and compatibility

The candidate passes the existing release tests, including independent schedule
fixtures, golden orders, large orders, cursor state, and seek/walk checks. Additional
tests cover 22,528 neighboring quantiles across tiny widths, one-ULP intervals,
nearly flat rates, and near-zero starting rates. A separate 150-digit Decimal
inverse checks 704 inputs. The maximum error in those samples is about 1.21 units,
where a unit is `max(ulp(start), ulp(end), epsilon*width)`. This accounts for the
finite resolution of endpoints: a one-ULP-wide interval has no representable
interior time. This sampled result is not a universal error bound.

A targeted compatibility search found changed order in 69 of 2,401 configurations.
Those cases deliberately probe ties; the fraction is not a workload prevalence
estimate. For example, mix 15 uniform items with 5 items fading linearly from time
0 to 1. Uniform item 6 and fading item 3 both have ideal time 1/2. The new inverse
puts the lower-index uniform source first; the old rounding puts it second. The
patch includes this independently derived regression. Other changed configurations
have not all been classified against the exact reference, so this does not claim
universally closer ordering.

Both implementations remain floating-point models. Structural guarantees—every
item exactly once, source-local order, and seeks matching walks—remain unchanged.
Existing checkpoints must continue using their original ordering version.

## Scope

This is a concrete first deletion, not a complete simplification of construction.
The uniform-remainder sweep and compensated sums remain. Its value is that it
removes numerical repair without reducing expressiveness or adding state. The
earlier exact scheduler and stepwise models remain unrecommended replacements.

## Reproduce

Run `python3 experiments/rational-bench/float_reproduce.py` from the repository.
It verifies baseline source hashes, prepares a temporary copy, applies the patch,
runs tests and the Decimal check, and measures both module assignments. It never
changes production files. Dependencies must be available to Cargo offline.
`--prepare-only` prepares and checks the candidate without running it.

Artifacts: [primary timings](float-results.json), [swapped-module timings](float-reversed-results.json),
[compatibility result](float-compatibility.json), [numeric check](float-numeric-summary.json),
[metadata](float-metadata.json), and [benchmark source](float_bench.rs).
