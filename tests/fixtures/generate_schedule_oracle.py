"""Independent schedule oracle: exact rational CDFs and 96-digit inverse CDFs.

Run: python3 tests/fixtures/generate_schedule_oracle.py [--check]
Uses only Python's standard library; never imports or executes dataorder.
Breakpoints are interpreted as their exact binary64 values, as in the Rust API.
Small cases list every position of the merged order; sparse cases list independently
computed windows around selected keys of orders up to MAX_MIX_LEN elements. Together
they cover overlapping schedules without uniform parts, gaps, interacting ramps,
reordered minorities, nearly coincident boundaries and exact ties.
"""
from decimal import Decimal, localcontext
from fractions import Fraction as F
import json
from pathlib import Path
import random
import sys


def profile(points):
    a, b, c, d = map(F, points)
    peak = 2 / ((c - b) + (d - a))
    return [(s, e, r0, r1) for s, e, r0, r1 in
            [(F(0), a, 0, 0), (a, b, 0, peak), (b, c, peak, peak),
             (c, d, peak, 0), (d, F(1), 0, 0)] if e > s]


def profiles(lens, schedules):
    # Each part has its own normalized CDF on the same virtual clock. No capacity
    # constraint or complement is involved, including when all parts are scheduled.
    result = [profile([0, 0, 1, 1] if p is None else p) for p in schedules]
    for p in result:
        assert sum((b - a) * (r0 + r1) / 2 for a, b, r0, r1 in p) == 1
    return result


def dec(x):
    x = F(x)
    return Decimal(x.numerator) / Decimal(x.denominator)


def inverse(p, y):
    share = F(0)
    for a, b, r0, r1 in p:
        mass = (r0 + r1) * (b - a) / 2
        if y <= share + mass:
            z = y - share
            if z == 0:
                return dec(a)
            if r0 == r1:
                return dec(a + z / r0)
            c = (r1 - r0) / (2 * (b - a))
            root = (dec(r0 * r0 + 4 * c * z)).sqrt()
            return dec(a) + 2 * dec(z) / (dec(r0) + root)
        share += mass
    raise AssertionError("unnormalized profile")


def small_case(lens, schedules):
    ps = profiles(lens, schedules)
    live = [s for s, n in enumerate(lens) if n]
    keys = []
    for rank, s in enumerate(live):
        for j in range(lens[s]):
            y = F(2 * len(live) * j + 2 * rank + 1, 2 * len(live) * lens[s])
            # Collapse only rounding at the oracle's 96-digit precision, far below
            # any distinctions in these fixtures, including exact cross-part ties.
            key = inverse(ps[s], y).quantize(Decimal("1e-75"))
            keys.append((key, s, j))
    keys.sort()
    return dict(lens=lens, schedules=schedules,
                samples=[[pos, s, j] for pos, (_, s, j) in enumerate(keys)])


def sparse_case(name, lens, schedules):
    """Find ranks of selected keys without walking a huge output or calling Rust."""
    ps = profiles(lens, schedules)
    live = [s for s, n in enumerate(lens) if n]
    k = len(live)
    phi = {s: F(2 * rank + 1, 2 * k) for rank, s in enumerate(live)}

    def key(s, j):
        return inverse(ps[s], (j + phi[s]) / lens[s]).quantize(Decimal("1e-75"))

    samples = {}
    for s in live:
        indices = range(lens[s]) if lens[s] < 20 else sorted({0, 1, lens[s] // 4, lens[s] // 2, lens[s] * 3 // 4, lens[s] - 2, lens[s] - 1})
        for j in indices:
            target = (key(s, j), s)
            rank = 0
            neighbors = []
            for t in live:
                lo, hi = 0, lens[t]
                while lo < hi:
                    mid = (lo + hi) // 2
                    if (key(t, mid), t) < target:
                        lo = mid + 1
                    else:
                        hi = mid
                rank += lo
                # A small neighborhood from each sorted part contains the merged
                # neighborhood, without enumerating any large part's full output.
                for index in range(max(0, lo - 4), min(lens[t], lo + 5)):
                    neighbors.append((key(t, index), t, index))
            neighbors.sort()
            center = next(i for i, (_, t, index) in enumerate(neighbors) if (t, index) == (s, j))
            for offset in range(-min(4, rank), min(4, sum(lens) - rank - 1) + 1):
                _, t, index = neighbors[center + offset]
                position = rank + offset
                assert samples.get(position, (t, index)) == (t, index)
                samples[position] = (t, index)
    return dict(name=name, lens=lens, schedules=schedules,
                samples=[[pos, s, j] for pos, (s, j) in sorted(samples.items())])


def fixtures():
    cases = []
    with localcontext() as ctx:
        ctx.prec = 96
        cases.append(small_case([101, 17, 13], [None, [0, 0, .5, .5], [.5, .75, 1, 1]]))
        cases.append(small_case([173, 19, 11], [None, [.125, .25, .5, .875], [0, .25, 1, 1]]))
        for name, lens, schedules in [
            ("equal lengths delayed past halfway", [100, 100], [None, [.6, .6, 1, 1]]),
            ("all delayed across a shared gap", [31, 31, 31], [[.8, .8, 1, 1]] * 3),
            ("independent overlapping ramps without uniform", [11, 23, 17], [[0, 1, 1, 1], [.2, .7, 1, 1], [0, 0, .4, .8]]),
            ("unequal separated supports", [11, 29], [[0, 0, .2, .2], [.8, .8, 1, 1]]),
            ("constant stops during a ramp", [100, 100], [[0, 0, .8, .8], [0, 1, 1, 1]]),
            ("complementary abrupt halves", [19, 19], [[0, 0, .5, .5], [.5, .5, 1, 1]]),
            ("complementary ramps with an exact tie", [5, 5], [[0, 1, 1, 1], [0, 0, 0, 1]]),
            ("overlapping ramps with uniform", [20, 20, 14], [[0, .5, 1, 1], [0, 0, .5, 1], None]),
            ("multiple scheduled minorities", [97, 3, 5, 7, 11], [None, [0, 0, .125, .125], [.25, .5, .75, 1], [0, .125, .375, .5], [.5, .5, 1, 1]]),
            ("nearly coincident boundaries", [101, 3, 5, 7], [None, [0, 0, .5, .5], [.5, .5 + 2**-30, .875, 1], [0, .25, .5 - 2**-30, .75]]),
            ("constant quantile ties", [1, 3, 0], [None, [0, 0, 1, 1], [0, 0, .5, .5]]),
        ]:
            for reverse in [False, True]:
                a, b = (lens[::-1], schedules[::-1]) if reverse else (lens, schedules)
                case = small_case(a, b)
                case["name"] = name + (" reversed" if reverse else "")
                cases.append(case)
        maximum = 2**46
        for name, lens, schedules in [
            ("MAX_MIX_LEN scheduled halves", [maximum // 2, maximum // 2], [[0, 0, .5, .5], [.5, .5, 1, 1]]),
            ("MAX_MIX_LEN complementary ramps", [maximum // 2, maximum // 2], [[0, 1, 1, 1], [0, 0, 0, 1]]),
            ("MAX_MIX_LEN uniform minorities", [1, 3, 7, maximum - 11], [None, None, None, [0, 0, 1, 1]]),
            ("large late virtual clocks", [2**40, 2**40, 3], [[.9, .9, 1, 1], [.95, .97, 1, 1], [0, 0, .1, .1]]),
            ("large interacting ramps", [2**42, 2**42, 13, 7], [[0, 1, 1, 1], [0, 0, 0, 1], None, None]),
        ]:
            cases.append(sparse_case(name, lens, schedules))
            cases.append(sparse_case(name + " reversed", lens[::-1], schedules[::-1]))
        rng = random.Random(0xDA7A5C4)
        for _ in range(12):
            points = sorted(rng.sample(range(1, 16), 4))
            cases.append(small_case([211, 7, 5], [None, [p / 16 for p in points], [0, 0, .75, 1]]))
        for n in [10**9, 10**12, 10**13]:
            # The singleton has stagger 1/4. The other part has stagger 3/4.
            # With constant rates, its exact rank is N/4-1 (the singleton wins ties).
            rank = n // 4 - 1
            for points in [[0, 0, 1, 1], [0, 1e-16, 1, 1]]:
                ps = profiles([1, n - 1], [None, points])
                t = inverse(ps[0], F(1, 4))
                # Count the dominant part's keys below the singleton by bisection,
                # independently of any production seek, using the exact oracle CDF.
                low, high = 0, n - 1
                while low < high:
                    mid = (low + high) // 2
                    if inverse(ps[1], F(4 * mid + 3, 4 * (n - 1))) < t:
                        low = mid + 1
                    else:
                        high = mid
                if points[1] == 0:
                    assert low == rank
                samples = [[p, 0 if p == low else 1, 0 if p == low else p - (p > low)]
                           for p in range(low - 3, low + 4)]
                cases.append(dict(lens=[1, n - 1], schedules=[None, points], samples=samples))
    return json.dumps(cases, separators=(",", ":")) + "\n"


path = Path(__file__).with_name("schedule_oracle.json")
content = fixtures()
if "--check" in sys.argv:
    assert path.read_text() == content, "schedule fixtures differ from the independent oracle"
else:
    path.write_text(content)
