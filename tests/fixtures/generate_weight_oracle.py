"""Regenerate exact quota fixtures using Python's arbitrary-precision integers.

Run: python3 tests/fixtures/generate_weight_oracle.py [--check]
No dataorder implementation or floating-point quota arithmetic is used.
"""
from pathlib import Path
import random
import struct
import sys

rng = random.Random(0xDA7A0D3)
LIMIT = 1 << 46


def quotas(total, bits):
    ratios = [struct.unpack('>d', x.to_bytes(8, 'big'))[0].as_integer_ratio() for x in bits]
    common = max(d for _, d in ratios)
    integers = [n * (common // d) for n, d in ratios]
    denominator = sum(integers)
    counts, remainders = zip(*(divmod(total * n, denominator) for n in integers))
    counts = list(counts)
    ranked = sorted(range(len(bits)), key=lambda i: (-remainders[i], i))
    for i in ranked[:total - sum(counts)]:
        counts[i] += 1
    return counts


cases = []
for case in range(256):
    size = 2 + rng.getrandbits(4)
    total = [1, 3, 17, 1009, (1 << 32) - 1, LIMIT - 1, LIMIT][case % 7]
    if case % 4 == 0:
        # Full exponent range with subnormals, zeros and a finite maximum.
        bits = [rng.randrange(0x7ff0000000000000) for _ in range(size)]
        bits[0] = 1
        bits[1] = 0x7fefffffffffffff
    elif case % 4 == 1:
        # Close weights make remainder ranking sensitive to low significand bits.
        exponent = rng.randrange(1, 2046)
        base = (exponent << 52) | rng.getrandbits(52)
        bits = [base + rng.randrange(16) for _ in range(size)]
    elif case % 4 == 2:
        # Exact ties and tiny positive terms which break them, spanning ~2000 bits.
        scale = [1.0, 2.0 ** 1000, 2.0 ** -900][case % 3]
        values = [7 * scale, scale, scale, 5e-324, 0.0]
        rng.shuffle(values)
        bits = [int.from_bytes(struct.pack('>d', x), 'big') for x in values]
    else:
        # Ordinary fast-path and wide-fallback mixtures, including signed zeros.
        exponents = [rng.randrange(2047) for _ in range(size)]
        bits = [(e << 52) | rng.getrandbits(52) for e in exponents]
        bits[0] = 0x8000000000000000
    cases.append((total, bits))

lines = ['# total | exact binary64 weight bits (hex) | expected integer quotas',
         '# Generated independently by generate_weight_oracle.py; do not bless Rust output.']
for total, bits in cases:
    lines.append(f'{total}|'+','.join(f'{x:016x}' for x in bits)+'|'+','.join(map(str, quotas(total, bits))))
content = '\n'.join(lines) + '\n'
path = Path(__file__).with_name('weight_oracle.txt')
if '--check' in sys.argv:
    assert path.read_text() == content, 'weight fixtures differ from the independent oracle'
else:
    path.write_text(content)
