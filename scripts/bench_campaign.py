#!/usr/bin/env python3
"""Pinned, repeated benchmark runs with per-row minima, tabulated across labelled steps.

    scripts/bench_campaign.py run LABEL DIR   # builds and runs DIR/examples/bench twice on one core (BENCH_CORE, default 2), keeps minima
    scripts/bench_campaign.py table [COL]     # 0 seek, 1 walk, 2 get (default 1)

Results accumulate in results.json next to this script."""
import json, os, re, subprocess, sys
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')

def load():
    return json.load(open(RES)) if os.path.exists(RES) else {'labels': [], 'rows': {}}

def run(label, d, rounds=2, core=os.environ.get('BENCH_CORE', '2')):
    d = os.path.abspath(d)
    subprocess.run(['cargo', 'build', '--release', '--example', 'bench'], cwd=d, check=True, capture_output=True)
    rows = {}
    for round_index in range(rounds):
        out = subprocess.run(['taskset', '-c', core, os.path.join(d, 'target/release/examples/bench')], cwd=d, check=True, capture_output=True, text=True).stdout
        current = {}
        for line in out.splitlines():
            m = re.match(r'^(.*?)\s{2,}([\d.]+) µs\s+([\d.]+) ns\s+([\d.]+) ns$', line)
            if m:
                name = m.group(1).strip()
                v = [float(m.group(2)), float(m.group(3)), float(m.group(4))]
                current[name] = v
        if not current:
            raise RuntimeError(f'benchmark run {round_index + 1} produced no recognizable measurement rows')
        for name, v in current.items():
            rows[name] = [min(a, b) for a, b in zip(rows[name], v)] if name in rows else v
    if not rows:
        raise ValueError('at least one benchmark run is required')
    res = load()
    if label in res['labels']:
        res['labels'].remove(label)
    res['labels'].append(label)
    for values in res['rows'].values():
        values.pop(label, None)
    for name, v in rows.items():
        res['rows'].setdefault(name, {})[label] = v
    res['rows'] = {name: values for name, values in res['rows'].items() if values}
    json.dump(res, open(RES, 'w'), indent=1)
    table(res)

def table(res=None, col=1, labels=None):
    res = res or load()
    labels = labels or res['labels']
    title = ['seek µs', 'walk ns/elem', 'get ns'][col]
    print(f"{title:<40}" + ''.join(f"{l[:13]:>14}" for l in labels))
    for name, vals in res['rows'].items():
        print(f"{name[:40]:<40}" + ''.join(f"{vals[l][col]:>14.1f}" if l in vals and col < len(vals[l]) else f"{'-':>14}" for l in labels))

if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'table':
        table(col=int(sys.argv[2]) if len(sys.argv) > 2 else 1)
