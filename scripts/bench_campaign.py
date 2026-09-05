#!/usr/bin/env python3
"""Pinned, repeated benchmark runs with per-row minima, tabulated across labelled steps.

    scripts/bench_campaign.py run LABEL DIR [MODE]  # MODE: default, phases, lifecycle, all
    scripts/bench_campaign.py table [COL]     # 0 seek, 1 walk, 2 get, 3 build, 4 reused seek, 5 cursor bytes (default 1)

Runs twice on one core (BENCH_CORE, default 2; "none" disables pinning on platforms without
taskset), keeping minima. BENCH_MODE sets the default mode. Original three-column output
and stored results remain supported. Results accumulate in results.json next to this script."""
import json, os, re, subprocess, sys
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')

def load():
    return json.load(open(RES)) if os.path.exists(RES) else {'labels': [], 'rows': {}}

def parse_output(out):
    rows = {}
    for line in out.splitlines():
        m = re.match(r'^(.*?)\s{2,}([\d.]+) µs\s+([\d.]+) ns\s+([\d.]+) ns$', line)
        if m:
            rows[m.group(1).strip()] = [float(m.group(i)) for i in (2, 3, 4)]
            continue
        m = re.match(r'^lifecycle (.*?)\s{2,}([\d.]+) µs\s+([\d.]+) µs\s+(\d+) B$', line)
        if m:
            rows['lifecycle: ' + m.group(1).strip()] = [None, None, None, float(m.group(2)), float(m.group(3)), int(m.group(4))]
    return rows

def run(label, d, rounds=2, core=os.environ.get('BENCH_CORE', '2'), mode=os.environ.get('BENCH_MODE', 'default')):
    if mode not in ('default', 'phases', 'lifecycle', 'all'):
        raise ValueError(f'unknown benchmark mode: {mode}')
    d = os.path.abspath(d)
    subprocess.run(['cargo', 'build', '--release', '--example', 'bench'], cwd=d, check=True, capture_output=True)
    rows = {}
    for round_index in range(rounds):
        command = [os.path.join(d, 'target/release/examples/bench')]
        if mode != 'default':
            command.append('--' + mode)
        if core != 'none':
            command = ['taskset', '-c', core] + command
        out = subprocess.run(command, cwd=d, check=True, capture_output=True, text=True).stdout
        current = parse_output(out)
        if not current:
            raise RuntimeError(f'benchmark run {round_index + 1} produced no recognizable measurement rows')
        for name, v in current.items():
            rows[name] = [min(a, b) if a is not None and b is not None else None for a, b in zip(rows[name], v)] if name in rows else v
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
    table(res, col=3 if mode == 'lifecycle' else 1)

def table(res=None, col=1, labels=None):
    res = res or load()
    labels = labels or res['labels']
    title = ['seek µs', 'walk ns/elem', 'get ns', 'build µs', 'reused seek µs', 'cursor requested bytes'][col]
    print(f"{title:<40}" + ''.join(f"{l[:13]:>14}" for l in labels))
    for name, vals in res['rows'].items():
        print(f"{name[:40]:<40}" + ''.join(f"{vals[l][col]:>14.1f}" if l in vals and col < len(vals[l]) and vals[l][col] is not None else f"{'-':>14}" for l in labels))

if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run(sys.argv[2], sys.argv[3], mode=sys.argv[4] if len(sys.argv) > 4 else os.environ.get('BENCH_MODE', 'default'))
    elif sys.argv[1] == 'table':
        table(col=int(sys.argv[2]) if len(sys.argv) > 2 else 1)
