"""Build the float simplification in an isolated copy; production stays unchanged."""
from decimal import Decimal as D, localcontext
from pathlib import Path
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
metadata = json.loads((HERE / "float-metadata.json").read_text())
baseline = {}
for name, digest in metadata["baseline_sha256"].items():
    data = (ROOT / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        saved = HERE / "float-baseline" / name
        data = saved.read_bytes() if saved.exists() else subprocess.check_output(
            ["git", "show", f"{metadata['baseline_git_revision']}:{name}"], cwd=ROOT
        )
    assert hashlib.sha256(data).hexdigest() == digest, name
    baseline[name] = data

audit = Path(tempfile.mkdtemp(prefix="dataorder-float-audit-"))
for name in subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0"):
    if not name or name.startswith((".git", "experiments/")):
        continue
    source = ROOT / name
    if source.is_file():
        target = audit / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
for name, data in baseline.items():
    (audit / name).write_bytes(data)
shutil.copytree(audit / "src/interleave", audit / "float-before/interleave")
subprocess.run(["patch", "-p1", "-i", str(HERE / "float-inverse.patch")], cwd=audit, check=True)
profile = audit / "src/interleave/profile.rs"
assert hashlib.sha256(profile.read_bytes()).hexdigest() == metadata["candidate_profile_sha256"]
profile.write_text(profile.read_text() + (HERE / "float-numeric-test.rs").read_text())
bench = (HERE / "float_bench.rs").read_text()
(audit / "examples/float_audit.rs").write_text(bench)
reversed_bench = bench.replace('"../float-before/interleave/mod.rs"] mod before', '"../src/interleave/mod.rs"] mod before')
reversed_bench = reversed_bench.replace('"../src/interleave/mod.rs"] mod after', '"../float-before/interleave/mod.rs"] mod after')
reversed_bench = reversed_bench.replace("target/float-results.json", "target/float-reversed-results.json")
(audit / "examples/float_audit_reversed.rs").write_text(reversed_bench)
print(f"Prepared {audit}", flush=True)
if "--prepare-only" in sys.argv:
    raise SystemExit(0)

def run(*args):
    subprocess.run(args, cwd=audit, check=True)

run("cargo", "test", "--release", "--offline", "--all-features", "--lib", "--tests")
run("cargo", "test", "--release", "--offline", "--all-features", "--doc")
rows = json.loads((audit / "target/float-numeric.json").read_text())
worst = D(0)
with localcontext() as context:
    context.prec = 150
    for row in rows:
        start, end, r0, r1, y, actual = map(D.from_float, row)
        width = end - start
        c = (r1 - r0) / (2 * width)
        root = (r0*r0 + 4*c*y).sqrt()
        x = 2*y / (r0 + root) if y else D(0)
        expected = min(end, max(start, start + x))
        unit = max(D.from_float(math.ulp(row[0])), D.from_float(math.ulp(row[1])), width * D.from_float(2**-52))
        worst = max(worst, abs(actual - expected) / unit)
assert worst < 8, worst
print(f"Independent Decimal reference: {len(rows)} samples, worst {worst} endpoint-resolution units", flush=True)
run("cargo", "build", "--release", "--offline", "--example", "float_audit", "--example", "float_audit_reversed")
run("./target/release/examples/float_audit", "--compatibility")
run("./target/release/examples/float_audit")
run("./target/release/examples/float_audit_reversed")
print(f"Results: {audit / 'target'}")
