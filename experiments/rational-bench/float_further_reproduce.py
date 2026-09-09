"""Reproduce the adopted float changes and the rejected direct-evaluation trial."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
metadata = json.loads((HERE / "float-further-metadata.json").read_text())
for name, digest in metadata["adopted_sha256"].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name

# The earlier reproducer verifies and restores the original float baseline.
prepared = subprocess.check_output([sys.executable, str(HERE / "float_reproduce.py"), "--prepare-only"], text=True)
audit = Path(next(line.removeprefix("Prepared ") for line in prepared.splitlines() if line.startswith("Prepared ")))
shutil.copy2(audit / "src/sum.rs", audit / "sum-before.rs")
before = audit / "float-before/interleave/profile.rs"
before.write_text(before.read_text().replace("use crate::sum::", "use crate::sum_before::"))
shutil.copytree(ROOT / "src", audit / "src", dirs_exist_ok=True)
for name in ["Cargo.toml", "Cargo.lock"]:
    shutil.copy2(ROOT / name, audit / name)
shutil.copytree(audit / "src/interleave", audit / "float-final/interleave")
shutil.copytree(audit / "src/interleave", audit / "float-direct/interleave")
subprocess.run(["patch", "-p2", "-i", str(HERE / "float-direct.patch")], cwd=audit / "float-direct", check=True)

for source, example in [("float_further_walk.rs", "final_audit"), ("float_further_build.rs", "final_build")]:
    text = (HERE / source).read_text()
    (audit / "examples" / (example + ".rs")).write_text(text)
    reverse = text.replace('"../float-before/interleave/mod.rs"] mod before', '"../float-final/interleave/mod.rs"] mod before')
    reverse = reverse.replace('"../float-final/interleave/mod.rs"] mod after', '"../float-before/interleave/mod.rs"] mod after')
    reverse = reverse.replace("target/final-results.json", "target/final-reversed-results.json")
    reverse = reverse.replace("target/final-build-results.json", "target/final-reversed-build-results.json")
    (audit / "examples" / (example + "_reversed.rs")).write_text(reverse)
    if example == "final_build":
        direct = text.replace("../float-final/interleave/mod.rs", "../float-direct/interleave/mod.rs")
        direct = direct.replace("target/final-build-results.json", "target/direct-final-build-results.json")
        (audit / "examples/direct_build_final.rs").write_text(direct)
print(f"Prepared {audit}", flush=True)
if "--prepare-only" in sys.argv:
    raise SystemExit(0)

def run(*args):
    subprocess.run(args, cwd=audit, check=True)

run("cargo", "test", "--release", "--offline", "--all-features", "--lib", "--tests")
examples = ["final_audit", "final_build", "final_audit_reversed", "final_build_reversed", "direct_build_final"]
run("cargo", "build", "--release", "--offline", *[argument for name in examples for argument in ("--example", name)])
for name in examples:
    run("./target/release/examples/" + name)
print(f"Results: {audit / 'target'}")
