//! Repeated benchmark runs and comparison tables. See `--help` for usage.
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::env;
use std::error::Error;
#[cfg(unix)]
use std::fs::File;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Measurements = Vec<Option<f64>>;
type Rows = BTreeMap<String, Measurements>;

static NONCE: AtomicU64 = AtomicU64::new(0);

const TITLES: [&str; 6] = ["seek µs", "walk ns/elem", "get ns", "build µs", "reused seek µs", "cursor requested bytes"];
const USAGE: &str = "cargo run --release --example bench_campaign -- run LABEL DIR [MODE]
cargo run --release --example bench_campaign -- table [COL]

MODE: default, phases, lifecycle, all (default: BENCH_MODE or default)
COL: 0 seek, 1 walk (default), 2 get, 3 build, 4 reused seek, 5 cursor bytes
Runs twice, unpinned by default. BENCH_CORE=N requests taskset affinity; none disables it.
Explicit affinity is checked before building. Raw runs and provenance are saved alongside minima.
Results: target/bench-campaign.json in the campaign runner's crate.";

#[derive(Default)]
struct Results {
    labels: Vec<String>,
    rows: BTreeMap<String, BTreeMap<String, Measurements>>,
    runs: BTreeMap<String, Value>,
}

impl Results {
    fn load(path: &Path) -> Result<Self> {
        let contents = match fs::read(path) {
            Ok(contents) => contents,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(error) => return Err(error.into()),
        };
        let mut value: Value = serde_json::from_slice(&contents)?;
        let runs = value.get_mut("runs").map(Value::take).map(serde_json::from_value).transpose()?.unwrap_or_default();
        Ok(Self { labels: serde_json::from_value(value["labels"].take())?, rows: serde_json::from_value(value["rows"].take())?, runs })
    }

    fn update(&mut self, label: String, rows: Rows) {
        self.runs.remove(&label);
        self.labels.retain(|old| old != &label);
        self.labels.push(label.clone());
        for values in self.rows.values_mut() {
            values.remove(&label);
        }
        for (name, values) in rows {
            self.rows.entry(name).or_default().insert(label.clone(), values);
        }
        self.rows.retain(|_, values| !values.is_empty());
    }

    /// Write a complete sibling file, sync it, then replace the old results in one rename.
    /// Readers observe either the old complete file or the new complete file.
    fn save(&self, path: &Path) -> Result<()> {
        let parent = path.parent().ok_or("results path has no parent")?;
        fs::create_dir_all(parent)?;
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let serial = NONCE.fetch_add(1, Ordering::Relaxed);
        let temporary = path.with_extension(format!("json.{}.{nonce}.{serial}.tmp", std::process::id()));
        let mut file = OpenOptions::new().write(true).create_new(true).open(&temporary)?;
        let result = (|| -> Result<()> {
            let bytes = serde_json::to_vec_pretty(&json!({ "labels": self.labels, "rows": self.rows, "runs": self.runs }))?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temporary, path)?;
            #[cfg(unix)]
            File::open(parent)?.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }

    fn table(&self, col: usize) {
        let name_width = self.rows.keys().map(|name| name.chars().count()).max().unwrap_or(0).max(40);
        print!("{:<name_width$}", TITLES[col]);
        for label in &self.labels {
            print!("{:>width$}", label, width = label.chars().count().max(14) + 2);
        }
        println!();
        for (name, values) in &self.rows {
            print!("{name:<name_width$}");
            for label in &self.labels {
                let rendered = values
                    .get(label)
                    .and_then(|row| row.get(col))
                    .copied()
                    .flatten()
                    .map_or_else(|| "-".into(), |value| measurement(value, col));
                print!("{:>width$}", rendered, width = label.chars().count().max(14) + 2);
            }
            println!();
        }
    }
}

fn measurement(value: f64, col: usize) -> String {
    match col {
        0 | 3 | 4 => format!("{value:.3}"),
        5 => format!("{value:.0}"),
        _ => format!("{value:.2}"),
    }
}

/// Lock only the final read/merge/write, after benchmarking. The persistent lock file
/// must not be removed: all writers need to lock the same inode. OS locks release on exit.
fn commit(path: &Path, label: String, rows: Rows, run: Value) -> Result<Results> {
    fs::create_dir_all(path.parent().ok_or("results path has no parent")?)?;
    let lock = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path.with_extension("json.lock"))?;
    lock.lock()?;
    let mut results = Results::load(path)?;
    results.update(label.clone(), rows);
    results.runs.insert(label, run);
    results.save(path)?;
    // Closing the file releases the lock even when loading or saving fails.
    drop(lock);
    Ok(results)
}

#[derive(Debug, PartialEq, Eq)]
enum Affinity {
    Unpinned,
    Core(usize),
}

impl Affinity {
    fn parse(value: Option<&str>) -> Result<Self> {
        match value {
            None | Some("none") => Ok(Self::Unpinned),
            Some(value) => Ok(Self::Core(value.parse().map_err(|_| "BENCH_CORE must be a nonnegative integer or none")?)),
        }
    }

    fn command(&self, executable: impl AsRef<std::ffi::OsStr>) -> Command {
        match self {
            Self::Unpinned => Command::new(executable),
            Self::Core(core) => {
                let mut command = Command::new("taskset");
                command.args(["-c", &core.to_string()]).arg(executable);
                command
            }
        }
    }

    fn validate(&self, directory: &Path) -> Result<()> {
        if matches!(self, Self::Core(_)) {
            checked_output(self.command("rustc").arg("--version").current_dir(directory))
                .map_err(|e| format!("requested BENCH_CORE affinity is unavailable; use BENCH_CORE=none to run unpinned: {e}"))?;
        }
        Ok(())
    }
}

fn optional_output(command: &mut Command) -> Option<String> {
    command.output().ok().filter(|output| output.status.success()).map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn provenance(directory: &Path, mode: &str, affinity: &Affinity) -> Result<Value> {
    let revision = optional_output(Command::new("git").args(["rev-parse", "HEAD"]).current_dir(directory));
    let changes = optional_output(Command::new("git").args(["status", "--porcelain"]).current_dir(directory));
    let compiler = String::from_utf8(checked_output(Command::new("rustc").arg("-Vv").current_dir(directory))?.stdout)?;
    let cpu = if cfg!(target_os = "macos") {
        optional_output(Command::new("sysctl").args(["-n", "machdep.cpu.brand_string"]))
    } else if cfg!(target_os = "linux") {
        fs::read_to_string("/proc/cpuinfo").ok().and_then(|info| {
            info.lines().find_map(|line| {
                line.strip_prefix("model name").and_then(|rest| rest.split_once(':')).map(|(_, model)| model.trim().to_owned())
            })
        })
    } else {
        env::var("PROCESSOR_IDENTIFIER").ok()
    };
    let system = if cfg!(windows) {
        optional_output(Command::new("cmd").args(["/C", "ver"]))
    } else {
        optional_output(Command::new("uname").arg("-a"))
    };
    Ok(json!({
        "started_at_unix_seconds": SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        "directory": directory, "revision": revision, "working_tree_status": changes,
        "compiler": compiler.trim(), "os": env::consts::OS, "arch": env::consts::ARCH,
        "system": system, "cpu": cpu, "mode": mode, "profile": "release",
        "affinity": match affinity { Affinity::Unpinned => Value::Null, Affinity::Core(core) => json!({"core": core}) },
        "rustflags": env::var("RUSTFLAGS").ok(), "encoded_rustflags": env::var("CARGO_ENCODED_RUSTFLAGS").ok(),
        "cargo_build_target": env::var("CARGO_BUILD_TARGET").ok(),
        "build_command": ["cargo", "build", "--locked", "--release", "--example", "bench", "--message-format=json"],
    }))
}

fn parse_output(output: &str) -> Rows {
    let mut rows = Rows::new();
    for line in output.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() < 7 {
            continue;
        }
        let (name, values) = fields.split_at(fields.len() - 6);
        let lifecycle = name[0] == "lifecycle";
        if [values[1], values[3], values[5]] != if lifecycle { ["µs", "µs", "B"] } else { ["µs", "ns", "ns"] } {
            continue;
        }
        let parsed: std::result::Result<Vec<f64>, _> = [values[0], values[2], values[4]].iter().map(|value| value.parse()).collect();
        let Ok(parsed) = parsed else { continue };
        if parsed.iter().any(|value| !value.is_finite() || *value < 0.0) {
            continue;
        }
        let name_text = values.iter().rev().fold(line.trim_end(), |rest, field| rest[..rest.len() - field.len()].trim_end()).trim();
        let mut measurements = Vec::new();
        let name = if lifecycle {
            if name.len() == 1 {
                continue;
            }
            measurements.extend([None; 3]);
            format!("lifecycle: {}", name_text["lifecycle".len()..].trim())
        } else {
            name_text.to_owned()
        };
        measurements.extend(parsed.into_iter().map(Some));
        rows.insert(name, measurements);
    }
    rows
}

fn merge_minima(rows: &mut Rows, current: Rows) {
    for (name, values) in current {
        rows.entry(name)
            .and_modify(|previous| {
                *previous = previous.iter().zip(&values).map(|(a, b)| a.zip(*b).map(|(a, b)| a.min(b))).collect();
            })
            .or_insert(values);
    }
}

fn checked_output(command: &mut Command) -> Result<Output> {
    let output = command.output().map_err(|error| format!("{command:?}: {error}"))?;
    if !output.status.success() {
        return Err(format!("{command:?} failed ({}):\n{}", output.status, String::from_utf8_lossy(&output.stderr)).into());
    }
    Ok(output)
}

fn run(label: String, directory: &Path, mode: &str, path: &Path) -> Result<()> {
    if !["default", "phases", "lifecycle", "all"].contains(&mode) {
        return Err(format!("unknown benchmark mode: {mode}").into());
    }
    let directory = directory.canonicalize()?;
    let affinity = Affinity::parse(env::var("BENCH_CORE").ok().as_deref())?;
    affinity.validate(&directory)?;
    let metadata = provenance(&directory, mode, &affinity)?;
    let build = checked_output(
        Command::new("cargo")
            .args(["build", "--locked", "--release", "--example", "bench", "--message-format=json"])
            .current_dir(&directory),
    )?;
    let executable = String::from_utf8(build.stdout)?
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find_map(|value| {
            (value["reason"] == "compiler-artifact" && value["target"]["name"] == "bench")
                .then(|| value["executable"].as_str().map(PathBuf::from))
                .flatten()
        })
        .ok_or("cargo did not report the bench executable")?;
    let mut samples = Vec::new();
    let mut rows = Rows::new();
    for round in 1..=2 {
        eprintln!("Benchmark run {round}/2 ({mode})");
        let mut command = affinity.command(&executable);
        if mode != "default" {
            command.arg(format!("--{mode}"));
        }
        let output = checked_output(command.current_dir(&directory))?;
        let raw = String::from_utf8(output.stdout)?;
        let current = parse_output(&raw);
        if current.is_empty() {
            return Err(format!("benchmark run {round} produced no recognizable measurement rows").into());
        }
        samples.push(json!({"round": round, "rows": current, "stdout": raw}));
        merge_minima(&mut rows, current);
    }
    let results = commit(path, label, rows, json!({"metadata": metadata, "samples": samples}))?;
    results.table(if mode == "lifecycle" { 3 } else { 1 });
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bench-campaign.json");
    match args.first().map(String::as_str) {
        Some("run") if (3..=4).contains(&args.len()) => {
            let mode = args.get(3).cloned().unwrap_or_else(|| env::var("BENCH_MODE").unwrap_or_else(|_| "default".into()));
            run(args[1].clone(), Path::new(&args[2]), &mode, &path)
        }
        Some("table") if args.len() <= 2 => {
            let col = args.get(1).map(|value| value.parse::<usize>()).transpose()?.unwrap_or(1);
            if col >= TITLES.len() {
                return Err("table column must be between 0 and 5".into());
            }
            Results::load(&path)?.table(col);
            Ok(())
        }
        Some("--help" | "-h") => {
            println!("{USAGE}");
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measurements_and_minima() {
        let mut rows = parse_output(
            "order  seek  walk / elem  get(pos)\nsource 1e9  0.02 µs  1.4 ns  3.1 ns\nlifecycle mix(100 sources)  8.20 µs  2.30 µs  4096 B\n",
        );
        merge_minima(&mut rows, parse_output("source 1e9  0.03 µs  1.2 ns  3.0 ns\nlifecycle mix(100 sources)  8.10 µs  2.40 µs  4000 B"));
        assert_eq!(rows["source 1e9"], vec![Some(0.02), Some(1.2), Some(3.0)]);
        assert_eq!(rows["lifecycle: mix(100 sources)"], vec![None, None, None, Some(8.1), Some(2.3), Some(4000.0)]);
        assert_eq!(rows.len(), 2);
        assert!(parse_output("invalid  NaN µs  1 ns  2 ns\ninvalid  1 µs  2 ms  3 ns").is_empty());
        assert!(parse_output("shuffle(mix)  [slow path]  1 µs  2 ns  3 ns").contains_key("shuffle(mix)  [slow path]"));
    }

    #[test]
    fn replacing_label_removes_stale_measurements() {
        let mut results = Results::default();
        results.update("baseline".into(), parse_output("source  1 µs  2 ns  3 ns"));
        results.update("candidate".into(), parse_output("source  2 µs  3 ns  4 ns\nold  3 µs  4 ns  5 ns"));
        results.update("candidate".into(), parse_output("new  4 µs  5 ns  6 ns"));
        assert_eq!(results.labels, ["baseline", "candidate"]);
        assert_eq!(results.rows["source"].len(), 1);
        assert!(!results.rows.contains_key("old"));
        assert!(results.rows["new"].contains_key("candidate"));
    }

    struct TestDirectory(PathBuf);
    impl TestDirectory {
        fn new() -> Self {
            let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
            let serial = NONCE.fetch_add(1, Ordering::Relaxed);
            let path = env::temp_dir().join(format!("dataorder-campaign-{}-{nonce}-{serial}", std::process::id()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn affinity_is_optional_and_small_timings_keep_precision() {
        assert_eq!(Affinity::parse(None).unwrap(), Affinity::Unpinned);
        assert_eq!(Affinity::parse(Some("none")).unwrap(), Affinity::Unpinned);
        assert_eq!(Affinity::parse(Some("2")).unwrap(), Affinity::Core(2));
        for bad in ["", "-1", "invalid"] {
            assert!(Affinity::parse(Some(bad)).is_err());
        }
        assert_eq!(Affinity::Unpinned.command("bench").get_program(), "bench");
        assert_eq!(Affinity::Core(2).command("bench").get_program(), "taskset");
        assert_eq!(measurement(0.02, 0), "0.020");
        assert_eq!(measurement(4096.0, 5), "4096");
    }

    #[test]
    fn legacy_results_load_and_new_runs_retain_raw_samples() {
        let directory = TestDirectory::new();
        let path = directory.0.join("results.json");
        fs::write(&path, r#"{"labels":["legacy"],"rows":{"source":{"legacy":[0.02,1.2,3.0]}}}"#).unwrap();
        let old = Results::load(&path).unwrap();
        assert!(old.runs.is_empty());
        let run = json!({"metadata":{"revision":"abc","compiler":"rustc"},"samples":[{"rows":{"source":[0.03,1.4,3.1]}},{"rows":{"source":[0.02,1.2,3.0]}}]});
        commit(&path, "new".into(), parse_output("source 0.02 µs 1.2 ns 3.0 ns"), run.clone()).unwrap();
        let saved = Results::load(&path).unwrap();
        assert_eq!(saved.labels, ["legacy", "new"]);
        assert_eq!(saved.runs["new"], run);
        commit(&path, "new".into(), parse_output("replacement 0.04 µs 2 ns 4 ns"), json!({"samples":[]})).unwrap();
        let saved = Results::load(&path).unwrap();
        assert!(!saved.rows["source"].contains_key("new"));
        assert_eq!(saved.runs["new"], json!({"samples":[]}));
    }

    /// Child entry point, also used to check that process death releases an OS lock.
    #[test]
    fn campaign_writer() {
        let Ok(directory) = env::var("DATAORDER_CAMPAIGN_TEST_DIR") else { return };
        let directory = Path::new(&directory);
        let label = env::var("DATAORDER_CAMPAIGN_TEST_LABEL").unwrap();
        let path = directory.join("results.json");
        if label == "lock-holder" {
            let lock =
                OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path.with_extension("json.lock")).unwrap();
            lock.lock().unwrap();
            fs::write(directory.join("ready-lock-holder"), "ready").unwrap();
            loop {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }
        fs::write(directory.join(format!("ready-{label}")), "ready").unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !directory.join("start").exists() {
            assert!(std::time::Instant::now() < deadline, "parent did not release writers");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        for round in 0..10 {
            commit(&path, label.clone(), parse_output("source 0.02 µs 1.2 ns 3.0 ns"), json!({"round":round})).unwrap();
        }
    }

    fn writer(directory: &Path, label: &str) -> std::process::Child {
        Command::new(env::current_exe().unwrap())
            .args(["--exact", "tests::campaign_writer"])
            .env("DATAORDER_CAMPAIGN_TEST_DIR", directory)
            .env("DATAORDER_CAMPAIGN_TEST_LABEL", label)
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap()
    }

    fn wait_ready(directory: &Path, labels: &[&str], children: &mut [std::process::Child]) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while labels.iter().any(|label| !directory.join(format!("ready-{label}")).exists()) {
            if std::time::Instant::now() >= deadline {
                for child in children {
                    let _ = child.kill();
                    let _ = child.wait();
                }
                panic!("campaign test children did not become ready");
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    #[test]
    fn concurrent_processes_merge_results_and_readers_see_complete_json() {
        let directory = TestDirectory::new();
        let path = directory.0.join("results.json");
        commit(&path, "initial".into(), Rows::new(), json!({})).unwrap();
        let labels = ["a", "b", "c", "d"];
        let mut children: Vec<_> = labels.iter().map(|label| writer(&directory.0, label)).collect();
        wait_ready(&directory.0, &labels, &mut children);
        fs::write(directory.0.join("start"), "start").unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut finished = vec![false; children.len()];
        loop {
            Results::load(&path).unwrap(); // Repeated reads race with atomic replacement.
            for (i, child) in children.iter_mut().enumerate() {
                if let Some(status) = child.try_wait().unwrap() {
                    assert!(status.success());
                    finished[i] = true;
                }
            }
            if finished.iter().all(|&done| done) {
                break;
            }
            if std::time::Instant::now() >= deadline {
                for child in &mut children {
                    let _ = child.kill();
                    let _ = child.wait();
                }
                panic!("campaign writers timed out");
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let saved = Results::load(&path).unwrap();
        assert_eq!(saved.labels.len(), 5);
        for label in labels {
            assert_eq!(saved.runs[label]["round"], 9);
        }
    }

    #[test]
    fn killed_process_releases_campaign_lock() {
        let directory = TestDirectory::new();
        let mut children = [writer(&directory.0, "lock-holder")];
        wait_ready(&directory.0, &["lock-holder"], &mut children);
        let lock = OpenOptions::new().read(true).write(true).open(directory.0.join("results.json.lock")).unwrap();
        assert!(matches!(lock.try_lock(), Err(std::fs::TryLockError::WouldBlock)));
        children[0].kill().unwrap();
        children[0].wait().unwrap();
        // Nonblocking acquisition makes a lock leak fail immediately rather than hang.
        lock.try_lock().unwrap();
    }
}
