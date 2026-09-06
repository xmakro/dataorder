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
#[path = "support/measurements.rs"]
#[allow(dead_code)]
mod measurements;
use measurements::{Measurements, Report, Rows, summarize};
const RUNS: usize = 6;

static NONCE: AtomicU64 = AtomicU64::new(0);

const TITLES: [&str; 6] = ["seek µs", "walk ns/elem", "get ns", "build µs", "reused seek µs", "cursor requested bytes"];
const USAGE: &str = "cargo run --release --example bench_campaign -- run LABEL DIR [MODE]
cargo run --release --example bench_campaign -- compare LABEL_A DIR_A LABEL_B DIR_B [MODE]
cargo run --release --example bench_campaign -- table [COL] [LABEL ...]

MODE: default, phases, lifecycle, all (default: BENCH_MODE or default)
COL: 0 seek, 1 walk (default), 2 get, 3 build, 4 reused seek, 5 cursor bytes
Six runs per revision; compare alternates revision order on each round.
Each benchmark reports five calibrated samples per metric. Tables show median [min..max].
Compared revisions must use identical bench.rs and support files and matching workloads.
Unpinned by default. BENCH_CORE=N requests taskset affinity; none disables it.
Raw samples, workload fingerprints and provenance are saved.
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

    fn validate_comparison(&self, labels: &[String]) -> Result<()> {
        for (i, label) in labels.iter().enumerate() {
            if !self.labels.contains(label) {
                return Err(format!("unknown campaign label: {label}").into());
            }
            for previous in &labels[..i] {
                let a = &self.runs.get(label).ok_or("legacy results have no comparable workload identity; select one label")?["metadata"];
                let b =
                    &self.runs.get(previous).ok_or("legacy results have no comparable workload identity; select one label")?["metadata"];
                if a["harness"].as_str().is_none() || a["harness"] != b["harness"] {
                    return Err(format!("{label} and {previous}: incompatible or unverified benchmark harnesses; select one label or rerun with identical harnesses").into());
                }
                for (name, values) in &self.rows {
                    if values.contains_key(label)
                        && values.contains_key(previous)
                        && (a["workloads"][name].as_str().is_none() || a["workloads"][name] != b["workloads"][name])
                    {
                        return Err(format!("{name}: workload differs between {label} and {previous}").into());
                    }
                }
            }
        }
        Ok(())
    }

    fn cell(&self, name: &str, label: &str, col: usize) -> String {
        let Some(value) = self.rows[name].get(label).and_then(|row| row.get(col)).copied().flatten() else { return "-".into() };
        let middle = measurement(value, col);
        let range = self.runs.get(label).and_then(|run| run["ranges"].get(name));
        if let Some((low, high)) = range.and_then(|r| r[0][col].as_f64().zip(r[1][col].as_f64()))
            && low != high
        {
            return format!("{middle} [{}..{}]", measurement(low, col), measurement(high, col));
        }
        middle
    }

    fn table(&self, col: usize, labels: &[String]) -> Result<()> {
        self.validate_comparison(labels)?;
        let name_width = self.rows.keys().map(|name| name.chars().count()).max().unwrap_or(0).max(40);
        let widths: Vec<_> = labels
            .iter()
            .map(|label| {
                self.rows
                    .keys()
                    .map(|name| self.cell(name, label, col).chars().count())
                    .max()
                    .unwrap_or(0)
                    .max(label.chars().count())
                    .max(14)
                    + 2
            })
            .collect();
        println!("Median [min..max] across retained samples; legacy single-label values retain their original statistic.");
        print!("{:<name_width$}", TITLES[col]);
        for (label, &width) in labels.iter().zip(&widths) {
            print!("{label:>width$}");
        }
        println!();
        for name in self.rows.keys().filter(|name| labels.iter().any(|label| self.rows[*name].contains_key(label))) {
            print!("{name:<name_width$}");
            for (label, &width) in labels.iter().zip(&widths) {
                print!("{:>width$}", self.cell(name, label, col));
            }
            println!();
        }
        Ok(())
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

type Ranges = BTreeMap<String, [Measurements; 2]>;

fn aggregate(reports: &[Report]) -> Result<(Rows, Ranges)> {
    let first = reports.first().ok_or("no benchmark samples")?;
    first.validate()?;
    for report in &reports[1..] {
        report.validate()?;
        first.compatible(report)?;
    }
    let (mut rows, mut ranges) = (Rows::new(), Ranges::new());
    for name in first.rows.keys() {
        let samples: Vec<_> = reports.iter().flat_map(|report| report.rows[name].samples.iter().cloned()).collect();
        let (median, low, high) = summarize(&samples);
        rows.insert(name.clone(), median);
        ranges.insert(name.clone(), [low, high]);
    }
    Ok((rows, ranges))
}

fn checked_output(command: &mut Command) -> Result<Output> {
    let output = command.output().map_err(|error| format!("{command:?}: {error}"))?;
    if !output.status.success() {
        return Err(format!("{command:?} failed ({}):\n{}", output.status, String::from_utf8_lossy(&output.stderr)).into());
    }
    Ok(output)
}

/// Keep each build independent even if checkouts share CARGO_TARGET_DIR or a
/// concurrent build replaces the original artifact between measurement rounds.
struct ExecutableSnapshot(PathBuf);

impl ExecutableSnapshot {
    fn copy(executable: &Path) -> Result<Self> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let serial = NONCE.fetch_add(1, Ordering::Relaxed);
        let directory = env::temp_dir().join(format!("dataorder-bench-{}-{nonce}-{serial}", std::process::id()));
        fs::create_dir(&directory)?;
        let snapshot = Self(directory.join(executable.file_name().ok_or("benchmark executable has no file name")?));
        fs::copy(executable, &snapshot.0)?;
        Ok(snapshot)
    }
}

impl Drop for ExecutableSnapshot {
    fn drop(&mut self) {
        if let Some(directory) = self.0.parent() {
            let _ = fs::remove_dir_all(directory);
        }
    }
}

struct Campaign {
    label: String,
    directory: PathBuf,
    affinity: Affinity,
    metadata: Value,
    executable: ExecutableSnapshot,
    samples: Vec<Value>,
    reports: Vec<Report>,
}

impl Campaign {
    fn prepare(label: String, directory: &Path, mode: &str) -> Result<Self> {
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
        let executable = ExecutableSnapshot::copy(&executable)?;
        Ok(Self { label, directory, affinity, metadata, executable, samples: Vec::new(), reports: Vec::new() })
    }

    fn sample(&mut self, mode: &str, round: usize) -> Result<()> {
        eprintln!("Benchmark {} run {}/{RUNS} ({mode})", self.label, round + 1);
        let mut command = self.affinity.command(&self.executable.0);
        if mode != "default" {
            command.arg(format!("--{mode}"));
        }
        let output = checked_output(command.current_dir(&self.directory))?;
        let raw = String::from_utf8(output.stdout)?;
        let report = Report::parse(&raw)?;
        if report.mode != mode {
            return Err("benchmark reported the wrong mode".into());
        }
        if let Some(first) = self.reports.first() {
            first.compatible(&report)?;
        }
        self.samples.push(json!({"round": round + 1, "report": report, "stdout": raw, "stderr": String::from_utf8_lossy(&output.stderr)}));
        self.reports.push(report);
        Ok(())
    }
}

fn run(targets: Vec<(String, PathBuf)>, mode: &str, path: &Path) -> Result<()> {
    if !["default", "phases", "lifecycle", "all"].contains(&mode) {
        return Err(format!("unknown benchmark mode: {mode}").into());
    }
    if targets.iter().enumerate().any(|(i, (label, _))| label.is_empty() || targets[..i].iter().any(|t| &t.0 == label)) {
        return Err("campaign labels must be nonempty and distinct".into());
    }
    let mut campaigns = targets.into_iter().map(|(label, dir)| Campaign::prepare(label, &dir, mode)).collect::<Result<Vec<_>>>()?;
    for round in 0..RUNS {
        let mut order: Vec<_> = (0..campaigns.len()).collect();
        if round % 2 == 1 {
            order.reverse();
        }
        for i in order {
            campaigns[i].sample(mode, round)?;
            if campaigns.len() > 1 && campaigns.iter().all(|c| !c.reports.is_empty()) {
                campaigns[0].reports[0].compatible(&campaigns[1].reports[0])?;
            }
        }
    }
    let labels: Vec<_> = campaigns.iter().map(|c| c.label.clone()).collect();
    for mut campaign in campaigns {
        let (rows, ranges) = aggregate(&campaign.reports)?;
        let report = &campaign.reports[0];
        campaign.metadata["harness"] = json!(report.harness);
        campaign.metadata["workloads"] = json!(report.rows.iter().map(|(name, row)| (name, &row.workload)).collect::<BTreeMap<_, _>>());
        campaign.metadata["statistic"] = json!("median of calibrated samples; range is minimum to maximum");
        campaign.metadata["rounds"] = json!(RUNS);
        commit(path, campaign.label, rows, json!({"metadata":campaign.metadata,"samples":campaign.samples,"ranges":ranges}))?;
    }
    Results::load(path)?.table(if mode == "lifecycle" { 3 } else { 1 }, &labels)
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bench-campaign.json");
    match args.first().map(String::as_str) {
        Some("run") if (3..=4).contains(&args.len()) => {
            let mode = args.get(3).cloned().unwrap_or_else(|| env::var("BENCH_MODE").unwrap_or_else(|_| "default".into()));
            run(vec![(args[1].clone(), PathBuf::from(&args[2]))], &mode, &path)
        }
        Some("compare") if (5..=6).contains(&args.len()) => {
            let mode = args.get(5).cloned().unwrap_or_else(|| env::var("BENCH_MODE").unwrap_or_else(|_| "default".into()));
            run(vec![(args[1].clone(), PathBuf::from(&args[2])), (args[3].clone(), PathBuf::from(&args[4]))], &mode, &path)
        }
        Some("table") => {
            let col = args.get(1).map(|value| value.parse::<usize>()).transpose()?.unwrap_or(1);
            if col >= TITLES.len() {
                return Err("table column must be between 0 and 5".into());
            }
            let results = Results::load(&path)?;
            let labels = if args.len() > 2 { &args[2..] } else { &results.labels };
            results.table(col, labels)
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

    fn row(name: &str, values: [f64; 3]) -> Rows {
        BTreeMap::from([(name.into(), values.into_iter().map(Some).collect())])
    }

    #[test]
    fn incompatible_saved_results_are_not_compared() {
        let mut results = Results::default();
        results.update("old".into(), row("source", [1.0, 2.0, 3.0]));
        results.update("new".into(), row("source", [1.0, 2.0, 3.0]));
        assert!(results.validate_comparison(&results.labels).is_err());
        for label in ["old", "new"] {
            results.runs.insert(label.into(), json!({"metadata":{"harness":"same","workloads":{"source":"same"}}}));
        }
        results.validate_comparison(&results.labels).unwrap();
        results.runs.get_mut("new").unwrap()["metadata"]["workloads"]["source"] = json!("different");
        assert!(results.validate_comparison(&results.labels).is_err());
        results.validate_comparison(&["old".into()]).unwrap();
    }

    #[test]
    fn aggregation_uses_every_sample_and_rejects_incomplete_runs() {
        use measurements::Measurement;
        let mut a = Report {
            schema: 1,
            harness: "same".into(),
            mode: "default".into(),
            rows: BTreeMap::from([(
                "source".into(),
                Measurement { workload: "same".into(), samples: vec![vec![Some(1.0), Some(2.0), Some(3.0), None, None, None]; 3] },
            )]),
        };
        let mut b = a.clone();
        for sample in &mut b.rows.get_mut("source").unwrap().samples {
            sample[0] = Some(9.0);
        }
        let (rows, ranges) = aggregate(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(rows["source"][0], Some(5.0));
        assert_eq!((ranges["source"][0][0], ranges["source"][1][0]), (Some(1.0), Some(9.0)));
        a.rows.insert("extra".into(), a.rows["source"].clone());
        assert!(aggregate(&[a, b]).is_err());
    }

    #[test]
    fn replacing_label_removes_stale_measurements() {
        let mut results = Results::default();
        results.update("baseline".into(), row("source", [1.0, 2.0, 3.0]));
        results.update("candidate".into(), row("source", [2.0, 3.0, 4.0]).into_iter().chain(row("old", [3.0, 4.0, 5.0])).collect());
        results.update("candidate".into(), row("new", [4.0, 5.0, 6.0]));
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
    fn executable_snapshots_survive_replaced_build_artifacts() {
        let directory = TestDirectory::new();
        let artifact = directory.0.join("bench");
        fs::write(&artifact, "before").unwrap();
        let before = ExecutableSnapshot::copy(&artifact).unwrap();
        fs::write(&artifact, "after").unwrap();
        let after = ExecutableSnapshot::copy(&artifact).unwrap();
        assert_eq!(fs::read_to_string(&before.0).unwrap(), "before");
        assert_eq!(fs::read_to_string(&after.0).unwrap(), "after");
        let parent = before.0.parent().unwrap().to_owned();
        drop(before);
        assert!(!parent.exists());
    }

    #[test]
    fn legacy_results_load_and_new_runs_retain_raw_samples() {
        let directory = TestDirectory::new();
        let path = directory.0.join("results.json");
        fs::write(&path, r#"{"labels":["legacy"],"rows":{"source":{"legacy":[0.02,1.2,3.0]}}}"#).unwrap();
        let old = Results::load(&path).unwrap();
        assert!(old.runs.is_empty());
        let run = json!({"metadata":{"revision":"abc","compiler":"rustc"},"samples":[{"rows":{"source":[0.03,1.4,3.1]}},{"rows":{"source":[0.02,1.2,3.0]}}]});
        commit(&path, "new".into(), row("source", [0.02, 1.2, 3.0]), run.clone()).unwrap();
        let saved = Results::load(&path).unwrap();
        assert_eq!(saved.labels, ["legacy", "new"]);
        assert_eq!(saved.runs["new"], run);
        commit(&path, "new".into(), row("replacement", [0.04, 2.0, 4.0]), json!({"samples":[]})).unwrap();
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
            commit(&path, label.clone(), row("source", [0.02, 1.2, 3.0]), json!({"round":round})).unwrap();
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
