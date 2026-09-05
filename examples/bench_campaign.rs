//! Repeated benchmark runs and comparison tables. See `--help` for usage.
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Measurements = Vec<Option<f64>>;
type Rows = BTreeMap<String, Measurements>;

const TITLES: [&str; 6] = ["seek µs", "walk ns/elem", "get ns", "build µs", "reused seek µs", "cursor requested bytes"];
const USAGE: &str = "cargo run --release --example bench_campaign -- run LABEL DIR [MODE]
cargo run --release --example bench_campaign -- table [COL]

MODE: default, phases, lifecycle, all (default: BENCH_MODE or default)
COL: 0 seek, 1 walk (default), 2 get, 3 build, 4 reused seek, 5 cursor bytes
Runs twice using BENCH_CORE (default: 2; none disables taskset).
Results: target/bench-campaign.json in the campaign runner's crate.";

#[derive(Default)]
struct Results {
    labels: Vec<String>,
    rows: BTreeMap<String, BTreeMap<String, Measurements>>,
}

impl Results {
    fn load(path: &Path) -> Result<Self> {
        let contents = match fs::read(path) {
            Ok(contents) => contents,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(error) => return Err(error.into()),
        };
        let mut value: Value = serde_json::from_slice(&contents)?;
        Ok(Self { labels: serde_json::from_value(value["labels"].take())?, rows: serde_json::from_value(value["rows"].take())? })
    }

    fn update(&mut self, label: String, rows: Rows) {
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

    fn save(&self, path: &Path) -> Result<()> {
        fs::create_dir_all(path.parent().ok_or("results path has no parent")?)?;
        fs::write(path, serde_json::to_vec_pretty(&json!({ "labels": self.labels, "rows": self.rows }))?)?;
        Ok(())
    }

    fn table(&self, col: usize) {
        print!("{:<40}", TITLES[col]);
        for label in &self.labels {
            print!("{:>14}", label.chars().take(13).collect::<String>());
        }
        println!();
        for (name, values) in &self.rows {
            print!("{:<40}", name.chars().take(40).collect::<String>());
            for label in &self.labels {
                match values.get(label).and_then(|row| row.get(col)).copied().flatten() {
                    Some(value) => print!("{value:>14.1}"),
                    None => print!("{:>14}", "-"),
                }
            }
            println!();
        }
    }
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
    let mut results = Results::load(path)?;
    let directory = directory.canonicalize()?;
    let build = checked_output(
        Command::new("cargo").args(["build", "--release", "--example", "bench", "--message-format=json"]).current_dir(&directory),
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
    let core = env::var("BENCH_CORE").unwrap_or_else(|_| "2".into());
    let mut rows = Rows::new();
    for round in 1..=2 {
        eprintln!("Benchmark run {round}/2 ({mode})");
        let mut command = if core == "none" {
            Command::new(&executable)
        } else {
            let mut command = Command::new("taskset");
            command.args(["-c", &core]).arg(&executable);
            command
        };
        if mode != "default" {
            command.arg(format!("--{mode}"));
        }
        let output = checked_output(command.current_dir(&directory))?;
        let current = parse_output(&String::from_utf8(output.stdout)?);
        if current.is_empty() {
            return Err(format!("benchmark run {round} produced no recognizable measurement rows").into());
        }
        merge_minima(&mut rows, current);
    }
    results.update(label, rows);
    results.save(path)?;
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
}
