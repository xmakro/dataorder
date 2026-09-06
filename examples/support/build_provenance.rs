//! Isolated benchmark builds and compiler-observed settings. Cargo resolves its
//! own configuration, compiler overrides and wrappers; the runner does not guess
//! those settings from its environment or parse Cargo's TOML hierarchy.
use super::{NONCE, Result, checked_output};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::Ordering;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct Build {
    directory: PathBuf,
    pub executable: PathBuf,
    pub settings: Value,
}

impl Drop for Build {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

impl Build {
    fn new(parent: &Path) -> Result<Self> {
        fs::create_dir_all(parent)?;
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let directory = parent.join(format!("{}-{nonce}-{}", std::process::id(), NONCE.fetch_add(1, Ordering::Relaxed)));
        fs::create_dir(&directory)?;
        Ok(Self { directory: directory.canonicalize()?, executable: PathBuf::new(), settings: Value::Null })
    }

    fn command(&self, checkout: &Path, subcommand: &str) -> Command {
        let mut command = Command::new("cargo");
        command
            .args([subcommand, "--locked", "--release", "--example", "bench", "--target-dir"])
            .arg(&self.directory)
            .arg("-vv")
            .current_dir(checkout)
            // New Cargo versions can separate intermediate and final artifacts.
            // Older Cargo versions simply ignore this environment variable.
            .env("CARGO_BUILD_BUILD_DIR", self.directory.join("intermediate"));
        command
    }

    pub fn prepare(checkout: &Path, parent: &Path) -> Result<Self> {
        let mut build = Self::new(parent)?;
        let version = checked_output(build.command(checkout, "rustc").args(["--", "-vV"]))?;
        let compiler = probe_output(version.stdout)?;
        if !compiler.lines().any(|line| line.starts_with("release: ")) {
            return Err("Cargo's compiler did not report its version".into());
        }
        let cfg = checked_output(build.command(checkout, "rustc").args(["--", "--print", "cfg"]))?;
        let mut target_cfg: Vec<_> = probe_output(cfg.stdout)?.lines().map(str::to_owned).collect();
        target_cfg.sort();
        if !target_cfg.iter().any(|line| line.starts_with("target_arch=")) {
            return Err("Cargo's compiler did not report the benchmark target".into());
        }
        let output = checked_output(build.command(checkout, "build").arg("--message-format=json"))?;
        let mut profiles = BTreeMap::new();
        for line in String::from_utf8(output.stdout)?.lines() {
            let Ok(value) = serde_json::from_str::<Value>(line) else { continue };
            if value["reason"] != "compiler-artifact" {
                continue;
            }
            if let Some(name @ ("dataorder" | "bench")) = value["target"]["name"].as_str() {
                profiles.insert(name.to_owned(), value["profile"].clone());
                if name == "bench" {
                    build.executable = value["executable"].as_str().map(PathBuf::from).ok_or("Cargo omitted the benchmark executable")?;
                }
            }
        }
        if !build.executable.canonicalize()?.starts_with(&build.directory) {
            return Err("Cargo's benchmark artifact is outside the isolated build directory".into());
        }
        let mut codegen = BTreeMap::new();
        for stderr in [version.stderr, cfg.stderr, output.stderr] {
            capture_settings(&String::from_utf8(stderr)?, &mut codegen)?;
        }
        if !["dataorder", "bench"].iter().all(|name| codegen.contains_key(*name) && profiles.contains_key(*name)) {
            return Err("could not verify compiler arguments for both dataorder and bench; benchmark not run".into());
        }
        build.settings = json!({"compiler":compiler, "target_cfg":target_cfg, "profiles":profiles, "codegen":codegen});
        Ok(build)
    }
}

// Verbose Cargo forwards build-script stdout as `[package version] ...` on the
// same stream as rustc's probe. Dependency execution order is nondeterministic;
// none of those lines describes the compiler version or its cfg output.
fn probe_output(bytes: Vec<u8>) -> Result<String> {
    Ok(String::from_utf8(bytes)?.lines().filter(|line| !line.starts_with('[')).collect::<Vec<_>>().join("\n").trim().to_owned())
}

/// Capture only compilation settings, never the environment prefix Cargo prints
/// in verbose output. Output paths and Cargo's revision-specific metadata are not
/// build-setting differences. Preserve flag order because later flags can win.
fn capture_settings(log: &str, into: &mut BTreeMap<String, Vec<String>>) -> Result<()> {
    for line in log.lines() {
        let Some(command) = line.trim().strip_prefix("Running `").and_then(|s| s.strip_suffix('`')) else { continue };
        let Some((_, arguments)) = command.split_once(" --crate-name ") else { continue };
        let args = words(arguments)?;
        let Some(name @ ("dataorder" | "bench")) = args.first().map(String::as_str) else { continue };
        if args.iter().any(|s| s == "-vV" || s == "--print") {
            continue;
        }
        let mut selected = Vec::new();
        let mut args = args.iter().skip(1);
        while let Some(arg) = args.next() {
            if ["-C", "-Z", "--cfg", "--target", "--edition", "--crate-type"].contains(&arg.as_str()) {
                let value = args.next().ok_or("missing compiler option value")?;
                record_setting(&mut selected, arg, value);
            } else if arg.starts_with("-C") || arg.starts_with("-Z") {
                record_setting(&mut selected, &arg[..2], &arg[2..]);
            } else if arg.starts_with("--edition=")
                || arg.starts_with("--target=")
                || arg.starts_with("--cfg=")
                || ["-O", "-g"].contains(&arg.as_str())
            {
                selected.push(arg.clone());
            }
        }

        into.insert(name.to_owned(), selected);
    }
    Ok(())
}

fn record_setting(into: &mut Vec<String>, flag: &str, value: &str) {
    if flag == "-C" && (value.starts_with("metadata=") || value.starts_with("extra-filename=")) {
        return;
    }
    let value = if flag == "-C" && value.starts_with("incremental=") { "incremental=enabled" } else { value };
    into.push(format!("{flag} {value}"));
}

/// Cargo's quoted Command display on Unix and Windows. Do not interpret shell
/// expansions; these strings are inspected as data and are never executed.
fn words(input: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let mut word = String::new();
    let mut quote = None;
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\'' | '"' if quote.is_none() => quote = Some(c),
            c if quote == Some(c) => quote = None,
            '\\' if quote != Some('\'')
                && chars.peek().is_some_and(|&next| next == '"' || next == '\\' || (quote.is_none() && next.is_whitespace())) =>
            {
                word.push(chars.next().unwrap());
            }
            c if c.is_whitespace() && quote.is_none() => {
                if !word.is_empty() {
                    out.push(std::mem::take(&mut word));
                }
            }
            c => word.push(c),
        }
    }
    if quote.is_some() {
        return Err("unrecognized quoting in Cargo compiler arguments".into());
    }
    if !word.is_empty() {
        out.push(word);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_effective_flags_without_environment_or_artifact_paths() {
        let mut settings = BTreeMap::new();
        let log = r#"Running `PRIVATE_ENV=do-not-save /tool/rustc --crate-name bench --edition=2024 examples/bench.rs -C opt-level=3 -C opt-level=0 -Clto=thin --cfg 'feature="serde"' -C metadata=revision --out-dir /private/build -L dependency=/private/build/deps`
Running `rustc --crate-name bench --edition=2024 examples/bench.rs -C opt-level=3 --print cfg`"#;
        capture_settings(log, &mut settings).unwrap();
        assert_eq!(settings["bench"], ["--edition=2024", "-C opt-level=3", "-C opt-level=0", "-C lto=thin", "--cfg feature=\"serde\""]);
        assert_eq!(
            words(r#"bench --cfg "feature=\"serde\"" "C:\checkout path\file.rs""#).unwrap(),
            ["bench", "--cfg", "feature=\"serde\"", "C:\\checkout path\\file.rs"]
        );
        assert!(words("'unterminated").is_err());
    }

    #[test]
    fn cargo_config_and_concurrent_shared_targets_are_observed() {
        let root = Build::new(&std::env::temp_dir().join("dataorder-build-provenance-tests")).unwrap();
        let shared = root.directory.join("shared");
        let checkouts: Vec<_> = ["before", "after"].into_iter().map(|name| {
            let dir = root.directory.join(name);
            fs::create_dir_all(dir.join(".cargo")).unwrap();
            fs::create_dir(dir.join("src")).unwrap();
            fs::create_dir(dir.join("examples")).unwrap();
            fs::write(dir.join("Cargo.toml"), "[workspace]\n[package]\nname = \"dataorder\"\nversion = \"0.0.0\"\nedition = \"2024\"\n[profile.release]\nopt-level = 0\n").unwrap();
            fs::write(dir.join("Cargo.lock"), "version = 4\n[[package]]\nname = \"dataorder\"\nversion = \"0.0.0\"\n").unwrap();
            fs::write(dir.join(".cargo/config.toml"), format!("[build]\ntarget-dir = {}\nrustflags = [\"--cfg\", \"dataorder_provenance_fixture\"]\n", serde_json::to_string(&shared).unwrap())).unwrap();
            fs::write(dir.join("build.rs"), "fn main() { println!(\"cargo:rustc-cfg=dataorder_build_fixture\"); }").unwrap();
            fs::write(dir.join("src/lib.rs"), format!("pub const LABEL: &str = {name:?};")).unwrap();
            fs::write(dir.join("examples/bench.rs"), "fn main() { println!(\"{}\", dataorder::LABEL); }").unwrap();
            dir
        }).collect();
        let parent = root.directory.join("isolated");
        let (a, b) = std::thread::scope(|scope| {
            let a = scope.spawn(|| Build::prepare(&checkouts[0], &parent).unwrap());
            let b = scope.spawn(|| Build::prepare(&checkouts[1], &parent).unwrap());
            (a.join().unwrap(), b.join().unwrap())
        });
        assert_ne!(a.executable, b.executable);
        for (build, expected) in [(&a, "before"), (&b, "after")] {
            let output = checked_output(&mut Command::new(&build.executable)).unwrap();
            assert_eq!(String::from_utf8(output.stdout).unwrap().trim(), expected);
            let compiler = build.settings["compiler"].as_str().unwrap();
            assert!(compiler.starts_with("rustc ") && compiler.contains("release:"));
            assert!(!compiler.contains("cargo:"));
            assert!(build.settings["profiles"]["dataorder"].is_object());
            assert!(build.settings["codegen"]["bench"].is_array());
            // Environment flags take precedence over build.rustflags in Cargo.
            if std::env::var_os("RUSTFLAGS").is_none() && std::env::var_os("CARGO_ENCODED_RUSTFLAGS").is_none() {
                assert!(build.settings["codegen"].to_string().contains("dataorder_provenance_fixture"));
            }
        }
        assert_eq!(a.settings, b.settings);
        assert!(!shared.exists(), "Cargo must never write the configured shared target");
    }

    #[test]
    fn build_artifacts_are_owned_and_isolated_before_compilation() {
        let parent = std::env::temp_dir().join("dataorder-build-isolation-tests");
        let a = Build::new(&parent).unwrap();
        let b = Build::new(&parent).unwrap();
        assert_ne!(a.directory, b.directory);
        for (build, contents) in [(&a, "before"), (&b, "after")] {
            fs::write(build.directory.join("bench"), contents).unwrap();
        }
        assert_eq!(fs::read_to_string(a.directory.join("bench")).unwrap(), "before");
        let command = a.command(Path::new("."), "build");
        assert!(command.get_args().any(|arg| arg == a.directory.as_os_str()));
        let saved = a.directory.clone();
        drop(a);
        assert!(!saved.exists());
        assert!(b.directory.exists());
    }
}
