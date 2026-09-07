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

const VERIFICATION_METHOD: &str = "rustup-compiler-path-v1";

pub fn invocations_verified(settings: &Value) -> bool {
    settings["verification_method"] == VERIFICATION_METHOD && settings["invocations_verified"] == true
}

// Resolve independently of Cargo's build.rustc / RUSTC setting. Trust the user's
// rustup-selected toolchain, not an arbitrary launcher named rustc. Without this
// independent identity, comparisons require the existing explicit override.
fn trusted_rustc(checkout: &Path) -> Option<PathBuf> {
    let output = checked_output(Command::new("rustup").args(["which", "rustc"]).current_dir(checkout)).ok()?;
    let path = PathBuf::from(String::from_utf8(output.stdout).ok()?.trim());
    path.is_absolute().then(|| path.canonicalize().ok()).flatten()
}

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
        let trusted = trusted_rustc(checkout);
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
        let mut direct = BTreeMap::new();
        for stderr in [version.stderr, cfg.stderr, output.stderr] {
            capture_settings(&String::from_utf8(stderr)?, trusted.as_deref(), &mut codegen, &mut direct)?;
        }
        if !["dataorder", "bench"].iter().all(|name| codegen.contains_key(*name) && profiles.contains_key(*name)) {
            return Err("could not verify compiler arguments for both dataorder and bench; benchmark not run".into());
        }
        // Cargo prints the arguments sent to a wrapper, not those it ultimately
        // sends to rustc. Equal displayed flags cannot verify wrapped builds.
        let invocations_verified = ["dataorder", "bench"].iter().all(|name| direct.get(*name) == Some(&true));
        build.settings = json!({"compiler":compiler, "target_cfg":target_cfg, "profiles":profiles,
            "codegen":codegen, "invocations_verified": invocations_verified, "verification_method": VERIFICATION_METHOD});
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
fn capture_settings(
    log: &str,
    trusted: Option<&Path>,
    into: &mut BTreeMap<String, Vec<String>>,
    direct: &mut BTreeMap<String, bool>,
) -> Result<()> {
    for line in log.lines() {
        let Some(command) = line.trim().strip_prefix("Running `").and_then(|s| s.strip_suffix('`')) else { continue };
        let Some((prefix, arguments)) = command.split_once(" --crate-name ") else { continue };
        let args = words(arguments)?;
        let Some(name @ ("dataorder" | "bench")) = args.first().map(String::as_str) else { continue };
        if args.iter().any(|s| s == "-vV" || s == "--print") {
            continue;
        }
        let verified = direct_rustc(prefix, trusted)?;
        direct.entry(name.to_owned()).and_modify(|old| *old &= verified).or_insert(verified);
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

/// Recognize an unwrapped rustc invocation conservatively. Inspect environment
/// assignments only to skip them; never retain their names or values in provenance.
fn direct_rustc(prefix: &str, trusted: Option<&Path>) -> Result<bool> {
    let words = words(prefix)?;
    let program = words.iter().position(|word| {
        !word.split_once('=').is_some_and(|(key, _)| !key.is_empty() && key.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_'))
    });
    Ok(program.is_some_and(|i| {
        let path = Path::new(&words[i]);
        i + 1 == words.len() && path.is_absolute() && trusted.is_some_and(|trusted| path.canonicalize().is_ok_and(|p| p == trusted))
    }))
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
        let mut direct = BTreeMap::new();
        let log = r#"Running `PRIVATE_ENV=do-not-save /tool/rustc --crate-name bench --edition=2024 examples/bench.rs -C opt-level=3 -C opt-level=0 -Clto=thin --cfg 'feature="serde"' -C metadata=revision --out-dir /private/build -L dependency=/private/build/deps`
Running `rustc --crate-name bench --edition=2024 examples/bench.rs -C opt-level=3 --print cfg`"#;
        capture_settings(log, &mut settings, &mut direct).unwrap();
        assert_eq!(direct["bench"], true);
        assert_eq!(settings["bench"], ["--edition=2024", "-C opt-level=3", "-C opt-level=0", "-C lto=thin", "--cfg feature=\"serde\""]);
        assert_eq!(
            words(r#"bench --cfg "feature=\"serde\"" "C:\checkout path\file.rs""#).unwrap(),
            ["bench", "--cfg", "feature=\"serde\"", "C:\\checkout path\\file.rs"]
        );
        assert!(words("'unterminated").is_err());
    }

    #[test]
    fn wrappers_cannot_be_certified_from_cargos_displayed_flags() {
        for prefix in [
            "PRIVATE_ENV=hidden /wrapper /tool/rustc",
            "/wrapper /workspace-wrapper /tool/rustc",
            r#""C:\wrapper path\cache.exe" "C:\tool\rustc.exe""#,
            "/custom-compiler",
        ] {
            let mut settings = BTreeMap::new();
            let mut direct = BTreeMap::new();
            capture_settings(&format!("Running `{prefix} --crate-name bench -C opt-level=3`"), &mut settings, &mut direct).unwrap();
            assert!(!direct["bench"], "{prefix}");
            assert_eq!(settings["bench"], ["-C opt-level=3"]);
        }
        assert!(direct_rustc(r#"PRIVATE_ENV='hidden value' "C:\tool path\rustc.exe""#).unwrap());
    }

    #[test]
    fn real_wrapper_flag_changes_require_an_override() {
        let root = Build::new(&std::env::temp_dir().join("dataorder-wrapper-tests")).unwrap();
        let dir = &root.directory;
        for subdir in ["src", "examples", ".cargo"] {
            fs::create_dir(dir.join(subdir)).unwrap();
        }
        fs::write(dir.join("Cargo.toml"), "[workspace]\n[package]\nname='dataorder'\nversion='0.0.0'\nedition='2024'\n").unwrap();
        fs::write(dir.join("Cargo.lock"), "version=4\n[[package]]\nname='dataorder'\nversion='0.0.0'\n").unwrap();
        fs::write(dir.join("src/lib.rs"), "pub fn go(x:u64)->u64{(0..x).map(|i|i*i).sum()}").unwrap();
        fs::write(dir.join("examples/bench.rs"), "fn main(){std::hint::black_box(dataorder::go(std::hint::black_box(100)));}").unwrap();
        let wrapper_source = dir.join("wrapper.rs");
        fs::write(
            &wrapper_source,
            r#"
fn main() {
    let mut args = std::env::args_os().skip(1);
    let compiler = args.next().unwrap();
    let mut args: Vec<_> = args.collect();
    if args.iter().any(|a| a == "--crate-name") && !args.iter().any(|a| a == "-vV" || a == "--print") {
        args.push("-C".into());
        args.push(format!("opt-level={}", include_str!("level")).into());
    }
    let status = std::process::Command::new(compiler).args(args).status().unwrap();
    std::process::exit(status.code().unwrap_or(1));
}
"#,
        )
        .unwrap();
        let mut settings = Vec::new();
        for level in ["0", "3"] {
            fs::write(dir.join("level"), level).unwrap();
            let wrapper = dir.join(format!("wrapper{level}{}", std::env::consts::EXE_SUFFIX));
            checked_output(Command::new("rustc").arg(&wrapper_source).arg("-o").arg(&wrapper)).unwrap();
            fs::write(dir.join(".cargo/config.toml"), format!("[build]\nrustc-wrapper={}\n", serde_json::to_string(&wrapper).unwrap()))
                .unwrap();
            let build = Build::prepare(dir, &dir.join("builds")).unwrap();
            assert_eq!(build.settings["invocations_verified"], false);
            settings.push(build.settings.clone());
        }
        assert_eq!(settings[0], settings[1], "Cargo cannot observe the injected flags");
        let metadata = json!({"environment":{"schema":1,"cpu":"fixture","system":"fixture","os":"fixture","arch":"fixture",
            "affinity":null,"build":settings[0]}});
        assert!(crate::validate_environment(&metadata, &metadata, "a", "b", false).is_err());
        assert!(crate::validate_environment(&metadata, &metadata, "a", "b", true).is_ok());
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
