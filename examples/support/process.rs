//! Bounded subprocess execution for benchmark builds and runs. Files replace pipes
//! so descendants cannot keep a pipe reader blocked after the main process exits.
use super::{NONCE, Result};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(windows)]
#[path = "process/windows.rs"]
mod windows;

pub fn timeout(value: Option<&str>) -> Result<Duration> {
    let seconds = value.unwrap_or("1800").parse::<u64>().map_err(|_| "BENCH_TIMEOUT_SECS must be a positive integer")?;
    if seconds == 0 {
        return Err("BENCH_TIMEOUT_SECS must be a positive integer".into());
    }
    Ok(Duration::from_secs(seconds))
}

struct Capture {
    directory: PathBuf,
    keep: bool,
}

impl Capture {
    fn new(parent: &Path) -> Result<Self> {
        fs::create_dir_all(parent)?;
        let directory = parent.join(format!(
            "command-{}-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos(),
            NONCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory)?;
        Ok(Self { directory, keep: false })
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        if !self.keep {
            let _ = fs::remove_dir_all(&self.directory);
        }
    }
}

struct Running {
    child: Child,
    finished: bool,
    #[cfg(windows)]
    job: windows::Job,
}

impl Running {
    fn spawn(command: &mut Command) -> Result<Self> {
        #[cfg(windows)]
        let job = {
            use std::os::windows::process::CommandExt;
            use windows_sys::Win32::System::Threading::CREATE_SUSPENDED;
            let job = windows::Job::new()?;
            command.creation_flags(CREATE_SUSPENDED);
            job
        };
        let running = Self {
            child: command.spawn()?,
            finished: false,
            #[cfg(windows)]
            job,
        };
        #[cfg(windows)]
        running.job.assign_and_resume(&running.child)?;
        Ok(running)
    }

    fn terminate(&mut self) {
        // Groups/jobs retain descendants after the original parent has exited.
        // Always fall back to killing/reaping the direct child as well.
        #[cfg(unix)]
        let _ = Command::new("/bin/kill")
            .args(["-KILL", "--", &format!("-{}", self.child.id())])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        #[cfg(windows)]
        self.job.terminate();
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.finished = true;
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        if !self.finished {
            self.terminate();
        }
    }
}

pub fn run(command: &mut Command, limit: Duration, diagnostics: &Path) -> Result<Output> {
    let mut capture = Capture::new(diagnostics)?;
    let stdout = capture.directory.join("stdout.log");
    let stderr = capture.directory.join("stderr.log");
    command.stdout(File::create(&stdout)?).stderr(File::create(&stderr)?).stdin(Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let result = (|| -> Result<Output> {
        let started = Instant::now();
        let mut running = Running::spawn(command)?;
        // Close the parent's capture handles too, so successful log directories
        // can be removed immediately on Windows as well as Unix.
        command.stdout(Stdio::null()).stderr(Stdio::null());
        let status = loop {
            if let Some(status) = running.child.try_wait()? {
                break status;
            }
            if started.elapsed() >= limit {
                running.terminate();
                return Err(format!("timed out after {} seconds", limit.as_secs_f64()).into());
            }
            std::thread::sleep(Duration::from_millis(10).min(limit.saturating_sub(started.elapsed())));
        };
        if !status.success() {
            // Keep cleanup armed: try_wait may have reaped the direct child,
            // while descendants still own files or continue background work.
            return Err(format!("failed ({status})").into());
        }
        running.finished = true;
        Ok(Output { status, stdout: fs::read(&stdout)?, stderr: fs::read(&stderr)? })
    })();
    result.map_err(|error| {
        capture.keep = true;
        // Do not print the command's environment. Preserve complete output on disk,
        // including progress emitted before a timeout, without flooding the terminal.
        format!("{}: {error}; diagnostics: {}", command.get_program().to_string_lossy(), capture.directory.display()).into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::io::Write;

    fn helper(mode: &str, directory: &Path) -> Command {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "process::tests::process_fixture", "--nocapture"])
            .env("DATAORDER_PROCESS_FIXTURE", mode)
            .env("DATAORDER_PROCESS_DIRECTORY", directory);
        command
    }

    #[test]
    fn process_fixture() {
        let Ok(mode) = std::env::var("DATAORDER_PROCESS_FIXTURE") else { return };
        let directory = PathBuf::from(std::env::var_os("DATAORDER_PROCESS_DIRECTORY").unwrap());
        if mode == "output" {
            // Larger than pipe buffers, on both streams.
            std::io::stdout().write_all(&vec![b'x'; 200_000]).unwrap();
            eprintln!("stderr marker");
            return;
        }
        if mode == "failure" {
            eprintln!("failure marker");
            std::process::exit(7);
        }
        if mode == "hang" || mode == "failure-descendant" {
            let mut child = helper("descendant", &directory).spawn().unwrap();
            eprintln!("parent marker");
            if mode == "failure-descendant" {
                let started = Instant::now();
                while !directory.join("ready").exists() {
                    assert!(started.elapsed() < Duration::from_secs(10), "descendant did not start");
                    std::thread::sleep(Duration::from_millis(10));
                }
                std::process::exit(7);
            }
            let _ = child.wait();
        } else {
            let lock =
                OpenOptions::new().read(true).write(true).create(true).truncate(false).open(directory.join("descendant.lock")).unwrap();
            lock.lock().unwrap();
            eprintln!("descendant marker");
            fs::write(directory.join("ready"), "ready").unwrap();
            // Bound even an unfixed regression so a failed test cannot leave an
            // indefinitely running background process.
            let started = Instant::now();
            while started.elapsed() < Duration::from_secs(30) {
                std::thread::sleep(Duration::from_secs(1));
            }
        }
    }

    #[test]
    fn output_and_failures_are_captured_without_pipe_deadlocks() {
        let root = Capture::new(&std::env::temp_dir().join("dataorder-process-tests")).unwrap();
        let diagnostics = root.directory.join("logs");
        let output = run(&mut helper("output", &root.directory), Duration::from_secs(15), &diagnostics).unwrap();
        assert!(output.stdout.windows(200_000).any(|w| w.iter().all(|&b| b == b'x')));
        assert!(String::from_utf8(output.stderr).unwrap().contains("stderr marker"));
        assert_eq!(fs::read_dir(&diagnostics).unwrap().count(), 0);
        let error = run(&mut helper("failure", &root.directory), Duration::from_secs(15), &diagnostics).unwrap_err().to_string();
        assert!(error.contains("failed") && error.contains("diagnostics:"));
        let log = fs::read_dir(&diagnostics).unwrap().next().unwrap().unwrap().path();
        assert!(fs::read_to_string(log.join("stderr.log")).unwrap().contains("failure marker"));
    }

    #[test]
    fn timeout_terminates_descendants_and_keeps_partial_output() {
        let root = Capture::new(&std::env::temp_dir().join("dataorder-process-tests")).unwrap();
        let diagnostics = root.directory.join("logs");
        let error = run(&mut helper("hang", &root.directory), Duration::from_secs(2), &diagnostics).unwrap_err().to_string();
        assert!(error.contains("timed out") && error.contains("diagnostics:"), "{error}");
        assert!(root.directory.join("ready").exists(), "descendant must have started");
        let log = fs::read_dir(&diagnostics).unwrap().next().unwrap().unwrap().path();
        let stderr = fs::read_to_string(log.join("stderr.log")).unwrap();
        assert!(stderr.contains("parent marker") && stderr.contains("descendant marker"));
        assert_descendant_stopped(&root.directory);
    }

    #[test]
    fn failed_parent_terminates_descendants_and_keeps_output() {
        let root = Capture::new(&std::env::temp_dir().join("dataorder-process-tests")).unwrap();
        let diagnostics = root.directory.join("logs");
        let error = run(&mut helper("failure-descendant", &root.directory), Duration::from_secs(15), &diagnostics).unwrap_err().to_string();
        assert!(error.contains("failed") && !error.contains("timed out"), "{error}");
        assert!(root.directory.join("ready").exists());
        let log = fs::read_dir(&diagnostics).unwrap().next().unwrap().unwrap().path();
        let stderr = fs::read_to_string(log.join("stderr.log")).unwrap();
        assert!(stderr.contains("parent marker") && stderr.contains("descendant marker"));
        assert_descendant_stopped(&root.directory);
    }

    fn assert_descendant_stopped(directory: &Path) {
        // Acquiring the descendant's lock proves it exited, without PID-reuse or
        // zombie-process ambiguities. Allow time for the OS to finish termination.
        let lock = OpenOptions::new().read(true).write(true).open(directory.join("descendant.lock")).unwrap();
        let started = Instant::now();
        while lock.try_lock().is_err() {
            assert!(started.elapsed() < Duration::from_secs(2), "descendant survived subprocess cleanup");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn deadlines_are_configurable_and_nonzero() {
        assert_eq!(timeout(None).unwrap(), Duration::from_secs(1800));
        assert_eq!(timeout(Some("12")).unwrap(), Duration::from_secs(12));
        for value in ["0", "-1", "NaN", ""] {
            assert!(timeout(Some(value)).is_err());
        }
    }
}
