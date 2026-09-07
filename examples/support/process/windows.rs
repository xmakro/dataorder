//! Contain descendants even after their parent exits. Start suspended so the
//! process cannot create children before it belongs to the job.
//! https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
use std::io;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::process::Child;
use windows_sys::Win32::Foundation::{HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JobObjectExtendedLimitInformation, SetInformationJobObject, TerminateJobObject,
};
use windows_sys::Win32::System::Threading::{OpenThread, ResumeThread, THREAD_SUSPEND_RESUME};

pub struct Job(OwnedHandle);

fn owned(handle: HANDLE) -> io::Result<OwnedHandle> {
    if handle.is_null() || handle == INVALID_HANDLE_VALUE {
        Err(io::Error::last_os_error())
    } else {
        // SAFETY: callers pass a newly created, uniquely owned Windows handle.
        Ok(unsafe { OwnedHandle::from_raw_handle(handle) })
    }
}

impl Job {
    pub fn new() -> io::Result<Self> {
        // SAFETY: null parameters request an unnamed job with default security.
        let job = Self(owned(unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) })?);
        // SAFETY: this C structure consists entirely of integer fields.
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        // SAFETY: the handle is live; the buffer and size match the requested class.
        if unsafe {
            SetInformationJobObject(
                job.0.as_raw_handle(),
                JobObjectExtendedLimitInformation,
                std::ptr::from_ref(&limits).cast(),
                size_of_val(&limits) as u32,
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(job)
    }

    pub fn assign_and_resume(&self, child: &Child) -> io::Result<()> {
        // SAFETY: both handles are live; the newly spawned process is suspended.
        if unsafe { AssignProcessToJobObject(self.0.as_raw_handle(), child.as_raw_handle()) } == 0 {
            return Err(io::Error::last_os_error());
        }
        // Stable std does not expose Child's primary thread handle. The suspended
        // process has not run user code, so its initial thread can be found before
        // resuming it. OwnedHandle closes both snapshot and thread on every path.
        // SAFETY: TH32CS_SNAPTHREAD requests a system thread snapshot (PID ignored).
        let snapshot = owned(unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) })?;
        // SAFETY: THREADENTRY32 consists entirely of integers; dwSize is required.
        let mut entry: THREADENTRY32 = unsafe { std::mem::zeroed() };
        entry.dwSize = size_of_val(&entry) as u32;
        // SAFETY: the snapshot is live and entry points to a correctly sized buffer.
        let mut present = unsafe { Thread32First(snapshot.as_raw_handle(), &mut entry) };
        while present != 0 {
            if entry.th32OwnerProcessID == child.id() {
                // SAFETY: open only the suspended child's thread, without inheritance.
                let thread = owned(unsafe { OpenThread(THREAD_SUSPEND_RESUME, 0, entry.th32ThreadID) })?;
                // SAFETY: this thread was suspended by CREATE_SUSPENDED and is in the job.
                if unsafe { ResumeThread(thread.as_raw_handle()) } == u32::MAX {
                    return Err(io::Error::last_os_error());
                }
                return Ok(());
            }
            entry.dwSize = size_of_val(&entry) as u32;
            // SAFETY: same valid snapshot and buffer as Thread32First.
            present = unsafe { Thread32Next(snapshot.as_raw_handle(), &mut entry) };
        }
        Err(io::Error::other("could not find the suspended subprocess thread"))
    }

    pub fn terminate(&self) {
        // SAFETY: the job handle stays live while all its processes are terminated.
        // Closing it also enforces KILL_ON_JOB_CLOSE as a final cleanup guard.
        unsafe { TerminateJobObject(self.0.as_raw_handle(), 1) };
    }
}
