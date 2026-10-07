//! Raw Linux syscalls for the terminal frontend (zero dependencies):
//! `ioctl(TCGETS/TCSETS)` for raw mode, `ioctl(TIOCGWINSZ)` for the
//! window size, `poll` for the event loop, `read`/`write` for I/O.
//! x86-64 ABI; the same `extern "C" syscall` pattern the feed crate's
//! io_uring layer uses.

#![allow(unsafe_code)]

const SYS_READ: i64 = 0;
const SYS_WRITE: i64 = 1;
const SYS_IOCTL: i64 = 16;
const SYS_POLL: i64 = 7;

const TCGETS: u64 = 0x5401;
const TCSETS: u64 = 0x5402;
const TIOCGWINSZ: u64 = 0x5413;

// termios bit masks (Linux)
const IGNBRK: u32 = 0o1;
const BRKINT: u32 = 0o2;
const PARMRK: u32 = 0o10;
const ISTRIP: u32 = 0o40;
const INLCR: u32 = 0o100;
const IGNCR: u32 = 0o200;
const ICRNL: u32 = 0o400;
const IXON: u32 = 0o2000;
const OPOST: u32 = 0o1;
const ECHO: u32 = 0o10;
const ECHONL: u32 = 0o400;
const ICANON: u32 = 0o2;
const ISIG: u32 = 0o1;
const IEXTEN: u32 = 0o100000;
const CSIZE: u32 = 0o60;
const PARENB: u32 = 0o400;
const CS8: u32 = 0o60;
const VMIN: usize = 6;
const VTIME: usize = 5;

#[repr(C)]
#[derive(Clone, Copy)]
struct Termios {
    c_iflag: u32,
    c_oflag: u32,
    c_cflag: u32,
    c_lflag: u32,
    c_line: u8,
    c_cc: [u8; 32],
    c_ispeed: u32,
    c_ospeed: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct WinSize {
    pub rows: u16,
    pub cols: u16,
    pub xpixel: u16,
    pub ypixel: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PollFd {
    fd: i32,
    events: i16,
    revents: i16,
}

pub const POLLIN: i16 = 0x001;

extern "C" {
    fn syscall(num: i64, ...) -> i64;
}

fn ioctl(fd: i32, req: u64, arg: *mut u8) -> i64 {
    unsafe { syscall(SYS_IOCTL, fd as i64, req, arg) }
}

/// Read up to `buf.len()` bytes; returns bytes read (0 = EOF/eagain).
pub fn read_fd(fd: i32, buf: &mut [u8]) -> i64 {
    unsafe { syscall(SYS_READ, fd as i64, buf.as_mut_ptr(), buf.len()) }
}

/// Write the whole buffer; returns bytes written or negative errno.
pub fn write_fd(fd: i32, buf: &[u8]) -> i64 {
    unsafe { syscall(SYS_WRITE, fd as i64, buf.as_ptr(), buf.len()) }
}

/// Poll fds with a timeout in milliseconds; returns the ready count or
/// negative errno.
pub fn poll_fds(fds: &mut [(i32, i16)], timeout_ms: i32) -> i32 {
    let mut pfds: Vec<PollFd> = fds
        .iter()
        .map(|&(fd, ev)| PollFd { fd, events: ev, revents: 0 })
        .collect();
    let rc = unsafe {
        syscall(SYS_POLL, pfds.as_mut_ptr(), pfds.len() as i64, timeout_ms)
    };
    for (i, p) in pfds.iter().enumerate() {
        fds[i].1 = p.revents;
    }
    rc as i32
}

/// Terminal window size (defaults to 80x24 on failure).
pub fn window_size(fd: i32) -> WinSize {
    let mut ws = WinSize::default();
    let rc = ioctl(fd, TIOCGWINSZ, &mut ws as *mut WinSize as *mut u8);
    if rc != 0 || ws.cols == 0 {
        ws.cols = 80;
        ws.rows = 24;
    }
    ws
}

/// Raw terminal mode guard: enters raw mode on construction and restores
/// the original attributes on drop.
pub struct RawMode {
    fd: i32,
    saved: Termios,
    pub active: bool,
}

impl RawMode {
    pub fn enter(fd: i32) -> Option<RawMode> {
        let mut t = Termios {
            c_iflag: 0,
            c_oflag: 0,
            c_cflag: 0,
            c_lflag: 0,
            c_line: 0,
            c_cc: [0; 32],
            c_ispeed: 0,
            c_ospeed: 0,
        };
        let rc = ioctl(fd, TCGETS, &mut t as *mut Termios as *mut u8);
        if rc != 0 {
            return None;
        }
        let saved = t;
        t.c_iflag &=
            !(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON);
        t.c_oflag &= !OPOST;
        t.c_lflag &= !(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
        t.c_cflag &= !(CSIZE | PARENB);
        t.c_cflag |= CS8;
        t.c_cc[VMIN] = 1;
        t.c_cc[VTIME] = 0;
        let rc = ioctl(fd, TCSETS, &t as *const Termios as *mut u8);
        if rc != 0 {
            return None;
        }
        Some(RawMode { fd, saved, active: true })
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        if self.active {
            unsafe {
                syscall(
                    SYS_IOCTL,
                    self.fd as i64,
                    TCSETS,
                    &self.saved as *const Termios as *mut u8,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poll_times_out_on_idle_fd() {
        use std::os::unix::io::AsRawFd;
        let (a, _b) = std::os::unix::net::UnixStream::pair().unwrap();
        let fd = a.as_raw_fd();
        let mut fds = [(fd, POLLIN)];
        let rc = poll_fds(&mut fds, 5);
        assert!(rc >= 0);
        assert_eq!(fds[0].1 & POLLIN, 0);
    }

    #[test]
    fn poll_wakes_on_ready_socket() {
        use std::os::unix::net::UnixStream;
        let (a, mut b) = UnixStream::pair().unwrap();
        use std::io::Write;
        b.write_all(b"x").unwrap();
        use std::os::unix::io::AsRawFd;
        let mut fds = [(a.as_raw_fd(), POLLIN)];
        let rc = poll_fds(&mut fds, 50);
        assert!(rc >= 1);
        assert_ne!(fds[0].1 & POLLIN, 0);
    }

    #[test]
    fn window_size_falls_back() {
        let ws = window_size(-1);
        assert_eq!(ws.cols, 80);
        assert_eq!(ws.rows, 24);
    }
}
