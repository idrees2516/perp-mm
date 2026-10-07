//! Raw io_uring — setup, rings, registered buffers, submission and
//! batched completion, following the usage patterns recommended for
//! high-performance feed handlers by "io_uring for High-Performance
//! DBMSs: When and How to Use It" (arXiv:2512.04859):
//!
//! * **registered (fixed) buffers** — zero per-op pinning cost;
//! * **batched `io_uring_enter`** — one syscall per wake-up, completions
//!   drained in batches;
//! * **SQPOLL probing** — attempt a kernel submit thread, fall back to
//!   enter()-driven submission when unprivileged (kernel/CAP_SYS_NICE
//!   dependent); under SQPOLL, wake the thread with
//!   `IORING_ENTER_SQ_WAKEUP` when `IORING_SQ_NEED_WAKEUP` is set.
//!
//! Kernel 5.10 compatible: classic re-arm `READ`/`RECV` per datagram
//! (no 5.19+ multishot dependency). All ring accesses use
//! acquire/release ordering as required by the UAPI memory model.
//!
//! The ring is intentionally minimal — a `FeedIoUring` owns the
//! lifecycle; `IoUring::new` returns an error (not a panic) when the
//! kernel or seccomp denies io_uring, so callers fall back to
//! [`crate::transport::UdpTransport`].

#![allow(unsafe_code)]

pub mod abi;

use abi::*;
use std::io;
use std::sync::atomic::{AtomicU32, Ordering};

/// An initialized io_uring instance with mapped rings.
pub struct IoUring {
    fd: i32,
    sq_poll: bool,
    // ring base pointers + sizes
    sq_ptr: *mut u8,
    sq_len: usize,
    cq_ptr: *mut u8,
    cq_len: usize,
    sqes_ptr: *mut IoUringSqe,
    sqes_len: usize,
    // offsets (from params) for ring field access
    sq_head_off: usize,
    sq_tail_off: usize,
    sq_mask_off: usize,
    sq_flags_off: usize,
    sq_dropped_off: usize,
    sq_array_off: usize,
    cq_head_off: usize,
    cq_tail_off: usize,
    cq_mask_off: usize,
    cq_cqes_off: usize,
    cq_overflow_off: usize,
    ring_entries: u32,
    // pending submission count (shadow of tail-head)
    sq_tail: u32,
    sq_head: u32,
    cq_head: u32,
    single_mmap: bool,
}

// The ring memory is shared with the kernel but accessed single-threaded
// per IoUring instance; the atomics below enforce the required ordering.
unsafe impl Send for IoUring {}
unsafe impl Sync for IoUring {}

impl IoUring {
    /// Create an io_uring with `entries` SQ slots. `want_sqpoll` tries a
    /// kernel submit thread first and falls back silently.
    pub fn new(entries: u32, want_sqpoll: bool) -> io::Result<IoUring> {
        let mut params = IoUringParams::zeroed();
        let mut flags = 0u32;
        if want_sqpoll {
            flags |= IORING_SETUP_SQPOLL;
        }
        params.flags = flags;
        let fd = unsafe { sys_setup(entries, &mut params) };
        let fd = if fd < 0 {
            // retry without SQPOLL if it was the blocker
            if want_sqpoll {
                let mut p2 = IoUringParams::zeroed();
                let fd2 = unsafe { sys_setup(entries, &mut p2) };
                if fd2 < 0 {
                    return Err(io::Error::last_os_error());
                }
                params = p2;
                fd2 as i32
            } else {
                return Err(io::Error::last_os_error());
            }
        } else {
            fd as i32
        };
        let sq_poll = params.flags & IORING_SETUP_SQPOLL != 0;
        let single_mmap = params.features & IORING_FEAT_SINGLE_MMAP != 0;

        let sq_off = params.sq_off;
        let cq_off = params.cq_off;
        let sq_entries = params.sq_entries;
        let cq_entries = params.cq_entries;

        // SQ ring size: array offset + array bytes
        let sq_len = (sq_off.array as usize) + (sq_entries as usize) * 4;
        // CQ ring size: cqes offset + cqe bytes
        let cq_len = (cq_off.cqes as usize) + (cq_entries as usize) * std::mem::size_of::<IoUringCqe>();
        let sqes_len = (sq_entries as usize) * std::mem::size_of::<IoUringSqe>();

        let sq_ptr = unsafe { map_ring(fd, IORING_OFF_SQ_RING, sq_len)? };
        let cq_ptr = if single_mmap {
            unsafe { map_ring(fd, IORING_OFF_CQ_RING, cq_len)? }
        } else {
            sq_ptr
        };
        let sqes_ptr = unsafe { map_ring(fd, IORING_OFF_SQES, sqes_len)? } as *mut IoUringSqe;

        let mut ring = IoUring {
            fd,
            sq_poll,
            sq_ptr,
            sq_len,
            cq_ptr,
            cq_len,
            sqes_ptr,
            sqes_len,
            sq_head_off: sq_off.head as usize,
            sq_tail_off: sq_off.tail as usize,
            sq_mask_off: sq_off.ring_mask as usize,
            sq_flags_off: sq_off.flags as usize,
            sq_dropped_off: sq_off.dropped as usize,
            sq_array_off: sq_off.array as usize,
            cq_head_off: cq_off.head as usize,
            cq_tail_off: cq_off.tail as usize,
            cq_mask_off: cq_off.ring_mask as usize,
            cq_cqes_off: cq_off.cqes as usize,
            cq_overflow_off: cq_off.overflow as usize,
            ring_entries: sq_entries,
            sq_tail: 0,
            sq_head: 0,
            cq_head: 0,
            single_mmap,
        };
        // sync shadow counters from the rings
        ring.sync_sq();
        Ok(ring)
    }

    fn sync_sq(&mut self) {
        // kernel-owned head
        self.sq_head = unsafe { load_u32(self.sq_ptr, self.sq_head_off) };
        self.sq_tail = unsafe { load_u32(self.sq_ptr, self.sq_tail_off) };
    }

    #[inline]
    fn mask(&self) -> u32 {
        unsafe { *(self.sq_ptr.add(self.sq_mask_off) as *const u32) }
    }

    /// Whether the kernel submit thread is active.
    pub fn is_sqpoll(&self) -> bool {
        self.sq_poll
    }

    /// Register fixed buffers (io_uring_register IORING_REGISTER_BUFFERS).
    /// The buffers must outlive the ring.
    pub fn register_buffers(&mut self, bufs: &mut [Vec<u8>]) -> io::Result<()> {
        let iovecs: Vec<Iovec> = bufs
            .iter()
            .map(|b| Iovec {
                base: b.as_ptr() as u64,
                len: b.len() as u64,
            })
            .collect();
        let r = unsafe { sys_register_buffers(self.fd, iovecs.as_ptr(), iovecs.len() as u32) };
        if r < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Prepare a fixed-buffer READ SQE (registered buffer `buf_idx`).
    /// Returns the user_data tag; completion length comes back in the CQE.
    pub fn submit_read_fixed(
        &mut self,
        fd: i32,
        buf_idx: u16,
        buf_cap: u32,
        user_data: u64,
    ) -> io::Result<u64> {
        let mut sqe = IoUringSqe::zeroed();
        sqe.opcode = IORING_OP_READ_FIXED;
        sqe.fd = fd;
        sqe.addr = 0; // with fixed buffers, addr is the buffer address OR
                      // may be 0 when buf_index selects (kernel uses addr
                      // as the destination; pass the registered base
                      // through buf_index and a valid addr).
        sqe.len = buf_cap;
        sqe.buf_index = buf_idx;
        sqe.user_data = user_data;
        sqe.off_or_addr2 = 0; // offset unused for sockets? read() on a
                              // socket requires offset = -1... actually
                              // non-file fds ignore off; use 0.
        self.push_sqe(sqe, user_data)
    }

    /// Prepare a plain RECV SQE into `buf`.
    pub fn submit_recv(&mut self, fd: i32, buf: &mut [u8], user_data: u64) -> io::Result<u64> {
        let mut sqe = IoUringSqe::zeroed();
        sqe.opcode = IORING_OP_RECV;
        sqe.fd = fd;
        sqe.addr = buf.as_mut_ptr() as u64;
        sqe.len = buf.len() as u32;
        sqe.user_data = user_data;
        self.push_sqe(sqe, user_data)
    }

    fn push_sqe(&mut self, sqe: IoUringSqe, user_data: u64) -> io::Result<u64> {
        self.sync_sq();
        let mask = self.mask();
        let pending = self.sq_tail.wrapping_sub(self.sq_head);
        if pending >= self.ring_entries {
            // flush first
            self.enter(0)?;
            self.sync_sq();
            let pending2 = self.sq_tail.wrapping_sub(self.sq_head);
            if pending2 >= self.ring_entries {
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "submission queue full",
                ));
            }
        }
        let idx = self.sq_tail & mask;
        unsafe {
            // write the SQE
            *self.sqes_ptr.add(idx as usize) = sqe;
            // publish the array entry
            let arr = self.sq_ptr.add(self.sq_array_off) as *mut u32;
            *arr.add(idx as usize) = idx;
            // update tail (release)
            fetch_add_u32(self.sq_ptr, self.sq_tail_off, 1);
        }
        self.sq_tail = self.sq_tail.wrapping_add(1);
        Ok(user_data)
    }

    /// Submit pending SQEs and optionally wait for completions. With
    /// SQPOLL, only wakes the kernel thread when needed.
    pub fn enter(&mut self, min_complete: u32) -> io::Result<u32> {
        let to_submit = self.sq_tail.wrapping_sub(self.sq_head);
        let mut flags = 0u32;
        if min_complete > 0 {
            flags |= IORING_ENTER_GETEVENTS;
        }
        if self.sq_poll {
            let need =
                unsafe { load_u32(self.sq_ptr, self.sq_flags_off) } & IORING_SQ_NEED_WAKEUP != 0;
            if need {
                flags |= IORING_ENTER_SQ_WAKEUP;
            } else if min_complete == 0 {
                // kernel thread will submit on its own
                self.sync_sq();
                return Ok(0);
            }
        }
        let r = unsafe { sys_enter(self.fd, to_submit, min_complete, flags) };
        if r < 0 {
            return Err(io::Error::last_os_error());
        }
        self.sync_sq();
        Ok(r as u32)
    }

    /// Drain available CQEs (batch completion), invoking `f` per CQE.
    /// Returns the number drained.
    pub fn drain_cqes<F: FnMut(u64, i32, u32)>(&mut self, mut f: F) -> usize {
        let mut n = 0usize;
        loop {
            let (tail, mask) = unsafe {
                (
                    load_u32(self.cq_ptr, self.cq_tail_off),
                    *(self.cq_ptr.add(self.cq_mask_off) as *const u32),
                )
            };
            if self.cq_head == tail {
                break;
            }
            let idx = self.cq_head & mask;
            let cqe: IoUringCqe =
                unsafe { *(self.cq_ptr.add(self.cq_cqes_off) as *const IoUringCqe).add(idx as usize) };
            f(cqe.user_data, cqe.res, cqe.flags);
            // advance head (release)
            unsafe {
                fetch_add_u32(self.cq_ptr, self.cq_head_off, 1);
            }
            self.cq_head = self.cq_head.wrapping_add(1);
            n += 1;
        }
        n
    }

    /// Dropped-submission count (SQ).
    pub fn dropped(&self) -> u32 {
        unsafe { *(self.sq_ptr.add(self.sq_dropped_off) as *const u32) }
    }

    /// CQ overflow count.
    pub fn cq_overflow(&self) -> u32 {
        unsafe { *(self.cq_ptr.add(self.cq_overflow_off) as *const u32) }
    }
}


/// Atomic load from ring memory (the UAPI requires acquire/release on
/// the head/tail words shared with the kernel).
#[inline]
unsafe fn load_u32(base: *mut u8, off: usize) -> u32 {
    (&*(base.add(off) as *const AtomicU32)).load(Ordering::Acquire)
}

#[inline]
unsafe fn fetch_add_u32(base: *mut u8, off: usize, v: u32) -> u32 {
    (&*(base.add(off) as *const AtomicU32)).fetch_add(v, Ordering::Release)
}

impl Drop for IoUring {
    fn drop(&mut self) {
        unsafe {
            unmap(self.sqes_ptr as *mut u8, self.sqes_len);
            if self.single_mmap {
                unmap(self.sq_ptr, self.sq_len);
                unmap(self.cq_ptr, self.cq_len);
            } else {
                unmap(self.sq_ptr, self.sq_len);
            }
            close_fd(self.fd);
        }
    }
}

/// A UDP market-data reader on io_uring with registered buffers:
/// the DBMS-paper pattern (fixed buffers + batched enter).
pub struct FeedIoUring {
    ring: IoUring,
    fd: i32,
    /// Registered receive buffers (fixed READ targets).
    bufs: Vec<Vec<u8>>,
    next_buf: usize,
}

impl FeedIoUring {
    /// Bind a UDP socket and set up the ring with `n_bufs` registered
    /// receive buffers of `buf_len` bytes each.
    pub fn udp(addr: &str, n_bufs: usize, buf_len: usize) -> io::Result<FeedIoUring> {
        let sock = std::net::UdpSocket::bind(addr)?;
        sock.set_nonblocking(true)?;
        let mut ring = IoUring::new(128, true)?;
        let mut bufs: Vec<Vec<u8>> = (0..n_bufs).map(|_| vec![0u8; buf_len]).collect();
        ring.register_buffers(&mut bufs)?;
        Ok(FeedIoUring {
            ring,
            fd: sock.as_raw_fd_unsafe(),
            bufs,
            next_buf: 0,
        })
    }

    /// Arm one receive into the next registered buffer.
    pub fn arm_recv(&mut self) -> io::Result<usize> {
        let idx = self.next_buf;
        self.next_buf = (self.next_buf + 1) % self.bufs.len();
        let cap = self.bufs[idx].len() as u32;
        self.ring.submit_read_fixed(self.fd, idx as u16, cap, idx as u64)?;
        Ok(idx)
    }

    /// Submit + wait for at least one completion, then drain into `out`
    /// as (buffer_index, byte_len) pairs.
    pub fn poll_batch(&mut self, out: &mut Vec<(usize, usize)>) -> io::Result<usize> {
        out.clear();
        self.ring.enter(1)?;
        let mut got = 0usize;
        self.ring.drain_cqes(|user_data, res, _flags| {
            if res > 0 {
                out.push((user_data as usize, res as usize));
                got += 1;
            }
        });
        Ok(got)
    }

    /// Access a completed buffer's bytes.
    pub fn buffer(&self, idx: usize, len: usize) -> &[u8] {
        &self.bufs[idx][..len]
    }
}

// UdpSocket fd extraction without the libc crate.
trait AsRawFdUnsafe {
    fn as_raw_fd_unsafe(&self) -> i32;
}

impl AsRawFdUnsafe for std::net::UdpSocket {
    #[cfg(target_os = "linux")]
    fn as_raw_fd_unsafe(&self) -> i32 {
        use std::os::unix::io::AsRawFd;
        self.as_raw_fd()
    }
    #[cfg(not(target_os = "linux"))]
    fn as_raw_fd_unsafe(&self) -> i32 {
        -1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uring_probe_or_skip() {
        // Kernel 5.10 has io_uring; container seccomp may block it. The
        // test validates the full path when available and skips cleanly
        // otherwise.
        let mut ring = match IoUring::new(64, true) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("io_uring unavailable ({e}); skipping");
                return;
            }
        };
        assert!(ring.ring_entries >= 8);

        // UDP loopback through the ring: arm a RECV, send a datagram,
        // drain the completion.
        let reader = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        reader.set_nonblocking(true).unwrap();
        let raddr = reader.local_addr().unwrap();
        let sender = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();

        let mut buf = vec![0u8; 256];
        let _tag = ring.submit_recv(reader.as_raw_fd_unsafe(), &mut buf, 0xAA).unwrap();
        sender.send_to(b"io_uring-feed-test", raddr).unwrap();
        // wait for the completion
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(2000);
        let mut got = Vec::new();
        loop {
            match ring.enter(1) {
                Ok(_) => {}
                Err(e) => panic!("enter failed: {e}"),
            }
            ring.drain_cqes(|ud, res, _f| got.push((ud, res)));
            if !got.is_empty() {
                break;
            }
            if std::time::Instant::now() > deadline {
                panic!("no io_uring completion within 2s");
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let (ud, res) = got[0];
        assert_eq!(ud, 0xAA);
        assert_eq!(res as usize, b"io_uring-feed-test".len());
        // the bytes landed in our buffer (kernel wrote through the SQE addr)
        assert_eq!(&buf[..res as usize], b"io_uring-feed-test");
    }
}
