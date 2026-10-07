//! io_uring ABI: constants and struct layouts, verified byte-for-byte
//! against `/usr/include/linux/io_uring.h` (kernel 5.10-compatible subset).

#![allow(unsafe_code)]

/// `io_uring_setup` syscall number (x86_64).
pub const SYS_IO_URING_SETUP: i64 = 425;
/// `io_uring_enter` syscall number (x86_64).
pub const SYS_IO_URING_ENTER: i64 = 426;
/// `io_uring_register` syscall number (x86_64).
pub const SYS_IO_URING_REGISTER: i64 = 427;

/// Submit-queue ring mmap offset.
pub const IORING_OFF_SQ_RING: u64 = 0;
/// Completion-queue ring mmap offset.
pub const IORING_OFF_CQ_RING: u64 = 0x8000000;
/// SQE array mmap offset.
pub const IORING_OFF_SQES: u64 = 0x10000000;

/// Kernel SQ thread polls the SQ (requires privileges or recent kernels).
pub const IORING_SETUP_SQPOLL: u32 = 1 << 1;
/// `io_uring_enter` blocks until `min_complete` CQEs.
pub const IORING_ENTER_GETEVENTS: u32 = 1 << 0;
/// Wake the SQ poll thread.
pub const IORING_ENTER_SQ_WAKEUP: u32 = 1 << 1;

/// SQ ring needs a wakeup (read from the SQ flags word under SQPOLL).
pub const IORING_SQ_NEED_WAKEUP: u32 = 1 << 0;
/// CQ ring overflowed (read from the SQ flags word).
pub const IORING_SQ_CQ_OVERFLOW: u32 = 1 << 1;

/// Single mmap shared by SQ and CQ rings.
pub const IORING_FEAT_SINGLE_MMAP: u32 = 1 << 0;

// opcodes (verified enum order from the UAPI header)
pub const IORING_OP_READ_FIXED: u8 = 4;
pub const IORING_OP_READ: u8 = 22;
pub const IORING_OP_RECV: u8 = 27;

/// `io_uring_register` opcode: register fixed buffers.
pub const IORING_REGISTER_BUFFERS: u32 = 0;

/// Submission queue entry (64 bytes) — UAPI layout.
#[repr(C)]
pub struct IoUringSqe {
    pub opcode: u8,
    pub flags: u8,
    pub ioprio: u16,
    pub fd: i32,
    /// union { off, addr2, ... }
    pub off_or_addr2: u64,
    /// union { addr (buffer pointer), splice_off_in, ... }
    pub addr: u64,
    pub len: u32,
    /// union { rw_flags, ... }
    pub rw_flags: u32,
    pub user_data: u64,
    /// union { buf_index, buf_group }
    pub buf_index: u16,
    pub personality: u16,
    /// union { splice_fd_in, file_index, ... }
    pub splice_fd_in_or_file_index: i32,
    pub __pad2: [u64; 2],
}

impl IoUringSqe {
    pub fn zeroed() -> IoUringSqe {
        IoUringSqe {
            opcode: 0,
            flags: 0,
            ioprio: 0,
            fd: 0,
            off_or_addr2: 0,
            addr: 0,
            len: 0,
            rw_flags: 0,
            user_data: 0,
            buf_index: 0,
            personality: 0,
            splice_fd_in_or_file_index: 0,
            __pad2: [0; 2],
        }
    }
}

/// Completion queue entry (16 bytes) — UAPI layout.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct IoUringCqe {
    pub user_data: u64,
    pub res: i32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct IoSqringOffsets {
    pub head: u32,
    pub tail: u32,
    pub ring_mask: u32,
    pub ring_entries: u32,
    pub flags: u32,
    pub dropped: u32,
    pub array: u32,
    pub resv1: u32,
    pub user_addr: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct IoCqringOffsets {
    pub head: u32,
    pub tail: u32,
    pub ring_mask: u32,
    pub ring_entries: u32,
    pub overflow: u32,
    pub cqes: u32,
    pub flags: u32,
    pub resv1: u32,
    pub user_addr: u64,
}

/// `struct io_uring_params` — UAPI layout.
#[repr(C)]
pub struct IoUringParams {
    pub sq_entries: u32,
    pub cq_entries: u32,
    pub flags: u32,
    pub sq_thread_cpu: u32,
    pub sq_thread_idle: u32,
    pub features: u32,
    pub wq_fd: u32,
    pub resv: [u32; 3],
    pub sq_off: IoSqringOffsets,
    pub cq_off: IoCqringOffsets,
}

impl IoUringParams {
    pub fn zeroed() -> IoUringParams {
        IoUringParams {
            sq_entries: 0,
            cq_entries: 0,
            flags: 0,
            sq_thread_cpu: 0,
            sq_thread_idle: 0,
            features: 0,
            wq_fd: 0,
            resv: [0; 3],
            sq_off: IoSqringOffsets::default(),
            cq_off: IoCqringOffsets::default(),
        }
    }
}

/// `struct iovec` for buffer registration.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Iovec {
    pub base: u64,
    pub len: u64,
}

extern "C" {
    fn syscall(num: i64, ...) -> i64;
    fn mmap(
        addr: *mut u8,
        length: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut u8;
    fn munmap(addr: *mut u8, length: usize) -> i32;
    fn close(fd: i32) -> i32;
}

/// # Safety
/// Raw FFI into the Linux syscall ABI: callers must pass valid
/// pointers/lengths per `io_uring_setup(2)`/`io_uring_enter(2)`/
/// `io_uring_register(2)`/`mmap(2)` semantics.
pub unsafe fn sys_setup(entries: u32, params: *mut IoUringParams) -> i64 {
    syscall(SYS_IO_URING_SETUP, entries, params)
}

/// # Safety
/// Raw FFI into the Linux syscall ABI: callers must pass valid
/// pointers/lengths per `io_uring_setup(2)`/`io_uring_enter(2)`/
/// `io_uring_register(2)`/`mmap(2)` semantics.
pub unsafe fn sys_enter(fd: i32, to_submit: u32, min_complete: u32, flags: u32) -> i64 {
    syscall(
        SYS_IO_URING_ENTER,
        fd,
        to_submit,
        min_complete,
        flags,
        0usize, // no sigset
    )
}

/// # Safety
/// Raw FFI into the Linux syscall ABI: callers must pass valid
/// pointers/lengths per `io_uring_setup(2)`/`io_uring_enter(2)`/
/// `io_uring_register(2)`/`mmap(2)` semantics.
pub unsafe fn sys_register_buffers(fd: i32, iovecs: *const Iovec, nr: u32) -> i64 {
    syscall(SYS_IO_URING_REGISTER, fd, IORING_REGISTER_BUFFERS, iovecs, nr)
}

/// # Safety
/// Raw FFI into the Linux syscall ABI: callers must pass valid
/// pointers/lengths per `io_uring_setup(2)`/`io_uring_enter(2)`/
/// `io_uring_register(2)`/`mmap(2)` semantics.
pub unsafe fn map_ring(
    fd: i32,
    offset: u64,
    length: usize,
) -> Result<*mut u8, std::io::Error> {
    // PROT_READ | PROT_WRITE = 3, MAP_SHARED = 1, MAP_POPULATE = 0x8000
    let prot = 3;
    let flags = 1 | 0x8000;
    let p = mmap(std::ptr::null_mut(), length, prot, flags, fd, offset as i64);
    if p as i64 == -1 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(p)
}

/// # Safety
/// Raw FFI into the Linux syscall ABI: callers must pass valid
/// pointers/lengths per `io_uring_setup(2)`/`io_uring_enter(2)`/
/// `io_uring_register(2)`/`mmap(2)` semantics.
pub unsafe fn unmap(addr: *mut u8, length: usize) {
    let _ = munmap(addr, length);
}

/// # Safety
/// Raw FFI into the Linux syscall ABI: callers must pass valid
/// pointers/lengths per `io_uring_setup(2)`/`io_uring_enter(2)`/
/// `io_uring_register(2)`/`mmap(2)` semantics.
pub unsafe fn close_fd(fd: i32) {
    let _ = close(fd);
}
