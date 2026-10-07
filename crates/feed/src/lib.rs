//! # feed — the market-data wire layer
//!
//! * [`codec`] — a compact binary frame format mirroring the venue's
//!   domain model: prices are `u64` ticks, quantities `u64` lots, money
//!   `u128` quote-minor; frames carry a per-session `seq` (the G-27
//!   snapshot/delta contract validated by [`ob::DeltaStream`]).
//! * [`transport`] — `Transport` trait with a std UDP implementation and
//!   an in-process channel transport (tests, engine wiring).
//! * [`replay`] — deterministic replay of recorded frame streams.
//! * [`uring`] — **raw-syscall io_uring** (extern "C" `syscall`/`mmap`,
//!   no external crates): setup, ring mmaps, registered buffers,
//!   SQPOLL probing with graceful fallback, batched
//!   `io_uring_enter(IORING_ENTER_GETEVENTS)` completions — the usage
//!   pattern recommended for high-performance feed handlers by
//!   "io_uring for High-Performance DBMSs" (arXiv:2512.04859).
//!   Kernel 5.10 compatible (no multishot dependency).
//! * [`handler`] — `FeedHandler`: transport -> decode -> book/delta
//!   application, the event-loop skeleton the engine and benches build
//!   on.

#![deny(unsafe_code)]
#![allow(unsafe_code)] // scoped to the uring module only

pub mod codec;
pub mod handler;
pub mod proto;
pub mod uds;
pub mod replay;
pub mod transport;
pub mod uring;

pub use codec::{decode_frame, encode_frame, Frame, MsgType, CODEC_MAGIC, CODEC_VERSION};
pub use handler::FeedHandler;
pub use replay::ReplayTransport;
pub use transport::{ChanTransport, Transport, UdpTransport};
