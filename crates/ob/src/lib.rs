#![forbid(unsafe_code)]
//! # ob — client-side order books for a high-performance market maker
//!
//! Implements the data-structure layer of the engine:
//!
//! * [`glass::Glass`] — the "glass" ordered-set from
//!   *Krapivensky, "glass: ordered set data structure for client-side order
//!   books" (arXiv:2506.13991)*: an uncompressed 32-way trie over integer
//!   price keys with **depth bounded by the key width (13 for 64-bit keys),
//!   independent of the number of levels**, and a cached path exploiting the
//!   sequential locality of market data (jump straight to the lowest common
//!   ancestor via one XOR + `clz`).
//!   The paper measures 6-30x faster than `std::map`-style containers on
//!   market-data-like workloads; we reproduce the design faithfully.
//! * [`l3::Level3`] — individual-order (L3) book with per-level intrusive
//!   FIFO queues giving O(1) cancel and exact queue-position tracking for
//!   execution-probability models (Cont–de Larrard style).
//! * [`book::OrderBook`] — the composite client-side book (L2 glass + L3 +
//!   trade prints + locked/crossed detection) maintained from venue events.
//! * [`delta::DeltaStream`] — snapshot/delta application with per-session
//!   sequence validation and resync-on-gap semantics (the same contract as
//!   the perp-options-clob gateway, invariant G-27).
//!
//! All prices are **integer ticks** (`u64`) and all quantities are **integer
//! lots** (`u64`), mirroring the venue's fixed-point domain model
//! (`tick_size_quote_minor`, `lot_size_base_minor`); no floating point is
//! used anywhere in this crate.

pub mod book;
pub mod delta;
pub mod glass;
pub mod l3;
pub mod types;

pub use book::{BookEvent, BookStats, OrderBook, Trade};
pub use delta::{Delta, DeltaStream, Op, Snapshot};
pub use glass::Glass;
pub use l3::{Fill, L3Order, Level3};
pub use types::{Side, TickIter};
