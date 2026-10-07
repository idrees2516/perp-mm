#![deny(unsafe_code)]
//! # bench — nanosecond latency + compute-unit (CU) metering
//!
//! * [`timer`] — `NsTimer`: `std::time::Instant` plus, on x86_64, raw
//!   `rdtsc` cycle counting (serialized with `lfence` via the
//!   `core::arch` intrinsics where available); percentile statistics.
//! * [`cu`] — **Solana-style compute-unit accounting**: every hot-path
//!   operation class carries a fixed CU cost; each feed event runs under
//!   a [`cu::Budget`] and the engine degrades gracefully (skipping
//!   cold-path estimator updates) when the budget is exceeded — the same
//!   discipline Solana programs live under (per-instruction compute
//!   metering with a per-transaction cap).
//! * The `bench_report` example measures: book op latencies, codec
//!   throughput, estimator tick, quote computation, the full
//!   decode->book->estimate->quote->encode pipeline, and the CU budget
//!   headroom — printing a table and writing CSV.

pub mod cu;
pub mod timer;

pub use cu::{Budget, ComputeMeter, OpClass};
pub use timer::{percentiles, NsTimer, Stats};
