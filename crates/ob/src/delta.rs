//! Snapshot / delta application with per-session sequence validation.
//!
//! Mirrors the perp-options-clob gateway contract (invariant G-27): the
//! client subscribes, receives a bootstrap **snapshot** at sequence `s`,
//! then **deltas** whose sequence must equal `last + 1`. A gap forces a
//! resync (re-request the snapshot); [`DeltaStream`] encodes exactly this
//! state machine so the feed layer and the engine share one implementation.

use crate::book::{BookEvent, OrderBook};
use crate::types::Side;

/// A single delta operation.
#[derive(Clone, Debug)]
pub enum Op {
    LevelDelta {
        side: Side,
        price_ticks: u64,
        delta_lots: i64,
    },
    SetLevel {
        side: Side,
        price_ticks: u64,
        lots: u64,
    },
    NewOrder {
        id: u64,
        side: Side,
        price_ticks: u64,
        lots: u64,
        ts_ns: i64,
    },
    CancelOrder { id: u64 },
    Print {
        price_ticks: u64,
        lots: u64,
        aggressor: Side,
        ts_ns: i64,
    },
}

/// Full-book snapshot at sequence `seq`.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub seq: u64,
    pub bids: Vec<(u64, u64)>,
    pub asks: Vec<(u64, u64)>,
}

/// Incremental delta at sequence `seq` (must be `last + 1`).
#[derive(Clone, Debug, Default)]
pub struct Delta {
    pub seq: u64,
    pub ops: Vec<Op>,
}

/// Errors from sequence validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeltaError {
    /// Delta arrived before any snapshot.
    NoSnapshot,
    /// Sequence gap: caller must resync via [`DeltaStream::apply_snapshot`].
    Gap { expected: u64, got: u64 },
    /// Non-monotonic / duplicate sequence.
    Stale { expected: u64, got: u64 },
}

/// Sequence-validated book stream.
pub struct DeltaStream {
    book: OrderBook,
    last_seq: Option<u64>,
    /// Set when a gap was reported; cleared by the next snapshot.
    pub needs_resync: bool,
}

impl DeltaStream {
    pub fn new() -> DeltaStream {
        DeltaStream {
            book: OrderBook::new(),
            last_seq: None,
            needs_resync: false,
        }
    }

    /// Install a snapshot (also the resync path after a gap).
    pub fn apply_snapshot(&mut self, snap: &Snapshot) {
        self.book.apply(BookEvent::Reset);
        for &(price, lots) in &snap.bids {
            self.book.apply(BookEvent::SetLevel {
                side: Side::Bid,
                price_ticks: price,
                lots,
            });
        }
        for &(price, lots) in &snap.asks {
            self.book.apply(BookEvent::SetLevel {
                side: Side::Ask,
                price_ticks: price,
                lots,
            });
        }
        self.last_seq = Some(snap.seq);
        self.needs_resync = false;
    }

    /// Apply a delta; validates sequencing.
    pub fn apply_delta(&mut self, delta: &Delta) -> Result<(), DeltaError> {
        let expected = match self.last_seq {
            None => return Err(DeltaError::NoSnapshot),
            Some(s) => s + 1,
        };
        if delta.seq < expected {
            return Err(DeltaError::Stale {
                expected,
                got: delta.seq,
            });
        }
        if delta.seq > expected {
            self.needs_resync = true;
            return Err(DeltaError::Gap {
                expected,
                got: delta.seq,
            });
        }
        for op in &delta.ops {
            let ev = match op {
                Op::LevelDelta {
                    side,
                    price_ticks,
                    delta_lots,
                } => BookEvent::LevelDelta {
                    side: *side,
                    price_ticks: *price_ticks,
                    delta_lots: *delta_lots,
                },
                Op::SetLevel {
                    side,
                    price_ticks,
                    lots,
                } => BookEvent::SetLevel {
                    side: *side,
                    price_ticks: *price_ticks,
                    lots: *lots,
                },
                Op::NewOrder {
                    id,
                    side,
                    price_ticks,
                    lots,
                    ts_ns,
                } => BookEvent::NewOrder {
                    id: *id,
                    side: *side,
                    price_ticks: *price_ticks,
                    lots: *lots,
                    ts_ns: *ts_ns,
                },
                Op::CancelOrder { id } => BookEvent::CancelOrder { id: *id },
                Op::Print {
                    price_ticks,
                    lots,
                    aggressor,
                    ts_ns,
                } => BookEvent::Print {
                    price_ticks: *price_ticks,
                    lots: *lots,
                    aggressor: *aggressor,
                    ts_ns: *ts_ns,
                },
            };
            self.book.apply(ev);
        }
        self.last_seq = Some(delta.seq);
        Ok(())
    }

    #[inline]
    pub fn book(&self) -> &OrderBook {
        &self.book
    }

    #[inline]
    pub fn book_mut(&mut self) -> &mut OrderBook {
        &mut self.book
    }

    #[inline]
    pub fn last_seq(&self) -> Option<u64> {
        self.last_seq
    }
}

impl Default for DeltaStream {
    fn default() -> Self {
        DeltaStream::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_then_deltas_with_gap() {
        let mut s = DeltaStream::new();
        let d = Delta {
            seq: 1,
            ops: vec![Op::LevelDelta {
                side: Side::Bid,
                price_ticks: 100,
                delta_lots: 5,
            }],
        };
        assert_eq!(s.apply_delta(&d), Err(DeltaError::NoSnapshot));

        s.apply_snapshot(&Snapshot {
            seq: 10,
            bids: vec![(99, 5)],
            asks: vec![(101, 5)],
        });
        assert_eq!(s.book().best_bid(), Some((99, 5)));

        let ok = Delta {
            seq: 11,
            ops: vec![Op::LevelDelta {
                side: Side::Bid,
                price_ticks: 98,
                delta_lots: 3,
            }],
        };
        assert!(s.apply_delta(&ok).is_ok());

        let gap = Delta {
            seq: 14,
            ops: vec![Op::LevelDelta {
                side: Side::Bid,
                price_ticks: 97,
                delta_lots: 1,
            }],
        };
        assert_eq!(
            s.apply_delta(&gap),
            Err(DeltaError::Gap { expected: 12, got: 14 })
        );
        assert!(s.needs_resync);

        // resync
        s.apply_snapshot(&Snapshot {
            seq: 14,
            bids: vec![(99, 8)],
            asks: vec![(101, 5)],
        });
        assert!(!s.needs_resync);
        assert_eq!(s.book().best_bid(), Some((99, 8)));
    }
}
