//! `FeedHandler`: transport -> codec -> [`ob::DeltaStream`], the
//! event-loop skeleton the engine and benches build on.

use crate::codec::{decode_frame, Frame, Payload};
use crate::transport::Transport;
use ob::{BookEvent, Delta, DeltaStream, Op, Snapshot, Side};

/// Events surfaced to the engine per pump.
#[derive(Clone, Debug, PartialEq)]
pub enum FeedEvent {
    BookReady,
    LevelUpdated,
    TradePrinted { price_ticks: u64, lots: u64, aggressor: Side },
    OurFill { order_id: u64, filled_lots: u64, remaining_lots: u64, price_ticks: u64 },
    GapDetected,
    Heartbeat,
}

/// Feeds a [`Transport`] into a sequence-validated book.
pub struct FeedHandler<T: Transport> {
    transport: T,
    stream: DeltaStream,
    buf: Vec<u8>,
    last_seq: Option<u64>,
    /// Cumulative frame counters (CU metering hooks read these).
    pub frames_decoded: u64,
    pub frames_rejected: u64,
}

impl<T: Transport> FeedHandler<T> {
    pub fn new(transport: T) -> FeedHandler<T> {
        FeedHandler {
            transport,
            stream: DeltaStream::new(),
            buf: vec![0u8; 65536],
            last_seq: None,
            frames_decoded: 0,
            frames_rejected: 0,
        }
    }

    /// Read one datagram, decode, apply to the book; returns surfaced
    /// events (empty when nothing new).
    pub fn pump_once(&mut self) -> Result<Vec<FeedEvent>, String> {
        let n = self
            .transport
            .recv(&mut self.buf)
            .map_err(|e| format!("transport: {e}"))?;
        if n == 0 {
            return Ok(vec![]);
        }
        let frame = match decode_frame(&self.buf[..n]) {
            Ok((f, _)) => f,
            Err(e) => {
                self.frames_rejected += 1;
                return Err(format!("decode: {e}"));
            }
        };
        self.frames_decoded += 1;
        Ok(self.apply_frame(frame))
    }

    /// Apply a decoded frame to the book stream (also the injection
    /// point for simulated venues).
    pub fn apply_frame(&mut self, frame: Frame) -> Vec<FeedEvent> {
        let mut events = Vec::new();
        match frame.payload {
            Payload::Snapshot { bids, asks } => {
                let snap = Snapshot {
                    seq: frame.seq,
                    bids,
                    asks,
                };
                self.stream.apply_snapshot(&snap);
                self.last_seq = Some(frame.seq);
                events.push(FeedEvent::BookReady);
            }
            Payload::NewOrder {
                order_id,
                side,
                price_ticks,
                lots,
            } => {
                let delta = Delta {
                    seq: frame.seq,
                    ops: vec![Op::NewOrder {
                        id: order_id,
                        side,
                        price_ticks,
                        lots,
                        ts_ns: frame.ts_ns,
                    }],
                };
                if let Err(e) = self.stream.apply_delta(&delta) {
                    events.push(FeedEvent::GapDetected);
                    let _ = e;
                } else {
                    self.last_seq = Some(frame.seq);
                    events.push(FeedEvent::LevelUpdated);
                }
            }
            Payload::CancelOrder { order_id } => {
                let delta = Delta {
                    seq: frame.seq,
                    ops: vec![Op::CancelOrder { id: order_id }],
                };
                if self.stream.apply_delta(&delta).is_ok() {
                    self.last_seq = Some(frame.seq);
                    events.push(FeedEvent::LevelUpdated);
                } else {
                    events.push(FeedEvent::GapDetected);
                }
            }
            Payload::LevelDelta {
                side,
                price_ticks,
                delta_lots,
            } => {
                let delta = Delta {
                    seq: frame.seq,
                    ops: vec![Op::LevelDelta {
                        side,
                        price_ticks,
                        delta_lots,
                    }],
                };
                if self.stream.apply_delta(&delta).is_ok() {
                    self.last_seq = Some(frame.seq);
                    events.push(FeedEvent::LevelUpdated);
                } else {
                    events.push(FeedEvent::GapDetected);
                }
            }
            Payload::Trade {
                price_ticks,
                lots,
                aggressor,
            } => {
                self.stream
                    .book_mut()
                    .apply(BookEvent::Print {
                        price_ticks,
                        lots,
                        aggressor,
                        ts_ns: frame.ts_ns,
                    });
                events.push(FeedEvent::TradePrinted {
                    price_ticks,
                    lots,
                    aggressor,
                });
            }
            Payload::Fill {
                order_id,
                filled_lots,
                remaining_lots,
                price_ticks,
            } => {
                events.push(FeedEvent::OurFill {
                    order_id,
                    filled_lots,
                    remaining_lots,
                    price_ticks,
                });
            }
            Payload::Heartbeat => {
                events.push(FeedEvent::Heartbeat);
            }
        }
        events
    }

    pub fn book(&self) -> &ob::OrderBook {
        self.stream.book()
    }

    pub fn book_mut(&mut self) -> &mut ob::OrderBook {
        self.stream.book_mut()
    }

    pub fn last_seq(&self) -> Option<u64> {
        self.last_seq
    }

    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::MsgType;
    use crate::replay::ReplayTransport;

    fn frames() -> Vec<Frame> {
        vec![
            Frame {
                msg_type: MsgType::Snapshot,
                seq: 10,
                ts_ns: 1,
                payload: Payload::Snapshot {
                    bids: vec![(99, 10)],
                    asks: vec![(101, 10)],
                },
            },
            Frame {
                msg_type: MsgType::LevelDelta,
                seq: 11,
                ts_ns: 2,
                payload: Payload::LevelDelta {
                    side: Side::Bid,
                    price_ticks: 98,
                    delta_lots: 5,
                },
            },
            Frame {
                msg_type: MsgType::Trade,
                seq: 12,
                ts_ns: 3,
                payload: Payload::Trade {
                    price_ticks: 100,
                    lots: 2,
                    aggressor: Side::Bid,
                },
            },
            Frame {
                msg_type: MsgType::LevelDelta,
                seq: 14, // GAP!
                ts_ns: 4,
                payload: Payload::LevelDelta {
                    side: Side::Ask,
                    price_ticks: 102,
                    delta_lots: 1,
                },
            },
        ]
    }

    #[test]
    fn handler_pipeline_with_gap() {
        let mut h = FeedHandler::new(ReplayTransport::from_frames(&frames()));
        let ev1 = h.pump_once().unwrap();
        assert!(ev1.contains(&FeedEvent::BookReady));
        assert_eq!(h.book().best_bid(), Some((99, 10)));
        let ev2 = h.pump_once().unwrap();
        assert!(ev2.contains(&FeedEvent::LevelUpdated));
        assert_eq!(h.book_mut().level_lots(Side::Bid, 98), 5);
        let ev3 = h.pump_once().unwrap();
        assert!(matches!(
            ev3[0],
            FeedEvent::TradePrinted { price_ticks: 100, lots: 2, .. }
        ));
        let ev4 = h.pump_once().unwrap();
        assert!(ev4.contains(&FeedEvent::GapDetected));
        assert_eq!(h.frames_decoded, 4);
    }
}
