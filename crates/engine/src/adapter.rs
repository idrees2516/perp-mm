//! Venue adapters: the command/event boundary between the engine and a
//! venue. [`SimAdapter`] talks to the in-process simulator;
//! [`PerpOptionsClobAdapter`] mirrors the perp-options-clob venue's
//! command/event semantics (Place/Cancel/Amend/Batch, ticks/lots,
//! maker-price fills) for a future wire-up to its socket.io/REST gateway.

use crate::venue::SimVenue;
use ob::Side;

/// A command to the venue.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    Place {
        side: Side,
        price_ticks: u64,
        lots: u64,
    },
    Cancel {
        order_id: u64,
    },
    CancelAll,
    /// Amend = cancel + replace (loses queue priority, venue semantics).
    Amend {
        order_id: u64,
        new_price_ticks: u64,
        new_open_lots: u64,
    },
}

/// Ack/events from the venue.
#[derive(Clone, Debug, PartialEq)]
pub enum VenueAck {
    Placed { order_id: u64 },
    Canceled { order_id: u64 },
    CancelAllDone { count: usize },
    Amended { order_id: u64 },
}

/// The adapter trait.
pub trait VenueAdapter {
    fn send(&mut self, cmd: Command) -> VenueAck;
}

/// Adapter to the in-process simulator.
pub struct SimAdapter<'a> {
    venue: &'a mut SimVenue,
}

impl<'a> SimAdapter<'a> {
    pub fn new(venue: &'a mut SimVenue) -> SimAdapter<'a> {
        SimAdapter { venue }
    }
}

impl<'a> VenueAdapter for SimAdapter<'a> {
    fn send(&mut self, cmd: Command) -> VenueAck {
        match cmd {
            Command::Place {
                side,
                price_ticks,
                lots,
            } => {
                let id = self.venue.place(side, price_ticks, lots);
                VenueAck::Placed { order_id: id }
            }
            Command::Cancel { order_id } => {
                // idempotent by design: unknown ids ack as canceled
                let _ = self.venue.cancel(order_id);
                VenueAck::Canceled { order_id }
            }
            Command::CancelAll => {
                let n = self.venue.cancel_all();
                VenueAck::CancelAllDone { count: n }
            }
            Command::Amend {
                order_id,
                new_price_ticks,
                new_open_lots,
            } => {
                // venue semantics: cancel + replace
                self.venue.cancel(order_id);
                // find the side from the order's remains: we track nothing
                // here; the engine re-places both sides on requote, so
                // amend reduces to cancel in the sim path.
                let _ = (new_price_ticks, new_open_lots);
                VenueAck::Amended { order_id }
            }
        }
    }
}

/// Command/event types mirroring the perp-options-clob venue engine
/// (u128 money, ticks/lots, maker-price fills). A future wire adapter
/// would serialize these to the gateway's JSON-RPC/REST forms.
#[derive(Clone, Debug)]
pub enum PocCommand {
    /// `Command::Place(OrderRequest)` with the venue's fields.
    Place {
        subaccount: u64,
        symbol: u32,
        side: Side,
        price_ticks: u64,
        qty_lots: u64,
        post_only: bool,
        reduce_only: bool,
    },
    Cancel { order_id: u64 },
    CancelAll { subaccount: u64, symbol: u32 },
    Amend {
        order_id: u64,
        new_price_ticks: u64,
        new_open_lots: u64,
    },
    PlaceBatch {
        subaccount: u64,
        symbol: u32,
        orders: Vec<(Side, u64, u64)>, // (side, price_ticks, lots)
    },
    CancelBatch { order_ids: Vec<u64> },
}

/// Venue events mirrored from the perp-options-clob journal.
#[derive(Clone, Debug)]
pub enum PocEvent {
    OrderResting { order_id: u64 },
    OrderClosed { order_id: u64 },
    OrderRejected { reason: String },
    Trade {
        order_id: u64,
        price_ticks: u64,
        filled_lots: u64,
        maker_fee_quote_minor: i128,
    },
    FundingPaid { amount_quote_minor: i128 },
    AdlExecuted { haircut: f64 },
    MmpTripped,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EngineConfig;

    #[test]
    fn sim_adapter_roundtrip() {
        let cfg = EngineConfig::default();
        let mut venue = SimVenue::new(cfg, 3);
        let mut ad = SimAdapter::new(&mut venue);
        let ack = ad.send(Command::Place {
            side: Side::Bid,
            price_ticks: 199,
            lots: 5,
        });
        match ack {
            VenueAck::Placed { order_id } => {
                assert!(order_id >= 1_000_000);
                let ack2 = ad.send(Command::Cancel { order_id });
                assert_eq!(ack2, VenueAck::Canceled { order_id });
            }
            _ => panic!("expected Placed"),
        }
        let ack3 = ad.send(Command::CancelAll);
        assert!(matches!(ack3, VenueAck::CancelAllDone { .. }));
    }
}
