//! Composite client-side order book: L2 glass + L3 queues + trade prints.

use crate::glass::Glass;
use crate::l3::{Fill, Level3};
use crate::types::{Side, TickIter};
use std::collections::VecDeque;

/// Cap on retained trade prints (ring buffer).
const TRADE_RING: usize = 4096;

/// A public trade print.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Trade {
    pub price_ticks: u64,
    pub lots: u64,
    /// Side of the *aggressor* (taker).
    pub aggressor: Side,
    pub ts_ns: i64,
}

/// Events that mutate the client-side book.
#[derive(Clone, Debug)]
pub enum BookEvent {
    /// A (possibly our own) order started resting.
    NewOrder {
        id: u64,
        side: Side,
        price_ticks: u64,
        lots: u64,
        ts_ns: i64,
    },
    /// A resting order was canceled / fully filled elsewhere.
    CancelOrder { id: u64 },
    /// L2-only level delta (venue aggregate feed): create/adjust/delete.
    LevelDelta {
        side: Side,
        price_ticks: u64,
        delta_lots: i64,
    },
    /// Absolute L2 level set (snapshot segment).
    SetLevel {
        side: Side,
        price_ticks: u64,
        lots: u64,
    },
    /// Public print: record only (does not touch resting orders).
    Print {
        price_ticks: u64,
        lots: u64,
        aggressor: Side,
        ts_ns: i64,
    },
    /// A taker sweep executing against resting liquidity up to `limit`
    /// (fills execute at the maker's price, matching the venue's
    /// price-time priority semantics). Consumes L3 FIFO queues.
    Sweep {
        limit_ticks: u64,
        lots: u64,
        aggressor: Side,
        ts_ns: i64,
    },
    /// Clear everything (precedes a snapshot).
    Reset,
}

/// Book health / stats.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BookStats {
    pub bid_levels: usize,
    pub ask_levels: usize,
    pub bid_lots: u64,
    pub ask_lots: u64,
    pub l3_orders: usize,
    /// best bid == best ask (locked).
    pub locked: bool,
    /// best bid > best ask (crossed — feed anomaly).
    pub crossed: bool,
}

/// Client-side order book for one instrument, maintained from venue events.
///
/// * L2: two [`Glass`] tries — bids as a max-glass, asks as a min-glass.
/// * L3: [`Level3`] intrusive FIFO queues per price level.
/// * Prints: bounded ring of [`Trade`]s.
pub struct OrderBook {
    bids: Glass,
    asks: Glass,
    l3: Level3,
    trades: VecDeque<Trade>,
    locked: bool,
    crossed: bool,
    last_ts_ns: i64,
    /// Running total resting lots per side (kept exact for stats).
    bid_lots_total: u64,
    ask_lots_total: u64,
}

impl OrderBook {
    pub fn new() -> OrderBook {
        OrderBook {
            bids: Glass::new(),
            asks: Glass::new(),
            l3: Level3::new(),
            trades: VecDeque::with_capacity(TRADE_RING),
            locked: false,
            crossed: false,
            last_ts_ns: 0,
            bid_lots_total: 0,
            ask_lots_total: 0,
        }
    }

    /// Apply an event; returns the fills produced by a [`BookEvent::Sweep`].
    pub fn apply(&mut self, ev: BookEvent) -> Vec<Fill> {
        match ev {
            BookEvent::NewOrder {
                id,
                side,
                price_ticks,
                lots,
                ts_ns,
            } => {
                self.l3.add(id, side, price_ticks, lots, ts_ns);
                let g = self.side_glass_mut(side);
                g.adjust(price_ticks, lots as i64);
                self.add_total(side, lots as i64);
                self.update_health();
                Vec::new()
            }
            BookEvent::CancelOrder { id } => {
                if let Some(o) = self.l3.cancel(id) {
                    let g = self.side_glass_mut(o.side);
                    g.adjust(o.price_ticks, -(o.open_lots as i64));
                    self.add_total(o.side, -(o.open_lots as i64));
                }
                self.update_health();
                Vec::new()
            }
            BookEvent::LevelDelta {
                side,
                price_ticks,
                delta_lots,
            } => {
                let g = self.side_glass_mut(side);
                let before = g.find(price_ticks).unwrap_or(0);
                let after = g.adjust(price_ticks, delta_lots).unwrap_or(0);
                self.add_total(side, after as i64 - before as i64);
                self.update_health();
                Vec::new()
            }
            BookEvent::SetLevel {
                side,
                price_ticks,
                lots,
            } => {
                let g = self.side_glass_mut(side);
                let before = g.find(price_ticks).unwrap_or(0);
                if lots == 0 {
                    g.erase(price_ticks);
                } else {
                    g.insert(price_ticks, lots);
                }
                self.add_total(side, lots as i64 - before as i64);
                self.update_health();
                Vec::new()
            }
            BookEvent::Print {
                price_ticks,
                lots,
                aggressor,
                ts_ns,
            } => {
                self.record_trade(price_ticks, lots, aggressor, ts_ns);
                Vec::new()
            }
            BookEvent::Sweep {
                limit_ticks,
                lots,
                aggressor,
                ts_ns,
            } => {
                let fills = self.sweep(limit_ticks, lots, aggressor);
                let (px, tot) = fills
                    .iter()
                    .fold((0u64, 0u64), |(_p, t), f| (f.price_ticks, t + f.filled_lots));
                // Record at last fill price (typical tape convention when a
                // sweep spans levels; total lots at the worst level).
                if tot > 0 {
                    self.record_trade(px, tot, aggressor, ts_ns);
                }
                fills
            }
            BookEvent::Reset => {
                self.bids.clear();
                self.asks.clear();
                self.l3.clear();
                self.trades.clear();
                self.locked = false;
                self.crossed = false;
                self.bid_lots_total = 0;
                self.ask_lots_total = 0;
                Vec::new()
            }
        }
    }

    /// Best bid (price, lots).
    #[inline]
    pub fn best_bid(&self) -> Option<(u64, u64)> {
        self.bids.max()
    }

    /// Best ask (price, lots).
    #[inline]
    pub fn best_ask(&self) -> Option<(u64, u64)> {
        self.asks.min()
    }

    /// `(best_bid + best_ask)` — integer mid doubled (avoids fp).
    #[inline]
    pub fn mid_sum_ticks(&self) -> Option<u64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((b, _)), Some((a, _))) => Some(b + a),
            _ => None,
        }
    }

    #[inline]
    pub fn spread_ticks(&self) -> Option<u64> {
        match (self.best_ask(), self.best_bid()) {
            (Some((a, _)), Some((b, _))) => Some(a.saturating_sub(b)),
            _ => None,
        }
    }

    /// The best `n` levels of `side`, best price first.
    pub fn ladder(&mut self, side: Side, n: usize) -> TickIter {
        let levels = match side {
            Side::Bid => self.bids.best_descending(n),
            Side::Ask => self.asks.best_ascending(n),
        };
        TickIter { levels, side }
    }

    /// Total lots resting on `side` within `[best..=price_ticks]`.
    pub fn depth_to(&mut self, side: Side, price_ticks: u64) -> u64 {
        match side {
            Side::Bid => match self.bids.max() {
                Some((best, _)) => self.bids.sum_between(price_ticks, best),
                None => 0,
            },
            Side::Ask => match self.asks.min() {
                Some((best, _)) => self.asks.sum_between(best, price_ticks),
                None => 0,
            },
        }
    }

    /// Lots resting at an exact level.
    pub fn level_lots(&mut self, side: Side, price_ticks: u64) -> u64 {
        let g = self.side_glass_mut(side);
        g.find(price_ticks).unwrap_or(0)
    }

    /// Recent prints, oldest first.
    pub fn trades(&self) -> impl Iterator<Item = &Trade> {
        self.trades.iter()
    }

    /// L3 access for queue-position models.
    #[inline]
    pub fn l3(&self) -> &Level3 {
        &self.l3
    }

    #[inline]
    pub fn l3_mut(&mut self) -> &mut Level3 {
        &mut self.l3
    }

    pub fn stats(&self) -> BookStats {
        BookStats {
            bid_levels: self.bids.len(),
            ask_levels: self.asks.len(),
            bid_lots: self.bid_lots_total,
            ask_lots: self.ask_lots_total,
            l3_orders: self.l3.order_count(),
            locked: self.locked,
            crossed: self.crossed,
        }
    }

    #[inline]
    fn add_total(&mut self, side: Side, delta: i64) {
        match side {
            Side::Bid => self.bid_lots_total = (self.bid_lots_total as i64 + delta).max(0) as u64,
            Side::Ask => self.ask_lots_total = (self.ask_lots_total as i64 + delta).max(0) as u64,
        }
    }

    // ------------------------------------------------------------------
    // internals
    // ------------------------------------------------------------------

    fn side_glass_mut(&mut self, side: Side) -> &mut Glass {
        match side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        }
    }

    fn record_trade(&mut self, price_ticks: u64, lots: u64, aggressor: Side, ts_ns: i64) {
        if self.trades.len() == TRADE_RING {
            self.trades.pop_front();
        }
        self.trades.push_back(Trade {
            price_ticks,
            lots,
            aggressor,
            ts_ns,
        });
        self.last_ts_ns = self.last_ts_ns.max(ts_ns);
    }

    /// Execute a taker order against resting liquidity. Bid aggressor
    /// consumes asks (maker prices), Ask aggressor consumes bids.
    fn sweep(&mut self, limit_ticks: u64, lots: u64, aggressor: Side) -> Vec<Fill> {
        let mut out = Vec::new();
        let mut remaining = lots;
        loop {
            if remaining == 0 {
                break;
            }
            let (level_price, level_lots) = match aggressor {
                Side::Bid => match self.asks.min() {
                    Some(x) => x,
                    None => break,
                },
                Side::Ask => match self.bids.max() {
                    Some(x) => x,
                    None => break,
                },
            };
            // Respect the taker's limit.
            let price_ok = match aggressor {
                Side::Bid => level_price <= limit_ticks,
                Side::Ask => level_price >= limit_ticks,
            };
            if !price_ok {
                break;
            }
            let take = level_lots.min(remaining);
            if take == 0 {
                break;
            }
            let before_l3 = self.l3.level_lots(level_price);
            let mut fills = self.l3.execute_fifo(level_price, take);
            let consumed_l3 = before_l3.saturating_sub(self.l3.level_lots(level_price));
            // L2 always drops `take` lots at this level; L3-only feeds may
            // track none of the resting orders (aggregate-only client), in
            // which case we synthesize the fill at the maker price.
            if fills.is_empty() && consumed_l3 == 0 {
                out.push(Fill {
                    order_id: 0,
                    side: aggressor.opposite(),
                    price_ticks: level_price,
                    filled_lots: take,
                    remaining_lots: 0,
                });
            } else {
                out.append(&mut fills);
            }
            match aggressor {
                Side::Bid => {
                    self.asks.adjust(level_price, -(take as i64));
                    self.ask_lots_total = self.ask_lots_total.saturating_sub(take);
                }
                Side::Ask => {
                    self.bids.adjust(level_price, -(take as i64));
                    self.bid_lots_total = self.bid_lots_total.saturating_sub(take);
                }
            }
            remaining -= take;
        }
        self.update_health();
        out
    }

    fn update_health(&mut self) {
        let b = self.bids.max().map(|(k, _)| k);
        let a = self.asks.min().map(|(k, _)| k);
        self.locked = matches!((b, a), (Some(b), Some(a)) if b == a);
        self.crossed = matches!((b, a), (Some(b), Some(a)) if b > a);
    }
}

impl Default for OrderBook {
    fn default() -> Self {
        OrderBook::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(events: Vec<BookEvent>) -> OrderBook {
        let mut b = OrderBook::new();
        for e in events {
            b.apply(e);
        }
        b
    }

    #[test]
    fn book_basics() {
        let b = mk(vec![
            BookEvent::NewOrder {
                id: 1,
                side: Side::Bid,
                price_ticks: 99,
                lots: 5,
                ts_ns: 1,
            },
            BookEvent::NewOrder {
                id: 2,
                side: Side::Bid,
                price_ticks: 99,
                lots: 7,
                ts_ns: 2,
            },
            BookEvent::NewOrder {
                id: 3,
                side: Side::Ask,
                price_ticks: 101,
                lots: 4,
                ts_ns: 3,
            },
        ]);
        assert_eq!(b.best_bid(), Some((99, 12)));
        assert_eq!(b.best_ask(), Some((101, 4)));
        assert_eq!(b.spread_ticks(), Some(2));
        assert_eq!(b.mid_sum_ticks(), Some(200));
        assert_eq!(b.l3().queue_ahead_orders(1), Some(0));
        assert_eq!(b.l3().queue_ahead_orders(2), Some(1));
        assert_eq!(b.l3().queue_ahead_lots(2), Some(5));
    }

    #[test]
    fn sweep_maker_price_fills() {
        let mut b = mk(vec![
            BookEvent::NewOrder {
                id: 10,
                side: Side::Ask,
                price_ticks: 100,
                lots: 3,
                ts_ns: 1,
            },
            BookEvent::NewOrder {
                id: 11,
                side: Side::Ask,
                price_ticks: 101,
                lots: 6,
                ts_ns: 2,
            },
            BookEvent::NewOrder {
                id: 12,
                side: Side::Ask,
                price_ticks: 102,
                lots: 9,
                ts_ns: 3,
            },
        ]);
        // Bid taker sweeps 8 lots with limit 101 -> 3 @ 100 + 5 @ 101.
        let fills = b.apply(BookEvent::Sweep {
            limit_ticks: 101,
            lots: 8,
            aggressor: Side::Bid,
            ts_ns: 4,
        });
        assert_eq!(fills.len(), 2);
        assert_eq!(fills[0].price_ticks, 100);
        assert_eq!(fills[0].filled_lots, 3);
        assert_eq!(fills[1].price_ticks, 101);
        assert_eq!(fills[1].filled_lots, 5);
        assert_eq!(fills[1].remaining_lots, 1);
        assert_eq!(b.best_ask(), Some((101, 1)));
        // print recorded
        assert_eq!(b.trades().count(), 1);
    }

    #[test]
    fn locked_and_crossed_detection() {
        let mut b = mk(vec![
            BookEvent::NewOrder {
                id: 1,
                side: Side::Bid,
                price_ticks: 100,
                lots: 1,
                ts_ns: 1,
            },
            BookEvent::NewOrder {
                id: 2,
                side: Side::Ask,
                price_ticks: 100,
                lots: 1,
                ts_ns: 2,
            },
        ]);
        assert!(b.stats().locked);
        b.apply(BookEvent::LevelDelta {
            side: Side::Bid,
            price_ticks: 101,
            delta_lots: 2,
        });
        assert!(b.stats().crossed);
    }
}
