//! The simulated venue: MQH-lite book dynamics + our fills + fees +
//! funding + margin/ADL, emitting feed frames.

use crate::config::{EngineConfig, MidModel};
use feed::codec::{encode_frame, Frame, MsgType, Payload};
use micro::Rng;
use ob::{BookEvent, OrderBook, Fill, Side};

/// Events the venue produces for the market maker (beyond feed frames).
#[derive(Clone, Debug, PartialEq)]
pub enum VenueEvent {
    /// One of our resting orders filled.
    OurFill {
        order_id: u64,
        side: Side,
        price_ticks: u64,
        filled_lots: u64,
        fee_quote: f64,
    },
    /// Funding accrued on our position.
    FundingAccrued { payment_quote: f64 },
}

/// A resting order we placed.
#[derive(Clone, Debug)]
struct OurOrder {
    id: u64,
    side: Side,
    price_ticks: u64,
    lots: u64,
}

/// The simulated CLOB.
pub struct SimVenue {
    pub cfg: EngineConfig,
    pub book: OrderBook,
    /// True mid (float, quote units) — the simulator's ground truth.
    pub mid: f64,
    /// Step counter.
    pub step: usize,
    /// Sigma path for rough mode.
    sigma_path: Vec<f64>,
    rough_pos: usize,
    /// Our resting orders.
    ours: Vec<OurOrder>,
    next_order_id: u64,
    /// Cash (quote), inventory (lots), accumulated PnL components.
    pub cash: f64,
    pub inventory: i64,
    pub fees_paid: f64,
    pub funding_paid: f64,
    /// Number of our fills.
    pub our_fills: u64,
    /// Number of our aggressive (taker) executions.
    pub taker_fills: u64,
    /// Accumulated hedge cost (slippage vs mid + taker fees).
    pub hedge_cost: f64,
    /// Sequence counter for frames.
    seq: u64,
    /// Pending frames to drain.
    frames: Vec<Frame>,
    /// Spread-capture accounting (fills at delta vs mid at fill time).
    pub spread_capture: f64,
    pub adverse_cost: f64,
    /// Equity high-water mark and max drawdown.
    pub equity_peak: f64,
    pub max_drawdown: f64,
}

impl SimVenue {
    pub fn new(cfg: EngineConfig, seed: u64) -> SimVenue {
        let mut rng = Rng::new(seed);
        let mid = cfg.s0;
        // initial book: 5 levels per side at `depth_lots`
        let mut book = OrderBook::new();
        let bid_tick = ((mid - cfg.tick_size) / cfg.tick_size).round() as u64;
        let ask_tick = bid_tick + 1;
        let mut order_id = 1u64;
        for i in 0..5u64 {
            book.apply(BookEvent::NewOrder {
                id: order_id,
                side: Side::Bid,
                price_ticks: bid_tick - i,
                lots: cfg.depth_lots,
                ts_ns: 0,
            });
            order_id += 1;
            book.apply(BookEvent::NewOrder {
                id: order_id,
                side: Side::Ask,
                price_ticks: ask_tick + i,
                lots: cfg.depth_lots,
                ts_ns: 0,
            });
            order_id += 1;
        }
        // rough-vol path (pre-generated)
        let sigma_path = match &cfg.mid {
            MidModel::Rough { h, nu, sigma0, .. } => {
                let model = micro::rough::RfsvModel {
                    h: *h,
                    nu: *nu,
                    alpha: 0.0,
                    m: 2.0 * sigma0.ln(),
                };
                let n = (cfg.horizon / cfg.dt) as usize + 8;
                let mut p = model.simulate(n, cfg.dt, &mut rng);
                let lo = 0.2 * sigma0;
                let hi = 5.0 * sigma0;
                for s in p.iter_mut() {
                    if !s.is_finite() || *s <= 0.0 {
                        *s = *sigma0;
                    }
                    *s = s.clamp(lo, hi);
                }
                p
            }
            MidModel::Gbm { .. } => Vec::new(),
        };
        let mut v = SimVenue {
            cfg,
            book,
            mid,
            step: 0,
            sigma_path,
            rough_pos: 0,
            ours: Vec::new(),
            next_order_id: 1_000_000,
            cash: 0.0,
            inventory: 0,
            fees_paid: 0.0,
            funding_paid: 0.0,
            our_fills: 0,
            taker_fills: 0,
            hedge_cost: 0.0,
            seq: 0,
            frames: Vec::new(),
            spread_capture: 0.0,
            adverse_cost: 0.0,
            equity_peak: 0.0,
            max_drawdown: 0.0,
        };
        v.emit_snapshot();
        v
    }

    /// Current sigma (per sqrt(second)).
    pub fn sigma_now(&self) -> f64 {
        match &self.cfg.mid {
            MidModel::Gbm { sigma } => *sigma,
            MidModel::Rough { sigma0, refresh_every, .. } => {
                let re = (*refresh_every).max(1);
                let idx = (self.rough_pos / re)
                    .min(self.sigma_path.len().saturating_sub(1));
                self.sigma_path.get(idx).copied().unwrap_or(*sigma0)
            }
        }
    }

    fn next_seq(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    fn emit_snapshot(&mut self) {
        let bids: Vec<(u64, u64)> = self.book.ladder(Side::Bid, 10).collect();
        let asks: Vec<(u64, u64)> = self.book.ladder(Side::Ask, 10).collect();
        let seq = self.next_seq();
        let ts = self.step as i64;
        self.frames.push(Frame {
            msg_type: MsgType::Snapshot,
            seq,
            ts_ns: ts,
            payload: Payload::Snapshot { bids, asks },
        });
    }

    fn emit_level(&mut self, side: Side, price_ticks: u64, delta_lots: i64) {
        let seq = self.next_seq();
        let ts = self.step as i64;
        self.frames.push(Frame {
            msg_type: MsgType::LevelDelta,
            seq,
            ts_ns: ts,
            payload: Payload::LevelDelta {
                side,
                price_ticks,
                delta_lots,
            },
        });
    }

    fn emit_trade(&mut self, price_ticks: u64, lots: u64, aggressor: Side) {
        let seq = self.next_seq();
        let ts = self.step as i64;
        self.frames.push(Frame {
            msg_type: MsgType::Trade,
            seq,
            ts_ns: ts,
            payload: Payload::Trade {
                price_ticks,
                lots,
                aggressor,
            },
        });
    }

    fn is_our_id(&self, id: u64) -> bool {
        // our ids live in [1_000_000, 1_500_000); public synthetic ids
        // start at 500_500_000
        (1_000_000..1_500_000).contains(&id)
    }

    fn synthetic_id(&mut self) -> u64 {
        self.next_order_id += 1;
        self.next_order_id + 500_000_000
    }

    /// Place one of our limit orders (rests at the level).
    pub fn place(&mut self, side: Side, price_ticks: u64, lots: u64) -> u64 {
        let id = self.next_order_id;
        self.next_order_id += 1;
        self.ours.push(OurOrder {
            id,
            side,
            price_ticks,
            lots,
        });
        let ts = self.step as i64;
        self.book.apply(BookEvent::NewOrder {
            id,
            side,
            price_ticks,
            lots,
            ts_ns: ts,
        });
        self.emit_level(side, price_ticks, lots as i64);
        id
    }

    /// Cancel our order.
    pub fn cancel(&mut self, id: u64) -> bool {
        let found = self.ours.iter().position(|o| o.id == id);
        if let Some(pos) = found {
            let o = self.ours.remove(pos);
            self.book.apply(BookEvent::CancelOrder { id });
            self.emit_level(o.side, o.price_ticks, -(o.lots as i64));
            true
        } else {
            false
        }
    }

    /// Cancel all our orders (requote path). Returns the count.
    pub fn cancel_all(&mut self) -> usize {
        let ids: Vec<u64> = self.ours.iter().map(|o| o.id).collect();
        let n = ids.len();
        for id in ids {
            self.cancel(id);
        }
        n
    }

    /// Our resting orders (id, side, price_ticks, lots).
    pub fn our_orders(&self) -> Vec<(u64, Side, u64, u64)> {
        self.ours
            .iter()
            .map(|o| (o.id, o.side, o.price_ticks, o.lots))
            .collect()
    }

    /// Aggressive take (market order) with self-match prevention:
    /// our resting orders on the opposite side that this take would
    /// cross are cancelled first (SMP), then the sweep consumes public
    /// liquidity at the touch with taker fees. Returns
    /// `(filled_lots, avg_price_quote, fee_quote)`; `(0, ..)` when the
    /// book is empty on that side.
    pub fn take(&mut self, side: Side, lots: u64) -> (u64, f64, f64) {
        if lots == 0 {
            return (0, 0.0, 0.0);
        }
        // SMP: pull our opposite-side orders that would be crossed.
        let limit = match side {
            Side::Bid => self
                .book
                .best_ask()
                .map(|(p, _)| p.saturating_add(3))
                .unwrap_or(u64::MAX / 4),
            Side::Ask => self
                .book
                .best_bid()
                .map(|(p, _)| p.saturating_sub(3).max(1))
                .unwrap_or(1),
        };
        let opposite = side.opposite();
        let smp_ids: Vec<u64> = self
            .ours
            .iter()
            .filter(|o| {
                o.side == opposite
                    && match side {
                        Side::Bid => o.price_ticks <= limit,
                        Side::Ask => o.price_ticks >= limit,
                    }
            })
            .map(|o| o.id)
            .collect();
        for id in smp_ids {
            self.cancel(id);
        }
        let fills: Vec<Fill> = self.book.apply(BookEvent::Sweep {
            limit_ticks: limit,
            lots,
            aggressor: side,
            ts_ns: self.step as i64,
        });
        let mut filled = 0u64;
        let mut notional = 0.0f64;
        let mut fee = 0.0f64;
        let mut print_price = 0u64;
        let mut by_price: std::collections::HashMap<(Side, u64), i64> =
            std::collections::HashMap::new();
        for f in &fills {
            filled += f.filled_lots;
            let price_quote = f.price_ticks as f64 * self.cfg.tick_size;
            notional += price_quote * f.filled_lots as f64;
            // fee on the per-lot NOTIONAL (price * lot_size), matching
            // the maker-fee accounting in book_our_fill
            fee += price_quote
                * self.cfg.lot_size
                * f.filled_lots as f64
                * self.cfg.fees.taker;
            *by_price
                .entry((f.side, f.price_ticks))
                .or_insert(0) -= f.filled_lots as i64;
            print_price = f.price_ticks;
        }
        for ((f_side, price), delta) in by_price {
            self.emit_level(f_side, price, delta);
        }
        if filled > 0 {
            let lot_q = notional * self.cfg.lot_size;
            match side {
                Side::Bid => {
                    self.cash -= lot_q + fee;
                    self.inventory += filled as i64;
                }
                Side::Ask => {
                    self.cash += lot_q - fee;
                    self.inventory -= filled as i64;
                }
            }
            self.fees_paid += fee;
            self.taker_fills += 1;
            // hedge slippage vs the mid at execution
            let avg = notional / filled as f64;
            self.hedge_cost += (avg - self.mid).abs() * filled as f64 * self.cfg.lot_size + fee;
            self.emit_trade(print_price, filled, side);
        }
        let avg = if filled > 0 {
            notional / filled as f64
        } else {
            0.0
        };
        (filled, avg, fee)
    }

    /// Mark equity: cash + inventory * mid * lot_size.
    pub fn equity(&self) -> f64 {
        self.cash + self.inventory as f64 * self.mid * self.cfg.lot_size
    }

    /// Unified fill accounting for one of our orders (used by both the
    /// Cox-intensity path and public sweeps).
    fn book_our_fill(&mut self, order_id: u64, price_ticks: u64, filled_lots: u64) -> VenueEvent {
        let (side, remaining) = {
            let o = self
                .ours
                .iter_mut()
                .find(|o| o.id == order_id)
                .expect("our order");
            o.lots = o.lots.saturating_sub(filled_lots);
            (o.side, o.lots)
        };
        if remaining == 0 {
            self.ours.retain(|o| o.id != order_id);
        }
        let price_quote = price_ticks as f64 * self.cfg.tick_size;
        let lot_q = price_quote * self.cfg.lot_size;
        let fee = filled_lots as f64 * lot_q.abs() * self.cfg.fees.maker;
        match side {
            Side::Bid => {
                self.cash -= filled_lots as f64 * lot_q + fee;
                self.inventory += filled_lots as i64;
            }
            Side::Ask => {
                self.cash += filled_lots as f64 * lot_q - fee;
                self.inventory -= filled_lots as i64;
            }
        }
        self.fees_paid += fee.abs();
        let signed = side.sign() as f64;
        self.spread_capture += signed * (price_quote - self.mid) * filled_lots as f64 * self.cfg.lot_size;
        self.our_fills += 1;
        // adverse selection: mid jumps against us right after our fill
        let adverse = self.cfg.adverse_ticks * self.cfg.tick_size * signed;
        self.mid -= adverse;
        self.adverse_cost += adverse.abs() * filled_lots as f64 * self.cfg.lot_size;
        VenueEvent::OurFill {
            order_id,
            side,
            price_ticks,
            filled_lots,
            fee_quote: fee,
        }
    }

    /// Advance one dt; returns venue events (our fills, funding).
    pub fn step(&mut self, rng: &mut Rng) -> Vec<VenueEvent> {
        let mut events = Vec::new();
        let dt = self.cfg.dt;
        // 1) mid evolves
        let sigma = self.sigma_now();
        match &self.cfg.mid {
            MidModel::Gbm { .. } => {
                self.mid += sigma * self.mid * rng.normal() * dt.sqrt();
            }
            MidModel::Rough { .. } => {
                self.mid *= (sigma * rng.normal() * dt.sqrt()).exp();
            }
        }
        self.mid = self.mid.max(self.cfg.tick_size);
        self.rough_pos += 1;

        // 2) public market order sweeps the touch (may consume our orders
        // resting there — the unified accounting path handles it).
        if rng.bernoulli(self.cfg.mo_rate * dt) {
            let aggressor = if rng.bernoulli(0.5) { Side::Bid } else { Side::Ask };
            let lots = 1 + rng.below(self.cfg.depth_lots);
            let limit = match aggressor {
                Side::Bid => self.book.best_ask().map(|(p, _)| p).unwrap_or(u64::MAX / 4),
                Side::Ask => self.book.best_bid().map(|(p, _)| p).unwrap_or(1),
            };
            let fills: Vec<Fill> = self.book.apply(BookEvent::Sweep {
                limit_ticks: limit,
                lots,
                aggressor,
                ts_ns: self.step as i64,
            });
            let mut print_price = 0u64;
            let mut print_lots = 0u64;
            for f in &fills {
                if self.is_our_id(f.order_id) && f.order_id != 0 {
                    events.push(self.book_our_fill(f.order_id, f.price_ticks, f.filled_lots));
                }
                print_price = f.price_ticks;
                print_lots += f.filled_lots;
            }
            // reflect the swept liquidity in frames (L2 was updated by the
            // book itself; emit deltas from the fill list)
            let mut by_price: std::collections::HashMap<(Side, u64), i64> =
                std::collections::HashMap::new();
            for f in &fills {
                *by_price
                    .entry((f.side, f.price_ticks))
                    .or_insert(0) -= f.filled_lots as i64;
            }
            for ((side, price), delta) in by_price {
                self.emit_level(side, price, delta);
            }
            if print_lots > 0 {
                self.emit_trade(print_price, print_lots, aggressor);
            }
        }

        // 3) public limit orders join the touch / inside the spread
        for _ in 0..rng.poisson(self.cfg.lo_rate * dt).min(4) {
            let side = if rng.bernoulli(0.5) { Side::Bid } else { Side::Ask };
            let price = self.touch_price(side, rng);
            let lots = 1 + rng.below(self.cfg.depth_lots);
            let id = self.synthetic_id();
            let ts = self.step as i64;
            self.book.apply(BookEvent::NewOrder {
                id,
                side,
                price_ticks: price,
                lots,
                ts_ns: ts,
            });
            self.emit_level(side, price, lots as i64);
        }

        // 4) public cancellations shrink the touch
        if rng.bernoulli(self.cfg.co_rate * dt) {
            let side = if rng.bernoulli(0.5) { Side::Bid } else { Side::Ask };
            if let Some((price, lots)) = self.best_level(side) {
                let remove = 1 + rng.below(lots.min(10));
                self.book.apply(BookEvent::LevelDelta {
                    side,
                    price_ticks: price,
                    delta_lots: -(remove as i64),
                });
                self.emit_level(side, price, -(remove as i64));
            }
        }

        // 5) our fills via Cox intensity lambda(delta)
        let mid_now = self.mid;
        let tick = self.cfg.tick_size;
        let candidates: Vec<(u64, u64)> = self
            .ours
            .iter()
            .map(|o| (o.id, o.price_ticks))
            .collect();
        for (id, price_ticks) in candidates {
            if self.ours.iter().all(|o| o.id != id) {
                continue; // consumed by the sweep above
            }
            let delta = (price_ticks as f64 - mid_now / tick).abs();
            let lam = self.cfg.intensity_a * (-self.cfg.intensity_kappa * delta).exp();
            if rng.bernoulli(1.0 - (-lam * dt).exp()) {
                let o_lots = self.ours.iter().find(|o| o.id == id).unwrap().lots;
                let filled = if rng.bernoulli(0.85) || o_lots <= 1 {
                    o_lots
                } else {
                    1 + rng.below(o_lots - 1)
                };
                let ev = self.book_our_fill(id, price_ticks, filled);
                // consume the L3 order exactly (queue-aware semantics:
                // the Cox intensity picked OUR order)
                self.book.l3_mut().reduce_order(id, filled);
                let side = ev_side(&ev);
                self.emit_level(side, price_ticks, -(filled as i64));
                self.emit_trade(price_ticks, filled, side.opposite());
                events.push(ev);
            }
        }

        // 6) funding accrual (longs pay when the rate is positive)
        let payment = self.cfg.funding_rate
            * (dt / self.cfg.funding_interval)
            * self.inventory as f64
            * self.mid
            * self.cfg.lot_size;
        self.cash -= payment;
        self.funding_paid += payment;
        if payment.abs() > 1e-12 {
            events.push(VenueEvent::FundingAccrued {
                payment_quote: payment,
            });
        }

        // 7) equity / drawdown tracking
        let eq = self.equity();
        if eq > self.equity_peak {
            self.equity_peak = eq;
        } else {
            let dd = self.equity_peak - eq;
            if dd > self.max_drawdown {
                self.max_drawdown = dd;
            }
        }

        // 8) venue invariant: repair any crossed/locked book (drop the
        // offending side's best level) — MMH-style self-correction.
        self.fix_crossed_if_needed();

        self.step += 1;
        events
    }

    /// Repair a crossed/locked book: cancel the smaller side's best level.
    fn fix_crossed_if_needed(&mut self) {
        loop {
            let b = self.book.best_bid().map(|(p, _)| p);
            let a = self.book.best_ask().map(|(p, _)| p);
            match (b, a) {
                (Some(b), Some(a)) if b >= a => {
                    let side = if b > a { Side::Bid } else { Side::Ask };
                    if let Some((price, lots)) = self.best_level(side) {
                        self.book.apply(BookEvent::LevelDelta {
                            side,
                            price_ticks: price,
                            delta_lots: -(lots as i64),
                        });
                        self.emit_level(side, price, -(lots as i64));
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
    }

    fn best_level(&mut self, side: Side) -> Option<(u64, u64)> {
        match side {
            Side::Bid => self.book.best_bid(),
            Side::Ask => self.book.best_ask(),
        }
    }

    /// Public touch price for a new limit order (join best or inside).
    fn touch_price(&self, side: Side, rng: &mut Rng) -> u64 {
        let mid_tick = (self.mid / self.cfg.tick_size).round() as i64;
        match side {
            Side::Bid => {
                let best = self
                    .book
                    .best_bid()
                    .map(|(p, _)| p as i64)
                    .unwrap_or(mid_tick - 1);
                if rng.bernoulli(0.2) {
                    (best + 1).min(mid_tick - 1).max(1) as u64
                } else {
                    best.max(1) as u64
                }
            }
            Side::Ask => {
                let best = self
                    .book
                    .best_ask()
                    .map(|(p, _)| p as i64)
                    .unwrap_or(mid_tick + 1);
                if rng.bernoulli(0.2) {
                    (best - 1).max(mid_tick + 1) as u64
                } else {
                    best as u64
                }
            }
        }
    }

    /// Drain pending feed frames (decoded, for the in-process engine).
    pub fn drain_frames_decoded(&mut self) -> Vec<Frame> {
        std::mem::take(&mut self.frames)
    }

    /// Drain pending feed frames (encoded, for real transports).
    pub fn drain_frames(&mut self) -> Vec<Vec<u8>> {
        let frames = std::mem::take(&mut self.frames);
        let mut out = Vec::with_capacity(frames.len());
        let mut buf = vec![0u8; 8192];
        for f in &frames {
            if let Ok(n) = encode_frame(f, &mut buf) {
                out.push(buf[..n].to_vec());
            }
        }
        out
    }
}

fn ev_side(ev: &VenueEvent) -> Side {
    match ev {
        VenueEvent::OurFill { side, .. } => *side,
        _ => Side::Bid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn venue_runs_and_fills() {
        let cfg = EngineConfig::default();
        let mut rng = Rng::new(7);
        let mut v = SimVenue::new(cfg.clone(), 11);
        // join the touch: delta ~ 1 tick -> healthy fill intensity while
        // the mid stays near the initial level (the test does not requote)
        let mid_tick = (v.mid / cfg.tick_size).round() as u64;
        v.place(Side::Bid, mid_tick - 1, 5);
        v.place(Side::Ask, mid_tick + 1, 5);
        let mut fills = 0u64;
        for _ in 0..2000 {
            let evs = v.step(&mut rng);
            fills += evs
                .iter()
                .filter(|e| matches!(e, VenueEvent::OurFill { .. }))
                .count() as u64;
        }
        assert!(fills >= 1, "expected at least one fill, got {fills}");
        assert_eq!(v.our_fills, fills);
        // book stays healthy
        let (b, a) = (v.book.best_bid(), v.book.best_ask());
        assert!(b.is_some() && a.is_some());
        assert!(b.unwrap().0 < a.unwrap().0);
        // frames were produced
        let frames = v.drain_frames_decoded();
        assert!(!frames.is_empty());
    }

    #[test]
    fn funding_signs() {
        let mut cfg = EngineConfig::default();
        cfg.funding_rate = 0.01; // 1% per interval
        let mut rng = Rng::new(3);
        let mut v = SimVenue::new(cfg, 5);
        v.inventory = 10; // long
        let cash_before = v.cash;
        v.step(&mut rng);
        assert!(v.cash < cash_before, "longs must pay positive funding");
        v.inventory = -10;
        let cash_before = v.cash;
        v.step(&mut rng);
        assert!(v.cash > cash_before, "shorts must receive");
    }

    #[test]
    fn cancel_all_cleans() {
        let cfg = EngineConfig::default();
        let mut v = SimVenue::new(cfg.clone(), 9);
        let mid_tick = (v.mid / cfg.tick_size).round() as u64;
        v.place(Side::Bid, mid_tick - 1, 5);
        v.place(Side::Ask, mid_tick + 1, 5);
        assert_eq!(v.our_orders().len(), 2);
        v.cancel_all();
        assert_eq!(v.our_orders().len(), 0);
    }
}
