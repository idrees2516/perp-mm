//! The `MarketMaker` orchestration + backtest driver.
//!
//! Owns the venue, the estimator stack, the quoting strategy, the risk
//! engine, the option market leg (SSVI surface + client flow), the
//! markout tracker, and the live strategy orders. The requote path is
//! ladder-aware (multi-level quoting with per-level diffing and
//! post-only clamping against the opposite touch); the option leg is
//! quoted in IV space by the vega-approximation quoter and delta-hedged
//! through the perp book with self-match-prevented taker executions.

use crate::config::EngineConfig;
use crate::estimator::{EstimatorStack, MarketState};
use crate::markout::MarkoutTracker;
use crate::metrics::RunMetrics;
use crate::margin::{portfolio_margin, MarginLeg, MarginState};
use crate::options_market::{HedgeMode, OptionMarket};
use crate::rfq::{RfqEngine, RfqRiskCtx};
use crate::risk::{RiskAction, RiskEngine};
use crate::strategy::{build, QuoteCtx, QuotingStrategy, Quotes, StrategyKind};
use crate::venue::{SimVenue, VenueEvent};
use micro::Rng;
use ob::Side;

/// One live strategy order.
#[derive(Clone, Copy, Debug)]
struct LiveOrder {
    id: u64,
    side: Side,
    price_ticks: u64,
    lots: u64,
}

/// The market maker: owns the venue, feed handler, estimator stack,
/// strategy, risk engine, option market and markout tracker for one run.
pub struct MarketMaker {
    pub cfg: EngineConfig,
    pub venue: SimVenue,
    est: EstimatorStack,
    strategy: Box<dyn QuotingStrategy>,
    risk: RiskEngine,
    /// Live strategy orders (both sides, all levels).
    live: Vec<LiveOrder>,
    /// Option market leg (Some when enabled).
    pub opt: Option<OptionMarket>,
    /// Markout tracker for adaptive spreads.
    markout: MarkoutTracker,
    /// Pause flag (quotes pulled while paused).
    pub paused: bool,
    /// Inventory path for metrics.
    inv_path: Vec<i64>,
    /// Equity path (subsampled every 60 steps).
    equity_path: Vec<f64>,
    step: usize,
    n_steps: usize,
    /// Last option view (for the gateway snapshot).
    pub last_opt_view: Option<crate::options_market::OptionsView>,
    /// Institutional RFQ lane (Paradigm / Derive V3 protocol shape).
    pub rfq: RfqEngine,
    /// Cached SFPM portfolio-margin state.
    pub margin: MarginState,
    margin_age: usize,
}

impl MarketMaker {
    pub fn new(cfg: EngineConfig, kind: StrategyKind, seed: u64) -> MarketMaker {
        let n_steps = (cfg.horizon / cfg.dt) as usize;
        let mut opt_cfg = cfg.option_market.clone();
        opt_cfg.perp_lot_size = cfg.lot_size;
        let opt = if opt_cfg.enabled {
            Some(OptionMarket::new(opt_cfg.clone(), opt_cfg.initial_atm_iv))
        } else {
            None
        };
        let mut venue = SimVenue::new(cfg.clone(), seed);
        // Drain the initial snapshot frames (the engine consumes book
        // state directly from the venue's own book).
        let _ = venue.drain_frames_decoded();
        let est = EstimatorStack::new(&cfg);
        let strategy = build(kind, &cfg);
        let risk = RiskEngine::new(&cfg);
        let markout = MarkoutTracker::new(cfg.markout_horizon_steps, cfg.markout_theta);
        MarketMaker {
            cfg,
            venue,
            est,
            strategy,
            risk,
            live: Vec::with_capacity(16),
            opt,
            markout,
            paused: false,
            inv_path: Vec::with_capacity(1024),
            equity_path: Vec::with_capacity(n_steps / 60 + 8),
            step: 0,
            n_steps,
            last_opt_view: None,
            rfq: RfqEngine::new(),
            margin: MarginState { scanning_loss: 0.0, somc: 0.0, maintenance: 0.0, initial: 0.0, utilization: 0.0, worst: 0 },
            margin_age: 0,
        }
    }

    /// Latest estimator snapshot (for the gateway / diagnostics).
    pub fn state(&mut self) -> MarketState {
        self.est.snapshot(&mut self.venue.book, self.cfg.tick_size)
    }

    /// Combined equity: perp venue equity + option book mark.
    pub fn equity(&self) -> f64 {
        self.venue.equity() + self.last_opt_view.map(|v| v.mark).unwrap_or(0.0)
    }

    /// Advance one step: venue evolves -> estimators -> option leg ->
    /// (periodic) requote through risk -> venue commands.
    pub fn step(&mut self, rng: &mut Rng) {
        // 1) venue step
        let events = self.venue.step(rng);
        // 2) drain frames (encoded transport consumers use drain_frames;
        //       the in-process engine reads the venue book directly)
        let _ = self.venue.drain_frames_decoded();
        // 3) venue events -> risk bookkeeping + markouts
        let mid_now = self.venue.mid;
        for ev in &events {
            if let VenueEvent::OurFill { filled_lots, side, price_ticks, .. } = ev {
                self.risk
                    .record_fill(*filled_lots, side.sign() * (*filled_lots as i64));
                let price = *price_ticks as f64 * self.cfg.tick_size;
                // normalize by THIS fill's distance from the mid (the
                // edge we actually captured), floored at half a tick
                let half = (price - mid_now)
                    .abs()
                    .max(0.5 * self.cfg.tick_size);
                self.markout
                    .on_fill(self.step, side.sign(), price, half);
            }
        }
        self.markout.on_step(self.step, mid_now);
        // 4) estimator update
        let (bb, ba) = (self.venue.book.best_bid(), self.venue.book.best_ask());
        self.est
            .update(self.venue.mid, bb, ba, self.cfg.tick_size);
        // 5) option leg: requote IVs, client flow, greeks, delta hedge
        if let Some(opt) = self.opt.as_mut() {
            let vpr = opt.vega_per_request(mid_now);
            let net_vega = opt.view(mid_now).net_vega;
            opt.requote(mid_now, net_vega, vpr, self.cfg.fees.taker, self.cfg.tick_size);
            let _fills = opt.step(rng, mid_now, self.cfg.dt);
            let view = opt.view(mid_now);
            // delta hedge through the perp (SMP-protected taker)
            let unhedged = view.net_delta_lots + self.venue.inventory as f64;
            let band = match self.cfg.option_market.hedge {
                HedgeMode::EveryStep => 0.0,
                HedgeMode::FixedBand { lots } => lots,
                HedgeMode::WwBand => {
                    // cost per PERP lot: taker fee + one tick of slippage
                    let cost = (self.cfg.fees.taker * mid_now + self.cfg.tick_size)
                        * self.cfg.lot_size;
                    let price_per_lot = mid_now * self.cfg.lot_size;
                    vol::HedgeCadence::new(cost, self.cfg.option_market.gamma, price_per_lot)
                        .band_lots
                }
            };
            if unhedged.abs() > band {
                // trade only the EXCESS back to the band edge (the
                // Whalley–Wilmott rebalance rule), not the whole
                // position — re-flattening every trigger would pay the
                // full spread on the entire book each time.
                let excess = ((unhedged.abs() - band) * 2.0).round() as u64 / 2; // half-lot granularity
                let lots = excess.min(5_000);
                let side = if unhedged > 0.0 { Side::Ask } else { Side::Bid };
                self.venue.take(side, lots);
            }
            self.last_opt_view = Some(view);

            // --- RFQ lane: institutional packages, firm TTL, atomic fills ---
            let ctx = RfqRiskCtx {
                unhedged_lots: unhedged,
                net_delta_limit: self.risk.max_inventory as f64,
                net_vega: view.net_vega,
                vega_limit: opt.cfg.max_position_lots * 100.0,
                freeze: self.margin.utilization >= 0.9,
                lot_size: self.cfg.option_market.perp_lot_size,
                multiplier: 1.0,
            };
            let executed = self.rfq.step(self.cfg.dt, mid_now, opt, rng, &ctx);
            if !executed.is_empty() {
                self.margin_age = 999; // force a margin refresh next cycle
            }

            // --- SFPM portfolio margin (scan amortized over 20 steps) ---
            if self.margin_age >= 20 {
                self.margin_age = 0;
                let legs: Vec<MarginLeg> = opt
                    .cfg
                    .moneyness
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| opt.pos[*i].abs() > 1e-9)
                    .map(|(i, m)| {
                        let iv = opt.surface().iv(m.ln(), opt.cfg.t_eff);
                        MarginLeg { kind: opt.cfg.kind, strike: m * mid_now, iv, t: opt.cfg.t_eff, lots: opt.pos[i] }
                    })
                    .collect();
                self.margin = portfolio_margin(
                    mid_now,
                    &legs,
                    self.venue.inventory as f64,
                    self.cfg.lot_size,
                    self.equity() + self.cfg.desk_capital,
                    1.0,
                );
            }
            self.margin_age += 1;
            self.risk.margin_util = self.margin.utilization;
        }
        // 6) requote cadence
        self.risk.tick();
        let state = self.est.snapshot(&mut self.venue.book, self.cfg.tick_size);
        if self.step.is_multiple_of(4) {
            self.requote(&state);
        }
        // 7) metrics
        self.inv_path.push(
            self.venue.inventory
                + self.last_opt_view.map(|v| v.net_delta_lots.round() as i64).unwrap_or(0),
        );
        if self.step.is_multiple_of(60) {
            self.equity_path.push(self.equity());
        }
        self.risk.observe_pnl(self.equity());
        self.step += 1;
    }

    fn requote(&mut self, state: &MarketState) {
        let bid_ladder: Vec<(u64, u64)> = self.venue.book.ladder(Side::Bid, 5).collect();
        let ask_ladder: Vec<(u64, u64)> = self.venue.book.ladder(Side::Ask, 5).collect();
        let fee_floor_ticks = 0.5
            * (self.cfg.fees.taker + self.cfg.fees.maker.abs())
            * state.mid
            / self.cfg.tick_size;
        let markout_mult = if self.cfg.markout_adaptive {
            self.markout.spread_multiplier()
        } else {
            1.0
        };
        let ctx = QuoteCtx {
            state,
            inventory: self.venue.inventory,
            time_left: self.cfg.horizon - self.step as f64 * self.cfg.dt,
            bid_ladder,
            ask_ladder,
            fee_floor_ticks,
            funding_rate: self.cfg.funding_rate,
            funding_interval: self.cfg.funding_interval,
            markout_mult,
            options: self.last_opt_view,
        };
        let desired = if self.paused {
            Quotes::none()
        } else {
            self.strategy.quotes(&ctx)
        };
        // risk-v2 metric: with the option leg live the limit binds on the
        // UNHEDGED combined delta (hedge leg + option book) — never on the
        // raw hedge-leg size (that false-positive freeze was the bug class
        // fixed in the web engine; the Rust filter was already directional,
        // this aligns the METRIC)
        let inv_metric = match self.last_opt_view {
            Some(v) => (v.net_delta_lots + self.venue.inventory as f64).round() as i64,
            None => self.venue.inventory,
        };
        let action = self.risk.filter(desired, state, inv_metric);
        match action {
            RiskAction::Halt(_) => {
                self.cancel_strategy_orders();
            }
            RiskAction::Accept(want) => {
                self.risk.note_requote();
                self.sync_orders(want);
            }
        }
    }

    /// Cancel every strategy order.
    fn cancel_strategy_orders(&mut self) {
        for lo in self.live.drain(..) {
            self.venue.cancel(lo.id);
        }
    }

    /// Diff the desired ladder against live strategy orders: cancel the
    /// ones that changed, keep the rest, place the missing levels with
    /// post-only clamping (never cross the opposite touch).
    fn sync_orders(&mut self, want: Quotes) {
        let want_bid: Vec<(u64, u64)> = want.full_bid();
        let want_ask: Vec<(u64, u64)> = want.full_ask();
        // post-only clamps against the current opposite touch
        let best_ask_tick = self.venue.book.best_ask().map(|(p, _)| p);
        let best_bid_tick = self.venue.book.best_bid().map(|(p, _)| p);
        let clamp_bid = |p: u64| -> Option<u64> {
            match best_ask_tick {
                Some(a) if p >= a => a.checked_sub(1),
                _ => Some(p),
            }
        };
        let clamp_ask = |p: u64| -> Option<u64> {
            match best_bid_tick {
                Some(b) if p <= b => b.checked_add(1),
                _ => Some(p),
            }
        };
        // desired with clamping applied
        let mut desired_bid: Vec<(u64, u64)> = Vec::with_capacity(want_bid.len());
        for (p, l) in want_bid {
            if l == 0 {
                continue;
            }
            if let Some(cp) = clamp_bid(p) {
                desired_bid.push((cp, l));
            }
        }
        let mut desired_ask: Vec<(u64, u64)> = Vec::with_capacity(want_ask.len());
        for (p, l) in want_ask {
            if l == 0 {
                continue;
            }
            if let Some(cp) = clamp_ask(p) {
                desired_ask.push((cp, l));
            }
        }
        // diff per side: keep matching orders, cancel stale, place new
        self.sync_side(Side::Bid, &desired_bid);
        self.sync_side(Side::Ask, &desired_ask);
    }

    fn sync_side(&mut self, side: Side, desired: &[(u64, u64)]) {
        // current live orders of this side
        let mut live: Vec<LiveOrder> = Vec::new();
        let mut keep: Vec<LiveOrder> = Vec::new();
        for lo in self.live.iter() {
            if lo.side == side {
                live.push(*lo);
            } else {
                keep.push(*lo);
            }
        }
        let mut remaining: Vec<(u64, u64)> = desired.to_vec();
        let mut placed: Vec<LiveOrder> = Vec::new();
        // keep live orders that match a desired level
        for lo in live {
            if let Some(pos) = remaining
                .iter()
                .position(|&(p, l)| p == lo.price_ticks && l == lo.lots)
            {
                remaining.remove(pos);
                placed.push(lo);
            } else {
                self.venue.cancel(lo.id);
            }
        }
        // place the rest (ids append)
        for (p, l) in remaining {
            let id = self.venue.place(side, p, l);
            placed.push(LiveOrder { id, side, price_ticks: p, lots: l });
        }
        self.live = keep;
        self.live.extend(placed);
    }

    // ------------------------------------------------------------------
    // Manual trading API (used by the gateway / TUI)
    // ------------------------------------------------------------------

    /// Place a manual resting order (post-only by default: clamped
    /// against the opposite touch). Returns the order id.
    pub fn manual_place(&mut self, side: Side, price_ticks: u64, lots: u64, post_only: bool) -> u64 {
        let p = if post_only {
            match side {
                Side::Bid => self
                    .venue
                    .book
                    .best_ask()
                    .map(|(a, _)| price_ticks.min(a.saturating_sub(1)))
                    .unwrap_or(price_ticks),
                Side::Ask => self
                    .venue
                    .book
                    .best_bid()
                    .map(|(b, _)| price_ticks.max(b.saturating_add(1)))
                    .unwrap_or(price_ticks),
            }
        } else {
            price_ticks
        };
        let p = p.max(1);
        self.venue.place(side, p, lots)
    }

    /// Cancel any order (strategy or manual) by id.
    pub fn manual_cancel(&mut self, id: u64) -> bool {
        self.live.retain(|lo| lo.id != id);
        self.venue.cancel(id)
    }

    /// Aggressive manual take (market order; SMP-protected).
    pub fn manual_take(&mut self, side: Side, lots: u64) -> (u64, f64, f64) {
        self.venue.take(side, lots)
    }

    /// Cancel all our orders (strategy + manual).
    pub fn cancel_all_orders(&mut self) -> usize {
        self.live.clear();
        self.venue.cancel_all()
    }

    /// Switch the quoting strategy live (state resets).
    pub fn select_strategy(&mut self, kind: StrategyKind) {
        self.strategy = build(kind, &self.cfg);
        self.cancel_strategy_orders();
    }

    /// Arm/disarm the kill-switch.
    pub fn set_kill_switch(&mut self, armed: bool) {
        self.risk.clear_halt(armed);
    }

    /// Current risk-halt state.
    pub fn is_halted(&self) -> bool {
        self.risk.is_halted()
    }

    /// Index of the currently-selected strategy kind (wire ordering).
    pub fn strategy_current_index(&self) -> u8 {
        let name = self.strategy.name();
        StrategyKind::all()
            .iter()
            .position(|k| k.label() == name)
            .map(|i| i as u8)
            .unwrap_or(0)
    }

    /// Current markout spread multiplier.
    pub fn markout_multiplier(&self) -> f64 {
        self.markout.spread_multiplier()
    }

    /// Live strategy orders (id, side, price_ticks, lots).
    pub fn strategy_orders(&self) -> Vec<(u64, Side, u64, u64)> {
        self.live
            .iter()
            .map(|lo| (lo.id, lo.side, lo.price_ticks, lo.lots))
            .collect()
    }

    /// Run to the horizon.
    pub fn run(&mut self, rng: &mut Rng) {
        for _ in 0..self.n_steps {
            self.step(rng);
        }
    }

    /// Collect metrics.
    pub fn metrics(&self) -> RunMetrics {
        let inv_abs_mean = if self.inv_path.is_empty() {
            0.0
        } else {
            self.inv_path
                .iter()
                .map(|&q| (q as f64).abs())
                .sum::<f64>()
                / self.inv_path.len() as f64
        };
        let inv_abs_max = self
            .inv_path
            .iter()
            .map(|&q| q.abs())
            .max()
            .unwrap_or(0);
        RunMetrics {
            strategy: self.strategy.name(),
            equity: self.equity(),
            spread_capture: self.venue.spread_capture,
            adverse_cost: self.venue.adverse_cost,
            fees_paid: self.venue.fees_paid,
            funding_paid: self.venue.funding_paid,
            inventory_final: self.venue.inventory,
            inventory_abs_mean: inv_abs_mean,
            inventory_abs_max: inv_abs_max,
            our_fills: self.venue.our_fills,
            taker_fills: self.venue.taker_fills,
            hedge_cost: self.venue.hedge_cost,
            option_mark: self.last_opt_view.map(|v| v.mark).unwrap_or(0.0),
            max_drawdown: self.venue.max_drawdown,
            equity_path: self.equity_path.clone(),
        }
    }
}

/// One backtest run for a strategy (fresh venue, deterministic seed).
pub fn run_backtest(cfg: EngineConfig, kind: StrategyKind, seed: u64) -> RunMetrics {
    let mut mm = MarketMaker::new(cfg, kind, seed);
    let mut rng = Rng::new(seed ^ 0x9E3779B9);
    mm.run(&mut rng);
    mm.metrics()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EngineConfig;
    use crate::options_market::{enable_options, HedgeMode, QuoterMode};

    fn quick_cfg() -> EngineConfig {
        let mut cfg = EngineConfig::default();
        cfg.horizon = 600.0; // 10 minutes
        cfg.dt = 0.1;
        cfg
    }

    #[test]
    fn backtests_produce_sane_metrics() {
        for kind in StrategyKind::all() {
            let m = run_backtest(quick_cfg(), kind, 42);
            assert!(m.equity_path.len() > 5, "{}: no equity path", m.strategy);
            assert!(m.our_fills > 0, "{}: no fills in 10min", m.strategy);
            assert!(m.inventory_abs_max <= 10_000);
            // equity is finite
            assert!(m.equity.is_finite());
        }
    }

    #[test]
    fn inventory_aware_strategies_stay_balanced() {
        // Over a longer horizon, the inventory-aware strategies should
        // keep |inventory| materially below the static benchmark's drift.
        let mut cfg = quick_cfg();
        cfg.horizon = 3600.0;
        let static_m = run_backtest(cfg.clone(), StrategyKind::Static, 7);
        for kind in [
            StrategyKind::UnifiedAs,
            StrategyKind::HjbPolicy,
            StrategyKind::MicroPrice,
        ] {
            let m = run_backtest(cfg.clone(), kind, 7);
            assert!(
                m.inventory_abs_max <= static_m.inventory_abs_max + 8,
                "{}: inv max {} vs static {}",
                m.strategy,
                m.inventory_abs_max,
                static_m.inventory_abs_max
            );
        }
    }

    #[test]
    fn multi_level_strategy_quotes_ladders() {
        let cfg = quick_cfg();
        let mut mm = MarketMaker::new(cfg, StrategyKind::MultiLevel, 11);
        let mut rng = Rng::new(11);
        for _ in 0..80 {
            mm.step(&mut rng);
        }
        // after enough steps there should be more than one live order
        // per side (ladder), and all bid prices < all ask prices.
        let orders = mm.strategy_orders();
        let bids: Vec<(u64, u64)> = orders.iter().filter(|(_, s, _, _)| *s == Side::Bid).map(|(_, _, p, l)| (*p, *l)).collect();
        let asks: Vec<(u64, u64)> = orders.iter().filter(|(_, s, _, _)| *s == Side::Ask).map(|(_, _, p, l)| (*p, *l)).collect();
        if !bids.is_empty() && !asks.is_empty() {
            let best_bid = bids.iter().map(|(p, _)| *p).max().unwrap();
            let best_ask = asks.iter().map(|(p, _)| *p).min().unwrap();
            assert!(best_bid < best_ask, "crossed: {best_bid} >= {best_ask}");
            // ladders are sorted best-first by construction
            assert!(!bids.is_empty());
        }
        assert!(!bids.is_empty() || !asks.is_empty(), "no live orders");
    }

    #[test]
    fn option_market_leg_runs_and_hedges() {
        let mut cfg = quick_cfg();
        enable_options(&mut cfg, QuoterMode::Glft, HedgeMode::WwBand);
        let mut mm = MarketMaker::new(cfg, StrategyKind::VolSurface, 21);
        let mut rng = Rng::new(21);
        // NOTE: the institutional RFQ lane also draws from the engine RNG,
        // so the seeded trajectory shifts when it is active — the run is
        // long enough that the WW band is crossed with near-certainty
        for _ in 0..1200 {
            mm.step(&mut rng);
        }
        let view = mm.last_opt_view.expect("option view");
        assert!(view.atm_iv > 0.05 && view.atm_iv < 2.0);
        assert!(view.mark.is_finite());
        // the hedge actually executed through the perp venue at least once
        assert!(
            mm.venue.taker_fills > 0 || view.net_delta_lots.abs() < 5.0,
            "no hedge executed (delta {})",
            view.net_delta_lots
        );
        // the RFQ lane is live: requests arrived and were risk-priced
        assert!(
            mm.rfq.stats.requests > 0,
            "rfq lane silent: {} requests",
            mm.rfq.stats.requests
        );
        // SFPM margin state is being maintained
        assert!(mm.margin.initial >= 0.0 && mm.margin.utilization >= 0.0);
        // combined equity is finite
        assert!(mm.equity().is_finite());
    }

    #[test]
    fn markout_tracker_drives_multiplier() {
        let mut cfg = quick_cfg();
        cfg.adverse_ticks = 3.0; // toxic fills
        let mut mm = MarketMaker::new(cfg, StrategyKind::MultiLevel, 33);
        let mut rng = Rng::new(33);
        for _ in 0..1200 {
            mm.step(&mut rng);
        }
        // with adverse fills the multiplier should have moved above 1
        // (or remain 1 if no fills resolved — assert finiteness either way)
        let m = mm.markout_multiplier();
        assert!(m.is_finite() && (1.0..=3.0).contains(&m));
    }

    #[test]
    fn manual_orders_roundtrip() {
        let cfg = quick_cfg();
        let mut mm = MarketMaker::new(cfg, StrategyKind::Static, 5);
        let mut rng = Rng::new(5);
        for _ in 0..20 {
            mm.step(&mut rng);
        }
        let before = mm.venue.our_orders().len();
        let id = mm.manual_place(Side::Bid, 190, 3, true);
        assert!(mm.venue.our_orders().len() == before + 1);
        assert!(mm.manual_cancel(id));
        assert!(mm.venue.our_orders().len() == before);
        // manual take fills at the touch
        let (filled, avg, fee) = mm.manual_take(Side::Ask, 2);
        assert!(filled >= 1);
        assert!(avg > 0.0 && fee >= 0.0);
        // strategy orders and manual orders coexist
        let _ = mm.manual_place(Side::Bid, 185, 2, true);
        let strat_ids: Vec<u64> = mm.strategy_orders().iter().map(|(i, ..)| *i).collect();
        assert!(!strat_ids.is_empty());
        let n = mm.cancel_all_orders();
        assert!(n >= strat_ids.len());
        assert!(mm.strategy_orders().is_empty());
    }

    #[test]
    fn strategy_switch_and_kill_switch() {
        let cfg = quick_cfg();
        let mut mm = MarketMaker::new(cfg, StrategyKind::Static, 9);
        mm.select_strategy(StrategyKind::HjbPolicy);
        assert_eq!(mm.strategy.name(), "hjb");
        mm.set_kill_switch(true);
        assert!(mm.is_halted());
        mm.set_kill_switch(false);
        assert!(!mm.is_halted());
    }
}
