//! Pre-trade risk engine: inventory bounds, participation, throttles,
//! halts, and the loss kill-switch (venue-MMP style).

use crate::config::EngineConfig;
use crate::estimator::MarketState;
use crate::strategy::Quotes;

/// Risk decision for a desired quote pair.
#[derive(Clone, Debug, PartialEq)]
pub enum RiskAction {
    /// Pass through (possibly modified).
    Accept(Quotes),
    /// Halt quoting entirely this cycle.
    Halt(&'static str),
}

/// Rolling risk state.
pub struct RiskEngine {
    pub max_inventory: i64,
    /// Max quote size (lots).
    pub max_quote_lots: u64,
    /// Max total quoted lots per side.
    pub max_side_lots: u64,
    /// Requote throttle: min steps between cancels/replaces.
    pub requote_every: usize,
    /// Jump-halt: steps to stop quoting after a detected jump.
    pub jump_halt_steps: usize,
    /// Vol-halt multiplier over the slow sigma baseline.
    pub vol_halt_mult: f64,
    /// Daily loss kill-switch (quote units).
    pub loss_limit: f64,
    /// SFPM portfolio-margin utilization (set by the engine each cycle).
    /// Hard breach (>= 0.9) gates the risk-ADDING side; the unwind side
    /// always stays live — a margin breach must never freeze the desk
    /// (the exposure could then never be worked off).
    pub margin_util: f64,
    // state
    step: usize,
    last_requote_step: usize,
    jump_halt_until: usize,
    sigma_baseline: f64,
    halted: Option<&'static str>,
    rolling_fills_abs: f64,
    rolling_net_delta: f64,
}

impl RiskEngine {
    pub fn new(cfg: &EngineConfig) -> RiskEngine {
        RiskEngine {
            max_inventory: cfg.max_inventory,
            max_quote_lots: 50,
            max_side_lots: 500,
            requote_every: 4,
            jump_halt_steps: 8,
            vol_halt_mult: 4.0,
            loss_limit: 5000.0,
            step: 0,
            last_requote_step: 0,
            jump_halt_until: 0,
            sigma_baseline: match &cfg.mid {
                crate::config::MidModel::Gbm { sigma } => *sigma,
                crate::config::MidModel::Rough { sigma0, .. } => *sigma0,
            },
            halted: None,
            rolling_fills_abs: 0.0,
            rolling_net_delta: 0.0,
            margin_util: 0.0,
        }
    }

    /// Advance internal state one step (call per venue step).
    pub fn tick(&mut self) {
        self.step += 1;
    }

    /// Record one of our fills (MMP-style rolling windows).
    pub fn record_fill(&mut self, lots: u64, signed: i64) {
        self.rolling_fills_abs += lots as f64;
        self.rolling_net_delta += signed as f64;
        // decay
        self.rolling_fills_abs *= 0.995;
        self.rolling_net_delta *= 0.995;
    }

    /// Mark the kill-switch from realized PnL.
    pub fn observe_pnl(&mut self, equity: f64) {
        if equity < -self.loss_limit {
            self.halted = Some("loss kill-switch");
        }
    }

    /// Apply gates to a desired quote pair (ladder-aware: level 0 plus
    /// the deeper levels share the same inventory/size/price gates).
    pub fn filter(&mut self, desired: Quotes, state: &MarketState, inventory: i64) -> RiskAction {
        if let Some(reason) = self.halted {
            return RiskAction::Halt(reason);
        }
        // jump halt
        if state.jump_flag {
            self.jump_halt_until = self.step + self.jump_halt_steps;
        }
        if self.step < self.jump_halt_until {
            return RiskAction::Halt("jump halt");
        }
        // vol spike halt
        if self.sigma_baseline > 0.0
            && state.sigma_fast > self.vol_halt_mult * self.sigma_baseline
        {
            return RiskAction::Halt("vol spike");
        }
        // inventory bounds: stop quoting the side that increases exposure.
        // NOTE: when the option leg is enabled the engine passes the
        // UNHEDGED combined delta here (hedge leg + option book) — a
        // delta-hedged book is the point of the desk, not a breach; the
        // raw hedge-leg size must never trip this gate.
        let mut q = desired;
        if inventory >= self.max_inventory {
            q.bid = None;
            q.bid_levels.clear();
        }
        if inventory <= -self.max_inventory {
            q.ask = None;
            q.ask_levels.clear();
        }
        // SFPM margin-utilization gate: hard breach blocks the risk-adding
        // side only (directional, self-recovering — never a full freeze)
        if self.margin_util >= 0.9 {
            if inventory > 0 {
                q.bid = None;
                q.bid_levels.clear();
            } else if inventory < 0 {
                q.ask = None;
                q.ask_levels.clear();
            }
        }
        // size clamps (level 0 + per-side totals across the ladder)
        if let Some((p, s)) = q.bid {
            q.bid = Some((p, s.min(self.max_quote_lots)));
        }
        if let Some((p, s)) = q.ask {
            q.ask = Some((p, s.min(self.max_quote_lots)));
        }
        let mut bid_total = q.bid.map(|(_, s)| s).unwrap_or(0);
        for lv in q.bid_levels.iter_mut() {
            lv.1 = lv.1.min(self.max_quote_lots);
            bid_total = bid_total.saturating_add(lv.1);
        }
        let mut ask_total = q.ask.map(|(_, s)| s).unwrap_or(0);
        for lv in q.ask_levels.iter_mut() {
            lv.1 = lv.1.min(self.max_quote_lots);
            ask_total = ask_total.saturating_add(lv.1);
        }
        // per-side displayed size cap: drop deepest levels first
        while bid_total > self.max_side_lots && !q.bid_levels.is_empty() {
            let removed = q.bid_levels.pop().map(|(_, s)| s).unwrap_or(0);
            bid_total -= removed;
        }
        while ask_total > self.max_side_lots && !q.ask_levels.is_empty() {
            let removed = q.ask_levels.pop().map(|(_, s)| s).unwrap_or(0);
            ask_total -= removed;
        }
        // price sanity: both sides positive and ordered (level 0)
        if let (Some((b, _)), Some((a, _))) = (q.bid, q.ask) {
            if b >= a {
                return RiskAction::Halt("crossed self-quote");
            }
        }
        // deeper levels must stay on their side of the mid-ish band
        q.bid_levels.retain(|(p, _)| {
            q.ask.map(|(a, _)| *p < a).unwrap_or(true)
        });
        q.ask_levels.retain(|(p, _)| {
            q.bid.map(|(b, _)| *p > b).unwrap_or(true)
        });
        // throttle: only allow a requote every `requote_every` steps
        if self.step < self.last_requote_step + self.requote_every {
            return RiskAction::Accept(q);
        }
        RiskAction::Accept(q)
    }

    /// Mark a requote as taken.
    pub fn note_requote(&mut self) {
        self.last_requote_step = self.step;
    }

    /// Manual kill-switch control: `armed = true` latches a halt,
    /// `armed = false` clears any halt (including the loss switch).
    pub fn clear_halt(&mut self, armed: bool) {
        if armed {
            self.halted = Some("manual kill-switch");
        } else {
            self.halted = None;
        }
    }

    pub fn is_halted(&self) -> bool {
        self.halted.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EngineConfig;

    fn state(sig: f64, jump: bool) -> MarketState {
        MarketState {
            mid: 100.0,
            best_bid: Some((99, 10)),
            best_ask: Some((101, 10)),
            spread: 1.0,
            sigma_fast: sig,
            sigma_slow: sig,
            sigma_rough: sig,
            hurst: 0.1,
            micro_price: 100.0,
            imbalance: 0.5,
            ofi: 0.0,
            ofi_impact: 0.0,
            spread_est_log: 0.0,
            jump_flag: jump,
            clf_bid: None,
            clf_ask: None,
        }
    }

    fn quotes() -> Quotes {
        Quotes {
            bid: Some((99, 5)),
            ask: Some((101, 5)),
            ..Quotes::none()
        }
    }

    #[test]
    fn ladder_gates_drop_deep_levels_first() {
        let cfg = EngineConfig::default();
        let mut r = RiskEngine::new(&cfg);
        let st = state(0.02, false);
        let mut q = quotes();
        q.bid_levels = vec![(98, 5), (97, 5), (96, 600)];
        // inventory gate at the long bound clears the whole bid ladder
        match r.filter(q.clone(), &st, cfg.max_inventory) {
            RiskAction::Accept(out) => {
                assert!(out.bid.is_none());
                assert!(out.bid_levels.is_empty());
                assert!(out.ask.is_some());
            }
            _ => panic!(),
        }
        // side-size cap drops the deepest (last) levels first
        let mut r2 = RiskEngine::new(&cfg);
        let mut q2 = quotes();
        q2.bid_levels = (0..12).map(|i| (98 - i, 50u64)).collect();
        match r2.filter(q2, &st, 0) {
            RiskAction::Accept(out) => {
                let total: u64 = out.bid.map(|(_, s)| s).unwrap_or(0)
                    + out.bid_levels.iter().map(|(_, s)| s).sum::<u64>();
                assert!(total <= r2.max_side_lots, "total {total}");
                assert!(out.bid_levels.len() < 12, "no levels dropped");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn inventory_gates() {
        let cfg = EngineConfig::default();
        let mut r = RiskEngine::new(&cfg);
        let st = state(0.02, false);
        // flat: both sides
        match r.filter(quotes(), &st, 0) {
            RiskAction::Accept(q) => assert!(q.bid.is_some() && q.ask.is_some()),
            _ => panic!(),
        }
        // max long: no bids
        match r.filter(quotes(), &st, cfg.max_inventory) {
            RiskAction::Accept(q) => assert!(q.bid.is_none() && q.ask.is_some()),
            _ => panic!(),
        }
        match r.filter(quotes(), &st, -cfg.max_inventory) {
            RiskAction::Accept(q) => assert!(q.ask.is_none() && q.bid.is_some()),
            _ => panic!(),
        }
    }

    #[test]
    fn halts_fire() {
        let cfg = EngineConfig::default();
        let mut r = RiskEngine::new(&cfg);
        // jump halt
        let st = state(0.02, true);
        match r.filter(quotes(), &st, 0) {
            RiskAction::Halt("jump halt") => {}
            _ => panic!("expected jump halt"),
        }
        // vol spike
        let mut r2 = RiskEngine::new(&cfg);
        let st2 = state(0.02 * 10.0, false);
        match r2.filter(quotes(), &st2, 0) {
            RiskAction::Halt("vol spike") => {}
            _ => panic!("expected vol spike halt"),
        }
        // loss kill-switch latches
        let mut r3 = RiskEngine::new(&cfg);
        r3.observe_pnl(-1e9);
        match r3.filter(quotes(), &state(0.02, false), 0) {
            RiskAction::Halt("loss kill-switch") => {}
            _ => panic!("expected kill-switch"),
        }
    }
}
