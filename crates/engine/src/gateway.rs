//! Engine gateway: builds [`GwSnapshot`]s from a live [`MarketMaker`]
//! and applies [`GwCommand`]s to it. This is the bridge the daemon
//! binary drives; it is also used directly by the in-process
//! integration tests.

use crate::estimator::MarketState;
use crate::mm::MarketMaker;
use crate::strategy::StrategyKind;
use feed::proto::{GwAck, GwCommand, GwSnapshot, OptionSnapshot, PnlParts};
use ob::Side;

/// Ring of engine-step latencies for p50/p99 reporting.
pub struct LatencyRing {
    buf: Vec<u64>,
    pos: usize,
    filled: usize,
}

impl LatencyRing {
    pub fn new(cap: usize) -> LatencyRing {
        LatencyRing { buf: vec![0; cap], pos: 0, filled: 0 }
    }

    pub fn push(&mut self, v: u64) {
        if self.buf.is_empty() {
            return;
        }
        self.buf[self.pos] = v;
        self.pos = (self.pos + 1) % self.buf.len();
        self.filled = (self.filled + 1).min(self.buf.len());
    }

    /// (p50, p99) over the recorded samples (nearest-rank convention).
    pub fn percentiles(&self) -> (u64, u64) {
        if self.filled == 0 {
            return (0, 0);
        }
        let mut v: Vec<u64> = self.buf[..self.filled].to_vec();
        v.sort_unstable();
        let n = v.len();
        let p = |q: f64| -> u64 {
            let idx = ((q * n as f64).ceil() as usize).clamp(1, n) - 1;
            v[idx]
        };
        (p(0.5), p(0.99))
    }
}

/// Tunable parameter slots (indices used by the TUI / `SetParam`).
pub const PARAM_GAMMA: u8 = 0;
pub const PARAM_INTENSITY_A: u8 = 1;
pub const PARAM_INTENSITY_KAPPA: u8 = 2;
pub const PARAM_MAX_INVENTORY: u8 = 3;
pub const PARAM_LEVELS: u8 = 4;
pub const PARAM_MARKOUT_THETA: u8 = 5;
pub const PARAM_VOL_OF_VOL: u8 = 6;

/// Strategy index <-> kind (stable ordering for the wire).
pub fn strategy_by_index(idx: u8) -> Option<StrategyKind> {
    StrategyKind::all().get(idx as usize).copied()
}

pub fn strategy_index(kind: StrategyKind) -> u8 {
    StrategyKind::all()
        .iter()
        .position(|&k| k == kind)
        .map(|i| i as u8)
        .unwrap_or(0)
}

/// Build a snapshot of the market maker's current state. `state` is the
/// latest estimator snapshot (recomputed by the daemon each tick);
/// `drops`/latency come from the gateway transport.
pub fn build_snapshot(
    mm: &mut MarketMaker,
    state: &MarketState,
    seq: u64,
    lat: &LatencyRing,
    drops: u64,
) -> GwSnapshot {
    let bids: Vec<(u64, u64)> = mm.venue.book.ladder(Side::Bid, 10).collect();
    let asks: Vec<(u64, u64)> = mm.venue.book.ladder(Side::Ask, 10).collect();
    let our_orders: Vec<(u8, u64, u64)> = mm
        .venue
        .our_orders()
        .into_iter()
        .map(|(_, s, p, l)| {
            (
                match s {
                    Side::Bid => 1u8,
                    Side::Ask => 2u8,
                },
                p,
                l,
            )
        })
        .collect();
    let (p50, p99) = lat.percentiles();
    let opt = mm.last_opt_view.map(|v| OptionSnapshot {
        atm_iv: v.atm_iv,
        net_delta_lots: v.net_delta_lots,
        net_vega: v.net_vega,
        net_gamma: v.net_gamma,
        mark: v.mark,
        fills: v.fills,
        leg_quotes: mm
            .opt
            .as_ref()
            .map(|o| o.quotes.clone())
            .unwrap_or_default(),
    });
    GwSnapshot {
        seq,
        step: mm.venue.step as u64,
        sim_time: mm.venue.step as f64 * mm.cfg.dt,
        mid: state.mid,
        micro_price: state.micro_price,
        spread: state.spread,
        sigma_fast: state.sigma_fast,
        sigma_slow: state.sigma_slow,
        sigma_rough: state.sigma_rough,
        hurst: state.hurst,
        imbalance: state.imbalance,
        ofi: state.ofi,
        jump_flag: state.jump_flag,
        book_bids: bids,
        book_asks: asks,
        our_orders,
        inventory: mm.venue.inventory,
        equity: mm.equity(),
        pnl: PnlParts {
            spread_capture: mm.venue.spread_capture,
            adverse_cost: mm.venue.adverse_cost,
            fees: mm.venue.fees_paid,
            funding: mm.venue.funding_paid,
            hedge_cost: mm.venue.hedge_cost,
        },
        markout_mult: mm.markout_multiplier(),
        halted: mm.is_halted(),
        paused: mm.paused,
        strategy: mm.strategy_current_index(),
        option: opt,
        step_ns_p50: p50,
        step_ns_p99: p99,
        drops,
    }
}

/// Apply one command to the market maker; returns the ack.
pub fn apply_command(mm: &mut MarketMaker, cmd: &GwCommand) -> GwAck {
    match cmd {
        GwCommand::Ping => GwAck::Pong,
        GwCommand::SelectStrategy(idx) => match strategy_by_index(*idx) {
            Some(kind) => {
                mm.select_strategy(kind);
                GwAck::Ok
            }
            None => GwAck::BadParam,
        },
        GwCommand::SetParam(id, v) => {
            if !v.is_finite() {
                return GwAck::BadParam;
            }
            let ok = match *id {
                PARAM_GAMMA => {
                    if *v > 0.0 && *v < 10.0 {
                        mm.cfg.gamma = *v;
                        true
                    } else {
                        false
                    }
                }
                PARAM_INTENSITY_A => {
                    if *v > 0.0 && *v < 100.0 {
                        mm.cfg.intensity_a = *v;
                        true
                    } else {
                        false
                    }
                }
                PARAM_INTENSITY_KAPPA => {
                    if *v > 0.0 && *v < 100.0 {
                        mm.cfg.intensity_kappa = *v;
                        true
                    } else {
                        false
                    }
                }
                PARAM_MAX_INVENTORY => {
                    if *v >= 1.0 && *v <= 10_000.0 {
                        mm.cfg.max_inventory = *v as i64;
                        true
                    } else {
                        false
                    }
                }
                PARAM_LEVELS => {
                    let n = *v as usize;
                    if (1..=8).contains(&n) {
                        mm.cfg.levels = n;
                        true
                    } else {
                        false
                    }
                }
                PARAM_MARKOUT_THETA => {
                    if *v >= 0.0 && *v <= 5.0 {
                        mm.cfg.markout_theta = *v;
                        true
                    } else {
                        false
                    }
                }
                PARAM_VOL_OF_VOL
                    if *v > 0.0 && *v <= 5.0 => {
                        mm.cfg.option_market.vol_of_vol = *v;
                        true
                    }
                _ => false,
            };
            if ok {
                // strategy internals depend on solved parameters: rebuild
                let idx = mm.strategy_current_index();
                if let Some(kind) = strategy_by_index(idx) {
                    mm.select_strategy(kind);
                }
                GwAck::Ok
            } else {
                GwAck::BadParam
            }
        }
        GwCommand::PauseQuotes(p) => {
            mm.paused = *p;
            if *p {
                let _ = mm.cancel_all_orders();
            }
            GwAck::Ok
        }
        GwCommand::KillSwitch(armed) => {
            mm.set_kill_switch(*armed);
            if *armed {
                let _ = mm.cancel_all_orders();
            }
            GwAck::Ok
        }
        GwCommand::ManualPlace { side, price_ticks, lots, post_only } => {
            if *lots == 0 || *price_ticks == 0 {
                return GwAck::BadParam;
            }
            mm.manual_place(*side, *price_ticks, *lots, *post_only);
            GwAck::Ok
        }
        GwCommand::ManualTake { side, lots } => {
            if *lots == 0 {
                return GwAck::BadParam;
            }
            mm.manual_take(*side, *lots);
            GwAck::Ok
        }
        GwCommand::ManualCancel { order_id } => {
            let _ = mm.manual_cancel(*order_id);
            GwAck::Ok
        }
        GwCommand::CancelAll => {
            let _ = mm.cancel_all_orders();
            GwAck::Ok
        }
        GwCommand::ResetSession => {
            // Reset accounting (positions/fees/PnL) without dropping the
            // book: full resets are done by restarting the daemon.
            mm.venue.cash = 0.0;
            mm.venue.inventory = 0;
            mm.venue.fees_paid = 0.0;
            mm.venue.funding_paid = 0.0;
            mm.venue.spread_capture = 0.0;
            mm.venue.adverse_cost = 0.0;
            mm.venue.hedge_cost = 0.0;
            mm.venue.our_fills = 0;
            mm.venue.taker_fills = 0;
            mm.venue.equity_peak = 0.0;
            mm.venue.max_drawdown = 0.0;
            if let Some(o) = mm.opt.as_mut() {
                o.pos.iter_mut().for_each(|p| *p = 0.0);
                o.cash = 0.0;
                o.fills = 0;
            }
            GwAck::Ok
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EngineConfig;
    use crate::estimator::EstimatorStack;
    use feed::proto::{decode_msg, encode_msg, GwMsg};
    use micro::Rng;

    fn mm_with(steps: usize, kind: StrategyKind) -> (MarketMaker, MarketState) {
        let cfg = EngineConfig::default();
        let mut mm = MarketMaker::new(cfg, kind, 99);
        let mut rng = Rng::new(99);
        for _ in 0..steps {
            mm.step(&mut rng);
        }
        // produce a state snapshot like the daemon does
        let est = EstimatorStack::new(&mm.cfg);
        let st = est.snapshot(&mut mm.venue.book, mm.cfg.tick_size);
        (mm, st)
    }

    #[test]
    fn snapshot_builds_and_roundtrips() {
        let (mut mm, st) = mm_with(120, StrategyKind::MultiLevel);
        let mut lat = LatencyRing::new(64);
        for i in 0..64 {
            lat.push(100 + i * 10);
        }
        let snap = build_snapshot(&mut mm, &st, 7, &lat, 3);
        assert_eq!(snap.seq, 7);
        assert_eq!(snap.drops, 3);
        assert!(snap.step_ns_p50 >= 100 && snap.step_ns_p99 >= snap.step_ns_p50);
        assert!(!snap.book_bids.is_empty() && !snap.book_asks.is_empty());
        assert!(snap.equity.is_finite());
        // MultiLevel is index 6 in the wire ordering
        assert_eq!(snap.strategy, 6);
        // wire round-trip (byte idempotence — robust to NaN payload
        // fields like `hurst` where NaN != NaN under PartialEq)
        let bytes = encode_msg(&GwMsg::Snapshot(snap));
        let back = decode_msg(&bytes).expect("decode");
        let bytes2 = encode_msg(&back);
        assert_eq!(bytes, bytes2);
        assert!(matches!(back, GwMsg::Snapshot(_)));
    }

    #[test]
    fn latency_percentiles() {
        let mut r = LatencyRing::new(8);
        for v in [10u64, 20, 30, 40, 50, 60, 70, 80] {
            r.push(v);
        }
        let (p50, p99) = r.percentiles();
        assert_eq!(p50, 40);
        assert_eq!(p99, 80);
        // wrap-around keeps the last 8
        for v in [1u64, 2, 3] {
            r.push(v);
        }
        let (p50, _) = r.percentiles();
        assert!(p50 < 80);
    }

    #[test]
    fn commands_apply() {
        let (mut mm, _) = mm_with(30, StrategyKind::Static);
        assert_eq!(apply_command(&mut mm, &GwCommand::Ping), GwAck::Pong);
        assert_eq!(
            apply_command(&mut mm, &GwCommand::SelectStrategy(2)),
            GwAck::Ok
        );
        assert_eq!(mm.strategy_current_index(), 2);
        assert_eq!(
            apply_command(&mut mm, &GwCommand::SelectStrategy(200)),
            GwAck::BadParam
        );
        assert_eq!(
            apply_command(&mut mm, &GwCommand::SetParam(PARAM_GAMMA, 0.5)),
            GwAck::Ok
        );
        assert!((mm.cfg.gamma - 0.5).abs() < 1e-12);
        assert_eq!(
            apply_command(&mut mm, &GwCommand::SetParam(PARAM_GAMMA, -1.0)),
            GwAck::BadParam
        );
        apply_command(&mut mm, &GwCommand::PauseQuotes(true));
        assert!(mm.paused);
        apply_command(&mut mm, &GwCommand::PauseQuotes(false));
        assert!(!mm.paused);
        apply_command(&mut mm, &GwCommand::KillSwitch(true));
        assert!(mm.is_halted());
        apply_command(&mut mm, &GwCommand::KillSwitch(false));
        assert!(!mm.is_halted());
        let eq_before = mm.equity();
        apply_command(&mut mm, &GwCommand::ResetSession);
        assert!((mm.equity() - eq_before).abs() < 1e-6); // flat: equity unchanged-ish
        // manual order flow
        let ack = apply_command(
            &mut mm,
            &GwCommand::ManualPlace { side: Side::Bid, price_ticks: 190, lots: 3, post_only: true },
        );
        assert_eq!(ack, GwAck::Ok);
        assert!(!mm.venue.our_orders().is_empty());
        let ack = apply_command(&mut mm, &GwCommand::ManualTake { side: Side::Ask, lots: 1 });
        assert_eq!(ack, GwAck::Ok);
        assert_eq!(apply_command(&mut mm, &GwCommand::CancelAll), GwAck::Ok);
        assert!(mm.venue.our_orders().is_empty());
    }
}
