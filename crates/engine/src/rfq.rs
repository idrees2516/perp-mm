//! Institutional RFQ lane, in the Paradigm / TradeParadigm-on-Paradex and
//! Derive V3 protocol shape: takers publish multi-leg package intents,
//! the desk responds with FIRM two-sided package quotes carrying a TTL,
//! execution is ATOMIC across legs (all-or-nothing), and combo margin
//! offsets price the risk netting of offsetting structures — a vertical
//! or a butterfly carries a fraction of the standalone margin of its
//! legs, so it quotes a fraction of the standalone spread.
//!
//! Operates over one option market's moneyness ladder (same kind/expiry):
//! packages are verticals (2 legs), butterflies (3 legs) and condors
//! (4 legs) plus outright singles.

use micro::Rng;
use vol::greeks::{full_greeks, price};

use crate::options_market::OptionMarket;

/// One package leg: ladder index + direction + lots.
#[derive(Clone, Copy, Debug)]
pub struct RfqLeg {
    pub leg: usize,
    pub dir: f64,
    pub lots: f64,
}

/// Package template (indices into the moneyness ladder, lots scaled by n).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RfqTemplate {
    Single,
    Vertical,
    Butterfly,
    Condor,
}

impl RfqTemplate {
    pub fn label(&self) -> &'static str {
        match self {
            RfqTemplate::Single => "single leg",
            RfqTemplate::Vertical => "vertical spread",
            RfqTemplate::Butterfly => "butterfly",
            RfqTemplate::Condor => "condor",
        }
    }

    /// `n` is the base lot size of the package (already in the option
    /// market's lot units — institutional size, a multiple of the retail
    /// request size).
    pub fn legs(&self, n: f64, atm: usize, len: usize) -> Vec<RfqLeg> {
        let cl = |i: isize| (i.clamp(0, len as isize - 1)) as usize;
        match self {
            RfqTemplate::Single => vec![RfqLeg { leg: atm, dir: 1.0, lots: n }],
            RfqTemplate::Vertical => vec![
                RfqLeg { leg: cl(atm as isize - 1), dir: 1.0, lots: n },
                RfqLeg { leg: cl(atm as isize + 1), dir: -1.0, lots: n },
            ],
            RfqTemplate::Butterfly => vec![
                RfqLeg { leg: cl(atm as isize - 2), dir: 1.0, lots: n },
                RfqLeg { leg: atm, dir: -1.0, lots: 2.0 * n },
                RfqLeg { leg: cl(atm as isize + 2), dir: 1.0, lots: n },
            ],
            RfqTemplate::Condor => vec![
                RfqLeg { leg: cl(atm as isize - 3), dir: 1.0, lots: n },
                RfqLeg { leg: cl(atm as isize - 1), dir: -1.0, lots: n },
                RfqLeg { leg: cl(atm as isize + 1), dir: -1.0, lots: n },
                RfqLeg { leg: cl(atm as isize + 3), dir: 1.0, lots: n },
            ],
        }
    }
}

/// A live firm quote on a package.
#[derive(Clone, Debug)]
pub struct RfqQuote {
    pub id: u64,
    pub template: RfqTemplate,
    pub legs: Vec<RfqLeg>,
    /// Package premium: desk buys at bid, sells at ask.
    pub bid: f64,
    pub ask: f64,
    pub fair: f64,
    pub margin_offset: f64,
    pub delta: f64,
    pub vega: f64,
    pub gamma: f64,
    /// Seconds of firmness remaining.
    pub ttl: f64,
    pub status: RfqStatus,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RfqStatus {
    Quoted,
    Executed,
    Expired,
}

/// Execution-risk context from the desk's risk engine.
pub struct RfqRiskCtx {
    pub unhedged_lots: f64,
    pub net_delta_limit: f64,
    pub net_vega: f64,
    pub vega_limit: f64,
    pub freeze: bool,
    pub lot_size: f64,
    pub multiplier: f64,
}

/// Result of an execution attempt.
pub enum RfqExec {
    Ok,
    Refused(&'static str),
}

#[derive(Default)]
pub struct RfqStats {
    pub requests: u64,
    pub executed: u64,
    pub refused: u64,
    /// Cumulative package premium received (desk view).
    pub premium_flow: f64,
}

pub struct RfqEngine {
    pub active: Vec<RfqQuote>,
    pub recent: Vec<RfqQuote>,
    pub stats: RfqStats,
    pub ttl_s: f64,
    pub auto_rate: f64,
    next_id: u64,
}

impl Default for RfqEngine {
    fn default() -> Self {
        RfqEngine::new()
    }
}

impl RfqEngine {
    pub fn new() -> RfqEngine {
        RfqEngine {
            active: Vec::new(),
            recent: Vec::new(),
            stats: RfqStats::default(),
            ttl_s: 5.0,
            auto_rate: 1.0 / 20.0,
            next_id: 1,
        }
    }

    /// Firm two-sided package quote with TTL, combo margin offset priced in.
    /// `quotes` are the desk's current per-leg (iv_bid, iv_ask).
    pub fn quote(
        &mut self,
        legs: Vec<RfqLeg>,
        template: RfqTemplate,
        spot: f64,
        mkt: &OptionMarket,
        ttl: f64,
    ) -> RfqQuote {
        self.stats.requests += 1;
        let id = self.next_id;
        self.next_id += 1;
        let mut bid = 0.0;
        let mut ask = 0.0;
        let mut fair = 0.0;
        let mut delta = 0.0;
        let mut vega = 0.0;
        let mut gamma = 0.0;
        let mut gross = 0.0;
        for l in &legs {
            let m = mkt.cfg.moneyness[l.leg];
            let strike = m * spot;
            let t = mkt.cfg.t_eff;
            let fair_iv = mkt.surface().iv(m.ln(), t);
            let (iv_b, iv_a) = mkt.quotes[l.leg];
            let pb = price(mkt.cfg.kind, spot, strike, 0.0, 0.0, iv_b, t);
            let pa = price(mkt.cfg.kind, spot, strike, 0.0, 0.0, iv_a, t);
            let pf = price(mkt.cfg.kind, spot, strike, 0.0, 0.0, fair_iv, t);
            bid += if l.dir > 0.0 { pb } else { -pa } * l.lots;
            ask += if l.dir > 0.0 { pa } else { -pb } * l.lots;
            fair += l.dir * pf * l.lots;
            let g = full_greeks(mkt.cfg.kind, spot, strike, 0.0, 0.0, fair_iv, t);
            delta += l.dir * g.delta * l.lots;
            vega += l.dir * g.vega * l.lots;
            gamma += l.dir * g.gamma * l.lots;
            gross += (g.delta * l.lots).abs() + 0.001 * (g.vega * l.lots).abs() + 0.5 * (g.gamma * l.lots).abs() * spot;
        }
        let net = delta.abs() + 0.001 * vega.abs() + 0.5 * gamma.abs() * spot;
        let margin_offset = (1.0 - net / gross.max(1e-9)).clamp(0.0, 0.6);
        let half = (ask - bid) / 2.0;
        let half2 = (half * (1.0 - margin_offset * 0.7)).max(0.05f64.max(fair.abs() * 0.001));
        RfqQuote {
            id,
            template,
            legs,
            bid: fair - half2,
            ask: fair + half2,
            fair,
            margin_offset,
            delta,
            vega,
            gamma,
            ttl,
            status: RfqStatus::Quoted,
        }
    }

    /// Institutional flow: TTL decay + Poisson arrivals + client decisions.
    /// Returns executed package ids (the engine refreshes margin after).
    pub fn step(
        &mut self,
        dt: f64,
        spot: f64,
        mkt: &mut OptionMarket,
        rng: &mut Rng,
        ctx: &RfqRiskCtx,
    ) -> Vec<u64> {
        for q in self.active.iter_mut() {
            q.ttl -= dt;
            if q.ttl <= 0.0 {
                q.status = RfqStatus::Expired;
            }
        }
        let expired: Vec<RfqQuote> = self
            .active
            .iter()
            .filter(|q| q.status == RfqStatus::Expired)
            .cloned()
            .collect();
        self.recent.extend(expired);
        self.active.retain(|q| q.status == RfqStatus::Quoted);
        let mut executed = Vec::new();
        if !ctx.freeze && rng.bernoulli(1.0 - (-self.auto_rate * dt).exp()) {
            let templates = [
                RfqTemplate::Single,
                RfqTemplate::Vertical,
                RfqTemplate::Butterfly,
                RfqTemplate::Condor,
            ];
            let tpl = templates[(rng.uniform() * templates.len() as f64) as usize];
            // institutional size: 5–25× the retail request size
            let n = (5.0 + rng.uniform() * 16.0) * mkt.cfg.lots_per_request;
            let atm = mkt.cfg.moneyness.len() / 2;
            let legs = tpl.legs(n, atm, mkt.cfg.moneyness.len());
            let q = self.quote(legs, tpl, spot, mkt, self.ttl_s);
            // client decision against an internal fair with noise
            let client_fair = q.fair * (1.0 + rng.normal() * 0.004);
            let side = if q.ask <= client_fair {
                Some(false) // desk sells
            } else if q.bid >= client_fair {
                Some(true) // desk buys
            } else {
                None
            };
            self.active.push(q);
            if let Some(buy) = side {
                let id = self.active.last().unwrap().id;
                match self.execute(id, buy, mkt, ctx) {
                    RfqExec::Ok => executed.push(id),
                    RfqExec::Refused(_) => {}
                }
            }
        }
        executed
    }

    /// Atomic package execution — all legs or nothing, risk-checked.
    pub fn execute(&mut self, id: u64, desk_buys: bool, mkt: &mut OptionMarket, ctx: &RfqRiskCtx) -> RfqExec {
        let Some(pos) = self.active.iter().position(|q| q.id == id && q.status == RfqStatus::Quoted) else {
            return RfqExec::Refused("quote not found / expired");
        };
        let sign = if desk_buys { 1.0 } else { -1.0 };
        let pkg_delta_lots = (self.active[pos].delta * ctx.multiplier) / ctx.lot_size;
        if (ctx.unhedged_lots + sign * pkg_delta_lots).abs() > ctx.net_delta_limit {
            self.stats.refused += 1;
            return RfqExec::Refused("net-delta limit");
        }
        let vega_after = ctx.net_vega + sign * self.active[pos].vega;
        if vega_after.abs() > ctx.vega_limit {
            self.stats.refused += 1;
            return RfqExec::Refused("vega limit");
        }
        // atomic pre-check on per-leg bounds
        for l in &self.active[pos].legs {
            let next = mkt.pos[l.leg] + sign * l.dir * l.lots;
            if next.abs() > mkt.cfg.max_position_lots {
                self.stats.refused += 1;
                return RfqExec::Refused("leg limit");
            }
        }
        let (legs, price, lots) = {
            let q = &self.active[pos];
            (q.legs.clone(), if desk_buys { q.bid } else { q.ask }, q.legs.iter().map(|l| l.lots).sum::<f64>())
        };
        for l in &legs {
            mkt.pos[l.leg] += sign * l.dir * l.lots;
        }
        // premium: desk buys pay the bid, desk sells receive the ask
        let flow = if desk_buys { -price * lots } else { price * lots };
        mkt.cash += flow;
        self.stats.premium_flow += flow;
        mkt.fills += legs.len() as u64;
        self.stats.executed += 1;
        let q = &mut self.active[pos];
        q.status = RfqStatus::Executed;
        let q = q.clone();
        self.active.remove(pos);
        self.recent.push(q);
        if self.recent.len() > 14 {
            self.recent.drain(0..self.recent.len() - 14);
        }
        RfqExec::Ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options_market::OptionMarketConfig;

    fn mkt() -> OptionMarket {
        let cfg = OptionMarketConfig::default();
        OptionMarket::new(cfg, 0.55)
    }

    #[test]
    fn templates_build_in_range() {
        let len = 7;
        let atm = 3;
        for tpl in [RfqTemplate::Single, RfqTemplate::Vertical, RfqTemplate::Butterfly, RfqTemplate::Condor] {
            for n in [5.0f64, 25.0] {
                let legs = tpl.legs(n, atm, len);
                assert!(!legs.is_empty());
                for l in &legs {
                    assert!(l.leg < len);
                    assert!(l.lots > 0.0);
                }
            }
        }
    }

    #[test]
    fn quote_two_sided_with_offset() {
        let mut eng = RfqEngine::new();
        let mkt = mkt();
        let legs = RfqTemplate::Vertical.legs(10.0, 3, mkt.cfg.moneyness.len());
        let q = eng.quote(legs, RfqTemplate::Vertical, 100.0, &mkt, 5.0);
        assert!(q.bid < q.fair && q.fair < q.ask, "bid {} fair {} ask {}", q.bid, q.fair, q.ask);
        assert!(q.margin_offset > 0.3, "vertical offset {}", q.margin_offset);
        let legs_s = RfqTemplate::Single.legs(10.0, 3, mkt.cfg.moneyness.len());
        let qs = eng.quote(legs_s, RfqTemplate::Single, 100.0, &mkt, 5.0);
        assert!(qs.margin_offset < 0.05, "single offset {}", qs.margin_offset);
        assert!(q.margin_offset > qs.margin_offset + 0.2);
    }

    #[test]
    fn atomic_execution_moves_every_leg() {
        let mut eng = RfqEngine::new();
        let mut mkt = mkt();
        let legs = RfqTemplate::Butterfly.legs(5.0, 3, mkt.cfg.moneyness.len());
        let q = eng.quote(legs, RfqTemplate::Butterfly, 100.0, &mkt, 5.0);
        eng.active.push(q);
        let id = eng.active[0].id;
        let before: Vec<f64> = mkt.pos.clone();
        let ctx = RfqRiskCtx {
            unhedged_lots: 0.0,
            net_delta_limit: 1000.0,
            net_vega: 0.0,
            vega_limit: 1e9,
            freeze: false,
            lot_size: 1.0,
            multiplier: 1.0,
        };
        let res = eng.execute(id, false, &mut mkt, &ctx);
        assert!(matches!(res, RfqExec::Ok));
        let moved = before.iter().zip(mkt.pos.iter()).filter(|(a, b)| (**a - **b).abs() > 1e-9).count();
        assert_eq!(moved, 3, "butterfly must move 3 legs atomically, moved {moved}");
        assert_eq!(eng.stats.executed, 1);
    }

    #[test]
    fn refusal_on_net_delta_limit() {
        let mut eng = RfqEngine::new();
        let mut mkt = mkt();
        let legs = RfqTemplate::Single.legs(10.0, 3, mkt.cfg.moneyness.len());
        let q = eng.quote(legs, RfqTemplate::Single, 100.0, &mkt, 5.0);
        eng.active.push(q);
        let id = eng.active[0].id;
        let before: Vec<f64> = mkt.pos.clone();
        let ctx = RfqRiskCtx {
            unhedged_lots: 0.0,
            net_delta_limit: 0.01,
            net_vega: 0.0,
            vega_limit: 1e9,
            freeze: false,
            lot_size: 1.0,
            multiplier: 1.0,
        };
        let res = eng.execute(id, true, &mut mkt, &ctx);
        assert!(matches!(res, RfqExec::Refused("net-delta limit")));
        assert_eq!(before, mkt.pos, "refusal must not touch the book");
        assert_eq!(eng.stats.refused, 1);
    }

    #[test]
    fn ttl_expiry_sweeps() {
        let mut eng = RfqEngine::new();
        let mut mkt = mkt();
        let legs = RfqTemplate::Single.legs(5.0, 3, mkt.cfg.moneyness.len());
        let q = eng.quote(legs, RfqTemplate::Single, 100.0, &mkt, 1.0);
        eng.active.push(q);
        let mut rng = micro::Rng::new(42);
        let ctx = RfqRiskCtx {
            unhedged_lots: 0.0,
            net_delta_limit: 100.0,
            net_vega: 0.0,
            vega_limit: 1e9,
            freeze: true, // no arrivals
            lot_size: 1.0,
            multiplier: 1.0,
        };
        eng.step(2.0, 100.0, &mut mkt, &mut rng, &ctx);
        assert!(eng.active.is_empty());
        assert_eq!(eng.recent.len(), 1);
        assert_eq!(eng.recent[0].status, RfqStatus::Expired);
    }
}
