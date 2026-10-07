//! Simulated option market: the missing venue leg for option market
//! making.
//!
//! The perp-options-clob venue trades everlasting option series quoted
//! in implied volatility against a perp CLOB. This module simulates the
//! option side end-to-end so the options strategies can be validated by
//! Monte Carlo against ground truth:
//!
//! - an ATM implied-volatility state driven by a vol-of-vol GBM,
//! - an arbitrage-free SSVI surface (Gatheral–Jacquier 2014) built from
//!   the ATM pillar each step,
//! - client requests arriving with Cox intensity `a_v·e^{−κ_v·δ}` where
//!   `δ` is the distance between our IV quote and the client's surface
//!   fair IV (the Bergault–Guéant vega-approximation request model),
//! - fills at our quoted IVs (premium = BSM at the quoted IV, priced
//!   with the high-precision pricer),
//! - net book greeks (delta/gamma/vega) against the surface, and a
//!   mark-to-market value used by the engine's equity.
//!
//! Two quoter modes are provided for the Monte-Carlo A/B studies:
//! [`QuoterMode::Fixed`] (naive fixed vol spread, no inventory
//! dependence — the benchmark) and [`QuoterMode::Glft`] (the
//! vega-approximation GLFT quotes from `vol::OptMm`).

use crate::config::EngineConfig;
use micro::Rng;
use models::options::Kind;
use ob::Side;
use vol::ssvi::SsviSurface;

/// How the per-strike IV quotes are produced.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuoterMode {
    /// Naive benchmark: fixed vol spread around the surface fair IV,
    /// no inventory skew.
    Fixed { vol_spread: f64 },
    /// Vega-approximation GLFT quotes (vol::OptMm).
    Glft,
}

/// How the delta hedge is triggered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HedgeMode {
    /// Hedge to zero every step (the naive benchmark).
    EveryStep,
    /// Fixed band in lots.
    FixedBand { lots: f64 },
    /// Whalley–Wilmott-style band from `vol::HedgeCadence` (cost/risk
    /// closed form, MC-validated in `vol`).
    WwBand,
}

/// Configuration of the option market.
#[derive(Clone, Debug)]
pub struct OptionMarketConfig {
    pub enabled: bool,
    /// Effective maturity (years) of the everlasting series.
    pub t_eff: f64,
    /// Strike moneyness ladder (multiples of spot), ascending.
    pub moneyness: Vec<f64>,
    /// Option kind per leg.
    pub kind: Kind,
    /// Client request intensity at zero vol spread (per second).
    pub a_v: f64,
    /// Client request decay vs vol spread.
    pub kappa_v: f64,
    /// Lots per client request.
    pub lots_per_request: f64,
    /// Annualized vol-of-vol driving the ATM IV.
    pub vol_of_vol: f64,
    /// SSVI skew parameters.
    pub surface_rho: f64,
    pub surface_eta: f64,
    pub surface_gamma: f64,
    /// Quoter mode.
    pub quoter: QuoterMode,
    /// Hedge mode.
    pub hedge: HedgeMode,
    /// CARA risk aversion (wealth) for the GLFT quoter.
    pub gamma: f64,
    /// Perp lot size (base units per perp lot) — converts the option
    /// book's delta (underlying units) into perp lots for hedging.
    pub perp_lot_size: f64,
    /// Initial ATM IV (the surface's own level; the option market is a
    /// separate vol market, not tied to the venue's realized vol).
    pub initial_atm_iv: f64,
    /// Per-leg position bound (option lots): clients are refused when a
    /// fill would breach it (the desk's vega limit).
    pub max_position_lots: f64,
}

impl Default for OptionMarketConfig {
    fn default() -> Self {
        OptionMarketConfig {
            enabled: false,
            t_eff: 30.0 / 365.0, // monthly series
            moneyness: vec![0.90, 0.95, 1.0, 1.05, 1.10],
            kind: Kind::Call,
            a_v: 0.05,
            kappa_v: 40.0,
            // option lot notional is matched to perp book depth: one
            // request trades ~10 perp lots of delta, hedgeable in one take
            lots_per_request: 0.01,
            vol_of_vol: 4.0,
            surface_rho: -0.7,
            surface_eta: 1.0,
            surface_gamma: 0.5,
            quoter: QuoterMode::Glft,
            hedge: HedgeMode::WwBand,
            gamma: 1e-3,
            perp_lot_size: 0.001,
            initial_atm_iv: 0.9,
            max_position_lots: 40.0,
        }
    }
}

/// One client fill on our option quote.
#[derive(Clone, Copy, Debug)]
pub struct OptionFill {
    /// Leg index into the moneyness ladder.
    pub leg: usize,
    /// We bought (`Bid`) or sold (`Ask`) — the side of OUR quote that
    /// was hit.
    pub side: Side,
    pub lots: f64,
    /// Premium exchanged (quote units; signed from our perspective).
    pub premium: f64,
    /// The IV we traded at.
    pub iv: f64,
    /// The client's fair IV at fill time.
    pub fair_iv: f64,
}

/// Read-only view for strategies / the gateway.
#[derive(Clone, Copy, Debug, Default)]
pub struct OptionsView {
    pub atm_iv: f64,
    /// Net book delta in perp lots (signed).
    pub net_delta_lots: f64,
    /// Net book vega ($ per unit vol).
    pub net_vega: f64,
    /// Net book gamma (per spot²).
    pub net_gamma: f64,
    /// Mark-to-market value of the book (quote units).
    pub mark: f64,
    pub fills: u64,
    /// Realized premium cash (quote units).
    pub cash: f64,
}

const YEAR_SECS: f64 = 365.0 * 24.0 * 3600.0;

pub struct OptionMarket {
    pub cfg: OptionMarketConfig,
    /// Current ATM implied vol (annualized).
    pub atm_iv: f64,
    /// Our per-strike IV quotes: (iv_bid, iv_ask).
    pub quotes: Vec<(f64, f64)>,
    /// Signed position per leg (lots).
    pub pos: Vec<f64>,
    /// Premium cash (quote units).
    pub cash: f64,
    pub fills: u64,
}

impl OptionMarket {
    pub fn new(cfg: OptionMarketConfig, initial_atm_iv: f64) -> OptionMarket {
        let n = cfg.moneyness.len();
        let atm_iv = initial_atm_iv.clamp(0.05, 2.0);
        let mut q = Vec::with_capacity(n);
        for _ in 0..n {
            q.push((atm_iv, atm_iv));
        }
        OptionMarket {
            cfg,
            atm_iv,
            quotes: q,
            pos: vec![0.0; n],
            cash: 0.0,
            fills: 0,
        }
    }

    /// The arbitrage-free surface at the current ATM IV.
    pub fn surface(&self) -> SsviSurface {
        SsviSurface::new(
            self.cfg.surface_rho,
            self.cfg.surface_eta,
            self.cfg.surface_gamma,
            &[(self.cfg.t_eff, self.atm_iv)],
        )
    }

    /// Fair IV of a leg.
    pub fn fair_iv(&self, surface: &SsviSurface, leg: usize, spot: f64) -> f64 {
        let k = (self.cfg.moneyness[leg] * spot / spot).ln(); // moneyness IS k
        surface.iv(k, self.cfg.t_eff)
    }

    /// Advance one step: evolve the ATM IV, then client arrivals.
    pub fn step(&mut self, rng: &mut Rng, spot: f64, dt: f64) -> Vec<OptionFill> {
        // 1) ATM IV: lognormal with annualized vol-of-vol.
        let dv = self.cfg.vol_of_vol * (dt / YEAR_SECS).sqrt() * rng.normal();
        self.atm_iv = (self.atm_iv * dv.exp()).clamp(0.05, 2.0);
        // 2) surface + arrivals per leg
        let surface = self.surface();
        let mut out = Vec::new();
        for leg in 0..self.cfg.moneyness.len() {
            let k = self.cfg.moneyness[leg].ln();
            let fair = surface.iv(k, self.cfg.t_eff);
            let (qb, qa) = self.quotes[leg];
            let strike = self.cfg.moneyness[leg] * spot;
            // client SELLS to our bid (we buy) when our bid is close to
            // their fair from below
            let db = (fair - qb).max(0.0);
            let lam_b = self.cfg.a_v * (-self.cfg.kappa_v * db).exp();
            let room_b = self.cfg.max_position_lots - self.pos[leg];
            if room_b >= self.cfg.lots_per_request
                && rng.bernoulli(1.0 - (-lam_b * dt).exp())
            {
                let premium =
                    vol::greeks::price(self.cfg.kind, spot, strike, 0.0, 0.0, qb, self.cfg.t_eff)
                        * self.cfg.lots_per_request;
                self.cash -= premium;
                self.pos[leg] += self.cfg.lots_per_request;
                self.fills += 1;
                out.push(OptionFill {
                    leg,
                    side: Side::Bid,
                    lots: self.cfg.lots_per_request,
                    premium: -premium,
                    iv: qb,
                    fair_iv: fair,
                });
            }
            let da = (qa - fair).max(0.0);
            let lam_a = self.cfg.a_v * (-self.cfg.kappa_v * da).exp();
            let room_a = self.cfg.max_position_lots + self.pos[leg];
            if room_a >= self.cfg.lots_per_request
                && rng.bernoulli(1.0 - (-lam_a * dt).exp())
            {
                let premium =
                    vol::greeks::price(self.cfg.kind, spot, strike, 0.0, 0.0, qa, self.cfg.t_eff)
                        * self.cfg.lots_per_request;
                self.cash += premium;
                self.pos[leg] -= self.cfg.lots_per_request;
                self.fills += 1;
                out.push(OptionFill {
                    leg,
                    side: Side::Ask,
                    lots: self.cfg.lots_per_request,
                    premium,
                    iv: qa,
                    fair_iv: fair,
                });
            }
        }
        out
    }

    /// Net greeks and mark at the current surface.
    pub fn view(&self, spot: f64) -> OptionsView {
        let surface = self.surface();
        let mut delta = 0.0;
        let mut gamma = 0.0;
        let mut vega = 0.0;
        let mut mark = self.cash;
        for (leg, &p) in self.pos.iter().enumerate() {
            if p == 0.0 {
                continue;
            }
            let k = self.cfg.moneyness[leg].ln();
            let iv = surface.iv(k, self.cfg.t_eff);
            let strike = self.cfg.moneyness[leg] * spot;
            let g = vol::greeks::full_greeks(self.cfg.kind, spot, strike, 0.0, 0.0, iv, self.cfg.t_eff);
            // delta converted to PERP lots (the hedge trades perp lots)
            delta += g.delta * p / self.cfg.perp_lot_size;
            gamma += g.gamma * p;
            vega += g.vega * p;
            mark += g.price * p;
        }
        OptionsView {
            atm_iv: self.atm_iv,
            net_delta_lots: delta,
            net_vega: vega,
            net_gamma: gamma,
            mark,
            fills: self.fills,
            cash: self.cash,
        }
    }

    /// Average vega of one client request (for the GLFT vega scale).
    pub fn vega_per_request(&self, spot: f64) -> f64 {
        let surface = self.surface();
        let mid = self.cfg.moneyness.len() / 2;
        let k = self.cfg.moneyness[mid].ln();
        let iv = surface.iv(k, self.cfg.t_eff);
        let strike = self.cfg.moneyness[mid] * spot;
        let g = vol::greeks::full_greeks(self.cfg.kind, spot, strike, 0.0, 0.0, iv, self.cfg.t_eff);
        (g.vega * self.cfg.lots_per_request).max(1.0)
    }

    /// Compute and install the per-strike IV quotes under the configured
    /// quoter mode. `net_vega` is the current book vega; `taker_fee_frac`
    /// and `tick` price the per-fill hedge cost.
    pub fn requote(
        &mut self,
        spot: f64,
        net_vega: f64,
        vega_per_request: f64,
        taker_fee_frac: f64,
        tick: f64,
    ) {
        let _ = spot; // strikes are moneyness-based; spot enters via view()
        let surface = self.surface();
        match self.cfg.quoter {
            QuoterMode::Fixed { vol_spread } => {
                for leg in 0..self.quotes.len() {
                    let k = self.cfg.moneyness[leg].ln();
                    let fair = surface.iv(k, self.cfg.t_eff);
                    self.quotes[leg] = (fair - vol_spread / 2.0, fair + vol_spread / 2.0);
                }
            }
            QuoterMode::Glft => {
                // per-fill cost in vol units: the delta hedge + fees each
                // request pays (approximated at the ATM delta), divided by
                // the vega per request
                let atm_delta = 0.5;
                let hedge_lots = atm_delta * self.cfg.lots_per_request / self.cfg.perp_lot_size;
                // $ per PERP lot = (fee fraction * price + one tick) * lot size
                let fill_cost_money =
                    (taker_fee_frac * spot + tick) * self.cfg.perp_lot_size * hedge_lots;
                let fill_cost_vol = (fill_cost_money / vega_per_request.max(1e-9)).max(0.0);
                let mm = vol::OptMm {
                    gamma: self.cfg.gamma,
                    sigma_vol: self.cfg.vol_of_vol / YEAR_SECS.sqrt(),
                    kappa_v: self.cfg.kappa_v,
                    a_v: self.cfg.a_v,
                    vega_per_fill: vega_per_request,
                    fill_cost_vol,
                };
                for leg in 0..self.quotes.len() {
                    let k = self.cfg.moneyness[leg].ln();
                    let fair = surface.iv(k, self.cfg.t_eff);
                    self.quotes[leg] = mm.quote_ivs(fair, net_vega);
                }
            }
        }
    }

    /// Hedge band in lots under the configured hedge mode.
    pub fn hedge_band(&self, cost_per_lot: f64, gamma: f64, price_per_lot: f64) -> f64 {
        match self.cfg.hedge {
            HedgeMode::EveryStep => 0.0,
            HedgeMode::FixedBand { lots } => lots,
            HedgeMode::WwBand => {
                vol::HedgeCadence::new(cost_per_lot, gamma, price_per_lot).band_lots
            }
        }
    }
}

/// Enable the option market on a config (helper for studies/tests).
pub fn enable_options(cfg: &mut EngineConfig, quoter: QuoterMode, hedge: HedgeMode) {
    cfg.option_market.enabled = true;
    cfg.option_market.quoter = quoter;
    cfg.option_market.hedge = hedge;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk() -> OptionMarket {
        let mut cfg = OptionMarketConfig::default();
        cfg.enabled = true;
        OptionMarket::new(cfg, 0.30)
    }

    #[test]
    fn surface_is_arbitrage_free() {
        let m = mk();
        let s = m.surface();
        let rep = s.arb_report();
        assert!(rep.butterfly_condition, "GJ condition violated: {:?}", rep);
        assert!(rep.butterfly_grid, "density negative: {:?}", rep);
        assert!(rep.calendar, "calendar violated: {:?}", rep);
        assert!(s.static_arb_free_bsm(100.0, 0.0, 0.0));
    }

    #[test]
    fn glft_quotes_skew_with_vega_inventory() {
        let mut m = mk();
        let spot = 100.0;
        let vpr = m.vega_per_request(spot);
        m.requote(spot, 0.0, vpr, 0.00045, 0.5);
        let (b0, a0) = m.quotes[2];
        // long vega pulls the bid down and the ask toward the fair
        m.requote(spot, 5.0 * vpr, vpr, 0.00045, 0.5);
        let (b1, a1) = m.quotes[2];
        assert!(b1 < b0, "{b1} vs {b0}");
        assert!(a1 < a0, "{a1} vs {a0}");
        // skew symmetry: short vega mirrors
        m.requote(spot, -5.0 * vpr, vpr, 0.00045, 0.5);
        let (b2, a2) = m.quotes[2];
        assert!(b2 > b0);
        assert!(a2 > a0);
    }

    #[test]
    fn client_flow_fills_and_marks() {
        let mut m = mk();
        // Make quotes very tight so arrivals happen quickly.
        let mut rng = Rng::new(7);
        let mut total_fills = 0;
        let spot = 100.0;
        let vpr = m.vega_per_request(spot);
        m.requote(spot, 0.0, vpr, 0.00045, 0.5);
        for _ in 0..2000 {
            for q in m.quotes.iter_mut() {
                *q = (0.29, 0.31); // tight: lambda ~ a_v * e^{-250*0.01} ~ 0.0043
            }
            let fills = m.step(&mut rng, spot, 1.0);
            total_fills += fills.len();
        }
        assert!(total_fills > 10, "no client flow: {total_fills}");
        let v = m.view(spot);
        assert!(v.fills == total_fills as u64);
        assert!(v.net_delta_lots.abs() > 1e-9 || v.net_vega.abs() > 1e-9);
        assert!(v.mark.is_finite());
    }

    #[test]
    fn hedge_bands_order_sensibly() {
        let mut cfg = OptionMarketConfig::default();
        cfg.enabled = true;
        let m = OptionMarket::new(cfg.clone(), 0.3);
        let ww = m.hedge_band(0.02, 1e-3, 100.0);
        assert!(ww > 0.0);
        cfg.hedge = HedgeMode::EveryStep;
        let tight = OptionMarket::new(cfg.clone(), 0.3);
        assert_eq!(tight.hedge_band(0.02, 1e-3, 100.0), 0.0);
        cfg.hedge = HedgeMode::FixedBand { lots: 4.0 };
        let fixed = OptionMarket::new(cfg, 0.3);
        assert_eq!(fixed.hedge_band(0.02, 1e-3, 100.0), 4.0);
    }
}
