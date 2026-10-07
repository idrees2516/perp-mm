//! Option market making via the vega approximation.
//!
//! # Model (Bergault & Guéant, "Algorithmic Market Making for Options";
//! GLFT arXiv:1105.3115 for the underlying quoting equations)
//!
//! The option book is summarized by its net vega `V` (in $ per unit
//! vol). Client requests arrive with intensity `λ(δ) = A_v e^{−κ_v δ}`
//! where `δ` is the vol half-spread. Each fill trades `vega_per_fill`
//! dollars of vega, so the inventory in "fill units" is
//! `q = V / vega_per_fill` and the wealth impact of a vol move is
//! `q · vega_per_fill · dσ`. Substituting the effective "price"
//! volatility `σ_eff = vega_per_fill · σ_vol` (σ_vol = vol-of-vol per
//! √second) into the GLFT asymptotics gives the optimal vol distances:
//! the bid/ask IV quotes around the surface's fair IV.
//!
//! # Hedge cadence (Whalley–Wilmott-style band, re-derived)
//!
//! Unhedged delta `δ` (lots) follows a Brownian motion with vol σ_S
//! (per √s); rebalancing to zero at exits of the band [−H, H] costs
//! `c` per lot (taker fee + half-spread). With CARA risk aversion γ
//! the steady-state loss rate is
//! `γ S² σ_S² H²/6 + c σ_S²/H`; minimizing over H gives the closed form
//! `H* = (3 c / (γ S²))^{1/3}` — the σ_S factors cancel (both terms are
//! linear in σ_S²), reproducing the classic (cost/risk-aversion)^{1/3}
//! structure. MC-validated below: the realized loss-rate is minimized
//! at H* (hump test).

use crate::ssvi::SsviSurface;
use crate::volga::VannaVolga;
use models::glft::GlftAsymptotic;
use models::options::Kind;

/// Option market-making quote engine (per-venue parameters).
#[derive(Clone, Debug)]
pub struct OptMm {
    /// CARA risk aversion (per $ of wealth).
    pub gamma: f64,
    /// Vol-of-vol per sqrt(second).
    pub sigma_vol: f64,
    /// Client-request intensity decay w.r.t. vol spread.
    pub kappa_v: f64,
    /// Client-request intensity at zero vol spread (per second).
    pub a_v: f64,
    /// Vega traded per request ($ per unit vol).
    pub vega_per_fill: f64,
    /// Per-fill execution cost expressed in vol units (the delta hedge
    /// and fees each fill pays, divided by the vega per fill): the
    /// dealer formula quotes `fill_cost + GLFT markup + skew`
    /// (Bergault–Guéant include hedging costs exactly this way).
    pub fill_cost_vol: f64,
}

impl OptMm {
    /// GLFT asymptotics in vega space: σ_eff = vega_per_fill · σ_vol,
    /// inventory measured in fill units.
    fn glft(&self) -> GlftAsymptotic {
        GlftAsymptotic::new(
            self.gamma,
            self.vega_per_fill * self.sigma_vol,
            self.kappa_v,
            self.a_v,
        )
    }

    /// Inventory in fill units (rounded to the nearest fill).
    pub fn q_fills(&self, net_vega: f64) -> i64 {
        (net_vega / self.vega_per_fill).round() as i64
    }

    /// Optimal vol distances `(bid, ask)` at net vega inventory `V`
    /// (vol units). Long vega (q > 0) pushes the bid further away and
    /// the ask closer — GLFT skew in vol space; the per-fill cost floor
    /// is added to both sides.
    pub fn vol_distances(&self, net_vega: f64) -> (f64, f64) {
        let g = self.glft();
        let q = self.q_fills(net_vega);
        (
            (g.delta_bid(q) + self.fill_cost_vol).max(1e-6),
            (g.delta_ask(q) + self.fill_cost_vol).max(1e-6),
        )
    }

    /// IV quotes around a fair IV.
    pub fn quote_ivs(&self, fair_iv: f64, net_vega: f64) -> (f64, f64) {
        let (db, da) = self.vol_distances(net_vega);
        (fair_iv - db, fair_iv + da)
    }

    /// IV quotes around the surface fair **with the vanna–volga
    /// overhedge** layered on top (Castagna–Mercurio): the fair IV is
    /// shifted by the cost of hedging the fill's second-order greeks
    /// with the 25Δ wing instruments, then the GLFT vol distances are
    /// applied — the quoting stack used by the VolSurface strategy.
    pub fn quote_ivs_vv(
        &self,
        kind: Kind,
        s: f64,
        k: f64,
        t: f64,
        surface: &SsviSurface,
        net_vega: f64,
        vv: &VannaVolga,
    ) -> (f64, f64, f64) {
        let fair = surface.iv((k / s).ln(), t);
        let oh = vv.overhedge(kind, k, fair);
        let centered = (fair + oh.vol_shift).max(0.03);
        let (iv_b, iv_a) = self.quote_ivs(centered, net_vega);
        (iv_b, iv_a, oh.vol_shift)
    }

    /// Premium quotes for one strike off the surface.
    pub fn quote_premiums(
        &self,
        kind: Kind,
        s: f64,
        k: f64,
        t: f64,
        surface: &SsviSurface,
        net_vega: f64,
    ) -> (f64, f64) {
        let fair = surface.iv((k / s).ln(), t);
        let (iv_b, iv_a) = self.quote_ivs(fair, net_vega);
        (
            crate::greeks::price(kind, s, k, 0.0, 0.0, iv_b, t),
            crate::greeks::price(kind, s, k, 0.0, 0.0, iv_a, t),
        )
    }

    /// Net book greeks for a ladder of positions `(kind, k, lots)`
    /// against the surface.
    pub fn book_greeks(
        &self,
        positions: &[(Kind, f64, f64)],
        s: f64,
        t: f64,
        surface: &SsviSurface,
    ) -> (f64, f64, f64) {
        // (net delta, net gamma, net vega)
        let mut delta = 0.0;
        let mut gamma = 0.0;
        let mut vega = 0.0;
        for &(kind, k, lots) in positions {
            let iv = surface.iv((k / s).ln(), t);
            let g = crate::greeks::full_greeks(kind, s, k, 0.0, 0.0, iv, t);
            delta += g.delta * lots;
            gamma += g.gamma * lots;
            vega += g.vega * lots;
        }
        (delta, gamma, vega)
    }
}

/// Delta-hedge no-trade band.
#[derive(Clone, Copy, Debug)]
pub struct HedgeCadence {
    /// Band half-width in lots.
    pub band_lots: f64,
}

impl HedgeCadence {
    /// Closed form `H* = (3 c / (γ S²))^{1/3}` with `c` the linear cost
    /// per lot ($/lot: fee fraction + half-spread, times price times lot
    /// size), `γ` the CARA risk aversion, `S` the price per lot.
    pub fn new(cost_per_lot: f64, gamma: f64, price_per_lot: f64) -> HedgeCadence {
        let c = cost_per_lot.max(0.0);
        let g = gamma.max(1e-9);
        let s = price_per_lot.max(1e-9);
        HedgeCadence {
            band_lots: (3.0 * c / (g * s * s)).powf(1.0 / 3.0),
        }
    }

    /// Hedge order (signed lots to trade, restoring flat), if outside the
    /// band. `unhedged` is the signed unhedged delta in lots.
    pub fn should_hedge(&self, unhedged: f64) -> Option<i64> {
        if unhedged.abs() < self.band_lots {
            None
        } else {
            Some(-unhedged.round() as i64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal xorshift normal RNG (tests only).
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.uniform().max(1e-12);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    #[test]
    fn vol_distances_skew_with_vega_inventory() {
        // Realistic scale: sigma_vol ~ 1.2 annualized, expressed per
        // sqrt(second); vega_per_fill = $1000 of vega per request.
        let mm = OptMm {
            gamma: 5e-4,
            sigma_vol: 1.2 / (365.0f64 * 24.0 * 3600.0).sqrt(),
            kappa_v: 600.0,
            a_v: 0.1,
            vega_per_fill: 1000.0,
            fill_cost_vol: 0.0,
        };
        let (db0, da0) = mm.vol_distances(0.0);
        assert!(db0 > 0.0 && da0 > 0.0);
        assert!(db0 < 0.1 && da0 < 0.1, "sane vol spreads {db0}/{da0}");
        // Long vega: bid distance grows, ask distance shrinks.
        let (db_long, da_long) = mm.vol_distances(5.0 * mm.vega_per_fill);
        assert!(db_long > db0, "{db_long} vs {db0}");
        assert!(da_long < da0, "{da_long} vs {da0}");
        // Short vega: mirrored.
        let (db_short, da_short) = mm.vol_distances(-5.0 * mm.vega_per_fill);
        assert!(da_short > da0);
        assert!(db_short < db0);
        // Exact mirror symmetry of the GLFT skew at moderate inventory.
        let (db2, da2) = mm.vol_distances(2.0 * mm.vega_per_fill);
        let (dbm2, dam2) = mm.vol_distances(-2.0 * mm.vega_per_fill);
        assert!((db2 - dam2).abs() < 1e-9);
        assert!((da2 - dbm2).abs() < 1e-9);
    }

    #[test]
    fn quote_ivs_wrap_the_fair() {
        let mm = OptMm {
            gamma: 5e-4,
            sigma_vol: 1.2 / (365.0f64 * 24.0 * 3600.0).sqrt(),
            kappa_v: 600.0,
            a_v: 0.1,
            vega_per_fill: 1000.0,
            fill_cost_vol: 0.0,
        };
        let (b, a) = mm.quote_ivs(0.30, 0.0);
        assert!(b < 0.30 && a > 0.30);
        assert!(a - b > 0.0);
    }

    #[test]
    fn hedge_band_monotonicity() {
        // Band grows with cost, shrinks with risk aversion.
        let small = HedgeCadence::new(0.01, 1e-3, 100.0).band_lots;
        let big = HedgeCadence::new(0.10, 1e-3, 100.0).band_lots;
        assert!(big > small);
        let risky = HedgeCadence::new(0.01, 1e-2, 100.0).band_lots;
        assert!(risky < small);
        // Band scales with price (notional per lot).
        let pricey = HedgeCadence::new(0.01, 1e-3, 200.0).band_lots;
        assert!(pricey < small);
    }

    /// MC ground truth for the hedge band: simulate reflected-band delta
    /// hedging over dt steps; loss = trading cost + CARA risk; the
    /// empirical loss-rate must be minimized near H*.
    #[test]
    fn hedge_band_mc_hump() {
        let s = 100.0f64;
        let sigma = 0.02f64; // per sqrt(second)
        let gamma = 1e-3f64;
        let c = 0.02f64; // $ per lot traded
        let dt = 1.0f64;
        let n = 200_000usize;
        let band = HedgeCadence::new(c, gamma, s).band_lots;
        let eval = |h: f64, seed: u64| -> f64 {
            let mut rng = Rng(seed);
            let mut delta = 0.0f64;
            let mut cost = 0.0f64;
            let mut risk = 0.0f64;
            for _ in 0..n {
                delta += sigma * dt.sqrt() * rng.normal();
                if delta.abs() > h {
                    cost += c * delta.abs();
                    delta = 0.0;
                }
                risk += 0.5 * gamma * (delta * s * sigma).powi(2) * dt;
            }
            (cost + risk) / (n as f64 * dt)
        };
        let opt = eval(band, 42);
        let below = eval(band * 0.45, 42);
        let above = eval(band / 0.45, 42);
        assert!(
            opt <= below * 1.02 && opt <= above * 1.02,
            "H* not optimal: band={band} opt={opt} below={below} above={above}"
        );
    }

    /// MC validation of the vega-approximation quoting: GLFT-quoted vol
    /// spreads must beat a fixed spread (same q=0 width) under
    /// vol-of-vol risk, in entropic certainty-equivalent terms.
    #[test]
    fn glft_vega_quotes_beat_fixed_spread_mc() {
        let mm = OptMm {
            gamma: 2e-4,
            sigma_vol: 1.2 / (365.0f64 * 24.0 * 3600.0).sqrt(), // per sqrt(second), annualized 1.2
            kappa_v: 400.0,
            a_v: 0.02,
            vega_per_fill: 1000.0,
            fill_cost_vol: 0.0,
        };
        let dt = 30.0f64; // seconds per step
        let n = 60_000usize;
        let vol0 = 0.30f64;
        // Simulate one shared path set: IV as GBM with vol sigma_vol,
        // requests Poisson with intensity A e^{-kappa delta}, direction
        // symmetric. Compare strategies on identical seeds.
        let run = |adaptive: bool, seed: u64| -> f64 {
            let mut rng = Rng(seed);
            let mut iv = vol0;
            let mut q = 0.0f64; // fills
            let mut cash = 0.0f64; // in vega-dollar units
            let (db0, da0) = mm.vol_distances(0.0);
            for _ in 0..n {
                iv *= (mm.sigma_vol * dt.sqrt() * rng.normal()).exp();
                iv = iv.clamp(0.05, 1.5);
                let fair = vol0; // client fair = anchor; we quote around fair
                let (db, da) = if adaptive {
                    mm.vol_distances(q * mm.vega_per_fill)
                } else {
                    (db0, da0)
                };
                // Poisson arrivals per side
                let lam_b = mm.a_v * (-mm.kappa_v * db).exp() * dt;
                let lam_a = mm.a_v * (-mm.kappa_v * da).exp() * dt;
                if rng.uniform() < lam_b.min(1.0) {
                    // client sells to our bid: we BUY vega at (fair - db)
                    cash -= mm.vega_per_fill * (fair - db);
                    q += 1.0;
                }
                if rng.uniform() < lam_a.min(1.0) {
                    cash += mm.vega_per_fill * (fair + da);
                    q -= 1.0;
                }
            }
            // Liquidate at the final IV.
            cash + q * mm.vega_per_fill * iv
        };
        
        
        let paths = 24;
        let mut w = Vec::new();
        for p in 0..paths {
            let g = run(true, 1000 + p as u64);
            let f = run(false, 1000 + p as u64);
            w.push((g, f));
        }
        // Entropic CE across paths.
        let ce = |xs: &[f64]| -> f64 {
            let mean = xs.iter().sum::<f64>() / xs.len() as f64;
            let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
            mean - 0.5 * mm.gamma * var
        };
        let ce_glft = ce(&w.iter().map(|&(g, _)| g).collect::<Vec<_>>());
        let ce_fixed = ce(&w.iter().map(|&(_, f)| f).collect::<Vec<_>>());
        assert!(
            ce_glft >= ce_fixed - 1e-6,
            "GLFT vega quotes CE {ce_glft} < fixed {ce_fixed}"
        );
    }
}
