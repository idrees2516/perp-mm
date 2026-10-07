//! Vanna–Volga overhedge for option quoting.
//!
//! Castagna & Mercurio, "The vanna-volga method for implied
//! volatilities" (2007); Wystup, "Vanna-Volga and Market Data" (2009).
//!
//! The vega-approximation option MM ([`crate::optmm::OptMm`], from
//! Bergault & Guéant) quotes around a fair IV with a GLFT vol spread,
//! but ignores the second-order smile risk each fill adds: vanna
//! (`d²V/dS dσ`) and volga (`d²V/dσ²`). The vanna–volga method charges
//! the cost of hedging that exposure with the liquid wing instruments:
//!
//! - 25Δ risk reversal `RR = C(K_c, σ_c) − P(K_p, σ_p)` (hedges vanna),
//! - 25Δ butterfly `BF = C(K_c, σ_c) + P(K_p, σ_p) − 2·ATM` (hedges volga),
//!
//! each leg priced at its own smile vol. For a target option X with
//! greeks `(vanna_X, volga_X)` the 2×2 system
//!
//! ```text
//! [ vanna_rr  vanna_bf ] [ w_rr ]   [ vanna_X ]
//! [ volga_rr  volga_bf ] [ w_bf ] = [ volga_X ]
//! ```
//!
//! gives the hedge weights, and the overhedge premium is
//! `λ·(w_rr·ΔV_rr + w_bf·ΔV_bf)` where `ΔV_i` is instrument i's smile
//! premium over the flat ATM vol and `λ ∈ (0, 1]` is the
//! survival-probability damping (exponential in T here). Translated to
//! vol units through the option's own vega it shifts the fair IV before
//! the GLFT distances are applied — an inventory-independent
//! second-order skew adjustment on top of the vega-approximation quotes.
//!
//! The vol-unit shift is clamped to a few ticks / ~0.15 vol points:
//! the classic market overhedge is a small second-order correction,
//! never a first-order smile move.

use crate::greeks::{full_greeks, price};
use crate::ssvi::SsviSurface;
use models::options::Kind;

/// Overhedge vol-shift clamp (vol fractions; 0.0015 = 0.15 vol points).
pub const VV_VOL_CLAMP: f64 = 0.0015;

/// The three liquid wing instruments at one expiry.
#[derive(Clone, Debug)]
pub struct Wings {
    /// ATM strike (50Δ forward).
    pub k_atm: f64,
    /// 25Δ call strike.
    pub k_c25: f64,
    /// 25Δ put strike.
    pub k_p25: f64,
    /// Smile vols at the wing strikes.
    pub sig_c: f64,
    pub sig_p: f64,
    pub sig_atm: f64,
    /// ATM leg greeks (vega/vanna/volga at the ATM strike; zero smile
    /// premium — the ATM instrument prices itself).
    pub atm: WingGreeks,
    /// RR instrument: greeks and smile premium (each leg at its own vol).
    pub rr: WingGreeks,
    /// BF instrument: greeks and smile premium.
    pub bf: WingGreeks,
    /// Market RR / BF quotes in vol points (diagnostics).
    pub rr_iv_pts: f64,
    pub bf_iv_pts: f64,
}

/// Greeks + smile premium of one wing structure.
#[derive(Clone, Copy, Debug, Default)]
pub struct WingGreeks {
    pub vega: f64,
    pub vanna: f64,
    pub volga: f64,
    /// Premium of the structure at smile vols minus at the flat ATM vol.
    pub cost: f64,
}

/// One vanna–volga overhedge evaluation.
#[derive(Clone, Copy, Debug)]
pub struct Overhedge {
    /// Shift of the fair IV (vol fractions, clamped to ±VV_VOL_CLAMP).
    pub vol_shift: f64,
    pub w_rr: f64,
    pub w_bf: f64,
    /// Premium charge in price units (damped).
    pub charge: f64,
}

/// Vanna–volga quoting layer over an SSVI surface at one expiry.
#[derive(Clone, Debug)]
pub struct VannaVolga {
    pub s: f64,
    pub t: f64,
    pub r: f64,
    pub q: f64,
    /// Survival damping rate (1/years): λ = exp(−rate·T); rate = 0 → full.
    pub damping_rate: f64,
    pub wings: Wings,
}

impl VannaVolga {
    /// Build the wing instruments off `surface` at expiry `t`.
    pub fn new(s: f64, t: f64, surface: &SsviSurface, r: f64, q: f64, damping_rate: f64) -> VannaVolga {
        let sig_atm = surface.iv(0.0, t);
        let k_atm = strike_from_delta(Kind::Call, s, r, q, sig_atm, t, 0.5);
        let k_c25 = strike_from_delta(Kind::Call, s, r, q, sig_atm, t, 0.25);
        let k_p25 = strike_from_delta(Kind::Put, s, r, q, sig_atm, t, -0.25);
        let kc = (k_c25 / s).ln();
        let kp = (k_p25 / s).ln();
        let sig_c = surface.iv(kc, t);
        let sig_p = surface.iv(kp, t);
        let g_c = full_greeks(Kind::Call, s, k_c25, r, q, sig_c, t);
        let g_p = full_greeks(Kind::Put, s, k_p25, r, q, sig_p, t);
        let g_a = full_greeks(Kind::Call, s, k_atm, r, q, sig_atm, t);
        let dv_c = price(Kind::Call, s, k_c25, r, q, sig_c, t)
            - price(Kind::Call, s, k_c25, r, q, sig_atm, t);
        let dv_p = price(Kind::Put, s, k_p25, r, q, sig_p, t)
            - price(Kind::Put, s, k_p25, r, q, sig_atm, t);
        let wings = Wings {
            k_atm,
            k_c25,
            k_p25,
            sig_c,
            sig_p,
            sig_atm,
            atm: WingGreeks {
                vega: g_a.vega,
                vanna: g_a.vanna,
                volga: g_a.volga,
                cost: 0.0,
            },
            rr: WingGreeks {
                vega: g_c.vega - g_p.vega,
                vanna: g_c.vanna - g_p.vanna,
                volga: g_c.volga - g_p.volga,
                cost: dv_c - dv_p,
            },
            bf: WingGreeks {
                vega: g_c.vega + g_p.vega - 2.0 * g_a.vega,
                vanna: g_c.vanna + g_p.vanna - 2.0 * g_a.vanna,
                volga: g_c.volga + g_p.volga - 2.0 * g_a.volga,
                cost: dv_c + dv_p,
            },
            rr_iv_pts: (sig_c - sig_p) * 100.0,
            bf_iv_pts: 0.5 * (sig_c + sig_p - 2.0 * sig_atm) * 100.0,
        };
        VannaVolga { s, t, r, q, damping_rate, wings }
    }

    fn lambda(&self) -> f64 {
        (-self.damping_rate * self.t).exp()
    }

    /// Overhedge vol shift for an option `(kind, strike)` whose
    /// smile-fair vol is `sig_k`.
    ///
    /// Castagna–Mercurio construction: `V_vv(X) = V_bs(X, σ_atm) +
    /// w_rr·ΔV_rr + w_bf·ΔV_bf` vs the surface-consistent price
    /// `V_surf(X) = V_bs(X, σ_k)`. The overhedge charges the RESIDUAL
    /// `w_rr·ΔV_rr + w_bf·ΔV_bf − ΔV_X` — zero when the surface is
    /// vanna–volga consistent, a few ticks of premium when it is not.
    pub fn overhedge(&self, kind: Kind, strike: f64, sig_k: f64) -> Overhedge {
        let g = full_greeks(kind, self.s, strike, self.r, self.q, sig_k, self.t);
        if g.vega <= 1e-12 {
            return Overhedge { vol_shift: 0.0, w_rr: 0.0, w_bf: 0.0, charge: 0.0 };
        }
        // The classic three-instrument vanna–volga portfolio (Castagna–
        // Mercurio): ATM + 25Δ RR + 25Δ BF solve the full 3×3 system in
        // (vega, vanna, volga) — under which the ATM prices itself exactly.
        let (w_atm, w_rr, w_bf) = solve_3x3(
            (self.wings.atm.vega, self.wings.atm.vanna, self.wings.atm.volga),
            (self.wings.rr.vega, self.wings.rr.vanna, self.wings.rr.volga),
            (self.wings.bf.vega, self.wings.bf.vanna, self.wings.bf.volga),
            (g.vega, g.vanna, g.volga),
        );
        if ![w_atm, w_rr, w_bf].iter().all(|x| x.is_finite()) {
            return Overhedge { vol_shift: 0.0, w_rr: 0.0, w_bf: 0.0, charge: 0.0 };
        }
        let _ = w_atm; // ATM smile premium is zero by construction
        // the target's own smile premium over the flat ATM vol
        let dv_x = price(kind, self.s, strike, self.r, self.q, sig_k, self.t)
            - price(kind, self.s, strike, self.r, self.q, self.wings.sig_atm, self.t);
        let lam = self.lambda();
        let charge = lam * (w_rr * self.wings.rr.cost + w_bf * self.wings.bf.cost - dv_x);
        let vol_shift = (charge / g.vega).clamp(-VV_VOL_CLAMP, VV_VOL_CLAMP);
        Overhedge { vol_shift, w_rr, w_bf, charge }
    }

    /// Portfolio-level charge for net `(vanna, volga)` hedged at the
    /// wing instruments, translated to vol points through `atm_vega`
    /// (the book treated as one option at the ATM vega centroid).
    pub fn portfolio_charge(&self, net_vanna: f64, net_volga: f64, atm_vega: f64) -> (f64, f64, f64, f64) {
        let (w_atm, w_rr, w_bf) = solve_3x3(
            (self.wings.atm.vega, self.wings.atm.vanna, self.wings.atm.volga),
            (self.wings.rr.vega, self.wings.rr.vanna, self.wings.rr.volga),
            (self.wings.bf.vega, self.wings.bf.vanna, self.wings.bf.volga),
            (atm_vega, net_vanna, net_volga),
        );
        if ![w_atm, w_rr, w_bf].iter().all(|x| x.is_finite()) || atm_vega <= 1e-12 {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let lam = self.lambda();
        let charge = lam * (w_rr * self.wings.rr.cost + w_bf * self.wings.bf.cost);
        let vol_shift = (charge / atm_vega).clamp(-4.0 * VV_VOL_CLAMP, 4.0 * VV_VOL_CLAMP);
        (charge, vol_shift, w_rr, w_bf)
    }
}

/// Solve the full 3×3 (vega, vanna, volga) system for the weights
/// `(w_atm, w_rr, w_bf)` by Cramer's rule; `(0, 0, 0)` if singular.
/// Each instrument contributes one column `(vega, vanna, volga)`.
fn solve_3x3(
    atm: (f64, f64, f64),
    rr: (f64, f64, f64),
    bf: (f64, f64, f64),
    target: (f64, f64, f64),
) -> (f64, f64, f64) {
    let det3 = |a: (f64, f64, f64), b: (f64, f64, f64), c: (f64, f64, f64)| -> f64 {
        a.0 * (b.1 * c.2 - b.2 * c.1) - a.1 * (b.0 * c.2 - b.2 * c.0) + a.2 * (b.0 * c.1 - b.1 * c.0)
    };
    let det = det3(atm, rr, bf);
    if det.abs() < 1e-14 {
        return (0.0, 0.0, 0.0);
    }
    let w_atm = det3(target, rr, bf) / det;
    let w_rr = det3(atm, target, bf) / det;
    let w_bf = det3(atm, rr, target) / det;
    (w_atm, w_rr, w_bf)
}

/// Strike from a target spot-delta (bisection on N(d1)); used to build
/// the 25Δ wing strikes. `target_delta` is signed (puts negative).
pub fn strike_from_delta(kind: Kind, s: f64, r: f64, q: f64, sigma: f64, t: f64, target_delta: f64) -> f64 {
    let eq = (-q * t).exp();
    let nd1 = match kind {
        Kind::Call => target_delta / eq,
        Kind::Put => target_delta / eq + 1.0,
    }
    .clamp(1e-9, 1.0 - 1e-9);
    // invert N(d1) by bisection
    let (mut lo, mut hi) = (-8.0f64, 8.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if crate::ncdf_hi(mid) < nd1 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let d1 = 0.5 * (lo + hi);
    let sq = sigma * t.sqrt();
    let d2 = d1 - sq;
    // from d2: ln(F/K) = d2·σ√T + σ²T/2, F = S e^{(r−q)T}
    let ln_fk = d2 * sq + 0.5 * sigma * sigma * t;
    let fwd = s * ((r - q) * t).exp();
    fwd * (-ln_fk).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn surface() -> SsviSurface {
        SsviSurface::new(
            -0.7,
            1.0,
            0.5,
            &[
                (1.0 / 52.0, 0.25),
                (1.0 / 12.0, 0.27),
                (0.25, 0.30),
                (0.5, 0.32),
                (1.0, 0.34),
            ],
        )
    }

    #[test]
    fn wing_strikes_bracket_the_atm() {
        let sf = surface();
        let vv = VannaVolga::new(100.0, 0.25, &sf, 0.0, 0.0, 0.25);
        let w = &vv.wings;
        assert!(w.k_p25 < w.k_atm && w.k_atm < w.k_c25, "k_p {} k_atm {} k_c {}", w.k_p25, w.k_atm, w.k_c25);
        // equity skew: the put wing is the richer wing
        assert!(w.sig_p > w.sig_atm, "sig_p {} vs atm {}", w.sig_p, w.sig_atm);
        assert!(w.sig_c < w.sig_atm || w.sig_c < w.sig_p);
        // RR in vol points is negative for an equity skew
        assert!(w.rr_iv_pts < 0.0, "rr {} pts", w.rr_iv_pts);
        // the wing structures have non-trivial greeks
        assert!(w.rr.vanna.abs() > 1e-6 && w.bf.volga.abs() > 1e-6);
    }

    #[test]
    fn weights_reproduce_target_greeks() {
        // The 3×3 system is exact: the weighted instruments reproduce the
        // target (vega, vanna, volga) to machine precision.
        let sf = surface();
        let vv = VannaVolga::new(100.0, 0.25, &sf, 0.0, 0.0, 0.0);
        for &strike in &[85.0f64, 92.0, 100.0, 108.0, 115.0] {
            for &kind in &[Kind::Call, Kind::Put] {
                let sig = sf.iv((strike / 100.0).ln(), 0.25);
                let g = full_greeks(kind, 100.0, strike, 0.0, 0.0, sig, 0.25);
                let (w_atm, w_rr, w_bf) = solve_3x3(
                    (vv.wings.atm.vega, vv.wings.atm.vanna, vv.wings.atm.volga),
                    (vv.wings.rr.vega, vv.wings.rr.vanna, vv.wings.rr.volga),
                    (vv.wings.bf.vega, vv.wings.bf.vanna, vv.wings.bf.volga),
                    (g.vega, g.vanna, g.volga),
                );
                let vega_h = w_atm * vv.wings.atm.vega + w_rr * vv.wings.rr.vega + w_bf * vv.wings.bf.vega;
                let vanna_h = w_atm * vv.wings.atm.vanna + w_rr * vv.wings.rr.vanna + w_bf * vv.wings.bf.vanna;
                let volga_h = w_atm * vv.wings.atm.volga + w_rr * vv.wings.rr.volga + w_bf * vv.wings.bf.volga;
                assert!((vega_h - g.vega).abs() < 1e-8, "vega {vega_h} vs {}", g.vega);
                assert!((vanna_h - g.vanna).abs() < 1e-8, "vanna {vanna_h} vs {}", g.vanna);
                assert!((volga_h - g.volga).abs() < 1e-8, "volga {volga_h} vs {}", g.volga);
            }
        }
    }

    #[test]
    fn atm_instrument_prices_itself_exactly() {
        // At the ATM strike the overhedge is ~zero by construction.
        let sf = surface();
        let vv = VannaVolga::new(100.0, 0.25, &sf, 0.0, 0.0, 0.25);
        let oh = vv.overhedge(Kind::Call, vv.wings.k_atm, sf.iv(0.0, 0.25));
        assert!(oh.vol_shift.abs() < 1e-6, "atm shift {}", oh.vol_shift);
    }

    #[test]
    fn atm_overhedge_is_small_and_wings_bigger() {
        let sf = surface();
        let vv = VannaVolga::new(100.0, 0.25, &sf, 0.0, 0.0, 0.25);
        let k_atm = vv.wings.k_atm;
        let atm = vv.overhedge(Kind::Call, k_atm, sf.iv(0.0, 0.25));
        // ATM vanna ~ 0 by symmetry: the overhedge is tiny.
        assert!(atm.vol_shift.abs() < 0.0004, "atm shift {}", atm.vol_shift);
        // wings carry the second-order risk: their shifts are larger
        // (in magnitude, clamped).
        let wing_put = vv.overhedge(Kind::Put, vv.wings.k_p25, sf.iv((vv.wings.k_p25 / 100.0).ln(), 0.25));
        let wing_call = vv.overhedge(Kind::Call, vv.wings.k_c25, sf.iv((vv.wings.k_c25 / 100.0).ln(), 0.25));
        assert!(
            wing_put.vol_shift.abs() > atm.vol_shift.abs(),
            "put wing {} vs atm {}",
            wing_put.vol_shift,
            atm.vol_shift
        );
        assert!(
            wing_call.vol_shift.abs() > atm.vol_shift.abs(),
            "call wing {} vs atm {}",
            wing_call.vol_shift,
            atm.vol_shift
        );
    }

    #[test]
    fn overhedge_is_clamped_and_damped() {
        let sf = surface();
        let full = VannaVolga::new(100.0, 0.25, &sf, 0.0, 0.0, 0.0);
        let damped = VannaVolga::new(100.0, 0.25, &sf, 0.0, 0.0, 4.0);
        let k = full.wings.k_p25;
        let sig = sf.iv((k / 100.0).ln(), 0.25);
        let a = full.overhedge(Kind::Put, k, sig);
        let b = damped.overhedge(Kind::Put, k, sig);
        // never beyond the clamp
        assert!(a.vol_shift.abs() <= VV_VOL_CLAMP + 1e-12);
        // damping shrinks the charge toward zero
        assert!(b.charge.abs() < a.charge.abs(), "damped {} vs full {}", b.charge, a.charge);
    }

    #[test]
    fn portfolio_charge_scale_sanity() {
        let sf = surface();
        let vv = VannaVolga::new(100.0, 0.25, &sf, 0.0, 0.0, 0.25);
        let (charge, vol_shift, _, _) = vv.portfolio_charge(5.0, 40.0, 12.0);
        assert!(charge.abs() < 10.0, "charge {charge} (price units)");
        assert!(vol_shift.abs() <= 4.0 * VV_VOL_CLAMP + 1e-12);
        // zero target → zero charge (no hedge needed)
        let (c0, v0, _, _) = vv.portfolio_charge(0.0, 0.0, 0.0);
        assert!(c0.abs() < 1e-12 && v0.abs() < 1e-12);
        // a pure-ATM book charges ~nothing (it IS the ATM instrument)
        let (ca, va, _, _) = vv.portfolio_charge(vv.wings.atm.vanna, vv.wings.atm.volga, vv.wings.atm.vega);
        assert!(ca.abs() < 1e-9, "pure-ATM book charge {ca}");
        assert!(va.abs() < 1e-12, "pure-ATM book vol shift {va}");
    }

    #[test]
    fn strike_from_delta_round_trips() {
        // The implied spot-delta of the returned strike matches the target.
        let (s, r, q, sig, t) = (100.0, 0.02, 0.01, 0.3, 0.5);
        for &target in &[0.75, 0.5, 0.25] {
            let k = strike_from_delta(Kind::Call, s, r, q, sig, t, target);
            let g = full_greeks(Kind::Call, s, k, r, q, sig, t);
            assert!((g.delta - target).abs() < 1e-6, "delta {} vs {target}", g.delta);
        }
        let k = strike_from_delta(Kind::Put, s, r, q, sig, t, -0.25);
        let g = full_greeks(Kind::Put, s, k, r, q, sig, t);
        assert!((g.delta - (-0.25)).abs() < 1e-6, "put delta {} vs -0.25", g.delta);
    }
}
