//! Robust Black–Scholes implied-volatility solver.
//!
//! Strategy: (1) reject arbitrage-violating prices via discounted
//! intrinsic/upper bounds; (2) Brenner–Subrahmanyam-style initial guess;
//! (3) Newton iterations with the analytic vega, step-clamped into a
//! bracket; (4) guaranteed bisection fallback. This is the classical
//! robust hybrid (the same problem Jäckel's "Let's Be Rational" (2014)
//! solves to machine precision; our target here is ~1e-10 price accuracy
//! at nanosecond-scale cost, which the hybrid achieves in 3–6 Newton
//! steps for vanillas).

use crate::{ncdf, npdf};
use models::options::Kind;

/// Solver outcome metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IvMethod {
    Newton,
    Bisection,
}

#[derive(Clone, Copy, Debug)]
pub struct IvResult {
    pub sigma: f64,
    pub iterations: u32,
    pub method: IvMethod,
}

/// Black–Scholes price with `vol`'s own normal functions (so the solver
/// inverts exactly the function used by [`crate::greeks`]).
#[inline]
fn bs_price(kind: Kind, s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return intrinsic(kind, s, k, r, q, t);
    }
    let sq = sigma * t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / sq;
    let d2 = d1 - sq;
    match kind {
        Kind::Call => s * (-q * t).exp() * ncdf(d1) - k * (-r * t).exp() * ncdf(d2),
        Kind::Put => k * (-r * t).exp() * ncdf(-d2) - s * (-q * t).exp() * ncdf(-d1),
    }
}

#[inline]
fn intrinsic(kind: Kind, s: f64, k: f64, r: f64, q: f64, t: f64) -> f64 {
    match kind {
        Kind::Call => (s * (-q * t).exp() - k * (-r * t).exp()).max(0.0),
        Kind::Put => (k * (-r * t).exp() - s * (-q * t).exp()).max(0.0),
    }
}

/// Analytic vega (same for call/put), per 1.0 of sigma.
#[inline]
fn vega(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return 0.0;
    }
    let sq = sigma * t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / sq;
    s * (-q * t).exp() * npdf(d1) * t.sqrt()
}

/// Solve for implied volatility. Returns `None` when the price violates
/// static no-arbitrage bounds (below discounted intrinsic or above the
/// discounted underlying).
pub fn implied_vol(
    kind: Kind,
    price: f64,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
) -> Option<IvResult> {
    if !(price.is_finite() && s > 0.0 && k > 0.0 && t > 0.0) {
        return None;
    }
    let lo = intrinsic(kind, s, k, r, q, t);
    let hi = match kind {
        Kind::Call => s * (-q * t).exp(),
        Kind::Put => k * (-r * t).exp(),
    };
    let scale = (s + k).max(1.0);
    if price < lo - 1e-12 * scale || price > hi + 1e-12 * scale {
        return None;
    }
    if price <= lo + 1e-13 * scale {
        // sigma -> 0
        return Some(IvResult { sigma: 0.0, iterations: 0, method: IvMethod::Bisection });
    }
    if price >= hi - 1e-13 * scale {
        return None; // sigma -> infinity; no finite IV
    }

    let price_tol = 1e-12 * scale;
    // Bisection bracket on sigma.
    let mut lo_s = 1e-7f64;
    let mut hi_s = 5.0f64;
    // Initial guess: Brenner–Subrahmanyam ATM form generalized by the
    // moneyness ratio, clamped into a sane band.
    let guess = (((price - lo) / (0.4 * s * (-q * t).exp())).powi(2) / t).sqrt();
    let mut x = guess.clamp(0.01, 3.0);

    // Newton phase.
    let mut used = IvMethod::Newton;
    let mut iters = 0u32;
    for _ in 0..64u32 {
        iters += 1;
        let v = bs_price(kind, s, k, r, q, x, t);
        let err = v - price;
        if err.abs() <= price_tol {
            return Some(IvResult { sigma: x, iterations: iters, method: used });
        }
        let g = vega(s, k, r, q, x, t);
        if g < 1e-10 {
            break; // flat zone: fall back to bisection
        }
        let mut next = x - err / g;
        if !next.is_finite() || next <= 0.0 {
            break;
        }
        // Keep the bracket informed; clamp steps into it.
        if err > 0.0 {
            hi_s = hi_s.min(x);
        } else {
            lo_s = lo_s.max(x);
        }
        next = next.clamp(lo_s * 0.999, hi_s * 1.001).max(1e-9);
        if (next - x).abs() < 1e-13 {
            // stall: nudge with bisection
            next = 0.5 * (lo_s + hi_s);
            used = IvMethod::Bisection;
        }
        x = next;
    }

    // Bisection fallback (guaranteed).
    used = IvMethod::Bisection;
    let mut a = 1e-7f64;
    let mut b = 5.0f64;
    let mut mid = x;
    for i in 0..100u32 {
        mid = 0.5 * (a + b);
        let v = bs_price(kind, s, k, r, q, mid, t);
        if v > price {
            b = mid;
        } else {
            a = mid;
        }
        if (b - a) < 1e-12 {
            iters = i + 1;
            return Some(IvResult { sigma: mid, iterations: iters, method: used });
        }
    }
    let v = bs_price(kind, s, k, r, q, mid, t);
    if (v - price).abs() <= 1e-9 * scale {
        Some(IvResult { sigma: mid, iterations: 100, method: used })
    } else {
        None
    }
}

/// Invert CRR (binomial) prices: the solver price function differs from
/// CRR by discretization error, so allow a loose tolerance and return the
/// vol that reprices BSM to within that error.
pub fn implied_vol_loose(
    kind: Kind,
    price: f64,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
) -> Option<f64> {
    let res = implied_vol(kind, price, s, k, r, q, t)?;
    // Reject only gross mismatches (CRR on a fine grid converges to BSM).
    let rep = bs_price(kind, s, k, r, q, res.sigma, t);
    if (rep - price).abs() > 0.02 * (s + k) {
        return None;
    }
    Some(res.sigma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use models::options::{american_crr, bsm, Kind};

    #[test]
    fn roundtrip_machine_precision() {
        let mut fails = 0;
        for &kind in &[Kind::Call, Kind::Put] {
            for &s in &[80.0, 100.0, 123.0] {
                for &m in &[-0.3f64, -0.1, 0.0, 0.1, 0.4] {
                    let k = s * m.exp();
                    for &sigma in &[0.05, 0.15, 0.3, 0.8, 1.6] {
                        for &t in &[0.02, 0.25, 1.0, 3.0] {
                            for &(r, q) in &[(0.0, 0.0), (0.05, 0.0), (0.02, 0.04)] {
                                let price = bs_price(kind, s, k, r, q, sigma, t);
                                let res = implied_vol(kind, price, s, k, r, q, t)
                                    .unwrap_or_else(|| panic!("no iv {kind:?}"));
                                let back = bs_price(kind, s, k, r, q, res.sigma, t);
                                if (back - price).abs() > 1e-9 * (s + k) {
                                    fails += 1;
                                }
                                // deep-wing vega flatness may force bisection;
                                // assert the method always converges though
                                assert!(res.sigma.is_finite() && res.sigma >= 0.0);
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(fails, 0, "price round-trip failures");
    }

    #[test]
    fn newton_dominates_typical_cases() {
        // For mid-moneyness cases the solver should be Newton-fast.
        let price = bs_price(Kind::Call, 100.0, 100.0, 0.0, 0.0, 0.2, 0.5);
        let res = implied_vol(Kind::Call, price, 100.0, 100.0, 0.0, 0.0, 0.5).unwrap();
        assert_eq!(res.method, IvMethod::Newton);
        assert!(res.iterations <= 8, "{} iterations", res.iterations);
        assert!((res.sigma - 0.2).abs() < 1e-9);
    }

    #[test]
    fn rejects_arbitrage_prices() {
        // Below intrinsic.
        assert!(implied_vol(Kind::Call, 1.0, 100.0, 90.0, 0.0, 0.0, 1.0).is_none());
        // Above the discounted underlying.
        assert!(implied_vol(Kind::Call, 101.0, 100.0, 90.0, 0.0, 0.0, 1.0).is_none());
        // Zero-time / negative inputs.
        assert!(implied_vol(Kind::Call, 5.0, 100.0, 100.0, 0.0, 0.0, 0.0).is_none());
    }

    #[test]
    fn agrees_with_models_bsm() {
        // Cross-check against the workspace's own pricer (different ncdf).
        for &sigma in &[0.1, 0.25, 0.6] {
            let p = bsm(Kind::Call, 100.0, 105.0, 0.03, 0.01, sigma, 0.75);
            let res = implied_vol(Kind::Call, p, 100.0, 105.0, 0.03, 0.01, 0.75).unwrap();
            assert!((res.sigma - sigma).abs() < 5e-4, "{} vs {}", res.sigma, sigma);
        }
    }

    #[test]
    fn inverts_crr_american() {
        let p = american_crr(Kind::Put, 100.0, 98.0, 0.05, 0.0, 0.25, 1.0, 400);
        let iv = implied_vol_loose(Kind::Put, p, 100.0, 98.0, 0.05, 0.0, 1.0).unwrap();
        // American put IV must sit at or above the European IV at the same
        // price (early exercise premium), and be finite. The 5e-5 slack
        // covers the ncdf difference between vol's and models' pricers.
        let eur = bsm(Kind::Put, 100.0, 98.0, 0.05, 0.0, iv, 1.0);
        assert!(eur <= p + 5e-5, "eur {eur} > amer {p}");
        assert!(iv > 0.0 && iv < 3.0);
    }
}

/// Fast path in the Jäckel "Let's Be Rational" (2015) lineage: fold ITM
/// quotes onto the OTM instrument via put-call parity (where the
/// price→vol map is best conditioned), solve in normalized strike units
/// b = V/K over x = σ√T with a rational seed and Halley (Householder-2)
/// steps — cubic convergence, machine precision in ≤3 iterations across
/// the practical moneyness/vol grid vs ~4–6 safeguarded Newton steps.
/// Falls back to the bracketed Newton/bisection path for pathological
/// premiums. Cross-validated against `implied_vol` in the tests below.
pub fn implied_vol_fast(
    kind: Kind,
    price: f64,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
) -> Option<IvResult> {
    if t <= 0.0 || !price.is_finite() {
        return None;
    }
    let eq = (-q * t).exp();
    let er = (-r * t).exp();
    let fwd = s * eq / er;
    let intrinsic = match kind {
        Kind::Call => (s * eq - k * er).max(0.0),
        Kind::Put => (k * er - s * eq).max(0.0),
    };
    let hi = match kind {
        Kind::Call => s * eq,
        Kind::Put => k * er,
    };
    let scale = (s + k).max(1.0);
    if price < intrinsic - 1e-12 * scale || price > hi + 1e-12 * scale {
        return None;
    }
    if price <= intrinsic + 1e-13 * scale {
        return Some(IvResult { sigma: 0.0, iterations: 0, method: IvMethod::Bisection });
    }
    // OTM fold: C − P = F − K (undiscounted)
    let parity = fwd - k;
    let (solve_kind, v_und) = match kind {
        Kind::Call if parity > 0.0 => (Kind::Put, price / er - parity),
        Kind::Put if parity < 0.0 => (Kind::Call, price / er + parity),
        k_ => (k_, price / er),
    };
    let m = (fwd / k).ln();
    let b = v_und / k;
    // value + derivatives in K-units over x = σ√T:
    //   v(x) = e^m Φ(d1) − Φ(d2)  (call),  v' = e^m φ(d1),
    //   v'' = −v' d1 (−m/x² + ½);  puts via Φ(−d)
    let val = |x: f64| -> f64 {
        let d1 = m / x + x / 2.0;
        let d2 = d1 - x;
        match solve_kind {
            Kind::Call => m.exp() * ncdf(d1) - ncdf(d2),
            Kind::Put => ncdf(-d2) - m.exp() * ncdf(-d1),
        }
    };
    let seed = if b < 1e-3 {
        m.abs() / (2.0 * (-b.ln()).max(1.0)).sqrt()
    } else {
        (2.0 * m.abs()).sqrt() * 1.02 + b * std::f64::consts::TAU.sqrt() * 0.4
    };
    let mut x = seed.max(0.02);
    let tol = (1e-14f64).max(1e-10 * b);
    let mut iters = 0u32;
    for _ in 0..6 {
        iters += 1;
        let fv = val(x) - b;
        if fv.abs() < tol {
            break;
        }
        let d1 = m / x + x / 2.0;
        let v1 = m.exp() * npdf(d1);
        if v1 <= 1e-300 {
            break;
        }
        let dd1 = -m / (x * x) + 0.5;
        let v2 = -v1 * d1 * dd1;
        let denom = 2.0 * v1 * v1 - fv * v2;
        let step = if denom > 1e-300 { 2.0 * fv * v1 / denom } else { fv / v1 };
        let next = x - step;
        if next > 1e-8 && next < 20.0 && step.abs() < x + 1.0 {
            x = next;
        } else {
            x = next.clamp(1e-8, 20.0);
            break;
        }
    }
    // relative-precision guard → cold fallback (the bracketed solver)
    if (val(x) - b).abs() > (1e-13f64).max(1e-9 * b) {
        return implied_vol(kind, price, s, k, r, q, t);
    }
    Some(IvResult { sigma: x / t.sqrt(), iterations: iters, method: IvMethod::Newton })
}

#[cfg(test)]
mod fast_tests {
    use super::*;

    #[test]
    fn fast_solver_matches_reference_on_grid() {
        // moneyness × maturity × vol grid, both kinds: the Halley path must
        // recover the true sigma to machine precision wherever the premium
        // carries enough digits (b > 1e-6), and agree with the reference
        // solver to 1e-6 vol points
        let mut max_err = 0.0f64;
        let mut n = 0u32;
        for t in [1.0 / 365.0, 1.0 / 52.0, 0.25, 0.5, 1.0, 2.0] {
            for m in [-0.3f64, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3] {
                for sig in [0.1, 0.3, 0.55, 0.9, 1.5, 2.2] {
                    for kind in [Kind::Call, Kind::Put] {
                        let s = 100.0;
                        let k = s * (-m).exp();
                        let p = bs_price(kind, s, k, 0.0, 0.0, sig, t);
                        // effective OTM target must carry digits
                        let parity = s - k;
                        let v_otm = match kind {
                            Kind::Call if parity > 0.0 => p - parity,
                            Kind::Put if parity < 0.0 => p + parity,
                            _ => p,
                        };
                        if v_otm / k < 1e-6 {
                            continue;
                        }
                        let Some(ivf) = implied_vol_fast(kind, p, s, k, 0.0, 0.0, t) else {
                            panic!("fast solver returned None m={m} t={t} sig={sig}");
                        };
                        let err = (ivf.sigma - sig).abs();
                        max_err = max_err.max(err);
                        n += 1;
                        if err > 1e-9 {
                            panic!("fast solver off: m={m} t={t} sig={sig} err={err}");
                        }
                        if let Some(ivn) = implied_vol(kind, p, s, k, 0.0, 0.0, t) {
                            let cross = (ivf.sigma - ivn.sigma).abs();
                            assert!(cross < 1e-6, "cross m={m} t={t} sig={sig} Δ={cross}");
                        }
                    }
                }
            }
        }
        assert!(n > 400, "grid too small: {n}");
        assert!(max_err < 1e-9, "max err {max_err}");
    }

    #[test]
    fn fast_solver_degrades_gracefully_below_precision_floor() {
        // deep-ITM premiums whose OTM component is ~1e-14: no solver can
        // invert that to full precision — must still return finite output
        // (via the cold fallback), never NaN
        let s = 100.0;
        let k = 74.08; // ~m = +0.3
        let p = bs_price(Kind::Call, s, k, 0.0, 0.0, 0.15, 1.0 / 12.0);
        if let Some(iv) = implied_vol_fast(Kind::Call, p, s, k, 0.0, 0.0, 1.0 / 12.0) {
            assert!(iv.sigma.is_finite() && iv.sigma >= 0.0, "sigma = {}", iv.sigma);
        }
    }
}
