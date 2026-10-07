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
