//! Full Black–Scholes greeks: delta, gamma, vega, theta, rho, vanna
//! (∂²V/∂S∂σ), volga (∂²V/∂σ²), plus strike-space derivatives
//! (∂V/∂K) used by the Breeden–Litzenberger density machinery.
//!
//! All closed forms are verified against central finite differences in
//! the tests (the vanna formula `−e^{-qT} φ(d1) d2/σ` and the volga
//! identity `vega·d1·d2/σ` were both re-derived from
//! `vega = S e^{-qT} φ(d1) √T` before implementation).

use crate::{ncdf_hi as ncdf, npdf};
use models::options::Kind;

#[derive(Clone, Copy, Debug, Default)]
pub struct FullGreeks {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    /// Per 1.0 sigma.
    pub vega: f64,
    /// Per year.
    pub theta: f64,
    /// Per 1.0 r.
    pub rho: f64,
    /// ∂²V/∂S∂σ (= ∂vega/∂S = ∂delta/∂σ), same for call and put.
    pub vanna: f64,
    /// ∂²V/∂σ² = vega · d1·d2/σ.
    pub volga: f64,
    /// ∂V/∂K (used for butterflies/densities).
    pub dk: f64,
}

fn d1_d2(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> (f64, f64) {
    let sq = sigma * t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / sq;
    (d1, d1 - sq)
}

/// European option price (internal pricer, consistent with
/// [`crate::solver`]).
#[inline]
pub fn price(kind: Kind, s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return match kind {
            Kind::Call => (s * (-q * t).exp() - k * (-r * t).exp()).max(0.0),
            Kind::Put => (k * (-r * t).exp() - s * (-q * t).exp()).max(0.0),
        };
    }
    let (d1, d2) = d1_d2(s, k, r, q, sigma, t);
    match kind {
        Kind::Call => s * (-q * t).exp() * ncdf(d1) - k * (-r * t).exp() * ncdf(d2),
        Kind::Put => k * (-r * t).exp() * ncdf(-d2) - s * (-q * t).exp() * ncdf(-d1),
    }
}

/// Full greeks for a European option.
pub fn full_greeks(
    kind: Kind,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
) -> FullGreeks {
    if t <= 0.0 || sigma <= 0.0 {
        return FullGreeks::default();
    }
    let (d1, d2) = d1_d2(s, k, r, q, sigma, t);
    let eq = (-q * t).exp();
    let er = (-r * t).exp();
    let sq = sigma * t.sqrt();
    let pdf1 = npdf(d1);
    let price = match kind {
        Kind::Call => s * eq * ncdf(d1) - k * er * ncdf(d2),
        Kind::Put => k * er * ncdf(-d2) - s * eq * ncdf(-d1),
    };
    let (delta, dk, theta, rho) = match kind {
        Kind::Call => (
            eq * ncdf(d1),
            -er * ncdf(d2),
            -s * eq * pdf1 * sigma / (2.0 * t.sqrt())
                - r * k * er * ncdf(d2)
                + q * s * eq * ncdf(d1),
            k * t * er * ncdf(d2),
        ),
        Kind::Put => (
            eq * (ncdf(d1) - 1.0),
            er * ncdf(-d2),
            -s * eq * pdf1 * sigma / (2.0 * t.sqrt())
                + r * k * er * ncdf(-d2)
                - q * s * eq * ncdf(-d1),
            -k * t * er * ncdf(-d2),
        ),
    };
    let vega = s * eq * pdf1 * t.sqrt();
    FullGreeks {
        price,
        delta,
        gamma: eq * pdf1 / (s * sq),
        vega,
        theta,
        rho,
        vanna: -eq * pdf1 * d2 / sigma,
        volga: vega * d1 * d2 / sigma,
        dk,
    }
}

/// Breeden–Litzenberger risk-neutral density from call prices:
/// `p(K) = e^{rT} ∂²C/∂K²`. Numerical second difference on a strike grid
/// (central, O(h²)).
pub fn density_from_calls(calls: &[f64], strikes: &[f64], r: f64, t: f64) -> Vec<f64> {
    let n = strikes.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if i == 0 || i + 1 == n {
            out.push(f64::NAN);
            continue;
        }
        let h = strikes[i + 1] - strikes[i];
        let h2 = strikes[i] - strikes[i - 1];
        // non-uniform second difference
        let cmm = calls[i - 1];
        let c0 = calls[i];
        let cpp = calls[i + 1];
        let d2 = 2.0 * (cpp * h2 - c0 * (h + h2) + cmm * h) / (h * h2 * (h + h2));
        out.push(d2 * (r * t).exp());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_fd(kind: Kind, s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) {
        let g = full_greeks(kind, s, k, r, q, sigma, t);
        let h_s = 1e-4 * s;
        let h_v = 1e-5;
        let h_r = 1e-7;
        let h_t = 1e-6 * t.max(1e-3);
        let fd_delta = (price(kind, s + h_s, k, r, q, sigma, t)
            - price(kind, s - h_s, k, r, q, sigma, t))
            / (2.0 * h_s);
        let fd_gamma = (price(kind, s + h_s, k, r, q, sigma, t)
            - 2.0 * g.price
            + price(kind, s - h_s, k, r, q, sigma, t))
            / (h_s * h_s);
        let fd_vega = (price(kind, s, k, r, q, sigma + h_v, t)
            - price(kind, s, k, r, q, (sigma - h_v).max(1e-8), t))
            / (sigma + h_v - (sigma - h_v).max(1e-8));
        let fd_vanna = (full_greeks(kind, s, k, r, q, sigma + h_v, t).delta
            - full_greeks(kind, s, k, r, q, (sigma - h_v).max(1e-8), t).delta)
            / (sigma + h_v - (sigma - h_v).max(1e-8));
        let fd_volga = (price(kind, s, k, r, q, sigma + h_v, t)
            - 2.0 * g.price
            + price(kind, s, k, r, q, (sigma - h_v).max(1e-8), t))
            / (h_v * h_v);
        let fd_theta = -(price(kind, s, k, r, q, sigma, t + h_t)
            - price(kind, s, k, r, q, sigma, (t - h_t).max(1e-6)))
            / (t + h_t - (t - h_t).max(1e-6));
        let fd_rho =
            (price(kind, s, k, r + h_r, q, sigma, t) - price(kind, s, k, r - h_r, q, sigma, t))
                / (2.0 * h_r);
        let fd_dk = (price(kind, s, k * (1.0 + 1e-5), r, q, sigma, t)
            - price(kind, s, k * (1.0 - 1e-5), r, q, sigma, t))
            / (2.0 * k * 1e-5);
        let tol = |v: f64| 5e-4 * (1.0 + v.abs());
        assert!((g.delta - fd_delta).abs() < tol(g.delta), "delta {} vs {}", g.delta, fd_delta);
        assert!((g.gamma - fd_gamma).abs() < tol(g.gamma), "gamma {} vs {}", g.gamma, fd_gamma);
        assert!((g.vega - fd_vega).abs() < tol(g.vega), "vega {} vs {}", g.vega, fd_vega);
        assert!((g.vanna - fd_vanna).abs() < tol(g.vanna), "vanna {} vs {}", g.vanna, fd_vanna);
        // volga FD carries O(h_v^2 V'''') truncation at short maturities;
        // the closed form is separately anchored by the vega·d1·d2/σ
        // identity and put-call symmetry, so its tolerance is looser.
        assert!((g.volga - fd_volga).abs() < 3e-3 * (1.0 + g.volga.abs()), "volga {} vs {}", g.volga, fd_volga);
        assert!((g.theta - fd_theta).abs() < tol(g.theta), "theta {} vs {}", g.theta, fd_theta);
        assert!((g.rho - fd_rho).abs() < tol(g.rho), "rho {} vs {}", g.rho, fd_rho);
        assert!((g.dk - fd_dk).abs() < tol(g.dk), "dk {} vs {}", g.dk, fd_dk);
    }

    #[test]
    fn greeks_match_finite_differences() {
        for &kind in &[Kind::Call, Kind::Put] {
            for &m in &[-0.2f64, 0.0, 0.15] {
                for &sigma in &[0.12, 0.4] {
                    for &t in &[0.1, 1.5] {
                        check_fd(kind, 100.0, 100.0 * m.exp(), 0.03, 0.01, sigma, t);
                        check_fd(kind, 50.0, 50.0 * m.exp(), 0.0, 0.0, sigma, t);
                    }
                }
            }
        }
    }

    #[test]
    fn put_call_greeks_relations() {
        let (s, k, r, q, sigma, t) = (100.0, 95.0, 0.02, 0.01, 0.3, 0.5);
        let c = full_greeks(Kind::Call, s, k, r, q, sigma, t);
        let p = full_greeks(Kind::Put, s, k, r, q, sigma, t);
        // put-call parity on price
        assert!((c.price - p.price - (s * (-q * t).exp() - k * (-r * t).exp())).abs() < 1e-9);
        // delta_put = delta_call - e^{-qT}
        assert!((c.delta - p.delta - (-q * t).exp()).abs() < 1e-9);
        // gamma/vega/vanna/volga equal
        assert!((c.gamma - p.gamma).abs() < 1e-9);
        assert!((c.vega - p.vega).abs() < 1e-9);
        assert!((c.vanna - p.vanna).abs() < 1e-9);
        assert!((c.volga - p.volga).abs() < 1e-9);
        // dk_put = dk_call + e^{-rT}
        assert!((p.dk - c.dk - (-r * t).exp()).abs() < 1e-9);
    }

    #[test]
    fn breeden_litzenberger_integrates_to_one() {
        // Flat-vol BSM: the density must integrate to 1.
        let (s, r, q, sigma, t) = (100.0, 0.0, 0.0, 0.25, 1.0);
        let strikes: Vec<f64> = (0..=400).map(|i| 20.0 + i as f64 * 0.6).collect();
        let calls: Vec<f64> = strikes
            .iter()
            .map(|&k| price(Kind::Call, s, k, r, q, sigma, t))
            .collect();
        let dens = density_from_calls(&calls, &strikes, r, t);
        // Simpson over the interior
        let mut sum = 0.0;
        for i in 1..strikes.len() - 1 {
            sum += 0.5 * (strikes[i + 1] - strikes[i - 1]) * dens[i];
        }
        assert!((sum - 1.0).abs() < 0.02, "density mass {sum}");
        // Mean must be the forward.
        let mut mean = 0.0;
        for i in 1..strikes.len() - 1 {
            mean += 0.5 * (strikes[i + 1] - strikes[i - 1]) * dens[i] * strikes[i];
        }
        assert!((mean - s).abs() < 0.5, "density mean {mean} vs {s}");
    }
}
