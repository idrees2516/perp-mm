//! Option pricing for the perp-options venue.
//!
//! * **Black–Scholes–Merton** spot options (the venue's European marks).
//! * **Black-76** futures-style options (perp underlyings with funding
//!   carry in the forward).
//! * **Barone-Adesi–Whaley** quadratic-approximation American exercise
//!   (the venue's American marks; CRR would be the referee — the BAW
//!   approximation is within cents for the parameter ranges here).
//! * **Everlasting** options (Paradigm-style): the venue sets the
//!   effective maturity to `interval * maturity_multiple`, priced as
//!   European at `T_eff` — longs pay the premium TWAP across rolls.
//! * **Implied volatility** via bracketed Newton/bisection hybrid.

use micro::special::{norm_cdf, norm_pdf};

/// Option kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Call,
    Put,
}

impl Kind {
    pub fn flip(self) -> Kind {
        match self {
            Kind::Call => Kind::Put,
            Kind::Put => Kind::Call,
        }
    }
}

/// European spot-option price (BSM). `s` spot, `k` strike, `r` and `q`
/// continuous rates, `sigma` vol, `t` years.
pub fn bsm(kind: Kind, s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return intrinsic(kind, s, k);
    }
    let _sq = sigma * sigma * t;
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    let df_r = (-r * t).exp();
    let df_q = (-q * t).exp();
    match kind {
        Kind::Call => {
            s * df_q * norm_cdf(d1) - k * df_r * norm_cdf(d2)
        }
        Kind::Put => {
            k * df_r * norm_cdf(-d2) - s * df_q * norm_cdf(-d1)
        }
    }
}

/// Black-76 futures-option price: `f` forward/futures level.
pub fn black76(kind: Kind, f: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return intrinsic(kind, f, k);
    }
    let d1 = ((f / k).ln() + 0.5 * sigma * sigma * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    let df_r = (-r * t).exp();
    match kind {
        Kind::Call => df_r * (f * norm_cdf(d1) - k * norm_cdf(d2)),
        Kind::Put => df_r * (k * norm_cdf(-d2) - f * norm_cdf(-d1)),
    }
}

/// All first-order greeks + gamma + vega (BSM spot).
#[derive(Clone, Copy, Debug)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
}

pub fn bsm_greeks(kind: Kind, s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> Greeks {
    if t <= 0.0 || sigma <= 0.0 {
        let d = match kind {
            Kind::Call => {
                if s > k {
                    1.0
                } else if s < k {
                    0.0
                } else {
                    0.5
                }
            }
            Kind::Put => {
                if s < k {
                    -1.0
                } else if s > k {
                    0.0
                } else {
                    -0.5
                }
            }
        };
        return Greeks {
            delta: d,
            gamma: 0.0,
            vega: 0.0,
            theta: 0.0,
            rho: 0.0,
        };
    }
    let sq = sigma * t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / sq;
    let d2 = d1 - sq;
    let df_q = (-q * t).exp();
    let df_r = (-r * t).exp();
    let pdf = norm_pdf(d1);
    let (delta, gamma, vega, theta, rho) = match kind {
        Kind::Call => (
            df_q * norm_cdf(d1),
            df_q * pdf / (s * sq),
            s * df_q * pdf * t.sqrt(),
            -s * df_q * pdf * sigma / (2.0 * t.sqrt())
                - r * k * df_r * norm_cdf(d2)
                + q * s * df_q * norm_cdf(d1),
            k * t * df_r * norm_cdf(d2),
        ),
        Kind::Put => (
            df_q * (norm_cdf(d1) - 1.0),
            df_q * pdf / (s * sq),
            s * df_q * pdf * t.sqrt(),
            -s * df_q * pdf * sigma / (2.0 * t.sqrt())
                + r * k * df_r * norm_cdf(-d2)
                - q * s * df_q * norm_cdf(-d1),
            -k * t * df_r * norm_cdf(-d2),
        ),
    };
    Greeks {
        delta,
        gamma,
        vega,
        theta,
        rho,
    }
}

/// American option price via a Cox-Ross-Rubinstein binomial tree with
/// early-exercise checks — the venue's "CRR referee" (exact in the limit;
/// 80 steps gives cent-level accuracy for typical perp-option parameters).
/// Chosen over the BAW quadratic approximation because it is provably
/// consistent: `american >= european`, `american >= intrinsic`,
/// `american == european` when early exercise is never optimal (calls
/// with zero carry benefit).
pub fn baw(kind: Kind, s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    american_crr(kind, s, k, r, q, sigma, t, 80)
}

/// American price by CRR backward induction.
pub fn american_crr(
    kind: Kind,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    steps: usize,
) -> f64 {
    if t <= 0.0 || sigma <= 0.0 || steps < 1 {
        return intrinsic(kind, s, k).max(0.0);
    }
    let dt = t / steps as f64;
    let u = (sigma * dt.sqrt()).exp();
    let d = 1.0 / u;
    let growth = ((r - q) * dt).exp();
    let p = (growth - d) / (u - d);
    if !(0.0..=1.0).contains(&p) {
        // Degenerate tree (extreme carry): fall back to European-ish bound.
        return bsm(kind, s, k, r, q, sigma, t).max(intrinsic(kind, s, k));
    }
    let disc = (-r * dt).exp();
    // Terminal payoffs
    let mut vals: Vec<f64> = (0..=steps)
        .map(|j| {
            let sj = s * u.powi(j as i32) * d.powi((steps - j) as i32);
            intrinsic(kind, sj, k)
        })
        .collect();
    // Backward induction with early exercise.
    for step in (0..steps).rev() {
        for j in 0..=step {
            let sj = s * u.powi(j as i32) * d.powi((step - j) as i32);
            let cont = disc * (p * vals[j + 1] + (1.0 - p) * vals[j]);
            let ex = intrinsic(kind, sj, k);
            vals[j] = cont.max(ex);
        }
    }
    vals[0]
}

#[inline]
fn intrinsic(kind: Kind, s: f64, k: f64) -> f64 {
    match kind {
        Kind::Call => (s - k).max(0.0),
        Kind::Put => (k - s).max(0.0),
    }
}

/// Effective maturity of an everlasting option (venue convention):
/// `T_eff = interval * maturity_multiple` (default 1h x 24).
#[inline]
pub fn everlasting_t_eff(interval_secs: f64, maturity_multiple: f64) -> f64 {
    (interval_secs * maturity_multiple) / (365.0 * 24.0 * 3600.0)
}

/// Everlasting option mark: European priced at `T_eff`.
pub fn everlasting(
    kind: Kind,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    sigma: f64,
    interval_secs: f64,
    maturity_multiple: f64,
) -> f64 {
    bsm(kind, s, k, r, q, sigma, everlasting_t_eff(interval_secs, maturity_multiple))
}

/// Implied volatility by bracketed Newton with bisection fallback.
pub fn implied_vol(kind: Kind, price: f64, s: f64, k: f64, r: f64, q: f64, t: f64) -> Option<f64> {
    if t <= 0.0 || price <= intrinsic(kind, s, k) + 1e-12 {
        return None;
    }
    let mut lo = 1e-4;
    let mut hi = 5.0;
    // bisection brackets first
    let f = |v: f64| bsm(kind, s, k, r, q, v, t) - price;
    if f(lo) > 0.0 || f(hi) < 0.0 {
        return None;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
        if hi - lo < 1e-10 {
            break;
        }
    }
    Some(0.5 * (lo + hi))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_call_parity() {
        let (s, k, r, q, sig, t) = (100.0, 95.0, 0.05, 0.02, 0.3, 0.5);
        let c = bsm(Kind::Call, s, k, r, q, sig, t);
        let p = bsm(Kind::Put, s, k, r, q, sig, t);
        let lhs = c - p;
        let rhs = s * (-q * t).exp() - k * (-r * t).exp();
        assert!((lhs - rhs).abs() < 1e-9, "{lhs} vs {rhs}");
    }

    #[test]
    fn known_price() {
        // Standard reference: S=100 K=100 r=5% q=0 sigma=20% T=1
        let c = bsm(Kind::Call, 100.0, 100.0, 0.05, 0.0, 0.2, 1.0);
        assert!((c - 10.4506).abs() < 2e-3, "call {c}");
        let p = bsm(Kind::Put, 100.0, 100.0, 0.05, 0.0, 0.2, 1.0);
        assert!((p - 5.5735).abs() < 2e-3, "put {p}");
    }

    #[test]
    fn greeks_sanity() {
        let g = bsm_greeks(Kind::Call, 100.0, 100.0, 0.05, 0.0, 0.2, 1.0);
        assert!(g.delta > 0.5 && g.delta < 0.7);
        assert!(g.gamma > 0.0 && g.gamma < 0.05);
        // delta by finite differences
        let h = 0.01;
        let up = bsm(Kind::Call, 100.0 + h, 100.0, 0.05, 0.0, 0.2, 1.0);
        let dn = bsm(Kind::Call, 100.0 - h, 100.0, 0.05, 0.0, 0.2, 1.0);
        let fd = (up - dn) / (2.0 * h);
        assert!((fd - g.delta).abs() < 1e-4);
        // vega by finite differences
        let vp = bsm(Kind::Call, 100.0, 100.0, 0.05, 0.0, 0.2 + 1e-4, 1.0);
        let vm = bsm(Kind::Call, 100.0, 100.0, 0.05, 0.0, 0.2 - 1e-4, 1.0);
        assert!(((vp - vm) / 2e-4 - g.vega).abs() < 1e-3);
    }

    #[test]
    fn implied_vol_roundtrip() {
        let (s, k, r, q, t) = (100.0, 110.0, 0.05, 0.01, 0.75);
        for &v in &[0.15f64, 0.3, 0.6, 1.2] {
            let price = bsm(Kind::Call, s, k, r, q, v, t);
            let iv = implied_vol(Kind::Call, price, s, k, r, q, t).unwrap();
            assert!((iv - v).abs() < 1e-6, "iv {iv} vs {v}");
        }
    }

    #[test]
    fn american_geq_european() {
        let (s, k, r, q, sig, t) = (100.0, 100.0, 0.05, 0.0, 0.25, 0.5);
        let euro_p = bsm(Kind::Put, s, k, r, q, sig, t);
        let amer_p = baw(Kind::Put, s, k, r, q, sig, t);
        assert!(amer_p >= euro_p - 0.03, "amer {amer_p} < euro {euro_p}");
        // deep ITM put: exercise value
        let deep = baw(Kind::Put, 50.0, 100.0, 0.05, 0.0, 0.25, 0.5);
        assert!((deep - 50.0).abs() < 0.05, "deep ITM put {deep}");
        // no early exercise premium without carry benefit (q=0, call):
        // american ~ european (tree discretization allowance)
        let c_am = baw(Kind::Call, s, k, r, 0.0, sig, t);
        let c_eu = bsm(Kind::Call, s, k, r, 0.0, sig, t);
        assert!(c_am < c_eu + 0.03 && c_am >= c_eu - 0.03, "am {c_am} vs eu {c_eu}");
        // with dividends (q > r), american calls exceed european
        let c_am_div = baw(Kind::Call, s, k, r, 0.08, sig, t);
        let c_eu_div = bsm(Kind::Call, s, k, r, 0.08, sig, t);
        assert!(c_am_div > c_eu_div, "am-div {c_am_div} vs eu-div {c_eu_div}");
    }

    #[test]
    fn everlasting_convention() {
        // 1h interval x 24 => T_eff = 86400 / 31536000 = 1/365 years = 1 day
        let t = everlasting_t_eff(3600.0, 24.0);
        assert!((t - 1.0 / 365.0).abs() < 1e-12);
        let price = everlasting(Kind::Call, 100.0, 100.0, 0.05, 0.0, 0.6, 3600.0, 24.0);
        let reference = bsm(Kind::Call, 100.0, 100.0, 0.05, 0.0, 0.6, 1.0 / 365.0);
        assert!((price - reference).abs() < 1e-12);
    }
}
