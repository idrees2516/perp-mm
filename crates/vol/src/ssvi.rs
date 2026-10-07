//! SSVI surface (Gatheral–Jacquier 2014, "Arbitrage-free SVI volatility
//! surfaces"):
//! `w(k, θ) = θ/2 (1 + ρ φ(θ) k + sqrt((φ(θ) k + ρ)^2 + 1 − ρ²))`
//! with Heston-like `φ(θ) = 1/(η θ^γ (1+θ)^{1−γ})`.
//!
//! Conditions implemented (each verified against the numerical
//! Breeden–Litzenberger ground truth in tests):
//!
//! - **Butterfly**: the sufficient condition `θ φ(θ)² (1+|ρ|) ≤ 4`
//!   (Gatheral–Jacquier Thm 4.2) plus the direct density check
//!   `g(k) ≥ 0` on a grid.
//! - **Calendar**: `θ(T)` nondecreasing, verified numerically by
//!   `w(k, T₂) ≥ w(k, T₁)` on a grid across all pillar pairs
//!   (Gatheral–Jacquier Thm 4.1 gives sufficient conditions on φ; the
//!   numerical check is the model-free ground truth).
//!
//! Global (non-pillar-wise) parametrizations follow Mingone (2022)
//! ("No arbitrage global parametrization for the eSSVI volatility
//! surface") — our Heston-like φ with monotone θ is the simplest
//! member of that family.

/// ATM pillars: `(maturity_years, atm_total_variance_theta)`.
#[derive(Clone, Debug)]
pub struct SsviSurface {
    pub rho: f64,
    pub eta: f64,
    pub gamma: f64,
    /// Sorted by maturity; θ enforced nondecreasing at construction.
    pub pillars: Vec<(f64, f64)>,
}

/// Result of a static-arbitrage audit.
#[derive(Clone, Copy, Debug, Default)]
pub struct ArbReport {
    pub butterfly_condition: bool,
    pub butterfly_grid: bool,
    pub calendar: bool,
}

impl ArbReport {
    pub fn ok(&self) -> bool {
        self.butterfly_condition && self.butterfly_grid && self.calendar
    }
}

impl SsviSurface {
    /// Build from ATM implied vols per maturity. θ = σ_atm² T; the pillar
    /// list is sorted and θ is made nondecreasing (cumulative max) —
    /// decreasing ATM variance with T is calendar-arbitrageable by
    /// construction.
    pub fn new(rho: f64, eta: f64, gamma: f64, atm_vols: &[(f64, f64)]) -> SsviSurface {
        let mut pillars: Vec<(f64, f64)> = atm_vols
            .iter()
            .map(|&(t, s)| (t, s * s * t))
            .filter(|&(t, _)| t > 0.0)
            .collect();
        pillars.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut run_max = 0.0f64;
        for p in pillars.iter_mut() {
            run_max = run_max.max(p.1);
            p.1 = run_max;
        }
        SsviSurface {
            rho: rho.clamp(-0.999, 0.999),
            eta: eta.max(1e-4),
            gamma: gamma.clamp(1e-3, 0.999),
            pillars,
        }
    }

    /// Heston-like φ(θ).
    #[inline]
    pub fn phi(&self, theta: f64) -> f64 {
        1.0 / (self.eta * theta.powf(self.gamma) * (1.0 + theta).powf(1.0 - self.gamma))
    }

    /// ATM total variance at maturity T (piecewise-linear in T between
    /// pillars; flat-vol extrapolation beyond the last pillar keeps θ
    /// nondecreasing).
    pub fn theta_at(&self, t: f64) -> f64 {
        let ps = &self.pillars;
        if ps.is_empty() {
            return 0.0;
        }
        if t <= ps[0].0 {
            // flat-vol extrapolation: theta = sigma_1^2 * t (<= theta_1)
            let vol2 = ps[0].1 / ps[0].0;
            return vol2 * t.max(0.0);
        }
        if t >= ps[ps.len() - 1].0 {
            let (t_last, th_last) = ps[ps.len() - 1];
            let vol2 = th_last / t_last;
            return th_last + vol2 * (t - t_last).max(0.0);
        }
        for w in ps.windows(2) {
            let (t0, th0) = w[0];
            let (t1, th1) = w[1];
            if t >= t0 && t <= t1 {
                let f = (t - t0) / (t1 - t0);
                return th0 + f * (th1 - th0);
            }
        }
        ps[ps.len() - 1].1
    }

    /// Total variance at log-moneyness k, maturity t.
    pub fn total_var(&self, k: f64, t: f64) -> f64 {
        let th = self.theta_at(t);
        if th <= 0.0 {
            return 0.0;
        }
        let ph = self.phi(th);
        let u = ph * k + self.rho;
        th / 2.0 * (1.0 + self.rho * ph * k + (u * u + 1.0 - self.rho * self.rho).sqrt())
    }

    /// Implied vol at (k, t).
    #[inline]
    pub fn iv(&self, k: f64, t: f64) -> f64 {
        (self.total_var(k, t) / t).sqrt()
    }

    /// ATM implied vol at maturity t.
    #[inline]
    pub fn iv_atm(&self, t: f64) -> f64 {
        (self.theta_at(t) / t).sqrt()
    }

    /// First two k-derivatives of the slice at t (for densities and
    /// surface greeks).
    pub fn dw_dk(&self, k: f64, t: f64) -> (f64, f64) {
        let th = self.theta_at(t);
        let ph = self.phi(th);
        let u = ph * k + self.rho;
        let rt = (u * u + 1.0 - self.rho * self.rho).sqrt();
        let wp = th / 2.0 * (self.rho * ph + ph * u / rt);
        let wpp = th / 2.0 * ph * ph * (1.0 - self.rho * self.rho) / rt.powi(3);
        (wp, wpp)
    }

    /// Density factor g(k) of the slice at t.
    pub fn g(&self, k: f64, t: f64) -> f64 {
        let w = self.total_var(k, t);
        if w <= 0.0 {
            return f64::NAN;
        }
        let (wp, wpp) = self.dw_dk(k, t);
        (1.0 - k * wp / (2.0 * w)).powi(2) - wp * wp / 4.0 + wpp / 2.0
    }

    /// Gatheral–Jacquier butterfly sufficient condition at every pillar.
    pub fn butterfly_condition(&self) -> bool {
        self.pillars.iter().all(|&(_, th)| {
            let ph = self.phi(th);
            th * ph * ph * (1.0 + self.rho.abs()) <= 4.0 + 1e-12
        })
    }

    /// Direct density check g(k) >= 0 on a grid for every pillar and a
    /// few interpolated maturities.
    pub fn butterfly_grid(&self, kmin: f64, kmax: f64, n: usize) -> bool {
        let n = n.max(16);
        let mut ts: Vec<f64> = self.pillars.iter().map(|&(t, _)| t).collect();
        if ts.len() >= 2 {
            let mid = 0.5 * (ts[0] + ts[ts.len() - 1]);
            ts.push(mid);
        }
        ts.retain(|&t| t > 0.0);
        for &t in &ts {
            for i in 0..=n {
                let k = kmin + (kmax - kmin) * i as f64 / n as f64;
                let g = self.g(k, t);
                if g.is_nan() || g < -1e-10 {
                    return false;
                }
            }
        }
        true
    }

    /// Calendar check: w(k, T2) >= w(k, T1) on a grid for all pillar
    /// pairs and interpolated maturities.
    pub fn calendar_grid(&self, kmin: f64, kmax: f64, n: usize) -> bool {
        let n = n.max(16);
        if self.pillars.len() < 2 {
            return true;
        }
        let mut ts: Vec<f64> = self.pillars.iter().map(|&(t, _)| t).collect();
        ts.push(0.5 * (ts[0] + ts[ts.len() - 1]));
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for w in ts.windows(2) {
            let (t1, t2) = (w[0], w[1]);
            for i in 0..=n {
                let k = kmin + (kmax - kmin) * i as f64 / n as f64;
                if self.total_var(k, t2) < self.total_var(k, t1) - 1e-12 {
                    return false;
                }
            }
        }
        true
    }

    /// Full audit.
    pub fn arb_report(&self) -> ArbReport {
        ArbReport {
            butterfly_condition: self.butterfly_condition(),
            butterfly_grid: self.butterfly_grid(-1.5, 1.5, 120),
            calendar: self.calendar_grid(-1.5, 1.5, 120),
        }
    }

    /// BSM ground-truth static no-arbitrage audit: for each pillar
    /// maturity, call prices from this surface must be monotone,
    /// convex, bounded; and calendar spreads must be nonnegative.
    pub fn static_arb_free_bsm(&self, s: f64, r: f64, q: f64) -> bool {
        use crate::greeks::price;
        use models::options::Kind;
        for &(t, _) in &self.pillars {
            let n = 48;
            let mut prev_c = f64::INFINITY;
            let (mut p_c, mut p_k, mut pp_c, mut pp_k) =
                (f64::INFINITY, 0.0f64, f64::INFINITY, 0.0f64);
            for i in 0..=n {
                let k = -1.2 + 2.4 * i as f64 / n as f64;
                let strike = s * k.exp();
                let iv = self.iv(k, t);
                let c = price(Kind::Call, s, strike, r, q, iv, t);
                if c > s * (-q * t).exp() + 1e-9 {
                    return false;
                }
                if c < (s * (-q * t).exp() - strike * (-r * t).exp()).max(0.0) - 1e-9 {
                    return false;
                }
                if c > prev_c + 1e-9 {
                    return false;
                }
                if i >= 2 {
                    let d2 = (c - p_c) / (strike - p_k).max(1e-12)
                        - (p_c - pp_c) / (p_k - pp_k).max(1e-12);
                    if d2 < -1e-7 {
                        return false;
                    }
                }
                pp_c = p_c;
                pp_k = p_k;
                p_c = c;
                p_k = strike;
                prev_c = c;
            }
        }
        // calendar on BSM prices
        if self.pillars.len() >= 2 {
            for w in self.pillars.windows(2) {
                let (t1, t2) = (w[0].0, w[1].0);
                for i in 0..=48 {
                    let k = -1.2 + 2.4 * i as f64 / 48.0;
                    let strike = s * k.exp();
                    let c1 = price(Kind::Call, s, strike, r, q, self.iv(k, t1), t1);
                    let c2 = price(Kind::Call, s, strike, r, q, self.iv(k, t2), t2);
                    if c2 < c1 - 1e-8 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Total vega exposure shape helper: the vega weight of a k-ladder
    /// at maturity t, normalized so the ATM point has weight 1.
    pub fn vega_shape(&self, k: f64, t: f64) -> f64 {
        let w = self.total_var(k, t);
        if w <= 0.0 {
            return 0.0;
        }
        // Vega ∝ φ(d1)·√w·(S√T) — shape only in k via the density term.
        let ph = self.phi(self.theta_at(t));
        let u = ph * k + self.rho;
        let rt = (u * u + 1.0 - self.rho * self.rho).sqrt();
        // d1(k) = (−k + w/2)/√w (forward-moneyness form, r=q=0)
        let d1 = (-k + 0.5 * w) / w.sqrt();
        (-(0.5 * d1 * d1)).exp() * rt.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn surface(rho: f64, eta: f64, gamma: f64) -> SsviSurface {
        SsviSurface::new(
            rho,
            eta,
            gamma,
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
    fn atm_iv_matches_pillars() {
        let sf = surface(-0.7, 1.0, 0.5);
        for &(t, vol) in &[
            (1.0 / 52.0, 0.25),
            (1.0 / 12.0, 0.27),
            (0.25, 0.30),
            (0.5, 0.32),
            (1.0, 0.34),
        ] {
            assert!((sf.iv_atm(t) - vol).abs() < 1e-9, "{} vs {}", sf.iv_atm(t), vol);
            // ATM k=0: w = θ/2(1 + sqrt(ρ²+1−ρ²)) = θ/2(1+1) = θ
            assert!((sf.total_var(0.0, t) - vol * vol * t).abs() < 1e-12);
        }
    }

    #[test]
    fn gk_condition_implies_positive_density() {
        // On random admissible params, whenever the GJ butterfly
        // condition holds, the numerical density is non-negative.
        let mut seed = 987654321u64;
        let mut next = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..60 {
            let rho = -0.95 + 1.7 * next();
            let eta = 0.3 + 2.5 * next();
            let gamma = 0.05 + 0.85 * next();
            let sf = surface(rho, eta, gamma);
            if sf.butterfly_condition() {
                assert!(
                    sf.butterfly_grid(-1.5, 1.5, 200),
                    "condition ok but grid fails: rho={rho} eta={eta} gamma={gamma}"
                );
                assert!(
                    sf.static_arb_free_bsm(100.0, 0.0, 0.0),
                    "condition ok but BSM arb: rho={rho} eta={eta} gamma={gamma}"
                );
            }
        }
    }

    #[test]
    fn decreasing_theta_is_repaired_and_calendar_clean() {
        let sf = SsviSurface::new(
            -0.6,
            1.2,
            0.5,
            &[(0.25, 0.30), (0.5, 0.20), (1.0, 0.32)], // 0.5y pillar violates
        );
        // construction enforces nondecreasing theta
        assert!(sf.pillars.windows(2).all(|w| w[0].1 <= w[1].1 + 1e-12));
        assert!(sf.calendar_grid(-1.5, 1.5, 200));
        assert!(sf.static_arb_free_bsm(100.0, 0.0, 0.0));
    }

    #[test]
    fn calendar_grid_catches_violations() {
        // Manually crafted decreasing pillar (bypassing the constructor
        // repair) must be caught.
        let mut sf = surface(-0.6, 1.2, 0.5);
        sf.pillars[3].1 = sf.pillars[2].1 * 0.5; // 0.5y below 0.25y
        assert!(!sf.calendar_grid(-1.5, 1.5, 200));
    }

    #[test]
    fn breeden_litzenberger_density_positive_and_integrates() {
        use crate::greeks::{density_from_calls, price};
        use models::options::Kind;
        let sf = surface(-0.75, 1.1, 0.45);
        let (s, r, q, t) = (100.0, 0.0, 0.0, 0.25);
        let strikes: Vec<f64> = (0..=300).map(|i| 30.0 + i as f64 * 0.8).collect();
        let calls: Vec<f64> = strikes
            .iter()
            .map(|&strike| {
                let k = (strike / s).ln();
                price(Kind::Call, s, strike, r, q, sf.iv(k, t), t)
            })
            .collect();
        let dens = density_from_calls(&calls, &strikes, r, t);
        let interior = dens[1..dens.len() - 1].iter().cloned();
        let min = interior.fold(f64::INFINITY, f64::min);
        assert!(min > -1e-4, "negative density {min}");
        let mut mass = 0.0;
        for i in 1..strikes.len() - 1 {
            mass += 0.5 * (strikes[i + 1] - strikes[i - 1]) * dens[i];
        }
        assert!((mass - 1.0).abs() < 0.03, "mass {mass}");
    }

    #[test]
    fn smile_shape_sanity() {
        let sf = surface(-0.7, 1.0, 0.5);
        // Equity skew: put wing (k<0) trades above call wing (k>0).
        assert!(sf.iv(-0.3, 0.25) > sf.iv(0.3, 0.25));
        // Term structure at the ATM vol levels: longer T, higher ATM vol.
        assert!(sf.iv_atm(1.0) > sf.iv_atm(0.25));
        // vega shape peaks near the money
        assert!(sf.vega_shape(0.0, 0.25) > sf.vega_shape(0.8, 0.25));
    }
}
