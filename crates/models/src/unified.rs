//! The unified Avellaneda–Stoikov / Cartea–Jaimungal framework
//! (arXiv:2606.01477).
//!
//! # The forced-unique objective
//!
//! Let the mid `S` be a continuous semimartingale (benchmark `dS = sigma dB`),
//! fills arrive with intensities `lambda^{a/b}(delta)`, inventory `q_t`
//! steps +-1 on fills, and terminal wealth is liquidation-adjusted:
//! `W_T^L = X_T + q_T S_T - L(q_T)` with `L` convex, `L(0) = 0`.
//! Theorem 10 (Forced Uniqueness): any dynamic preference functional
//! satisfying (J1) cash-additivity, (J2) normalization, (J3) concavity
//! (strict), (J4) strong dynamic consistency, (J5) law-invariance is
//! **necessarily**
//! ```text
//! J_t(pi) = -(1/gamma) log E[ exp(-gamma * W_T^L(pi)) | F_t ]
//! ```
//! for a unique constant `gamma > 0` (the entropic certainty-equivalent).
//! `gamma` is invariant under the business-time clock change (Cor. 12) and
//! shared across all assets of a desk (Theorem 49).
//!
//! # The forced parameter relations
//!
//! * Running inventory penalty (CJ) **forced**: `phi = gamma*sigma^2/2`
//!   (paper convention; the Cartea et al. 2015 book convention omits the
//!   factor 1/2, giving `gamma*sigma^2` — see [`ForcedRelations`]).
//! * Terminal penalty forced: `alpha = L''(0)/2` (Cor. 22; for quadratic
//!   impact `L(q) = (kappa/2) q^2`, `alpha = kappa/2`).
//! * Calibration inversion (Cor. 23): `gamma = 2*phi/sigma^2`, pointwise
//!   in stochastic volatility: `gamma_t = 2*phi_t/sigma_t^2` must be
//!   constant — a desk-level diagnostic.
//!
//! # The AS benchmark quotes (Corollaries 19-20)
//!
//! With `lambda(delta) = A e^{-kappa delta}`, `L = 0`:
//! ```text
//! reservation  r(s,q,t)   = s - gamma*sigma^2 * q * (T - t)
//! half-spread  h(t)       = (gamma*sigma^2/2)*(T-t) + (1/gamma)*ln(1 + gamma/kappa)
//! ask = r + h,  bid = r - h
//! ```
//! These are also what the CJ objective produces at `phi = gamma*sigma^2/2`
//! (identical reservation price and half-spread, Cor. 20). The exact
//! finite-horizon optimal quotes (with inventory-dependent skew) come from
//! the [`crate::hjb`] solver; the AS closed forms above are its
//! second-order expansion and the industry-standard fast path.

/// The three "forced" relations of the unified framework.
pub struct ForcedRelations;

impl ForcedRelations {
    /// CJ running inventory coefficient pinned by the entropic objective:
    /// `phi = gamma * sigma^2 / 2` (paper convention).
    #[inline]
    pub fn phi_from_gamma(gamma: f64, sigma: f64) -> f64 {
        gamma * sigma * sigma / 2.0
    }

    /// Desk-calibration inversion (Corollary 23): `gamma = 2 * phi / sigma^2`.
    #[inline]
    pub fn gamma_from_phi(phi: f64, sigma: f64) -> f64 {
        2.0 * phi / (sigma * sigma)
    }

    /// Terminal penalty pinned by the liquidation-cost curvature:
    /// `alpha = L''(0) / 2`.
    #[inline]
    pub fn alpha_from_liquidation(l_double_prime_at_zero: f64) -> f64 {
        l_double_prime_at_zero / 2.0
    }

    /// The Cartea et al. (2015) book convention defines phi as *twice* the
    /// paper's quantity; the forced relation in that convention is
    /// `phi_book = gamma * sigma^2`.
    #[inline]
    pub fn phi_book_convention(gamma: f64, sigma: f64) -> f64 {
        gamma * sigma * sigma
    }

    /// Running inventory cost rate at the current (possibly stochastic)
    /// volatility: `(gamma/2) * sigma_t^2 * q^2` per unit time —
    /// `(gamma/2) * q^2 * d<S>_t` in the quadratic-variation clock
    /// (Proposition 37). Constant `gamma`, clock-invariant (Corollary 12).
    #[inline]
    pub fn running_inventory_cost(gamma: f64, sigma_t: f64, q: f64) -> f64 {
        0.5 * gamma * sigma_t * sigma_t * q * q
    }
}

/// Parameters of the unified AS/CJ quoting model.
#[derive(Clone, Debug)]
pub struct UnifiedParams {
    /// Risk-aversion (the single forced scalar; constant intraday).
    pub gamma: f64,
    /// Mid volatility (per sqrt time).
    pub sigma: f64,
    /// Fill-intensity decay: `lambda(delta) = A * exp(-kappa * delta)`.
    pub kappa: f64,
    /// Fill-intensity scale.
    pub a: f64,
    /// Horizon.
    pub t: f64,
    /// Optional quadratic liquidation cost `L(q) = kappa_liq/2 * q^2`
    /// (0 = liquidate at mid).
    pub kappa_liq: f64,
}

impl UnifiedParams {
    /// Benchmark parameters of the paper's Appendix D
    /// (`S0=100, sigma=2, T=1, gamma=0.1, A=140, kappa=1.5, L=0`),
    /// for which `phi_forced = 0.1*4/2 = 0.2`.
    pub fn benchmark() -> UnifiedParams {
        UnifiedParams {
            gamma: 0.1,
            sigma: 2.0,
            kappa: 1.5,
            a: 140.0,
            t: 1.0,
            kappa_liq: 0.0,
        }
    }

    /// The forced CJ running penalty for these parameters
    /// (`gamma * sigma^2 / 2`).
    pub fn phi_forced(&self) -> f64 {
        ForcedRelations::phi_from_gamma(self.gamma, self.sigma)
    }

    /// Reservation price `r = s - gamma*sigma^2*q*(T-t)`.
    pub fn reservation(&self, s: f64, q: f64, t: f64) -> f64 {
        s - self.gamma * self.sigma * self.sigma * q * (self.t - t)
    }

    /// Optimal half-spread
    /// `h = (gamma*sigma^2/2)*(T-t) + (1/gamma)*ln(1 + gamma/kappa)`.
    pub fn half_spread(&self, t: f64) -> f64 {
        0.5 * self.gamma * self.sigma * self.sigma * (self.t - t)
            + (1.0 / self.gamma) * (1.0 + self.gamma / self.kappa).ln()
    }

    /// AS quote pair `(bid, ask)` at time `t`, mid `s`, inventory `q`.
    pub fn quotes(&self, s: f64, q: f64, t: f64) -> (f64, f64) {
        let r = self.reservation(s, q, t);
        let h = self.half_spread(t);
        (r - h, r + h)
    }

    /// Fill intensity at distance `delta`.
    #[inline]
    pub fn lambda(&self, delta: f64) -> f64 {
        self.a * (-self.kappa * delta).exp()
    }

    /// The value-function factorization scale of the exact HJB: the running
    /// diffusion penalty on the reduced `v` system is `gamma^2 sigma^2 q^2/2`
    /// (see [`crate::hjb`]).
    #[inline]
    pub fn v_diffusion_coeff(&self, q: f64) -> f64 {
        0.5 * self.gamma * self.gamma * self.sigma * self.sigma * q * q
    }
}

/// Entropic certainty-equivalent of a sample of terminal wealths:
/// `-(1/gamma) * log( mean( exp(-gamma * W) ) )`.
pub fn entropic_ce(samples: &[f64], gamma: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut m = 0.0f64;
    for &w in samples {
        m += (-gamma * w).exp();
    }
    m /= samples.len() as f64;
    -(1.0 / gamma) * m.ln()
}

/// Multi-asset extension (Proposition 50): with constant covariance
/// `Sigma`, the forced running-cost matrix is `Phi_ij = gamma*Sigma_ij/2`
/// and the reservation price vector shifts by `-gamma * Sigma * q * (T-t)`.
pub struct MultiAsset {
    pub gamma: f64,
    /// Covariance matrix (row-major, K x K).
    pub sigma_cov: Vec<f64>,
    pub k: usize,
    pub t: f64,
}

impl MultiAsset {
    pub fn new(gamma: f64, sigma_cov: Vec<f64>, k: usize, t: f64) -> MultiAsset {
        debug_assert_eq!(sigma_cov.len(), k * k);
        MultiAsset {
            gamma,
            sigma_cov,
            k,
            t,
        }
    }

    /// Forced running-cost matrix `Phi = gamma * Sigma / 2` (row-major).
    pub fn forced_phi(&self) -> Vec<f64> {
        self.sigma_cov
            .iter()
            .map(|&x| self.gamma * x / 2.0)
            .collect()
    }

    /// Reservation price vector: `r_k = s_k - gamma * (Sigma q)_k * (T-t)`.
    pub fn reservation(&self, s: &[f64], q: &[f64], t: f64) -> Vec<f64> {
        let mut out = vec![0.0; self.k];
        for i in 0..self.k {
            let mut sq = 0.0;
            for j in 0..self.k {
                sq += self.sigma_cov[i * self.k + j] * q[j];
            }
            out[i] = s[i] - self.gamma * sq * (self.t - t);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_relations_roundtrip() {
        let (gamma, sigma) = (0.1f64, 2.0);
        let phi = ForcedRelations::phi_from_gamma(gamma, sigma);
        assert!((phi - 0.2).abs() < 1e-12, "benchmark phi_forced = 0.2");
        assert!((ForcedRelations::gamma_from_phi(phi, sigma) - gamma).abs() < 1e-12);
        // book convention = 2x paper convention
        assert!((ForcedRelations::phi_book_convention(gamma, sigma) - 2.0 * phi).abs() < 1e-12);
        // terminal penalty for quadratic impact L(q) = (kappa/2) q^2
        assert!((ForcedRelations::alpha_from_liquidation(1.5) - 0.75).abs() < 1e-12);
        // stochastic-vol running cost follows sigma_t^2
        let c1 = ForcedRelations::running_inventory_cost(0.1, 2.0, 3.0);
        let c2 = ForcedRelations::running_inventory_cost(0.1, 4.0, 3.0);
        assert!((c2 / c1 - 4.0).abs() < 1e-12);
    }

    #[test]
    fn as_quotes_benchmark() {
        let p = UnifiedParams::benchmark();
        // At t = T the spread is purely the fill-intensity term.
        let h_t = p.half_spread(p.t);
        let expect = (1.0f64 / 0.1) * (1.0f64 + 0.1 / 1.5).ln();
        assert!((h_t - expect).abs() < 1e-12);
        // Reservation skew: r(s, +q) is below s by gamma*sigma^2*q*tau.
        let r = p.reservation(100.0, 2.0, 0.5);
        assert!((r - (100.0 - 0.1 * 4.0 * 2.0 * 0.5)).abs() < 1e-12);
        // Quotes bracket the reservation price.
        let (bid, ask) = p.quotes(100.0, 0.0, 0.0);
        let h = p.half_spread(0.0);
        assert!((ask - bid - 2.0 * h).abs() < 1e-12);
        // Positive inventory skews both quotes down.
        let (bid_l, ask_l) = p.quotes(100.0, 1.0, 0.0);
        assert!(bid_l < bid && ask_l < ask);
    }

    #[test]
    fn entropic_ce_ordering() {
        // Deterministic wealth -> CE = wealth; risk penalizes dispersion.
        let det = entropic_ce(&[10.0; 8], 0.5);
        assert!((det - 10.0).abs() < 1e-9);
        let risky = entropic_ce(&[0.0, 20.0], 0.5);
        assert!(risky < 10.0);
        // More risk aversion -> lower CE for a dispersed payoff.
        let ce1 = entropic_ce(&[0.0, 20.0], 0.1);
        let ce2 = entropic_ce(&[0.0, 20.0], 1.0);
        assert!(ce2 < ce1);
        // CE is monotone in wealth.
        assert!(entropic_ce(&[15.0, 15.0], 1.0) > det);
    }

    #[test]
    fn multi_asset_phi_and_reservation() {
        let cov = vec![4.0, 1.0, 1.0, 9.0];
        let ma = MultiAsset::new(0.1, cov.clone(), 2, 1.0);
        let phi = ma.forced_phi();
        assert!((phi[0] - 0.2).abs() < 1e-12);
        assert!((phi[1] - 0.05).abs() < 1e-12);
        assert!((phi[3] - 0.45).abs() < 1e-12);
        let s = [100.0, 50.0];
        let q = [1.0, -2.0];
        let r = ma.reservation(&s, &q, 0.0);
        // r_0 = 100 - 0.1*(4*1 + 1*(-2))*1 = 100 - 0.2
        assert!((r[0] - 99.8).abs() < 1e-12);
        // r_1 = 50 - 0.1*(1*1 + 9*(-2)) = 50 + 1.7
        assert!((r[1] - 51.7).abs() < 1e-12);
    }
}
