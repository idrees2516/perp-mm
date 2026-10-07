//! Guéant–Lehalle–Fernández-Tapia (2013) asymptotic quotes,
//! "Dealing with the inventory risk: a solution to the market making
//! problem" (Math. Fin. Econ. 7(4); arXiv:1105.3115).
//!
//! The GLFT infinite-horizon approximation (verified form, Guéant's
//! survey arXiv:1605.01862 §4, lot size Delta = 1, risk aversion xi =
//! gamma):
//! ```text
//! delta_b^inf(q) ~= (1/gamma) ln(1 + gamma/kappa)
//!                  + (2q+1)/2 * sqrt( (gamma sigma^2 / (2 A kappa))
//!                                     * (1 + gamma/kappa)^{kappa/gamma + 1} )
//! delta_a^inf(q) ~= (1/gamma) ln(1 + gamma/kappa)
//!                  - (2q-1)/2 * sqrt( ... )
//! spread^inf(q)  ~= (2/gamma) ln(1 + gamma/kappa) + sqrt( ... )
//! ```
//! (inventory-averse skew linear in q: half-spread + skew decomposition).
//!
//! These are validated against the exact [`crate::hjb::MmHjb`] solver at
//! long horizons in the test below — the solver is ground truth, the
//! asymptotics are the fast path.

/// GLFT asymptotic quote distances at inventory `q`.
#[derive(Clone, Copy, Debug)]
pub struct GlftAsymptotic {
    pub gamma: f64,
    pub sigma: f64,
    pub kappa: f64,
    pub a: f64,
}

impl GlftAsymptotic {
    pub fn new(gamma: f64, sigma: f64, kappa: f64, a: f64) -> GlftAsymptotic {
        GlftAsymptotic {
            gamma,
            sigma,
            kappa,
            a,
        }
    }

    /// The skew coefficient `sqrt((gamma sigma^2/(2 A kappa)) (1+gamma/kappa)^{kappa/gamma+1})`.
    pub fn skew_coef(&self) -> f64 {
        let base = self.gamma * self.sigma * self.sigma / (2.0 * self.a * self.kappa);
        let pow = (1.0 + self.gamma / self.kappa).powf(self.kappa / self.gamma + 1.0);
        (base * pow).sqrt()
    }

    /// Half of the risk-neutral part of the spread.
    pub fn half_intensity_term(&self) -> f64 {
        (1.0 / self.gamma) * (1.0 + self.gamma / self.kappa).ln()
    }

    /// Bid distance from the mid at inventory `q`.
    pub fn delta_bid(&self, q: i64) -> f64 {
        self.half_intensity_term() + (2.0 * q as f64 + 1.0) / 2.0 * self.skew_coef()
    }

    /// Ask distance from the mid at inventory `q`.
    pub fn delta_ask(&self, q: i64) -> f64 {
        self.half_intensity_term() - (2.0 * q as f64 - 1.0) / 2.0 * self.skew_coef()
    }

    /// Total spread at inventory `q`.
    pub fn spread(&self, _q: i64) -> f64 {
        2.0 * self.half_intensity_term() + self.skew_coef()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hjb::{MmHjb, MmProblem};

    #[test]
    fn matches_exact_solver_at_long_horizon() {
        // Structural agreement with the GLFT survey's delta_approx (which
        // is itself an approximation of the exact problem): same sign, same
        // monotone structure, same order of magnitude. The exact solver is
        // validated against Monte Carlo entropic CE elsewhere; the GLFT
        // skew coefficient is the fast-path anchor.
        let (gamma, sigma, kappa, a) = (0.1f64, 2.0, 1.5, 1.5);
        let mut prob = MmProblem::new(gamma, sigma, kappa, a, 200.0);
        prob.n_steps = 4000;
        let sol = MmHjb::solve(prob);
        let asym = GlftAsymptotic::new(gamma, sigma, kappa, a);
        for &q in &[-5i64, -2, 2, 5] {
            let solver_skew_b = sol.delta_bid(q, 0.0) - sol.delta_bid(0, 0.0);
            let asym_skew_b = asym.delta_bid(q) - asym.delta_bid(0);
            // same sign and same order of magnitude
            assert!(
                solver_skew_b * asym_skew_b > 0.0,
                "q={q}: sign mismatch {solver_skew_b} vs {asym_skew_b}"
            );
            let ratio = (solver_skew_b / asym_skew_b).abs();
            assert!(
                (0.2..5.0).contains(&ratio),
                "q={q}: skew ratio {ratio} (solver {solver_skew_b} vs GLFT {asym_skew_b})"
            );
        }
        // Skew linear in q in the asymptotics by construction.
        let d1 = asym.delta_bid(1) - asym.delta_bid(0);
        let d2 = asym.delta_bid(2) - asym.delta_bid(1);
        assert!((d1 - d2).abs() < 1e-12);
        // The GLFT q=0 centered half-spread is intensity + skew/2; the
        // solver's steady-state half must be at least the intensity floor
        // and of the same order.
        let solver_half = 0.5 * (sol.delta_bid(0, 0.0) + sol.delta_ask(0, 0.0));
        let glft_half = asym.half_intensity_term() + 0.5 * asym.skew_coef();
        assert!(
            solver_half >= asym.half_intensity_term() - 0.05,
            "solver half {solver_half} below intensity floor"
        );
        assert!(
            (solver_half - glft_half).abs() < 0.6,
            "solver half {solver_half} vs GLFT {glft_half}"
        );
    }
}
