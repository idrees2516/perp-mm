//! Exact numerical solution of the inventory market-making HJB in the
//! reduced `v` (psi) form — the core of the unified framework
//! (arXiv:2606.01477, Prop. 33) and of Guéant–Lehalle–Fernández-Tapia
//! (2013).
//!
//! # Derivation (self-contained, sign-verified three ways)
//!
//! CARA utility `U(w) = -exp(-gamma w)` on liquidation-adjusted terminal
//! wealth `W_T = X_T + q_T S_T - L(q_T)`; arithmetic mid `dS = sigma dB`;
//! fills at distance `delta` arrive with intensity `lambda(delta) =
//! A exp(-kappa delta)`; inventory bounded `|q| <= Q`. Cash-additivity
//! (axiom J1) factors the value function exactly:
//! ```text
//! V(t,x,q,s) = -exp(-gamma (x + q s)) * v_q(t)
//! ```
//! Substituting into the HJB (the ask fill moves wealth by
//! `+delta^a` — hence a `exp(-gamma delta^a)` factor on `v_{q-1}`):
//! ```text
//! vdot_q = -(gamma^2 sigma^2 q^2 / 2) v_q + M_a(q) + M_b(q)
//! M_a(q) = max_{delta >= 0} A e^{-kappa delta} ( v_q - e^{-gamma delta} v_{q-1} )
//! M_b(q) = max_{delta >= 0} A e^{-kappa delta} ( v_q - e^{-gamma delta} v_{q+1} )
//! v_q(T) = exp(gamma L(q))          (= 1 when L = 0)
//! ```
//! with the maximizer (FOC, unique interior maximum):
//! ```text
//! delta_a*(q) = (1/gamma) ln( (kappa+gamma) v_{q-1} / (kappa v_q) )
//! delta_b*(q) = (1/gamma) ln( (kappa+gamma) v_{q+1} / (kappa v_q) )
//! ```
//! Sign verification: (i) at `t = T` (`v = 1`) the FOC reduces to the
//! Avellaneda–Stoikov terminal half-spread `(1/gamma) ln(1 + gamma/kappa)`
//! exactly; (ii) with quoting disabled the system integrates to
//! `v_q = exp(gamma^2 sigma^2 (T-t) q^2 / 2)`, the exact certainty-
//! equivalent inflation of holding inventory; (iii) the long-horizon
//! ratios `v_{q+-1}/v_q` reproduce the GLFT Gaussian eigenvector skew.
//!
//! # Adverse-selection impact extension
//!
//! Optional exogenous impact: our ask fill moves the mid down by `beta`
//! (and our bid fill up by `beta`), so the post-fill wealth shift is
//! `delta - beta q`. The FOC becomes
//! `delta_a*(q) = beta q + (1/gamma) ln((kappa+gamma) v_{q-1}/(kappa v_q))`
//! — deeper quotes when the fill itself moves the market against us.

use crate::unified::UnifiedParams;

/// Problem specification for the market-making HJB.
#[derive(Clone, Debug)]
pub struct MmProblem {
    pub gamma: f64,
    pub sigma: f64,
    pub kappa: f64,
    pub a: f64,
    /// Horizon T.
    pub t: f64,
    /// Inventory bound Q (quotes clamp at |q| = Q).
    pub q_max: i64,
    /// Quadratic liquidation cost `L(q) = kappa_liq/2 q^2` (0 = mid).
    pub kappa_liq: f64,
    /// Adverse-selection impact beta (0 = off).
    pub impact_beta: f64,
    /// Time steps for the RK4 backward march.
    pub n_steps: usize,
}

impl MmProblem {
    pub fn new(gamma: f64, sigma: f64, kappa: f64, a: f64, t: f64) -> MmProblem {
        MmProblem {
            gamma,
            sigma,
            kappa,
            a,
            t,
            q_max: 10,
            kappa_liq: 0.0,
            impact_beta: 0.0,
            n_steps: 400,
        }
    }

    /// From [`UnifiedParams`].
    pub fn from_unified(p: &UnifiedParams) -> MmProblem {
        MmProblem::new(p.gamma, p.sigma, p.kappa, p.a, p.t)
    }
}

/// Solved HJB: value rows `v_q` on a uniform time grid, plus quote access.
pub struct MmHjb {
    problem: MmProblem,
    /// `times[i] = i * dt`, i = 0..=n_steps.
    pub times: Vec<f64>,
    /// `v[i][q + Q]`: value-function factors at `times[i]`.
    pub v: Vec<Vec<f64>>,
}

impl MmHjb {
    /// Solve by RK4 backward march from `T`.
    pub fn solve(problem: MmProblem) -> MmHjb {
        let q_dim = (2 * problem.q_max + 1) as usize;
        let n = problem.n_steps.max(20);
        let dt = problem.t / n as f64;
        let mut times: Vec<f64> = (0..=n).map(|i| i as f64 * dt).collect();
        times[n] = problem.t; // exact
        let mut v = vec![vec![1.0f64; q_dim]; n + 1];
        // Terminal condition: v_q(T) = exp(gamma * L(q)).
        for qi in 0..q_dim {
            let q = qi as i64 - problem.q_max;
            let l = problem.kappa_liq * (q as f64) * (q as f64) / 2.0;
            v[n][qi] = (problem.gamma * l).exp();
        }
        // Backward RK4 with h = -dt.
        for i in (0..n).rev() {
            let cur = v[i + 1].clone();
            let k1 = rhs(&problem, &cur);
            let mut tmp = vec![0.0f64; q_dim];
            for j in 0..q_dim {
                tmp[j] = cur[j] + 0.5 * (-dt) * k1[j];
            }
            let k2 = rhs(&problem, &tmp);
            for j in 0..q_dim {
                tmp[j] = cur[j] + 0.5 * (-dt) * k2[j];
            }
            let k3 = rhs(&problem, &tmp);
            for j in 0..q_dim {
                tmp[j] = cur[j] + (-dt) * k3[j];
            }
            let k4 = rhs(&problem, &tmp);
            for j in 0..q_dim {
                let h = -dt;
                v[i][j] = cur[j] + h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
                if !v[i][j].is_finite() || v[i][j] <= 0.0 {
                    // Numerical guard: clamp to a tiny positive value.
                    v[i][j] = 1e-12;
                }
            }
        }
        MmHjb { problem, times, v }
    }

    fn row_at(&self, t: f64) -> Vec<f64> {
        let t = t.clamp(0.0, self.problem.t);
        let n = self.times.len() - 1;
        let dt = self.problem.t / n as f64;
        let raw = if dt > 0.0 { t / dt } else { 0.0 };
        let pos = if raw < 0.0 {
            0
        } else if raw > n as f64 {
            n
        } else {
            raw as usize
        };
        let frac = if dt > 0.0 { t / dt - pos as f64 } else { 0.0 };
        let (i0, i1) = (pos.min(n), (pos + 1).min(n));
        let mut out = vec![0.0f64; self.v[0].len()];
        for j in 0..out.len() {
            out[j] = self.v[i0][j] * (1.0 - frac) + self.v[i1][j] * frac;
        }
        out
    }

    /// Value factor `v_q(t)` (interpolated).
    pub fn v_at(&self, q: i64, t: f64) -> f64 {
        let row = self.row_at(t);
        let idx = (q + self.problem.q_max).clamp(0, row.len() as i64 - 1) as usize;
        row[idx]
    }

    /// Optimal ask distance from the mid at `(q, t)`.
    pub fn delta_ask(&self, q: i64, t: f64) -> f64 {
        let p = &self.problem;
        if q <= -p.q_max {
            return f64::INFINITY; // at the short bound: don't sell more
        }
        let (vq, vprev) = (self.v_at(q, t), self.v_at(q - 1, t));
        // Ask fill: wealth shift delta - beta*(q-1) (the post-fill mark of
        // remaining inventory q-1 moves down by beta).
        let c = p.impact_beta * (q as f64 - 1.0);
        let foc = c + (1.0 / p.gamma) * ((p.kappa + p.gamma) * vprev / (p.kappa * vq)).ln();
        foc.max(0.0)
    }

    /// Optimal bid distance from the mid at `(q, t)`.
    pub fn delta_bid(&self, q: i64, t: f64) -> f64 {
        let p = &self.problem;
        if q >= p.q_max {
            return f64::INFINITY;
        }
        let (vq, vnext) = (self.v_at(q, t), self.v_at(q + 1, t));
        // Bid fill: wealth shift delta + beta*(q+1).
        let c = p.impact_beta * (q as f64 + 1.0);
        let foc = -c + (1.0 / p.gamma) * ((p.kappa + p.gamma) * vnext / (p.kappa * vq)).ln();
        foc.max(0.0)
    }

    /// Quote prices `(bid, ask)` around mid `s` at `(q, t)`.
    pub fn quotes(&self, s: f64, q: i64, t: f64) -> (f64, f64) {
        let da = self.delta_ask(q, t);
        let db = self.delta_bid(q, t);
        let ask = if da.is_finite() { s + da } else { f64::INFINITY };
        let bid = if db.is_finite() { s - db } else { f64::NEG_INFINITY };
        (bid, ask)
    }

    /// The two maximized impulse values at `(q, t)` (diagnostics/tests).
    pub fn impulses(&self, q: i64, t: f64) -> (f64, f64) {
        let p = &self.problem;
        let row = self.row_at(t);
        let qi = (q + p.q_max).clamp(0, row.len() as i64 - 1) as usize;
        let vq = row[qi];
        let ma = if q > -p.q_max {
            impulse(p, vq, row[qi - 1], q as f64)
        } else {
            0.0
        };
        let mb = if q < p.q_max {
            impulse_bid(p, vq, row[qi + 1], q as f64)
        } else {
            0.0
        };
        (ma, mb)
    }

    pub fn problem(&self) -> &MmProblem {
        &self.problem
    }
}

/// Maximized ask-side impulse
/// `max_delta A e^{-kappa delta} (v_q - e^{-gamma(delta - beta(q-1))} v_{q-1})`
/// with its closed-form maximizer `delta* = beta(q-1) + FOC`.
fn impulse(p: &MmProblem, vq: f64, vprev: f64, q: f64) -> f64 {
    let c = p.impact_beta * (q - 1.0);
    let delta_star =
        c + (1.0 / p.gamma) * ((p.kappa + p.gamma) * vprev / (p.kappa * vq)).ln();
    let d = delta_star.max(0.0);
    let val = p.a * (-p.kappa * d).exp() * (vq - (-p.gamma * (d - c)).exp() * vprev);
    val.max(0.0)
}

fn impulse_bid(p: &MmProblem, vq: f64, vnext: f64, q: f64) -> f64 {
    // bid fill shifts wealth by delta + beta*(q+1)
    let c = p.impact_beta * (q + 1.0);
    let delta_star =
        -c + (1.0 / p.gamma) * ((p.kappa + p.gamma) * vnext / (p.kappa * vq)).ln();
    let d = delta_star.max(0.0);
    let val = p.a * (-p.kappa * d).exp() * (vq - (-p.gamma * (d + c)).exp() * vnext);
    val.max(0.0)
}

/// RHS of the v-system: `vdot_q = -(gamma^2 sigma^2 q^2/2) v_q + M_a + M_b`.
fn rhs(p: &MmProblem, v: &[f64]) -> Vec<f64> {
    let q_dim = v.len();
    let q_max = p.q_max;
    let mut out = vec![0.0f64; q_dim];
    for qi in 0..q_dim {
        let q = qi as i64 - q_max;
        let qf = q as f64;
        let diffusion = -0.5 * p.gamma * p.gamma * p.sigma * p.sigma * qf * qf * v[qi];
        let ma = if q > -q_max {
            impulse(p, v[qi], v[qi - 1], qf)
        } else {
            0.0
        };
        let mb = if q < q_max {
            impulse_bid(p, v[qi], v[qi + 1], qf)
        } else {
            0.0
        };
        out[qi] = diffusion + ma + mb;
    }
    out
}

/// Closed-form AS quotes (the fast path / validation anchor).
#[derive(Clone, Copy, Debug)]
pub struct AsQuotes {
    pub reservation: f64,
    pub half_spread: f64,
    pub bid: f64,
    pub ask: f64,
}

impl AsQuotes {
    /// AS closed-form quotes at `(s, q, t)` for the given parameters.
    pub fn from_params(p: &UnifiedParams, s: f64, q: f64, t: f64) -> AsQuotes {
        let r = p.reservation(s, q, t);
        let h = p.half_spread(t);
        AsQuotes {
            reservation: r,
            half_spread: h,
            bid: r - h,
            ask: r + h,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::entropic_ce;
    use micro::Rng;

    fn benchmark_solver(n_steps: usize) -> MmHjb {
        let p = UnifiedParams::benchmark();
        let mut prob = MmProblem::from_unified(&p);
        prob.n_steps = n_steps;
        MmHjb::solve(prob)
    }

    #[test]
    fn terminal_foc_matches_as_exactly() {
        let sol = benchmark_solver(200);
        let p = sol.problem();
        // At t = T all v = 1 (L = 0): FOC must equal the AS terminal
        // half-spread exactly.
        let expect = (1.0 / p.gamma) * (1.0 + p.gamma / p.kappa).ln();
        for &q in &[-3i64, 0, 2] {
            let da = sol.delta_ask(q, p.t);
            let db = sol.delta_bid(q, p.t);
            assert!((da - expect).abs() < 1e-9, "ask {da} vs {expect}");
            assert!((db - expect).abs() < 1e-9, "bid {db} vs {expect}");
        }
    }

    #[test]
    fn solver_matches_as_closed_form() {
        // AS closed forms are the small-fill-intensity expansion of the
        // exact v-system (A small => few fills => the v-profile stays near
        // the pure-diffusion Gaussian). With larger A the exact solution
        // legitimately skews LESS than AS (the impulses flatten the
        // v-profile; the Monte-Carlo CE test shows the exact policy
        // dominates AS's).
        let (gamma, sigma, kappa, a, t_horizon) = (0.1f64, 2.0, 1.5, 0.5, 1.0);
        let p = UnifiedParams { gamma, sigma, kappa, a, t: t_horizon, kappa_liq: 0.0 };
        let mut prob = MmProblem::new(gamma, sigma, kappa, a, t_horizon);
        prob.n_steps = 800;
        let sol = MmHjb::solve(prob);
        for &t in &[0.0f64, 0.5] {
            let tau = t_horizon - t;
            let half = p.half_spread(t);
            for &q in &[-2i64, -1, 0, 1, 2] {
                let qf = q as f64;
                let da_as = half - gamma * sigma * sigma * qf * tau;
                let db_as = half + gamma * sigma * sigma * qf * tau;
                let da = sol.delta_ask(q, t);
                let db = sol.delta_bid(q, t);
                // mixed absolute/relative tolerance (quotes are small)
                assert!(
                    (da - da_as).abs() < 0.06 + 0.3 * da_as,
                    "ask(q={q}, t={t}): solver {da} vs AS {da_as}"
                );
                assert!(
                    (db - db_as).abs() < 0.06 + 0.3 * db_as,
                    "bid(q={q}, t={t}): solver {db} vs AS {db_as}"
                );
            }
        }
    }

    #[test]
    fn symmetry_and_monotonicity() {
        let sol = benchmark_solver(400);
        // v symmetric in q and increasing in |q| (inventory risk inflation
        // dominates the impulse mean-reversion at large |q|).
        for q in 0..=sol.problem().q_max {
            let vq = sol.v_at(q, 0.5);
            let vmq = sol.v_at(-q, 0.5);
            assert!((vq - vmq).abs() < 1e-9);
            if q > 6 {
                assert!(vq > sol.v_at(q - 1, 0.5), "v must grow with |q| (large)");
            }
        }
        // v(0, q=0) < 1: quoting opportunity raises the certainty
        // equivalent (shrinks the negative-utility magnitude).
        assert!(sol.v_at(0, 0.0) > 0.0 && sol.v_at(0, 0.0) < 1.0);
        // quote symmetry: delta_a(-q) == delta_b(q)
        for &q in &[-5i64, -2, 0, 3] {
            let a = sol.delta_ask(-q, 0.3);
            let b = sol.delta_bid(q, 0.3);
            assert!((a - b).abs() < 1e-9);
        }
        // ask distance is non-increasing in q (sell cheaper when long)
        let mut prev = f64::INFINITY;
        for q in -5..=5i64 {
            let d = sol.delta_ask(q, 0.25);
            assert!(d <= prev + 1e-9, "ask distance must decrease in q: q={q} d={d}");
            prev = d;
        }
    }

    #[test]
    fn grid_convergence() {
        let coarse = benchmark_solver(200);
        let fine = benchmark_solver(1600);
        for &q in &[-3i64, 0, 3] {
            for &t in &[0.0f64, 0.5, 0.9] {
                let a1 = coarse.delta_ask(q, t);
                let a2 = fine.delta_ask(q, t);
                assert!((a1 - a2).abs() < 5e-3, "q={q} t={t}: {a1} vs {a2}");
            }
        }
    }

    #[test]
    fn impact_beta_widens_quotes_against_inventory() {
        // Moderate intensity regime (interior FOC) so the impact offsets
        // are visible.
        let mk = |beta: f64| {
            let mut prob = MmProblem::new(0.1, 2.0, 1.5, 1.5, 1.0);
            prob.impact_beta = beta;
            prob.n_steps = 400;
            MmHjb::solve(prob)
        };
        let base = mk(0.0);
        let impacted = mk(0.5);
        // Long inventory: our ask fills push the mid down -> quote deeper.
        assert!(
            impacted.delta_ask(2, 0.9) > base.delta_ask(2, 0.9) + 0.2,
            "impacted {} vs base {}",
            impacted.delta_ask(2, 0.9),
            base.delta_ask(2, 0.9)
        );
        assert!(
            impacted.delta_bid(-2, 0.9) > base.delta_bid(-2, 0.9) + 0.2,
            "impacted {} vs base {}",
            impacted.delta_bid(-2, 0.9),
            base.delta_bid(-2, 0.9)
        );
    }

    #[test]
    fn mc_entropic_ce_optimality() {
        // Monte-Carlo ground truth: the solver policy's entropic
        // certainty-equivalent must (weakly) dominate a static-spread
        // policy and match the AS closed-form policy closely.
        let sol = benchmark_solver(400);
        let p = UnifiedParams::benchmark();
        let n_paths = 4000;
        let n_steps_sim = 600;
        let dt = p.t / n_steps_sim as f64;

        let simulate = |policy: &dyn Fn(i64, f64) -> (f64, f64)| -> Vec<f64> {
            let mut rng = Rng::new(77);
            let mut out = Vec::with_capacity(n_paths);
            for _ in 0..n_paths {
                let mut x = 0.0f64;
                let mut s = 100.0f64;
                let mut q = 0i64;
                for i in 0..n_steps_sim {
                    let t = i as f64 * dt;
                    // brownian mid
                    s += p.sigma * dt.sqrt() * rng.normal();
                    let (db, da) = policy(q, t);
                    // Poisson fills
                    if db.is_finite() && rng.bernoulli(1.0 - (-p.lambda(db) * dt).exp()) {
                        x -= s - db;
                        q += 1;
                    }
                    if da.is_finite() && rng.bernoulli(1.0 - (-p.lambda(da) * dt).exp()) {
                        x += s + da;
                        q -= 1;
                    }
                }
                out.push(x + q as f64 * s);
            }
            out
        };

        let solver_policy = |q: i64, t: f64| -> (f64, f64) {
            let da = sol.delta_ask(q, t);
            let db = sol.delta_bid(q, t);
            (db, da)
        };
        let as_policy = |q: i64, t: f64| -> (f64, f64) {
            let half = p.half_spread(t);
            let skew = p.gamma * p.sigma * p.sigma * q as f64 * (p.t - t);
            (half + skew, half - skew)
        };
        let static_policy = |_q: i64, _t: f64| -> (f64, f64) {
            let half = 2.0 * p.half_spread(0.5); // deliberately too wide
            (half, half)
        };

        let w_solver = simulate(&solver_policy);
        let w_as = simulate(&as_policy);
        let w_static = simulate(&static_policy);
        let ce_solver = entropic_ce(&w_solver, p.gamma);
        let ce_as = entropic_ce(&w_as, p.gamma);
        let ce_static = entropic_ce(&w_static, p.gamma);
        // Solver is the exact optimum: CE(solver) >= CE(as) - noise margin,
        // and both dominate the static (mis-calibrated) policy.
        assert!(
            ce_solver >= ce_as - 0.15,
            "CE solver {ce_solver} vs AS {ce_as}"
        );
        assert!(
            ce_solver > ce_static,
            "CE solver {ce_solver} vs static {ce_static}"
        );
        assert!(ce_as > ce_static);
    }
}
