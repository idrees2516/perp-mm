//! Stoikov (2018), "The Micro-Price" — two constructions:
//!
//! 1. [`MicroPriceFit`] — practical online fit of `E[Delta mid | imbalance]`
//!    with a cubic polynomial in imbalance via recursive least squares
//!    (RLS with forgetting), symmetrized as in the paper (every sample
//!    `(I, dM)` is paired with `(1-I, -dM)`), so the fit passes through
//!    zero at `I = 0.5`. `micro = mid + f(I)`.
//!
//! 2. [`MicroPriceChain`] — the paper's full discrete-state Markov-chain
//!    construction on states `(imbalance bin, spread bin)`:
//!    ```text
//!    G1 = (I - Q)^-1 R K        (expected mid change until the first move)
//!    B  = (I - Q)^-1 T          (state chain between mid moves)
//!    G* = G1 + sum_{i>=1} B^i G1 = P_micro - mid
//!    ```
//!    with `Q` = no-move transitions, `T` = post-move state transitions,
//!    `R` = mid-change distribution. Data is symmetrized the same way.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// RLS polynomial fit
// ---------------------------------------------------------------------------

/// Online `E[Delta mid | I]` fit: features `(i - 0.5, (i-0.5)^3)`.
pub struct MicroPriceFit {
    /// Forgetting factor (1.0 = plain LS).
    pub lambda: f64,
    theta: [f64; 2],
    p: [[f64; 2]; 2],
    n: u64,
}

impl MicroPriceFit {
    pub fn new(lambda: f64) -> MicroPriceFit {
        MicroPriceFit {
            lambda,
            theta: [0.0, 0.0],
            p: [[1e4, 0.0], [0.0, 1e4]],
            n: 0,
        }
    }

    /// Record an observation: top-of-book imbalance `i` (0..1) and the
    /// realized mid change (in ticks) over the following event/interval.
    /// Both the sample and its mirror are fitted.
    pub fn update(&mut self, i: f64, d_mid_ticks: f64) {
        if !(0.0..=1.0).contains(&i) || !d_mid_ticks.is_finite() {
            return;
        }
        self.n += 1;
        self.rls(i, d_mid_ticks);
        self.rls(1.0 - i, -d_mid_ticks);
    }

    fn rls(&mut self, i: f64, y: f64) {
        let x = [i - 0.5, (i - 0.5).powi(3)];
        // x' P
        let xp = [
            x[0] * self.p[0][0] + x[1] * self.p[1][0],
            x[0] * self.p[0][1] + x[1] * self.p[1][1],
        ];
        let denom = self.lambda + x[0] * xp[0] + x[1] * xp[1];
        if denom.abs() < 1e-12 {
            return;
        }
        let k = [xp[0] / denom, xp[1] / denom];
        let yhat = x[0] * self.theta[0] + x[1] * self.theta[1];
        let err = y - yhat;
        self.theta[0] += k[0] * err;
        self.theta[1] += k[1] * err;
        // P <- (P - k x' P) / lambda
        for r in 0..2 {
            for c in 0..2 {
                self.p[r][c] = (self.p[r][c] - k[r] * xp[c]) / self.lambda;
            }
        }
    }

    /// Fitted expected mid change (ticks) at imbalance `i`.
    pub fn predict(&self, i: f64) -> f64 {
        let x = i - 0.5;
        self.theta[0] * x + self.theta[1] * x * x * x
    }

    /// Micro-price: `mid + f(I)` (all in ticks; mid may be fractional).
    pub fn micro(&self, mid_ticks: f64, i: f64) -> f64 {
        mid_ticks + self.predict(i)
    }

    pub fn samples(&self) -> u64 {
        self.n
    }
}

// ---------------------------------------------------------------------------
// Discrete-state Markov chain (the paper's construction)
// ---------------------------------------------------------------------------

const MAX_DMID_BIN: i64 = 3; // -3..=3 ticks

/// Full Markov-chain micro-price over states (imbalance bin, spread bin).
pub struct MicroPriceChain {
    pub i_bins: usize,
    pub s_bins: usize,
    /// Q counts: no-mid-move transitions.
    q_counts: HashMap<(u16, u16, u16, u16), u64>,
    /// T counts: post-mid-move state transitions.
    t_counts: HashMap<(u16, u16, u16, u16), u64>,
    /// R counts: mid changes (binned) per starting state.
    r_counts: HashMap<(u16, u16, i64), u64>,
    /// Cached G* vector, recomputed lazily.
    g_star: Vec<f64>,
    dirty: bool,
    n: u64,
}

fn bin_i(i: f64, i_bins: usize) -> u16 {
    let clamped = i.clamp(0.0, 1.0 - 1e-9);
    (clamped * i_bins as f64) as u16
}

fn bin_s(s_ticks: u64, s_bins: usize) -> u16 {
    (s_ticks.clamp(1, s_bins as u64) - 1) as u16
}

impl MicroPriceChain {
    /// `i_bins` imbalance bins x `s_bins` spread bins (1..=s_bins ticks).
    pub fn new(i_bins: usize, s_bins: usize) -> MicroPriceChain {
        MicroPriceChain {
            i_bins: i_bins.max(2),
            s_bins: s_bins.max(2),
            q_counts: HashMap::new(),
            t_counts: HashMap::new(),
            r_counts: HashMap::new(),
            g_star: Vec::new(),
            dirty: true,
            n: 0,
        }
    }

    /// Record a top-of-book transition: state `(i, spread_ticks)` moved to
    /// `(i_next, spread_next)` with mid change `d_mid_ticks` (binned to
    /// -3..=3). Symmetrization doubles every observation.
    pub fn update(
        &mut self,
        i: f64,
        spread_ticks: u64,
        i_next: f64,
        spread_next: u64,
        d_mid_ticks: f64,
    ) {
        if !(0.0..=1.0).contains(&i) || !(0.0..=1.0).contains(&i_next) {
            return;
        }
        self.n += 1;
        let d_bin = d_mid_ticks.round().clamp(-MAX_DMID_BIN as f64, MAX_DMID_BIN as f64) as i64;
        for sign in [1.0f64, -1.0f64] {
            let (a, b) = (
                bin_i(if sign > 0.0 { i } else { 1.0 - i }, self.i_bins),
                bin_s(spread_ticks, self.s_bins),
                // next state
            );
            let ni = if sign > 0.0 { i_next } else { 1.0 - i_next };
            let (c, d) = (bin_i(ni, self.i_bins), bin_s(spread_next, self.s_bins));
            let dd = (d_bin as f64 * sign).round() as i64;
            if dd == 0 {
                *self.q_counts.entry((a, b, c, d)).or_insert(0) += 1;
            } else {
                *self.t_counts.entry((a, b, c, d)).or_insert(0) += 1;
                *self.r_counts.entry((a, b, dd)).or_insert(0) += 1;
            }
        }
        self.dirty = true;
    }

    fn states(&self) -> usize {
        self.i_bins * self.s_bins
    }

    fn idx(&self, i_bin: u16, s_bin: u16) -> usize {
        i_bin as usize * self.s_bins + s_bin as usize
    }

    /// Recompute `G*` (called by [`Self::micro`]).
    pub fn recompute(&mut self) {
        let ns = self.states();
        // Build Q, T (row-normalized), R.K.
        let mut q = vec![vec![0.0f64; ns]; ns];
        let mut t = vec![vec![0.0f64; ns]; ns];
        let mut rk = vec![0.0f64; ns];
        let mut row_sum_q = vec![0.0f64; ns];
        let mut row_sum_t = vec![0.0f64; ns];
        for (&(a, b, c, d), &cnt) in &self.q_counts {
            let (fr, to) = (self.idx(a, b), self.idx(c, d));
            q[fr][to] += cnt as f64;
            row_sum_q[fr] += cnt as f64;
        }
        for (&(a, b, c, d), &cnt) in &self.t_counts {
            let (fr, to) = (self.idx(a, b), self.idx(c, d));
            t[fr][to] += cnt as f64;
            row_sum_t[fr] += cnt as f64;
        }
        for (&(a, b, d), &cnt) in &self.r_counts {
            let fr = self.idx(a, b);
            rk[fr] += d as f64 * cnt as f64;
        }
        for s in 0..ns {
            if row_sum_q[s] > 0.0 {
                for c in 0..ns {
                    q[s][c] /= row_sum_q[s];
                }
            }
            if row_sum_t[s] > 0.0 {
                for c in 0..ns {
                    t[s][c] /= row_sum_t[s];
                }
            }
            if row_sum_t[s] > 0.0 {
                rk[s] /= row_sum_t[s];
            }
        }
        // M = (I - Q)^{-1}
        let m = match invert_i_minus_q(&q, ns) {
            Some(m) => m,
            None => {
                self.g_star = rk;
                self.dirty = false;
                return;
            }
        };
        // G1 = M R K ; B = M T
        let mut g1 = vec![0.0f64; ns];
        for i in 0..ns {
            for j in 0..ns {
                g1[i] += m[i][j] * rk[j];
            }
        }
        let mut b = vec![vec![0.0f64; ns]; ns];
        for i in 0..ns {
            for j in 0..ns {
                let mut s = 0.0;
                for k2 in 0..ns {
                    s += m[i][k2] * t[k2][j];
                }
                b[i][j] = s;
            }
        }
        // G* = G1 + B G1 + B^2 G1 + ...
        let mut g = g1.clone();
        let mut bi = b.clone();
        for _ in 0..500 {
            let mut bg = vec![0.0f64; ns];
            let mut norm = 0.0f64;
            for i in 0..ns {
                let mut s = 0.0;
                for j in 0..ns {
                    s += bi[i][j] * g1[j];
                }
                bg[i] = s;
                norm += s.abs();
            }
            for i in 0..ns {
                g[i] += bg[i];
            }
            if norm < 1e-10 {
                break;
            }
            // bi <- bi * b
            let mut nb = vec![vec![0.0f64; ns]; ns];
            for i in 0..ns {
                for j in 0..ns {
                    let mut s = 0.0;
                    for k2 in 0..ns {
                        s += bi[i][k2] * b[k2][j];
                    }
                    nb[i][j] = s;
                }
            }
            bi = nb;
        }
        self.g_star = g;
        self.dirty = false;
    }

    /// Micro-price (ticks): `mid + G*(state)` — call after enough updates.
    pub fn micro(&mut self, mid_ticks: f64, i: f64, spread_ticks: u64) -> f64 {
        if self.dirty || self.g_star.is_empty() {
            self.recompute();
        }
        let s = self.idx(bin_i(i, self.i_bins), bin_s(spread_ticks, self.s_bins));
        mid_ticks + self.g_star.get(s).copied().unwrap_or(0.0)
    }

    pub fn samples(&self) -> u64 {
        self.n
    }
}

/// Invert `I - Q` via Gauss-Jordan; None if singular ( absorbing loops ).
fn invert_i_minus_q(q: &[Vec<f64>], n: usize) -> Option<Vec<Vec<f64>>> {
    let mut a = vec![vec![0.0f64; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = if i == j { 1.0 } else { 0.0 } - q[i][j];
        }
        a[i][n + i] = 1.0;
    }
    for col in 0..n {
        let mut piv = col;
        for row in col + 1..n {
            if a[row][col].abs() > a[piv][col].abs() {
                piv = row;
            }
        }
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        let d = a[col][col];
        for c in 0..2 * n {
            a[col][c] /= d;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let f = a[row][col];
            if f == 0.0 {
                continue;
            }
            for c in 0..2 * n {
                a[row][c] -= f * a[col][c];
            }
        }
    }
    let mut inv = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = a[i][n + j];
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::Rng;

    #[test]
    fn microprice_fit_learns_drift() {
        // Construct: when imbalance > 0.5 the next mid tends +1 tick and
        // vice versa (linear in imbalance). The fit must recover the sign
        // and rough magnitude.
        let mut fit = MicroPriceFit::new(0.999);
        let mut rng = Rng::new(61);
        for _ in 0..30_000 {
            let i = rng.uniform();
            let drift = (i - 0.5) * 4.0; // ticks
            let d = drift + rng.normal() * 0.5;
            fit.update(i, d);
        }
        assert!(fit.predict(0.9) > 1.0, "predict(0.9)={}", fit.predict(0.9));
        assert!(fit.predict(0.1) < -1.0, "predict(0.1)={}", fit.predict(0.1));
        assert!(fit.predict(0.5).abs() < 1e-6);
        assert!((fit.predict(0.75) - 1.0).abs() < 0.3);
    }

    #[test]
    fn microprice_chain_builds() {
        let mut chain = MicroPriceChain::new(10, 4);
        let mut rng = Rng::new(62);
        let mut mid = 100.0f64;
        let (mut qb, mut qa) = (50u64, 50u64);
        for step in 0..200_000 {
            let bid = mid - 0.5;
            let ask = mid + 0.5;
            let i = qb as f64 / (qb + qa) as f64;
            // Imbalance pushes the mid: strong buy imbalance -> mid up.
            let d_mid = (i - 0.5) * 2.0 + rng.normal() * 0.3;
            let d_bin = d_mid.round();
            if d_bin != 0.0 {
                mid += d_bin;
            }
            // sizes mean-revert
            qb = (qb as i64 + rng.below(7) as i64 - 3).clamp(5, 200) as u64;
            qa = (qa as i64 + rng.below(7) as i64 - 3).clamp(5, 200) as u64;
            let i_next = qb as f64 / (qb + qa) as f64;
            let _ = (bid, ask, step);
            chain.update(i, 1, i_next, 1, d_mid);
        }
        // micro at high imbalance should exceed mid; at low, below.
        let hi = chain.micro(100.0, 0.9, 1);
        let lo = chain.micro(100.0, 0.1, 1);
        assert!(hi > 100.05, "hi={hi}");
        assert!(lo < 99.95, "lo={lo}");
        assert!(hi > lo);
    }
}
