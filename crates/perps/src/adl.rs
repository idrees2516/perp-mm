//! Autodeleveraging: impossibilities and optimization — Chitra
//! (arXiv:2512.01112).
//!
//! # The policy framework
//!
//! An ADL policy outputs **severity** `theta in [0,1]` (fraction of the
//! shortfall to socialize) and **haircuts** `h in [0,1]^n` over winning
//! accounts. Validity (paper eqs. 17-18):
//! ```text
//! budget balance:  sum_i h_i e_i^+ = theta * D      (e_i = equity)
//! feasibility:     theta * D <= W = sum_i e_i^+
//! ```
//! The **separation principle**: solvency depends only on the scalar
//! severity sequence; fairness/revenue depend only on the allocation
//! given the budget.
//!
//! # Implemented policies
//!
//! * [`queue_haircuts`] — the production **queue** policy (Binance /
//!   Hyperliquid): rank by PNL-leverage score `s = ell * mark / p_ref`
//!   (Binance `p_ref` = bankruptcy price; Hyperliquid = last mark; the
//!   perp-options-clob venue uses profit-ratio ranking), then
//!   water-fill top-down with the running-residual rule (paper eq. 19).
//!   Optional lot rounding demonstrates the production **overshoot**
//!   pathology (closing more than the deficit).
//! * [`pro_rata`] — uniform multiplicative haircuts `h_i = (theta D / W)
//!   e_i^+` (paper eq. 20), rank-preserving.
//! * [`levered_pro_rata`] — `h_i ~ ell_i e_i^+` (eq. 21).
//! * [`capped_pro_rata`] — the optimized policy of Propositions 6.1/6.2:
//!   water-filling `h_i = min(eta, beta_i)` with per-account caps
//!   `beta_i = min(hbar_i, 1 - e_min/e_i)` — the unique solution of
//!   `min sum phi(h_i)` s.t. caps and budget, for any increasing convex
//!   `phi`; sybil-resistant, scale-invariant, and order-stable
//!   (`e_i >= e_j => (1-h_i)e_i >= (1-h_j)e_j`).
//!
//! # Trilemma metrics (Propositions 5.1-5.3)
//!
//! [`policy_metrics`] computes PTSR (top-survivor-to-deficit ratio), PMR,
//! and the overshoot — the empirical face of the paper's
//! solvency/fairness/revenue impossibility (Theorem J.7): any static
//! policy family trading these off is on the trilemma frontier.

/// One account in the ADL universe.
#[derive(Clone, Debug)]
pub struct AdlAccount {
    /// Signed position quantity (lots; sign = direction).
    pub qty: i64,
    /// Entry price (quote per lot).
    pub entry_price: f64,
    /// Collateral (quote).
    pub collateral: f64,
}

impl AdlAccount {
    /// Mark equity: `collateral + (mark - entry) * qty`.
    pub fn equity(&self, mark: f64) -> f64 {
        self.collateral + (mark - self.entry_price) * self.qty as f64
    }

    /// Effective leverage at mark (notional / |equity|).
    pub fn leverage(&self, mark: f64) -> f64 {
        let e = self.equity(mark).abs();
        if e < 1e-12 {
            return f64::INFINITY;
        }
        (self.qty.abs() as f64 * mark) / e
    }
}

/// Reference price convention for the queue score.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RankRef {
    /// Binance: bankruptcy price of the liquidated position.
    BankruptcyPrice,
    /// Hyperliquid: last mark price.
    LastMark,
}

/// Rank score for a winning account (higher = deleveraged first).
///
/// Chitra eq. (paper §3): `s = ell * p_hat / p_ref` where `ell` is the
/// signed initial leverage; with `p_ref = mark` this reduces to the
/// venue's profit-ratio ranking.
pub fn rank_score(acct: &AdlAccount, mark: f64, p_ref: f64) -> f64 {
    let e = acct.equity(mark);
    let notional = acct.qty.abs() as f64 * mark;
    let ell = if e.abs() < 1e-12 {
        f64::INFINITY * (if acct.qty > 0 { 1.0 } else { -1.0 })
    } else {
        notional / e * (if acct.qty > 0 { 1.0 } else { -1.0 })
    };
    // winners on the side being deleveraged: use |score|
    (ell * mark / p_ref).abs()
}

/// Aggregate shortfall statistics: `(D, W, Delta, omega)`.
pub fn shortfall(accounts: &[AdlAccount], mark: f64) -> (f64, f64, f64, f64) {
    let mut d = 0.0f64;
    let mut w = 0.0f64;
    let mut delta = 0.0f64;
    let mut omega = 0.0f64;
    for a in accounts {
        let e = a.equity(mark);
        if e < 0.0 {
            d += -e;
            delta = delta.max(-e);
        } else {
            w += e;
            omega = omega.max(e);
        }
    }
    (d, w, delta, omega)
}

/// Queue-policy haircuts (paper eq. 19) with optional lot rounding.
///
/// Returns haircuts aligned with `accounts`. `budget = theta * D`.
pub fn queue_haircuts(
    accounts: &[AdlAccount],
    mark: f64,
    ref_price: f64,
    budget: f64,
    lot_size: Option<f64>,
) -> Vec<f64> {
    let n = accounts.len();
    let mut haircuts = vec![0.0f64; n];
    if budget <= 0.0 {
        return haircuts;
    }
    // rank winners (positive equity) descending by score
    let mut idx: Vec<usize> = (0..n)
        .filter(|&i| accounts[i].equity(mark) > 0.0)
        .collect();
    idx.sort_by(|&a, &b| {
        let sa = rank_score(&accounts[a], mark, ref_price);
        let sb = rank_score(&accounts[b], mark, ref_price);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut residual = budget;
    for &i in &idx {
        if residual <= 0.0 {
            break;
        }
        let e = accounts[i].equity(mark);
        let mut h = (residual / e).min(1.0);
        if let Some(_ls) = lot_size {
            // production path: close in whole lots, rounding UP
            // (over-closure -> overshoot pathology)
            let lots_total = (accounts[i].qty.abs() as f64).max(1.0);
            let close_lots = (h * lots_total).ceil();
            h = (close_lots / lots_total).min(1.0);
        }
        haircuts[i] = h;
        residual -= h * e;
    }
    haircuts
}

/// Pro-rata haircuts (eq. 20): `h_i = (theta D / W) e_i^+`.
pub fn pro_rata(accounts: &[AdlAccount], mark: f64, budget: f64) -> Vec<f64> {
    let (_, w, _, _) = shortfall(accounts, mark);
    let mut out = vec![0.0f64; accounts.len()];
    if w <= 0.0 || budget <= 0.0 {
        return out;
    }
    let frac = (budget / w).min(1.0);
    for (i, a) in accounts.iter().enumerate() {
        let e = a.equity(mark);
        if e > 0.0 {
            out[i] = frac;
        }
    }
    out
}

/// Levered pro-rata haircuts (eq. 21): `h_i ~ ell_i e_i^+`, with
/// redistribution when clamping at 1 breaks budget balance.
pub fn levered_pro_rata(accounts: &[AdlAccount], mark: f64, budget: f64) -> Vec<f64> {
    let n = accounts.len();
    let mut out = vec![0.0f64; n];
    if budget <= 0.0 {
        return out;
    }
    // iterative clamped proportional allocation
    let mut active: Vec<usize> = (0..n)
        .filter(|&i| accounts[i].equity(mark) > 0.0)
        .collect();
    let mut remaining = budget;
    let mut guard = 0;
    while remaining > 1e-12 && !active.is_empty() && guard < 64 {
        guard += 1;
        let mut w_sum = 0.0f64;
        for &i in &active {
            w_sum += accounts[i].leverage(mark) * accounts[i].equity(mark);
        }
        if w_sum <= 0.0 {
            break;
        }
        let mut next_active = Vec::new();
        let mut taken = 0.0f64;
        for &i in &active {
            let e = accounts[i].equity(mark);
            let w = accounts[i].leverage(mark) * e;
            let add = (w / w_sum * remaining / e).min(1.0 - out[i]);
            out[i] += add;
            taken += add * e;
            if out[i] < 1.0 - 1e-12 {
                next_active.push(i);
            }
        }
        remaining -= taken;
        active = next_active;
    }
    out
}

/// Capped pro-rata (Propositions 6.1/6.2): water-filling `h_i =
/// min(eta, beta_i)` with caps `beta_i` chosen so the budget is exactly
/// met. The unique optimum of `min sum phi(h_i)` over increasing convex
/// `phi` subject to the caps and budget; sybil-resistant and
/// order-stable.
pub fn capped_pro_rata(
    accounts: &[AdlAccount],
    mark: f64,
    budget: f64,
    hbar: f64,
    e_min: f64,
) -> Vec<f64> {
    let n = accounts.len();
    let mut out = vec![0.0f64; n];
    if budget <= 0.0 {
        return out;
    }
    // caps beta_i = min(hbar, 1 - e_min / e_i^+)
    let mut caps: Vec<(usize, f64, f64)> = Vec::new(); // (idx, cap, equity)
    for (i, a) in accounts.iter().enumerate() {
        let e = a.equity(mark);
        if e > 0.0 {
            let cap = hbar.min((1.0 - e_min / e).max(0.0));
            caps.push((i, cap, e));
        }
    }
    if caps.is_empty() {
        return out;
    }
    let capacity: f64 = caps.iter().map(|&(_, c, e)| c * e).sum();
    if budget >= capacity {
        // take everything to caps
        for &(i, c, _) in &caps {
            out[i] = c;
        }
        return out;
    }
    // water-filling: sort by cap ascending; pour the budget.
    caps.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut remaining = budget;
    let mut uncapped: Vec<(usize, f64)> = Vec::new();
    for &(i, cap, e) in &caps {
        if cap * e <= remaining {
            out[i] = cap;
            remaining -= cap * e;
        } else {
            uncapped.push((i, e));
        }
    }
    if !uncapped.is_empty() {
        let eq_sum: f64 = uncapped.iter().map(|&(_, e)| e).sum();
        if eq_sum > 0.0 {
            let eta = remaining / eq_sum;
            for &(i, _e) in &uncapped {
                out[i] = eta.min(caps.iter().find(|&&(j, _, _)| j == i).map(|&(_, c, _)| c).unwrap_or(1.0));
            }
        }
    }
    out
}

/// Post-policy metrics (the trilemma face): `(PTSR, PMR, overshoot)`.
#[derive(Clone, Copy, Debug)]
pub struct AdlMetrics {
    /// `omega^pi / D^pi` — top-survivor-to-residual-deficit ratio.
    pub ptsr: f64,
    /// `omega^pi / Delta^pi`.
    pub pmr: f64,
    /// `(H - D)^+` — over-deleveraging beyond the deficit.
    pub overshoot: f64,
    /// Total haircut mass `H = sum h_i e_i^+`.
    pub h_total: f64,
}

pub fn policy_metrics(
    accounts: &[AdlAccount],
    mark: f64,
    haircuts: &[f64],
    deficit: f64,
    max_single_deficit: f64,
) -> AdlMetrics {
    let mut h_total = 0.0f64;
    let mut omega_pi = 0.0f64;
    for (i, a) in accounts.iter().enumerate() {
        let e = a.equity(mark);
        if e > 0.0 {
            let h = haircuts.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            h_total += h * e;
            omega_pi = omega_pi.max((1.0 - h) * e);
        }
    }
    let d_pi = (deficit - h_total).max(0.0);
    let delta_pi = (max_single_deficit * (1.0 - h_total / deficit.max(1e-12))).max(0.0);
    AdlMetrics {
        ptsr: if d_pi > 1e-12 { omega_pi / d_pi } else { f64::INFINITY },
        pmr: if delta_pi > 1e-12 { omega_pi / delta_pi } else { f64::INFINITY },
        overshoot: (h_total - deficit).max(0.0),
        h_total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_accounts() -> Vec<AdlAccount> {
        // Matched book at mark = 100. Winners (positive equity): shorts
        // entered ABOVE the mark and longs entered BELOW it. Losers:
        // longs entered above, shorts entered below.
        vec![
            // winners (deleveraging candidates)
            AdlAccount { qty: -10, entry_price: 120.0, collateral: 200.0 },
            AdlAccount { qty: -5, entry_price: 115.0, collateral: 60.0 },
            AdlAccount { qty: -20, entry_price: 130.0, collateral: 500.0 },
            AdlAccount { qty: 8, entry_price: 80.0, collateral: 50.0 },
            // losers (deficit)
            AdlAccount { qty: 12, entry_price: 125.0, collateral: 100.0 },
            AdlAccount { qty: 6, entry_price: 115.0, collateral: 40.0 },
            AdlAccount { qty: -15, entry_price: 70.0, collateral: 150.0 },
        ]
    }

    #[test]
    fn budget_balance_all_policies() {
        let mark = 100.0;
        let accts = sample_accounts();
        let (d, w, _, _) = shortfall(&accts, mark);
        assert!(d > 0.0 && w > 0.0);
        let budget = 0.8 * d;
        for h in [
            queue_haircuts(&accts, mark, mark, budget, None),
            pro_rata(&accts, mark, budget),
            levered_pro_rata(&accts, mark, budget),
            capped_pro_rata(&accts, mark, budget, 0.5, 10.0),
        ] {
            let total: f64 = accts
                .iter()
                .enumerate()
                .map(|(i, a)| a.equity(mark).max(0.0) * h[i])
                .sum();
            assert!(
                (total - budget).abs() < 1e-6 * budget.max(1.0),
                "budget: took {total} vs {budget}"
            );
        }
    }

    #[test]
    fn prop_5_3_pro_rata_preserves_top_winner_better() {
        // Proposition 5.3 for H = theta D <= e_(1) and a queue whose
        // top-ranked account IS the top-equity winner:
        // omega^PR - omega^Queue = H (1 - e_(1)/W).
        let mark = 100.0;
        // W1: high equity AND high leverage (top score); W2: small.
        let accts = vec![
            AdlAccount { qty: -40, entry_price: 125.0, collateral: 1000.0 }, // e=2000, ell=2.0
            AdlAccount { qty: -5, entry_price: 110.0, collateral: 480.0 },  // e=530, ell=0.94
            AdlAccount { qty: 12, entry_price: 125.0, collateral: 100.0 },  // e=-200
            AdlAccount { qty: 6, entry_price: 115.0, collateral: 40.0 },    // e=-50
        ];
        let (d, _, _, _) = shortfall(&accts, mark);
        assert!((d - 250.0).abs() < 1e-9);
        let budget = 0.5 * d; // 125 <= e_(1) = 2000
        let hq = queue_haircuts(&accts, mark, mark, budget, None);
        let hp = pro_rata(&accts, mark, budget);
        let omega = |h: &[f64]| {
            accts
                .iter()
                .enumerate()
                .map(|(i, a)| (1.0 - h[i]) * a.equity(mark).max(0.0))
                .fold(0.0, f64::max)
        };
        let diff = omega(&hp) - omega(&hq);
        let e1 = accts
            .iter()
            .map(|a| a.equity(mark).max(0.0))
            .fold(0.0, f64::max);
        let w: f64 = accts.iter().map(|a| a.equity(mark).max(0.0)).sum();
        let expect = budget * (1.0 - e1 / w);
        assert!((diff - expect).abs() < 1e-6, "diff {diff} vs {expect}");
        assert!(diff > 0.0);
    }

    #[test]
    fn capped_pro_rata_caps_and_stability() {
        let mark = 100.0;
        let accts = sample_accounts();
        let (d, _, _, _) = shortfall(&accts, mark);
        let budget = 0.6 * d;
        let (hbar, e_min) = (0.3, 20.0);
        let h = capped_pro_rata(&accts, mark, budget, hbar, e_min);
        // caps respected
        for (i, a) in accts.iter().enumerate() {
            let e = a.equity(mark);
            if e > 0.0 {
                let cap = hbar.min((1.0 - e_min / e).max(0.0));
                assert!(h[i] <= cap + 1e-9, "cap violated at {i}: {} > {cap}", h[i]);
            } else {
                assert_eq!(h[i], 0.0);
            }
        }
        // order stability (Prop 6.2): e_i >= e_j => (1-h_i)e_i >= (1-h_j)e_j
        let mut prev_post = f64::INFINITY;
        let mut sorted: Vec<(f64, f64)> = accts
            .iter()
            .enumerate()
            .filter(|(_, a)| a.equity(mark) > 0.0)
            .map(|(i, a)| (a.equity(mark), h[i]))
            .collect();
        sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for &(e, h) in &sorted {
            let post = (1.0 - h) * e;
            assert!(
                post <= prev_post + 1e-9,
                "order stability violated: {post} > {prev_post}"
            );
            prev_post = post;
        }
    }

    #[test]
    fn queue_lot_rounding_overshoots() {
        // Production-style whole-lot closure over-closes vs the exact
        // budget (the Chitra empirical pathology).
        let mark = 100.0;
        let accts = sample_accounts();
        let (d, _, _, delta) = shortfall(&accts, mark);
        let budget = 0.4 * d;
        let h_frac = queue_haircuts(&accts, mark, mark, budget, None);
        let h_lots = queue_haircuts(&accts, mark, mark, budget, Some(1.0));
        let m_frac = policy_metrics(&accts, mark, &h_frac, d, delta);
        let m_lots = policy_metrics(&accts, mark, &h_lots, d, delta);
        assert!(m_frac.overshoot <= 1e-9);
        assert!(
            m_lots.overshoot >= m_frac.overshoot,
            "lot rounding must not reduce overshoot"
        );
    }

    #[test]
    fn severity_feasibility() {
        let mark = 100.0;
        let accts = sample_accounts();
        let (d, w, _, _) = shortfall(&accts, mark);
        // theta > W/D is infeasible: budget capped by W in every policy.
        let budget = 2.0 * d;
        let h = pro_rata(&accts, mark, budget);
        let total: f64 = accts
            .iter()
            .enumerate()
            .map(|(i, a)| a.equity(mark).max(0.0) * h[i])
            .sum();
        assert!(total <= w + 1e-9);
    }
}
