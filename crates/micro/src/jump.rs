//! Real-time jump detection for high-frequency prices.
//!
//! Three complementary tests:
//!
//! 1. [`LeeMykland`] — Lee & Mykland (2008), "Jumps in financial markets: a
//!    new nonparametric test and jump dynamics" (RFS 21(6)). Streaming
//!    statistic `L(i) = |r_i| / sqrt(V(i))` with rolling bipower
//!    volatility; the normalized max converges to a standard Gumbel under
//!    the no-jump null:
//!    ```text
//!    V(i)  = (pi/2) * (1/(K-1)) * sum_{j=i-K+2..i} |r_{j-1}| |r_j|
//!    C(M)  = sqrt(2 ln M) - (ln pi + ln ln M) / (2 sqrt(2 ln M))
//!    S(M)  = 1 / sqrt(2 ln M)
//!    jump  <=>  (L(i) - C(M)) / S(M) > q_{1-alpha} = -ln(-ln(1-alpha))
//!    ```
//!
//! 2. [`BnsRatioTest`] — Barndorff-Nielsen & Shephard-style realized /
//!    bipower ratio test over a return window: `RV = sum r^2`,
//!    `BV = (pi/2) sum |r_{j-1} r_j|`. Self-contained variance derivation
//!    (recovering exactly the BNS constant theta = pi^2/4 + pi - 5;
//!    Monte-Carlo size-validated in the test suite): for iid N(0, s^2)
//!    returns, `Var(BV - RV) = theta * n * s^4`; plugging in the
//!    jump-robust `s^2 ~ BV/n` gives the standardization
//!    `z = (BV - RV) / sqrt(theta * BV^2 / n)`.
//!
//! 3. [`BhrJumpTest`] — Bibinger–Hautsch–Ristig (arXiv:2403.00819), "Jump
//!    detection in high-frequency order prices": designed for *order
//!    prices* (best bid/ask) observed with **one-sided** microstructure
//!    noise. Uses block minima `m_k` (noise-free asymptotics), the
//!    min-based spot variance
//!    `sigma2_k = pi / (2 (pi-2) K) * sum_j (m_{j} - m_{j-1})^2 / h`,
//!    the global statistic `T = max_k |m_k - m_{k-1}| / sigma_k` and the
//!    Gumbel normalization `n^{1/3} T - B_n`, with
//!    `B(m) = 2 ln m - ln(pi ln m)`, `m = 2/h - 2`. Also provides jump
//!    localization and size estimation `Delta_X = min-after - min-before`.

use crate::special::{gumbel_ppf, norm_ppf};

// ---------------------------------------------------------------------------
// Lee & Mykland (2008)
// ---------------------------------------------------------------------------

/// Streaming Lee–Mykland jump detector over log returns.
pub struct LeeMykland {
    /// Window size K (number of returns in the bipower estimator).
    pub window: usize,
    /// Number of observations in the reference period M (for the Gumbel
    /// normalization; e.g. one trading day of bars).
    pub period_m: usize,
    /// Significance level.
    pub alpha: f64,
    buf: std::collections::VecDeque<f64>,
    /// Cached Gumbel-normalized threshold on the raw statistic L.
    threshold_l: f64,
    last: Option<LmResult>,
}

/// Result of one Lee–Mykland update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LmResult {
    /// Bipower variance estimate (window, current return included).
    pub sigma2: f64,
    /// Statistic `L = |r| / sqrt(V)`.
    pub stat: f64,
    /// Gumbel-normalized `(L - C(M)) / S(M)`.
    pub normalized: f64,
    /// Jump flag at the 1-alpha level.
    pub is_jump: bool,
}

impl LeeMykland {
    /// New detector with window K and reference period M observations.
    pub fn new(window: usize, period_m: usize, alpha: f64) -> LeeMykland {
        let m = period_m.max(10) as f64;
        let ln_m = m.ln();
        let c = (2.0 * ln_m).sqrt() - (std::f64::consts::PI.ln() + ln_m.ln())
            / (2.0 * (2.0 * ln_m).sqrt());
        let s = 1.0 / (2.0 * ln_m).sqrt();
        let q = gumbel_ppf(1.0 - alpha);
        LeeMykland {
            window: window.max(4),
            period_m,
            alpha,
            buf: std::collections::VecDeque::with_capacity(window + 1),
            threshold_l: c + s * q,
            last: None,
        }
    }

    /// Push the next log return; returns `None` until the window is full.
    pub fn update(&mut self, r: f64) -> Option<LmResult> {
        self.buf.push_back(r);
        if self.buf.len() > self.window {
            self.buf.pop_front();
        }
        if self.buf.len() < self.window {
            return None;
        }
        // V(i) = (pi/2) * (1/(K-1)) * sum_{j} |r_{j-1}||r_j| over the K-1
        // consecutive pairs inside the window (includes the current return).
        let mut sum = 0.0f64;
        let pts: Vec<f64> = self.buf.iter().copied().collect();
        for w in pts.windows(2) {
            sum += w[0].abs() * w[1].abs();
        }
        let v = (std::f64::consts::PI / 2.0) * sum / (self.window as f64 - 1.0);
        let res = if v > 0.0 && r.is_finite() {
            let stat = r.abs() / v.sqrt();
            let ln_m = self.period_m.max(10) as f64;
            let lnm = (2.0 * ln_m.ln()).sqrt();
            let c = lnm - (std::f64::consts::PI.ln() + ln_m.ln().ln()) / (2.0 * lnm);
            let s = 1.0 / lnm;
            let normalized = (stat - c) / s;
            Some(LmResult {
                sigma2: v,
                stat,
                normalized,
                is_jump: stat > self.threshold_l,
            })
        } else {
            None
        };
        self.last = res;
        res
    }

    /// Most recent result.
    pub fn last(&self) -> Option<LmResult> {
        self.last
    }
}

// ---------------------------------------------------------------------------
// BNS-style ratio test (batch)
// ---------------------------------------------------------------------------

/// Batch realized/bipower ratio test over a slice of log returns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BnsResult {
    pub rv: f64,
    pub bv: f64,
    /// Realized jump variation `max(RV - BV, 0)`.
    pub rjv: f64,
    /// Share of variance attributable to jumps.
    pub jump_share: f64,
    /// Standardized statistic (negative => jumps).
    pub z: f64,
    /// One-sided jump flag at level alpha.
    pub is_jump: bool,
}

/// Run the ratio test on `returns` (log returns, equal spacing).
///
/// Variance derivation: for iid N(0, s^2) returns,
/// `Var(BV) = 2.608 n s^4`, `Var(RV) = 2 n s^4`, `Cov(BV, RV) = 2 n s^4`,
/// hence `Var(BV - RV) = theta n s^4` with
/// `theta = pi^2/4 + pi - 5 = 0.6090` — exactly the BNS constant. The
/// jump-robust plug-in `s^2 ~ BV/n` yields
/// `z = (BV - RV) / sqrt(theta * BV^2 / n)` (N(0,1) under the no-jump
/// null, one-sided to the left under jumps).
pub fn bns_ratio_test(returns: &[f64], alpha: f64) -> Option<BnsResult> {
    if returns.len() < 10 {
        return None;
    }
    let n = returns.len() as f64;
    let mut rv = 0.0;
    let mut bv = 0.0;
    for i in 1..returns.len() {
        let (a, b) = (returns[i - 1], returns[i]);
        rv += b * b;
        bv += a.abs() * b.abs();
    }
    rv += returns[0] * returns[0];
    bv *= std::f64::consts::PI / 2.0;
    let rjv = (rv - bv).max(0.0);
    let theta = std::f64::consts::PI.powi(2) / 4.0 + std::f64::consts::PI - 5.0;
    let var = theta * bv * bv / n;
    let z = if var > 0.0 { (bv - rv) / var.sqrt() } else { 0.0 };
    Some(BnsResult {
        rv,
        bv,
        rjv,
        jump_share: if rv > 0.0 { rjv / rv } else { 0.0 },
        z,
        is_jump: z < norm_ppf(alpha),
    })
}

/// Convenience alias.
pub type BnsRatioTest = BnsResult;

// ---------------------------------------------------------------------------
// Bibinger–Hautsch–Ristig block-minima test (order prices)
// ---------------------------------------------------------------------------

/// Batch BHR jump test on an observed price series (best ask or best bid
/// prices) under one-sided microstructure noise.
pub struct BhrJumpTest {
    /// Observations per block (h_n * n).
    pub block_obs: usize,
    /// Blocks in the rolling variance window (K_n).
    pub vol_window: usize,
    /// Significance level.
    pub alpha: f64,
}

/// Result of the BHR global test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BhrResult {
    /// Global statistic `n^{1/3} * T`.
    pub stat: f64,
    /// Gumbel centering `B_n`.
    pub b_n: f64,
    /// `stat - b_n` (compare against Gumbel quantile).
    pub normalized: f64,
    pub is_jump: bool,
    /// Estimated jump location (block index) when a jump is found.
    pub jump_block: Option<usize>,
    /// Estimated jump size (price units of the input series).
    pub jump_size: Option<f64>,
}

impl BhrJumpTest {
    /// `block_obs` observations per block, `vol_window` blocks for the
    /// spot-variance estimate.
    pub fn new(block_obs: usize, vol_window: usize, alpha: f64) -> BhrJumpTest {
        BhrJumpTest {
            block_obs: block_obs.max(4),
            vol_window: vol_window.max(4),
            alpha,
        }
    }

    /// Suggested observations per block for a sample of length n, following
    /// the paper's rate `h_n ~ 2 ln(2/h_n - 2) n^{-2/3}` (self-consistent
    /// fixed point, floored at 8 observations).
    pub fn suggest_block_obs(n: usize) -> usize {
        let nf = n as f64;
        let mut h = (2.0 * (2.0 * nf.powf(-2.0 / 3.0)).ln()) * nf.powf(-2.0 / 3.0);
        for _ in 0..4 {
            let m = (2.0 / h - 2.0).max(std::f64::consts::E);
            h = 2.0 * m.ln() * nf.powf(-2.0 / 3.0);
        }
        (h * nf).round().max(8.0) as usize
    }

    /// Run the global test on the observed series `y` (index-aligned
    /// prices, e.g. best ask). Returns `None` when too short.
    pub fn test(&self, y: &[f64]) -> Option<BhrResult> {
        let n = y.len();
        let h = self.block_obs;              // observations per block
        let h_frac = h as f64 / n as f64;    // block length as a time fraction
        let n_blocks = n / h;
        if n_blocks < self.vol_window + 4 {
            return None;
        }
        // Block minima.
        let m: Vec<f64> = (0..n_blocks)
            .map(|k| {
                let blk = &y[k * h..(k + 1) * h];
                blk.iter().copied().fold(f64::INFINITY, f64::min)
            })
            .collect();
        // Block increments of minima.
        let dm: Vec<f64> = (1..n_blocks).map(|k| m[k] - m[k - 1]).collect();
        // Rolling min-based spot variance on the [0,1] time scale:
        // sigma2_k = pi / (2 (pi-2) K) * sum_{j in window} dm_j^2 / h_frac
        let k_w = self.vol_window;
        let const_c = std::f64::consts::PI / (2.0 * (std::f64::consts::PI - 2.0));
        let sigma2 = |lo: usize, hi: usize| -> f64 {
            let mut s = 0.0;
            for j in lo..hi {
                s += dm[j] * dm[j];
            }
            const_c * s / (k_w as f64 * h_frac)
        };
        // Statistic: T = max_k |dm_k| / sigma_k with a centered window.
        let mut t_max = 0.0f64;
        let mut argmax = 0usize;
        for k in 0..dm.len() {
            let lo = k.saturating_sub(k_w / 2);
            let hi = (k + k_w / 2 + 1).min(dm.len());
            let s2 = sigma2(lo, hi);
            if s2 <= 0.0 {
                continue;
            }
            let t = dm[k].abs() / s2.sqrt();
            if t > t_max {
                t_max = t;
                argmax = k;
            }
        }
        // Gumbel normalization (m = number of effective blocks).
        let m_blocks = (2.0 / h_frac - 2.0).max(std::f64::consts::E);
        let b_n = 2.0 * m_blocks.ln() - (std::f64::consts::PI * m_blocks.ln()).ln();
        let stat = (n as f64).powf(1.0 / 3.0) * t_max;
        let normalized = stat - b_n;
        let is_jump = normalized > gumbel_ppf(1.0 - self.alpha);
        let (jump_block, jump_size) = if is_jump {
            // Location: dm index k => jump between blocks k and k+1.
            // Size: min of the block after the jump minus min of the block
            // before it (paper eqs. 6-7).
            let k = argmax; // dm[k] = m[k+1] - m[k]
            let size = m[k + 1] - m[k];
            (Some(k + 1), Some(size))
        } else {
            (None, None)
        };
        Some(BhrResult {
            stat,
            b_n,
            normalized,
            is_jump,
            jump_block,
            jump_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::Rng;

    #[test]
    fn lm08_power_and_size() {
        // Size: pure Gaussian returns -> essentially no false flags.
        let mut lm = LeeMykland::new(100, 5000, 0.001);
        let mut rng = Rng::new(1);
        let mut false_pos = 0;
        for _ in 0..5000 {
            if let Some(r) = lm.update(rng.normal()) {
                if r.is_jump {
                    false_pos += 1;
                }
            }
        }
        assert!(false_pos <= 3, "false positives: {false_pos}");

        // Power: 5 injected jumps of 8 sigma.
        let mut lm = LeeMykland::new(100, 5000, 0.001);
        let mut rng = Rng::new(2);
        let jump_idx: Vec<usize> = vec![700, 1500, 2600, 3400, 4300];
        let mut detected = 0;
        for i in 0..5000 {
            let mut r = rng.normal();
            if jump_idx.contains(&i) {
                r += 8.0;
            }
            if let Some(res) = lm.update(r) {
                if res.is_jump && jump_idx.contains(&i) {
                    detected += 1;
                }
            }
        }
        assert!(detected >= 4, "detected {detected}/5 jumps");
    }

    #[test]
    fn bns_size_and_power() {
        // Size under pure diffusion.
        let mut rng = Rng::new(3);
        let mut rejections = 0;
        for _ in 0..200 {
            let rets: Vec<f64> = (0..1000).map(|_| rng.normal()).collect();
            let res = bns_ratio_test(&rets, 0.05).unwrap();
            if res.is_jump {
                rejections += 1;
            }
            assert!(res.z.abs() < 4.0);
        }
        assert!(rejections <= 20, "BNS size too big: {rejections}/200");

        // Power: three large jumps.
        let mut rng = Rng::new(4);
        let mut rets: Vec<f64> = (0..1000).map(|_| rng.normal()).collect();
        for &i in &[300, 600, 900] {
            rets[i] += 10.0;
        }
        let res = bns_ratio_test(&rets, 0.05).unwrap();
        assert!(res.is_jump);
        assert!(res.z < -3.0, "z = {}", res.z);
        assert!(res.jump_share > 0.05, "share = {}", res.jump_share);
    }

    #[test]
    fn bhr_detects_jump_in_ask_prices() {
        // Ask prices: efficient random walk + one-sided exponential noise.
        let n = 6000;
        let sigma_obs = 0.02;
        let mut rng = Rng::new(5);
        let mut x = 100.0f64;
        let mut y: Vec<f64> = Vec::with_capacity(n);
        let jump_at = 3000usize;
        let jump_size = -3.5; // sharp downward move, >> block-level noise
        for i in 0..n {
            x += sigma_obs * rng.normal();
            if i == jump_at {
                x += jump_size;
            }
            y.push(x + rng.exponential(50.0)); // one-sided noise
        }
        let bhr = BhrJumpTest::new(BhrJumpTest::suggest_block_obs(n), 20, 0.01);
        let res = bhr.test(&y).unwrap();
        assert!(res.is_jump, "BHR failed to detect: normalized={}", res.normalized);
        // Location within +/- 3 blocks of the true one.
        let true_block = jump_at / bhr.block_obs;
        let est = res.jump_block.unwrap();
        assert!(
            (est as i64 - true_block as i64).abs() <= 3,
            "location off: est {est} vs true {true_block}"
        );
        // Size estimate within 35%.
        let sz = res.jump_size.unwrap();
        assert!(
            ((sz - jump_size) / jump_size).abs() < 0.35,
            "size estimate {sz} vs {jump_size}"
        );

        // Size: no jump -> generally not rejected.
        let mut rng = Rng::new(6);
        let mut x = 100.0f64;
        let mut y2: Vec<f64> = Vec::with_capacity(n);
        for _ in 0..n {
            x += sigma_obs * rng.normal();
            y2.push(x + rng.exponential(50.0));
        }
        let res2 = bhr.test(&y2).unwrap();
        assert!(!res2.is_jump, "BHR false positive: {:?}", res2.normalized);
    }
}
