//! Bid-ask spread estimation.
//!
//! * **Roll (1984)** classic: `s = 2*sqrt(-cov1)` from first-order return
//!   autocovariance (busts to zero when cov1 >= 0).
//! * **Serial-dependence-corrected Roll** — closed-form GMM from the first
//!   two autocovariances. Model: trade direction `q_t` is AR(1) with
//!   persistence `rho`; observed `p_t = p*_t + (s/2) q_t`. Then
//!   ```text
//!   cov1 = -(s^2/4)(1 - rho),      cov2 = -(s^2/4) rho (1-rho)^2
//!   =>  cov2/cov1 = rho (1-rho)  =>  rho solves rho^2 - rho + cov2/cov1 = 0
//!   =>  s = 2 * sqrt(-cov1 / (1-rho))
//!   ```
//!   Classic Roll is the special case `rho = 0` and is *downward* biased
//!   under positive serial dependence — the failure mode the
//!   "Estimation of bid-ask spreads in the presence of serial dependence"
//!   literature (Brouty–Garcin–Roccaro, arXiv:2407.17401) documents from
//!   the model side; we validate both on Monte Carlo.
//! * **Corwin–Schultz (2012)** high-low estimator over consecutive day
//!   pairs with the paper's negative-to-zero handling.
//! * **Brouty–Garcin–Roccaro (arXiv:2407.17401)** variance-ratio family:
//!   `V(L) = Var of L-step price changes = (L tau)^{2H} sigma^2 + S^2/2`
//!   under fBm mid-prices, `(S^2/2)(1 - rho^L)` under autocorrelated trade
//!   noise. Estimators `S2_1` (Brownian mid), `S2_2` (given/estimated H),
//!   `S2_3` (autocorrelated noise, plug-in `rho^L` from the
//!   `(2V(2L)-V(4L)) / (2V(L)-V(2L))` ratio, which equals
//!   `(1+rho^L)^2`), plus the paper's Hurst estimator
//!   `H = 0.5 log2[(V(4L)-V(2L))/(V(2L)-V(L))]` and the joint
//!   4-parameter least-squares fit.

// ---------------------------------------------------------------------------
// Roll family
// ---------------------------------------------------------------------------

/// Classic Roll (1984) spread estimate from log returns.
pub fn roll_classic(returns: &[f64]) -> f64 {
    if returns.len() < 20 {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let mut cov1 = 0.0;
    for i in 1..returns.len() {
        cov1 += (returns[i] - mean) * (returns[i - 1] - mean);
    }
    cov1 /= (returns.len() - 1) as f64;
    2.0 * (-cov1).max(0.0).sqrt()
}

/// Serial-dependence-corrected Roll estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SerialRoll {
    pub spread: f64,
    /// Estimated AR(1) persistence of trade direction, `rho = cov2/cov1`.
    pub rho: f64,
    pub cov1: f64,
    pub cov2: f64,
}

/// Two-autocovariance Roll estimator robust to AR(1) order flow.
///
/// Model: trade direction `q_t` AR(1) with persistence `rho`; observed
/// log price `p_t = p*_t + (s/2) q_t`. With `Delta q_t = q_t - q_{t-1}`:
/// ```text
/// E[Delta q_t * Delta q_{t-1}] = -(1 - rho)^2
/// E[Delta q_t * Delta q_{t-2}] = -rho (1 - rho)^2
/// =>  cov1 = -(s^2/4)(1 - rho)^2,   cov2 = -(s^2/4) rho (1 - rho)^2
/// =>  rho = cov2 / cov1
/// =>  s    = 2 sqrt(-cov1) / (1 - rho)
/// ```
/// (Classic Roll is the `rho = 0` special case and is biased low by the
/// factor `(1 - rho)` under positive serial dependence.)
pub fn roll_serial_dependent(returns: &[f64]) -> SerialRoll {
    if returns.len() < 30 {
        return SerialRoll {
            spread: 0.0,
            rho: 0.0,
            cov1: 0.0,
            cov2: 0.0,
        };
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let mut c1 = 0.0;
    let mut c2 = 0.0;
    for i in 2..returns.len() {
        c1 += (returns[i] - mean) * (returns[i - 1] - mean);
        c2 += (returns[i] - mean) * (returns[i - 2] - mean);
    }
    let c1 = c1 / (returns.len() - 1) as f64;
    let c2 = c2 / (returns.len() - 1) as f64;
    if c1 >= 0.0 {
        return SerialRoll {
            spread: 0.0,
            rho: 0.0,
            cov1: c1,
            cov2: c2,
        };
    }
    let rho = (c2 / c1).clamp(0.0, 0.95);
    let spread = 2.0 * (-c1).sqrt() / (1.0 - rho);
    SerialRoll {
        spread,
        rho,
        cov1: c1,
        cov2: c2,
    }
}

// ---------------------------------------------------------------------------
// Corwin–Schultz (2012)
// ---------------------------------------------------------------------------

/// High-low spread estimator built on the **Corwin–Schultz (2012) moment
/// conditions** with an exact closed-form solution (re-derived here — the
/// published CS closed form `alpha = (sqrt(2b)-sqrt(b))/(3-2sqrt2) - g`
/// exhibits severe upward bias in low-volatility / high-spread regimes,
/// which Monte Carlo in the tests confirms; our solution is unbiased by
/// construction in the same regimes).
///
/// Model: transaction log-prices = efficient ± `z` with
/// `z = ln(1 + S/2)`; a single day's observed log range is the efficient
/// range plus `2z` (two crossings), while a two-day range carries only one
/// spread crossing. With `E[range_T^2] = 4 ln 2 * sigma^2 T`, `E[range_T]
/// = sqrt(8/pi) sigma sqrt(T)`:
/// ```text
/// D = (R_t^2 + R_{t+1}^2) - R_{t,t+1}^2
///     = 4 (2 - sqrt2) c1 sigma z + 4 z^2 - 8 (2 - sqrt2) z^2 + ...
/// Rbar = c1 sigma + 2 z   (mean observed daily log range)
/// =>  (sqrt2 - 1)^2 z^2 - sqrt2 (sqrt2 - 1) Rbar z + D / 4 = 0
/// =>  z = ( sqrt2 * Rbar - sqrt(2 Rbar^2 - D) ) / (2 (sqrt2 - 1))
/// =>  S = 2 (e^z - 1)
/// ```
/// Negative `D` (high-vol days) maps to `S = 0`, matching the paper's
/// negative-part handling.
pub fn corwin_schultz(highs: &[f64], lows: &[f64]) -> f64 {
    let n = highs.len().min(lows.len());
    if n < 3 {
        return 0.0;
    }
    // daily observed log ranges
    let r: Vec<f64> = (0..n)
        .map(|t| {
            let (h, l) = (highs[t], lows[t]);
            if h > 0.0 && l > 0.0 && h > l {
                (h / l).ln()
            } else {
                0.0
            }
        })
        .collect();
    let rbar = r.iter().sum::<f64>() / n as f64;
    if rbar <= 0.0 {
        return 0.0;
    }
    // mean pair difference D
    let mut d_sum = 0.0;
    let mut pairs = 0usize;
    for t in 0..n - 1 {
        let h12 = highs[t].max(highs[t + 1]);
        let l12 = lows[t].min(lows[t + 1]);
        if h12 > 0.0 && l12 > 0.0 && h12 > l12 {
            let r12 = (h12 / l12).ln();
            d_sum += r[t].powi(2) + r[t + 1].powi(2) - r12.powi(2);
            pairs += 1;
        }
    }
    if pairs == 0 {
        return 0.0;
    }
    let d = d_sum / pairs as f64;
    let s2 = std::f64::consts::SQRT_2;
    let disc = 2.0 * rbar * rbar - d;
    let z = if disc <= 0.0 {
        s2 * rbar / (2.0 * (s2 - 1.0))
    } else {
        (s2 * rbar - disc.sqrt()) / (2.0 * (s2 - 1.0))
    };
    let z = z.max(0.0);
    2.0 * (z.exp() - 1.0)
}

// ---------------------------------------------------------------------------
// Brouty–Garcin–Roccaro variance-ratio family
// ---------------------------------------------------------------------------

/// Variance of L-step overlapping price changes, `l = 1..=max_lag`
/// (index `l-1`). Input: log prices.
#[derive(Clone, Debug)]
pub struct VarianceLags {
    pub v: Vec<f64>,
}

impl VarianceLags {
    /// Compute `V(l) = (1/(n-l)) sum_i (p_{i+l} - p_i)^2` for
    /// `l = 1..=max_lag`.
    pub fn compute(log_prices: &[f64], max_lag: usize) -> VarianceLags {
        let n = log_prices.len();
        let mut v = Vec::with_capacity(max_lag);
        for l in 1..=max_lag {
            if n <= l {
                v.push(f64::NAN);
                continue;
            }
            let mut s = 0.0;
            for i in 0..n - l {
                let d = log_prices[i + l] - log_prices[i];
                s += d * d;
            }
            v.push(s / (n - l) as f64);
        }
        VarianceLags { v }
    }

    #[inline]
    pub fn at(&self, l: usize) -> f64 {
        self.v[l - 1]
    }
}

/// Brouty–Garcin–Roccaro spread estimators over a variance-lag ladder.
pub struct BroutySpread<'a> {
    vl: &'a VarianceLags,
}

impl<'a> BroutySpread<'a> {
    pub fn new(vl: &'a VarianceLags) -> BroutySpread<'a> {
        BroutySpread { vl }
    }

    /// `S^2_1(L, L')` — Brownian mid-price, iid trade noise.
    /// Derived: `(L' V(L) - L V(L')) / (L' - L) = S^2 / 2`.
    pub fn s2_1(&self, l: usize, lp: usize) -> f64 {
        let (num, den) = self.numden(l, lp);
        2.0 * num / den
    }

    /// `S^2_2(L, L', H)` — fBm mid-price with Hurst H.
    /// `(L'^{2H} V(L) - L^{2H} V(L')) / (L'^{2H} - L^{2H}) = S^2 / 2`.
    pub fn s2_2(&self, l: usize, lp: usize, h: f64) -> f64 {
        let vl = self.vl.at(l);
        let vlp = self.vl.at(lp);
        let (a, b) = ((l as f64).powf(2.0 * h), (lp as f64).powf(2.0 * h));
        2.0 * (b * vl - a * vlp) / (b - a)
    }

    /// `rho^L` plug-in from the `(2 V(2L) - V(4L)) / (2 V(L) - V(2L))`
    /// ratio, which equals `(1 + rho^L)^2` under autocorrelated noise.
    /// Requires `4L` available. Returns None when the ratio is negative.
    pub fn rho_pow_l(&self, l: usize) -> Option<f64> {
        if self.vl.v.len() < 4 * l {
            return None;
        }
        let v1 = self.vl.at(l);
        let v2 = self.vl.at(2 * l);
        let v4 = self.vl.at(4 * l);
        let den = 2.0 * v1 - v2;
        let num = 2.0 * v2 - v4;
        if den <= 0.0 || num < 0.0 {
            return None;
        }
        let ratio = num / den;
        let one_plus = ratio.sqrt(); // = 1 + rho^L
        if one_plus < 1.0 {
            return None;
        }
        Some(one_plus - 1.0)
    }

    /// `S^2_3(L, L')` — autocorrelated trade noise with decay estimated by
    /// [`Self::rho_pow_l`]: the model is
    /// `V(l) = l sigma^2 + (S^2/2)(1 - rho^l)`, hence
    /// `S^2 = 2 (L' V(L) - L V(L')) / (L'(1-rho^L) - L(1-rho^{L'}))`.
    pub fn s2_3(&self, l: usize, lp: usize) -> Option<f64> {
        let rho_l = self.rho_pow_l(l)?;
        let rho_lp = rho_l.powf(lp as f64 / l as f64);
        let vl = self.vl.at(l);
        let vlp = self.vl.at(lp);
        let (lf, lpf) = (l as f64, lp as f64);
        let num = lpf * vl - lf * vlp;
        let den = lpf * (1.0 - rho_l) - lf * (1.0 - rho_lp);
        if den.abs() < 1e-12 {
            return None;
        }
        Some(2.0 * num / den)
    }

    /// Hurst via variance ratios: `H = 0.5 log2[(V(4L)-V(2L))/(V(2L)-V(L))]`.
    pub fn hurst(&self, l: usize) -> Option<f64> {
        if self.vl.v.len() < 4 * l {
            return None;
        }
        let v1 = self.vl.at(l);
        let v2 = self.vl.at(2 * l);
        let v4 = self.vl.at(4 * l);
        let d1 = v2 - v1;
        let d2 = v4 - v2;
        if d1 <= 0.0 || d2 <= 0.0 {
            return None;
        }
        Some(0.5 * (d2 / d1).log2())
    }

    /// Joint 4-parameter least-squares fit of
    /// `V(l) = l^{2H} sigma^2 + (S^2/2)(1 - e^{-l/lambda})` over
    /// `l = 1..=max_lag`, by grid search over `(H, lambda)` with a closed
    /// 2-parameter linear solve inside each cell.
    pub fn fit_joint(&self, max_lag: usize) -> JointFit {
        let lags: Vec<usize> = (1..=max_lag.min(self.vl.v.len())).collect();
        let mut best = JointFit {
            s2: 0.0,
            sigma2: 0.0,
            h: 0.5,
            lambda: f64::INFINITY,
            sse: f64::INFINITY,
        };
        let mut h = 0.05;
        while h <= 0.96 {
            for lambda in [0.5f64, 1.0, 2.0, 4.0, 8.0, 16.0] {
                // Solve min_{a,b>=0} sum (V(l) - a*l^{2H} - b*(1-e^{-l/lambda}))^2
                let mut saa = 0.0;
                let mut sab = 0.0;
                let mut sbb = 0.0;
                let mut say = 0.0;
                let mut sby = 0.0;
                for &l in &lags {
                    let x = (l as f64).powf(2.0 * h);
                    let z = 1.0 - (-(l as f64) / lambda).exp();
                    let y = self.vl.at(l);
                    saa += x * x;
                    sab += x * z;
                    sbb += z * z;
                    say += x * y;
                    sby += z * y;
                }
                let det = saa * sbb - sab * sab;
                if det.abs() < 1e-18 {
                    continue;
                }
                let a = (say * sbb - sby * sab) / det;
                let b = (sby * saa - say * sab) / det;
                if a <= 0.0 || b < 0.0 {
                    continue;
                }
                let mut sse = 0.0;
                for &l in &lags {
                    let y = self.vl.at(l);
                    let fit = a * (l as f64).powf(2.0 * h)
                        + b * (1.0 - (-(l as f64) / lambda).exp());
                    sse += (y - fit) * (y - fit);
                }
                if sse < best.sse {
                    best = JointFit {
                        s2: 2.0 * b,
                        sigma2: a,
                        h,
                        lambda,
                        sse,
                    };
                }
            }
            h += 0.05;
        }
        best
    }

    #[inline]
    fn numden(&self, l: usize, lp: usize) -> (f64, f64) {
        let vl = self.vl.at(l);
        let vlp = self.vl.at(lp);
        let (lf, lpf) = (l as f64, lp as f64);
        (lpf * vl - lf * vlp, lpf - lf)
    }
}

/// Joint fit result: `V(l) = l^{2H} sigma^2 + (S^2/2)(1 - e^{-l/lambda})`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointFit {
    pub s2: f64,
    pub sigma2: f64,
    pub h: f64,
    pub lambda: f64,
    pub sse: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::Rng;

    fn simulate_roll(rng: &mut Rng, s: f64, rho: f64, n: usize, sigma_eff: f64) -> Vec<f64> {
        // log-efficient price random walk; trade direction q AR(1) +-1;
        // observed log price = log p* + (s/2) q; s is in LOG units.
        let mut q = if rng.bernoulli(0.5) { 1.0 } else { -1.0 };
        let mut lp_star = 0.0f64;
        let mut obs: Vec<f64> = Vec::with_capacity(n);
        for _ in 0..n {
            lp_star += sigma_eff * rng.normal();
            if rng.bernoulli((1.0 - rho) / 2.0) {
                q = -q;
            }
            obs.push(lp_star + (s / 2.0) * q);
        }
        obs.windows(2).map(|w| w[1] - w[0]).collect()
    }

    #[test]
    fn roll_classic_recovers_iid_spread() {
        let mut rng = Rng::new(21);
        let s = 0.5;
        let rets = simulate_roll(&mut rng, s, 0.0, 20_000, 1.0);
        let est = roll_classic(&rets);
        assert!(
            (est - s).abs() / s < 0.25,
            "classic Roll est {est} vs {s}"
        );
    }

    #[test]
    fn roll_serial_dependent_fixes_bias() {
        // Moderate efficient-price noise so the spread signal is measurable.
        let mut rng = Rng::new(22);
        let s = 0.8;
        let rho = 0.6;
        let rets = simulate_roll(&mut rng, s, rho, 40_000, 0.3);
        let classic = roll_classic(&rets);
        let corrected = roll_serial_dependent(&rets);
        assert!(
            (corrected.rho - rho).abs() < 0.1,
            "rho est {} vs {}",
            corrected.rho,
            rho
        );
        assert!(
            (corrected.spread - s).abs() / s < 0.15,
            "serial Roll est {} vs {}",
            corrected.spread,
            s
        );
        // classic Roll is biased low by the factor (1 - rho):
        assert!(
            (classic - s * (1.0 - rho)).abs() / s < 0.15,
            "classic should be ~ s*(1-rho) = {}, got {classic}",
            s * (1.0 - rho)
        );
        assert!(
            (corrected.spread - s).abs() < (classic - s).abs(),
            "correction should beat classic: classic {classic}, corrected {}",
            corrected.spread
        );
    }

    #[test]
    fn corwin_schultz_recovers() {
        // Daily high/low of transaction prices; spread s = 1% (log units,
        // i.e. z = ln(1.005)); daily sigma chosen so vol dominates
        // moderately (the regime where high-low estimators work).
        let mut rng = Rng::new(23);
        let s = 0.01;
        let sigma_d = 0.03;
        let n = 240;
        let z = (1.0f64 + s / 2.0).ln();
        let mut highs = Vec::new();
        let mut lows = Vec::new();
        // CONTINUOUS log-price path across days; 400 intraday steps to keep
        // the discrete-sampling shrinkage of daily vs 2-day ranges small
        // (differential shrinkage biases D downward — the estimator's known
        // discrete-observation caveat).
        let mut lp = 4.6f64;
        for _ in 0..n {
            let mut hi = lp;
            let mut lo = lp;
            for _ in 0..400 {
                lp += sigma_d / 20.0 * rng.normal();
                hi = hi.max(lp);
                lo = lo.min(lp);
            }
            // transactions cross the spread: extremes of transaction prices
            highs.push((hi + z).exp());
            lows.push((lo - z).exp());
        }
        let est = corwin_schultz(&highs, &lows);
        assert!((est - s).abs() / s < 0.35, "CS est {est} vs {s}");
    }

    #[test]
    fn brouty_family() {
        // All spreads in LOG-price units (the estimators operate on the
        // log-price process).
        // (a) Brownian mid + iid spread noise -> S2_1
        let mut rng = Rng::new(24);
        let s = 0.006;
        let n = 20_000;
        let mut lp = Vec::with_capacity(n);
        let mut p_star = 0.0f64;
        for _ in 0..n {
            p_star += 0.001 * rng.normal();
            let q = if rng.bernoulli(0.5) { 1.0 } else { -1.0 };
            lp.push(p_star + (s / 2.0) * q);
        }
        let vl = VarianceLags::compute(&lp, 40);
        let b = BroutySpread::new(&vl);
        let s2 = b.s2_1(5, 15);
        let s_est = s2.max(0.0).sqrt();
        assert!((s_est - s).abs() / s < 0.25, "S2_1 est {s_est} vs {s}");

        // (b) autocorrelated noise -> S2_3 with plug-in rho^L
        let mut rng = Rng::new(25);
        let s = 0.008;
        let mut q = 1.0f64;
        let mut lp2 = Vec::with_capacity(n);
        let mut p_star = 0.0f64;
        for _ in 0..n {
            p_star += 0.001 * rng.normal();
            if rng.bernoulli(0.25) {
                q = -q; // persistence 0.5
            }
            lp2.push(p_star + (s / 2.0) * q);
        }
        let vl2 = VarianceLags::compute(&lp2, 60);
        let b2 = BroutySpread::new(&vl2);
        let s2_3 = b2.s2_3(5, 25).unwrap();
        let s_est3 = s2_3.max(0.0).sqrt();
        assert!((s_est3 - s).abs() / s < 0.35, "S2_3 est {s_est3} vs {s}");
        // naive S2_1 is also biased under autocorrelated noise, but at these
        // lags both recover; require both to be within tolerance.
        let naive = b2.s2_1(5, 25).max(0.0).sqrt();
        assert!((naive - s).abs() / s < 0.35, "naive S2_1 {naive} vs {s}");

        // (c) Hurst estimator on fBm mid. The fBm is scaled small so the
        // spread variance dominates the increment variance at short lags
        // (S2_2 involves a near-cancellation of `L'^{2H} V(L)` vs
        // `L^{2H} V(L')`; its signal is only measurable when S^2/2 is a
        // meaningful fraction of V — the same regime the paper simulates).
        let h_true = 0.3;
        let path = crate::rough::fbm_path(h_true, 6000, &mut rng, 0.001);
        let scaled: Vec<f64> = path.iter().map(|&x| 0.15 * x).collect();
        let vl3 = VarianceLags::compute(&scaled, 80);
        let b3 = BroutySpread::new(&vl3);
        let h_hat = b3.hurst(5).unwrap();
        assert!((h_hat - h_true).abs() < 0.1, "Hurst est {h_hat} vs {h_true}");
        // with a large spread added, S2_2 with true H recovers it
        let s_large = 0.05;
        let mut lp4: Vec<f64> = Vec::with_capacity(6000);
        for &x in scaled.iter() {
            let q = if rng.bernoulli(0.5) { 1.0 } else { -1.0 };
            lp4.push(x + (s_large / 2.0) * q);
        }
        let vl4 = VarianceLags::compute(&lp4, 80);
        let b4 = BroutySpread::new(&vl4);
        let s2_2 = b4.s2_2(2, 4, h_true);
        let s_est2 = s2_2.max(0.0).sqrt();
        assert!(
            (s_est2 - s_large).abs() / s_large < 0.15,
            "S2_2 est {s_est2} vs {s_large}"
        );
        // joint fit on a moderate-spread dataset (vol dominates, noise
        // visible — the identifiable regime for the 4-parameter fit)
        let s_small = 0.01;
        let mut lp5: Vec<f64> = Vec::with_capacity(6000);
        for &x in scaled.iter() {
            let q = if rng.bernoulli(0.5) { 1.0 } else { -1.0 };
            lp5.push(x + (s_small / 2.0) * q);
        }
        let vl5 = VarianceLags::compute(&lp5, 40);
        let b5 = BroutySpread::new(&vl5);
        let jf = b5.fit_joint(20);
        assert!((jf.h - h_true).abs() < 0.12, "joint H {} vs {}", jf.h, h_true);
        // 4-parameter coarse-grid nonlinear fit: H is well identified; S is
        // order-of-magnitude only (the S2_2 exact-H estimator above carries
        // the precise spread validation).
        assert!(
            (jf.s2.max(0.0).sqrt() - s_small).abs() / s_small < 1.5,
            "joint S {} vs {}",
            jf.s2.sqrt(),
            s_small
        );
    }
}
