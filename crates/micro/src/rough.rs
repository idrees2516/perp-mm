//! Rough volatility: fBm simulation (Davies–Harte), the `m(q,Delta)`
//! Hurst estimator, the RFSV model, and Gaussian-conditional forecasts.
//!
//! Source: Gatheral–Jaisson–Rosenbaum, "Volatility is rough"
//! (arXiv:1410.3394; Fin. & Stoch. 22, 2018):
//! * log-volatility increments scale as `Delta^H` with `H ~ 0.1`;
//! * `m(q, Delta) = (1/N) sum_k |log sigma_{kDelta} - log sigma_{(k-1)Delta}|^q`
//!   satisfies `log m(q,Delta) ~ log(b_q) + q H log Delta` — the Hurst
//!   estimator is the slope regression of `log m(2,Delta)` on `log Delta`;
//! * RFSV: `X_t = log sigma^2_t` is a fractional OU process
//!   `X_t = nu * int_{-inf}^t e^{-alpha(t-s)} dW^H_s + m` with `H < 1/2`
//!   and `alpha*T << 1`, so locally `X ~ nu W^H`;
//! * forecasting uses the fBm Gaussian conditional mean.

/// Radix-2 iterative FFT (in place). `re`/`im` length must be a power of 2.
fn fft(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    // bit reversal permutation
    let mut j = 0usize;
    for i in 0..n {
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
        let mut m = n >> 1;
        while m >= 1 && j & m != 0 {
            j &= !m;
            m >>= 1;
        }
        j |= m;
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2usize;
    while len <= n {
        let ang = sign * 2.0 * std::f64::consts::PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let mut cur_r = 1.0f64;
            let mut cur_i = 0.0f64;
            for k in 0..len / 2 {
                let a = i + k;
                let b = i + k + len / 2;
                let tr = re[b] * cur_r - im[b] * cur_i;
                let ti = re[b] * cur_i + im[b] * cur_r;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let nr = cur_r * wr - cur_i * wi;
                cur_i = cur_r * wi + cur_i * wr;
                cur_r = nr;
            }
            i += len;
        }
        len <<= 1;
    }
    if inverse {
        for i in 0..n {
            re[i] /= n as f64;
            im[i] /= n as f64;
        }
    }
}

/// Fractional Gaussian noise autocovariance (unit variance at lag 0):
/// `gamma(k) = 0.5 (|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H})`.
#[inline]
fn fgn_cov(h: f64, k: usize) -> f64 {
    let k = k as f64;
    0.5 * ((k + 1.0).powf(2.0 * h) - 2.0 * k.powf(2.0 * h) + (k - 1.0).abs().powf(2.0 * h))
}

/// Exact fractional Brownian motion path via Davies–Harte circulant
/// embedding (O(n log n), exact covariance). Returns `n+1` values
/// `B_0=0, ..., B_{n dt}` with `Var(B_{t+dt}-B_t) = dt^{2H}`.
pub fn fbm_path(h: f64, n: usize, rng: &mut crate::sampling::Rng, dt: f64) -> Vec<f64> {
    debug_assert!((0.0..=1.0).contains(&h));
    let n = n.max(2);
    let mut m = (2 * n).next_power_of_two();
    for _ in 0..4 {
        let (re, im) = build_embedded(h, n, m);
        let mut re = re;
        let mut im = im;
        fft(&mut re, &mut im, false);
        let ok = re.iter().zip(im.iter()).all(|(&r, &i)| r > -1e-10 && i.abs() < 1e-8);
        if ok {
            // sample: x = sqrt(m) * ifft(sqrt(lambda) * fft(g)), g real N(0,1)
            let mut gr = vec![0.0f64; m];
            let mut gi = vec![0.0f64; m];
            for k in 0..m {
                gr[k] = rng.normal();
                gi[k] = 0.0;
            }
            fft(&mut gr, &mut gi, false);
            for k in 0..m {
                let lam = re[k].max(0.0);
                let s = lam.sqrt();
                let (a, b) = (gr[k], gi[k]);
                gr[k] = s * a;
                gi[k] = s * b;
            }
            fft(&mut gr, &mut gi, true);
            // scale to increments with Var = dt^{2H}, cumsum to B^H
            // (x = ifft(sqrt(lambda) * fft(g)) is already covariance-exact)
            let scale = dt.powf(h);
            let mut out = Vec::with_capacity(n + 1);
            out.push(0.0);
            let mut acc = 0.0f64;
            for k in 0..n {
                acc += gr[k] * scale;
                out.push(acc);
            }
            return out;
        }
        m *= 2;
    }
    // fallback: clamp eigenvalues (very unlikely for H in (0,1))
    let (mut re, mut im) = build_embedded(h, n, m);
    fft(&mut re, &mut im, false);
    let mut gr = vec![0.0f64; m];
    let mut gi = vec![0.0f64; m];
    for k in 0..m {
        gr[k] = rng.normal();
    }
    fft(&mut gr, &mut gi, false);
    for k in 0..m {
        let s = re[k].max(0.0).sqrt();
        gr[k] *= s;
        gi[k] *= s;
    }
    fft(&mut gr, &mut gi, true);
    let scale = dt.powf(h);
    let mut out = vec![0.0f64; n + 1];
    let mut acc = 0.0;
    for k in 0..n {
        acc += gr[k] * scale;
        out[k + 1] = acc;
    }
    out
}

fn build_embedded(h: f64, n: usize, m: usize) -> (Vec<f64>, Vec<f64>) {
    let mut re = vec![0.0f64; m];
    let im = vec![0.0f64; m];
    for k in 0..n {
        re[k] = fgn_cov(h, k);
    }
    for k in 1..n {
        re[m - k] = fgn_cov(h, k);
    }
    (re, im)
}

/// `m(q, Delta)` Hurst estimation: regress `log m(q,Delta)` on
/// `log Delta` over the provided lags; slope / q = H.
/// `vol_proxy` is a series of positive volatility (or variance) proxies at
/// fixed spacing; lags are multiples of that spacing.
///
/// Returns `(H, r_squared)`.
pub fn hurst_estimate(
    vol_proxy: &[f64],
    lags: &[usize],
    q: f64,
    robust: bool,
) -> (f64, f64) {
    let n = vol_proxy.len();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for &lag in lags {
        if lag == 0 || lag >= n {
            continue;
        }
        let mut vals = Vec::with_capacity(n - lag);
        for i in 0..n - lag {
            vals.push((vol_proxy[i + lag] - vol_proxy[i]).abs());
        }
        if vals.is_empty() {
            continue;
        }
        let m = if robust {
            // GJR robust version: median of |diff|^q, then log.
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = vals.len() / 2;
            let med = if vals.len() % 2 == 1 {
                vals[mid]
            } else {
                0.5 * (vals[mid - 1] + vals[mid])
            };
            med.powf(q).ln()
        } else {
            // m(q, Delta) = mean of |diff|^q, then log.
            let acc: f64 = vals.iter().map(|v| v.powf(q)).sum();
            (acc / vals.len() as f64).ln()
        };
        if m.is_finite() {
            xs.push((lag as f64).ln());
            ys.push(m);
        }
    }
    if xs.len() < 3 {
        return (f64::NAN, 0.0);
    }
    let nx = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / nx;
    let my = ys.iter().sum::<f64>() / nx;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for i in 0..xs.len() {
        sxy += (xs[i] - mx) * (ys[i] - my);
        sxx += (xs[i] - mx) * (xs[i] - mx);
        syy += (ys[i] - my) * (ys[i] - my);
    }
    let slope = sxy / sxx;
    let r2 = if syy > 0.0 {
        (sxy * sxy) / (sxx * syy)
    } else {
        0.0
    };
    (slope / q, r2)
}

/// Realized-variance proxy series over a trailing window.
pub fn realized_variance_series(returns: &[f64], window: usize) -> Vec<f64> {
    if returns.len() < window {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(returns.len() - window + 1);
    let mut s = 0.0;
    for i in 0..window {
        s += returns[i] * returns[i];
    }
    out.push(s);
    for i in window..returns.len() {
        s += returns[i] * returns[i] - returns[i - window] * returns[i - window];
        out.push(s);
    }
    out
}

/// RFSV volatility model: `X_t = log sigma^2_t` follows fractional OU
/// `X ~ nu W^H` with slow mean reversion (`alpha << 1/T`).
#[derive(Clone, Debug)]
pub struct RfsvModel {
    pub h: f64,
    pub nu: f64,
    /// Mean-reversion rate (use ~0 for the pure RFSV behavior).
    pub alpha: f64,
    /// Long-run mean of `X = log sigma^2`.
    pub m: f64,
}

impl RfsvModel {
    /// Simulate `n` volatility observations at spacing `dt`
    /// (returns sigma_t, i.e. `exp(X_t / 2)`).
    pub fn simulate(&self, n: usize, dt: f64, rng: &mut crate::sampling::Rng) -> Vec<f64> {
        let path = fbm_path(self.h, n, rng, dt);
        let mut x = self.m;
        let decay = (-self.alpha * dt).exp();
        let mut out = Vec::with_capacity(n);
        for k in 1..=n {
            let db = path[k] - path[k - 1];
            x = decay * x + (1.0 - decay) * self.m + self.nu * db;
            out.push((x / 2.0).exp());
        }
        out
    }

    /// Gaussian-conditional forecast of `X_{t+horizon}` given the last `m`
    /// observations of `X = log sigma^2` (exact for pure fBm, `alpha=0`;
    /// an approximation otherwise). Uses the fBm covariance
    /// `R(s,t) = 0.5 (s^{2H} + t^{2H} - |t-s|^{2H})`.
    pub fn forecast(&self, x_hist: &[f64], dt: f64, horizon: f64) -> f64 {
        let m = x_hist.len().min(128);
        if m < 2 {
            return x_hist.last().copied().unwrap_or(self.m);
        }
        let hist = &x_hist[x_hist.len() - m..];
        let t_now = (m - 1) as f64 * dt;
        let t_fut = t_now + horizon;
        // k_i = R(t_fut, t_i), K_ij = R(t_i, t_j)
        let r = |s: f64, t: f64| 0.5 * (s.powf(2.0 * self.h) + t.powf(2.0 * self.h) - (t - s).abs().powf(2.0 * self.h));
        let mut k = vec![0.0f64; m];
        for i in 0..m {
            let t_i = t_now - (m - 1 - i) as f64 * dt;
            k[i] = r(t_fut, t_i);
        }
        let mut a = vec![vec![0.0f64; m]; m];
        for i in 0..m {
            let t_i = t_now - (m - 1 - i) as f64 * dt;
            for j in 0..m {
                let t_j = t_now - (m - 1 - j) as f64 * dt;
                a[i][j] = r(t_i, t_j);
            }
        }
        // Solve A w = k (Gaussian elimination with partial pivoting).
        for col in 0..m {
            let mut piv = col;
            for row in col + 1..m {
                if a[row][col].abs() > a[piv][col].abs() {
                    piv = row;
                }
            }
            a.swap(col, piv);
            k.swap(col, piv);
            let d = a[col][col];
            if d.abs() < 1e-12 {
                continue;
            }
            for row in col + 1..m {
                let f = a[row][col] / d;
                if f == 0.0 {
                    continue;
                }
                for c2 in col..m {
                    a[row][c2] -= f * a[col][c2];
                }
                k[row] -= f * k[col];
            }
        }
        let mut w = vec![0.0f64; m];
        for row in (0..m).rev() {
            let mut s = k[row];
            for c2 in row + 1..m {
                s -= a[row][c2] * w[c2];
            }
            w[row] = if a[row][row].abs() > 1e-12 {
                s / a[row][row]
            } else {
                0.0
            };
        }
        // shrink weights toward the last observation for numerical safety
        let wsum: f64 = w.iter().sum();
        if !(0.5..=1.5).contains(&wsum) {
            return *hist.last().unwrap();
        }
        let mean = self.m;
        let cond = w
            .iter()
            .zip(hist.iter())
            .map(|(&wi, &xi)| wi * (xi - mean))
            .sum::<f64>();
        mean + cond
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::Rng;

    #[test]
    fn fbm_variance_scaling() {
        // Var(B_{t+Delta} - B_t) ~ Delta^{2H}
        let h = 0.3;
        let n = 4000;
        let dt = 1.0;
        let mut rng = Rng::new(31);
        let paths: Vec<Vec<f64>> = (0..30).map(|_| fbm_path(h, n, &mut rng, dt)).collect();
        for &lag in &[4usize, 16, 64] {
            let mut acc = 0.0;
            for p in &paths {
                for i in 0..n - lag {
                    let d = p[i + lag] - p[i];
                    acc += d * d;
                }
            }
            let count = (paths.len() * (n - lag)) as f64;
            let var = acc / count;
            let expect = (lag as f64 * dt).powf(2.0 * h);
            assert!(
                (var / expect - 1.0).abs() < 0.25,
                "lag {lag}: var {var} vs {expect}"
            );
        }
    }

    #[test]
    fn hurst_recovery() {
        // Simulate RFSV-ish log-vol as nu * fBm; estimator should find H.
        for &h_true in &[0.1f64, 0.3, 0.45] {
            let mut rng = Rng::new(41);
            let nu = 0.3;
            let n = 4000;
            let path = fbm_path(h_true, n, &mut rng, 1.0);
            let log_vol: Vec<f64> = path.iter().map(|&b| nu * b).collect();
            let lags: Vec<usize> = vec![1, 2, 4, 8, 16, 32, 64];
            let (h_hat, r2) = hurst_estimate(&log_vol, &lags, 2.0, false);
            assert!(r2 > 0.9, "r2 {r2}");
            assert!(
                (h_hat - h_true).abs() < 0.08,
                "h_hat {h_hat} vs {h_true}"
            );
        }
    }

    #[test]
    fn rfsv_forecast_beats_climatology() {
        let model = RfsvModel {
            h: 0.1,
            nu: 0.3,
            alpha: 0.0,
            m: -9.0,
        };
        let mut rng = Rng::new(51);
        let n = 3000;
        let dt = 1.0;
        let sig = model.simulate(n, dt, &mut rng);
        let x: Vec<f64> = sig.iter().map(|&s| 2.0 * s.ln()).collect();
        // Walk-forward: forecast horizon 20 from history of 128.
        let mut err_model = 0.0;
        let mut err_clim = 0.0;
        let mut cnt = 0.0;
        for t in (600..n - 20).step_by(7) {
            let hist = &x[t - 128..t];
            let f = model.forecast(hist, dt, 20.0);
            let actual = x[t + 20];
            let clim: f64 = hist.iter().sum::<f64>() / hist.len() as f64;
            err_model += (f - actual).powi(2);
            err_clim += (clim - actual).powi(2);
            cnt += 1.0;
        }
        assert!(
            err_model < err_clim,
            "RFSV forecast MSE {} vs climatological {}",
            err_model / cnt,
            err_clim / cnt
        );
    }
}
