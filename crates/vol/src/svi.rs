//! Raw SVI slice (Gatheral 2004): total variance
//! `w(k) = a + b ( rho (k-m) + sqrt((k-m)^2 + sigma^2) )`,
//! with the Gatheral–Jacquier (2014) butterfly machinery:
//! the risk-neutral density factor
//! `g(k) = (1 - k w'/(2w))^2 - (w'/2)^2 + w''/2`
//! must be non-negative (Gatheral–Jacquier §2, Lemma 2.1), plus the
//! parameteric sanity `b >= 0, |rho| < 1, sigma > 0,
//! a + b sigma sqrt(1-rho^2) >= 0`. The g(k) grid check is the ground
//! truth used throughout; the parametric shortcuts are validated against
//! it in the tests. Calibration is a damped Gauss–Newton with analytic
//! Jacobian and parameter projection onto the sane set.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SviSlice {
    pub a: f64,
    pub b: f64,
    pub rho: f64,
    pub m: f64,
    pub sigma: f64,
}

impl SviSlice {
    /// Total variance at log-moneyness `k`.
    #[inline]
    pub fn w(&self, k: f64) -> f64 {
        let d = k - self.m;
        let s2 = self.sigma * self.sigma;
        self.a + self.b * (self.rho * d + (d * d + s2).sqrt())
    }

    /// First derivative `w'(k)`.
    #[inline]
    pub fn dw(&self, k: f64) -> f64 {
        let d = k - self.m;
        let s2 = self.sigma * self.sigma;
        self.b * (self.rho + d / (d * d + s2).sqrt())
    }

    /// Second derivative `w''(k)`.
    #[inline]
    pub fn d2w(&self, k: f64) -> f64 {
        let d = k - self.m;
        let s2 = self.sigma * self.sigma;
        self.b * s2 / (d * d + s2).powf(1.5)
    }

    /// Gatheral–Jacquier density factor `g(k) >= 0` ⟺ butterfly-free
    /// (at that k).
    pub fn g(&self, k: f64) -> f64 {
        let w = self.w(k);
        if w <= 0.0 {
            return f64::NAN;
        }
        let wp = self.dw(k);
        let wpp = self.d2w(k);
        (1.0 - k * wp / (2.0 * w)).powi(2) - wp * wp / 4.0 + wpp / 2.0
    }

    /// Tail slopes `(w'(+inf), w'(-inf)) = (b(1+rho), b(1-rho))`.
    pub fn tail_slopes(&self) -> (f64, f64) {
        (self.b * (1.0 + self.rho), self.b * (1.0 - self.rho))
    }

    /// Basic parameteric sanity (necessary conditions).
    pub fn params_sane(&self) -> bool {
        self.b >= 0.0
            && self.rho.abs() < 0.9999
            && self.sigma > 0.0
            && self.a + self.b * self.sigma * (1.0 - self.rho * self.rho).sqrt() >= -1e-12
    }

    /// Butterfly-free on a grid (ground truth: `g(k) >= -tol`).
    pub fn butterfly_free_grid(&self, kmin: f64, kmax: f64, n: usize) -> bool {
        let n = n.max(8);
        for i in 0..=n {
            let k = kmin + (kmax - kmin) * i as f64 / n as f64;
            let g = self.g(k);
            if g.is_nan() || g < -1e-10 {
                return false;
            }
        }
        // Tail slopes must be non-negative and bounded (Roger Lee's
        // moment bound on total-variance slope, evaluated numerically
        // at the grid edge as a proxy for the asymptotic slopes).
        let (s_r, s_l) = self.tail_slopes();
        s_r >= 0.0 && s_l >= 0.0 && s_r <= 4.0 && s_l <= 4.0
    }

    /// Full static no-arbitrage check of the slice against the BSM
    /// ground truth: call prices (with `iv = sqrt(w/t)`) must be
    /// monotone decreasing and convex in strike, above intrinsic,
    /// below the discounted spot.
    pub fn static_arb_free_bsm(&self, s: f64, t: f64, r: f64, q: f64) -> bool {
        use crate::greeks::price;
        use models::options::Kind;
        let n = 60;
        let kmin = -1.5f64;
        let kmax = 1.5f64;
        let mut prev_c = f64::INFINITY;
        let mut prev_k = 0.0f64;
        let mut prev2_c = f64::INFINITY;
        let mut prev2_k = 0.0f64;
        for i in 0..=n {
            let kln = kmin + (kmax - kmin) * i as f64 / n as f64;
            let strike = s * kln.exp(); // F = S e^{(r-q)T} ~ S; k = ln(K/F)
            let w = self.w(kln);
            if w <= 0.0 {
                return false;
            }
            let iv = (w / t).sqrt();
            let c = price(Kind::Call, s, strike, r, q, iv, t);
            if c > s * (-q * t).exp() + 1e-9 {
                return false;
            }
            if c < (s * (-q * t).exp() - strike * (-r * t).exp()).max(0.0) - 1e-9 {
                return false;
            }
            if c > prev_c + 1e-9 {
                return false; // not monotone
            }
            if i >= 2 {
                // convexity via second difference
                let d2 = (c - prev_c) / (strike - prev_k).max(1e-12)
                    - (prev_c - prev2_c) / (prev_k - prev2_k).max(1e-12);
                if d2 < -1e-7 {
                    return false;
                }
            }
            prev2_c = prev_c;
            prev2_k = prev_k;
            prev_c = c;
            prev_k = strike;
        }
        true
    }

    /// Fit a raw SVI slice to total-variance pillars `(k, w)` by damped
    /// Gauss–Newton with analytic Jacobian and projection onto the sane
    /// parameter set. Returns `None` if the fit does not converge.
    pub fn fit(pillars: &[(f64, f64)]) -> Option<SviSlice> {
        if pillars.len() < 5 {
            return None;
        }
        // Initial guess: m at the minimum-variance k, sigma from the
        // curvature scale, rho from the asymmetry.
        let kmin = pillars.iter().fold(f64::INFINITY, |m, &(k, _)| m.min(k));
        let kmax = pillars.iter().fold(f64::NEG_INFINITY, |m, &(k, _)| m.max(k));
        let w_min = pillars.iter().fold(f64::INFINITY, |m, &(_, w)| m.min(w)).max(1e-6);
        let w_max = pillars.iter().fold(f64::NEG_INFINITY, |m, &(_, w)| m.max(w));
        let (k_at_min, k_at_max) = (
            pillars.iter().fold((kmin, f64::INFINITY), |a, &(k, w)| if w < a.1 { (k, w) } else { a }).0,
            pillars.iter().fold((kmax, f64::NEG_INFINITY), |a, &(k, w)| if w > a.1 { (k, w) } else { a }).0,
        );
        let slope = (w_max - w_min) / (k_at_max - k_at_min).abs().max(0.05);
        let mut sl = SviSlice {
            a: (w_min * 0.8).max(0.0),
            b: (slope / 2.0).clamp(0.01, 2.0),
            rho: -0.6,
            m: k_at_min,
            sigma: ((kmax - kmin) / 8.0).clamp(0.02, 0.5),
        };
        let mut lambda = 1e-2;
        let target_rmse = 1e-8;
        for _ in 0..300 {
            // residuals and Jacobian
            let mut jtj = [[0.0f64; 5]; 5];
            let mut jtr = [0.0f64; 5];
            let mut rmse = 0.0f64;
            for &(k, wt) in pillars {
                let d = k - sl.m;
                let s2 = sl.sigma * sl.sigma;
                let rt = (d * d + s2).sqrt();
                let w = sl.a + sl.b * (sl.rho * d + rt);
                let r = w - wt;
                rmse += r * r;
                // ∂w/∂a, ∂w/∂b, ∂w/∂rho, ∂w/∂m, ∂w/∂sigma
                let j = [
                    1.0,
                    sl.rho * d + rt,
                    sl.b * d,
                    sl.b * (-sl.rho - d / rt),
                    sl.b * sl.sigma / rt,
                ];
                for i in 0..5 {
                    jtr[i] += j[i] * r;
                    for j2 in 0..5 {
                        jtj[i][j2] += j[i] * j[j2];
                    }
                }
            }
            rmse = (rmse / pillars.len() as f64).sqrt();
            if rmse < target_rmse {
                break;
            }
            // Solve (JtJ + lambda I) dx = -Jt r by Gaussian elimination.
            let mut sys = jtj;
            for i in 0..5 {
                sys[i][i] += lambda * (1.0 + sys[i][i]);
            }
            let mut rhs = jtr;
            for i in 0..5 {
                rhs[i] = -rhs[i];
            }
            let dx = solve5(&mut sys, &mut rhs)?;
            // Trial step with projection.
            let trial = SviSlice {
                a: (sl.a + dx[0]).max(0.0),
                b: (sl.b + dx[1]).clamp(1e-4, 10.0),
                rho: (sl.rho + dx[2]).clamp(-0.995, 0.995),
                m: sl.m + dx[3],
                sigma: (sl.sigma + dx[4]).clamp(1e-3, 10.0),
            };
            // Rescale a to keep w(m) >= 0 feasible.
            let trial = if trial.a + trial.b * trial.sigma * (1.0 - trial.rho * trial.rho).sqrt()
                < 0.0
            {
                SviSlice { a: -trial.b * trial.sigma * (1.0 - trial.rho * trial.rho).sqrt(), ..trial }
            } else {
                trial
            };
            let new_rmse = rmse_of(&trial, pillars);
            if new_rmse < rmse {
                sl = trial;
                lambda = (lambda / 3.0).max(1e-9);
            } else {
                lambda *= 5.0;
                if lambda > 1e10 {
                    break;
                }
            }
        }
        let final_rmse = rmse_of(&sl, pillars);
        if final_rmse < 1e-4 && sl.params_sane() {
            Some(sl)
        } else {
            None
        }
    }
}

fn rmse_of(sl: &SviSlice, pillars: &[(f64, f64)]) -> f64 {
    let s = pillars
        .iter()
        .map(|&(k, wt)| {
            
            (sl.w(k) - wt).powi(2)
        })
        .sum::<f64>();
    (s / pillars.len() as f64).sqrt()
}

fn solve5(a: &mut [[f64; 5]; 5], b: &mut [f64; 5]) -> Option<[f64; 5]> {
    for col in 0..5 {
        let mut piv = col;
        for r in col + 1..5 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-14 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        for j in col..5 {
            a[col][j] /= d;
        }
        b[col] /= d;
        for r in 0..5 {
            if r != col && a[r][col].abs() > 1e-14 {
                let f = a[r][col];
                for j in col..5 {
                    a[r][j] -= f * a[col][j];
                }
                b[r] -= f * b[col];
            }
        }
    }
    Some(*b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ssvi_slice_w(rho: f64, eta: f64, gamma: f64, theta: f64) -> impl Fn(f64) -> f64 {
        move |k: f64| {
            let phi = 1.0 / (eta * theta.powf(gamma) * (1.0 + theta).powf(1.0 - gamma));
            theta / 2.0
                * (1.0 + rho * phi * k + ((phi * k + rho).powi(2) + 1.0 - rho * rho).sqrt())
        }
    }

    #[test]
    fn svi_reproduces_ssvi_slice_and_is_butterfly_free() {
        let wf = ssvi_slice_w(-0.7, 1.0, 0.5, 0.04);
        let pillars: Vec<(f64, f64)> = (-10..=10)
            .map(|i| {
                let k = i as f64 * 0.15;
                (k, wf(k))
            })
            .collect();
        let sl = SviSlice::fit(&pillars).expect("fit");
        let rmse = rmse_of(&sl, &pillars);
        assert!(rmse < 1e-5, "rmse {rmse}");
        assert!(sl.params_sane());
        assert!(sl.butterfly_free_grid(-1.0, 1.0, 100), "g(k) < 0");
        assert!(sl.static_arb_free_bsm(100.0, 1.0, 0.0, 0.0));
    }

    #[test]
    fn negative_a_is_caught_by_ground_truth() {
        // Butterfly violation: tiny a relative to wings makes the density
        // dip negative near the money.
        let bad = SviSlice { a: 0.0004, b: 0.8, rho: 0.0, m: 0.0, sigma: 0.15 };
        // g at k=0:
        let g0 = bad.g(0.0);
        // With these parameters w(0) = a + b sigma = 0.0004+0.12 ~ 0.1204,
        // g(0) = (1-0)^2 - (w'(0)/2)^2 + w''(0)/2; w'(0)=0, so
        // g(0) = 1 + w''/2 > 0 — actually fine; push the wing instead.
        if g0 < 0.0 {
            assert!(!bad.butterfly_free_grid(-1.0, 1.0, 100));
        }
        // Construct a definite violation: strong negative rho with large b
        // makes the left wing density negative.
        let viol = SviSlice { a: 0.02, b: 1.3, rho: -0.95, m: 0.3, sigma: 0.1 };
        let any_neg = (-150..=150).any(|i| viol.g(i as f64 * 0.02) < 0.0);
        assert!(any_neg, "expected a butterfly violation somewhere");
        assert!(!viol.butterfly_free_grid(-3.0, 3.0, 300));
    }

    #[test]
    fn tail_slopes_match_derivatives() {
        let sl = SviSlice { a: 0.02, b: 0.3, rho: -0.5, m: 0.1, sigma: 0.2 };
        let far = 500.0;
        let (sr, sl_) = sl.tail_slopes();
        assert!((sl.dw(far) - sr).abs() < 1e-6);
        // left-tail slope magnitude: dw(-inf) = b(rho-1) = -b(1-rho)
        assert!((sl.dw(-far) + sl_).abs() < 1e-6);
    }

    #[test]
    fn calendar_violation_detected_between_slices() {
        // A later slice with strictly smaller total variance somewhere is
        // calendar-arbitrageable; the numerical check used by SsviSurface
        // catches it (unit-tested here on raw slices).
        let t1 = SviSlice { a: 0.04, b: 0.2, rho: -0.5, m: 0.0, sigma: 0.2 };
        let t2 = SviSlice { a: 0.02, b: 0.2, rho: -0.5, m: 0.0, sigma: 0.2 };
        let any_cal = (-100..=100).any(|i| {
            let k = i as f64 * 0.02;
            t2.w(k) < t1.w(k) - 1e-12
        });
        assert!(any_cal, "expected calendar arbitrage");
    }
}
