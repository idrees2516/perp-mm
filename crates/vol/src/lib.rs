//! # vol — volatility surface & option market-making mathematics
//!
//! Zero-dependency (workspace-internal only) implementations of:
//! - a robust Black–Scholes implied-volatility solver
//!   (Newton with analytic vega + safeguarded bisection; cf. Jäckel,
//!   "Let's Be Rational" (2014) for the general problem),
//! - the raw SVI slice of Gatheral (2004) with Gatheral–Jacquier
//!   "Arbitrage-free SVI volatility surfaces" (2014) butterfly
//!   (density `g(k) >= 0`) and calendar checks,
//! - the SSVI surface `w(k,theta)` with Heston-like `phi(theta)`
//!   and its (sufficient) no-butterfly condition
//!   `theta phi(theta)^2 (1+|rho|) <= 4` (Gatheral–Jacquier Thm 4.2),
//!   plus Mingone, "No arbitrage global parametrization for the eSSVI
//!   volatility surface" (2022) as the global-conditions reference,
//! - full BSM greeks including vanna and volga (all closed form, all
//!   finite-difference verified),
//! - the vega-approximation option market-making model
//!   (Bergault & Guéant, "Algorithmic Market Making for Options":
//!   the option book collapses to its vega, and the quoting problem
//!   becomes the GLFT problem in vol space) with a
//!   Whalley–Wilmott-style no-trade hedge band whose closed form is
//!   re-derived here and Monte-Carlo validated.
//!
//! Every no-arbitrage condition is additionally verified against the
//! numerical ground truth (Breeden–Litzenberger densities and
//! call-price monotonicity/convexity on strike grids).

pub mod greeks;
pub mod optmm;
pub mod solver;
pub mod ssvi;
pub mod svi;
pub mod volga;

pub use greeks::FullGreeks;
pub use optmm::{HedgeCadence, OptMm};
pub use solver::{implied_vol, IvResult};
pub use ssvi::SsviSurface;
pub use svi::SviSlice;
pub use volga::{Overhedge, VannaVolga, Wings};

/// Standard normal density.
#[inline]
pub fn npdf(x: f64) -> f64 {
    (-0.5 * x * x - 0.5 * (2.0 * std::f64::consts::PI).ln()).exp()
}

/// Standard normal CDF (Zelen–Severo rational approximation, ~1e-7
/// absolute — the fast path for hot loops).
#[inline]
pub fn ncdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t * (0.319381530
        + t * (-0.356563782
            + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let phi = 1.0 - npdf(x) * poly;
    if x >= 0.0 {
        phi
    } else {
        1.0 - phi
    }
}

/// Machine-precision standard normal CDF: 12-node Gauss–Legendre
/// quadrature of the density over unit-half segments, with nodes
/// computed once by Newton iteration on Legendre polynomials (cached).
/// Used by the greeks pricer where 1e-7 CDF noise would otherwise be
/// amplified by finite-difference denominators.
pub fn ncdf_hi(x: f64) -> f64 {
    const CAP: f64 = 8.4; // tail below 1e-16
    if x >= CAP {
        return 1.0;
    }
    if x <= -CAP {
        return 0.0;
    }
    static GL: std::sync::OnceLock<(Vec<f64>, Vec<f64>)> = std::sync::OnceLock::new();
    let (nodes, weights) = GL.get_or_init(|| gauss_legendre(12));
    let a = x.abs();
    // integrate phi over [0, a] in segments of width <= 0.5
    let n_seg = (a / 0.5).ceil().max(1.0) as usize;
    let w = a / n_seg as f64;
    let mut sum = 0.0;
    for s in 0..n_seg {
        let lo = s as f64 * w;
        let mid = lo + 0.5 * w;
        for (i, &xi) in nodes.iter().enumerate() {
            let t = mid + 0.5 * w * xi;
            sum += weights[i] * npdf(t);
        }
    }
    // integral over the segment = half-width * GL sum
    let mass = 0.5 * w * sum;
    if x >= 0.0 {
        0.5 + mass
    } else {
        0.5 - mass
    }
}

/// Gauss–Legendre nodes/weights on [-1, 1] (Newton on P_n, standard
/// initial guesses; machine-precision convergence for smooth weights).
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0f64; n];
    let mut wgt = vec![0.0f64; n];
    for i in 0..n {
        // initial guess (Chebyshev-like)
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 0.0;
        for _ in 0..100 {
            // evaluate P_n(z), P_n'(z) by recurrence
            let mut p0 = 1.0f64;
            let mut p1 = z;
            for k in 2..=n {
                let p2 = ((2.0 * k as f64 - 1.0) * z * p1 - (k as f64 - 1.0) * p0)
                    / k as f64;
                p0 = p1;
                p1 = p2;
            }
            // P_n' via the relation n (z P_n - P_{n-1}) / (z^2 - 1)
            dp = n as f64 * (z * p1 - p0) / (z * z - 1.0);
            let dz = p1 / dp;
            z -= dz;
            if dz.abs() < 1e-15 {
                break;
            }
        }
        x[i] = z;
        wgt[i] = 2.0 / ((1.0 - z * z) * dp * dp);
    }
    (x, wgt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ncdf_hi_matches_reference_values() {
        let refs = [
            (0.0, 0.5),
            (1.0, 0.841_344_746_068_542_9),
            (-1.0, 0.158_655_253_931_457_07),
            (2.0, 0.977_249_868_051_820_8),
            (-2.5, 0.006209665325776159),
            (3.0, 0.998_650_101_968_369_9),
            (5.37, 0.999999960631679),
            (-6.1, 5.303423256108886e-10),
        ];
        for &(z, r) in &refs {
            assert!((ncdf_hi(z) - r).abs() < 1e-13, "ncdf_hi({z}) = {} vs {r}", ncdf_hi(z));
        }
        // Integration consistency: mass over a symmetric range.
        assert!((ncdf_hi(0.7) - ncdf_hi(-0.7) - 0.5160726955538539).abs() < 1e-13);
    }

    #[test]
    fn fast_ncdf_within_tolerance() {
        for z in [-4.0, -2.0, -0.5, 0.0, 0.5, 2.0, 4.0] {
            assert!((ncdf(z) - ncdf_hi(z)).abs() < 2e-7, "z={z}");
        }
    }
}
