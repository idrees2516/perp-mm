//! Special functions: erf/erfc, normal pdf/cdf/quantile, Gumbel quantile.
//!
//! Hand-rolled (zero external dependencies):
//! * `erf`/`erfc` — Abramowitz & Stegun 7.1.26 (|err| <= 1.5e-7), with the
//!   continued-fraction style large-|x| guard via the same series.
//! * `norm_ppf` — Acklam's rational initial guess + one Newton polish
//!   against `norm_cdf` (final |err| ~ 1e-9 across (0,1)).
//! * Gumbel (standard): cdf `exp(-exp(-x))`, quantile `-ln(-ln p)`.

const SQRT2: f64 = std::f64::consts::SQRT_2;

/// Error function. Taylor series for |x| <= 3.5 (double precision),
/// optimal-truncated asymptotic expansion beyond (tail-accurate — needed
/// because the Lee–Mykland threshold sits beyond 6 sigma).
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    if ax <= 3.5 {
        let x2 = ax * ax;
        let mut term = ax;
        let mut sum = ax;
        for k in 1..80 {
            term *= -x2 / k as f64;
            let add = term / (2 * k + 1) as f64;
            sum += add;
            if add.abs() < 1e-18 {
                break;
            }
        }
        sign * (2.0 / std::f64::consts::PI.sqrt()) * sum
    } else {
        // erfc(ax) = e^{-ax^2} / (ax sqrt(pi)) * sum_k (-1)^k (2k-1)!!/(2ax^2)^k
        let two_x2 = 2.0 * ax * ax;
        let mut term = 1.0f64;
        let mut sum = 1.0f64;
        for k in 1..14 {
            term *= -(2 * k - 1) as f64 / two_x2;
            if term.abs() > 1e-17 {
                sum += term;
            } else {
                break;
            }
        }
        let erfc = (-ax * ax).exp() / (ax * std::f64::consts::PI.sqrt()) * sum;
        sign * (1.0 - erfc)
    }
}

/// Complementary error function.
pub fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

/// Standard normal density.
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard normal CDF.
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / SQRT2))
}

/// Standard normal quantile (inverse CDF), p in (0, 1).
pub fn norm_ppf(p: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&p));
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    // Acklam's rational approximation.
    const A: [f64; 6] = [
        -3.969_683_028_665_376e+01,
        2.209_460_984_245_205e+02,
        -2.759_285_104_469_687e+02,
        1.383_577_518_672_69e+02,
        -3.066_479_806_614_716e+01,
        2.506_628_277_459_239e+00,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e+01,
        1.615_858_368_580_409e+02,
        -1.556_989_798_598_866e+02,
        6.680_131_188_771_972e+01,
        -1.328_068_155_288_572e+01,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-03,
        -3.223_964_580_411_365e-01,
        -2.400_758_277_161_838e+00,
        -2.549_732_539_343_734e+00,
        4.374_664_141_464_968e+00,
        2.938_163_982_698_783e+00,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-03,
        3.224_671_290_700_398e-01,
        2.445_134_137_142_996e+00,
        3.754_408_661_907_416e+00,
    ];
    const P_LOW: f64 = 0.024_25;
    let mut x;
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        x = (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        x = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0);
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        x = -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }
    // Newton polish.
    let e = norm_cdf(x) - p;
    let u = e / norm_pdf(x);
    x -= u / (1.0 + 0.5 * x * u);
    x
}

/// Standard Gumbel CDF: `exp(-exp(-x))`.
pub fn gumbel_cdf(x: f64) -> f64 {
    (-(-x).exp()).exp()
}

/// Standard Gumbel quantile: `-ln(-ln p)`.
pub fn gumbel_ppf(p: f64) -> f64 {
    -((-p.ln()).ln())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_known_values() {
        assert!(erf(0.0).abs() < 1e-17);
        assert!((erf(1.0) - 0.842_700_792_949_715).abs() < 1e-12);
        assert!((erf(2.0) - 0.995_322_265_018_953).abs() < 1e-12);
        assert!((erf(-1.0) + 0.842_700_792_949_715).abs() < 1e-12);
        assert!((erf(3.0) - 0.999_977_909_503_001).abs() < 1e-12);
        // tail accuracy (critical for LM thresholds)
        assert!((erf(4.0) - 0.999_999_984_582_742).abs() < 1e-11);
        assert!((erf(5.0) - 0.999_999_999_998_463).abs() < 1e-12);
    }

    #[test]
    fn norm_roundtrip() {
        for &x in &[
            -4.0f64, -2.5, -1.0, -0.3, 0.0, 0.25, 1.0, 2.0, 3.5, 5.0,
        ] {
            let p = norm_cdf(x);
            let back = norm_ppf(p);
            assert!(
                (back - x).abs() < 1e-8,
                "roundtrip failed at x={x}: {back}"
            );
        }
        // reference values
        assert!((norm_cdf(1.96) - 0.975_002_104_851_779_5).abs() < 1e-6);
        assert!((norm_ppf(0.975) - 1.959_963_984_540_054).abs() < 1e-8);
    }

    #[test]
    fn gumbel() {
        assert!((gumbel_cdf(0.0) - std::f64::consts::E.powi(-1)).abs() < 1e-12);
        assert!((gumbel_ppf(0.5) - 0.366_512_920_581_664).abs() < 1e-9);
        let q99 = gumbel_ppf(0.99);
        assert!((gumbel_cdf(q99) - 0.99).abs() < 1e-12);
    }
}
