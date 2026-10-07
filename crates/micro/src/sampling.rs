//! Deterministic RNG (xoshiro256++) and samplers — zero dependencies,
//! bit-reproducible across platforms, shared by all downstream crates so
//! simulations are exactly repeatable from a seed.

/// xoshiro256++ PRNG.
pub struct Rng {
    s: [u64; 4],
    spare_normal: Option<f64>,
}

impl Rng {
    /// Seeded generator (SplitMix64 expansion; odd seeds not required).
    pub fn new(seed: u64) -> Rng {
        let mut sm = seed;
        let mut next = || {
            sm = sm.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Rng {
            s: [next(), next(), next(), next()],
            spare_normal: None,
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in [0, n).
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }

    /// True with probability p.
    #[inline]
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.uniform() < p
    }

    /// Standard normal via Box–Muller (with cached spare).
    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare_normal.take() {
            return z;
        }
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        let z0 = r * theta.cos();
        let z1 = r * theta.sin();
        self.spare_normal = Some(z1);
        z0
    }

    /// Exponential with rate `lambda`.
    pub fn exponential(&mut self, lambda: f64) -> f64 {
        (-self.uniform().max(1e-300).ln()) / lambda
    }

    /// Poisson(lambda) via Knuth (fine for lambda < ~30) with a normal
    /// fallback for large rates.
    pub fn poisson(&mut self, lambda: f64) -> u64 {
        if lambda < 30.0 {
            let l = (-lambda).exp();
            let mut k = 0u64;
            let mut p = 1.0f64;
            loop {
                p *= self.uniform();
                if p <= l {
                    return k;
                }
                k += 1;
            }
        } else {
            let n = (lambda + self.normal() * lambda.sqrt()).round();
            n.max(0.0) as u64
        }
    }

    /// Gamma(shape k integer, rate 1) — Erlang via exponentials / normals.
    pub fn erlang(&mut self, k: u32, rate: f64) -> f64 {
        let mut s = 0.0;
        for _ in 0..k.max(1) {
            s += self.exponential(rate);
        }
        s
    }

    /// AR(1) sequence with coefficient rho, length n, N(0,1) innovations.
    pub fn ar1(&mut self, rho: f64, n: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(n);
        let mut x = self.normal();
        out.push(x);
        for _ in 1..n {
            x = rho * x + (1.0 - rho * rho).sqrt() * self.normal();
            out.push(x);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn normal_moments() {
        let mut r = Rng::new(7);
        let n = 200_000;
        let (mut s, mut s2) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let z = r.normal();
            s += z;
            s2 += z * z;
        }
        let mean = s / n as f64;
        let var = s2 / n as f64 - mean * mean;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "var {var}");
    }

    #[test]
    fn poisson_mean() {
        let mut r = Rng::new(11);
        let n = 50_000;
        let lambda = 5.0;
        let s: u64 = (0..n).map(|_| r.poisson(lambda)).sum();
        let m = s as f64 / n as f64;
        assert!((m - lambda).abs() < 0.1, "poisson mean {m}");
    }
}
