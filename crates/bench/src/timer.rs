//! Nanosecond-precision timing.

#![allow(unsafe_code)] // rdtsc intrinsic only

use std::time::Instant;

/// A monotonic nanosecond stopwatch.
#[derive(Clone, Copy)]
pub struct NsTimer {
    start: Instant,
}

impl NsTimer {
    #[inline]
    pub fn start() -> NsTimer {
        NsTimer {
            start: Instant::now(),
        }
    }

    /// Elapsed nanoseconds.
    #[inline]
    pub fn elapsed_ns(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }
}

/// Raw TSC cycles (x86_64 only; 0 elsewhere). Serialized read with
/// `lfence` around `_rdtsc`.
#[inline]
pub fn rdtsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        let lo: u32;
        let hi: u32;
        unsafe {
            core::arch::x86_64::_mm_lfence();
            let t = core::arch::x86_64::_rdtsc();
            lo = t as u32;
            hi = (t >> 32) as u32;
        }
        ((hi as u64) << 32) | lo as u64
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0
    }
}

/// Summary statistics over a sample.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub n: usize,
    pub min: f64,
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
    pub max: f64,
    pub mean: f64,
}

/// Percentile summary of a sample (values in ns or cycles).
pub fn percentiles(mut vals: Vec<f64>) -> Stats {
    if vals.is_empty() {
        return Stats::default();
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vals.len();
    let q = |p: f64| -> f64 {
        let idx = ((p / 100.0) * (n as f64 - 1.0)).round() as usize;
        vals[idx.min(n - 1)]
    };
    let mean = vals.iter().sum::<f64>() / n as f64;
    Stats {
        n,
        min: vals[0],
        p50: q(50.0),
        p90: q(90.0),
        p99: q(99.0),
        max: vals[n - 1],
        mean,
    }
}

/// Run `f` `n` times (after `warmup` unmeasured runs), returning ns stats.
pub fn measure_ns<F: FnMut()>(warmup: usize, n: usize, mut f: F) -> Stats {
    for _ in 0..warmup {
        f();
    }
    let mut vals = Vec::with_capacity(n);
    for _ in 0..n {
        let t = NsTimer::start();
        f();
        vals.push(t.elapsed_ns() as f64);
    }
    percentiles(vals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timer_monotonic() {
        let t = NsTimer::start();
        let a = t.elapsed_ns();
        std::thread::sleep(std::time::Duration::from_micros(200));
        let b = t.elapsed_ns();
        assert!(b > a);
        assert!(b > 100_000, "slept 200us, measured {b}ns");
    }

    #[test]
    fn stats_percentiles() {
        let s = percentiles((1..=100).map(|x| x as f64).collect());
        assert_eq!(s.n, 100);
        assert!((s.min - 1.0).abs() < 1e-9);
        assert!((s.max - 100.0).abs() < 1e-9);
        assert!((s.p50 - 50.0).abs() < 1.5); // round-to-nearest index
        assert!((s.mean - 50.5).abs() < 1e-9);
    }

    #[test]
    fn measure_works() {
        let s = measure_ns(10, 1000, || {
            let mut x = 0u64;
            for i in 0..100u64 {
                x += i.wrapping_mul(3);
            }
            std::hint::black_box(x);
        });
        assert!(s.p50 > 0.0);
        assert!(s.p99 >= s.p50);
        assert!(s.max >= s.p99);
    }
}
