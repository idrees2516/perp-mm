//! Execution probabilities for passive orders at a given queue position —
//! the Cont–de Larrard (2013) queueing layer, in the exact form the
//! queue-aware quoting strategy needs.
//!
//! Model: our order rests behind `queue_ahead` lots. Front-of-queue
//! consumption events (market orders + cancellations ahead) arrive as a
//! Poisson process at rate `mu_eff`; refills join *behind* us and can
//! never delay our turn. Therefore:
//!
//! * Our fill time is `Erlang(queue_ahead + 1, mu_eff)` — so the
//!   probability of filling within a horizon is the Erlang CDF complement
//!   (a Poisson tail), computed in closed form by
//!   [`fill_probability_within`].
//! * The level cannot die while our order is inside it and refills
//!   protect it (the away-move consumes us first) — the "queue empties
//!   before we fill" race is *degenerate* (probability 1), which Monte
//!   Carlo in the previous revision confirmed. The non-degenerate race is
//!   against an **exogenous** away-move clock (price ticking away because
//!   of the adjacent side or a quote jump), giving the exact closed form
//!   [`fill_probability_vs_clock`]: `P(Erlang(m, mu) < Exp(nu)) =
//!   (mu/(mu+nu))^m` (Laplace transform of the Erlang at `nu`).
//! * [`expected_fill_time`] = `(queue_ahead + 1) / mu_eff`.

/// `P(our order fills within `horizon` seconds)`:
/// `P(Erlang(m, mu_eff) <= h) = 1 - e^{-mu h} sum_{j<m} (mu h)^j / j!`
/// with `m = queue_ahead + 1`.
pub fn fill_probability_within(queue_ahead: u64, mu_eff: f64, horizon: f64) -> f64 {
    if mu_eff <= 0.0 || horizon <= 0.0 {
        return 0.0;
    }
    let m = queue_ahead + 1;
    let x = mu_eff * horizon;
    // Poisson upper tail: P(N >= m) for N ~ Poisson(x).
    // Sum the lower terms with Kahan-ish accumulation; x is moderate.
    let mut term = (-x).exp(); // j = 0
    let mut lower = term;
    for j in 1..m {
        term *= x / j as f64;
        lower += term;
        if term < 1e-30 {
            break;
        }
    }
    (1.0 - lower).clamp(0.0, 1.0)
}

/// `P(fill before an exogenous away-move clock fires)`:
/// `P(Erlang(m, mu_eff) < Exp(nu)) = (mu/(mu+nu))^m`, `m = queue_ahead+1`.
#[inline]
pub fn fill_probability_vs_clock(queue_ahead: u64, mu_eff: f64, away_rate: f64) -> f64 {
    if mu_eff <= 0.0 {
        return 0.0;
    }
    if away_rate <= 0.0 {
        return 1.0;
    }
    let m = (queue_ahead + 1) as f64;
    (mu_eff / (mu_eff + away_rate)).powf(m)
}

/// Expected time (seconds) until our order fills: `(qa + 1) / mu_eff`.
#[inline]
pub fn expected_fill_time(queue_ahead: u64, mu_eff: f64) -> f64 {
    if mu_eff <= 0.0 {
        return f64::INFINITY;
    }
    (queue_ahead + 1) as f64 / mu_eff
}

#[cfg(test)]
mod tests {
    use super::*;
    use micro::Rng;

    #[test]
    fn within_horizon_matches_mc() {
        // MC: fill time = time of the (qa+1)-th event of a Poisson(mu).
        let cases = [
            (3u64, 1.0f64, 4.0f64),
            (8, 0.5, 20.0),
            (0, 2.0, 1.0),
            (12, 0.7, 30.0),
        ];
        for (i, &(qa, mu, h)) in cases.iter().enumerate() {
            let mut rng = Rng::new(300 + i as u64);
            let n = 20_000;
            let mut filled = 0usize;
            for _ in 0..n {
                let mut t = 0.0;
                let mut ok = false;
                for _ in 0..(qa + 1) {
                    t += rng.exponential(mu);
                    if t > h {
                        break;
                    }
                }
                if t <= h {
                    ok = true;
                }
                if ok {
                    filled += 1;
                }
            }
            let mc = filled as f64 / n as f64;
            let exact = fill_probability_within(qa, mu, h);
            assert!(
                (exact - mc).abs() < 0.02,
                "qa={qa} mu={mu} h={h}: exact {exact} vs mc {mc}"
            );
        }
    }

    #[test]
    fn vs_clock_matches_mc() {
        // MC: race Erlang(qa+1, mu) vs Exp(nu).
        let cases = [(3u64, 1.0f64, 0.8f64), (8, 0.5, 0.4), (0, 2.0, 1.0)];
        for (i, &(qa, mu, nu)) in cases.iter().enumerate() {
            let mut rng = Rng::new(400 + i as u64);
            let n = 20_000;
            let mut wins = 0usize;
            for _ in 0..n {
                let death = rng.exponential(nu);
                let mut t = 0.0;
                for _ in 0..(qa + 1) {
                    t += rng.exponential(mu);
                }
                let fill = t;
                if fill < death {
                    wins += 1;
                }
            }
            let mc = wins as f64 / n as f64;
            let exact = fill_probability_vs_clock(qa, mu, nu);
            assert!(
                (exact - mc).abs() < 0.02,
                "qa={qa} mu={mu} nu={nu}: exact {exact} vs mc {mc}"
            );
        }
    }

    #[test]
    fn edge_cases() {
        // Front of the queue, no away-clock: certain fill.
        assert!((fill_probability_vs_clock(0, 1.0, 0.0) - 1.0).abs() < 1e-12);
        assert_eq!(fill_probability_within(0, 1.0, 0.0), 0.0);
        // monotone in horizon and in mu
        let a = fill_probability_within(5, 1.0, 5.0);
        let b = fill_probability_within(5, 1.0, 10.0);
        let c = fill_probability_within(5, 2.0, 5.0);
        assert!(b > a && c > a);
        assert!((expected_fill_time(10, 2.0) - 5.5).abs() < 1e-12);
        assert!(expected_fill_time(10, 0.0).is_infinite());
    }
}
