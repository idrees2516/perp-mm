//! Post-fill markout tracking and adverse-selection-adaptive spreads.
//!
//! Practical market-making desks widen quotes when their fills are
//! "toxic": filled quotes whose mid drifts through them shortly after
//! the fill (negative markouts). This module tracks per-fill markouts
//! at a configurable horizon, maintains an EWMA of the markout-to-
//! spread ratio, and produces a spread multiplier
//! `1 + theta * toxicity` (clamped) that the quoting strategies apply.
//! Cf. "Optimal Quoting under Adverse Selection and Price Reading"
//! (2026) and the client-flow-segmentation/markouts literature.

/// Markout tracker state.
pub struct MarkoutTracker {
    /// Horizon at which markouts are measured (steps).
    horizon_steps: usize,
    /// EWMA decay per resolved markout.
    alpha: f64,
    /// Toxicity sensitivity of the multiplier.
    theta: f64,
    /// Maximum multiplier.
    max_mult: f64,
    // state
    pending: Vec<PendingFill>,
    /// EWMA of markout / half-spread (signed; negative = adverse).
    ewma_ratio: f64,
    /// EWMA of the half-spread (quote units).
    spread_ewma: f64,
    resolved: u64,
}

#[derive(Clone, Copy)]
struct PendingFill {
    due_step: usize,
    /// +1 bought, -1 sold.
    signed: f64,
    /// Fill price (quote units).
    price: f64,
    /// Half-spread at fill time (quote units).
    half_spread: f64,
}

impl MarkoutTracker {
    pub fn new(horizon_steps: usize, theta: f64) -> MarkoutTracker {
        MarkoutTracker {
            horizon_steps: horizon_steps.max(1),
            alpha: 0.05,
            theta,
            max_mult: 3.0,
            pending: Vec::with_capacity(64),
            ewma_ratio: 0.0,
            spread_ewma: 0.0,
            resolved: 0,
        }
    }

    /// Record one of our passive fills.
    pub fn on_fill(&mut self, step: usize, signed: i64, price: f64, half_spread: f64) {
        self.pending.push(PendingFill {
            due_step: step + self.horizon_steps,
            signed: signed as f64,
            price,
            half_spread: half_spread.max(1e-9),
        });
        self.spread_ewma = if self.spread_ewma <= 0.0 {
            half_spread
        } else {
            self.spread_ewma * (1.0 - self.alpha) + half_spread * self.alpha
        };
    }

    /// Advance one step and resolve due markouts against the current mid.
    pub fn on_step(&mut self, step: usize, mid: f64) {
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].due_step <= step {
                let f = self.pending.remove(i);
                // markout: favorable when the mid moves in our direction
                // after the fill (bought + mid up).
                let markout = f.signed * (mid - f.price);
                let ratio = markout / f.half_spread;
                self.ewma_ratio = self.ewma_ratio * (1.0 - self.alpha) + ratio * self.alpha;
                self.resolved += 1;
            } else {
                i += 1;
            }
        }
    }

    /// Number of resolved markouts.
    pub fn resolved_count(&self) -> u64 {
        self.resolved
    }

    /// Current EWMA markout-to-half-spread ratio (negative = adverse).
    pub fn markout_ratio(&self) -> f64 {
        self.ewma_ratio
    }

    /// Toxicity in [0, 1]: zero when markouts are non-negative.
    pub fn toxicity(&self) -> f64 {
        (-self.ewma_ratio).clamp(0.0, 1.0)
    }

    /// Spread multiplier `1 + theta * toxicity` (clamped).
    pub fn spread_multiplier(&self) -> f64 {
        (1.0 + self.theta * self.toxicity()).min(self.max_mult)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_flow_keeps_multiplier_at_one() {
        let mut t = MarkoutTracker::new(5, 1.5);
        // Bought at 100, mid drifts up (favorable) each time.
        for step in 0..60 {
            if step % 2 == 0 {
                t.on_fill(step, 1, 100.0, 0.5);
            }
            t.on_step(step, 100.6);
        }
        assert!(t.resolved_count() > 5);
        assert!((t.spread_multiplier() - 1.0).abs() < 1e-9);
        assert!(t.toxicity() <= 0.0 + 1e-9);
    }

    #[test]
    fn toxic_flow_widens_the_spread() {
        let mut t = MarkoutTracker::new(5, 1.5);
        // We keep buying at 100 and the mid keeps dropping: adverse.
        for step in 0..200 {
            if step % 2 == 0 {
                t.on_fill(step, 1, 100.0, 0.5);
            }
            t.on_step(step, 99.4);
        }
        assert!(t.toxicity() > 0.2, "toxicity {}", t.toxicity());
        assert!(t.spread_multiplier() > 1.3, "mult {}", t.spread_multiplier());
        assert!(t.spread_multiplier() <= 3.0);
        assert!(t.markout_ratio() < 0.0);
    }

    #[test]
    fn mixed_flow_blends() {
        let mut t = MarkoutTracker::new(3, 2.0);
        // buys only; the mid alternates in slow blocks (4 steps up, 4
        // down) so resolved markouts mix favorable and adverse roughly
        // evenly — toxicity stays moderate, far from the pure-toxic case.
        for step in 0..600 {
            if step % 2 == 0 {
                t.on_fill(step, 1, 100.0, 0.5);
            }
            let up = (step / 4) % 2 == 0;
            let mid = if up { 100.4 } else { 99.6 };
            t.on_step(step, mid);
        }
        assert!(t.toxicity() < 0.35, "toxicity {}", t.toxicity());
        assert!(t.markout_ratio() > -1.0);
    }
}
