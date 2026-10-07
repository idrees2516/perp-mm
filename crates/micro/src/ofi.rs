//! Order-flow imbalance (Cont–Kukanov–Stoikov 2014, arXiv:1011.6402,
//! "The price impact of order book events").
//!
//! Exact per-event contribution (verified against the paper):
//! ```text
//! e_n = 1{Pb_n >= Pb_{n-1}} qb_n  -  1{Pb_n <= Pb_{n-1}} qb_{n-1}
//!     - 1{Pa_n <= Pa_{n-1}} qa_n  +  1{Pa_n >= Pa_{n-1}} qa_{n-1}
//! ```
//! (bid-side quantities signed +, ask-side -; the four cases: same-price
//! size change -> `q_n - q_{n-1}`; best-price improvement -> new queue
//! size; best-price decrease -> previous queue size).
//!
//! The stylized model in the paper: `Delta P = OFI / (2 D) + eps` with `D`
//! the book depth — we fit the impact coefficient online with RLS and
//! expose the depth-implied prior for sanity-checking.

/// Rolling OFI accumulator with online impact regression.
pub struct OfiTracker {
    prev_bid: Option<(u64, u64)>,
    prev_ask: Option<(u64, u64)>,
    /// OFI summed over the active window (lots).
    pub window_ofi: f64,
    /// Window length in events (reset by caller).
    pub window_len: usize,
    /// Last event contribution awaiting a mid-change observation.
    pending: Option<f64>,
    // RLS for dMid = beta * ofi
    beta: f64,
    p: f64,
    n: u64,
    /// Total events seen.
    pub events: u64,
}

impl OfiTracker {
    pub fn new() -> OfiTracker {
        OfiTracker {
            prev_bid: None,
            prev_ask: None,
            window_ofi: 0.0,
            window_len: 0,
            pending: None,
            beta: 0.0,
            p: 1e6,
            n: 0,
            events: 0,
        }
    }

    /// Feed the current best bid/ask (price ticks, size lots). Returns the
    /// per-event contribution `e_n` (None for the first event).
    pub fn update(&mut self, bid: u64, bid_sz: u64, ask: u64, ask_sz: u64) -> Option<f64> {
        self.events += 1;
        let e = match (self.prev_bid, self.prev_ask) {
            (Some((pb, pqb)), Some((pa, pqa))) => {
                let mut e = 0.0f64;
                // bid side
                if bid >= pb {
                    e += bid_sz as f64;
                }
                if bid <= pb {
                    e -= pqb as f64;
                }
                // ask side
                if ask <= pa {
                    e -= ask_sz as f64;
                }
                if ask >= pa {
                    e += pqa as f64;
                }
                e
            }
            _ => {
                self.prev_bid = Some((bid, bid_sz));
                self.prev_ask = Some((ask, ask_sz));
                return None;
            }
        };
        // RLS update: dMid (ticks) ~ beta * e (lots)
        // The caller records mid changes via `observe_mid_change`.
        self.pending = Some(e);
        self.prev_bid = Some((bid, bid_sz));
        self.prev_ask = Some((ask, ask_sz));
        self.window_ofi += e;
        self.window_len += 1;
        Some(e)
    }

    /// Record the mid change (ticks) since the previous event, associating
    /// it with the last event's OFI contribution (impact regression).
    pub fn observe_mid_change(&mut self, d_mid_ticks: f64) {
        if let Some(e) = self.pending.take() {
            if e != 0.0 && d_mid_ticks.is_finite() {
                self.n += 1;
                // recursive least squares, one regressor
                let p_x = self.p * e;
                let denom = 1.0 + e * p_x;
                let k = p_x / denom;
                let err = d_mid_ticks - self.beta * e;
                self.beta += k * err;
                self.p = (self.p - k * e * self.p) / 0.9995;
            }
        }
    }

    /// Fitted impact coefficient `dMid_ticks ~ beta * OFI_lots`.
    pub fn impact_coef(&self) -> f64 {
        self.beta
    }

    /// Number of regression observations.
    pub fn impact_samples(&self) -> u64 {
        self.n
    }

    /// Depth-implied prior `beta0 = 1 / (2 D)` for depth `d` lots.
    pub fn depth_prior(&self, d: u64) -> f64 {
        if d == 0 {
            0.0
        } else {
            1.0 / (2.0 * d as f64)
        }
    }

    /// Rolling-window OFI (lots).
    pub fn ofi(&self) -> f64 {
        self.window_ofi
    }

    /// Reset the rolling window (call at window boundaries).
    pub fn reset_window(&mut self) {
        self.window_ofi = 0.0;
        self.window_len = 0;
    }

    /// Short-horizon mid forecast in ticks: `beta * ofi` (decayed).
    pub fn mid_forecast_ticks(&self, decay: f64) -> f64 {
        self.beta * self.window_ofi * decay
    }
}

impl Default for OfiTracker {
    fn default() -> Self {
        OfiTracker::new()
    }
}
