//! BitMEX-style funding for perpetuals, matching the venue's calculator.
//!
//! * Premium: `clamp((mark_twap - index_twap) / index_twap, +/-5bps)`.
//! * Rate: `clamp(interest(1bp/8h) + premium, +/-75bps)` per 8-hour
//!   interval, longs pay shorts.
//! * Payments computed once per lot with **ceil on the magnitude**
//!   (house-favorable rounding), exactly zero-sum in `u128` quote-minor.

/// A TWAP ring buffer over the last `cap` samples.
pub struct TwapRing {
    buf: std::collections::VecDeque<f64>,
    cap: usize,
    sum: f64,
}

impl TwapRing {
    pub fn new(cap: usize) -> TwapRing {
        TwapRing {
            buf: std::collections::VecDeque::with_capacity(cap.max(1)),
            cap: cap.max(1),
            sum: 0.0,
        }
    }

    pub fn push(&mut self, sample: f64) {
        if self.buf.len() == self.cap {
            if let Some(old) = self.buf.pop_front() {
                self.sum -= old;
            }
        }
        self.buf.push_back(sample);
        self.sum += sample;
    }

    /// TWAP of the retained samples (0 when empty).
    pub fn twap(&self) -> f64 {
        if self.buf.is_empty() {
            0.0
        } else {
            self.sum / self.buf.len() as f64
        }
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

/// Compute `(premium, rate)` in **rate units per interval** (not bps).
///
/// `mark_twap`, `index_twap`: TWAP inputs; `interest`: per-interval
/// interest rate (venue default 1bp = 0.0001); clamps in rate units
/// (5bps premium, 75bps rate).
pub fn funding_rate(
    mark_twap: f64,
    index_twap: f64,
    interest: f64,
    premium_clamp: f64,
    rate_clamp: f64,
) -> (f64, f64) {
    let premium = if index_twap > 0.0 {
        (mark_twap - index_twap) / index_twap
    } else {
        0.0
    };
    let premium = premium.clamp(-premium_clamp, premium_clamp);
    let rate = (interest + premium).clamp(-rate_clamp, rate_clamp);
    (premium, rate)
}

/// Venue-default parameters: 1bp interest per 8h, +/-5bps premium clamp,
/// +/-75bps rate clamp.
pub const DEFAULT_INTEREST_8H: f64 = 0.0001;
pub const DEFAULT_PREMIUM_CLAMP: f64 = 0.0005;
pub const DEFAULT_RATE_CLAMP: f64 = 0.0075;

/// Exact funding payment in quote-minor units for a position of
/// `signed_lots` (positive = long; longs pay shorts when rate > 0),
/// with ceil-on-magnitude rounding.
///
/// `rate` is per interval; `lot_notional_quote_minor_at_1` =
/// `lot_size_base_minor * tick scaling` — the venue computes
/// `payment = ceil(|rate| * price * lots * lot_size)`, so we take the
/// position's mark `price_quote_minor` per base unit.
pub fn funding_payment_exact(
    signed_lots: i64,
    rate: f64,
    price_quote_minor_per_base: u128,
    lot_size_base_minor: u128,
) -> i128 {
    if signed_lots == 0 || rate == 0.0 {
        return 0;
    }
    // notional_base_minor * price_quote_minor / 1e(base precision) handled
    // by caller; here: |rate| * notional in quote-minor with ceil.
    let notional = (signed_lots.unsigned_abs() as u128)
        .checked_mul(lot_size_base_minor)
        .and_then(|b| b.checked_mul(price_quote_minor_per_base));
    let notional = match notional {
        Some(n) => n,
        None => return 0,
    };
    // rate in parts per 1e9 for fixed-point precision
    let rate_pp = (rate.abs() * 1e9) as u128;
    let numerator = notional
        .checked_mul(rate_pp)
        .and_then(|n| n.checked_div(1_000_000_000))
        .unwrap_or(0);
    let ceil_adj = if numerator
        .checked_mul(1_000_000_000)
        .map(|x| x < notional.saturating_mul(rate_pp))
        .unwrap_or(false)
    {
        1
    } else {
        0
    };
    let payment = numerator + ceil_adj;
    // longs pay when rate > 0: their account decreases
    let sign = if rate > 0.0 { -1 } else { 1 };
    if signed_lots > 0 {
        sign * payment as i128
    } else {
        -sign * payment as i128
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twap_ring() {
        let mut r = TwapRing::new(4);
        for &x in &[1.0, 2.0, 3.0, 4.0, 5.0] {
            r.push(x);
        }
        assert_eq!(r.len(), 4);
        assert!((r.twap() - 3.5).abs() < 1e-12); // mean of 2..5
    }

    #[test]
    fn funding_rate_clamps() {
        // premium clamped at +5bps
        let (p, r) = funding_rate(1.02, 1.0, DEFAULT_INTEREST_8H, DEFAULT_PREMIUM_CLAMP, DEFAULT_RATE_CLAMP);
        assert!((p - 0.0005).abs() < 1e-12);
        assert!((r - (DEFAULT_INTEREST_8H + 0.0005)).abs() < 1e-12);
        // negative premium clamped at -5bps
        let (p2, r2) = funding_rate(0.999, 1.0, DEFAULT_INTEREST_8H, DEFAULT_PREMIUM_CLAMP, DEFAULT_RATE_CLAMP);
        assert!((p2 + 0.0005).abs() < 1e-12);
        assert!((r2 - (DEFAULT_INTEREST_8H - 0.0005)).abs() < 1e-12);
        // rate floor clamp binds only with negative interest
        let (_, r3) = funding_rate(0.5, 1.0, -0.01, DEFAULT_PREMIUM_CLAMP, DEFAULT_RATE_CLAMP);
        assert!((r3 + DEFAULT_RATE_CLAMP).abs() < 1e-12);
    }

    #[test]
    fn payment_rounding_and_signs() {
        // long pays when rate positive (negative account delta)
        let pay = funding_payment_exact(3, 0.0001, 100_000_000, 10_000);
        // notional = 3 * 10_000 * 100_000_000 = 3e12; * 1e-4 = 3e8
        assert_eq!(pay, -300_000_000);
        // short receives
        let recv = funding_payment_exact(-3, 0.0001, 100_000_000, 10_000);
        assert_eq!(recv, 300_000_000);
        // zero rate / zero position
        assert_eq!(funding_payment_exact(3, 0.0, 100, 10), 0);
        assert_eq!(funding_payment_exact(0, 0.01, 100, 10), 0);
    }
}
