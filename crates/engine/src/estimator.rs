//! The streaming estimator stack fed by book/trade events.

use crate::config::EngineConfig;
use micro::jump::LeeMykland;
use micro::microprice::MicroPriceFit;
use micro::ofi::OfiTracker;
use ob::OrderBook;

/// The market state snapshot handed to strategies each requote.
#[derive(Clone, Debug)]
pub struct MarketState {
    /// Mid (quote units).
    pub mid: f64,
    /// Best bid/ask (ticks) and sizes (lots).
    pub best_bid: Option<(u64, u64)>,
    pub best_ask: Option<(u64, u64)>,
    /// Spread in quote units.
    pub spread: f64,
    /// Volatility estimate per sqrt(second) (EWMA fast).
    pub sigma_fast: f64,
    /// Volatility estimate per sqrt(second) (EWMA slow).
    pub sigma_slow: f64,
    /// Rough-vol forecast for the quoting horizon (per sqrt(second)).
    pub sigma_rough: f64,
    /// Hurst estimate (refreshed on the slow cadence; NaN before).
    pub hurst: f64,
    /// Micro-price (quote units).
    pub micro_price: f64,
    /// Top-of-book imbalance (0..1, >0.5 = buy pressure).
    pub imbalance: f64,
    /// Rolling OFI (lots).
    pub ofi: f64,
    /// Fitted OFI impact coefficient (ticks per lot).
    pub ofi_impact: f64,
    /// Rolling effective spread estimate (Roll/serial-dep; log units).
    pub spread_est_log: f64,
    /// Lee-Mykland jump flag on the latest bar.
    pub jump_flag: bool,
    /// CLF liquidity factor of the bid/ask ladders (lower = deeper).
    pub clf_bid: Option<f64>,
    pub clf_ask: Option<f64>,
}

/// The estimator stack.
pub struct EstimatorStack {
    /// EWMA fast (per ~10 steps).
    lambda_fast: f64,
    /// EWMA slow (per ~200 steps).
    lambda_slow: f64,
    var_fast: f64,
    var_slow: f64,
    /// Mid log-return series for the spread estimators (ring).
    rets: std::collections::VecDeque<f64>,
    rets_cap: usize,
    /// Last mid (for returns).
    last_mid: Option<f64>,
    /// Jump detector.
    lm: LeeMykland,
    /// Micro-price fitter.
    mp: MicroPriceFit,
    /// OFI tracker.
    ofi: OfiTracker,
    /// Rough-vol cadence state.
    rough_every: usize,
    rough_counter: usize,
    vol_hist: std::collections::VecDeque<f64>,
    hurst: f64,
    sigma_rough: f64,
    /// Book imbalance + best sizes cache.
    last_bid: Option<(u64, u64)>,
    last_ask: Option<(u64, u64)>,
    /// Spread estimate.
    spread_est_log: f64,
    dt: f64,
}

impl EstimatorStack {
    pub fn new(cfg: &EngineConfig) -> EstimatorStack {
        let steps_per_day = (cfg.horizon / cfg.dt) as usize;
        EstimatorStack {
            lambda_fast: 1.0 - 1.0 / 20.0,
            lambda_slow: 1.0 - 1.0 / 400.0,
            var_fast: cfg.sigma_per_step().powi(2),
            var_slow: cfg.sigma_per_step().powi(2),
            rets: std::collections::VecDeque::with_capacity(4096),
            rets_cap: 3000,
            last_mid: None,
            lm: LeeMykland::new(64, steps_per_day.max(100), 0.001),
            mp: MicroPriceFit::new(0.999),
            ofi: OfiTracker::new(),
            rough_every: 128,
            rough_counter: 0,
            vol_hist: std::collections::VecDeque::with_capacity(1024),
            hurst: f64::NAN,
            sigma_rough: cfg.sigma_per_step(),
            last_bid: None,
            last_ask: None,
            spread_est_log: 0.0,
            dt: cfg.dt,
        }
    }

    /// Feed one observation (called per venue step).
    pub fn update(
        &mut self,
        mid: f64,
        best_bid: Option<(u64, u64)>,
        best_ask: Option<(u64, u64)>,
        tick_size: f64,
    ) {
        // log return
        if let Some(lm) = self.last_mid {
            if lm > 0.0 && mid > 0.0 {
                let r = (mid / lm).ln();
                // variance EWMAs (per step)
                self.var_fast =
                    self.lambda_fast * self.var_fast + (1.0 - self.lambda_fast) * r * r;
                self.var_slow =
                    self.lambda_slow * self.var_slow + (1.0 - self.lambda_slow) * r * r;
                self.rets.push_back(r);
                if self.rets.len() > self.rets_cap {
                    self.rets.pop_front();
                }
                // jump test on the streaming return
                let _ = self.lm.update(r);
                // micro-price: predict next mid change from imbalance
                if let (Some((_bp, bq)), Some((_ap, aq))) = (best_bid, best_ask) {
                    let i = bq as f64 / (bq + aq).max(1) as f64;
                    self.mp.update(i, r * mid); // approx next-mid change in quote units
                }
                // OFI mid-change observation
                self.ofi.observe_mid_change(r * mid / tick_size);
            }
        }
        self.last_mid = Some(mid);
        // OFI event
        if let (Some(prev), Some(cur)) = (self.last_bid, best_bid) {
            // bid contributes; ask handled next block
            let _ = self.ofi.update(cur.0, cur.1, self.last_ask.map(|a| a.0).unwrap_or(cur.0 + 1), self.last_ask.map(|a| a.1).unwrap_or(cur.1));
            let _ = prev;
        } else if let Some(cur) = best_bid {
            let _ = self.ofi.update(cur.0, cur.1, best_ask.map(|a| a.0).unwrap_or(cur.0 + 1), best_ask.map(|a| a.1).unwrap_or(cur.1));
        }
        self.last_bid = best_bid;
        self.last_ask = best_ask;
        // spread estimate refresh (cheap; every 64 steps)
        if self.rets.len().is_multiple_of(64) && self.rets.len() >= 256 {
            let slice: Vec<f64> = self.rets.iter().copied().collect();
            let sr = micro::spread::roll_serial_dependent(&slice);
            self.spread_est_log = sr.spread;
        }
        // rough-vol cadence
        self.rough_counter += 1;
        if self.rough_counter.is_multiple_of(self.rough_every) {
            let rv = micro::rough::realized_variance_series(
                &self.rets.iter().copied().collect::<Vec<f64>>(),
                64,
            );
            for v in rv {
                self.vol_hist.push_back(v);
                if self.vol_hist.len() > 512 {
                    self.vol_hist.pop_front();
                }
            }
            if self.vol_hist.len() >= 128 {
                let proxy: Vec<f64> =
                    self.vol_hist.iter().map(|&v| (v.max(1e-18)).ln()).collect();
                let lags = [8usize, 16, 32, 64];
                let (h, _) = micro::rough::hurst_estimate(&proxy, &lags, 2.0, false);
                if h.is_finite() {
                    self.hurst = h;
                    // RFSV forecast of log-variance over the quoting horizon
                    let model = micro::rough::RfsvModel {
                        h: h.clamp(0.02, 0.48),
                        nu: 0.3,
                        alpha: 0.0,
                        m: proxy.last().copied().unwrap_or(0.0),
                    };
                    let hist: Vec<f64> = proxy.iter().rev().take(128).rev().copied().collect();
                    let f = model.forecast(&hist, self.dt, 64.0 * self.dt);
                    self.sigma_rough = (f / 2.0).exp().max(1e-6);
                }
            }
        }
    }

    /// Build the strategy-facing snapshot.
    pub fn snapshot(&self, book: &mut OrderBook, tick_size: f64) -> MarketState {
        let best_bid = book.best_bid();
        let best_ask = book.best_ask();
        let mid = match (best_bid, best_ask) {
            (Some((b, _)), Some((a, _))) => {
                (b + a) as f64 / 2.0 * tick_size
            }
            _ => self.last_mid.unwrap_or(tick_size),
        };
        let spread = match (best_bid, best_ask) {
            (Some((b, _)), Some((a, _))) => a.saturating_sub(b) as f64 * tick_size,
            _ => f64::INFINITY,
        };
        let imbalance = match (best_bid, best_ask) {
            (Some((_, bq)), Some((_, aq))) => bq as f64 / (bq + aq).max(1) as f64,
            _ => 0.5,
        };
        let micro_px = match (best_bid, best_ask) {
            (Some((_bp, bq)), Some((_ap, aq))) => {
                let i = bq as f64 / (bq + aq).max(1) as f64;
                self.mp.micro(mid, i)
            }
            _ => mid,
        };
        let clf_bid = {
            let l: Vec<(u64, u64)> = book.ladder(ob::Side::Bid, 5).collect();
            micro::clf::clf_score(&l, 5)
        };
        let clf_ask = {
            let l: Vec<(u64, u64)> = book.ladder(ob::Side::Ask, 5).collect();
            micro::clf::clf_score(&l, 5)
        };
        let dt = self.dt;
        MarketState {
            mid,
            best_bid,
            best_ask,
            spread,
            sigma_fast: (self.var_fast.max(1e-18) / dt).sqrt(),
            sigma_slow: (self.var_slow.max(1e-18) / dt).sqrt(),
            sigma_rough: self.sigma_rough,
            hurst: self.hurst,
            micro_price: micro_px,
            imbalance,
            ofi: self.ofi.ofi(),
            ofi_impact: self.ofi.impact_coef(),
            spread_est_log: self.spread_est_log,
            jump_flag: self.lm.last().map(|l| l.is_jump).unwrap_or(false),
            clf_bid,
            clf_ask,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use micro::Rng;

    #[test]
    fn estimators_track_sim() {
        let cfg = EngineConfig::default();
        let mut est = EstimatorStack::new(&cfg);
        let mut rng = Rng::new(3);
        let mut mid = cfg.s0;
        let sigma = 0.02;
        for _ in 0..3000 {
            mid *= (sigma * rng.normal() * cfg.dt.sqrt()).exp();
            est.update(mid, Some((199, 50)), Some((201, 50)), cfg.tick_size);
        }
        let mut book = OrderBook::new();
        book.apply(ob::BookEvent::NewOrder {
            id: 1,
            side: ob::Side::Bid,
            price_ticks: 199,
            lots: 50,
            ts_ns: 0,
        });
        book.apply(ob::BookEvent::NewOrder {
            id: 2,
            side: ob::Side::Ask,
            price_ticks: 201,
            lots: 50,
            ts_ns: 0,
        });
        let st = est.snapshot(&mut book, cfg.tick_size);
        // fast vol within 40% of true
        assert!(
            (st.sigma_fast - sigma).abs() / sigma < 0.4,
            "sigma_fast {} vs {}",
            st.sigma_fast,
            sigma
        );
        assert!((st.imbalance - 0.5).abs() < 1e-9);
        assert_eq!(st.best_bid, Some((199, 50)));
    }
}
