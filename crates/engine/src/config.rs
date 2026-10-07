//! Engine configuration (plain structs, no serialization dependency).

use crate::options_market::OptionMarketConfig;

/// Mid-price model for the simulated venue.
#[derive(Clone, Debug, PartialEq)]
pub enum MidModel {
    /// Arithmetic/GBM drift-free mid with constant sigma (per sqrt(second)).
    Gbm { sigma: f64 },
    /// RFSV rough volatility driving a log-normal mid (H, nu, base sigma).
    Rough { h: f64, nu: f64, sigma0: f64, refresh_every: usize },
}

impl Default for MidModel {
    fn default() -> Self {
        MidModel::Gbm { sigma: 0.02 }
    }
}

/// Venue fee model (maker rebate can be negative = rebate).
#[derive(Clone, Debug)]
pub struct Fees {
    /// taker fee in fractional (4.5bps = 0.00045)
    pub taker: f64,
    /// maker fee (top tier can be -0.5bp rebate = -0.00005)
    pub maker: f64,
}

impl Default for Fees {
    fn default() -> Self {
        Fees {
            taker: 0.00045,
            maker: -0.00005,
        }
    }
}

/// Full engine configuration.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Tick size in quote units.
    pub tick_size: f64,
    /// Lot size in base units.
    pub lot_size: f64,
    /// Starting mid.
    pub s0: f64,
    /// Mid model.
    pub mid: MidModel,
    /// Simulation step (seconds).
    pub dt: f64,
    /// Horizon (seconds).
    pub horizon: f64,
    /// Fill intensity scale A (per second at delta=0).
    pub intensity_a: f64,
    /// Fill intensity decay kappa.
    pub intensity_kappa: f64,
    /// Adverse-selection: mid jump against us on our own fill (ticks).
    pub adverse_ticks: f64,
    /// MQH-lite book event rates: limit-order arrivals per second per side
    /// (top + depth), cancellations, market orders.
    pub lo_rate: f64,
    pub co_rate: f64,
    pub mo_rate: f64,
    /// Initial depth at each of the first levels (lots).
    pub depth_lots: u64,
    /// Fees.
    pub fees: Fees,
    /// Funding interval (seconds) and rate per interval.
    pub funding_interval: f64,
    pub funding_rate: f64,
    /// Inventory bound (lots).
    pub max_inventory: i64,
    /// Desk capital base (quote units) for margin-utilization accounting —
    /// SFPM utilization is initial margin / (capital + realized PnL).
    pub desk_capital: f64,
    /// Risk aversion gamma (the unified framework's single scalar).
    pub gamma: f64,
    /// HJB solver time steps.
    pub hjb_steps: usize,
    /// Maximum ladder levels per side (multi-level quoting).
    pub levels: usize,
    /// Enable markout-driven adaptive spread widening.
    pub markout_adaptive: bool,
    /// Markout horizon (steps) and spread-multiplier sensitivity.
    pub markout_horizon_steps: usize,
    pub markout_theta: f64,
    /// The option market leg (SSVI surface, client flow, vega quotes).
    pub option_market: OptionMarketConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            tick_size: 0.5,
            lot_size: 0.001,
            s0: 100.0,
            mid: MidModel::Gbm { sigma: 0.02 },
            dt: 0.05,
            horizon: 6.5 * 3600.0,
            intensity_a: 1.2,
            intensity_kappa: 1.5,
            adverse_ticks: 0.8,
            lo_rate: 3.0,
            co_rate: 2.2,
            mo_rate: 0.8,
            depth_lots: 40,
            fees: Fees::default(),
            funding_interval: 8.0 * 3600.0,
            funding_rate: 0.0001,
            max_inventory: 20,
            desk_capital: 1000.0,
            gamma: 0.1,
            hjb_steps: 300,
            levels: 3,
            markout_adaptive: true,
            markout_horizon_steps: 40,
            markout_theta: 1.5,
            option_market: OptionMarketConfig::default(),
        }
    }
}

impl EngineConfig {
    /// Per-step volatility used by strategies (estimated live, but the
    /// config also exposes the true annualization-free sigma for sims).
    pub fn sigma_per_step(&self) -> f64 {
        match &self.mid {
            MidModel::Gbm { sigma } => *sigma * self.dt.sqrt(),
            MidModel::Rough { sigma0, .. } => *sigma0 * self.dt.sqrt(),
        }
    }
}
