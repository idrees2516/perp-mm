#![forbid(unsafe_code)]
//! # engine — the market-making engine
//!
//! Orchestrates everything the prior crates built:
//!
//! ```text
//! [SimVenue] --feed frames--> [FeedHandler/book] --> [EstimatorStack]
//!                                     |                     |
//!                                     v                     v
//!                        [RiskEngine] <--------------- [QuotingStrategy]
//!                                     |
//!                        orders (venue adapter) --> [SimVenue] --> fills
//!                                     |
//!                               [Metrics]
//! ```
//!
//! * [`venue`] — the simulated CLOB: MQH-lite top-of-book dynamics
//!   (meta-queue birth/death + in-spread events, following arXiv:
//!   2410.08744's structure in reduced form), GBM or RFSV rough-volatility
//!   mids, Cox-process fills at `lambda(delta)` with adverse-selection
//!   impact, maker/taker fees with the venue's rebate tiers, 8h funding
//!   accrual, isolated-margin liquidations and ADL events.
//! * [`estimator`] — the streaming stack: multi-scale EWMA volatility,
//!   Lee–Mykland jump flags, Roll + serial-dependence spread, micro-price
//!   (RLS), OFI with impact regression, imbalance, CLF liquidity factor,
//!   RFSV rough-vol forecast on a slow cadence.
//! * [`strategy`] — [`StaticSpread`], [`UnifiedAs`] (closed form),
//!   [`HjbPolicy`] (exact solver policy), [`QueueAwareLadder`] (fill
//!   probabilities per queue position), [`MicroPriceAdjusted`] (AS centered
//!   on the micro-price with OFI drift), [`OptionsMarketMaker`] (everlasting
//!   option series quoted around BSM with delta hedging through the perp).
//! * [`risk`] — pre-trade gates: inventory bounds, participation caps,
//!   quote-size clamps, tick rounding, post-only guard, throttle
//!   (MMP-style rolling |fill| + |net delta|), jump/vol halts, loss
//!   kill-switch.
//! * [`adapter`] — `VenueAdapter` with the sim adapter and the
//!   perp-options-clob command/event semantics (Place/Cancel/Amend/Batch,
//!   u128 money, ticks/lots, maker-price fills).
//! * [`metrics`] — PnL attribution (spread capture, inventory, adverse
//!   selection, fees, funding), Sharpe, drawdown, inventory stats, CSV.
//! * [`mm`] — the `MarketMaker` orchestration + backtest driver.

pub mod adapter;
pub mod config;
pub mod estimator;
pub mod gateway;
pub mod margin;
pub mod markout;
pub mod options_market;
pub mod metrics;
pub mod mm;
pub mod rfq;
pub mod risk;
pub mod strategy;
pub mod venue;

pub use config::EngineConfig;
pub use mm::{run_backtest, MarketMaker};
pub use strategy::{QuotingStrategy, StrategyKind};
