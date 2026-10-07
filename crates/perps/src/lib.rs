#![forbid(unsafe_code)]
//! # perps — perpetuals mechanics for the market-making engine
//!
//! * [`funding`] — BitMEX-style premium+interest funding exactly as the
//!   perp-options-clob venue computes it: `premium = clamp((mark_twap -
//!   index_twap)/index_twap, +/-5bps)`, `rate = clamp(interest_8h +
//!   premium, +/-75bps)`, longs pay shorts, TWAP ring buffers, exact
//!   `u128` payments with ceil-on-magnitude rounding.
//! * [`liquidation`] — **Donnelly–Lin–Lorig, "Optimal Liquidation of
//!   Perpetual Contracts" (arXiv:2601.10812)**: the closed-form optimal
//!   trading speed for the identity payoff (Theorem 2), the small-impact
//!   target-inventory rule (Proposition 3), the small-beta asymptotic
//!   strategy (Theorem 6, with the Almgren–Chriss base term), the
//!   short-time strategy (Theorem 8), and the substitution rule
//!   (Proposition 9) for arbitrary payoff `psi` (everlasting options).
//! * [`adl`] — **Chitra, "Autodeleveraging: Impossibilities and
//!   Optimization" (arXiv:2512.01112)**: the formal ADL policy framework —
//!   queue ranking (Binance bankruptcy-price / Hyperliquid mark-price
//!   reference conventions), pro-rata, levered pro-rata, capped
//!   water-filling pro-rata (Propositions 6.1/6.2), the trilemma metrics
//!   (PTSR/PMR/overshoot), and the severity/allocation separation
//!   principle.
//! * [`margin`] — isolated-margin accounting (equity, maintenance,
//!   liquidation trigger, deficit) used by the venue simulator.

pub mod adl;
pub mod funding;
pub mod liquidation;
pub mod margin;

pub use adl::{
    capped_pro_rata, levered_pro_rata, policy_metrics, pro_rata, queue_haircuts, AdlAccount,
    AdlMetrics,
};
pub use funding::{funding_payment_exact, funding_rate, TwapRing};
pub use liquidation::{LiqParams, Liquidator};
pub use margin::{isolated_account, IsolatedMargin};
