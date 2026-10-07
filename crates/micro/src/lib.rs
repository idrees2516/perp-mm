#![forbid(unsafe_code)]
//! # micro — market-microstructure analytics kernel
//!
//! Streaming estimators feeding the quoting engine, each tied to its
//! source paper (see `RESEARCH.md` at the workspace root for the full
//! paper-to-code map):
//!
//! * [`jump`] — real-time jump detection:
//!   - **Lee & Mykland (2008)** rolling bipower-volatility test with
//!     Gumbel critical values;
//!   - **BNS-style** realized/bipower ratio test (self-derived variance,
//!     Monte-Carlo size-validated);
//!   - **Bibinger–Hautsch–Ristig (arXiv:2403.00819)** block-minima test
//!     for *order prices* under one-sided microstructure noise.
//! * [`spread`] — bid-ask spread estimation:
//!   - **Roll (1984)** classic autocovariance estimator;
//!   - serial-dependence-corrected Roll (two-autocovariance GMM, derived
//!     in closed form for AR(1) order flow);
//!   - **Corwin–Schultz (2012)** high-low estimator;
//!   - **Brouty–Garcin–Roccaro (arXiv:2407.17401)** variance-ratio
//!     estimator family `S^2_{1..4}` robust to fBm mid-prices and
//!     autocorrelated trades, including their Hurst and noise-decay
//!     estimators.
//! * [`microprice`] — **Stoikov (2018)** micro-price: (a) polynomial
//!   imbalance fit via recursive least squares, (b) the full
//!   discrete-state Markov-chain construction `G* = G1 + sum B^i G1`.
//! * [`ofi`] — **Cont–Kukanov–Stoikov (2014)** order-flow imbalance with
//!   the exact per-event contribution formula and online impact
//!   regression (`dP = OFI / (2D)` stylized model).
//! * [`rough`] — **Gatheral–Jaisson–Rosenbaum "Volatility is rough"**:
//!   fractional Brownian motion simulation, the `m(q,Delta)` scaling
//!   Hurst estimator, the RFSV model, and Gaussian-conditional forecasts.
//! * [`clf`] — **Composite Liquidity Factor** (arXiv:2507.05749) used for
//!   reference-leg selection in multi-contract quoting.
//! * [`special`] — error function, normal/Gumbel quantiles (no external
//!   dependencies anywhere in this crate).
//! * [`sampling`] — deterministic RNG (xoshiro256++) + normals, shared
//!   with downstream crates for reproducible simulations.

pub mod clf;
pub mod jump;
pub mod microprice;
pub mod ofi;
pub mod rough;
pub mod sampling;
pub mod special;
pub mod spread;

pub use clf::clf_score;
pub use jump::{BhrJumpTest, BnsRatioTest, LeeMykland};
pub use microprice::{MicroPriceChain, MicroPriceFit};
pub use ofi::OfiTracker;
pub use rough::{fbm_path, hurst_estimate, RfsvModel};
pub use sampling::Rng;
pub use spread::{
    corwin_schultz, roll_classic, roll_serial_dependent, BroutySpread, VarianceLags,
};
