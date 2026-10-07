#![forbid(unsafe_code)]
//! # models — the quoting mathematics
//!
//! * [`unified`] — **Avellaneda–Stoikov and Cartea–Jaimungal as One
//!   Framework** (arXiv:2606.01477, Feys 2026): the axiomatic core. Five
//!   axioms on the market maker's dynamic preference functional
//!   (cash-additivity, normalization, concavity, strong dynamic
//!   consistency, law-invariance) *force* the objective to be the entropic
//!   certainty-equivalent on liquidation-adjusted terminal wealth —
//!   Theorem 10 (Forced Uniqueness) — with a single scalar `gamma`:
//!   `J_t = -(1/gamma) log E[exp(-gamma * W_T^L) | F_t]`.
//!   Consequences implemented:
//!   * Corollary 17 (single-parameter pinning): the Cartea–Jaimungal
//!     running inventory penalty is **forced** to `phi = gamma*sigma^2/2`
//!     (in the paper's convention; `gamma*sigma^2` in the doubled
//!     convention used by Cartea et al. 2015 — see
//!     [`unified::ForcedRelations`]), invertible as `gamma = 2*phi/sigma^2`
//!     (Corollary 23) — a consistency cross-check on independently
//!     calibrated desk parameters.
//!   * Corollary 22: the terminal liquidation penalty is forced to
//!     `alpha = L''(0)/2` — microstructure data, not preference.
//!   * Corollary 19: with `lambda(delta) = A*exp(-kappa*delta)` and
//!     `L = 0`, the optimal strategy is exactly Avellaneda–Stoikov (2008).
//!   * Corollary 20: CJ at `phi = gamma*sigma^2/2` is the second-order
//!     expansion of the entropic functional and produces the identical
//!     reservation price and half-spread.
//!   * Proposition 37 (stochastic volatility): the running inventory cost
//!     is `(gamma/2) q^2 d<S>_t` — quadratic-variation clock, `gamma`
//!     constant in business time (Corollary 12).
//!   * Proposition 50 (multi-asset): `Phi_ij = gamma * Sigma_ij / 2`.
//! * [`hjb`] — exact numerical solution of the inventory market-making
//!   HJB in the reduced `v` (psi) form, incl. optional adverse-selection
//!   impact, plus the closed-form AS quotes used as validation anchors.
//! * [`glft`] — Guéant–Lehalle–Fernández-Tapia (2013) asymptotic quotes,
//!   validated against the exact solver at long horizons.
//! * [`queue`] — Cont–de-Larrard-style execution probabilities for an
//!   order at a given queue position (exact truncated birth-death solve,
//!   Monte-Carlo validated), used by the queue-aware quoting strategy.
//! * [`options`] — BSM / Black-76 pricing + greeks, implied volatility,
//!   Barone-Adesi–Whaley American approximation, and the everlasting
//!   effective-maturity convention used by the venue.

pub mod glft;
pub mod multilevel;
pub mod hjb;
pub mod options;
pub mod queue;
pub mod unified;

pub use hjb::{AsQuotes, MmHjb, MmProblem};
pub use unified::{entropic_ce, ForcedRelations, MultiAsset, UnifiedParams};
