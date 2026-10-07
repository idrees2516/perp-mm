# RESEARCH.md — paper-to-code map

Every module in this workspace traces to a specific paper (or a
self-contained derivation validated by Monte Carlo). Papers are grouped
by the crate implementing them.

## models — the quoting core

### Avellaneda-Stoikov and Cartea-Jaimungal as One Framework: A Forced Uniqueness Theorem for Inventory Market Making (arXiv:2606.01477v1, Feys 2026)

- **Theorem 10 (Forced Uniqueness)** — five axioms (cash-additivity,
  normalization, concavity, strong dynamic consistency, law-invariance)
  force the objective to be the entropic certainty-equivalent on
  liquidation-adjusted terminal wealth with a single scalar `gamma`.
  → `models::unified::entropic_ce` (used as the MC validation objective).
- **Corollary 17 (single-parameter pinning)** `phi = gamma*sigma^2/2`,
  **Corollary 23 (calibration inversion)** `gamma = 2*phi/sigma^2`,
  **Corollary 22** `alpha = L''(0)/2` (with the Cartea et al. 2015
  double-convention warning `phi_book = gamma*sigma^2`)
  → `models::unified::ForcedRelations`.
- **Corollaries 19-20 (AS quotes / CJ equivalence at the forced phi)**:
  `r = s - gamma*sigma^2*q*(T-t)`,
  `half = (gamma*sigma^2/2)(T-t) + (1/gamma) ln(1+gamma/kappa)`
  → `models::unified::UnifiedParams::{reservation, half_spread, quotes}`
  and the engine's `UnifiedAs` strategy.
- **Proposition 33 (the HJB / psi-system)** with impulse operators
  `I^a[psi] = sup_delta lambda(delta)[e^{-gamma delta} psi(q-1) - psi(q)]`
  → `models::hjb` (full re-derivation, sign-verified three ways: the
  terminal FOC reduces exactly to the AS half-spread; the no-quoting
  limit integrates to `v_q = exp(gamma^2 sigma^2 tau q^2 / 2)`; and the
  Monte-Carlo entropic-CE optimality test).
- **Proposition 37 (stochastic volatility)**: running cost
  `(gamma/2) q^2 d<S>_t` (quadratic-variation clock, gamma constant)
  → `ForcedRelations::running_inventory_cost` + the estimator stack's
  rough-vol sigma feeding `UnifiedAs`/`HjbPolicy`.
- **Proposition 50 (multi-asset)**: `Phi_ij = gamma*Sigma_ij/2`
  → `models::unified::MultiAsset`.

### Guéant-Lehalle-Fernández-Tapia, "Dealing with the inventory risk" (2013, arXiv:1105.3115)

- The infinite-horizon asymptotic quotes
  `delta_b^inf(q) ≈ (1/gamma)ln(1+gamma/kappa) + (2q+1)/2 * sqrt(...)`
  (survey form arXiv:1605.01862 §4)
  → `models::glft::GlftAsymptotic`, structurally validated against the
  exact solver at long horizons.

### Cont & de Larrard (2013), "Price dynamics in a Markovian limit order market"

- The queue-race execution layer. Our implementation
  (`models::queue`) documents and corrects a subtle degeneracy: when
  refills join *behind* our order (they cannot delay our turn), the
  "queue empties before we fill" race is degenerate (probability 1 —
  confirmed by Monte Carlo). The non-degenerate races are:
  `P(fill within h) = Poisson tail` and
  `P(fill before an exogenous away-move clock) = (mu/(mu+nu))^m`
  (self-derived via the Erlang Laplace transform, MC-validated).

## micro — microstructure analytics

### Lee & Mykland (2008), "Jumps in financial markets: a new nonparametric test and jump dynamics" (RFS 21(6))

- Rolling bipower volatility `V(i) = (pi/2)/(K-1) sum |r_{j-1} r_j|`,
  statistic `L(i) = |r_i|/sqrt(V(i))`, Gumbel normalization
  `C(M), S(M)` → `micro::jump::LeeMykland` (power/size MC-validated).

### Barndorff-Nielsen & Shephard (2004-2006) — realized/bipower jump tests

- `RV`, `BV = (pi/2) sum |r_{j-1} r_j|`, standardized difference with
  the BNS constant `theta = pi^2/4 + pi - 5` — we re-derive the variance
  from scratch (`Var(BV-RV) = theta n s^4`, recovering theta exactly)
  and jump-robustify via `s^2 ~ BV/n` → `micro::jump::bns_ratio_test`.

### Bibinger-Hautsch-Ristig, "Jump detection in high-frequency order prices" (arXiv:2403.00819)

- Block-minima statistics for one-sided-noise order prices: min-based
  spot variance `pi/(2(pi-2)K) sum dm^2/h`, global statistic
  `n^{1/3} T - B_n` vs Gumbel, block-size rate `h_n ~ 2 ln(2/h-2) n^{-2/3}`
  → `micro::jump::BhrJumpTest` (location/size recovery MC-validated).

### Roll (1984); serial dependence corrections

- Classic `s = 2 sqrt(-cov1)` → `micro::spread::roll_classic`.
- AR(1) order-flow correction, re-derived in closed form:
  `E[Delta q_t Delta q_{t-1}] = -(1-rho)^2`,
  `E[Delta q_t Delta q_{t-2}] = -rho(1-rho)^2` ⟹
  `rho = cov2/cov1`, `s = 2 sqrt(-cov1)/(1-rho)`; classic Roll is biased
  low by the factor `(1-rho)` → `micro::spread::roll_serial_dependent`
  (both biases verified by simulation).

### Corwin-Schultz (2012), high-low spread estimator

- Our implementation solves the CS moment conditions in **exact closed
  form** (a quadratic in `z = ln(1+S/2)`) rather than the published
  approximation, which MC shows is severely upward-biased in
  low-vol/high-spread regimes → `micro::spread::corwin_schultz`.

### Brouty-Garcin-Roccaro (arXiv:2407.17401v3), "Estimation of bid-ask spreads in the presence of serial dependence"

- The variance-ratio family `V(L) = L tau sigma^2 + S^2/2` (fBm mid,
  autocorrelated noise): `S2_1`, `S2_2(H)`, `S2_3(rho)` with the
  plug-in `rho^L` from `(2V(2L)-V(4L))/(2V(L)-V(2L)) = (1+rho^L)^2`,
  the Hurst variance-ratio estimator, and the joint 4-parameter fit
  → `micro::spread::{VarianceLags, BroutySpread}` (all MC-recovered).

### Stoikov (2018), "The Micro-Price"

- Both constructions: the imbalance-polynomial fit
  `E[Delta mid | I]` via RLS (symmetrized), and the full discrete-state
  Markov chain `G1 = (I-Q)^{-1} R K`, `B = (I-Q)^{-1} T`,
  `G* = G1 + sum B^i G1` → `micro::microprice::{MicroPriceFit, MicroPriceChain}`.

### Cont-Kukanov-Stoikov (2014, arXiv:1011.6402), "The price impact of order book events"

- The exact per-event OFI contribution
  `e_n = 1{Pb_n >= Pb_{n-1}} qb_n - 1{Pb_n <= Pb_{n-1}} qb_{n-1} - ...`
  and the stylized model `Delta P = OFI/(2D)` → `micro::ofi::OfiTracker`
  (impact coefficient fit by RLS; used by the `micro` strategy).

### Gatheral-Jaisson-Rosenbaum, "Volatility is rough" (arXiv:1410.3394)

- fBm simulation by Davies-Harte circulant embedding (hand-rolled
  radix-2 FFT), the `m(q,Delta)` scaling Hurst estimator, the RFSV
  model, and Gaussian-conditional forecasting
  → `micro::rough` (Hurst recovery + forecast-beats-climatology tests).

### Composite Liquidity Factor — "High Frequency Quoting Under Liquidity Constraints" (arXiv:2507.05749v1)

- `CLF^b_i = log(p^b_1/p^b_{i+1}) / log(cum q)` (and the ask-side
  mirror) → `micro::clf::clf_score`; the same paper's reference-leg
  selection rule `choose argmin CLF` → `micro::clf::choose_reference_leg`.

## perps — perpetuals mechanics

### Donnelly-Lin-Lorig, "Optimal Liquidation of Perpetual Contracts" (arXiv:2601.10812v1)

- Dynamics: `dQ = nu dt`, `dP = b nu dt + eta dW^P`,
  `P_hat = P + k nu`, funding `beta(P - psi(S))`,
  `dX = -[P_hat nu + beta Q (P - psi(S))] dt` → `perps::liquidation`.
- **Theorem 2** closed form
  `nu* = (1/4k)[(xi+pi)q + (xi-pi)/b (p-s)]` with `a = 2 sqrt(k(b beta+phi))`,
  `C`, `omega = a/2k`, `xi(t)`, `pi(t)` → `Liquidator::closed_form`
  (a double-exponential transcription bug in `xi/pi` was caught and
  fixed by hand-computation + the sign test).
- **Proposition 3** target inventory `Q* = -beta Z/(b beta + 2 phi)` →
  `Liquidator::prop3_target`.
- **Theorem 8** short-time strategy → `Liquidator::small_time`.
- **Proposition 9** substitution rule (`psi(s)` in place of `s`) →
  `closed_form(.., s_override)`.
- Theorem 6's `nu_1` correction requires the paper's eq. (32) `gamma_2`
  which our extraction did not transcribe; the identity-payoff exact
  form covers that regime and the substitution rule is the paper's own
  production shortcut (documented in the module).
- MC optimality: exact > AC under funding; exact = AC at beta=0;
  AC > TWAP in the inventory-risk regime.

### Chitra, "Autodeleveraging: Impossibilities and Optimization" (arXiv:2512.01112v2)

- The policy formalism (severity theta, haircuts h, budget balance
  `sum h e+ = theta D`, feasibility) → `perps::adl`.
- Queue ranking (Binance bankruptcy-price / Hyperliquid mark reference;
  the venue's profit-ratio convention), running-residual haircuts
  (eq. 19), with the whole-lot rounding that produces the production
  **overshoot** pathology → `queue_haircuts`.
- Pro-rata (eq. 20), levered pro-rata (eq. 21) with clamped
  redistribution, **capped water-filling pro-rata** (Propositions
  6.1/6.2 — unique convex optimum, sybil-resistant, order-stable) →
  `pro_rata`, `levered_pro_rata`, `capped_pro_rata`.
- Trilemma metrics PTSR/PMR/overshoot (Propositions 5.1-5.3; the
  Proposition 5.3 identity `omega^PR - omega^Queue = H(1-e_(1)/W)` is
  verified numerically) → `policy_metrics`.

### BitMEX funding (as implemented by the venue)

- `premium = clamp((mark_twap - index_twap)/index_twap, ±5bps)`,
  `rate = clamp(interest + premium, ±75bps)`, longs pay shorts, TWAP
  rings, exact `u128` payments with ceil-on-magnitude →
  `perps::funding`.

## ob — data structures

### Krapivensky, "glass: ordered set data structure for client-side order books" (arXiv:2506.13991v1)

- Uncompressed 32-way trie over integer price keys with depth bounded by
  the key width (13 for 64-bit keys — independent of the number of
  levels), cached path with the XOR+clz lowest-common-ancestor jump,
  O(1) slot allocator, `adjust()` delta semantics
  → `ob::glass::Glass` (validated against a `BTreeMap` oracle over
  clustered random workloads; the paper measures 6-30x over std::map —
  see `bench` for this machine's numbers).

## feed — the wire layer

### "io_uring for High-Performance DBMSs: When and How to Use It" (arXiv:2512.04859v1)

- The recommended feed-handler pattern: **registered (fixed) buffers**,
  batched `io_uring_enter(IORING_ENTER_GETEVENTS)` completions, SQPOLL
  probing with graceful fallback to enter()-driven submission
  → `feed::uring` (raw syscalls via `extern "C"`, zero external crates;
  ABI verified byte-for-byte against `/usr/include/linux/io_uring.h`;
  the UDP loopback test passes on this kernel-5.10 machine with real
  completions drained through the ring).

## engine — the venue simulation

### "No Tick-Size Too Small" (arXiv:2410.08744v3, MQH model)

- The engine's simulated book dynamics follow the MQH structure in
  reduced form ("MQH-lite"): meta-queue birth/death at the touch (LO/CO
  rates), market-order sweeps, in-spread limit-order arrivals (20% of
  joins quote inside the spread), and self-repair of crossed books
  → `engine::venue::SimVenue::step`. The full 5-meta-queue/9-state/12-
  event Hawkes model is out of scope for the simulator (documented);
  the in-spread intensity factor `(delta(s-1)/alpha)^beta` and the
  stylized scaling laws are the basis of the reduced form.

### The venue's own semantics (perp-options-clob repo)

- Ticks/lots integer domain, price-time priority with maker-price
  fills, MMP-style rolling throttles, isolated margin + maintenance
  trigger, profit-ranked ADL, 8h funding — mirrored in `engine::venue`,
  `engine::risk`, `engine::adapter` (`PocCommand`/`PocEvent`).

## Papers researched but not implemented (out of the chosen scope)

- **COMMON Order Book with Privacy** (IACR 2023/1868) — MPC order book;
  out of scope per the clarified focus (quoting + microstructure).
- **Parallel Clearing Mechanisms in CLOBs** (arXiv:2509.01683) — batch
  clearing parallelism; the bench crate demonstrates the CU/pipeline
  discipline instead.
- **Limit Order Book Dynamics in Matching Markets** (arXiv:2511.20606v2)
  — matching-market microstructure; the theta/T execution gate informed
  the venue's sweep semantics.
- **Cyfrin audit of Deriverse DEX v2.0** — used as risk-design input
  (wash-trade/socialized-loss patterns the engine's risk gates defend
  against).

## vol — volatility surfaces & option market making (stage 2)

### Gatheral-Jacquier, "Arbitrage-Free SVI Volatility Surfaces" (2014, Quant Finance 14)
- **Implemented**: `vol::ssvi::SsviSurface` — `w(k,θ) = θ/2(1+ρφ(θ)k+√((φ(θ)k+ρ)²+1−ρ²))`
  with Heston-like `φ(θ) = 1/(ηθ^γ(1+θ)^{1−γ})`.
- Butterfly condition `θφ(θ)²(1+|ρ|) ≤ 4` (Thm 4.2) enforced + the direct
  density `g(k) = (1−kw'/2w)² − w'²/4 + w''/2 ≥ 0` grid check; calendar
  verified by `w(k,T₂) ≥ w(k,T₁)` on grids across pillar pairs.
- **Validation**: `gk_condition_implies_positive_density` (the analytic
  condition ⟹ numerical density ≥ 0 on 60 random surfaces),
  `breeden_litzenberger_density_positive_and_integrates` (call-price
  densities integrate to 1), and `static_arb_free_bsm` audits
  (monotone/convex/bounded calls). Raw SVI slice + damped Gauss-Newton
  fit in `vol::svi` with the same ground-truth checks.

### Mingone, "No arbitrage global parametrization for the eSSVI volatility surface" (2022)
- **Reference for** the global (all-pillar) no-arb parametrization our
  monotone-θ Heston-like φ family instantiates.

### Zhang, "Risk-Sensitive Option Market Making with Arbitrage-Free eSSVI Surfaces" (arXiv:2510.04569, 2025)
- **Implemented (deterministic core)**: option MM as a constrained,
  risk-sensitive control problem: differentiable eSSVI surface → our
  SSVI layer with no-arb audits; half-spreads and hedge intensity as
  the controls → our GLFT vol quotes + WW hedge band; the paper's RL
  loop is replaced by the closed-form GLFT/WW policies and validated by
  Monte Carlo instead.

### Bergault & Guéant, "Algorithmic Market Making for Options" / "Algorithmic market making: the case of equity derivatives"
- **Implemented**: `vol::optmm::OptMm` — the option book collapses to its
  vega; client requests arrive at `a_v·e^{−κ_v·δ}` on the vol spread;
  the quoting problem becomes GLFT in vol space with
  `σ_eff = vega_per_fill · vol_of_vol`; per-fill hedge cost (the dealer
  formula `fill_cost + GLFT markup + skew`) is priced into both quotes.
- **Validation**: `glft_vega_quotes_beat_fixed_spread_mc` (entropic CE
  across 24 paths), mirror-symmetry of the skew, and the engine-level
  Study B.

### Whalley-Wilmott no-trade band (re-derived)
- `H* = (3c/(γS²))^{1/3}` from minimizing the steady-state
  loss rate `γS²σ²H²/6 + cσ²/H` of reflected-band delta hedging
  (`vol::optmm::HedgeCadence`).
- **Validation**: `hedge_band_mc_hump` — the simulated loss-rate is
  minimized at H* (hump test over ±2.2×); engine Study B shows the band
  adds +3.2 equity vs hedging every step.

### Castagna & Mercurio, "The Vanna-Volga Method for Implied Volatilities" (2007); Wystup, "Vanna-Volga and Market Data" (2009)
- The second-order layer the vega-approximation ignores: each option
  fill adds vanna (∂²V/∂S∂σ) and volga (∂²V/∂σ²) exposure. The classic
  three-instrument hedge portfolio — ATM + 25Δ risk reversal + 25Δ
  butterfly — solves the full 3×3 system in (vega, vanna, volga) by
  Cramer's rule (`vol::volga::VannaVolga::overhedge`,
  `solve_3x3`). Under this system the ATM prices itself exactly.
- **Overhedge charge** = λ·(w_rr·ΔV_rr + w_bf·ΔV_bf − ΔV_X): the
  vanna–volga replication cost minus the target's own smile premium —
  zero when the surface is vv-consistent, a few ticks otherwise —
  translated to vol units through the option's own vega and clamped to
  ±0.15 vol points (the market-standard overhedge is a second-order
  correction, never a first-order smile move). λ is the
  survival-probability damping, exponential in T.
- Applied in the quoting stack as `OptMm::quote_ivs_vv` (fair IV →
  overhedge shift → GLFT vega distances) — the same stack as the web
  terminal's VolSurface strategy.
- **Validation**: `weights_reproduce_target_greeks` (the 3×3 hedge is
  exact to machine precision on a strike×kind grid),
  `atm_instrument_prices_itself_exactly` (zero ATM overhedge by
  construction), `wing_strikes_bracket_the_atm` + RR sign for the
  equity skew, `overhedge_is_clamped_and_damped` (clamp bounds +
  monotone damping), `portfolio_charge_scale_sanity` (zero target →
  zero charge; pure-ATM book → ~zero charge),
  `strike_from_delta_round_trips` (the 25Δ strikes invert exactly).

### Jäckel, "Let's Be Rational" (2014) — reference for the IV problem
- **Implemented** as the classical robust hybrid: Newton with analytic
  vega + step clamping into a bracket + guaranteed bisection fallback
  (`vol::solver`); arbitrage-bound rejection (discounted intrinsic /
  discounted underlying).
- **Validation**: machine-precision price round-trips over 1,800
  parameter combinations; CRR-American inversion; cross-check against
  `models::options::bsm` within ncdf tolerance.

## models / engine — multi-level quoting & flow adaptation (stage 2)

### Barzykin-Bergault-Guéant, "Market making by an FX dealer: tiers, pricing ladders and hedging rates" (arXiv:2112.02269)
- **Implemented**: `models::multilevel::LadderPolicy` — size-tiered
  quote ladders (distance multipliers + growing tier sizes) around the
  GLFT base; per-level net edge
  `s_i·d_i·(1−α·m_i) − γσ²·s_i²·hold/2` prices the fill-probability /
  post-fill-returns trade-off; `optimal_levels` searches the level count.
- **Validation**: clean regime ladders / toxic regime collapses to a
  single level; engine Study A (ladder beats hjb & static under stress).

### "The Market Maker's Dilemma: Navigating the Fill Probability vs. Post-Fill Returns Trade-Off" (arXiv, Nov 2025)
- **Cited & modelled** as the `adverse_alpha` tier toxicity above: deeper
  quotes fill predominantly on flow that is more adversely selected.

### Le, "Funding-Aware Optimal Market Making for Perpetual DEXs" (arXiv:2605.06405, 2026)
- **Implemented** as the funding carry skew: positive funding makes long
  inventory costly, shifting both quotes down by
  `rate·(τ/interval)·mid·sign(q)` (ladder & volmm strategies).

### Markouts / adverse-selection adaptation (desk practice; cf. "Optimal
Quoting under Adverse Selection and Price Reading", 2026; flow-toxicity
segmentation literature)
- **Implemented**: `engine::markout::MarkoutTracker` — per-fill markouts
  at a horizon, EWMA of markout-to-captured-edge, spread multiplier
  `1 + θ·toxicity` (capped), normalization by the fill's own distance
  from the mid.
- **Validation**: engine Study C — under toxic fills (adverse 1.2
  ticks), adaptive widening improves mean equity +71% and the
  controller provably converges: `mult² − mult − θ·A/base = 0` ⟹
  `mult·base ≈ A + base` ≈ the impact-aware HJB optimum `A + (1/γ)ln(1+γ/κ)`.

## Notable bugs caught by the stage-2 validation methodology
- **ob::l3 LevelHead derived Default** produced `head = tail = 0`
  instead of `NIL = u32::MAX`: every fresh level's queue was silently
  linked to slab handle 0 (a valid order), corrupting FIFO chains —
  latent until multi-level quoting stressed it; caught by the new
  `invariant_violation()` walk + regression test
  (`engine/tests/l3_invariant.rs`).
- **Option-delta unit mismatch**: option delta (underlying units) vs
  perp lots (`lot_size` units) made hedges 1000× too small; caught by
  Study B's impossible CE ordering.
- **take() fee missing lot_size** (1000× inflated taker fees on hedge
  sweeps) and **re-flattening the whole position** at every band trigger
  instead of trading the excess to the band edge; both caught by the
  P&L attribution breakdown (hedge_cost dominating everything).
- **Markout normalization** by the observed book spread instead of the
  fill's own captured edge made the toxicity estimator saturate; caught
  by Study C's adaptive-worse-than-static result, fixed so the
  controller converges to the impact-aware optimum.

---

# Stage 4 — risk-engine v2, protocol lanes, and the Φ bug (2026-10)

*Sources: Paradex/TradeParadigm RFQ integration coverage (OI ×2.6 to
$202M after the Paradigm RFQ engine went live), Derive V3 launch
coverage (RFQ options, sub-ms matching, SFPM portfolio margin), Deribit
listed conventions (SOMC, block trades), Albers et al. 2025 "The Market
Maker's Dilemma: Navigating the Fill Probability vs. Post-Fill Returns
Trade-Off", Le 2025/26 "Funding-Aware Optimal Market Making for
Perpetual DEXs", Zhang 2025 "Risk-Sensitive Option Market Making with
Arbitrage-Free eSSVI Surfaces", Jäckel 2015 "Let's Be Rational".*

## The bug that motivated stage 4

The web engine's risk check halted the desk when the RAW perp position
— which is the delta HEDGE leg — crossed a lot-count limit. A properly
delta-hedged options book (unhedged ≈ 2 lots, hedge ≈ 62) was treated as
a breach; the halt then froze quoting, the option market AND the
hedger, so the position could never work off: `risk-manager resume`
re-halted within one step and the desk spent up to 63% of its life
deadlocked (reproduced across 6 seeds). The fix is the real-desk
doctrine, now in both engines:

1. **Limits bind on the UNHEDGED combined delta** (hedge leg + option
   book), never on the raw hedge size; a gross hedge-leg cap remains as
   a sanity bound.
2. **Limit breaches gate directionally and self-recover** — soft band
   (75%): drop the risk-adding side, tighten + deepen the unwind side;
   hard breach: unwind-only quotes at the touch. Never both-sides-off:
   blocking risk-REDUCING trades freezes the exposure a risk engine
   exists to shed.
3. **Kill-switch (drawdown only) keeps risk-reducing paths live**: the
   hedge, marks and surface keep running through risk-off; firm RFQ
   quotes are pulled (Derive MMP / cancel-on-disconnect semantics).
4. **Margin utilization is the protocol-grade limit** (SFPM): soft 70%
   gates, hard 90% winds the option book down (only position-reducing
   fills served).

Post-fix: 40-seed scan shows zero limit-halt loops; the worst seed
(63% deadlocked) now quotes both sides 3000/3000 steps; 12-seed
sustainability at 0.62% halted (kill-switch cycles only).

## New engine capability (paper → implementation map)

| Source | Implementation | Validation |
|---|---|---|
| Paradigm/TradeParadigm-on-Paradex + Derive V3 RFQ | `engine::rfq` (Rust) + `src/lib/engine/rfq.ts` (web): multi-leg packages (verticals, butterflies, condors / 10 templates incl. straddles, risk reversals, calendars, boxes on the web side), firm quotes with TTL, ATOMIC all-or-nothing execution, combo margin offsets shared with the taker as tighter package spreads, single-leg executions anchor the governed vol surface (Paradex shape) | atomicity (all legs move or none), box ≫ straddle offset ordering, net-delta/vega/leg-limit refusals leave the book untouched, TTL expiry sweeps, wall-clock firm windows for the manual taker |
| Derive V3 SFPM / Paradex SCAN / Deribit SOMC | `engine::margin` + `src/lib/engine/margin.ts`: spot×vol scenario grid + time-decay scan, worst-loss ∨ short-option-minimum floor, 1.2× initial buffer, utilization-gated risk | hedged books cut spot scenarios, short straddles require margin, perp-only books price off the spot scan, flat books are free |
| Albers et al. 2025 (fill-prob vs post-fill trade-off) | per-side markout trackers: the spread response widens only the TOXIC side | wired into all 8 strategies on the web engine |
| Le 2025 (funding-aware MM) | funding stream: premium-index EWMA, BitMEX-shape rate with clamps, per-interval payments, reservation-price carry skew | rate bounded by the clamp, payments signed correctly, carry leans against the paying side |
| Zhang 2025 (eSSVI risk-sensitive control) | per-expiry vega buckets drive per-pillar GLFT quotes (term-structure-aware inventory) instead of blinding the whole chain with the aggregate | vegaPerExpiry feeds both the quoter and the RFQ package pricer |
| Jäckel 2015 (Let's Be Rational) | `vol::solver::implied_vol_fast` + `bsm.impliedVolFast`: OTM parity fold, normalized strike units, rational seed + Halley (Householder-2) iterations, relative-precision fallback to the bracketed solver | machine precision (|Δσ| < 1e-9) across the moneyness/maturity/vol grid wherever the premium carries digits (b > 1e-6); graceful degradation below the double-precision floor; ~2-5× fewer evaluations than Newton+bisection |
| Self-learned intensity (adaptive-desk lineage) | online Cox-intensity MLE: exposure integral at the running κ̂, method-of-moments for A, score climbing for κ, exponential forgetting, count-based shrinkage; calibrated on non-swept prints only (real desks exclude swept/odd-lot) | recovers (A, κ) on a synthetic pure-Cox process; tracks the venue's realized mixture process (Cox core + informed boosts) |

## The Φ bug (web engine)

The Hart/West ncdf port evaluated its rational polynomial at (z/2)²
instead of z² — every normal CDF value was wrong (ncdf(0.25) = 0.71 vs
0.5987 true). The engine had been *self-consistent* on a wrong Φ: all
premiums, greeks, SSVI no-arb checks and vanna-volga shifts were
computed with it, so nothing visibly broke until the IV-solver grid
test disagreed with the reference. Rebuilt from first principles:
erf Maclaurin (exact term recurrence) in the central region + the
Laplace continued fraction for the Mills ratio in the tail —
|ΔΦ| ≤ 8e-15 over ±8 against C-libm erf, ~211 ns/call. The Rust
`micro::special::erf` was already correct (its own reference tests
caught nothing because there was nothing to catch); only the TS port
was garbled.

## Desk calibration notes

- Web-engine capital raised 1000 → 2500: at the sim's book scale
  (gross short ~150-200 lots against ~2.5-5k equity) SOMC binds at
  60-85% utilization — the desk trades against its capital constraint
  exactly as a real SFPM desk does (risk events show margin-gated
  cycles with self-recovery).
- RFQ freeze binds at the hard margin band only (0.9): the soft band
  tightens, it does not freeze.
- Institutional arrival intensity: one package per ~40 s (web) / ~20 s
  (Rust sim clock), sizes 5-25× the retail request.
