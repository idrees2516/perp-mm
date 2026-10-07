//! Quoting strategies.

use crate::estimator::MarketState;
use crate::config::EngineConfig;
use crate::options_market::OptionsView;
use models::glft::GlftAsymptotic;
use models::hjb::{MmHjb, MmProblem};
use models::multilevel::LadderPolicy;
use models::unified::UnifiedParams;

/// A desired quote pair (tick prices + size in lots). `None` = no quote
/// on that side. Optional deeper ladder levels follow level 0.
#[derive(Clone, Debug, PartialEq)]
pub struct Quotes {
    pub bid: Option<(u64, u64)>,
    pub ask: Option<(u64, u64)>,
    /// Deeper bid levels `(price_ticks, lots)`, best-first.
    pub bid_levels: Vec<(u64, u64)>,
    /// Deeper ask levels `(price_ticks, lots)`, best-first.
    pub ask_levels: Vec<(u64, u64)>,
}

impl Quotes {
    pub fn none() -> Quotes {
        Quotes {
            bid: None,
            ask: None,
            bid_levels: Vec::new(),
            ask_levels: Vec::new(),
        }
    }

    /// Single-level pair.
    pub fn new(bid: Option<(u64, u64)>, ask: Option<(u64, u64)>) -> Quotes {
        Quotes {
            bid,
            ask,
            bid_levels: Vec::new(),
            ask_levels: Vec::new(),
        }
    }

    /// Full bid ladder including level 0.
    pub fn full_bid(&self) -> Vec<(u64, u64)> {
        let mut v = Vec::with_capacity(1 + self.bid_levels.len());
        if let Some(l0) = self.bid {
            v.push(l0);
        }
        v.extend_from_slice(&self.bid_levels);
        v
    }

    /// Full ask ladder including level 0.
    pub fn full_ask(&self) -> Vec<(u64, u64)> {
        let mut v = Vec::with_capacity(1 + self.ask_levels.len());
        if let Some(l0) = self.ask {
            v.push(l0);
        }
        v.extend_from_slice(&self.ask_levels);
        v
    }
}

/// Strategy-facing context.
pub struct QuoteCtx<'a> {
    pub state: &'a MarketState,
    pub inventory: i64,
    /// Remaining time to the quoting horizon (seconds).
    pub time_left: f64,
    /// Book top levels for queue-aware placement: (price_ticks, lots)
    /// best-first.
    pub bid_ladder: Vec<(u64, u64)>,
    pub ask_ladder: Vec<(u64, u64)>,
    /// Fee floor for the half-spread, in ticks (round-trip fee
    /// amortization: half of `taker + |maker|` in ticks).
    pub fee_floor_ticks: f64,
    /// Current funding rate per interval (signed; longs pay when > 0).
    pub funding_rate: f64,
    /// Funding interval seconds.
    pub funding_interval: f64,
    /// Markout-driven spread multiplier (1 = off / clean flow).
    pub markout_mult: f64,
    /// Option market view (Some when the option leg is enabled).
    pub options: Option<OptionsView>,
}

/// The strategy trait: state -> desired quotes.
pub trait QuotingStrategy {
    /// Human-readable name.
    fn name(&self) -> &'static str;
    /// Compute desired quotes.
    fn quotes(&mut self, ctx: &QuoteCtx) -> Quotes;
}

/// Which strategy to run (for config-driven selection).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrategyKind {
    Static,
    UnifiedAs,
    HjbPolicy,
    QueueAware,
    MicroPrice,
    OptionsMm,
    /// Multi-level ladder quoting (pricing ladders + toxicity-aware
    /// level count + fee floor + funding skew).
    MultiLevel,
    /// Vol-surface option market maker (SSVI + vega-approx GLFT quotes,
    /// perp legs quote around the combined delta).
    VolSurface,
}

impl StrategyKind {
    pub fn all() -> Vec<StrategyKind> {
        vec![
            StrategyKind::Static,
            StrategyKind::UnifiedAs,
            StrategyKind::HjbPolicy,
            StrategyKind::QueueAware,
            StrategyKind::MicroPrice,
            StrategyKind::OptionsMm,
            StrategyKind::MultiLevel,
            StrategyKind::VolSurface,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            StrategyKind::Static => "static",
            StrategyKind::UnifiedAs => "as",
            StrategyKind::HjbPolicy => "hjb",
            StrategyKind::QueueAware => "queue",
            StrategyKind::MicroPrice => "micro",
            StrategyKind::OptionsMm => "options",
            StrategyKind::MultiLevel => "ladder",
            StrategyKind::VolSurface => "volmm",
        }
    }
}

/// Build the strategy instance for a kind (solved once, reused).
pub fn build(kind: StrategyKind, cfg: &EngineConfig) -> Box<dyn QuotingStrategy> {
    match kind {
        StrategyKind::Static => Box::new(StaticSpread::new(cfg)),
        StrategyKind::UnifiedAs => Box::new(UnifiedAs::new(cfg)),
        StrategyKind::HjbPolicy => Box::new(HjbPolicy::new(cfg)),
        StrategyKind::QueueAware => Box::new(QueueAwareLadder::new(cfg)),
        StrategyKind::MicroPrice => Box::new(MicroPriceAdjusted::new(cfg)),
        StrategyKind::OptionsMm => Box::new(OptionsMarketMaker::new(cfg)),
        StrategyKind::MultiLevel => Box::new(MultiLevelLadder::new(cfg)),
        StrategyKind::VolSurface => Box::new(VolSurfaceMm::new(cfg)),
    }
}

// ---------------------------------------------------------------------------
// 1. Static spread benchmark
// ---------------------------------------------------------------------------

/// Fixed half-spread around the mid (the naive benchmark).
pub struct StaticSpread {
    tick: f64,
    half_ticks: i64,
    size: u64,
}

impl StaticSpread {
    pub fn new(cfg: &EngineConfig) -> StaticSpread {
        // half spread ~ 2 ticks
        StaticSpread {
            tick: cfg.tick_size,
            half_ticks: 2,
            size: 5, // lots
        }
    }
}

impl QuotingStrategy for StaticSpread {
    fn name(&self) -> &'static str {
        "static"
    }

    fn quotes(&mut self, ctx: &QuoteCtx) -> Quotes {
        let mid_tick = (ctx.state.mid / self.tick).round() as i64;
        let b = (mid_tick - self.half_ticks).max(1) as u64;
        let a = (mid_tick + self.half_ticks).max(1) as u64;
        Quotes {
            bid: Some((b, self.size)),
            ask: Some((a, self.size)),
            ..Quotes::none()
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Unified AS (closed form)
// ---------------------------------------------------------------------------

/// Avellaneda–Stoikov closed-form quotes via the unified framework
/// (Corollaries 19-20), with live sigma from the estimator stack.
pub struct UnifiedAs {
    tick: f64,
    size: u64,
    gamma: f64,
    kappa: f64,
    a: f64,
    horizon: f64,
}

impl UnifiedAs {
    pub fn new(cfg: &EngineConfig) -> UnifiedAs {
        UnifiedAs {
            tick: cfg.tick_size,
            size: 5, // lots
            gamma: cfg.gamma,
            kappa: cfg.intensity_kappa,
            a: cfg.intensity_a,
            horizon: 600.0, // rolling quoting horizon (seconds)
        }
    }
}

impl QuotingStrategy for UnifiedAs {
    fn name(&self) -> &'static str {
        "as"
    }

    fn quotes(&mut self, ctx: &QuoteCtx) -> Quotes {
        let sigma = ctx.state.sigma_slow.max(ctx.state.sigma_rough).max(1e-4);
        let p = UnifiedParams {
            gamma: self.gamma,
            sigma,
            kappa: self.kappa,
            a: self.a,
            t: self.horizon,
            kappa_liq: 0.0,
        };
        // rolling horizon: quote for the full remaining time (t = 0 in
        // the AS parameterization, T = horizon)
        let q = ctx.inventory as f64;
        let (bid, ask) = p.quotes(ctx.state.mid, q, 0.0);
        let mid_tick = (ctx.state.mid / self.tick).round() as i64;
        let bid_tick = ((bid / self.tick).round() as i64).max(1);
        let ask_tick = ((ask / self.tick).round() as i64).max(1);
        // sanity: keep quotes within +/- 20 ticks of the mid
        let bid_tick = bid_tick.clamp(mid_tick - 20, mid_tick - 1);
        let ask_tick = ask_tick.clamp(mid_tick + 1, mid_tick + 20);
        Quotes {
            bid: Some((bid_tick as u64, self.size)),
            ask: Some((ask_tick as u64, self.size)),
            ..Quotes::none()
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Exact HJB policy
// ---------------------------------------------------------------------------

/// The exact HJB solver's optimal quotes (models::hjb), with live sigma.
pub struct HjbPolicy {
    tick: f64,
    size: u64,
    solver: MmHjb,
    sigma_live: f64,
    sigma_solver: f64,
}

impl HjbPolicy {
    pub fn new(cfg: &EngineConfig) -> HjbPolicy {
        // solve with the config sigma; re-solve lazily when live sigma
        // drifts materially (sigma enters quadratically, so we track it)
        let sigma = match &cfg.mid {
            crate::config::MidModel::Gbm { sigma } => *sigma,
            crate::config::MidModel::Rough { sigma0, .. } => *sigma0,
        };
        // per-second sigma; the HJB problem uses sigma per sqrt(time) of
        // the *quoting clock* — we quote on a 600s horizon
        let mut prob = MmProblem::new(cfg.gamma, sigma, cfg.intensity_kappa, cfg.intensity_a, 600.0);
        prob.q_max = cfg.max_inventory;
        prob.n_steps = cfg.hjb_steps;
        let solver = MmHjb::solve(prob);
        HjbPolicy {
            tick: cfg.tick_size,
            size: 5, // lots
            solver,
            sigma_live: sigma,
            sigma_solver: sigma,
        }
    }
}

impl QuotingStrategy for HjbPolicy {
    fn name(&self) -> &'static str {
        "hjb"
    }

    fn quotes(&mut self, ctx: &QuoteCtx) -> Quotes {
        // track live sigma; re-solve if it moved > 50%
        let sigma = ctx.state.sigma_slow.max(ctx.state.sigma_rough).max(1e-4);
        self.sigma_live = sigma;
        if (self.sigma_live - self.sigma_solver).abs() / self.sigma_solver > 0.5 {
            let p = self.solver.problem().clone();
            let mut prob = MmProblem::new(
                p.gamma,
                self.sigma_live,
                p.kappa,
                p.a,
                p.t,
            );
            prob.q_max = p.q_max;
            prob.n_steps = p.n_steps.min(200); // fast re-solve
            self.solver = MmHjb::solve(prob);
            self.sigma_solver = self.sigma_live;
        }
        // The solver returns distances in quote units; convert to ticks
        let mid = ctx.state.mid;
        let mid_tick = (mid / self.tick).round() as i64;
        let q = ctx.inventory.clamp(-self.solver.problem().q_max, self.solver.problem().q_max);
        // t = 0 (start of the rolling horizon) — the conservative choice
        let da = self.solver.delta_ask(q, 0.0);
        let db = self.solver.delta_bid(q, 0.0);
        let bid_tick = if db.is_finite() {
            (mid_tick - (db / self.tick).round() as i64).max(1)
        } else {
            mid_tick - 1
        };
        let ask_tick = if da.is_finite() {
            (mid_tick + (da / self.tick).round() as i64).max(mid_tick + 1)
        } else {
            mid_tick + 1
        };
        let bid_tick = bid_tick.clamp(mid_tick - 20, mid_tick - 1);
        let ask_tick = ask_tick.clamp(mid_tick + 1, mid_tick + 20);
        Quotes {
            bid: Some((bid_tick as u64, self.size)),
            ask: Some((ask_tick as u64, self.size)),
            ..Quotes::none()
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Queue-aware ladder
// ---------------------------------------------------------------------------

/// Multi-level quoting with per-level fill probabilities
/// (models::queue): choose the level maximizing
/// `lambda_eff(delta) * edge`, blending the AS target spread.
pub struct QueueAwareLadder {
    tick: f64,
    size: u64,
    gamma: f64,
    kappa: f64,
    a: f64,
    horizon: f64,
    /// Depletion rate estimate (events/sec at the touch) — from the
    /// venue's MO + CO rates; live-tracked via the book feed in a full
    /// deployment, configured here.
    mu_eff: f64,
    away_rate: f64,
}

impl QueueAwareLadder {
    pub fn new(cfg: &EngineConfig) -> QueueAwareLadder {
        QueueAwareLadder {
            tick: cfg.tick_size,
            size: 5, // lots
            gamma: cfg.gamma,
            kappa: cfg.intensity_kappa,
            a: cfg.intensity_a,
            horizon: 600.0,
            mu_eff: cfg.mo_rate + cfg.co_rate,
            away_rate: cfg.lo_rate * 0.25,
        }
    }
}

impl QuotingStrategy for QueueAwareLadder {
    fn name(&self) -> &'static str {
        "queue"
    }

    fn quotes(&mut self, ctx: &QuoteCtx) -> Quotes {
        let sigma = ctx.state.sigma_slow.max(ctx.state.sigma_rough).max(1e-4);
        // AS target distances for the edge benchmark
        let p = UnifiedParams {
            gamma: self.gamma,
            sigma,
            kappa: self.kappa,
            a: self.a,
            t: self.horizon,
            kappa_liq: 0.0,
        };
        let half = p.half_spread(0.0);
        let skew = self.gamma * sigma * sigma * ctx.inventory as f64 * self.horizon;
        let target_bid_d = half + skew;
        let target_ask_d = half - skew;

        // candidate levels: 1..=4 ticks from the mid on each side
        let mid_tick = (ctx.state.mid / self.tick).round() as i64;
        let best = |ladder: &[(u64, u64)]| ladder.first().copied();
        let (bid_best, ask_best) = (best(&ctx.bid_ladder), best(&ctx.ask_ladder));

        let score_level = |ticks_out: i64, queue_ahead: u64| -> f64 {
            let delta = ticks_out as f64 * self.tick;
            let p_fill =
                models::queue::fill_probability_vs_clock(queue_ahead, self.mu_eff, self.away_rate);
            let edge = (delta - 0.0).max(0.0);
            self.a * (-self.kappa * delta / self.tick).exp() * p_fill * edge
        };

        let mut best_bid: Option<(u64, u64)> = None;
        let mut best_bid_score = -1.0f64;
        for ticks_out in 1..=4i64 {
            let price_tick = (mid_tick - ticks_out).max(1) as u64;
            // joining an existing level: queue ahead = its size
            let queue_ahead = ctx
                .bid_ladder
                .iter()
                .find(|&&(p, _)| p == price_tick)
                .map(|&(_, l)| l)
                .unwrap_or(0);
            let s = score_level(ticks_out, queue_ahead);
            // prefer levels near the AS target distance
            let d = ticks_out as f64 * self.tick;
            let target_bonus = 1.0 / (1.0 + (d - target_bid_d).abs().max(0.1));
            let s = s * target_bonus;
            if s > best_bid_score {
                best_bid_score = s;
                best_bid = Some((price_tick, self.size));
            }
        }
        let _ = bid_best;

        let mut best_ask: Option<(u64, u64)> = None;
        let mut best_ask_score = -1.0f64;
        for ticks_out in 1..=4i64 {
            let price_tick = (mid_tick + ticks_out).max(1) as u64;
            let queue_ahead = ctx
                .ask_ladder
                .iter()
                .find(|&&(p, _)| p == price_tick)
                .map(|&(_, l)| l)
                .unwrap_or(0);
            let s = score_level(ticks_out, queue_ahead);
            let d = ticks_out as f64 * self.tick;
            let target_bonus = 1.0 / (1.0 + (d - target_ask_d).abs().max(0.1));
            let s = s * target_bonus;
            if s > best_ask_score {
                best_ask_score = s;
                best_ask = Some((price_tick, self.size));
            }
        }
        let _ = ask_best;
        Quotes {
            bid: best_bid,
            ask: best_ask,
            ..Quotes::none()
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Micro-price adjusted AS
// ---------------------------------------------------------------------------

/// AS quotes centered on the **micro-price** with an OFI drift
/// adjustment (Stoikov 2018 + CKS 2014 layered on the unified framework).
pub struct MicroPriceAdjusted {
    tick: f64,
    size: u64,
    gamma: f64,
    kappa: f64,
    a: f64,
    horizon: f64,
}

impl MicroPriceAdjusted {
    pub fn new(cfg: &EngineConfig) -> MicroPriceAdjusted {
        MicroPriceAdjusted {
            tick: cfg.tick_size,
            size: 5, // lots
            gamma: cfg.gamma,
            kappa: cfg.intensity_kappa,
            a: cfg.intensity_a,
            horizon: 600.0,
        }
    }
}

impl QuotingStrategy for MicroPriceAdjusted {
    fn name(&self) -> &'static str {
        "micro"
    }

    fn quotes(&mut self, ctx: &QuoteCtx) -> Quotes {
        let sigma = ctx.state.sigma_slow.max(ctx.state.sigma_rough).max(1e-4);
        let p = UnifiedParams {
            gamma: self.gamma,
            sigma,
            kappa: self.kappa,
            a: self.a,
            t: self.horizon,
            kappa_liq: 0.0,
        };
        let half = p.half_spread(self.horizon);
        let skew = self.gamma * sigma * sigma * ctx.inventory as f64 * self.horizon;
        // center on the micro-price, plus the OFI drift forecast (ticks)
        let center = ctx.state.micro_price + ctx.state.ofi_impact * ctx.state.ofi * self.tick;
        let mid_tick = (ctx.state.mid / self.tick).round() as i64;
        let bid = center - half - skew;
        let ask = center + half - skew;
        let bid_tick = ((bid / self.tick).round() as i64).clamp(mid_tick - 20, mid_tick - 1);
        let ask_tick = ((ask / self.tick).round() as i64).clamp(mid_tick + 1, mid_tick + 20);
        Quotes {
            bid: Some((bid_tick.max(1) as u64, self.size)),
            ask: Some((ask_tick.max(1) as u64, self.size)),
            ..Quotes::none()
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Options market maker with delta hedging
// ---------------------------------------------------------------------------

/// Quotes an everlasting call option series around BSM (vol from the
/// rough-vol forecast) and passes the net delta into the perp quoting as
/// inventory adjustment (the perp legs run the AS strategy).
pub struct OptionsMarketMaker {
    tick: f64,
    size: u64,
    gamma: f64,
    kappa: f64,
    a: f64,
    horizon: f64,
    /// Option strike (quote units).
    pub strike: f64,
    /// Everlasting interval seconds and multiple (venue convention 1h x 24).
    pub interval_secs: f64,
    pub maturity_multiple: f64,
    /// Fixed IV override; None = use rough-vol forecast sigma.
    pub iv_override: Option<f64>,
    /// Option quote spread in premium units.
    pub opt_spread: f64,
    /// Delta hedge band (|net delta| in lots before rehedging).
    pub hedge_band: f64,
    // internal state
    last_option_quote: Option<(f64, f64)>, // (bid premium, ask premium)
    net_option_delta: f64,
}

impl OptionsMarketMaker {
    pub fn new(cfg: &EngineConfig) -> OptionsMarketMaker {
        OptionsMarketMaker {
            tick: cfg.tick_size,
            size: 5, // lots
            gamma: cfg.gamma,
            kappa: cfg.intensity_kappa,
            a: cfg.intensity_a,
            horizon: 600.0,
            strike: cfg.s0,
            interval_secs: 3600.0,
            maturity_multiple: 24.0,
            iv_override: None,
            opt_spread: 0.15,
            hedge_band: 2.0,
            last_option_quote: None,
            net_option_delta: 0.0,
        }
    }

    /// Current option premium pair (bid, ask) — surfaced to the venue
    /// adapter's option series.
    pub fn option_quote(&self, state: &MarketState) -> (f64, f64) {
        let sigma = self.iv_override.unwrap_or_else(|| {
            (state.sigma_rough * (365.0f64 * 24.0 * 3600.0).sqrt()).clamp(0.05, 2.0)
        });
        let t_eff =
            models::options::everlasting_t_eff(self.interval_secs, self.maturity_multiple);
        let mid = models::options::bsm(
            models::options::Kind::Call,
            state.mid,
            self.strike,
            0.0,
            0.0,
            sigma,
            t_eff,
        );
        (mid - self.opt_spread / 2.0, mid + self.opt_spread / 2.0)
    }

    /// Record a signed option fill (calls bought add +delta).
    pub fn record_option_fill(&mut self, lots_signed: f64, state: &MarketState) {
        let sigma = self.iv_override.unwrap_or_else(|| {
            (state.sigma_rough * (365.0f64 * 24.0 * 3600.0).sqrt()).clamp(0.05, 2.0)
        });
        let t_eff =
            models::options::everlasting_t_eff(self.interval_secs, self.maturity_multiple);
        let g = models::options::bsm_greeks(
            models::options::Kind::Call,
            state.mid,
            self.strike,
            0.0,
            0.0,
            sigma,
            t_eff,
        );
        self.net_option_delta += lots_signed * g.delta;
    }

    /// Hedge delta in perp lots (rounded); None when inside the band.
    pub fn hedge_lots(&mut self) -> Option<i64> {
        if self.net_option_delta.abs() < self.hedge_band {
            return None;
        }
        let h = self.net_option_delta.round() as i64;
        self.net_option_delta -= h as f64;
        Some(-h) // short the option delta via the perp
    }
}

impl QuotingStrategy for OptionsMarketMaker {
    fn name(&self) -> &'static str {
        "options"
    }

    fn quotes(&mut self, ctx: &QuoteCtx) -> Quotes {
        // Track the option quote for the adapter.
        let (ob, oa) = self.option_quote(ctx.state);
        self.last_option_quote = Some((ob, oa));
        // Perp quoting: AS with the delta-hedge adjustment folded into
        // inventory (the hedge output is consumed by the engine loop).
        let hedge = self.hedge_lots().unwrap_or(0);
        let sigma = ctx.state.sigma_slow.max(ctx.state.sigma_rough).max(1e-4);
        let p = UnifiedParams {
            gamma: self.gamma,
            sigma,
            kappa: self.kappa,
            a: self.a,
            t: self.horizon,
            kappa_liq: 0.0,
        };
        let half = p.half_spread(0.0);
        let q_eff = ctx.inventory as f64 + hedge as f64;
        let skew = self.gamma * sigma * sigma * q_eff * self.horizon;
        let mid_tick = (ctx.state.mid / self.tick).round() as i64;
        let bid_tick =
            (((ctx.state.mid - half - skew) / self.tick).round() as i64).clamp(mid_tick - 20, mid_tick - 1);
        let ask_tick =
            (((ctx.state.mid + half - skew) / self.tick).round() as i64).clamp(mid_tick + 1, mid_tick + 20);
        Quotes {
            bid: Some((bid_tick.max(1) as u64, self.size)),
            ask: Some((ask_tick.max(1) as u64, self.size)),
            ..Quotes::none()
        }
    }
}


// ---------------------------------------------------------------------------
// 7. Multi-level ladder (pricing ladders + toxicity-aware level count)
// ---------------------------------------------------------------------------

/// Multi-level quoting: Barzykin–Bergault–Guéant tiers around the GLFT
/// base distance, with
/// - a fee floor (the half-spread must amortize the round-trip fee),
/// - the markout-driven spread multiplier,
/// - a funding-aware skew (Le, "Funding-Aware Optimal Market Making
///   for Perpetual DEXs", arXiv:2605.06405: positive funding makes long
///   inventory costly, shifting both quotes down),
/// - inventory taper of the displayed sizes.
pub struct MultiLevelLadder {
    tick: f64,
    base_size: f64,
    gamma: f64,
    kappa: f64,
    a: f64,
    horizon: f64,
    policy: LadderPolicy,
    q_max: i64,
}

impl MultiLevelLadder {
    pub fn new(cfg: &EngineConfig) -> MultiLevelLadder {
        MultiLevelLadder {
            tick: cfg.tick_size,
            base_size: 5.0,
            gamma: cfg.gamma,
            kappa: cfg.intensity_kappa,
            a: cfg.intensity_a,
            horizon: 600.0,
            policy: LadderPolicy {
                levels: cfg.levels.max(1),
                ..LadderPolicy::default()
            },
            q_max: cfg.max_inventory,
        }
    }
}

impl QuotingStrategy for MultiLevelLadder {
    fn name(&self) -> &'static str {
        "ladder"
    }

    fn quotes(&mut self, ctx: &QuoteCtx) -> Quotes {
        let sigma = ctx
            .state
            .sigma_slow
            .max(ctx.state.sigma_rough)
            .max(1e-4);
        let glft = GlftAsymptotic::new(self.gamma, sigma, self.kappa, self.a);
        let (db_tiers, da_tiers) = self.policy.distances(&glft, ctx.inventory);
        // fee floor + markout multiplier on the level-0 distance
        let fee_floor = ctx.fee_floor_ticks.max(1.0) * self.tick;
        let mult = ctx.markout_mult.max(0.5);
        // funding-aware skew: positive funding + long inventory shifts
        // both quotes down by the expected carry over the quoting horizon
        let carry = ctx.funding_rate
            * (self.horizon / ctx.funding_interval.max(1.0)).min(1.0)
            * ctx.state.mid;
        let fund_skew = carry * ctx.inventory.signum() as f64;
        let mid = ctx.state.mid;
        let mid_tick = (mid / self.tick).round() as i64;
        let mut out = Quotes::none();
        // level 0
        let half_b = (db_tiers[0] * mult).max(fee_floor);
        let half_a = (da_tiers[0] * mult).max(fee_floor);
        let bid_tick = (((mid - half_b - fund_skew) / self.tick).round() as i64)
            .clamp(mid_tick - 20, mid_tick - 1);
        let ask_tick = (((mid + half_a - fund_skew) / self.tick).round() as i64)
            .clamp(mid_tick + 1, mid_tick + 20);
        let taper_b = self.policy.taper(ctx.inventory, self.q_max, true);
        let taper_a = self.policy.taper(ctx.inventory, self.q_max, false);
        let s0_b = ((self.base_size * taper_b).round() as u64).max(1);
        let s0_a = ((self.base_size * taper_a).round() as u64).max(1);
        if taper_b > 0.0 {
            out.bid = Some((bid_tick.max(1) as u64, s0_b));
        }
        if taper_a > 0.0 {
            out.ask = Some((ask_tick.max(1) as u64, s0_a));
        }
        // deeper levels: distances already carry the tier multiplier
        let (_, tiers_s) = self.policy.tiers();
        for i in 1..db_tiers.len() {
            let d_b = (db_tiers[i] * mult).max(fee_floor);
            let d_a = (da_tiers[i] * mult).max(fee_floor);
            let sz_b = ((tiers_s[i] * taper_b).round() as u64).max(1);
            let sz_a = ((tiers_s[i] * taper_a).round() as u64).max(1);
            let bp = (((mid - d_b - fund_skew) / self.tick).round() as i64)
                .clamp(mid_tick - 40, mid_tick - 1);
            let ap = (((mid + d_a - fund_skew) / self.tick).round() as i64)
                .clamp(mid_tick + 1, mid_tick + 40);
            if taper_b > 0.0 && bp < bid_tick {
                out.bid_levels.push((bp.max(1) as u64, sz_b));
            }
            if taper_a > 0.0 && ap > ask_tick {
                out.ask_levels.push((ap.max(1) as u64, sz_a));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// 8. Vol-surface option market maker
// ---------------------------------------------------------------------------

/// Option market making with an arbitrage-free SSVI surface: the option
/// quotes are produced by the engine's option market (vega-approximation
/// GLFT quotes); this strategy runs the PERP legs, quoting around the
/// COMBINED delta (perp inventory + option book delta) so passive perp
/// flow pays for the hedge, with the AS skew, fee floor, markout
/// multiplier and funding-aware carry.
pub struct VolSurfaceMm {
    tick: f64,
    size: u64,
    gamma: f64,
    kappa: f64,
    a: f64,
    horizon: f64,
}

impl VolSurfaceMm {
    pub fn new(cfg: &EngineConfig) -> VolSurfaceMm {
        VolSurfaceMm {
            tick: cfg.tick_size,
            size: 5,
            gamma: cfg.gamma,
            kappa: cfg.intensity_kappa,
            a: cfg.intensity_a,
            horizon: 600.0,
        }
    }
}

impl QuotingStrategy for VolSurfaceMm {
    fn name(&self) -> &'static str {
        "volmm"
    }

    fn quotes(&mut self, ctx: &QuoteCtx) -> Quotes {
        let sigma = ctx
            .state
            .sigma_slow
            .max(ctx.state.sigma_rough)
            .max(1e-4);
        // combined delta exposure: perp inventory + option book delta
        let opt_delta = ctx.options.map(|o| o.net_delta_lots).unwrap_or(0.0);
        let q_eff = ctx.inventory as f64 + opt_delta;
        let p = UnifiedParams {
            gamma: self.gamma,
            sigma,
            kappa: self.kappa,
            a: self.a,
            t: self.horizon,
            kappa_liq: 0.0,
        };
        let half = (p.half_spread(self.horizon) * ctx.markout_mult.max(0.5))
            .max(ctx.fee_floor_ticks.max(1.0) * self.tick);
        let skew = self.gamma * sigma * sigma * q_eff * self.horizon;
        // funding carry against the combined delta
        let carry = ctx.funding_rate
            * (self.horizon / ctx.funding_interval.max(1.0)).min(1.0)
            * ctx.state.mid
            * q_eff.signum();
        let mid = ctx.state.mid;
        let mid_tick = (mid / self.tick).round() as i64;
        let bid_tick = (((mid - half - skew - carry) / self.tick).round() as i64)
            .clamp(mid_tick - 20, mid_tick - 1);
        let ask_tick = (((mid + half - skew - carry) / self.tick).round() as i64)
            .clamp(mid_tick + 1, mid_tick + 20);
        Quotes::new(
            Some((bid_tick.max(1) as u64, self.size)),
            Some((ask_tick.max(1) as u64, self.size)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EngineConfig;

    fn ctx_for(state: &MarketState) -> QuoteCtx<'_> {
        QuoteCtx {
            state,
            inventory: 0,
            time_left: 600.0,
            bid_ladder: vec![(99, 40), (98, 30)],
            ask_ladder: vec![(101, 40), (102, 30)],
            fee_floor_ticks: 1.0,
            funding_rate: 0.0,
            funding_interval: 8.0 * 3600.0,
            markout_mult: 1.0,
            options: None,
        }
    }

    fn sample_state() -> MarketState {
        MarketState {
            mid: 100.0,
            best_bid: Some((99, 40)),
            best_ask: Some((101, 40)),
            spread: 1.0,
            sigma_fast: 0.02,
            sigma_slow: 0.02,
            sigma_rough: 0.02,
            hurst: 0.1,
            micro_price: 100.1,
            imbalance: 0.55,
            ofi: 5.0,
            ofi_impact: 0.001,
            spread_est_log: 0.001,
            jump_flag: false,
            clf_bid: Some(0.1),
            clf_ask: Some(0.12),
        }
    }

    #[test]
    fn all_strategies_quote_sensibly() {
        let cfg = EngineConfig::default();
        let state = sample_state();
        for kind in StrategyKind::all() {
            let mut s = build(kind, &cfg);
            let q = s.quotes(&ctx_for(&state));
            match (q.bid, q.ask) {
                (Some((b, bs)), Some((a, asz))) => {
                    assert!(b < a, "{}: bid {b} >= ask {a}", s.name());
                    assert!(bs > 0 && asz > 0);
                    // quotes within 20 ticks of the mid
                    assert!(b >= 180 && a <= 220, "{}: {b}/{a}", s.name());
                }
                _ => panic!("{}: missing quotes", s.name()),
            }
        }
    }

    #[test]
    fn as_skews_with_inventory() {
        let cfg = EngineConfig::default();
        let mut s = build(StrategyKind::UnifiedAs, &cfg);
        let state = sample_state();
        let q_flat = s.quotes(&ctx_for(&state));
        let mut ctx_long = ctx_for(&state);
        ctx_long.inventory = 10;
        let q_long = s.quotes(&ctx_long);
        // long inventory: both quotes move DOWN
        assert!(q_long.bid.unwrap().0 < q_flat.bid.unwrap().0);
        assert!(q_long.ask.unwrap().0 <= q_flat.ask.unwrap().0);
    }

    #[test]
    fn options_mm_quotes_and_hedges() {
        let cfg = EngineConfig::default();
        let mut s = OptionsMarketMaker::new(&cfg);
        let state = sample_state();
        let (ob, oa) = s.option_quote(&state);
        assert!(ob > 0.0 && oa > ob);
        // no fill: no hedge
        assert_eq!(s.hedge_lots(), None);
        // simulate a 10-lot option buy: delta ~0.5 -> hedge of -5
        s.record_option_fill(10.0, &state);
        assert_eq!(s.hedge_lots(), Some(-5));
        assert_eq!(s.hedge_lots(), None);
    }
}
