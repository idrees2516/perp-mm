// The market-making engine: 8 quoting strategies over the perp CLOB +
// option market, estimator stack, risk engine, markout-adaptive spreads,
// WW-band delta hedging, vanna–volga overhedge, CU metering and latency
// percentiles. Port of the Rust `engine` crate's hot loop, arranged for
// a browser simulation loop.
//
// Risk-parameter scaling (all three books share one γ* slider):
//  - perp AS/HJB quotes live in price space over a 600 s horizon:
//    γ_perp = γ* (skew ≈ γ*σ_H²·q — visible at a few lots);
//  - option GLFT quotes live in vega space: γ_opt = γ*·30000 so the
//    vol-point skew is visible at a few fills of vega;
//  - the WW band uses per-$-wealth risk aversion γ_ww = γ*·2e-5.

import { Rng, clamp } from "./math";
import { SsviSurface } from "./ssvi";
import { fullGreeks, FullGreeks } from "./bsm";
import { MmHjb, defaultProblem, asQuotes, asHalfSpread, asReservation, AsParams, OptMm, HedgeCadence } from "./models";
import { VannaVolga } from "./vannaVolga";
import { Ewma, RollSpread, microPrice, OfiTracker, JumpDetector, HurstEstimator, IntensityLearner } from "./micro";
import { PerpVenue, defaultPerpConfig, PerpQuoteLevel } from "./venue";
import { OptionMarket, defaultOptionConfig, OptionFill, RiskGate } from "./options";
import { RfqEngine, RfqRiskCtx, RfqQuoteView, RfqExecEvent, RFQ_TEMPLATES } from "./rfq";
import { portfolioMargin, MarginState } from "./margin";

export type StrategyId = "static" | "as" | "hjb" | "queue" | "micro" | "optmm" | "ladder" | "volsurf";

export interface StrategyInfo {
  id: StrategyId;
  name: string;
  blurb: string;
}

export const STRATEGIES: StrategyInfo[] = [
  { id: "static", name: "Static", blurb: "Fixed spread benchmark — no inventory dependence." },
  { id: "as", name: "Unified AS", blurb: "Avellaneda–Stoikov reservation price + closed-form half-spread." },
  { id: "hjb", name: "HJB Policy", blurb: "Exact HJB solve (RK4 backward on the v-system) — impulse FOC quotes." },
  { id: "queue", name: "Queue-Aware", blurb: "Cont–de Larrard Erlang fill probability vs the away-move clock." },
  { id: "micro", name: "Micro-Price", blurb: "Stoikov micro-price anchor instead of the mid." },
  { id: "optmm", name: "Options MM", blurb: "Vega-approx GLFT quotes + WW-band delta hedging." },
  { id: "ladder", name: "Multi-Level", blurb: "Pricing ladders: tiered distances/sizes, markout-toxicity aware." },
  { id: "volsurf", name: "Vol-Surface MM", blurb: "SSVI fair + vanna–volga overhedge + GLFT vega quotes + combined-delta perp legs." },
];

export interface EngineParams {
  /** Risk aversion slider γ* ∈ [0.02, 1]. */
  gamma: number;
  /** Fill intensity decay (price units) for the perp book. */
  kappa: number;
  a: number;
  /** Quoting horizon (seconds). */
  horizon: number;
  staticTicks: number;
  ladderBase: number;
  levels: number;
  hedgeMode: "off" | "every" | "ww";
  vvOn: boolean;
  sigmaVolAnnual: number;
  kappaV: number;
  aV: number;
  vegaPerFill: number;
  fillCostVol: number;
}

export const defaultParams: EngineParams = {
  gamma: 0.35,
  kappa: 0.8,
  a: 0.9,
  horizon: 600,
  staticTicks: 4,
  ladderBase: 3,
  levels: 3,
  hedgeMode: "ww",
  vvOn: true,
  sigmaVolAnnual: 0.9,
  kappaV: 600,
  aV: 0.02,
  vegaPerFill: 25,
  fillCostVol: 0.0015,
};

const YEAR_SECS = 365 * 24 * 3600;
/** Vega-space risk aversion scale (γ_opt = γ*·OPT_GAMMA_SCALE). */
const OPT_GAMMA_SCALE = 4.3;
/** Per-$-wealth risk aversion for the WW band. */
const WW_GAMMA_SCALE = 2e-5;
/** Desk starting capital (quote units) — PnL is measured against it. */
export const INITIAL_CAPITAL = 2500;

export interface ChainRow {
  moneyness: number;
  strike: number;
  call: LegQ;
  put: LegQ;
  greeks: FullGreeks;
}

export interface LegQ {
  ivBid: number;
  ivFair: number;
  ivAsk: number;
  vvShift: number;
  premBid: number;
  premAsk: number;
  pos: number;
}

export interface EngineSnapshot {
  clock: number;
  steps: number;
  running: boolean;
  spot: number;
  params: EngineParams;
  micro: {
    sigmaFast: number;
    sigmaSlow: number;
    hurst: number;
    microPrice: number;
    imbalance: number;
    ofi: number;
    ofiImpact: number;
    rollSpread: number;
    jumpStat: number;
    jumpExceed: number;
  };
  book: { bid: number; ask: number; bidSz: number; askSz: number; spread: number };
  midHistory: number[];
  equity: number;
  equityHistory: number[];
  position: number;
  pnl: {
    spreadCapture: number;
    inventory: number;
    fees: number;
    hedgeCost: number;
    funding: number;
    optionPremium: number;
    optionMark: number;
    total: number;
  };
  quotes: { bid: PerpQuoteLevel[]; ask: PerpQuoteLevel[] };
  strategy: StrategyId;
  markouts: { ratio: number; toxicity: number; multiplier: number; resolved: number; multBid: number; multAsk: number };
  funding: {
    rate: number;
    premiumIndex: number;
    nextIn: number;
    interval: number;
    paid: number;
    index: number;
    carrySkew: number;
  };
  learned: { aRaw: number; kappaRaw: number; aUsed: number; kappaUsed: number; fills: number; exposure: number };
  rfq: {
    active: RfqQuoteView[];
    recent: RfqQuoteView[];
    executions: RfqExecEvent[];
    requests: number;
    executed: number;
    refused: number;
    premiumFlow: number;
  };
  margin: MarginState;
  risk: {
    netDeltaLimit: number;
    grossHedgeCap: number;
    vegaLimit: number;
    vegaUsed: number;
    /** Unhedged combined delta (hedge leg + option book), lots. */
    netDelta: number;
    /** Gross hedge-leg size, lots. */
    grossHedge: number;
    drawdown: number;
    /** nominal | gated (soft band) | breach (hard limit) | halted (kill-switch). */
    state: "nominal" | "gated" | "breach" | "halted";
    gatedOn: string;
    /** Option book is being wound down (gross hedge cap breach). */
    windDown: boolean;
    halted: boolean;
    haltReason: string;
    events: Array<{ t: number; msg: string }>;
  };
  options: {
    expiries: number[];
    moneyness: number[];
    selectedExp: number;
    atmTerm: number[];
    /** Net vega per expiry ($ per unit vol). */
    vegaPerExp: number[];
    surfaceGrid: number[][];
    arb: { butterflyCondition: boolean; butterflyGrid: boolean; calendar: boolean };
    chain: ChainRow[];
    net: { delta: number; gamma: number; vega: number; theta: number; vanna: number; volga: number; mark: number };
    wings: Array<{ t: number; rrIv: number; bfIv: number; sigAtm: number }>;
    recentFills: OptionFill[];
    vvEnabled: boolean;
    vvPortfolio: { charge: number; wRr: number; wBf: number; volShift: number };
    hedgeBand: number;
    unhedgedDeltaLots: number;
    hedgeLotsTotal: number;
    fills: number;
  };
  perf: { p50: number; p99: number; cu: number; cuBudget: number; cuPct: number; ops: number };
}

export class MarketMaker {
  rng: Rng;
  venue: PerpVenue;
  options: OptionMarket;
  params: EngineParams;
  strategy: StrategyId = "volsurf";
  selectedExp = 1;

  sigmaFast = new Ewma(0.06, 0.0004);
  sigmaSlow = new Ewma(0.008, 0.0004);
  roll = new RollSpread(64);
  ofi = new OfiTracker(32);
  jumps = new JumpDetector(128);
  hurstE = new HurstEstimator(256);

  private hjb: MmHjb | null = null;
  private hjbAge = 0;

  hedgeBand: HedgeCadence;
  hedgeLotsTotal = 0;
  lastHedgeAction = 0;

  /** Kill-switch halt (drawdown only): quotes off, hedge + marks stay live. */
  halted = false;
  haltReason = "";
  /** Sim-clock timestamp of the halt (for the auto risk-on cooldown). */
  private haltedAt = -1;
  peakEquity = 0;
  /** Risk v2: the position limit binds on the UNHEDGED combined delta —
   *  a delta-hedged options book is the point of the desk, not a breach. */
  netDeltaLimit = 60;
  /** Sanity cap on the gross hedge leg (lots) — wind-down territory. */
  grossHedgeCap = 240;
  vegaLimit = defaultOptionConfig.vegaLimit;
  /** Directional gating state — self-recovering, never a full freeze. */
  riskState: "nominal" | "gated" | "breach" = "nominal";
  gatedOn = "";
  private lastRiskState: "nominal" | "gated" | "breach" = "nominal";
  riskEventTape: Array<{ t: number; msg: string }> = [];

  private latencies: number[] = [];
  private cu = 0;
  private cuBudget = 600;
  private ops = 0;
  /** Online Cox-intensity learner for the perp fill model. */
  learner = new IntensityLearner(defaultPerpConfig.aIntensity, defaultPerpConfig.kappaIntensity);
  /** Institutional RFQ lane (Paradigm / Derive V3 protocol shape). */
  rfqEngine: RfqEngine;
  /** Cached portfolio-margin state (recomputed every N steps). */
  marginState: MarginState;
  private marginAge = 999;
  /** Per-expiry net vega (term-structure buckets for per-pillar quotes). */
  vegaPerExp: number[] = [];

  clock = 0;
  steps = 0;
  private prevMid = 0;
  private optNet: { delta: number; gamma: number; vega: number; theta: number; vanna: number; volga: number; mark: number } = {
    delta: 0, gamma: 0, vega: 0, theta: 0, vanna: 0, volga: 0, mark: 0,
  };
  private recentOptFills: OptionFill[] = [];
  private lastQuotes: { bid: PerpQuoteLevel[]; ask: PerpQuoteLevel[] } = { bid: [], ask: [] };

  constructor(seed = 12345) {
    this.rng = new Rng(BigInt(seed));
    this.venue = new PerpVenue(defaultPerpConfig, this.rng);
    this.venue.cash = INITIAL_CAPITAL;
    this.options = new OptionMarket(defaultOptionConfig, this.rng, 0.55);
    this.params = { ...defaultParams };
    this.hedgeBand = new HedgeCadence(0.3, this.params.gamma * WW_GAMMA_SCALE, 100);
    this.prevMid = this.venue.mid;
    this.peakEquity = INITIAL_CAPITAL;
    this.rfqEngine = new RfqEngine(this.rng, 5, 1 / 40);
    this.marginState = portfolioMargin(this.venue.mid, [], 0, defaultPerpConfig.lotSize, INITIAL_CAPITAL, defaultOptionConfig.multiplier);
  }

  optMm(): OptMm {
    return new OptMm({
      gamma: this.params.gamma * OPT_GAMMA_SCALE,
      sigmaVol: this.params.sigmaVolAnnual / Math.sqrt(YEAR_SECS),
      kappaV: this.params.kappaV,
      aV: this.params.aV,
      vegaPerFill: this.params.vegaPerFill,
      fillCostVol: this.params.fillCostVol,
    });
  }

  /** Per-second relative sigma estimate (clamped to a sane regime). */
  estSigma(): number {
    const s = 0.7 * this.sigmaFast.value + 0.3 * this.sigmaSlow.value;
    return clamp(s, 2.5e-4, 0.0012);
  }

  /** One simulation step of dt seconds. */
  step(dt: number) {
    const t0 = performance.now();
    this.cu = 0;
    if (this.halted) {
      // auto risk-on: after a 30-s cooldown the desk re-enables quoting
      // with a re-anchored peak (risk-off / risk-on cycling)
      if (this.clock - this.haltedAt > 30) {
        this.halted = false;
        this.haltReason = "";
        this.peakEquity = this.totalEquity();
        this.logRiskEvent("risk-on after cooldown (peak re-anchored)");
      } else {
        // Risk-off: quoting is suppressed everywhere, but risk-REDUCING
        // paths stay live — the venue keeps marking, the surface keeps
        // evolving and the delta hedge keeps running. A kill-switch that
        // froze the hedge would trap the very risk it is meant to shed.
        // Firm RFQ quotes are pulled immediately (Derive MMP /
        // cancel-on-disconnect semantics: a tripped desk stops being firm).
        for (const q of this.rfqEngine.active) {
          q.status = "expired";
          q.note = "pulled: kill-switch (MMP)";
          this.rfqEngine.pushRecentPublic(q);
        }
        this.rfqEngine.active = [];
        this.venue.step(dt, this.estSigma(), { bid: [], ask: [] }, this.steps);
        this.options.step(dt);
        this.optNet = this.options.netGreeks(this.venue.mid, this.options.surface());
        this.hedge();
        this.clock += dt;
        this.steps++;
        this.peakEquity = Math.max(this.peakEquity, this.totalEquity());
        this.recordPerf(t0);
        return;
      }
    }
    const mid0 = this.venue.mid;
    const sigma = this.estSigma();

    // --- strategy quotes ---
    this.lastQuotes = this.perpQuotes(dt, sigma);
    this.cu += 8;

    // --- venue step with our quotes resting ---
    const fills = this.venue.step(dt, sigma, this.lastQuotes, this.steps);
    this.cu += fills.length * 10 + 6;
    this.ops += fills.length;
    // intensity learning: realized fills at their distances (ticks from
    // the pre-step mid) feed the online Cox MLE
    for (const f of fills) {
      // intensity calibration excludes swept/toxic prints — they are not
      // Cox-distance fills and would flatten the estimated decay
      if (!f.toxic) this.learner.observeFill(Math.abs(mid0 - f.price) / this.venue.cfg.tickSize);
    }
    this.cu += 2;

    // --- option market ---
    const surface = this.options.surface();
    this.options.step(dt);
    this.cu += 6;
    const optMm = this.optMm();
    const vv = this.vvFor(surface);
    // per-expiry vega buckets: each pillar's quotes lean against its own
    // term-structure inventory (eSSVI-style surface control)
    this.vegaPerExp = this.options.vegaPerExpiry(this.venue.mid, surface);
    this.cu += 6;
    // risk-gated option intake: one-sided vol quoting at the vega limit,
    // per-leg wind-down of the book at the gross hedge cap
    const optFills = this.options.clientStep(dt, this.venue.mid, surface, optMm, this.vegaPerExp, vv, this.riskGate());
    this.cu += 12 + optFills.length * 14;
    this.ops += optFills.length;
    this.recentOptFills.push(...optFills);
    if (this.recentOptFills.length > 24) this.recentOptFills.splice(0, this.recentOptFills.length - 24);

    // --- RFQ lane: institutional flow, TTL decay, atomic executions ---
    const rfqCtx: RfqRiskCtx = {
      unhedgedLots: this.unhedgedLots(),
      netDeltaLimit: this.netDeltaLimit,
      netVega: this.optNet.vega,
      vegaLimit: this.vegaLimit,
      freeze: this.marginState.utilization >= 0.9,
      lotSize: this.venue.cfg.lotSize,
      multiplier: this.options.cfg.multiplier,
      clock: this.clock,
    };
    const touches = this.rfqEngine.step(dt, this.venue.mid, this.options, optMm, surface, this.vegaPerExp, vv, rfqCtx);
    // governed surface: executed single-leg RFQ touches anchor the pillars
    for (const t of touches) this.options.anchorAtm(t.expIdx, t.iv, 0.12);
    this.cu += 10;

    // --- portfolio margin (SFPM scan, amortized over 20 steps) ---
    if (this.marginAge >= 20) {
      this.marginAge = 0;
      this.marginState = portfolioMargin(
        this.venue.mid,
        this.options.marginLegs(this.venue.mid, surface),
        this.venue.position,
        this.venue.cfg.lotSize,
        this.totalEquity(),
        this.options.cfg.multiplier,
      );
    }
    this.marginAge++;
    this.cu += 2;

    // --- estimators ---
    const mid = this.venue.mid;
    const ret = (mid - this.prevMid) / Math.max(mid, 1e-9);
    this.prevMid = mid;
    this.sigmaFast.update(Math.abs(ret) / Math.sqrt(Math.max(dt, 1e-9)) / Math.SQRT2);
    this.sigmaSlow.update(Math.abs(ret) / Math.sqrt(Math.max(dt, 1e-9)) / Math.SQRT2);
    this.roll.update(mid);
    this.jumps.update(ret, Math.abs(ret) / Math.max(sigma * Math.sqrt(dt), 1e-12));
    this.hurstE.update(ret);
    this.ofi.update(mid > mid0 ? 1 : -1, mid, mid0);
    this.cu += 14;

    // --- option book greeks ---
    this.optNet = this.options.netGreeks(this.venue.mid, surface);
    this.cu += 16;

    // --- delta hedging ---
    this.hedge();

    // --- risk engine ---
    this.riskCheck();

    this.clock += dt;
    this.steps++;
    this.peakEquity = Math.max(this.peakEquity, this.totalEquity());
    this.recordPerf(t0);
  }

  private vvFor(surface: SsviSurface): VannaVolga {
    // reference expiry for the wing instruments: the selected one
    const t = this.options.cfg.expiries[clamp(this.selectedExp, 0, this.options.cfg.expiries.length - 1)];
    return new VannaVolga(this.venue.mid, t, surface, 0, 0, this.params.vvOn ? this.options.cfg.vvDamping : 1e9);
  }

  private recordPerf(t0: number) {
    const ns = (performance.now() - t0) * 1e6;
    this.latencies.push(ns);
    if (this.latencies.length > 512) this.latencies.shift();
  }

  /** Perp quote ladder for the active strategy, risk-gated on the UNHEDGED
   *  combined delta (hedge leg + option book): near the limit the desk
   *  drops the risk-adding side and tightens the unwind side; at the limit
   *  it quotes unwind-only at the touch. Never both-sides-off — blocking
   *  risk-REDUCING trades would freeze the exposure (the classic dead-lock
   *  risk engines must avoid: the unwind path always stays live). */
  perpQuotes(dt: number, sigma: number): { bid: PerpQuoteLevel[]; ask: PerpQuoteLevel[] } {
    const full = this.perpQuotesInner(dt, sigma);
    const unhedged = this.unhedgedLots();
    const lim = this.netDeltaLimit;
    const tick = this.venue.cfg.tickSize;
    const mid = this.venue.mid;
    const lvl = (p: number, sz: number): PerpQuoteLevel => ({ p: Math.round(p / tick) * tick, sz });
    let out: { bid: PerpQuoteLevel[]; ask: PerpQuoteLevel[] };
    if (Math.abs(unhedged) >= lim) {
      // hard breach: unwind-only at the touch with boosted size — Cox
      // fill intensity at δ≈0 sheds the excess within seconds, then the
      // book self-recovers to two-sided quoting (hysteresis by band)
      out =
        unhedged > 0
          ? { bid: [], ask: [lvl(mid + tick, 8), lvl(mid + 2 * tick, 12)] }
          : { bid: [lvl(mid - 2 * tick, 12), lvl(mid - tick, 8)], ask: [] };
    } else if (Math.abs(unhedged) >= lim * 0.75) {
      // soft band: drop the adding side; pull the unwind side closer and
      // deepen it so the book actively trades out of the excess
      const k = 0.6;
      const tighten = (lv: PerpQuoteLevel[]) =>
        lv.map((l) => ({ p: Math.round((mid + (l.p - mid) * k) / tick) * tick, sz: Math.max(2, Math.round(l.sz * 1.5)) }));
      out = unhedged > 0 ? { bid: [], ask: tighten(full.ask) } : { bid: tighten(full.bid), ask: [] };
    } else {
      out = full;
    }
    // intensity learning: record what is actually RESTING (post-gating)
    this.learner.observeExposure(
      [
        ...out.bid.map((l) => ({ distTicks: Math.max(0, (mid - l.p) / tick), active: true })),
        ...out.ask.map((l) => ({ distTicks: Math.max(0, (l.p - mid) / tick), active: true })),
      ],
      dt,
    );
    return out;
  }

  private perpQuotesInner(dt: number, sigma: number): { bid: PerpQuoteLevel[]; ask: PerpQuoteLevel[] } {
    const v = this.venue;
    const tick = v.cfg.tickSize;
    const mid = v.mid;
    const q = v.position + this.optNet.delta / v.cfg.lotSize;
    // directional spread adaptation (Albers et al. 2025): fill likelihood
    // and post-fill returns trade off per side — widen the TOXIC side only
    const multB = v.markoutsBid.multiplier;
    const multA = v.markoutsAsk.multiplier;
    // learned fill intensity (online Cox MLE, shrunk toward the prior)
    const kappa = clamp(this.learner.kappaUsed(), 0.1, 3);
    const aI = clamp(this.learner.aUsed(), 1e-3, 3);
    const lvl = (p: number, sz: number): PerpQuoteLevel => ({ p: Math.round(p / tick) * tick, sz });
    const size0 = 2;
    // price-units vol over the quoting horizon
    const sigmaH = Math.max(0.2, sigma * mid * Math.sqrt(this.params.horizon));
    // funding carry (Le 2025 funding-aware MM): expected funding on the
    // quoting horizon shifts the reservation price against the carry side
    const carry = clamp(
      (-q * v.fundingRate * Math.min(this.params.horizon, v.cfg.fundingInterval) * mid) / v.cfg.fundingInterval / 2,
      -10 * tick,
      10 * tick,
    );
    const params: AsParams = {
      gamma: this.params.gamma,
      sigma: sigmaH,
      kappa,
      a: aI,
      t: 1,
    };
    const asHalf = asHalfSpread(params, 0);

    switch (this.strategy) {
      case "static": {
        const h = this.params.staticTicks * tick;
        return { bid: [lvl(mid - h * multB, size0)], ask: [lvl(mid + h * multA, size0)] };
      }
      case "as": {
        // unified AS: reservation price (inventory + funding carry) ± spread
        const h = clamp(asHalf, 1.2 * tick, 20 * tick);
        const r = mid - this.params.gamma * sigmaH * sigmaH * q + carry;
        return { bid: [lvl(r - h * multB, size0)], ask: [lvl(r + h * multA, size0)] };
      }
      case "hjb": {
        if (!this.hjb || this.hjbAge > 120) {
          const prob = defaultProblem(this.params.gamma, sigmaH, kappa, aI, 1);
          this.hjb = new MmHjb({ ...prob, qMax: 10, nSteps: 240 });
          this.hjbAge = 0;
        }
        this.hjbAge++;
        const da = this.hjb.deltaAsk(clamp(Math.round(q), -10, 10), 0.02);
        const db = this.hjb.deltaBid(clamp(Math.round(q), -10, 10), 0.02);
        const ha = clamp(isFinite(da) ? da : 20 * tick, 1.2 * tick, 20 * tick) * multA;
        const hb = clamp(isFinite(db) ? db : 20 * tick, 1.2 * tick, 20 * tick) * multB;
        return { bid: [lvl(mid + carry - hb, size0)], ask: [lvl(mid + carry + ha, size0)] };
      }
      case "queue": {
        // AS distance + Erlang fill-probability vs the away-move clock.
        const base = clamp(asHalf / tick, 1, 30);
        const mu = aI * 1.2;
        const nu = clamp(sigma * 40, 0.05, 3);
        const pick = (dT: number) => {
          const m = Math.max(1, Math.round(dT));
          const p = Math.pow(mu / (mu + nu), m + 1);
          return { m, score: m * tick * p * Math.exp(-kappa * (m * tick)) };
        };
        let best = pick(base);
        for (let i = 1; i <= 4; i++) {
          const c = pick(base + i);
          if (c.score > best.score) best = c;
        }
        return { bid: [lvl(mid - best.m * tick * multB, size0)], ask: [lvl(mid + best.m * tick * multA, size0)] };
      }
      case "micro": {
        const mp = microPrice(v.top.bid, v.top.ask, v.top.bidSz, v.top.askSz);
        const h = Math.max(2 * tick, v.spread * 0.8 + asHalf * 0.4);
        return { bid: [lvl(mp - h * multB, size0)], ask: [lvl(mp + h * multA, size0)] };
      }
      case "optmm": {
        const h = Math.max(2 * tick, this.params.ladderBase * tick + asHalf * 0.35);
        return { bid: [lvl(mid - h * multB, size0)], ask: [lvl(mid + h * multA, size0)] };
      }
      case "ladder": {
        const outB: PerpQuoteLevel[] = [];
        const outA: PerpQuoteLevel[] = [];
        const baseD = Math.max(this.params.ladderBase * tick, asHalf * 0.6);
        for (let i = 0; i < this.params.levels; i++) {
          const d = baseD * (1 + 0.7 * i);
          const sz = size0 * (1 + 0.6 * i);
          outB.push(lvl(mid - d * multB, sz));
          outA.push(lvl(mid + d * multA, sz));
        }
        return { bid: outB, ask: outA };
      }
      case "volsurf":
      default: {
        // combined-delta reservation skew + funding carry + tiered ladder
        // (skew clamped to 10 ticks — the reservation price is a tilt,
        // not a regime)
        const skew = clamp(-this.params.gamma * sigmaH * sigmaH * q + carry, -10 * tick, 10 * tick);
        const outB: PerpQuoteLevel[] = [];
        const outA: PerpQuoteLevel[] = [];
        const baseD = Math.max(2 * tick, asHalf * 0.7);
        for (let i = 0; i < this.params.levels; i++) {
          const d = baseD * (1 + 0.6 * i);
          outB.push(lvl(mid + skew - d * multB, size0 * (1 + 0.5 * i)));
          outA.push(lvl(mid + skew + d * multA, size0 * (1 + 0.5 * i)));
        }
        return { bid: outB, ask: outA };
      }
    }
  }

  /** Delta hedging: every-step / WW band / off. */
  private hedge() {
    if (this.params.hedgeMode === "off") return;
    const lotSize = this.venue.cfg.lotSize;
    const optDeltaLots = (this.optNet.delta * this.options.cfg.multiplier) / lotSize;
    const unhedged = this.venue.position + optDeltaLots;
    if (this.params.hedgeMode === "every") {
      if (Math.abs(unhedged) >= 1) {
        const lots = -Math.round(unhedged);
        this.venue.take(lots > 0 ? "buy" : "sell", Math.abs(lots));
        this.hedgeLotsTotal += Math.abs(lots);
        this.lastHedgeAction = this.steps;
        this.cu += 10;
      }
      return;
    }
    const cPerLot = this.venue.cfg.takerFee * this.venue.mid * lotSize + this.venue.spread / 2;
    this.hedgeBand = new HedgeCadence(cPerLot, this.params.gamma * WW_GAMMA_SCALE, this.venue.mid * lotSize);
    const action = this.hedgeBand.shouldHedge(unhedged);
    if (action !== null && Math.abs(action) >= 1) {
      this.venue.take(action > 0 ? "buy" : "sell", Math.abs(action));
      this.hedgeLotsTotal += Math.abs(action);
      this.lastHedgeAction = this.steps;
      this.cu += 10;
    }
  }

  private riskCheck() {
    if (this.halted) {
      // auto risk-on: after a 30-s cooldown the desk re-enables quoting
      // with a re-anchored peak (risk-off / risk-on cycling)
      if (this.clock - this.haltedAt > 30) {
        this.halted = false;
        this.haltReason = "";
        this.peakEquity = this.totalEquity();
        this.logRiskEvent("risk-on after cooldown (peak re-anchored)");
      }
      return;
    }
    const eq = this.totalEquity();
    // relative drawdown kill-switch (20% of peak, floor 180 units — the
    // option book's gamma MTM swings are an order above the perp drift)
    const ddLimit = Math.max(180, this.peakEquity * 0.2);
    if (this.peakEquity - eq > ddLimit) {
      this.halted = true;
      this.haltedAt = this.clock;
      this.haltReason = `drawdown kill-switch (${(this.peakEquity - eq).toFixed(1)} > ${ddLimit.toFixed(1)})`;
      this.logRiskEvent(`KILL-SWITCH — ${this.haltReason}; quotes off, hedge + marks live`);
      return;
    }
    // --- limit gating: directional, self-recovering, never a freeze ---
    // Limits bind on the UNHEDGED combined delta (not the raw hedge leg —
    // the hedge is how a short-gamma book gets flat), on gross hedge-leg
    // size as a sanity cap, on net vega, and on MARGIN UTILIZATION — the
    // protocol-grade constraint (Derive SFPM / Paradex SCAN lineage):
    // capital is the true limit, not lot counts. Breaches gate quoting
    // one-sided toward unwind; they never halt the desk.
    const unhedged = this.unhedgedLots();
    const gross = Math.abs(this.venue.position);
    const vega = Math.abs(this.optNet.vega);
    const util = this.marginState.utilization;
    let state: "nominal" | "gated" | "breach" = "nominal";
    const on: string[] = [];
    if (Math.abs(unhedged) >= this.netDeltaLimit) {
      state = "breach";
      on.push("net-delta");
    } else if (Math.abs(unhedged) >= this.netDeltaLimit * 0.75) {
      state = "gated";
      on.push("net-delta");
    }
    if (gross >= this.grossHedgeCap) {
      state = "breach";
      on.push("gross-hedge");
    }
    if (vega >= this.vegaLimit) {
      state = "breach";
      on.push("vega");
    } else if (vega >= this.vegaLimit * 0.75 && state === "nominal") {
      state = "gated";
      on.push("vega");
    }
    if (util >= 0.9) {
      state = "breach";
      on.push("margin");
    } else if (util >= 0.7 && state === "nominal") {
      state = "gated";
      on.push("margin");
    }
    this.riskState = state;
    this.gatedOn = on.join("+");
    if (this.riskState !== this.lastRiskState) {
      this.logRiskEvent(
        this.riskState === "nominal"
          ? "risk nominal — quoting live both sides"
          : `risk ${this.riskState} (${this.gatedOn}) — risk-adding side gated, unwind side live`,
      );
      this.lastRiskState = this.riskState;
    }
  }

  totalEquity(): number {
    return this.venue.equity + this.options.cash + this.optNet.mark;
  }

  /** Risk-manager resume: clear a kill-switch halt and re-anchor the peak. */
  resume() {
    if (this.halted) this.logRiskEvent("manual risk-manager resume");
    this.halted = false;
    this.haltReason = "";
    this.haltedAt = -1;
    this.peakEquity = this.totalEquity();
  }

  /** The combined delta the desk actually carries: hedge leg + option book. */
  unhedgedLots(): number {
    return this.venue.position + (this.optNet.delta * this.options.cfg.multiplier) / this.venue.cfg.lotSize;
  }

  /** Option-flow risk gate: vega one-siding at the limit, per-leg wind-down
   *  of the book at the gross hedge cap or hard margin breach (only
   *  risk-REDUCING fills served). */
  private riskGate(): RiskGate {
    const vegaOver = Math.abs(this.optNet.vega) >= this.vegaLimit;
    const grossOver = Math.abs(this.venue.position) >= this.grossHedgeCap;
    const marginHard = this.marginState.utilization >= 0.9;
    return {
      blockVolBid: vegaOver && this.optNet.vega > 0,
      blockVolAsk: vegaOver && this.optNet.vega < 0,
      windDown: grossOver || marginHard,
    };
  }

  /** Manual RFQ: the UI taker requests a firm package quote. Returns the
   *  quote id (firm for the TTL window — Paradigm/Derive hold-for-time). */
  requestRfq(templateId: string, lots: number, expIdx?: number): number | null {
    const tpl = RFQ_TEMPLATES.find((t) => t.id === templateId);
    if (!tpl) return null;
    const e = clamp(expIdx ?? this.selectedExp, 0, this.options.cfg.expiries.length - 1);
    const { legs, label } = this.rfqEngine.buildPackage(tpl, e, lots, this.options);
    const surface = this.options.surface();
    const optMm = this.optMm();
    const vpe = this.vegaPerExp.length ? this.vegaPerExp : this.options.vegaPerExpiry(this.venue.mid, surface);
    const q = this.rfqEngine.quote(legs, label, "manual", this.venue.mid, this.options, optMm, surface, vpe, this.vvFor(surface), 99999);
    this.rfqEngine.active.push(q);
    return q.id;
  }

  /** Cancel a manual quote (the UI taker's real-time firm window elapsed). */
  cancelRfq(id: number) {
    const q = this.rfqEngine.active.find((x) => x.id === id);
    if (!q) return;
    q.status = "expired";
    q.note = "expired — firm window elapsed";
    this.rfqEngine.pushRecentPublic(q);
    this.rfqEngine.active = this.rfqEngine.active.filter((x) => x.status === "quoted");
  }

  /** Execute a quoted RFQ package atomically (desk side). */
  executeRfq(id: number, side: "desk-buys" | "desk-sells"): { ok: boolean; reason?: string } {
    const surface = this.options.surface();
    const optMm = this.optMm();
    const vpe = this.vegaPerExp.length ? this.vegaPerExp : this.options.vegaPerExpiry(this.venue.mid, surface);
    const ctx: RfqRiskCtx = {
      unhedgedLots: this.unhedgedLots(),
      netDeltaLimit: this.netDeltaLimit,
      netVega: this.optNet.vega,
      vegaLimit: this.vegaLimit,
      freeze: this.marginState.utilization >= 0.9,
      lotSize: this.venue.cfg.lotSize,
      multiplier: this.options.cfg.multiplier,
      clock: this.clock,
    };
    const res = this.rfqEngine.execute(id, side, this.venue.mid, this.options, optMm, surface, vpe, this.vvFor(surface), ctx);
    if (res.ok) {
      // the margin scan is stale after a package execution — refresh now
      this.marginState = portfolioMargin(
        this.venue.mid,
        this.options.marginLegs(this.venue.mid, surface),
        this.venue.position,
        this.venue.cfg.lotSize,
        this.totalEquity(),
        this.options.cfg.multiplier,
      );
      this.marginAge = 0;
    }
    return res;
  }

  private logRiskEvent(msg: string) {
    this.riskEventTape.push({ t: Math.round(this.clock), msg });
    if (this.riskEventTape.length > 16) this.riskEventTape.shift();
  }

  /** Build the immutable UI snapshot. */
  snapshot(): EngineSnapshot {
    const v = this.venue;
    const surface = this.options.surface();
    const optMm = this.optMm();
    const s = v.mid;
    const vv = this.vvFor(surface);
    const net = this.options.netGreeks(s, surface);
    const e = this.options.cfg.expiries.length;
    const m = this.options.cfg.moneyness.length;
    // per-expiry net vega (term-structure buckets) — before the chain so
    // pillar quotes lean against their own inventory
    const vegaPerExp = this.vegaPerExp.length === e ? this.vegaPerExp.slice() : this.options.vegaPerExpiry(s, surface);
    const surfaceGrid: number[][] = [];
    for (let ei = 0; ei < e; ei++) {
      const row: number[] = [];
      for (let li = 0; li < m; li++) {
        row.push(surface.iv(this.options.cfg.moneyness[li], this.options.cfg.expiries[ei]));
      }
      surfaceGrid.push(row);
    }
    const sel = clamp(this.selectedExp, 0, e - 1);
    const vegaSel = vegaPerExp[sel] ?? net.vega;
    const chain: ChainRow[] = this.options.cfg.moneyness.map((mo, li) => {
      const strike = Math.exp(mo) * s;
      const qc = this.options.quotes(sel, li, "call", s, surface, optMm, vegaSel, vv);
      const qp = this.options.quotes(sel, li, "put", s, surface, optMm, vegaSel, vv);
      const gc = fullGreeks("call", s, strike, 0, 0, qc.ivFair, this.options.cfg.expiries[sel]);
      return {
        moneyness: mo,
        strike,
        call: { ivBid: qc.ivBid, ivFair: qc.ivFair, ivAsk: qc.ivAsk, vvShift: qc.vvShift, premBid: qc.premBid, premAsk: qc.premAsk, pos: this.options.pos[sel][li][0] },
        put: { ivBid: qp.ivBid, ivFair: qp.ivFair, ivAsk: qp.ivAsk, vvShift: qp.vvShift, premBid: qp.premBid, premAsk: qp.premAsk, pos: this.options.pos[sel][li][1] },
        greeks: gc,
      };
    });
    const vvp = vv.portfolioCharge(net.vanna, net.volga, Math.max(Math.abs(net.vega), 1e-9));
    const lat = this.latencies.slice().sort((a, b) => a - b);
    const p50 = lat.length ? lat[Math.floor(lat.length * 0.5)] : 0;
    const p99 = lat.length ? lat[Math.floor(lat.length * 0.99)] : 0;
    const unhedged = v.position + (net.delta * this.options.cfg.multiplier) / v.cfg.lotSize;
    return {
      clock: this.clock,
      steps: this.steps,
      running: !this.halted,
      spot: s,
      params: { ...this.params },
      micro: {
        sigmaFast: this.sigmaFast.value,
        sigmaSlow: this.sigmaSlow.value,
        hurst: this.hurstE.estimate(),
        microPrice: microPrice(v.top.bid, v.top.ask, v.top.bidSz, v.top.askSz),
        imbalance: v.top.bidSz / Math.max(1, v.top.bidSz + v.top.askSz),
        ofi: this.ofi.value,
        ofiImpact: this.ofi.impactFit,
        rollSpread: this.roll.estimate(),
        jumpStat: this.jumps.stat(),
        jumpExceed: this.jumps.exceedRate(),
      },
      book: { bid: v.top.bid, ask: v.top.ask, bidSz: v.top.bidSz, askSz: v.top.askSz, spread: v.spread },
      midHistory: v.midHistory.slice(-240),
      equity: this.totalEquity(),
      equityHistory: v.equityHistory.slice(-240),
      position: v.position,
      pnl: {
        spreadCapture: v.spreadCapture,
        inventory: v.inventoryPnl(),
        fees: -v.feesPaid,
        hedgeCost: -v.hedgeFees,
        funding: -v.fundingPaid,
        optionPremium: this.options.cash,
        optionMark: net.mark,
        total: this.totalEquity(),
      },
      quotes: this.lastQuotes,
      strategy: this.strategy,
      markouts: { ...v.markouts.state(), multBid: v.markoutsBid.multiplier, multAsk: v.markoutsAsk.multiplier },
      funding: {
        rate: v.fundingRate,
        premiumIndex: v.premiumIndex,
        nextIn: Math.max(0, v.cfg.fundingInterval - v.fundingElapsed),
        interval: v.cfg.fundingInterval,
        paid: v.fundingPaid,
        index: v.index,
        carrySkew: clamp(
          (-(v.position + net.delta / v.cfg.lotSize) * v.fundingRate * Math.min(this.params.horizon, v.cfg.fundingInterval) * s) /
            v.cfg.fundingInterval /
            2,
          -10 * v.cfg.tickSize,
          10 * v.cfg.tickSize,
        ),
      },
      learned: this.learner.state(),
      rfq: {
        active: this.rfqEngine.active.map((qq) => ({ ...qq, legs: qq.legs.map((l) => ({ ...l })) })),
        recent: this.rfqEngine.recent.slice(-8),
        executions: this.rfqEngine.executions.slice(-10),
        requests: this.rfqEngine.requests,
        executed: this.rfqEngine.executed,
        refused: this.rfqEngine.refused,
        premiumFlow: this.rfqEngine.premiumFlow,
      },
      margin: this.marginState,
      risk: {
        netDeltaLimit: this.netDeltaLimit,
        grossHedgeCap: this.grossHedgeCap,
        vegaLimit: this.vegaLimit,
        vegaUsed: Math.abs(net.vega),
        netDelta: unhedged,
        grossHedge: Math.abs(v.position),
        drawdown: this.peakEquity - this.totalEquity(),
        state: this.halted ? "halted" : this.riskState,
        gatedOn: this.gatedOn,
        windDown: Math.abs(v.position) >= this.grossHedgeCap,
        halted: this.halted,
        haltReason: this.haltReason,
        events: this.riskEventTape.slice(-8),
      },
      options: {
        expiries: this.options.cfg.expiries,
        moneyness: this.options.cfg.moneyness,
        selectedExp: sel,
        atmTerm: this.options.atm.slice(),
        vegaPerExp,
        surfaceGrid,
        arb: surface.arbReport(),
        chain,
        net,
        wings: this.options.wingState(surface, s),
        recentFills: this.recentOptFills.slice(-10),
        vvEnabled: this.params.vvOn,
        vvPortfolio: { charge: vvp.charge, wRr: vvp.wRr, wBf: vvp.volShift, volShift: vvp.volShift },
        hedgeBand: this.hedgeBand.bandLots,
        unhedgedDeltaLots: unhedged,
        hedgeLotsTotal: this.hedgeLotsTotal,
        fills: this.options.fills,
      },
      perf: {
        p50,
        p99,
        cu: this.cu,
        cuBudget: this.cuBudget,
        cuPct: (this.cu / this.cuBudget) * 100,
        ops: this.ops,
      },
    };
  }
}
