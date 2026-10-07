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
import { Ewma, RollSpread, microPrice, OfiTracker, JumpDetector, HurstEstimator } from "./micro";
import { PerpVenue, defaultPerpConfig, PerpQuoteLevel } from "./venue";
import { OptionMarket, defaultOptionConfig, OptionFill } from "./options";

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
export const INITIAL_CAPITAL = 1000;

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
    optionPremium: number;
    optionMark: number;
    total: number;
  };
  quotes: { bid: PerpQuoteLevel[]; ask: PerpQuoteLevel[] };
  strategy: StrategyId;
  markouts: { ratio: number; toxicity: number; multiplier: number; resolved: number };
  risk: {
    positionLimit: number;
    vegaLimit: number;
    vegaUsed: number;
    drawdown: number;
    halted: boolean;
    haltReason: string;
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

  halted = false;
  haltReason = "";
  peakEquity = 0;
  positionLimit = 60;
  vegaLimit = defaultOptionConfig.vegaLimit;

  private latencies: number[] = [];
  private cu = 0;
  private cuBudget = 600;
  private ops = 0;

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
      this.venue.step(dt, this.estSigma(), { bid: [], ask: [] }, this.steps);
      this.clock += dt;
      this.steps++;
      this.recordPerf(t0);
      return;
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

    // --- option market ---
    const surface = this.options.surface();
    this.options.step(dt);
    this.cu += 6;
    const optMm = this.optMm();
    const vv = this.vvFor(surface);
    const optFills = this.options.clientStep(dt, this.venue.mid, surface, optMm, this.optNet.vega, vv);
    this.cu += 12 + optFills.length * 14;
    this.ops += optFills.length;
    this.recentOptFills.push(...optFills);
    if (this.recentOptFills.length > 24) this.recentOptFills.splice(0, this.recentOptFills.length - 24);

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

  /** Perp quote ladder for the active strategy (one-sided near limits). */
  perpQuotes(dt: number, sigma: number): { bid: PerpQuoteLevel[]; ask: PerpQuoteLevel[] } {
    const full = this.perpQuotesInner(dt, sigma);
    // inventory gating: stop quoting the side that adds inventory when
    // within 25% of the hard limit (the desk trades out, not breaches)
    const pos = this.venue.position;
    const lim = this.positionLimit;
    if (pos > lim * 0.75) full.bid = [];
    if (pos < -lim * 0.75) full.ask = [];
    return full;
  }

  private perpQuotesInner(dt: number, sigma: number): { bid: PerpQuoteLevel[]; ask: PerpQuoteLevel[] } {
    const v = this.venue;
    const tick = v.cfg.tickSize;
    const mid = v.mid;
    const q = v.position + this.optNet.delta / v.cfg.lotSize;
    const mult = v.markouts.multiplier;
    const lvl = (p: number, sz: number): PerpQuoteLevel => ({ p: Math.round(p / tick) * tick, sz });
    const size0 = 2;
    // price-units vol over the quoting horizon
    const sigmaH = Math.max(0.2, sigma * mid * Math.sqrt(this.params.horizon));
    const params: AsParams = {
      gamma: this.params.gamma,
      sigma: sigmaH,
      kappa: this.params.kappa,
      a: this.params.a,
      t: 1,
    };
    const asHalf = asHalfSpread(params, 0);

    switch (this.strategy) {
      case "static": {
        const h = this.params.staticTicks * tick * mult;
        return { bid: [lvl(mid - h, size0)], ask: [lvl(mid + h, size0)] };
      }
      case "as": {
        const h = clamp(asHalf, 1.2 * tick, 20 * tick) * mult;
        return { bid: [lvl(mid - h, size0)], ask: [lvl(mid + h, size0)] };
      }
      case "hjb": {
        if (!this.hjb || this.hjbAge > 120) {
          const prob = defaultProblem(this.params.gamma, sigmaH, this.params.kappa, this.params.a, 1);
          this.hjb = new MmHjb({ ...prob, qMax: 10, nSteps: 240 });
          this.hjbAge = 0;
        }
        this.hjbAge++;
        const da = this.hjb.deltaAsk(clamp(Math.round(q), -10, 10), 0.02);
        const db = this.hjb.deltaBid(clamp(Math.round(q), -10, 10), 0.02);
        const ha = clamp(isFinite(da) ? da : 20 * tick, 1.2 * tick, 20 * tick) * mult;
        const hb = clamp(isFinite(db) ? db : 20 * tick, 1.2 * tick, 20 * tick) * mult;
        return { bid: [lvl(mid - hb, size0)], ask: [lvl(mid + ha, size0)] };
      }
      case "queue": {
        // AS distance + Erlang fill-probability vs the away-move clock.
        const base = clamp(asHalf / tick, 1, 30);
        const mu = this.params.a * 1.2;
        const nu = clamp(sigma * 40, 0.05, 3);
        const pick = (dT: number) => {
          const m = Math.max(1, Math.round(dT));
          const p = Math.pow(mu / (mu + nu), m + 1);
          return { m, score: m * tick * p * Math.exp(-this.params.kappa * (m * tick)) };
        };
        let best = pick(base);
        for (let i = 1; i <= 4; i++) {
          const c = pick(base + i);
          if (c.score > best.score) best = c;
        }
        const d = best.m * tick * mult;
        return { bid: [lvl(mid - d, size0)], ask: [lvl(mid + d, size0)] };
      }
      case "micro": {
        const mp = microPrice(v.top.bid, v.top.ask, v.top.bidSz, v.top.askSz);
        const h = Math.max(2 * tick, v.spread * 0.8 + asHalf * 0.4) * mult;
        return { bid: [lvl(mp - h, size0)], ask: [lvl(mp + h, size0)] };
      }
      case "optmm": {
        const h = Math.max(2 * tick, this.params.ladderBase * tick + asHalf * 0.35) * mult;
        return { bid: [lvl(mid - h, size0)], ask: [lvl(mid + h, size0)] };
      }
      case "ladder": {
        const outB: PerpQuoteLevel[] = [];
        const outA: PerpQuoteLevel[] = [];
        const baseD = Math.max(this.params.ladderBase * tick, asHalf * 0.6);
        for (let i = 0; i < this.params.levels; i++) {
          const d = baseD * (1 + 0.7 * i) * mult;
          const sz = size0 * (1 + 0.6 * i);
          outB.push(lvl(mid - d, sz));
          outA.push(lvl(mid + d, sz));
        }
        return { bid: outB, ask: outA };
      }
      case "volsurf":
      default: {
        // combined-delta reservation skew + tiered ladder (skew clamped to
        // 10 ticks — the reservation price is a tilt, not a regime)
        const skew = clamp(-this.params.gamma * sigmaH * sigmaH * q, -10 * tick, 10 * tick);
        const outB: PerpQuoteLevel[] = [];
        const outA: PerpQuoteLevel[] = [];
        const baseD = Math.max(2 * tick, asHalf * 0.7);
        for (let i = 0; i < this.params.levels; i++) {
          const d = baseD * (1 + 0.6 * i) * mult;
          outB.push(lvl(mid + skew - d, size0 * (1 + 0.5 * i)));
          outA.push(lvl(mid + skew + d, size0 * (1 + 0.5 * i)));
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
    const eq = this.totalEquity();
    // relative drawdown kill-switch (15% of peak, floor 180 units — the
    // option book's gamma MTM swings are an order above the perp drift)
    const ddLimit = Math.max(180, this.peakEquity * 0.15);
    if (this.peakEquity - eq > ddLimit) {
      this.halted = true;
      this.haltReason = `drawdown kill-switch (${(this.peakEquity - eq).toFixed(1)} > ${ddLimit.toFixed(1)})`;
    } else if (Math.abs(this.venue.position) > this.positionLimit) {
      this.halted = true;
      this.haltReason = "perp position limit breach";
    } else if (Math.abs(this.optNet.vega) > this.vegaLimit) {
      this.halted = true;
      this.haltReason = "vega limit breach";
    }
  }

  totalEquity(): number {
    return this.venue.equity + this.options.cash + this.optNet.mark;
  }

  /** Risk-manager resume: clear the halt and re-anchor the peak. */
  resume() {
    this.halted = false;
    this.haltReason = "";
    this.peakEquity = this.totalEquity();
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
    const surfaceGrid: number[][] = [];
    for (let ei = 0; ei < e; ei++) {
      const row: number[] = [];
      for (let li = 0; li < m; li++) {
        row.push(surface.iv(this.options.cfg.moneyness[li], this.options.cfg.expiries[ei]));
      }
      surfaceGrid.push(row);
    }
    const sel = clamp(this.selectedExp, 0, e - 1);
    const chain: ChainRow[] = this.options.cfg.moneyness.map((mo, li) => {
      const strike = Math.exp(mo) * s;
      const qc = this.options.quotes(sel, li, "call", s, surface, optMm, net.vega, vv);
      const qp = this.options.quotes(sel, li, "put", s, surface, optMm, net.vega, vv);
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
    // per-expiry net vega
    const vegaPerExp: number[] = [];
    for (let e = 0; e < this.options.cfg.expiries.length; e++) {
      const t = this.options.cfg.expiries[e];
      let vegaE = 0;
      for (let l = 0; l < this.options.cfg.moneyness.length; l++) {
        const strike = Math.exp(this.options.cfg.moneyness[l]) * s;
        const iv = surface.iv(this.options.cfg.moneyness[l], t);
        for (let ki = 0; ki < 2; ki++) {
          const lots = this.options.pos[e][l][ki];
          if (lots === 0) continue;
          vegaE += fullGreeks(ki === 0 ? "call" : "put", s, strike, 0, 0, iv, t).vega * lots;
        }
      }
      vegaPerExp.push(vegaE);
    }
    const lat = this.latencies.slice().sort((a, b) => a - b);
    const p50 = lat.length ? lat[Math.floor(lat.length * 0.5)] : 0;
    const p99 = lat.length ? lat[Math.floor(lat.length * 0.99)] : 0;
    const unhedged = v.position + net.delta / v.cfg.lotSize;
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
        optionPremium: this.options.cash,
        optionMark: net.mark,
        total: this.totalEquity(),
      },
      quotes: this.lastQuotes,
      strategy: this.strategy,
      markouts: v.markouts.state(),
      risk: {
        positionLimit: this.positionLimit,
        vegaLimit: this.vegaLimit,
        vegaUsed: Math.abs(net.vega),
        drawdown: this.peakEquity - this.totalEquity(),
        halted: this.halted,
        haltReason: this.haltReason,
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
