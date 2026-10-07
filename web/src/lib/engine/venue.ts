// Simulated perp CLOB venue — the MQH-lite venue of the Rust `engine`
// crate: latent fair price with drift/jumps, agent flow at the touch,
// our quotes filled through a Cox-intensity model λ(δ) = A·e^{−κδ} plus
// informed sweeps (adverse selection), fee accounting, per-fill
// markouts and PnL attribution.

import { Rng } from "./math";

export interface PerpQuoteLevel {
  /** Price. */
  p: number;
  /** Size in lots. */
  sz: number;
}

export interface Fill {
  side: "bid" | "ask";
  price: number;
  lots: number;
  fee: number;
  /** True when the fill came from an informed sweep (toxic). */
  toxic: boolean;
}

export interface MarkoutTrackerState {
  ratio: number;
  toxicity: number;
  multiplier: number;
  resolved: number;
}

export class MarkoutTracker {
  private pending: Array<{ step: number; signed: number; price: number; halfSpread: number }> = [];
  private resolved: Array<number> = [];
  private ewmaRatio = 0;
  constructor(public horizonSteps: number, public theta: number, public maxMult = 4) {}

  onFill(step: number, signed: number, price: number, halfSpread: number) {
    this.pending.push({ step, signed, price, halfSpread });
  }

  onStep(step: number, mid: number) {
    const keep: typeof this.pending = [];
    for (const f of this.pending) {
      if (step - f.step >= this.horizonSteps) {
        const drift = (mid - f.price) * f.signed;
        // normalized by the captured edge: >0 means we won the markout
        const edge = Math.max(f.halfSpread, 1e-9);
        this.resolved.push(drift / edge);
        if (this.resolved.length > 512) this.resolved.shift();
      } else {
        keep.push(f);
      }
    }
    this.pending = keep;
    if (this.resolved.length) {
      const recent = this.resolved.slice(-64);
      const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
      this.ewmaRatio = 0.1 * mean + 0.9 * this.ewmaRatio;
    }
  }

  get ratio(): number {
    return this.ewmaRatio;
  }
  get toxicity(): number {
    return clamp01(-this.ewmaRatio);
  }
  get multiplier(): number {
    return Math.min(1 + this.theta * this.toxicity(), this.maxMult);
  }
  private toxicity(): number {
    return clamp01(-this.ewmaRatio);
  }
  get resolvedCount(): number {
    return this.resolved.length;
  }
  state(): MarkoutTrackerState {
    return { ratio: this.ewmaRatio, toxicity: this.toxicity(), multiplier: this.multiplier, resolved: this.resolved.length };
  }
}

function clamp01(x: number): number {
  return Math.min(Math.max(x, 0), 1);
}

export interface PnlAttribution {
  spreadCapture: number;
  inventory: number;
  fees: number;
  hedgeCost: number;
  total: number;
}

export interface PerpConfig {
  tickSize: number;
  lotSize: number;
  /** Maker fee fraction (per lot notional). */
  makerFee: number;
  /** Taker fee fraction. */
  takerFee: number;
  /** Fill intensity scale A (per second). */
  aIntensity: number;
  /** Fill intensity decay per tick of distance. */
  kappaIntensity: number;
  /** Informed-arrival intensity (per second). */
  informedRate: number;
  /** Informed impact in ticks. */
  informedImpact: number;
  /** Initial price. */
  s0: number;
}

export const defaultPerpConfig: PerpConfig = {
  tickSize: 0.5,
  lotSize: 1,
  makerFee: 0.00002,
  takerFee: 0.0005,
  aIntensity: 0.12,
  kappaIntensity: 0.5,
  informedRate: 0.012,
  informedImpact: 6,
  s0: 100,
};

export interface BookTop {
  bid: number;
  ask: number;
  bidSz: number;
  askSz: number;
}

/**
 * The perp venue: latent fair value + displayed book + our ladder.
 * Our resting quotes fill via Cox intensity on their distance; informed
 * agents occasionally sweep the near side right before a move.
 */
export class PerpVenue {
  /** Latent fair price. */
  fair: number;
  /** Displayed top-of-book (synthetic agents, exogenous of us). */
  top: BookTop;
  /** Our net position in lots. */
  position = 0;
  /** Realized cash. */
  cash = 0;
  feesPaid = 0;
  hedgeFees = 0;
  spreadCapture = 0;
  private markSum = 0;
  private prevMark = 0;
  private drift = 0;
  private sigmaState = 0.00035; // per sqrt-second price vol (~0.55%/√s? scaled below)
  readonly markouts = new MarkoutTracker(5, 1.5);
  tape: Array<{ t: number; price: number; lots: number; side: "bid" | "ask"; ours: boolean }> = [];
  midHistory: number[] = [];
  equityHistory: number[] = [];

  constructor(public cfg: PerpConfig, private rng: Rng) {
    this.fair = cfg.s0;
    // initial top consistent with the step-0 rebuild structure (half-spread
    // = tick/2) so the estimator is not poisoned by a one-off structural jump
    this.top = { bid: cfg.s0 - cfg.tickSize / 2, ask: cfg.s0 + cfg.tickSize / 2, bidSz: 40, askSz: 40 };
  }

  get mid(): number {
    return 0.5 * (this.top.bid + this.top.ask);
  }

  get spread(): number {
    return this.top.ask - this.top.bid;
  }

  get equity(): number {
    return this.cash + this.position * this.mid;
  }

  /** Advance the market one step of `dt` seconds with our quotes resting. */
  step(
    dt: number,
    sigma: number,
    quotes: { bid: PerpQuoteLevel[]; ask: PerpQuoteLevel[] },
    stepIndex: number,
    volOfVol = 0.6,
  ): Fill[] {
    const cfg = this.cfg;
    // --- latent price: OU drift + diffusion + vol regime ---
    // gentle mean reversion to s0 keeps the sim in a sane regime
    this.drift = 0.99 * this.drift - 0.05 * this.drift * dt + 0.0008 * Math.log(this.cfg.s0 / this.fair);
    this.sigmaState = clampNum(this.sigmaState * Math.exp(volOfVol * Math.sqrt(dt) * 0.02 * this.rng.normal()), 1e-5, 0.004);
    const effSigma = sigma;
    let jump = 0;
    // informed flow: spike fill intensity on one side, then move the fair
    let informedDir = 0;
    if (this.rng.uniform() < cfg.informedRate * dt) {
      informedDir = this.rng.uniform() < 0.5 ? 1 : -1;
      // relative log-move of informedImpact ticks (e.g. 6 ticks ≈ 3% at 100)
      jump = (informedDir * cfg.informedImpact * cfg.tickSize * (0.5 + this.rng.uniform())) / this.fair;
    }
    this.fair = Math.max(1, this.fair * Math.exp(this.drift * dt + effSigma * Math.sqrt(dt) * this.rng.normal() + jump));
    // --- displayed book rebuilds around the fair ---
    const halfSpr = Math.max(cfg.tickSize, effSigma * Math.sqrt(dt) * 2.2 * 100) / 2;
    // quote-flicker noise kept well below the per-step diffusion so the
    // estimator measures the latent process, not the flicker
    const noise = (this.rng.uniform() - 0.5) * cfg.tickSize * 0.3;
    this.top.bid = roundTick(this.fair - halfSpr + noise, cfg.tickSize);
    this.top.ask = this.top.bid + Math.ceil((2 * halfSpr) / cfg.tickSize) * cfg.tickSize;
    this.top.bidSz = Math.max(4, 40 + Math.round(30 * this.rng.normal() * 0.2 + 30 * (0.5 - Math.abs(this.fair / this.top.bid - 1) * 50)));
    this.top.askSz = Math.max(4, 40 + Math.round(30 * this.rng.normal() * 0.2));
    // --- our fills: Cox intensity per level + informed sweeps ---
    const fills: Fill[] = [];
    for (const lvl of quotes.bid) {
      const delta = Math.max(0, this.mid - lvl.p);
      let lam = cfg.aIntensity * Math.exp(-cfg.kappaIntensity * (delta / cfg.tickSize)) * dt;
      if (informedDir < 0) lam += cfg.informedRate * dt * 8; // informed selling hits our bid
      const n = this.rng.poisson(Math.min(lam, 0.95));
      if (n > 0) fills.push(this.applyFill("bid", lvl.p, Math.min(n * lvl.sz, lvl.sz), false, stepIndex));
    }
    for (const lvl of quotes.ask) {
      const delta = Math.max(0, lvl.p - this.mid);
      let lam = cfg.aIntensity * Math.exp(-cfg.kappaIntensity * (delta / cfg.tickSize)) * dt;
      if (informedDir > 0) lam += cfg.informedRate * dt * 8;
      const n = this.rng.poisson(Math.min(lam, 0.95));
      if (n > 0) fills.push(this.applyFill("ask", lvl.p, Math.min(n * lvl.sz, lvl.sz), false, stepIndex));
    }
    // hard sweep: market order crossing several levels
    if (this.rng.uniform() < 0.05 * dt) {
      const dir = this.rng.uniform() < 0.5 ? "bid" : "ask";
      const depth = (1 + this.rng.poisson(2)) * cfg.tickSize * 2;
      const levels = dir === "bid" ? quotes.bid : quotes.ask;
      for (const lvl of levels) {
        const dist = dir === "bid" ? this.mid - lvl.p : lvl.p - this.mid;
        if (dist <= depth) {
          fills.push(this.applyFill(dir, lvl.p, lvl.sz, true, stepIndex));
        }
      }
    }
    // --- accounting ---
    const midNow = this.mid;
    this.markouts.onStep(stepIndex, midNow);
    this.midHistory.push(midNow);
    if (this.midHistory.length > 720) this.midHistory.shift();
    // equity mark decomposition
    const mark = this.position * midNow;
    this.markSum = mark;
    this.equityHistory.push(this.equity);
    if (this.equityHistory.length > 720) this.equityHistory.shift();
    this.prevMark = mark;
    return fills;
  }

  private applyFill(side: "bid" | "ask", price: number, lots: number, toxic: boolean, stepIndex: number): Fill {
    const signed = side === "bid" ? 1 : -1; // buy on our bid
    const notional = Math.abs(price * lots * this.cfg.lotSize);
    const fee = notional * this.cfg.makerFee;
    this.position += signed * lots;
    this.cash -= signed * price * lots;
    this.feesPaid += fee;
    this.cash -= fee;
    this.spreadCapture += (this.mid - price) * signed * lots;
    this.markouts.onFill(stepIndex, signed, price, Math.max(this.spread / 2, this.cfg.tickSize / 2));
    const f: Fill = { side, price, lots, fee, toxic };
    this.tape.push({ t: Date.now(), price, lots, side, ours: true });
    if (this.tape.length > 120) this.tape.shift();
    return f;
  }

  /** Aggressive (taker) execution for hedging / manual trading. */
  take(side: "buy" | "sell", lots: number): Fill {
    const price = side === "buy" ? this.top.ask : this.top.bid;
    const signed = side === "buy" ? 1 : -1;
    const notional = Math.abs(price * lots * this.cfg.lotSize);
    const fee = notional * this.cfg.takerFee;
    this.position += signed * lots;
    this.cash -= signed * price * lots;
    this.feesPaid += fee;
    this.hedgeFees += fee;
    this.cash -= fee;
    this.tape.push({ t: Date.now(), price, lots, side: side === "buy" ? "ask" : "bid", ours: true });
    if (this.tape.length > 120) this.tape.shift();
    return { side: side === "buy" ? "ask" : "bid", price, lots, fee, toxic: false };
  }

  /** Inventory PnL since the last call (mark-to-market drift). */
  inventoryPnl(): number {
    const m = this.position * this.mid;
    const d = m - this.prevMark;
    this.prevMark = m;
    return d;
  }

  attribution(): PnlAttribution {
    const total = this.equity;
    const inv = this.position * this.mid - this.markSum;
    return {
      spreadCapture: this.spreadCapture,
      inventory: inv,
      fees: -this.feesPaid,
      hedgeCost: -this.hedgeFees,
      total,
    };
  }
}

function roundTick(x: number, tick: number): number {
  return Math.round(x / tick) * tick;
}

function clampNum(x: number, lo: number, hi: number): number {
  return Math.min(Math.max(x, lo), hi);
}
