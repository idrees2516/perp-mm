// Simulated option market: multi-expiry chain on an arbitrage-free SSVI
// surface whose ATM pillars are driven by a vol-of-vol process, client
// requests arriving at Cox intensity a_v·e^{−κ_v·δ} on the vol spread
// (Bergault–Guéant request model), fills at our quoted IVs, per-leg
// vanna–volga overhedge on top of the vega-approximation GLFT quotes,
// net book greeks and Whalley–Wilmott-band delta hedging.

import { Rng, clamp } from "./math";
import { SsviSurface } from "./ssvi";
import { Kind, price, fullGreeks } from "./bsm";
import { OptMm } from "./models";
import { VannaVolga } from "./vannaVolga";

export interface OptionConfig {
  /** Expiries in years. */
  expiries: number[];
  /** Moneyness ladder shared by each expiry. */
  moneyness: number[];
  kinds: Kind[];
  /** Client request intensity at zero vol spread (per second). */
  aV: number;
  /** Client decay vs vol spread. */
  kappaV: number;
  /** Lots per request. */
  lotsPerRequest: number;
  /** Annualized vol-of-vol driving ATM IVs. */
  volOfVol: number;
  /** SSVI shape. */
  rho: number;
  eta: number;
  gamma: number;
  /** Per-leg position bound (lots). */
  maxLots: number;
  /** Vega limit ($ per unit vol) for the risk engine. */
  vegaLimit: number;
  /** Vanna–volga damping rate. */
  vvDamping: number;
  /** Enable the vanna–volga overhedge. */
  useVv: boolean;
  /** Contract multiplier (underlying units per option lot). */
  multiplier: number;
}

export const defaultOptionConfig: OptionConfig = {
  expiries: [7 / 365, 30 / 365, 90 / 365, 180 / 365, 365 / 365],
  moneyness: [-0.08, -0.04, -0.02, 0, 0.02, 0.04, 0.08],
  kinds: ["call", "put"],
  aV: 0.02,
  kappaV: 55,
  lotsPerRequest: 1,
  volOfVol: 0.9,
  rho: -0.7,
  eta: 1.0,
  gamma: 0.5,
  maxLots: 60,
  vegaLimit: 4000,
  vvDamping: 0.25,
  useVv: true,
  multiplier: 1,
};

export interface LegQuotes {
  ivBid: number;
  ivAsk: number;
  ivFair: number;
  /** Vanna–volga overhedge shift applied to the fair. */
  vvShift: number;
  premBid: number;
  premAsk: number;
}

export interface OptionPosition {
  expIdx: number;
  legIdx: number;
  kind: Kind;
  lots: number;
}

export interface OptionFill {
  expIdx: number;
  legIdx: number;
  kind: Kind;
  side: "bid" | "ask";
  lots: number;
  iv: number;
  fairIv: number;
  premium: number;
  vvShift: number;
}

/** Directional risk gate for option client flow. The risk-REDUCING side
 *  is never blocked: buying back a short leg / selling a long leg always
 *  stays quotable — a risk engine that blocks its own unwind path turns a
 *  limit breach into a permanent freeze. */
export interface RiskGate {
  /** Long vega over the limit: our vol-bid is pulled (no more vega buys). */
  blockVolBid: boolean;
  /** Short vega over the limit: our vol-ask is pulled (no more vega sells). */
  blockVolAsk: boolean;
  /** Gross hedge-cap breach: wind the book down — only fills that reduce
   *  the |per-leg position| are served; flat legs stop filling. */
  windDown: boolean;
}

export interface NetGreeks {
  delta: number;
  gamma: number;
  vega: number;
  theta: number;
  vanna: number;
  volga: number;
  /** Mark-to-market of the book. */
  mark: number;
}

const YEAR_SECS = 365 * 24 * 3600;

export class OptionMarket {
  /** ATM IV per expiry (annualized). */
  atm: number[];
  /** Term-structure shape noise per expiry. */
  termNoise: number[];
  /** Position per [expiry][leg][kind] in lots. */
  pos: number[][][];
  cash = 0;
  fills = 0;
  vvChargeTotal = 0;
  lastVvShifts: number[][] = [];
  lastWings: Array<{ t: number; rrIv: number; bfIv: number; sigAtm: number }> = [];
  hedgeLots = 0;

  constructor(public cfg: OptionConfig, private rng: Rng, initialAtm = 0.55) {
    const n = cfg.expiries.length;
    this.atm = cfg.expiries.map((t, i) => clamp(initialAtm * (1 + 0.06 * Math.log(t * 12 + 1.0) + 0.01 * i), 0.08, 2.5));
    this.termNoise = new Array(n).fill(0);
    // pos[expiry][leg][kindIdx: 0=call, 1=put] in lots
    this.pos = cfg.expiries.map(() => cfg.moneyness.map(() => [0, 0]));
    this.lastVvShifts = cfg.expiries.map(() => cfg.moneyness.map(() => 0));
  }

  /** The current surface (shared vol-of-vol factor + term structure). */
  surface(): SsviSurface {
    const pillars = this.cfg.expiries.map((t, i) => [t, this.atm[i]] as [number, number]);
    return new SsviSurface(this.cfg.rho, this.cfg.eta, this.cfg.gamma, pillars);
  }

  /** Advance ATM vols one step (correlated vol-of-vol). */
  step(dt: number) {
    const common = this.rng.normal() * 0.7;
    const vvPerSec = this.cfg.volOfVol / Math.sqrt(YEAR_SECS);
    for (let i = 0; i < this.atm.length; i++) {
      const z = common + this.rng.normal() * 0.45;
      const noise = this.rng.normal() * 0.02;
      this.termNoise[i] = 0.98 * this.termNoise[i] + noise * Math.sqrt(dt) * 0.05;
      this.atm[i] = clamp(
        this.atm[i] * Math.exp(vvPerSec * Math.sqrt(dt) * z + this.termNoise[i] * 0.001),
        0.08,
        2.5,
      );
    }
  }

  kindIdx(kind: Kind): number {
    return kind === "call" ? 0 : 1;
  }

  /** Net book greeks against the surface. */
  netGreeks(s: number, surface: SsviSurface): NetGreeks {
    let delta = 0;
    let gamma = 0;
    let vega = 0;
    let theta = 0;
    let vanna = 0;
    let volga = 0;
    let mark = 0;
    for (let e = 0; e < this.cfg.expiries.length; e++) {
      const t = this.cfg.expiries[e];
      for (let l = 0; l < this.cfg.moneyness.length; l++) {
        const k = Math.exp(this.cfg.moneyness[l]) * s;
        const iv = surface.iv(this.cfg.moneyness[l], t);
        for (let ki = 0; ki < 2; ki++) {
          const lots = this.pos[e][l][ki];
          if (lots === 0) continue;
          const kind: Kind = ki === 0 ? "call" : "put";
          const g = fullGreeks(kind, s, k, 0, 0, iv, t);
          delta += g.delta * lots;
          gamma += g.gamma * lots;
          vega += g.vega * lots;
          theta += g.theta * lots;
          vanna += g.vanna * lots;
          volga += g.volga * lots;
          mark += g.price * lots;
        }
      }
    }
    return { delta, gamma, vega, theta, vanna, volga, mark };
  }

  /** Per-expiry net vega ($ per unit vol) — the term-structure buckets
   *  the per-pillar quotes lean against. */
  vegaPerExpiry(s: number, surface: SsviSurface): number[] {
    const out: number[] = [];
    for (let e = 0; e < this.cfg.expiries.length; e++) {
      const t = this.cfg.expiries[e];
      let vegaE = 0;
      for (let l = 0; l < this.cfg.moneyness.length; l++) {
        const k = Math.exp(this.cfg.moneyness[l]) * s;
        const iv = surface.iv(this.cfg.moneyness[l], t);
        for (let ki = 0; ki < 2; ki++) {
          const lots = this.pos[e][l][ki];
          if (lots === 0) continue;
          vegaE += fullGreeks(ki === 0 ? "call" : "put", s, k, 0, 0, iv, t).vega * lots;
        }
      }
      out.push(vegaE);
    }
    return out;
  }

  /** Per-leg marks for the SFPM scenario grid (portfolio margin). */
  marginLegs(s: number, surface: SsviSurface): Array<{ kind: Kind; strike: number; iv: number; t: number; lots: number }> {
    const out: Array<{ kind: Kind; strike: number; iv: number; t: number; lots: number }> = [];
    for (let e = 0; e < this.cfg.expiries.length; e++) {
      const t = this.cfg.expiries[e];
      for (let l = 0; l < this.cfg.moneyness.length; l++) {
        const k = Math.exp(this.cfg.moneyness[l]) * s;
        const iv = surface.iv(this.cfg.moneyness[l], t);
        for (let ki = 0; ki < 2; ki++) {
          const lots = this.pos[e][l][ki];
          if (lots === 0) continue;
          out.push({ kind: ki === 0 ? "call" : "put", strike: k, iv, t, lots });
        }
      }
    }
    return out;
  }

  /** Governed-surface anchor (Paradex shape): an executed market touch
   *  EWMA's its ATM pillar toward the touched IV, with a clamped move and
   *  bounded total drift — the surface follows the market, it does not
   *  chase it. */
  anchorAtm(expIdx: number, iv: number, weight = 0.1) {
    if (expIdx < 0 || expIdx >= this.atm.length) return;
    const cur = this.atm[expIdx];
    const move = clamp(Math.log(Math.max(iv, 0.08) / cur) * weight, -0.02, 0.02);
    this.atm[expIdx] = clamp(cur * Math.exp(move), 0.08, 2.5);
  }

  /**
   * Quotes for one (expiry, leg, kind): fair IV from the surface,
   * vanna–volga overhedge shift, GLFT vega distances around it.
   * `netVega` is the PER-EXPIRY net vega inventory — the GLFT skew
   * then leans each pillar against its own term-structure bucket
   * (eSSVI-style surface-level inventory control) instead of blinding
   * the whole chain with the aggregate.
   */
  quotes(
    e: number,
    l: number,
    kind: Kind,
    s: number,
    surface: SsviSurface,
    optMm: OptMm,
    netVega: number,
    vv: VannaVolga | null,
  ): LegQuotes {
    const t = this.cfg.expiries[e];
    const k = Math.exp(this.cfg.moneyness[l]) * s;
    const fair = surface.iv(this.cfg.moneyness[l], t);
    let vvShift = 0;
    if (vv && this.cfg.useVv) {
      const o = vv.overhedge(kind, k, fair);
      // scale: overhedge is per 1 lot of vega exposure — expressed in vol
      // points it is small; scale by request size for the charge.
      vvShift = o.volShift;
    }
    const centered = Math.max(0.03, fair + vvShift);
    const [ivB, ivA] = optMm.quoteIvs(centered, netVega);
    return {
      ivBid: ivB,
      ivAsk: ivA,
      ivFair: fair,
      vvShift,
      premBid: price(kind, s, k, 0, 0, ivB, t),
      premAsk: price(kind, s, k, 0, 0, ivA, t),
    };
  }

  /**
   * Client flow step: requests arrive at Cox intensity on the vol spread
   * and fill at our quotes. Returns the fills.
   */
  clientStep(
    dt: number,
    s: number,
    surface: SsviSurface,
    optMm: OptMm,
    vegaPerExp: number[],
    vv: VannaVolga | null,
    gate: RiskGate = { blockVolBid: false, blockVolAsk: false, windDown: false },
  ): OptionFill[] {
    const out: OptionFill[] = [];
    const cfg = this.cfg;
    for (let e = 0; e < cfg.expiries.length; e++) {
      const t = cfg.expiries[e];
      const vegaE = vegaPerExp[e] ?? 0;
      for (let l = 0; l < cfg.moneyness.length; l++) {
        const wingFactor = Math.exp(-2.2 * Math.abs(cfg.moneyness[l]));
        for (let ki = 0; ki < 2; ki++) {
          const kind: Kind = ki === 0 ? "call" : "put";
          const q = this.quotes(e, l, kind, s, surface, optMm, vegaE, vv);
          const fair = q.ivFair;
          const db = Math.max(fair - q.ivBid, 1e-6);
          const da = Math.max(q.ivAsk - fair, 1e-6);
          // risk gate: the risk-REDUCING side is never blocked — buying
          // back a short leg (our bid) / selling a long leg (our ask)
          const posHere = this.pos[e][l][ki];
          const allowBid = !gate.blockVolBid && (!gate.windDown || posHere < 0);
          const allowAsk = !gate.blockVolAsk && (!gate.windDown || posHere > 0);
          // client hits our bid (we buy) / lifts our ask (we sell)
          const lamB = allowBid ? cfg.aV * wingFactor * Math.exp(-cfg.kappaV * db) * dt : 0;
          const lamA = allowAsk ? cfg.aV * wingFactor * Math.exp(-cfg.kappaV * da) * dt : 0;
          if (this.rng.uniform() < Math.min(lamB, 0.5)) {
            this.pushFill(out, e, l, kind, "bid", q, s, t);
          }
          if (this.rng.uniform() < Math.min(lamA, 0.5)) {
            this.pushFill(out, e, l, kind, "ask", q, s, t);
          }
        }
      }
    }
    return out;
  }

  private pushFill(out: OptionFill[], e: number, l: number, kind: Kind, side: "bid" | "ask", q: LegQuotes, s: number, t: number) {
    const cfg = this.cfg;
    const ki = this.kindIdx(kind);
    const lots = cfg.lotsPerRequest;
    const cur = this.pos[e][l][ki];
    const next = side === "bid" ? cur + lots : cur - lots;
    if (Math.abs(next) > cfg.maxLots) return; // desk limit refusal
    this.pos[e][l][ki] = next;
    const iv = side === "bid" ? q.ivBid : q.ivAsk;
    const prem = (side === "bid" ? q.premBid : q.premAsk) * lots;
    this.cash += side === "bid" ? -prem : prem;
    this.fills += 1;
    this.vvChargeTotal += q.vvShift;
    this.lastVvShifts[e][l] = q.vvShift;
    out.push({ expIdx: e, legIdx: l, kind, side, lots, iv, fairIv: q.ivFair, premium: prem, vvShift: q.vvShift });
  }

  /** Wing instrument state per expiry (for the panel). */
  wingState(surface: SsviSurface, _s: number): Array<{ t: number; rrIv: number; bfIv: number; sigAtm: number }> {
    const out: Array<{ t: number; rrIv: number; bfIv: number; sigAtm: number }> = [];
    for (let e = 0; e < this.cfg.expiries.length; e++) {
      const t = this.cfg.expiries[e];
      const sigAtm = surface.iv(0, t);
      const kC = surface.iv(0.2, t); // approximate 25Δ strikes
      const kP = surface.iv(-0.2, t);
      out.push({ t, rrIv: (kC - kP) * 100, bfIv: 0.5 * (kC + kP - 2 * sigAtm) * 100, sigAtm });
    }
    return out;
  }
}
