// Quoting models: GLFT asymptotics, the exact HJB solver, AS unified
// quotes, the vega-approximation option MM and the Whalley–Wilmott hedge
// band. Ports of the Rust `models::{glft, hjb, unified}` + `vol::optmm`.

import { clamp } from "./math";
import { SsviSurface } from "./ssvi";
import { Kind, price, fullGreeks } from "./bsm";

// ---------------------------------------------------------------- GLFT ----

/** Guéant–Lehalle–Fernández-Tapia (2013) asymptotic quote distances. */
export class GlftAsymptotic {
  constructor(
    public gamma: number,
    public sigma: number,
    public kappa: number,
    public a: number,
  ) {}

  /** Skew coefficient sqrt((γσ²/(2Aκ))(1+γ/κ)^{κ/γ+1}). */
  skewCoef(): number {
    const base = (this.gamma * this.sigma * this.sigma) / (2 * this.a * this.kappa);
    const pow = Math.pow(1 + this.gamma / this.kappa, this.kappa / this.gamma + 1);
    return Math.sqrt(base * pow);
  }

  halfIntensityTerm(): number {
    return (1 / this.gamma) * Math.log(1 + this.gamma / this.kappa);
  }

  deltaBid(q: number): number {
    return this.halfIntensityTerm() + ((2 * q + 1) / 2) * this.skewCoef();
  }

  deltaAsk(q: number): number {
    return this.halfIntensityTerm() - ((2 * q - 1) / 2) * this.skewCoef();
  }

  spread(): number {
    return 2 * this.halfIntensityTerm() + this.skewCoef();
  }
}

// ----------------------------------------------------------------- HJB ----

export interface MmProblem {
  gamma: number;
  sigma: number;
  kappa: number;
  a: number;
  t: number;
  qMax: number;
  kappaLiq: number;
  impactBeta: number;
  nSteps: number;
}

export function defaultProblem(gamma: number, sigma: number, kappa: number, a: number, t: number): MmProblem {
  return { gamma, sigma, kappa, a, t, qMax: 10, kappaLiq: 0, impactBeta: 0, nSteps: 400 };
}

/** Impulse maximizer values of the reduced v-system. */
function impulseAsk(p: MmProblem, vq: number, vprev: number, q: number): number {
  if (vq <= 0 || vprev <= 0) return 0;
  const foc = (p.impactBeta * (q - 1)) + (1 / p.gamma) * Math.log(((p.kappa + p.gamma) * vprev) / (p.kappa * vq));
  const d = Math.max(foc, 0);
  return p.a * Math.exp(-p.kappa * d) * (vq - Math.exp(-p.gamma * d) * vprev);
}

function impulseBid(p: MmProblem, vq: number, vnext: number, q: number): number {
  if (vq <= 0 || vnext <= 0) return 0;
  const foc = -p.impactBeta * (q + 1) + (1 / p.gamma) * Math.log(((p.kappa + p.gamma) * vnext) / (p.kappa * vq));
  const d = Math.max(foc, 0);
  return p.a * Math.exp(-p.kappa * d) * (vq - Math.exp(-p.gamma * d) * vnext);
}

function rhs(p: MmProblem, v: number[]): number[] {
  const qDim = v.length;
  const out = new Array<number>(qDim).fill(0);
  for (let qi = 0; qi < qDim; qi++) {
    const q = qi - p.qMax;
    let acc = -0.5 * p.gamma * p.gamma * p.sigma * p.sigma * q * q * v[qi];
    if (q > -p.qMax) acc += impulseAsk(p, v[qi], v[qi - 1], q);
    if (q < p.qMax) acc += impulseBid(p, v[qi], v[qi + 1], q);
    out[qi] = acc;
  }
  return out;
}

/** Exact HJB solution: RK4 backward march on the reduced v-system. */
export class MmHjb {
  p: MmProblem;
  times: number[];
  /** v[i][q + Q]. */
  v: number[][];

  constructor(p: MmProblem) {
    this.p = p;
    const qDim = 2 * p.qMax + 1;
    const n = Math.max(p.nSteps, 20);
    const dt = p.t / n;
    const times: number[] = [];
    for (let i = 0; i <= n; i++) times.push((i * dt));
    times[n] = p.t;
    const v: number[][] = [];
    for (let i = 0; i <= n; i++) v.push(new Array<number>(qDim).fill(1));
    for (let qi = 0; qi < qDim; qi++) {
      const q = qi - p.qMax;
      v[n][qi] = Math.exp(p.gamma * (p.kappaLiq * q * q) / 2);
    }
    for (let i = n - 1; i >= 0; i--) {
      const cur = v[i + 1].slice();
      const k1 = rhs(p, cur);
      const tmp = cur.map((x, j) => x - 0.5 * dt * k1[j]);
      const k2 = rhs(p, tmp);
      const tmp2 = cur.map((x, j) => x - 0.5 * dt * k2[j]);
      const k3 = rhs(p, tmp2);
      const tmp3 = cur.map((x, j) => x - dt * k3[j]);
      const k4 = rhs(p, tmp3);
      v[i] = cur.map((x, j) => {
        const nv = x - (dt / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
        return isFinite(nv) && nv > 0 ? nv : 1e-12;
      });
    }
    this.times = times;
    this.v = v;
  }

  rowAt(t: number): number[] {
    const clamped = clamp(t, 0, this.p.t);
    const n = this.times.length - 1;
    const dt = this.p.t / n;
    const raw = dt > 0 ? clamped / dt : 0;
    const pos = Math.max(0, Math.min(n, Math.floor(raw)));
    const frac = dt > 0 ? clamped / dt - pos : 0;
    const i0 = Math.min(pos, n);
    const i1 = Math.min(pos + 1, n);
    return this.v[i0].map((x, j) => x * (1 - frac) + this.v[i1][j] * frac);
  }

  vAt(q: number, t: number): number {
    const row = this.rowAt(t);
    return row[clamp(q + this.p.qMax, 0, row.length - 1)];
  }

  deltaAsk(q: number, t: number): number {
    const p = this.p;
    if (q <= -p.qMax) return Infinity;
    const vq = this.vAt(q, t);
    const vprev = this.vAt(q - 1, t);
    const c = p.impactBeta * (q - 1);
    const foc = c + (1 / p.gamma) * Math.log(((p.kappa + p.gamma) * vprev) / (p.kappa * vq));
    return Math.max(foc, 0);
  }

  deltaBid(q: number, t: number): number {
    const p = this.p;
    if (q >= p.qMax) return Infinity;
    const vq = this.vAt(q, t);
    const vnext = this.vAt(q + 1, t);
    const c = p.impactBeta * (q + 1);
    const foc = -c + (1 / p.gamma) * Math.log(((p.kappa + p.gamma) * vnext) / (p.kappa * vq));
    return Math.max(foc, 0);
  }

  quotes(s: number, q: number, t: number): [number, number] {
    const da = this.deltaAsk(q, t);
    const db = this.deltaBid(q, t);
    return [isFinite(db) ? s - db : -Infinity, isFinite(da) ? s + da : Infinity];
  }
}

// -------------------------------------------------------- AS unified ----

export interface AsParams {
  gamma: number;
  sigma: number;
  kappa: number;
  a: number;
  t: number;
}

export function asReservation(p: AsParams, s: number, q: number, t: number): number {
  return s - p.gamma * p.sigma * p.sigma * q * (p.t - t);
}

export function asHalfSpread(p: AsParams, t: number): number {
  return 0.5 * p.gamma * p.sigma * p.sigma * (p.t - t) + (1 / p.gamma) * Math.log(1 + p.gamma / p.kappa);
}

export function asQuotes(p: AsParams, s: number, q: number, t: number): [number, number] {
  const r = asReservation(p, s, q, t);
  const h = asHalfSpread(p, t);
  return [r - h, r + h];
}

// ------------------------------------------------- option MM (vega-G) ----

export interface OptMmParams {
  gamma: number;
  /** Vol-of-vol per sqrt(second). */
  sigmaVol: number;
  kappaV: number;
  aV: number;
  /** Vega traded per request ($ per unit vol). */
  vegaPerFill: number;
  /** Per-fill execution cost in vol units. */
  fillCostVol: number;
}

export class OptMm {
  constructor(public p: OptMmParams) {}

  private glft(): GlftAsymptotic {
    return new GlftAsymptotic(this.p.gamma, this.p.vegaPerFill * this.p.sigmaVol, this.p.kappaV, this.p.aV);
  }

  qFills(netVega: number): number {
    return Math.round(netVega / this.p.vegaPerFill);
  }

  /** Optimal vol distances (bid, ask) at net vega inventory V. */
  volDistances(netVega: number): [number, number] {
    const g = this.glft();
    const q = this.qFills(netVega);
    return [
      Math.max(g.deltaBid(q) + this.p.fillCostVol, 1e-6),
      Math.max(g.deltaAsk(q) + this.p.fillCostVol, 1e-6),
    ];
  }

  quoteIvs(fairIv: number, netVega: number): [number, number] {
    const [db, da] = this.volDistances(netVega);
    return [fairIv - db, fairIv + da];
  }

  quotePremiums(kind: Kind, s: number, k: number, t: number, surface: SsviSurface, netVega: number): [number, number] {
    const fair = surface.iv(Math.log(k / s), t);
    const [ivB, ivA] = this.quoteIvs(fair, netVega);
    return [price(kind, s, k, 0, 0, ivB, t), price(kind, s, k, 0, 0, ivA, t)];
  }

  /** Net book greeks (delta, gamma, vega) for positions [kind, strike, lots]. */
  bookGreeks(positions: Array<[Kind, number, number]>, s: number, t: number, surface: SsviSurface): [number, number, number] {
    let delta = 0;
    let gamma = 0;
    let vega = 0;
    for (const [kind, k, lots] of positions) {
      const iv = surface.iv(Math.log(k / s), t);
      const g = fullGreeks(kind, s, k, 0, 0, iv, t);
      delta += g.delta * lots;
      gamma += g.gamma * lots;
      vega += g.vega * lots;
    }
    return [delta, gamma, vega];
  }
}

// -------------------------------------------------- Whalley–Wilmott ------

/**
 * Delta-hedge no-trade band, closed form `H* = (3c/(γS²))^{1/3}`
 * (re-derived in the Rust crate and MC-validated: the loss-rate hump).
 */
export class HedgeCadence {
  bandLots: number;

  constructor(costPerLot: number, gamma: number, pricePerLot: number) {
    const c = Math.max(costPerLot, 0);
    const g = Math.max(gamma, 1e-9);
    const s = Math.max(pricePerLot, 1e-9);
    this.bandLots = Math.pow((3 * c) / (g * s * s), 1 / 3);
  }

  /**
   * Signed hedge lots if outside the band — trades back to the band
   * EDGE (the classic reflected-band control), not to flat: the
   * regression the Rust engine's validation caught (full flatten per
   * trigger churns hedge volume ~2× the band width).
   */
  shouldHedge(unhedged: number): number | null {
    if (Math.abs(unhedged) < this.bandLots) return null;
    const edge = Math.sign(unhedged) * this.bandLots;
    const excess = -(unhedged - edge);
    return Math.abs(excess) >= 1 ? Math.round(excess) : null;
  }
}
