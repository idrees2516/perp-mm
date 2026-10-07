// Black–Scholes–Merton: price, full greeks (delta, gamma, vega, theta,
// rho, vanna, volga, dV/dK) and an implied-volatility solver
// (Newton with bisection safeguard, arbitrage-bound rejection).
// Faithful port of the Rust `vol::greeks` + `vol::solver`.

import { ncdf, npdf, clamp } from "./math";

export type Kind = "call" | "put";

export interface FullGreeks {
  price: number;
  delta: number;
  gamma: number;
  /** Per 1.0 sigma (per unit vol), like the Rust crate. */
  vega: number;
  /** Per year. */
  theta: number;
  rho: number;
  /** d²V/dS dσ. */
  vanna: number;
  /** d²V/dσ² = vega·d1·d2/σ. */
  volga: number;
  /** dV/dK. */
  dk: number;
}

function d1d2(s: number, k: number, r: number, q: number, sigma: number, t: number): [number, number] {
  const sq = sigma * Math.sqrt(t);
  const d1 = (Math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / sq;
  return [d1, d1 - sq];
}

export function price(kind: Kind, s: number, k: number, r: number, q: number, sigma: number, t: number): number {
  if (t <= 0.0 || sigma <= 0.0) {
    return kind === "call"
      ? Math.max(s * Math.exp(-q * t) - k * Math.exp(-r * t), 0.0)
      : Math.max(k * Math.exp(-r * t) - s * Math.exp(-q * t), 0.0);
  }
  const [d1, d2] = d1d2(s, k, r, q, sigma, t);
  const eq = Math.exp(-q * t);
  const er = Math.exp(-r * t);
  return kind === "call"
    ? s * eq * ncdf(d1) - k * er * ncdf(d2)
    : k * er * ncdf(-d2) - s * eq * ncdf(-d1);
}

export function fullGreeks(kind: Kind, s: number, k: number, r: number, q: number, sigma: number, t: number): FullGreeks {
  if (t <= 0.0 || sigma <= 0.0) {
    return { price: price(kind, s, k, r, q, sigma, t), delta: 0, gamma: 0, vega: 0, theta: 0, rho: 0, vanna: 0, volga: 0, dk: 0 };
  }
  const [d1, d2] = d1d2(s, k, r, q, sigma, t);
  const eq = Math.exp(-q * t);
  const er = Math.exp(-r * t);
  const sq = sigma * Math.sqrt(t);
  const pdf1 = npdf(d1);
  const isCall = kind === "call";
  const px = isCall
    ? s * eq * ncdf(d1) - k * er * ncdf(d2)
    : k * er * ncdf(-d2) - s * eq * ncdf(-d1);
  const delta = isCall ? eq * ncdf(d1) : eq * (ncdf(d1) - 1.0);
  const dk = isCall ? -er * ncdf(d2) : er * ncdf(-d2);
  const theta = isCall
    ? -s * eq * pdf1 * sigma / (2 * Math.sqrt(t)) - r * k * er * ncdf(d2) + q * s * eq * ncdf(d1)
    : -s * eq * pdf1 * sigma / (2 * Math.sqrt(t)) + r * k * er * ncdf(-d2) - q * s * eq * ncdf(-d1);
  const rho = isCall ? k * t * er * ncdf(d2) : -k * t * er * ncdf(-d2);
  const vega = s * eq * pdf1 * Math.sqrt(t);
  return {
    price: px,
    delta,
    gamma: (eq * pdf1) / (s * sq),
    vega,
    theta,
    rho,
    vanna: (-eq * pdf1 * d2) / sigma,
    volga: (vega * d1 * d2) / sigma,
    dk,
  };
}

/**
 * Implied volatility from an option premium.
 * Newton on the Vega with a bisection safeguard (converges in ~4–6
 * iterations for realistic quotes; rejects no-arbitrage-violating
 * premiums by returning NaN). Mirrors the Rust `vol::solver::implied_vol`.
 */
export function impliedVol(kind: Kind, s: number, k: number, r: number, q: number, t: number, premium: number): number {
  if (t <= 0 || !isFinite(premium)) return NaN;
  const eq = Math.exp(-q * t);
  const er = Math.exp(-r * t);
  const fwd = s * eq / er;
  const intrinsic = kind === "call"
    ? Math.max(fwd * er - k * er, 0)
    : Math.max(k * er - fwd * er, 0);
  const disc = er;
  if (premium < intrinsic - 1e-12 || premium > (kind === "call" ? fwd * eq : k) * disc + 1e-12) return NaN;
  if (Math.abs(premium - intrinsic) < 1e-12) return 0.0;
  let lo = 1e-6;
  let hi = 5.0;
  // Newton from a decent start
  let sig = clamp(Math.sqrt(Math.abs(Math.log(k / fwd)) / t + 0.04), 0.05, 3.0);
  for (let i = 0; i < 60; i++) {
    const p = price(kind, s, k, r, q, sig, t);
    const diff = p - premium;
    if (Math.abs(diff) < 1e-12) return sig;
    const g = fullGreeks(kind, s, k, r, q, sig, t).vega;
    if (g > 1e-14) {
      const next = sig - diff / g;
      if (next > lo && next < hi) {
        sig = next;
        continue;
      }
    }
    // bisection fallback
    const mid = 0.5 * (lo + hi);
    const pm = price(kind, s, k, r, q, mid, t);
    if (pm > premium) hi = mid;
    else lo = mid;
    sig = 0.5 * (lo + hi);
    if (hi - lo < 1e-12) return sig;
  }
  return sig;
}

/**
 * Strike from a target delta (forward-moneyness inversion) — used to
 * build the 25-delta hedge legs for the vanna–volga overhedge.
 */
export function strikeFromDelta(kind: Kind, s: number, r: number, q: number, sigma: number, t: number, targetDelta: number): number {
  // delta = eq*N(d1) (call). Invert N(d1) then solve for K.
  const eq = Math.exp(-q * t);
  const nd1 = clamp(kind === "call" ? targetDelta / eq : targetDelta / eq + 1.0, 1e-9, 1 - 1e-9);
  // invert N by bisection on d1
  let lo = -8;
  let hi = 8;
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    if (ncdf(mid) < nd1) lo = mid;
    else hi = mid;
  }
  const d1 = 0.5 * (lo + hi);
  const sq = sigma * Math.sqrt(t);
  const d2 = d1 - sq;
  // from d2: ln(F/K) = d2·σ√T + σ²T/2, F = S·e^{(r−q)T}
  const lnFK = d2 * sq + 0.5 * sigma * sigma * t;
  const F = s * Math.exp((r - q) * t);
  return F * Math.exp(-lnFK);
}
