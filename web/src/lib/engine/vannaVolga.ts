// Vanna–Volga overhedge for option quoting — NEW depth item
// (Castagna & Mercurio 2007, "The vanna-volga method for implied
// volatilities"; Wystup, "Vanna-Volga and Market Data").
//
// The vega-approximation option MM (Bergault–Guéant) quotes around a
// fair IV with a GLFT vol spread, but ignores the second-order smile
// risk each fill adds: vanna (d²V/dS dσ) and volga (d²V/dσ²). The
// vanna–volga method charges the cost of hedging that exposure with the
// liquid wing instruments:
//
//   - 25Δ risk reversal  RR = C(K_c, σ_c) − P(K_p, σ_p)   (hedges vanna)
//   - 25Δ butterfly      BF = C(K_c, σ_c) + P(K_p, σ_p) − 2·ATM  (hedges volga)
//
// For a target option X with greeks (vanna_X, volga_X), solve the 2×2
// system [vanna_rr vanna_bf; volga_rr volga_bf]·[w_rr; w_bf] =
// [vanna_X; volga_X]; the overhedge premium is λ·(w_rr·ΔV_rr +
// w_bf·ΔV_bf) where ΔV_i is instrument i's smile premium over the flat
// ATM vol and λ ∈ (0, 1] is the survival-probability damping (the
// classic Castagna damping; here exponential in T with a tunable rate).
// Translated to vol units via the option's own vega, this shifts the
// fair IV before GLFT distances are applied — an inventory-independent
// second-order skew adjustment on top of the vega-approximation quotes.

import { Kind, price, fullGreeks, strikeFromDelta } from "./bsm";
import { SsviSurface } from "./ssvi";
import { clamp } from "./math";

/** Overhedge vol-shift clamp: the classic market overhedge is a few
 *  ticks to ~0.15 vol points — never a first-order smile move. */
const VV_VOL_CLAMP = 0.0015;

export interface Wings {
  kAtm: number;
  kC25: number;
  kP25: number;
  /** Smile vols at the wing strikes. */
  sigC: number;
  sigP: number;
  sigAtm: number;
  /** ATM leg greeks (zero smile premium — prices itself). */
  atm: { vega: number; vanna: number; volga: number; cost: number };
  /** RR / BF instrument greeks (each leg at its own smile vol). */
  rr: { vega: number; vanna: number; volga: number; cost: number };
  bf: { vega: number; vanna: number; volga: number; cost: number };
  /** Market quotes of the structures (in IV points, for the panel). */
  rrIv: number;
  bfIv: number;
}

export interface OverhedgeResult {
  /** Vol-point shift of the fair IV. */
  volShift: number;
  /** Hedge weights (RR, BF). */
  wRr: number;
  wBf: number;
  /** Premium charge in price units (damped). */
  charge: number;
}

export class VannaVolga {
  wings: Wings;
  /** Damping rate (1/years); λ = exp(−rate·T), rate = 0 → full charge. */
  dampingRate: number;

  constructor(
    public s: number,
    public t: number,
    public surface: SsviSurface,
    public r = 0,
    public q = 0,
    dampingRate = 0.25,
  ) {
    this.dampingRate = dampingRate;
    this.wings = this.buildWings();
  }

  private buildWings(): Wings {
    const { s, t, surface, r, q } = this;
    const sigAtm = surface.iv(0, t);
    const kAtm = strikeFromDelta("call", s, r, q, sigAtm, t, 0.5);
    const kC25 = strikeFromDelta("call", s, r, q, sigAtm, t, 0.25);
    const kP25 = strikeFromDelta("put", s, r, q, sigAtm, t, -0.25);
    const kc = Math.log(kC25 / s);
    const kp = Math.log(kP25 / s);
    const sigC = surface.iv(kc, t);
    const sigP = surface.iv(kp, t);
    // Greeks of each leg at its own smile vol.
    const gC = fullGreeks("call", s, kC25, r, q, sigC, t);
    const gP = fullGreeks("put", s, kP25, r, q, sigP, t);
    const gA = fullGreeks("call", s, kAtm, r, q, sigAtm, t);
    // Smile premiums over the flat ATM vol.
    const dvC = price("call", s, kC25, r, q, sigC, t) - price("call", s, kC25, r, q, sigAtm, t);
    const dvP = price("put", s, kP25, r, q, sigP, t) - price("put", s, kP25, r, q, sigAtm, t);
    return {
      kAtm,
      kC25,
      kP25,
      sigC,
      sigP,
      sigAtm,
      atm: { vega: gA.vega, vanna: gA.vanna, volga: gA.volga, cost: 0 },
      rr: {
        vega: gC.vega - gP.vega,
        vanna: gC.vanna - gP.vanna,
        volga: gC.volga - gP.volga,
        cost: dvC - dvP,
      },
      bf: {
        vega: gC.vega + gP.vega - 2 * gA.vega,
        vanna: gC.vanna + gP.vanna - 2 * gA.vanna,
        volga: gC.volga + gP.volga - 2 * gA.volga,
        cost: dvC + dvP,
      },
      rrIv: (sigC - sigP) * 100,
      bfIv: 0.5 * (sigC + sigP - 2 * sigAtm) * 100,
    };
  }

  private lambda(): number {
    return Math.exp(-this.dampingRate * this.t);
  }

  /**
   * Overhedge vol shift for an option (kind, strike) whose smile-fair
   * vol is `sigK`.
   *
   * Castagna–Mercurio construction: V_vv(X) = V_bs(X, σ_atm) +
   * w_rr·ΔV_rr + w_bf·ΔV_bf vs the surface-consistent price
   * V_surf(X) = V_bs(X, σ_k). The overhedge charges the RESIDUAL
   * w_rr·ΔV_rr + w_bf·ΔV_bf − ΔV_X — zero when the surface is
   * vanna–volga consistent, a few ticks of premium when it is not.
   */
  overhedge(kind: Kind, strike: number, sigK: number): OverhedgeResult {
    const { s, t, r, q } = this;
    const g = fullGreeks(kind, s, strike, r, q, sigK, t);
    if (Math.abs(g.vanna) < 1e-12 && Math.abs(g.volga) < 1e-12) {
      return { volShift: 0, wRr: 0, wBf: 0, charge: 0 };
    }
    const { rr, bf } = this.wings;
    // The classic three-instrument vanna–volga portfolio (Castagna–
    // Mercurio): ATM + 25Δ RR + 25Δ BF solve the full 3×3 system in
    // (vega, vanna, volga) — under which the ATM prices itself exactly.
    const w = solve3x3(
      [this.wings.atm.vega, this.wings.atm.vanna, this.wings.atm.volga],
      [rr.vega, rr.vanna, rr.volga],
      [bf.vega, bf.vanna, bf.volga],
      [g.vega, g.vanna, g.volga],
    );
    if (!w) {
      return { volShift: 0, wRr: 0, wBf: 0, charge: 0 };
    }
    const wRr = w[1];
    const wBf = w[2];
    // the target's own smile premium over the flat ATM vol
    const dvX = price(kind, s, strike, r, q, sigK, t) - price(kind, s, strike, r, q, this.wings.sigAtm, t);
    const lam = this.lambda();
    const charge = lam * (wRr * rr.cost + wBf * bf.cost - dvX);
    const volShift = clamp(charge / g.vega, -VV_VOL_CLAMP, VV_VOL_CLAMP);
    return { volShift, wRr, wBf, charge };
  }

  /**
   * Portfolio-level overhedge: net (vanna, volga) of the whole book
   * charged at the wing-instrument smile costs. Returns the vol-point
   * shift that would be applied at the ATM vega centroid, and the
   * dollar charge.
   */
  portfolioCharge(netVanna: number, netVolga: number, atmVega: number): { charge: number; volShift: number; wRr: number; wBf: number } {
    const { rr, bf } = this.wings;
    const w = solve3x3(
      [this.wings.atm.vega, this.wings.atm.vanna, this.wings.atm.volga],
      [rr.vega, rr.vanna, rr.volga],
      [bf.vega, bf.vanna, bf.volga],
      [atmVega, netVanna, netVolga],
    );
    if (!w || atmVega <= 1e-12) return { charge: 0, volShift: 0, wRr: 0, wBf: 0 };
    const lam = this.lambda();
    const charge = lam * (w[1] * rr.cost + w[2] * bf.cost);
    const volShift = clamp(charge / atmVega, -VV_VOL_CLAMP * 4, VV_VOL_CLAMP * 4);
    return { charge, volShift, wRr: w[1], wBf: w[2] };
  }
}

/** Solve the 3×3 (vega, vanna, volga) system by Cramer's rule;
 *  null when singular. Columns = instruments, rhs = target. */
function solve3x3(
  atm: [number, number, number],
  rr: [number, number, number],
  bf: [number, number, number],
  target: [number, number, number],
): [number, number, number] | null {
  const det3 = (a: number[], b: number[], c: number[]): number =>
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
  const det = det3(atm, rr, bf);
  if (Math.abs(det) < 1e-14) return null;
  const wAtm = det3(target, rr, bf) / det;
  const wRr = det3(atm, target, bf) / det;
  const wBf = det3(atm, rr, target) / det;
  if (![wAtm, wRr, wBf].every((x) => isFinite(x))) return null;
  return [wAtm, wRr, wBf];
}
