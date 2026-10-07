// SSVI volatility surface (Gatheral–Jacquier 2014, "Arbitrage-free SVI
// volatility surfaces") with the full static no-arbitrage audit.
// Faithful port of the Rust `vol::ssvi`.

import { clamp } from "./math";

export interface Pillar {
  /** Maturity in years. */
  t: number;
  /** ATM total variance θ = σ_atm²·T. */
  theta: number;
}

export interface ArbReport {
  butterflyCondition: boolean;
  butterflyGrid: boolean;
  calendar: boolean;
}

export class SsviSurface {
  rho: number;
  eta: number;
  gamma: number;
  /** Sorted by maturity; θ enforced nondecreasing at construction. */
  pillars: Pillar[];

  constructor(rho: number, eta: number, gamma: number, atmVols: [number, number][]) {
    this.rho = clamp(rho, -0.999, 0.999);
    this.eta = Math.max(eta, 1e-4);
    this.gamma = clamp(gamma, 1e-3, 0.999);
    let pillars: Pillar[] = atmVols
      .map(([t, s]) => ({ t, theta: s * s * t }))
      .filter((p) => p.t > 0);
    pillars.sort((a, b) => a.t - b.t);
    let runMax = 0;
    for (const p of pillars) {
      runMax = Math.max(runMax, p.theta);
      p.theta = runMax;
    }
    this.pillars = pillars;
  }

  /** Heston-like φ(θ) = 1/(η θ^γ (1+θ)^{1−γ}). */
  phi(theta: number): number {
    return 1.0 / (this.eta * Math.pow(theta, this.gamma) * Math.pow(1 + theta, 1 - this.gamma));
  }

  /** ATM total variance at maturity t (piecewise-linear; flat-vol extrapolation). */
  thetaAt(t: number): number {
    const ps = this.pillars;
    if (ps.length === 0) return 0;
    if (t <= ps[0].t) {
      return (ps[0].theta / ps[0].t) * Math.max(t, 0);
    }
    const last = ps[ps.length - 1];
    if (t >= last.t) {
      return last.theta + (last.theta / last.t) * Math.max(t - last.t, 0);
    }
    for (let i = 0; i + 1 < ps.length; i++) {
      const a = ps[i];
      const b = ps[i + 1];
      if (t >= a.t && t <= b.t) {
        const f = (t - a.t) / (b.t - a.t);
        return a.theta + f * (b.theta - a.theta);
      }
    }
    return last.theta;
  }

  /** Total variance at log-moneyness k, maturity t. */
  totalVar(k: number, t: number): number {
    const th = this.thetaAt(t);
    if (th <= 0) return 0;
    const ph = this.phi(th);
    const u = ph * k + this.rho;
    return (th / 2) * (1 + this.rho * ph * k + Math.sqrt(u * u + 1 - this.rho * this.rho));
  }

  /** Implied vol at (k, t). */
  iv(k: number, t: number): number {
    return Math.sqrt(this.totalVar(k, t) / t);
  }

  ivAtm(t: number): number {
    return Math.sqrt(this.thetaAt(t) / t);
  }

  /** First two k-derivatives of the slice. */
  dwDk(k: number, t: number): [number, number] {
    const th = this.thetaAt(t);
    const ph = this.phi(th);
    const u = ph * k + this.rho;
    const rt = Math.sqrt(u * u + 1 - this.rho * this.rho);
    const wp = (th / 2) * (this.rho * ph + (ph * u) / rt);
    const wpp = (th / 2) * ph * ph * (1 - this.rho * this.rho) * Math.pow(rt, -3);
    return [wp, wpp];
  }

  /** Gatheral–Jacquier density factor g(k) of the slice. */
  g(k: number, t: number): number {
    const w = this.totalVar(k, t);
    if (w <= 0) return NaN;
    const [wp, wpp] = this.dwDk(k, t);
    return Math.pow(1 - (k * wp) / (2 * w), 2) - (wp * wp) / 4 + wpp / 2;
  }

  /** GJ butterfly sufficient condition θφ²(1+|ρ|) ≤ 4 at every pillar. */
  butterflyCondition(): boolean {
    return this.pillars.every((p) => this.phi(p.theta) ** 2 * p.theta * (1 + Math.abs(this.rho)) <= 4 + 1e-12);
  }

  /** Direct density check g(k) ≥ 0 on a grid. */
  butterflyGrid(kmin: number, kmax: number, n: number): boolean {
    n = Math.max(n, 16);
    const ts = this.pillars.map((p) => p.t);
    if (ts.length >= 2) ts.push(0.5 * (ts[0] + ts[ts.length - 1]));
    for (const t of ts) {
      if (t <= 0) continue;
      for (let i = 0; i <= n; i++) {
        const k = kmin + ((kmax - kmin) * i) / n;
        const g = this.g(k, t);
        if (isNaN(g) || g < -1e-10) return false;
      }
    }
    return true;
  }

  /** Calendar check: w(k,T2) ≥ w(k,T1) on a grid. */
  calendarGrid(kmin: number, kmax: number, n: number): boolean {
    n = Math.max(n, 16);
    if (this.pillars.length < 2) return true;
    const ts = this.pillars.map((p) => p.t);
    ts.push(0.5 * (ts[0] + ts[ts.length - 1]));
    ts.sort((a, b) => a - b);
    for (let i = 0; i + 1 < ts.length; i++) {
      for (let j = 0; j <= n; j++) {
        const k = kmin + ((kmax - kmin) * j) / n;
        if (this.totalVar(k, ts[i + 1]) < this.totalVar(k, ts[i]) - 1e-12) return false;
      }
    }
    return true;
  }

  arbReport(): ArbReport {
    return {
      butterflyCondition: this.butterflyCondition(),
      butterflyGrid: this.butterflyGrid(-1.5, 1.5, 60),
      calendar: this.calendarGrid(-1.5, 1.5, 60),
    };
  }

  /** Vega shape (normalized, ATM ≈ 1) for ladder weighting. */
  vegaShape(k: number, t: number): number {
    const w = this.totalVar(k, t);
    if (w <= 0) return 0;
    const ph = this.phi(this.thetaAt(t));
    const u = ph * k + this.rho;
    const rt = Math.sqrt(u * u + 1 - this.rho * this.rho);
    const d1 = (-k + 0.5 * w) / Math.sqrt(w);
    return Math.exp(-0.5 * d1 * d1) * Math.sqrt(rt);
  }
}
