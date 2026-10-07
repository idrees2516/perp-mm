// Market microstructure estimators — port of the Rust `micro` crate's
// hot path subset used by the engine: EWMA multi-scale sigma, Roll
// spread with serial-dependence correction, Stoikov micro-price,
// order-flow imbalance + OFI with impact fit, LM08-style jump statistic
// and a rough-volatility Hurst estimate.

import { clamp } from "./math";

export class Ewma {
  private v: number;
  constructor(public alpha: number, init = NaN) {
    this.v = init;
  }
  update(x: number): number {
    if (isNaN(this.v)) this.v = x;
    else this.v = this.alpha * x + (1 - this.alpha) * this.v;
    return this.v;
  }
  get value(): number {
    return isNaN(this.v) ? 0 : this.v;
  }
}

/** Roll effective spread from trade-price series with the serial-dep cov2/cov1 correction. */
export class RollSpread {
  private prices: number[] = [];
  constructor(public window = 64) {}
  update(p: number) {
    this.prices.push(p);
    if (this.prices.length > this.window) this.prices.shift();
  }
  /** Effective half-spread estimate (returns 0 when undefined). */
  estimate(): number {
    const ps = this.prices;
    if (ps.length < 4) return 0;
    let s1 = 0;
    let s2 = 0;
    const n = ps.length - 1;
    const mean = ps.slice(1).reduce((a, b) => a + b, 0) / n;
    for (let i = 1; i < ps.length - 1; i++) {
      s1 += (ps[i] - mean) * (ps[i + 1] - mean);
    }
    const mean0 = ps.slice(0, -1).reduce((a, b) => a + b, 0) / n;
    for (let i = 0; i < ps.length - 1; i++) {
      s2 += (ps[i] - mean0) * (ps[i + 1] - mean0);
    }
    const cov1 = s1 / (n - 1);
    const cov2 = s2 / (n - 1);
    let gamma = cov1; // lag-1 autocov of mid-changes proxy
    if (Math.abs(cov2) > 1e-18 && Math.abs(cov1) > 1e-18) {
      // serial-dependence correction: gamma = cov2·(cov2/cov1) rho-adjusted
      gamma = (cov2 * cov2) / cov1;
    }
    const spread = 2 * Math.sqrt(Math.max(-gamma, 0));
    return isFinite(spread) ? spread : 0;
  }
}

/** Stoikov micro-price: bid + spread·I with the logistic imbalance weight. */
export function microPrice(bid: number, ask: number, bidSz: number, askSz: number, a = 3.2, b = 6.0): number {
  const total = bidSz + askSz;
  if (total <= 0) return 0.5 * (bid + ask);
  const imb = askSz > bidSz ? Math.exp(-a * (1 - bidSz / total) - b) : 1 - Math.exp(-a * (bidSz / total) - b);
  const mid = 0.5 * (bid + ask);
  return mid + (ask - bid) * (imb - 0.5);
}

export class OfiTracker {
  private ofi = 0;
  private impact = 0.02;
  private queue: Array<[number, number]> = [];
  constructor(public window = 32) {}
  /** Order flow imbalance event: signed size at the touch. */
  update(signedLots: number, midNow: number, midPrev: number) {
    this.ofi += signedLots;
    this.queue.push([signedLots, midNow - midPrev]);
    if (this.queue.length > this.window) this.queue.shift();
    // RLS-lite impact fit: slope of mid-move on cumulative ofi.
    if (this.queue.length >= 8) {
      let sx = 0;
      let sy = 0;
      let sxx = 0;
      let sxy = 0;
      for (const [x, y] of this.queue) {
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
      }
      const n = this.queue.length;
      const den = n * sxx - sx * sx;
      if (Math.abs(den) > 1e-12) this.impact = clamp((n * sxy - sx * sy) / den, -5, 5);
    }
  }
  get value(): number {
    return this.ofi;
  }
  get impactFit(): number {
    return this.impact;
  }
}

/** LM08-style jump statistic: standardized |return| against EWMA vol. */
export class JumpDetector {
  private z: number[] = [];
  constructor(public window = 128) {}
  update(ret: number, sigma: number) {
    if (sigma > 1e-12) {
      this.z.push(Math.abs(ret) / sigma);
      if (this.z.length > this.window) this.z.shift();
    }
  }
  /** Latest z (LM08 max-statistic flavor). */
  stat(): number {
    return this.z.length ? this.z[this.z.length - 1] : 0;
  }
  /** Threshold exceedance fraction. */
  exceedRate(threshold = 3.5): number {
    if (!this.z.length) return 0;
    return this.z.filter((z) => z > threshold).length / this.z.length;
  }
}

/** Rough-volatility Hurst estimate via variance of log-variance scaling. */
export class HurstEstimator {
  private rets: number[] = [];
  constructor(public window = 256) {}
  update(ret: number) {
    this.rets.push(ret);
    if (this.rets.length > this.window) this.rets.shift();
  }
  /** H in (0, 1); NaN until enough data. */
  estimate(): number {
    const r = this.rets;
    if (r.length < 64) return NaN;
    // m-scaling of realized variance increments (RFSV: log var ~ fBm).
    const ms = [4, 8, 16, 32];
    const pts: Array<[number, number]> = [];
    for (const m of ms) {
      const vars: number[] = [];
      for (let i = 0; i + m <= r.length; i += m) {
        let acc = 0;
        for (let j = i; j < i + m; j++) acc += r[j] * r[j];
        vars.push(Math.log(Math.max(acc, 1e-18)));
      }
      if (vars.length < 4) continue;
      const mean = vars.reduce((a, b) => a + b, 0) / vars.length;
      const vvar = vars.reduce((a, b) => a + (b - mean) * (b - mean), 0) / vars.length;
      pts.push([Math.log(m), Math.log(Math.max(vvar, 1e-18))]);
    }
    if (pts.length < 3) return NaN;
    let sx = 0;
    let sy = 0;
    let sxx = 0;
    let sxy = 0;
    for (const [x, y] of pts) {
      sx += x;
      sy += y;
      sxx += x * x;
      sxy += x * y;
    }
    const n = pts.length;
    const den = n * sxx - sx * sx;
    if (Math.abs(den) < 1e-12) return NaN;
    return clamp((n * sxy - sx * sy) / den / 2, 0.05, 0.95);
  }
}
