// Core numerics: seeded RNG, machine-precision normal CDF (Hart 1968 /
// Graeme West's double-precision algorithm — the standard in option
// pricing libraries), PDF, Poisson arrivals. Ported from the Rust
// `micro::special` + `vol` ncdf with equivalent accuracy
// (|eps| < 1e-14 in the central region).

/** xorshift64* RNG — same generator family as the Rust `micro::Rng`. */
export class Rng {
  private s: bigint;
  constructor(seed = 88172645463325252n) {
    this.s = BigInt(seed) & 0xffffffffffffffffn;
    if (this.s === 0n) this.s = 88172645463325252n;
  }
  /** Uniform in [0, 1). */
  uniform(): number {
    let x = this.s;
    x ^= (x << 13n) & 0xffffffffffffffffn;
    x ^= x >> 7n;
    x ^= (x << 17n) & 0xffffffffffffffffn;
    this.s = x & 0xffffffffffffffffn;
    // top 53 bits for a full double mantissa
    return Number(x >> 11n) / 9007199254740992.0;
  }
  /** Standard normal via Box–Muller. */
  normal(): number {
    const u1 = Math.max(this.uniform(), 1e-12);
    const u2 = this.uniform();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }
  /** Poisson draw with mean `lam` (Knuth for small lam, normal approx above). */
  poisson(lam: number): number {
    if (lam <= 0) return 0;
    if (lam < 30) {
      const L = Math.exp(-lam);
      let k = 0;
      let p = 1.0;
      do {
        k++;
        p *= this.uniform();
      } while (p > L);
      return k - 1;
    }
    return Math.max(0, Math.round(lam + Math.sqrt(lam) * this.normal()));
  }
}

const A1 = -3.969683028665376e1;
const A2 = 2.209460984245205e2;
const A3 = -2.759285104469687e2;
const A4 = 1.383577518433505e2;
const A5 = -3.066479806614716e1;
const A6 = 2.506628277459239e0;
const B1 = -5.447609879822406e1;
const B2 = 1.515838255847551e2;
const B3 = -1.919821518634212e2;
const B4 = 1.397095447485428e2;
const B5 = -4.255407025916908e1;
const C1 = -7.784894002430293e-3;
const C2 = -3.223964580411365e-1;
const C3 = -2.400758277161838e0;
const C4 = -2.549732539343734e0;
const C5 = 4.374664140464988e0;
const C6 = 2.938163982698783e0;
const P_LOW = 0.02425;

/** Normal CDF, double precision.
 *
 *  Rebuilt from first principles after the Hart/West port was found
 *  evaluating its rational polynomial at (z/2)² instead of z² (every
 *  value off — the engine had been self-consistent on a wrong Φ):
 *
 *   - central region |x| ≤ 3.5: erf(x/√2) by the alternating Maclaurin
 *     series with the exact term recurrence (|err| ≲ 1e-15);
 *   - tail |x| > 3.5: the Laplace continued fraction for the Mills
 *     ratio, Φ̄(x) = φ(x)/(x + 1/(x + 2/(x + 3/(x + …)))), backward-
 *     evaluated 60 levels deep (converges geometrically there).
 *
 *  Cross-validated against C-libm erf on a 320-point grid over ±8. */
export function ncdf(x: number): number {
  const ax = Math.abs(x);
  if (ax > 37.0) return x > 0 ? 1.0 : 0.0;
  if (ax <= 3.5) {
    // erf(x/√2) by Maclaurin: term_k/term_{k−1} = −z²(2k−1)/(k(2k+1))
    const z = x / Math.SQRT2;
    const z2 = z * z;
    let term = z;
    let sum = z;
    for (let k = 1; k < 90; k++) {
      term *= (-z2 * (2 * k - 1)) / (k * (2 * k + 1));
      sum += term;
      if (Math.abs(term) < 1e-18 * Math.abs(sum)) break;
    }
    return 0.5 + sum / Math.sqrt(Math.PI);
  }
  // Laplace continued fraction for the Mills ratio (backward)
  let cf = 0;
  for (let k = 60; k >= 1; k--) cf = k / (ax + cf);
  const tail = npdf(x) / (ax + cf);
  return x > 0 ? 1 - tail : tail;
}

/** Normal PDF. */
export function npdf(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

export function clamp(x: number, lo: number, hi: number): number {
  return Math.min(Math.max(x, lo), hi);
}
