// Portfolio margin in the Derive V3 SFPM / Paradex SCAN lineage: the
// whole book — options AND the perp hedge leg — is repriced under a
// scenario grid of spot×vol shocks plus a time-decay scenario; the
// maintenance requirement is the worst grid loss floored by the
// short-option minimum charge (Deribit's listed SOMC convention, which
// covers the deep-OTM tail a finite scenario grid understates), and
// initial margin carries a 1.2× buffer. Risk gating then binds on
// MARGIN UTILIZATION — the protocol-grade replacement for raw position
// limits: capital is the true constraint, not lot counts.

import { Kind, price } from "./bsm";

export interface MarginLeg {
  kind: Kind;
  strike: number;
  iv: number;
  t: number;
  lots: number;
}

export interface MarginScenario {
  name: string;
  dSpot: number;
  /** Absolute IV shock (vol points, e.g. +0.10 = +10 vol pts). */
  dVol: number;
  /** Calendar days of decay applied under the scenario. */
  decayDays: number;
}

/** The SFPM scanning grid: spot shocks crossed with vol shocks, plus a
 *  pure time-decay scenario for short-gamma books. */
export const SFPM_GRID: MarginScenario[] = [
  { name: "s−15% v−10pt", dSpot: -0.15, dVol: -0.1, decayDays: 0 },
  { name: "s−15% v+10pt", dSpot: -0.15, dVol: 0.1, decayDays: 0 },
  { name: "s−7.5% v−5pt", dSpot: -0.075, dVol: -0.05, decayDays: 0 },
  { name: "s−7.5% v+5pt", dSpot: -0.075, dVol: 0.05, decayDays: 0 },
  { name: "s−7.5% v+15pt", dSpot: -0.075, dVol: 0.15, decayDays: 0 },
  { name: "s+7.5% v−5pt", dSpot: 0.075, dVol: -0.05, decayDays: 0 },
  { name: "s+7.5% v+5pt", dSpot: 0.075, dVol: 0.05, decayDays: 0 },
  { name: "s+15% v−10pt", dSpot: 0.15, dVol: -0.1, decayDays: 0 },
  { name: "s+15% v+10pt", dSpot: 0.15, dVol: 0.1, decayDays: 0 },
  { name: "t+1d (decay)", dSpot: 0, dVol: 0, decayDays: 1 },
];

export interface MarginState {
  /** Worst scenario loss (pre-buffer). */
  scanningLoss: number;
  /** Short-option minimum charge. */
  somc: number;
  maintenance: number;
  initial: number;
  /** initial / equity. */
  utilization: number;
  worst: string;
  perScenario: Array<{ name: string; loss: number }>;
}

const INITIAL_BUFFER = 1.2;
/** SOMC: per short lot, 12% of the lesser of spot/strike (Deribit SOM
 *  lineage, scaled to this sim's book/capital ratio). */
const SOMC_FRACTION = 0.12;

export function portfolioMargin(
  s: number,
  legs: MarginLeg[],
  perpLots: number,
  lotSize: number,
  equity: number,
  multiplier: number,
  grid: MarginScenario[] = SFPM_GRID,
): MarginState {
  const baseBook = legs.reduce((acc, l) => acc + price(l.kind, s, l.strike, 0, 0, l.iv, l.t) * l.lots * multiplier, 0);
  let worst = 0;
  let worstName = "—";
  const perScenario: Array<{ name: string; loss: number }> = [];
  for (const sc of grid) {
    const s2 = Math.max(0.05, s * (1 + sc.dSpot));
    const dt = Math.max(1e-6, sc.decayDays / 365);
    let v = 0;
    for (const l of legs) {
      const t2 = Math.max(1e-6, l.t - dt);
      const iv2 = Math.max(0.005, l.iv + sc.dVol * Math.sqrt(t2 / Math.max(l.t, 1e-6)));
      // spot shock moves moneyness; vol shock is applied to the leg's pillar
      v += price(l.kind, s2, l.strike, 0, 0, iv2, t2) * l.lots * multiplier;
    }
    // perp leg marks with the shocked spot
    v += perpLots * (s2 - s) * lotSize;
    const loss = baseBook + perpLots * 0 * lotSize - v;
    perScenario.push({ name: sc.name, loss: Math.max(0, loss) });
    if (loss > worst) {
      worst = loss;
      worstName = sc.name;
    }
  }
  // short-option minimum charge: the grid's deep-OTM blind spot
  let somc = 0;
  for (const l of legs) {
    if (l.lots < 0) {
      somc += Math.abs(l.lots) * SOMC_FRACTION * Math.min(s, l.strike) * multiplier;
    }
  }
  const maintenance = Math.max(worst, somc);
  const initial = maintenance * INITIAL_BUFFER;
  const utilization = equity > 1e-9 ? initial / equity : Infinity;
  return { scanningLoss: worst, somc, maintenance, initial, utilization, worst: worstName, perScenario };
}
