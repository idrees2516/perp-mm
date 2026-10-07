// The institutional RFQ lane, in the Paradigm / TradeParadigm-on-Paradex
// and Derive V3 protocol shape: takers publish multi-leg package intents
// (spreads, straddles, risk reversals, flies, calendars, boxes), the desk
// responds with FIRM two-sided package quotes carrying a TTL, execution
// is ATOMIC across legs (all-or-nothing), and combo margin offsets price
// the risk netting of offsetting structures — a call spread or a box
// carries a fraction of the standalone margin of its legs, so it quotes
// a fraction of the standalone spread. Single-leg executions act as
// market touches that anchor the governed vol surface (Paradex shape).

import { Rng, clamp } from "./math";
import { SsviSurface } from "./ssvi";
import { Kind, price, fullGreeks } from "./bsm";
import { OptionMarket } from "./options";
import { OptMm } from "./models";
import { VannaVolga } from "./vannaVolga";

export interface RfqLeg {
  expIdx: number;
  legIdx: number;
  kind: Kind;
  dir: 1 | -1;
  lots: number;
}

export interface RfqLegQuote {
  expIdx: number;
  legIdx: number;
  kind: Kind;
  dir: number;
  strike: number;
  t: number;
  ivBid: number;
  ivAsk: number;
  ivFair: number;
  /** Package fair premium contribution of this leg (per package lot). */
  premFair: number;
}

export interface RfqQuoteView {
  id: number;
  label: string;
  legs: RfqLegQuote[];
  /** Package premium per lot: desk buys at bid, sells at ask. */
  bid: number;
  ask: number;
  fair: number;
  marginOffset: number;
  /** Package greeks per lot (signed, underlying units for delta). */
  delta: number;
  vega: number;
  gamma: number;
  lots: number;
  ttl: number;
  totalTtl: number;
  origin: "institutional" | "manual";
  status: "quoted" | "executed" | "expired";
  note?: string;
}

export interface RfqExecEvent {
  id: number;
  label: string;
  side: "desk-buys" | "desk-sells";
  price: number;
  lots: number;
  t: number;
  delta: number;
  vega: number;
  origin: "institutional" | "manual";
}

/** Execution-risk context the desk's risk engine supplies. */
export interface RfqRiskCtx {
  unhedgedLots: number;
  netDeltaLimit: number;
  netVega: number;
  vegaLimit: number;
  freeze: boolean;
  lotSize: number;
  multiplier: number;
  clock: number;
}

export interface RfqTemplate {
  id: string;
  label: string;
  blurb: string;
  weight: number;
  make: (e: number, atm: number, lots: number, nL: number, nE: number) => RfqLeg[];
}

const L = (expIdx: number, legIdx: number, kind: Kind, dir: 1 | -1, lots: number): RfqLeg => ({ expIdx, legIdx, kind, dir, lots });

export const RFQ_TEMPLATES: RfqTemplate[] = [
  { id: "single-call", label: "Single Call", blurb: "outright call — also acts as a surface touch", weight: 0.16, make: (e, a, n) => [L(e, a, "call", 1, n)] },
  { id: "single-put", label: "Single Put", blurb: "outright put — also acts as a surface touch", weight: 0.14, make: (e, a, n) => [L(e, a, "put", 1, n)] },
  { id: "call-spread", label: "Call Spread", blurb: "vertical: buy low strike, sell high", weight: 0.12, make: (e, a, n) => [L(e, a - 1, "call", 1, n), L(e, a + 1, "call", -1, n)] },
  { id: "put-spread", label: "Put Spread", blurb: "vertical: buy high strike, sell low", weight: 0.1, make: (e, a, n) => [L(e, a + 1, "put", 1, n), L(e, a - 1, "put", -1, n)] },
  { id: "straddle", label: "Straddle", blurb: "ATM call + put — pure gamma/vega", weight: 0.1, make: (e, a, n) => [L(e, a, "call", 1, n), L(e, a, "put", 1, n)] },
  { id: "strangle", label: "Strangle", blurb: "OTM call + OTM put", weight: 0.08, make: (e, a, n) => [L(e, a + 2, "call", 1, n), L(e, a - 2, "put", 1, n)] },
  { id: "risk-reversal", label: "Risk Reversal", blurb: "long call, short put — the skew instrument", weight: 0.1, make: (e, a, n) => [L(e, a + 2, "call", 1, n), L(e, a - 2, "put", -1, n)] },
  { id: "butterfly", label: "Butterfly", blurb: "long wings, short 2× body", weight: 0.08, make: (e, a, n) => [L(e, a - 2, "call", 1, n), L(e, a, "call", -1, 2 * n), L(e, a + 2, "call", 1, n)] },
  { id: "calendar", label: "Calendar", blurb: "short near ATM, long far ATM", weight: 0.06, make: (e, a, n, _nL, nE) => [L(e, a, "call", -1, n), L(Math.min(e + 1, nE - 1), a, "call", 1, n)] },
  { id: "box", label: "Box", blurb: "synthetic forward lockout — near-riskless combo", weight: 0.06, make: (e, a, n) => [L(e, a - 2, "call", 1, n), L(e, a + 2, "call", -1, n), L(e, a - 2, "put", -1, n), L(e, a + 2, "put", 1, n)] },
];

export class RfqEngine {
  active: RfqQuoteView[] = [];
  recent: RfqQuoteView[] = [];
  executions: RfqExecEvent[] = [];
  requests = 0;
  executed = 0;
  refused = 0;
  /** Cumulative RFQ premium cash flow (desk view). */
  premiumFlow = 0;
  private nextId = 1;

  constructor(private rng: Rng, public ttlS = 5, public autoRate = 1 / 40) {}

  private templateByWeight(): RfqTemplate {
    const r = this.rng.uniform();
    let acc = 0;
    for (const t of RFQ_TEMPLATES) {
      acc += t.weight;
      if (r <= acc) return t;
    }
    return RFQ_TEMPLATES[0];
  }

  /** Build a package from a template on the given expiry. */
  buildPackage(tpl: RfqTemplate, e: number, lots: number, options: OptionMarket): { legs: RfqLeg[]; label: string } {
    const nL = options.cfg.moneyness.length;
    const nE = options.cfg.expiries.length;
    const atm = Math.floor(nL / 2);
    const legs = tpl.make(clamp(e, 0, nE - 1), atm, lots, nL, nE).map((l) => ({ ...l, legIdx: clamp(l.legIdx, 0, nL - 1), expIdx: clamp(l.expIdx, 0, nE - 1) }));
    return { legs, label: tpl.label };
  }

  /** Firm two-sided package quote with TTL, combo margin offset priced in. */
  quote(
    legs: RfqLeg[],
    label: string,
    origin: "institutional" | "manual",
    s: number,
    options: OptionMarket,
    optMm: OptMm,
    surface: SsviSurface,
    vegaPerExp: number[],
    vv: VannaVolga | null,
    ttl?: number,
  ): RfqQuoteView {
    const id = this.nextId++;
    this.requests++;
    const legViews: RfqLegQuote[] = [];
    let fair = 0;
    let bid = 0;
    let ask = 0;
    let delta = 0;
    let vega = 0;
    let gamma = 0;
    let grossRisk = 0;
    const lots = legs.length ? legs[0].lots : 1;
    for (const leg of legs) {
      const t = options.cfg.expiries[leg.expIdx];
      const k = Math.exp(options.cfg.moneyness[leg.legIdx]) * s;
      const vegaE = vegaPerExp[leg.expIdx] ?? 0;
      const q = options.quotes(leg.expIdx, leg.legIdx, leg.kind, s, surface, optMm, vegaE, vv);
      const premFair = price(leg.kind, s, k, 0, 0, q.ivFair, t);
      // package prices are TOTALS across the legs (each scaled by its lots)
      bid += (leg.dir > 0 ? q.premBid : -q.premAsk) * leg.lots;
      ask += (leg.dir > 0 ? q.premAsk : -q.premBid) * leg.lots;
      fair += leg.dir * premFair * leg.lots;
      const g = fullGreeks(leg.kind, s, k, 0, 0, q.ivFair, t);
      delta += leg.dir * g.delta * leg.lots;
      vega += leg.dir * g.vega * leg.lots;
      gamma += leg.dir * g.gamma * leg.lots;
      // gross risk: delta + vega + gamma all count — offsetting structures
      // (spreads, boxes) net; additive ones (straddles) stack
      grossRisk +=
        Math.abs(g.delta * leg.lots) +
        0.001 * Math.abs(g.vega * leg.lots) +
        0.5 * Math.abs(g.gamma * leg.lots) * s;
      legViews.push({ expIdx: leg.expIdx, legIdx: leg.legIdx, kind: leg.kind, dir: leg.dir, strike: k, t, ivBid: q.ivBid, ivAsk: q.ivAsk, ivFair: q.ivFair, premFair: leg.dir * premFair * leg.lots });
    }
    // combo margin offset: how much of the gross leg risk nets inside the
    // package — the SFPM offset the venue grants, shared with the taker as
    // a tighter package spread
    const netRisk = Math.abs(delta) + 0.001 * Math.abs(vega) + 0.5 * Math.abs(gamma) * s;
    const marginOffset = clamp(1 - netRisk / Math.max(grossRisk, 1e-9), 0, 0.6);
    const half = (ask - bid) / 2;
    // floor the package half-spread: vol-riskless structures (boxes) would
    // otherwise quote a zero-width market
    const half2 = Math.max(half * (1 - marginOffset * 0.7), Math.max(0.05, Math.abs(fair) * 0.001));
    const totalTtl = ttl ?? this.ttlS;
    return {
      id,
      label,
      legs: legViews,
      bid: fair - half2,
      ask: fair + half2,
      fair,
      marginOffset,
      delta,
      vega,
      gamma,
      lots,
      ttl: totalTtl,
      totalTtl,
      origin,
      status: "quoted",
    };
  }

  /** Institutional flow: arrivals, client decisions, TTL decay. Returns
   *  surface touches from executed single-leg packages. */
  step(
    dt: number,
    s: number,
    options: OptionMarket,
    optMm: OptMm,
    surface: SsviSurface,
    vegaPerExp: number[],
    vv: VannaVolga | null,
    ctx: RfqRiskCtx,
  ): Array<{ expIdx: number; iv: number }> {
    const touches: Array<{ expIdx: number; iv: number }> = [];
    // TTL decay
    for (const q of this.active) {
      q.ttl -= dt;
      if (q.ttl <= 0) {
        q.status = "expired";
        this.pushRecent(q);
      }
    }
    this.active = this.active.filter((q) => q.status === "quoted");
    // arrival
    if (!ctx.freeze && this.rng.uniform() < this.autoRate * dt) {
      const tpl = this.templateByWeight();
      const e = Math.floor(this.rng.uniform() * options.cfg.expiries.length);
      const lots = 5 + Math.floor(this.rng.uniform() * 21);
      const { legs, label } = this.buildPackage(tpl, e, lots, options);
      const q = this.quote(legs, label, "institutional", s, options, optMm, surface, vegaPerExp, vv);
      this.active.push(q);
      // client decision against an internal fair with independent noise
      const clientFair = q.fair * (1 + this.rng.normal() * 0.004);
      if (q.ask <= clientFair) {
        this.execute(q.id, "desk-sells", s, options, optMm, surface, vegaPerExp, vv, ctx, touches);
      } else if (q.bid >= clientFair) {
        this.execute(q.id, "desk-buys", s, options, optMm, surface, vegaPerExp, vv, ctx, touches);
      }
    }
    return touches;
  }

  /** Atomic package execution — all legs or nothing, risk-checked. */
  execute(
    id: number,
    side: "desk-buys" | "desk-sells",
    s: number,
    options: OptionMarket,
    optMm: OptMm,
    surface: SsviSurface,
    vegaPerExp: number[],
    vv: VannaVolga | null,
    ctx: RfqRiskCtx,
    touches?: Array<{ expIdx: number; iv: number }>,
  ): { ok: boolean; reason?: string } {
    const q = this.active.find((x) => x.id === id && x.status === "quoted");
    if (!q) return { ok: false, reason: "quote not found / expired" };
    // rebuild legs from the view (strikes are stale-mark safe: quote is live)
    const pkgDeltaLots = (q.delta * ctx.multiplier) / ctx.lotSize;
    if (side === "desk-sells" && Math.abs(ctx.unhedgedLots - pkgDeltaLots) > ctx.netDeltaLimit) {
      this.refused++;
      q.note = "refused: net-delta limit";
      return { ok: false, reason: "net-delta limit" };
    }
    if (side === "desk-buys" && Math.abs(ctx.unhedgedLots + pkgDeltaLots) > ctx.netDeltaLimit) {
      this.refused++;
      q.note = "refused: net-delta limit";
      return { ok: false, reason: "net-delta limit" };
    }
    const vegaAfter = ctx.netVega + (side === "desk-sells" ? -q.vega : q.vega);
    if (Math.abs(vegaAfter) > ctx.vegaLimit) {
      this.refused++;
      q.note = "refused: vega limit";
      return { ok: false, reason: "vega limit" };
    }
    // reconstruct legs (positions mutate via the option market)
    const tplLegs: RfqLeg[] = [];
    const sign = side === "desk-sells" ? -1 : 1; // desk-sells takes the short side of the package
    for (const lv of q.legs) {
      tplLegs.push({ expIdx: lv.expIdx, legIdx: lv.legIdx, kind: lv.kind, dir: (sign > 0 ? 1 : -1) as 1 | -1, lots: q.lots });
    }
    // atomic pre-check on per-leg bounds
    for (const leg of tplLegs) {
      const ki = options.kindIdx(leg.kind);
      const next = options.pos[leg.expIdx][leg.legIdx][ki] + leg.dir * leg.lots;
      if (Math.abs(next) > options.cfg.maxLots) {
        this.refused++;
        q.note = "refused: leg limit";
        return { ok: false, reason: "leg limit" };
      }
    }
    // execute all legs
    const execIv = side === "desk-sells" ? q.legs.map((l) => (l.dir > 0 ? l.ivAsk : l.ivBid)) : q.legs.map((l) => (l.dir > 0 ? l.ivBid : l.ivAsk));
    let li = 0;
    for (const leg of tplLegs) {
      const ki = options.kindIdx(leg.kind);
      options.pos[leg.expIdx][leg.legIdx][ki] += leg.dir * leg.lots;
      const t = options.cfg.expiries[leg.expIdx];
      const k = Math.exp(options.cfg.moneyness[leg.legIdx]) * s;
      const prem = price(leg.kind, s, k, 0, 0, execIv[li] ?? q.legs[li].ivFair, t) * leg.lots;
      // premium: desk-sells receives on +dir legs, pays on −dir legs
      options.cash += leg.dir * prem;
      this.premiumFlow += leg.dir * prem;
      // single-leg packages are market touches: anchor the governed surface
      if (tplLegs.length === 1 && touches) {
        const ivTouch = execIv[li] ?? q.legs[li].ivFair;
        touches.push({ expIdx: leg.expIdx, iv: ivTouch });
      }
      li++;
    }
    options.fills += tplLegs.length;
    q.status = "executed";
    q.note = `executed: ${side} @ ${(side === "desk-sells" ? q.ask : q.bid).toFixed(2)}`;
    this.executed++;
    const ev: RfqExecEvent = {
      id: q.id,
      label: q.label,
      side,
      price: side === "desk-sells" ? q.ask : q.bid,
      lots: q.lots,
      t: ctx.clock,
      delta: (side === "desk-sells" ? -1 : 1) * q.delta,
      vega: (side === "desk-sells" ? -1 : 1) * q.vega,
      origin: q.origin,
    };
    this.executions.push(ev);
    if (this.executions.length > 40) this.executions.shift();
    this.pushRecent(q);
    this.active = this.active.filter((x) => x.status === "quoted");
    return { ok: true };
  }

  pushRecentPublic(q: RfqQuoteView) {
    this.pushRecent(q);
  }

  private pushRecent(q: RfqQuoteView) {
    this.recent.push(q);
    if (this.recent.length > 14) this.recent.shift();
  }
}
