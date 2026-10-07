// Post-fix verification: worst-case seed 26 + kill-switch resume path +
// quote-side presence under gating + long sustainability.
import { MarketMaker } from "../src/lib/engine/engine";

console.log("=== seed 26 (was 63% of time deadlocked) ===");
{
  const mm = new MarketMaker(26);
  let states = new Map<string, number>();
  let killHalts = 0;
  let bidEmptySteps = 0;
  let bothEmptySteps = 0;
  let lastLogged = -1;
  for (let i = 0; i < 3000; i++) {
    mm.step(1.0);
    states.set(mm.riskState, (states.get(mm.riskState) ?? 0) + 1);
    if (mm.halted) killHalts++;
    const q = mm.snapshot().quotes;
    if (q.bid.length === 0 && q.ask.length === 0) bothEmptySteps++;
    else if (q.bid.length === 0) bidEmptySteps++;
    if (mm.riskState !== "nominal" && mm.riskState !== lastLogged + "" && i - lastLogged > 20) {
      console.log(`  t=${i}s: ${mm.riskState} (${mm.gatedOn}) netDelta=${mm.unhedgedLots().toFixed(1)} raw=${mm.venue.position}`);
      lastLogged = i;
    }
  }
  const s = mm.snapshot();
  console.log(`  states: ${[...states.entries()].map(([k, v]) => `${k}:${v}`).join(" ")}`);
  console.log(`  kill-switch halts: ${killHalts}/3000, both-sides-empty: ${bothEmptySteps}, bid-empty: ${bidEmptySteps}`);
  console.log(`  equity ${s.equity.toFixed(0)} optFills ${s.options.fills} hedgeLots ${s.options.hedgeLotsTotal}`);
  console.log(`  risk events: ${mm.riskEventTape.slice(-6).map((e) => `t${e.t} ${e.msg.slice(0, 60)}`).join(" | ")}`);
}

console.log("\n=== kill-switch resume path (seed 4, resume on first halt) ===");
{
  const mm = new MarketMaker(4);
  let resumedAt = -1;
  for (let i = 0; i < 2600; i++) {
    mm.step(1.0);
    if (mm.halted && resumedAt < 0) {
      console.log(`  t=${i}s HALT: ${mm.haltReason}`);
      mm.resume();
      resumedAt = i;
      continue;
    }
    if (resumedAt >= 0 && i === resumedAt + 10) {
      const q = mm.snapshot().quotes;
      console.log(`  t=${i}s (+10s after resume): halted=${mm.halted} bids=${q.bid.length} asks=${q.ask.length} state=${mm.riskState}`);
      break;
    }
  }
}

console.log("\n=== 5000-step sustainability, 12 seeds ===");
let totHalted = 0;
let totSteps = 0;
let minEq = Infinity;
let eqs: number[] = [];
for (let seed = 1; seed <= 12; seed++) {
  const mm = new MarketMaker(seed);
  let haltedSteps = 0;
  for (let i = 0; i < 5000; i++) {
    mm.step(1.0);
    if (mm.halted) haltedSteps++;
  }
  const s = mm.snapshot();
  totHalted += haltedSteps;
  totSteps += 5000;
  eqs.push(s.equity);
  minEq = Math.min(minEq, s.equity);
  console.log(`  seed ${seed}: halted ${(haltedSteps / 50).toFixed(1)}% eq ${s.equity.toFixed(0)} fills ${s.options.fills} state=${s.risk.state}`);
}
console.log(`  MEAN halted: ${((totHalted / totSteps) * 100).toFixed(2)}% · min equity ${minEq.toFixed(0)} · mean equity ${(eqs.reduce((a, b) => a + b, 0) / eqs.length).toFixed(0)}`);
