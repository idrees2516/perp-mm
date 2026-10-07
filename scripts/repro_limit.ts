// Reproduce the user-reported bug:
//   "risk-manager resume, after 5-10 seconds HALTED — perp position limit
//    breach appears again and the bid side disappears"
import { MarketMaker } from "../src/lib/engine/engine";

const mm = new MarketMaker(12345);
let haltedOnce = false;
let resumedAt = -1;
let haltedSteps = 0;

for (let i = 0; i < 3000; i++) {
  const wasHalted = mm.halted;
  mm.step(1.0);

  if (mm.halted && !wasHalted && !haltedOnce) {
    haltedOnce = true;
    console.log(`[t=${i}s] FIRST HALT: ${mm.haltReason}`);
    console.log(`   raw perp pos=${mm.venue.position} limit=${mm.positionLimit}`);
    const net = mm.optNet;
    console.log(`   opt delta=${net.delta.toFixed(1)} lots  (hedge leg = ${(-net.delta).toFixed(1)})`);
    console.log(`   unhedged (pos + optDelta) = ${(mm.venue.position + net.delta).toFixed(2)}`);
    console.log(`   vega=${net.vega.toFixed(0)}/${mm.vegaLimit}  fills=${mm.options.fills}`);
    // simulate the user clicking "risk-manager resume"
    mm.resume();
    resumedAt = i;
    console.log(`   -> user clicked risk-manager resume`);
    continue;
  }
  if (resumedAt >= 0 && i - resumedAt <= 15) {
    const q = mm.snapshot().quotes;
    if (mm.halted && !wasHalted) {
      console.log(`[t=${i}s] RE-HALT ${i - resumedAt}s after resume: ${mm.haltReason}  pos=${mm.venue.position}`);
    }
    if (i === resumedAt + 5 || i === resumedAt + 10 || i === resumedAt + 15) {
      console.log(`[t=${i}s] (${i - resumedAt}s post-resume) halted=${mm.halted} bids=${q.bid.length} asks=${q.ask.length} pos=${mm.venue.position}`);
    }
  }
  if (mm.halted) haltedSteps++;
}
const s = mm.snapshot();
console.log(`\n3000 steps: ${haltedSteps} halted steps (${((haltedSteps / 3000) * 100).toFixed(1)}%), equity ${s.equity.toFixed(0)}, optFills ${s.options.fills}, hedgeLots ${s.options.hedgeLotsTotal}`);
