"use client";

import { useState, useEffect, useRef } from "react";
import { useEngine } from "@/lib/engine/useEngine";
import type { StrategyId } from "@/lib/engine/engine";
import { INITIAL_CAPITAL } from "@/lib/engine/engine";
import { RFQ_TEMPLATES } from "@/lib/engine/rfq";
import { Sparkline, LineChart, Heatmap, SignedBar, DepthLadder, StatChip, Panel } from "@/components/terminal/charts";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Pause, Play, RotateCcw, Zap, Activity, Layers, Gauge, LineChart as LineIcon, Grid3x3, Scale, Brain, ShieldAlert, Handshake, Landmark, Coins } from "lucide-react";

const fmt = (v: number, d = 2) => (isFinite(v) ? v.toFixed(d) : "—");
const vp = (v: number, d = 2) => (isFinite(v) ? (v * 100).toFixed(d) : "—"); // vol points
const signed = (v: number) => (v > 0 ? "text-emerald-400" : v < 0 ? "text-rose-400" : "text-neutral-300");
const expLabel = (t: number) => {
  const d = t * 365;
  if (d < 10) return `${d.toFixed(0)}D`;
  if (d < 100) return `${d.toFixed(0)}D`;
  return `${(d / 365).toFixed(2).replace(/\.?0+$/, "")}Y`;
};

function SmileChart({ chain }: { chain: Array<{ moneyness: number; call: { ivBid: number; ivFair: number; ivAsk: number } }> }) {
  const w = 640;
  const h = 190;
  if (!chain.length) return null;
  const ivs = chain.flatMap((c) => [c.call.ivBid, c.call.ivAsk]);
  const min = Math.min(...ivs) * 0.985;
  const max = Math.max(...ivs) * 1.015;
  const range = max - min || 1;
  const yOf = (v: number) => h - 12 - ((v - min) / range) * (h - 24);
  const xOf = (i: number) => 30 + (i / (chain.length - 1)) * (w - 50);
  const line = (sel: (c: typeof chain[0]) => number, color: string, dash?: string) => (
    <polyline
      points={chain.map((c, i) => `${xOf(i).toFixed(1)},${yOf(sel(c)).toFixed(1)}`).join(" ")}
      fill="none"
      stroke={color}
      strokeWidth={1.6}
      strokeDasharray={dash}
    />
  );
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full block">
      {[0.25, 0.5, 0.75].map((f) => (
        <line key={f} x1={30} x2={w - 20} y1={h * f} y2={h * f} stroke="#27272a" strokeWidth={0.5} />
      ))}
      {line((c) => c.call.ivAsk, "#fb7185")}
      {line((c) => c.call.ivBid, "#34d399")}
      {line((c) => c.call.ivFair, "#e4e4e7", "4 3")}
      {chain.map((c, i) => (
        <g key={i}>
          <circle cx={xOf(i)} cy={yOf(c.call.ivAsk)} r={2.2} fill="#fb7185" />
          <circle cx={xOf(i)} cy={yOf(c.call.ivBid)} r={2.2} fill="#34d399" />
          <text x={xOf(i)} y={h - 1} textAnchor="middle" fill="#71717a" fontSize={9} fontFamily="monospace">
            {c.moneyness === 0 ? "ATM" : fmt(c.moneyness * 100, 1)}
          </text>
        </g>
      ))}
      <text x={34} y={14} fill="#fb7185" fontSize={9} fontFamily="monospace">ask IV</text>
      <text x={74} y={14} fill="#34d399" fontSize={9} fontFamily="monospace">bid IV</text>
      <text x={118} y={14} fill="#a1a1aa" fontSize={9} fontFamily="monospace">fair (SSVI+VV)</text>
    </svg>
  );
}

function WwBandViz({ unhedged, band }: { unhedged: number; band: number }) {
  const scale = Math.max(band * 3.2, Math.abs(unhedged) * 1.25, 4);
  const x = (v: number) => 50 + (v / scale) * 50;
  const clamped = Math.max(-scale, Math.min(scale, unhedged));
  const outside = Math.abs(unhedged) >= band;
  return (
    <div>
      <svg viewBox="0 0 100 26" className="w-full block h-14">
        <line x1={0} x2={100} y1={13} y2={13} stroke="#3f3f46" strokeWidth={0.4} />
        <rect x={x(-band)} y={5} width={Math.max(x(band) - x(-band), 0.5)} height={16} fill="#f59e0b" opacity={0.18} rx={1} />
        <line x1={x(-band)} x2={x(-band)} y1={4} y2={22} stroke="#f59e0b" strokeWidth={0.5} strokeDasharray="1.5 1.5" />
        <line x1={x(band)} x2={x(band)} y1={4} y2={22} stroke="#f59e0b" strokeWidth={0.5} strokeDasharray="1.5 1.5" />
        <line x1={50} x2={50} y1={4} y2={22} stroke="#71717a" strokeWidth={0.6} />
        <circle cx={x(clamped)} cy={13} r={2.6} fill={outside ? "#fb7185" : "#34d399"} />
      </svg>
      <div className="flex justify-between text-[9px] font-mono text-neutral-500 mt-0.5">
        <span>{-scale.toFixed(1)}</span>
        <span className="text-amber-500/80">± H* = {band.toFixed(2)} lots</span>
        <span>+{scale.toFixed(1)}</span>
      </div>
    </div>
  );
}

export default function Home() {
  const eng = useEngine(12345);
  const snap = eng.snap;
  const [tab, setTab] = useState("terminal");
  const [rfqTpl, setRfqTpl] = useState("call-spread");
  const [rfqLots, setRfqLots] = useState(10);
  const [rfqMsg, setRfqMsg] = useState<string | null>(null);
  /** Real-time firm windows for manual quotes (id → wall-clock deadline ms). */
  const [manualFirm, setManualFirm] = useState<Record<number, number>>({});
  const firmRef = useRef<Record<number, number>>({});
  useEffect(() => {
    const iv = setInterval(() => {
      const now = Date.now();
      for (const [id, dl] of Object.entries(firmRef.current)) {
        if (now > dl) {
          eng.cancelRfq(Number(id));
          delete firmRef.current[Number(id)];
          setManualFirm({ ...firmRef.current });
        }
      }
    }, 400);
    return () => clearInterval(iv);
  }, [eng]);

  if (!snap) {
    return (
      <div className="min-h-screen bg-neutral-950 text-neutral-300 flex items-center justify-center">
        <div className="font-mono text-xs text-neutral-500">booting engine…</div>
      </div>
    );
  }

  const pnl = snap.equity - INITIAL_CAPITAL;
  const pnlTone = pnl > 0 ? "pos" : pnl < 0 ? "neg" : "neutral";
  const pnlPct = (pnl / INITIAL_CAPITAL) * 100;
  const o = snap.options;
  const cur = o.expiries[o.selectedExp];

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-200 font-sans">
      {/* ------------------------------------------------ header */}
      <header className="sticky top-0 z-20 border-b border-neutral-800 bg-neutral-950/95 backdrop-blur px-3 py-2">
        <div className="max-w-[1440px] mx-auto flex flex-wrap items-center gap-x-5 gap-y-2">
          <div className="flex items-center gap-2">
            <span className="inline-flex h-6 w-6 items-center justify-center rounded bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
              <Zap className="h-3.5 w-3.5" />
            </span>
            <div>
              <div className="text-sm font-semibold tracking-tight leading-none">perp-mm</div>
              <div className="text-[9px] text-neutral-500 leading-none mt-0.5">options quoting engine · live</div>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5">
            <StatChip label="spot" value={fmt(snap.spot)} sub={`mid ${fmt(snap.micro.microPrice)}`} />
            <StatChip label="pnl" value={`${pnl > 0 ? "+" : ""}${fmt(pnl, 1)}`} tone={pnlTone} sub={`${pnlPct > 0 ? "+" : ""}${pnlPct.toFixed(1)}% · eq ${fmt(snap.equity, 0)}`} />
            <StatChip label="perp pos" value={`${snap.position > 0 ? "+" : ""}${snap.position}`} tone={snap.position > 0 ? "pos" : snap.position < 0 ? "neg" : "neutral"} sub={`net Δ ${fmt(snap.risk.netDelta, 0)} / ${snap.risk.netDeltaLimit}`} />
            <StatChip label="net vega" value={fmt(o.net.vega, 0)} tone={o.net.vega > 0 ? "pos" : "neg"} sub={`limit ${snap.risk.vegaLimit}`} />
            <StatChip label="opt fills" value={`${o.fills}`} sub={`hedges ${o.hedgeLotsTotal}`} />
            <StatChip label="markout ×" value={fmt(snap.markouts.multiplier)} tone={snap.markouts.multiplier > 1.3 ? "warn" : "neutral"} sub={`tox ${(snap.markouts.toxicity * 100).toFixed(0)}%`} />
            <StatChip label="CU" value={`${snap.perf.cuPct.toFixed(0)}%`} tone={snap.perf.cuPct > 80 ? "warn" : "accent"} sub={`${snap.perf.cu}/${snap.perf.cuBudget}`} />
            <StatChip label="tick p50" value={snap.perf.p50 > 0 ? `${(snap.perf.p50 / 1000).toFixed(0)}µs` : "<1µs"} sub={`p99 ${(snap.perf.p99 / 1000).toFixed(0)}µs`} />
            <StatChip label="sim clock" value={`${Math.floor(snap.clock / 60)}m ${Math.floor(snap.clock % 60)}s`} sub={`${snap.steps} steps`} />
          </div>
          <div className="ml-auto flex items-center gap-2">
            <div className="flex items-center gap-1.5">
              <span className="text-[9px] uppercase tracking-wider text-neutral-500">speed</span>
              <Slider
                value={[eng.speed]}
                min={1}
                max={40}
                step={1}
                onValueChange={(v) => eng.setSpeed(v[0])}
                className="w-24"
                aria-label="simulation speed"
              />
              <span className="font-mono text-[10px] text-neutral-400 w-6">{eng.speed}x</span>
            </div>
            <Button size="sm" variant="outline" onClick={eng.running ? eng.pause : eng.play} aria-label={eng.running ? "pause" : "play"} className="h-7 px-2.5">
              {eng.running ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
            </Button>
            <Button size="sm" variant="outline" onClick={() => eng.reset()} aria-label="reset" className="h-7 px-2.5">
              <RotateCcw className="h-3 w-3" />
            </Button>
          </div>
        </div>
        {!snap.running && (
          <div className="max-w-[1440px] mx-auto mt-1.5 flex items-center gap-3">
            <Badge variant="destructive" className="text-[10px] font-mono">HALTED — {snap.risk.haltReason}</Badge>
            <span className="font-mono text-[9px] text-neutral-500">quotes off · hedge + marks live · auto risk-on in 30 s</span>
            <Button size="sm" variant="outline" onClick={eng.resume} className="h-6 px-2 text-[10px] font-mono border-amber-600/50 text-amber-300 hover:bg-amber-500/10">
              risk-manager resume
            </Button>
          </div>
        )}
        {(snap.risk.state === "gated" || snap.risk.state === "breach") && snap.running && (
          <div className="max-w-[1440px] mx-auto mt-1.5 flex flex-wrap items-center gap-2">
            <Badge className="text-[10px] font-mono border-amber-600/50 bg-amber-500/10 text-amber-300 hover:bg-amber-500/10">
              RISK-{snap.risk.state.toUpperCase()} · {snap.risk.gatedOn} — {snap.risk.netDelta > 0 ? "bids gated" : snap.risk.netDelta < 0 ? "asks gated" : "intake gated"}, unwind side live
            </Badge>
            <span className="font-mono text-[9px] text-neutral-500">self-recovering: desk trades out instead of halting</span>
          </div>
        )}
      </header>

      {/* ------------------------------------------------ body */}
      <main className="max-w-[1440px] mx-auto px-3 py-3">
        <Tabs value={tab} onValueChange={setTab}>
          <TabsList className="h-8 bg-neutral-900 border border-neutral-800 w-full justify-start overflow-x-auto rounded-md">
            <TabsTrigger value="terminal" className="text-[11px] gap-1.5 data-[state=active]:bg-emerald-500/15 data-[state=active]:text-emerald-300"><Activity className="h-3 w-3" />Terminal</TabsTrigger>
            <TabsTrigger value="chain" className="text-[11px] gap-1.5 data-[state=active]:bg-emerald-500/15 data-[state=active]:text-emerald-300"><Layers className="h-3 w-3" />Option Chain</TabsTrigger>
            <TabsTrigger value="surface" className="text-[11px] gap-1.5 data-[state=active]:bg-emerald-500/15 data-[state=active]:text-emerald-300"><Grid3x3 className="h-3 w-3" />Vol Surface</TabsTrigger>
            <TabsTrigger value="greeks" className="text-[11px] gap-1.5 data-[state=active]:bg-emerald-500/15 data-[state=active]:text-emerald-300"><Scale className="h-3 w-3" />Greeks & Hedge</TabsTrigger>
            <TabsTrigger value="rfq" className="text-[11px] gap-1.5 data-[state=active]:bg-emerald-500/15 data-[state=active]:text-emerald-300"><Handshake className="h-3 w-3" />RFQ · Margin</TabsTrigger>
            <TabsTrigger value="strategy" className="text-[11px] gap-1.5 data-[state=active]:bg-emerald-500/15 data-[state=active]:text-emerald-300"><Brain className="h-3 w-3" />Strategy</TabsTrigger>
            <TabsTrigger value="risk" className="text-[11px] gap-1.5 data-[state=active]:bg-emerald-500/15 data-[state=active]:text-emerald-300"><ShieldAlert className="h-3 w-3" />Micro · Risk · PnL</TabsTrigger>
          </TabsList>

          {/* ============================================ TAB: terminal */}
          <TabsContent value="terminal" className="mt-3 grid gap-3 lg:grid-cols-3">
            <Panel title="perp quote ladder — strategy output" className="lg:col-span-1">
              <DepthLadder bid={snap.quotes.bid} ask={snap.quotes.ask} mid={snap.spot} tickLabel="dist = |price − mid|, bar = size" />
              <div className="mt-2 grid grid-cols-2 gap-2 font-mono text-[10px]">
                <div className="rounded border border-neutral-800 bg-neutral-900/50 p-2">
                  <div className="text-neutral-500 text-[9px] uppercase">book bid / ask</div>
                  <div className="mt-1 flex justify-between">
                    <span className="text-emerald-400">{fmt(snap.book.bid)} × {snap.book.bidSz}</span>
                    <span className="text-rose-400">{fmt(snap.book.ask)} × {snap.book.askSz}</span>
                  </div>
                </div>
                <div className="rounded border border-neutral-800 bg-neutral-900/50 p-2">
                  <div className="text-neutral-500 text-[9px] uppercase">micro-price vs mid</div>
                  <div className="mt-1 flex justify-between">
                    <span className="text-teal-300">{fmt(snap.micro.microPrice)}</span>
                    <span className={snap.micro.microPrice > snap.spot ? "text-emerald-400" : "text-rose-400"}>
                      {snap.micro.microPrice > snap.spot ? "↑" : "↓"} {fmt(Math.abs(snap.micro.microPrice - snap.spot), 3)}
                    </span>
                  </div>
                </div>
              </div>
            </Panel>

            <Panel title="mid price — latent fair + displayed book" className="lg:col-span-2">
              <LineChart data={snap.midHistory} h={150} color="#e4e4e7" label={`spot ${fmt(snap.spot)} · σ_fast ${(snap.micro.sigmaFast * 100).toFixed(3)}%/√s`} />
              <div className="mt-2 grid grid-cols-4 gap-3">
                <StatChip label="σ fast" value={`${(snap.micro.sigmaFast * 100).toFixed(3)}%`} sub="per √s" />
                <StatChip label="σ slow" value={`${(snap.micro.sigmaSlow * 100).toFixed(3)}%`} sub="per √s" />
                <StatChip label="hurst" value={isFinite(snap.micro.hurst) ? fmt(snap.micro.hurst, 2) : "…"} sub={isFinite(snap.micro.hurst) && snap.micro.hurst < 0.45 ? "rough regime" : "MBM regime"} />
                <StatChip label="roll spread" value={fmt(snap.micro.rollSpread, 3)} sub="effective" />
              </div>
            </Panel>

            <Panel title="equity curve — total book" className="lg:col-span-2">
              <LineChart data={snap.equityHistory} h={140} color="#34d399" baseline={0} label={`equity ${fmt(snap.equity)} · drawdown ${fmt(snap.risk.drawdown)}`} />
            </Panel>

            <Panel title="markouts · toxicity · adaptive spread">
              <div className="space-y-2.5">
                <div>
                  <div className="flex justify-between text-[10px] font-mono mb-1">
                    <span className="text-neutral-500">markout ratio</span>
                    <span className={signed(snap.markouts.ratio)}>{fmt(snap.markouts.ratio, 3)}</span>
                  </div>
                  <SignedBar value={snap.markouts.ratio} max={2} />
                </div>
                <div>
                  <div className="flex justify-between text-[10px] font-mono mb-1">
                    <span className="text-neutral-500">toxicity [0,1]</span>
                    <span className="text-amber-400">{(snap.markouts.toxicity * 100).toFixed(1)}%</span>
                  </div>
                  <SignedBar value={snap.markouts.toxicity} max={1} colorPos="#f59e0b" colorNeg="#f59e0b" />
                </div>
                <div className="rounded border border-neutral-800 bg-neutral-900/50 p-2 font-mono text-[10px] space-y-1">
                  <div className="flex justify-between"><span className="text-neutral-500">spread multiplier</span><span className="text-amber-300">×{fmt(snap.markouts.multiplier, 2)}</span></div>
                  <div className="flex justify-between"><span className="text-neutral-500">resolved markouts</span><span>{snap.markouts.resolved}</span></div>
                  <div className="flex justify-between"><span className="text-neutral-500">jump stat z</span><span className={snap.micro.jumpStat > 3.5 ? "text-rose-400" : ""}>{fmt(snap.micro.jumpStat, 2)}</span></div>
                  <div className="flex justify-between"><span className="text-neutral-500">jump exceed rate</span><span>{(snap.micro.jumpExceed * 100).toFixed(1)}%</span></div>
                </div>
              </div>
            </Panel>
          </TabsContent>

          {/* ============================================ TAB: option chain */}
          <TabsContent value="chain" className="mt-3 space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[10px] uppercase tracking-wider text-neutral-500">expiry</span>
              {o.expiries.map((t, i) => (
                <button
                  key={t}
                  onClick={() => eng.setSelectedExp(i)}
                  className={`rounded px-2.5 py-1 font-mono text-[10px] border transition-colors ${i === o.selectedExp ? "border-emerald-500/50 bg-emerald-500/15 text-emerald-300" : "border-neutral-800 bg-neutral-900/60 text-neutral-400 hover:border-neutral-700"}`}
                >
                  {expLabel(t)}
                </button>
              ))}
              <span className="ml-3 font-mono text-[10px] text-neutral-500">
                T = {(cur * 365).toFixed(0)}d · ATM {vp(o.atmTerm[o.selectedExp])} vol pts · quotes = SSVI fair {o.vvEnabled ? "+ vanna–volga overhedge" : ""} ± GLFT vega spread
              </span>
            </div>

            <Panel title={`option chain — ${expLabel(cur)} · calls | puts`}>
              <div className="overflow-x-auto">
                <table className="w-full text-[10px] font-mono tabular-nums">
                  <thead>
                    <tr className="text-neutral-500 border-b border-neutral-800">
                      <th className="text-left py-1.5 font-normal">k</th>
                      <th className="text-right font-normal">strike</th>
                      <th className="text-right font-normal text-emerald-500/80">bid IV</th>
                      <th className="text-right font-normal">fair IV</th>
                      <th className="text-right font-normal text-rose-500/80">ask IV</th>
                      <th className="text-right font-normal text-amber-500/80">VV shift</th>
                      <th className="text-right font-normal">bid ¥</th>
                      <th className="text-right font-normal">ask ¥</th>
                      <th className="text-right font-normal">Δ</th>
                      <th className="text-right font-normal">Γ</th>
                      <th className="text-right font-normal">vega</th>
                      <th className="text-right font-normal">vanna</th>
                      <th className="text-right font-normal">volga</th>
                      <th className="text-right font-normal text-emerald-500/80">C bid IV</th>
                      <th className="text-right font-normal text-rose-500/80">C ask IV</th>
                      <th className="text-right font-normal">C pos</th>
                      <th className="text-right font-normal text-emerald-500/80">P bid IV</th>
                      <th className="text-right font-normal text-rose-500/80">P ask IV</th>
                      <th className="text-right font-normal">P pos</th>
                    </tr>
                  </thead>
                  <tbody>
                    {o.chain.map((row) => {
                      const g = row.greeks;
                      const posCls = (v: number) => (v > 0 ? "text-emerald-400" : v < 0 ? "text-rose-400" : "text-neutral-600");
                      return (
                        <tr key={row.moneyness} className={`border-b border-neutral-900 ${row.moneyness === 0 ? "bg-neutral-800/40" : ""}`}>
                          <td className="text-left py-1 text-neutral-400">{row.moneyness === 0 ? "ATM" : fmt(row.moneyness * 100, 1)}</td>
                          <td className="text-right text-neutral-200">{fmt(row.strike, 1)}</td>
                          <td className="text-right text-emerald-400">{vp(row.put.ivBid)}</td>
                          <td className="text-right text-neutral-300">{vp(row.put.ivFair)}</td>
                          <td className="text-right text-rose-400">{vp(row.put.ivAsk)}</td>
                          <td className="text-right text-amber-400/90">{vp(row.put.vvShift, 3)}</td>
                          <td className="text-right text-neutral-400">{fmt(row.put.premBid, 2)}</td>
                          <td className="text-right text-neutral-400">{fmt(row.put.premAsk, 2)}</td>
                          <td className="text-right text-neutral-400">{fmt(g.delta, 3)}</td>
                          <td className="text-right text-neutral-500">{fmt(g.gamma, 4)}</td>
                          <td className="text-right text-neutral-400">{fmt(g.vega, 2)}</td>
                          <td className="text-right text-neutral-500">{fmt(g.vanna, 4)}</td>
                          <td className="text-right text-neutral-500">{fmt(g.volga, 4)}</td>
                          <td className="text-right text-emerald-400">{vp(row.call.ivBid)}</td>
                          <td className="text-right text-rose-400">{vp(row.call.ivAsk)}</td>
                          <td className={`text-right ${posCls(row.call.pos)}`}>{row.call.pos || "·"}</td>
                          <td className="text-right text-emerald-400">{vp(row.put.ivBid)}</td>
                          <td className="text-right text-rose-400">{vp(row.put.ivAsk)}</td>
                          <td className={`text-right ${posCls(row.put.pos)}`}>{row.put.pos || "·"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Panel>

            <div className="grid gap-3 lg:grid-cols-2">
              <Panel title="smile slice — bid / fair / ask IV">
                <SmileChart chain={o.chain} />
              </Panel>
              <Panel title="wing instruments · vanna–volga overhedge">
                <div className="overflow-x-auto">
                  <table className="w-full text-[10px] font-mono tabular-nums">
                    <thead>
                      <tr className="text-neutral-500 border-b border-neutral-800">
                        <th className="text-left py-1.5 font-normal">expiry</th>
                        <th className="text-right font-normal">ATM IV</th>
                        <th className="text-right font-normal text-amber-500/80">25Δ RR</th>
                        <th className="text-right font-normal text-amber-500/80">25Δ BF</th>
                      </tr>
                    </thead>
                    <tbody>
                      {o.wings.map((w) => (
                        <tr key={w.t} className={`border-b border-neutral-900 ${w.t === cur ? "bg-neutral-800/40" : ""}`}>
                          <td className="py-1 text-neutral-400">{expLabel(w.t)}</td>
                          <td className="text-right text-neutral-300">{vp(w.sigAtm)}</td>
                          <td className="text-right text-amber-400/90">{fmt(w.rrIv, 2)}</td>
                          <td className="text-right text-amber-400/90">{fmt(w.bfIv, 2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="mt-2 grid grid-cols-3 gap-2">
                  <StatChip label="portfolio charge" value={fmt(o.vvPortfolio.charge, 3)} tone="warn" sub="$ per wing hedge" />
                  <StatChip label="weight RR" value={fmt(o.vvPortfolio.wRr, 2)} sub="vanna hedge" />
                  <StatChip label="weight BF" value={fmt(o.vvPortfolio.volShift, 3)} sub="volga hedge (vol pts)" />
                </div>
              </Panel>
            </div>

            <Panel title="recent option fills — client flow at our quotes">
              {o.recentFills.length === 0 ? (
                <div className="text-[10px] text-neutral-500 font-mono">waiting for client requests… (Cox intensity on the vol spread)</div>
              ) : (
                <div className="max-h-48 overflow-y-auto">
                  <table className="w-full text-[10px] font-mono tabular-nums">
                    <thead className="sticky top-0 bg-neutral-900">
                      <tr className="text-neutral-500 border-b border-neutral-800">
                        <th className="text-left py-1 font-normal">expiry</th>
                        <th className="text-left font-normal">k</th>
                        <th className="text-left font-normal">side</th>
                        <th className="text-right font-normal">traded IV</th>
                        <th className="text-right font-normal">fair IV</th>
                        <th className="text-right font-normal">edge (vol pts)</th>
                        <th className="text-right font-normal">premium</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...o.recentFills].reverse().map((f, i) => {
                        const edge = f.side === "bid" ? f.fairIv - f.iv : f.iv - f.fairIv;
                        return (
                          <tr key={i} className="border-b border-neutral-900">
                            <td className="py-1 text-neutral-400">{expLabel(o.expiries[f.expIdx])}</td>
                            <td className="text-neutral-400">{fmt(o.moneyness[f.legIdx] * 100, 1)}</td>
                            <td className={f.side === "bid" ? "text-emerald-400" : "text-rose-400"}>{f.side === "bid" ? "we buy" : "we sell"}</td>
                            <td className="text-right text-neutral-200">{vp(f.iv)}</td>
                            <td className="text-right text-neutral-500">{vp(f.fairIv)}</td>
                            <td className={`text-right ${edge > 0 ? "text-emerald-400" : "text-rose-400"}`}>{vp(edge)}</td>
                            <td className="text-right text-neutral-400">{fmt(f.premium, 2)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </Panel>
          </TabsContent>

          {/* ============================================ TAB: vol surface */}
          <TabsContent value="surface" className="mt-3 grid gap-3 lg:grid-cols-2">
            <Panel title="SSVI implied vol surface — Gatheral–Jacquier (2014)" className="lg:col-span-2">
              <Heatmap
                grid={o.surfaceGrid}
                rowLabels={o.expiries.map(expLabel)}
                colLabels={o.moneyness.map((k) => (k === 0 ? "ATM" : fmt(k * 100, 1)))}
                format={(v) => vp(v, 1)}
                unit=" vol pts"
              />
              <div className="mt-3 flex flex-wrap gap-2">
                <Badge variant="outline" className={`text-[10px] font-mono border ${o.arb.butterflyCondition ? "border-emerald-600/40 text-emerald-400" : "border-rose-600/40 text-rose-400"}`}>
                  butterfly θφ²(1+|ρ|) ≤ 4 {o.arb.butterflyCondition ? "✓" : "✗"}
                </Badge>
                <Badge variant="outline" className={`text-[10px] font-mono border ${o.arb.butterflyGrid ? "border-emerald-600/40 text-emerald-400" : "border-rose-600/40 text-rose-400"}`}>
                  density g(k) ≥ 0 {o.arb.butterflyGrid ? "✓" : "✗"}
                </Badge>
                <Badge variant="outline" className={`text-[10px] font-mono border ${o.arb.calendar ? "border-emerald-600/40 text-emerald-400" : "border-rose-600/40 text-rose-400"}`}>
                  calendar w(k,T₂) ≥ w(k,T₁) {o.arb.calendar ? "✓" : "✗"}
                </Badge>
              </div>
            </Panel>

            <Panel title="ATM term structure — vol-of-vol driven pillars">
              <LineChart data={o.atmTerm.map((v) => v * 100)} w={560} h={150} color="#f59e0b" label="ATM IV (vol pts) per expiry" />
              <div className="mt-1 flex justify-between text-[9px] font-mono text-neutral-500">
                {o.expiries.map((t) => <span key={t}>{expLabel(t)}</span>)}
              </div>
            </Panel>

            <Panel title="surface model">
              <div className="font-mono text-[10px] leading-relaxed text-neutral-400 space-y-1.5">
                <div>w(k,T) = θ/2 · (1 + ρφk + √((φk+ρ)² + 1−ρ²))</div>
                <div>φ(θ) = 1/(η·θ^γ·(1+θ)^(1−γ))&nbsp;&nbsp;— Heston-like</div>
                <div className="text-neutral-500">θ(T) interpolated, flat-vol extrapolated, monotone-repaired at construction.</div>
                <div className="pt-1 border-t border-neutral-800">Pillars re-driven every step by a correlated vol-of-vol GBM (annual ν = {(0.9).toFixed(1)}), per-expiry term noise OU.</div>
                <div className="pt-1 border-t border-neutral-800">ρ = −0.70 · η = 1.0 · γ = 0.5 · 5 expiries × 7 strikes</div>
              </div>
            </Panel>
          </TabsContent>

          {/* ============================================ TAB: greeks & hedge */}
          <TabsContent value="greeks" className="mt-3 grid gap-3 lg:grid-cols-3">
            <Panel title="net book greeks — all expiries" className="lg:col-span-2">
              <div className="grid grid-cols-2 gap-x-6 gap-y-3">
                {([
                  ["delta", o.net.delta, 5, "per unit spot"],
                  ["gamma", o.net.gamma, 0.3, "per spot²"],
                  ["vega", o.net.vega, 4000, "$ per unit vol"],
                  ["theta", o.net.theta, 30, "per year"],
                  ["vanna", o.net.vanna, 40, "∂²V/∂S∂σ"],
                  ["volga", o.net.volga, 300, "∂²V/∂σ²"],
                ] as const).map(([label, v, max, sub]) => (
                  <div key={label}>
                    <div className="flex justify-between text-[10px] font-mono mb-1">
                      <span className="text-neutral-500 uppercase">{label}</span>
                      <span className={signed(v)}>{fmt(v, Math.abs(v) < 1 ? 4 : 2)}</span>
                    </div>
                    <SignedBar value={v} max={max} />
                    <div className="text-[9px] text-neutral-600 font-mono mt-0.5">{sub}</div>
                  </div>
                ))}
              </div>
              <div className="mt-3 pt-3 border-t border-neutral-800 grid grid-cols-4 gap-3">
                <StatChip label="book mark" value={fmt(o.net.mark, 2)} sub="MTM value" />
                <StatChip label="premium cash" value={fmt(snap.pnl.optionPremium, 2)} sub="client premiums" />
                <StatChip label="unhedged Δ" value={`${fmt(o.unhedgedDeltaLots, 1)} lots`} tone={Math.abs(o.unhedgedDeltaLots) > o.hedgeBand ? "neg" : "pos"} sub={`band ±${fmt(o.hedgeBand, 2)}`} />
                <StatChip label="hedge volume" value={`${o.hedgeLotsTotal}`} sub="lots traded" />
              </div>
            </Panel>

            <Panel title="WW delta-hedge band — Whalley–Wilmott">
              <WwBandViz unhedged={o.unhedgedDeltaLots} band={o.hedgeBand} />
              <div className="mt-2 font-mono text-[10px] text-neutral-500 space-y-1">
                <div>H* = (3c/(γS²))^(1/3) — re-derived & MC-validated</div>
                <div>hedge fires only on band exits; trades the excess, not the flat.</div>
                <div className="pt-1.5 border-t border-neutral-800 flex items-center gap-2">
                  <Switch checked={eng.snap?.options.vvEnabled ?? false} onCheckedChange={(c) => eng.setParam("vvOn", c)} />
                  <span className="text-neutral-400">vanna–volga overhedge {o.vvEnabled ? "ON" : "OFF"}</span>
                </div>
              </div>
            </Panel>

            <Panel title="vega per expiry — term exposure">
              <div className="grid gap-2 sm:grid-cols-5">
                {o.expiries.map((t, ei) => {
                  const vE = o.vegaPerExp?.[ei] ?? 0;
                  const maxV = Math.max(...(o.vegaPerExp ?? [0]).map(Math.abs), 1);
                  return (
                    <button
                      key={t}
                      onClick={() => eng.setSelectedExp(ei)}
                      className={`text-left rounded border p-2 transition-colors ${ei === o.selectedExp ? "border-emerald-500/50 bg-emerald-500/10" : "border-neutral-800 bg-neutral-900/50 hover:border-neutral-700"}`}
                    >
                      <div className="text-[9px] uppercase text-neutral-500">{expLabel(t)}</div>
                      <div className={`font-mono text-sm ${signed(vE)}`}>{fmt(vE, 1)} <span className="text-[9px] text-neutral-500">vega</span></div>
                      <div className="mt-1"><SignedBar value={vE} max={maxV} height={6} /></div>
                      <div className="text-[9px] text-neutral-500 font-mono mt-1">ATM {vp(o.atmTerm[ei])}</div>
                    </button>
                  );
                })}
              </div>
              <div className="mt-2 text-[9px] font-mono text-neutral-600">
                net vega {fmt(o.net.vega, 0)} $/vol across {o.expiries.length} expiries · GLFT vega-space inventory q = {fmt(Math.round(o.net.vega / 25), 0)} fills · vanna {fmt(o.net.vanna, 2)} · volga {fmt(o.net.volga, 2)}
              </div>
            </Panel>
          </TabsContent>

          {/* ============================================ TAB: rfq / margin */}
          <TabsContent value="rfq" className="mt-3 grid gap-3 lg:grid-cols-3">
            <Panel title="RFQ desk — multi-leg packages · firm quotes (TTL) · atomic execution" className="lg:col-span-2">
              <div className="space-y-3">
                <div className="flex flex-wrap items-center gap-1.5">
                  {RFQ_TEMPLATES.map((t) => (
                    <button
                      key={t.id}
                      onClick={() => setRfqTpl(t.id)}
                      title={t.blurb}
                      className={`rounded border px-2 py-1 font-mono text-[10px] transition-colors ${
                        rfqTpl === t.id
                          ? "border-emerald-600/60 bg-emerald-500/15 text-emerald-300"
                          : "border-neutral-800 bg-neutral-900/60 text-neutral-400 hover:border-neutral-700 hover:text-neutral-200"
                      }`}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>
                <div className="flex items-center gap-3">
                  <div className="flex-1">
                    <div className="flex justify-between text-[10px] font-mono mb-1">
                      <span className="text-neutral-500">package size (lots)</span>
                      <span className="text-neutral-200">{rfqLots}</span>
                    </div>
                    <Slider value={[rfqLots]} min={5} max={25} step={1} onValueChange={(v) => setRfqLots(v[0])} aria-label="rfq lots" />
                  </div>
                  <Button
                    size="sm"
                    onClick={() => {
                      const id = eng.requestRfq(rfqTpl, rfqLots);
                      if (id !== null) {
                        firmRef.current[id] = Date.now() + 15000;
                        setManualFirm({ ...firmRef.current });
                      }
                      setRfqMsg(id === null ? "unknown template" : `quoted #${id} — firm for 15 s wall-clock (Paradigm/Derive hold-for-time)`);
                    }}
                    className="h-7 px-3 text-[10px] font-mono border-emerald-600/50 bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20"
                    variant="outline"
                  >
                    request quote
                  </Button>
                </div>
                {rfqMsg && <div className="font-mono text-[9px] text-neutral-500">{rfqMsg}</div>}
                <div className="space-y-2">
                  {snap.rfq.active.map((q) => (
                    <div key={q.id} className="rounded border border-neutral-800 bg-neutral-900/50 p-2">
                      <div className="flex flex-wrap items-center gap-2 mb-1.5">
                        <span className="font-mono text-[11px] text-neutral-200">{q.label}</span>
                        <Badge className={`text-[9px] font-mono ${q.origin === "manual" ? "border-sky-600/50 bg-sky-500/10 text-sky-300" : "border-violet-600/50 bg-violet-500/10 text-violet-300"}`}>
                          {q.origin === "manual" ? "MANUAL TAKER" : "INSTITUTIONAL"}
                        </Badge>
                        <Badge className="text-[9px] font-mono border-amber-600/40 bg-amber-500/10 text-amber-300">
                          offset {(q.marginOffset * 100).toFixed(0)}%
                        </Badge>
                        <span className="font-mono text-[9px] text-neutral-500">{q.lots} lots</span>
                        <div className="ml-auto flex items-center gap-2">
                          <div className="w-16 h-1.5 rounded bg-neutral-800 overflow-hidden">
                            {q.origin === "manual" ? (
                              <div className="h-full bg-sky-500/70" style={{ width: `${Math.max(0, Math.min(100, (((manualFirm[q.id] ?? Date.now()) - Date.now()) / 15000) * 100))}%` }} />
                            ) : (
                              <div className="h-full bg-amber-500/70" style={{ width: `${Math.max(0, Math.min(100, (q.ttl / q.totalTtl) * 100))}%` }} />
                            )}
                          </div>
                          <span className="font-mono text-[9px] text-neutral-500">
                            {q.origin === "manual"
                              ? `${Math.max(0, ((manualFirm[q.id] ?? Date.now()) - Date.now()) / 1000).toFixed(1)}s`
                              : `${q.ttl.toFixed(1)}s`}
                          </span>
                        </div>
                      </div>
                      <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 font-mono text-[9px]">
                        {q.legs.map((l, i) => (
                          <div key={i} className="contents">
                            <span className={l.dir > 0 ? "text-emerald-400" : "text-rose-400"}>
                              {l.dir > 0 ? "+" : "−"}{l.kind === "call" ? "C" : "P"} {fmt(l.strike, 1)} {expLabel(l.t)}
                            </span>
                            <span className="text-neutral-500">
                              iv {vp(l.ivBid, 1)} / {vp(l.ivAsk, 1)} · leg fair {fmt(l.premFair, 2)}
                            </span>
                          </div>
                        ))}
                      </div>
                      <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[10px]">
                        <span className="text-emerald-400">desk buys @ {fmt(q.bid)}</span>
                        <span className="text-neutral-400">fair {fmt(q.fair)}</span>
                        <span className="text-rose-400">desk sells @ {fmt(q.ask)}</span>
                        <span className="text-neutral-500">Δ {fmt(q.delta, 1)} · Γ {fmt(q.gamma, 3)} · V {fmt(q.vega, 0)}</span>
                        {q.note && <span className="text-amber-400/80 text-[9px]">{q.note}</span>}
                      </div>
                      <div className="mt-1.5 flex gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          disabled={q.status !== "quoted"}
                          onClick={() => {
                            const res = eng.executeRfq(q.id, "desk-buys");
                            setRfqMsg(res.ok ? `executed #${q.id} ${q.label} — desk BUYS @ ${fmt(q.bid)}` : `refused: ${res.reason}`);
                          }}
                          className="h-6 px-2 text-[9px] font-mono border-emerald-600/50 text-emerald-300 hover:bg-emerald-500/10"
                        >
                          execute — desk buys
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          disabled={q.status !== "quoted"}
                          onClick={() => {
                            const res = eng.executeRfq(q.id, "desk-sells");
                            setRfqMsg(res.ok ? `executed #${q.id} ${q.label} — desk SELLS @ ${fmt(q.ask)}` : `refused: ${res.reason}`);
                          }}
                          className="h-6 px-2 text-[9px] font-mono border-rose-600/50 text-rose-300 hover:bg-rose-500/10"
                        >
                          execute — desk sells
                        </Button>
                      </div>
                    </div>
                  ))}
                  {!snap.rfq.active.length && (
                    <div className="rounded border border-dashed border-neutral-800 p-3 font-mono text-[10px] text-neutral-600">
                      no firm quotes — request one above, or wait for the institutional flow (arrivals every ~40 s)
                    </div>
                  )}
                </div>
                <div className="grid grid-cols-4 gap-2 pt-1">
                  <StatChip label="requests" value={`${snap.rfq.requests}`} />
                  <StatChip label="executed" value={`${snap.rfq.executed}`} tone="pos" />
                  <StatChip label="refused" value={`${snap.rfq.refused}`} tone={snap.rfq.refused > 0 ? "warn" : "neutral"} sub="risk-gated" />
                  <StatChip label="premium flow" value={fmt(snap.rfq.premiumFlow, 0)} tone={snap.rfq.premiumFlow > 0 ? "pos" : "neg"} />
                </div>
                <div className="rounded border border-neutral-800 bg-neutral-900/50 p-2">
                  <div className="text-[9px] uppercase text-neutral-500 mb-1 font-mono">recent package executions</div>
                  <div className="space-y-0.5 max-h-24 overflow-y-auto">
                    {[...snap.rfq.executions].reverse().map((ev) => (
                      <div key={`${ev.id}-${ev.t}`} className="flex gap-2 font-mono text-[9px]">
                        <span className="text-neutral-600 w-10 shrink-0">{ev.t}s</span>
                        <span className="text-neutral-400 w-28 shrink-0 truncate">{ev.label}</span>
                        <span className={ev.side === "desk-buys" ? "text-emerald-400" : "text-rose-400"}>{ev.side === "desk-buys" ? "BUY" : "SELL"}</span>
                        <span className="text-neutral-400">@ {fmt(ev.price)}</span>
                        <span className="text-neutral-500">{ev.lots} lots</span>
                        <span className="text-neutral-600">Δ {fmt(ev.delta, 0)} V {fmt(ev.vega, 0)}</span>
                        <span className="ml-auto text-neutral-700">{ev.origin}</span>
                      </div>
                    ))}
                    {!snap.rfq.executions.length && <div className="font-mono text-[9px] text-neutral-600">—</div>}
                  </div>
                </div>
              </div>
            </Panel>

            <Panel title="portfolio margin — SFPM scenario scan + SOMC floor">
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-[10px] font-mono mb-1">
                    <span className="text-neutral-500">margin utilization</span>
                    <span className={snap.margin.utilization >= 0.9 ? "text-rose-400" : snap.margin.utilization >= 0.7 ? "text-amber-300" : "text-neutral-300"}>
                      {vp(snap.margin.utilization, 1)}%
                    </span>
                  </div>
                  <div className="relative">
                    <SignedBar value={snap.margin.utilization} max={1.4} colorPos={snap.margin.utilization >= 0.9 ? "#fb7185" : snap.margin.utilization >= 0.7 ? "#f59e0b" : "#34d399"} colorNeg="#34d399" />
                    <div className="absolute top-0 bottom-0 border-l border-dashed border-amber-600/60" style={{ left: `${(0.7 / 1.4) * 100}%` }} />
                    <div className="absolute top-0 bottom-0 border-l border-dashed border-rose-600/60" style={{ left: `${(0.9 / 1.4) * 100}%` }} />
                  </div>
                  <div className="flex justify-between text-[8px] font-mono text-neutral-600 mt-0.5">
                    <span>0</span><span className="text-amber-600">gated 70%</span><span className="text-rose-600">wind-down 90%</span><span>140%</span>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <StatChip label="initial margin" value={fmt(snap.margin.initial, 0)} sub="1.2× maintenance" />
                  <StatChip label="maintenance" value={fmt(snap.margin.maintenance, 0)} sub="scan ∨ SOMC" />
                  <StatChip label="SOMC floor" value={fmt(snap.margin.somc, 0)} tone="accent" sub="gross short wings" />
                  <StatChip label="scan loss" value={fmt(snap.margin.scanningLoss, 0)} sub={`worst: ${snap.margin.worst}`} />
                </div>
                <div className="rounded border border-neutral-800 bg-neutral-900/50 p-2 space-y-1">
                  <div className="text-[9px] uppercase text-neutral-500 font-mono mb-0.5">scenario grid losses</div>
                  {snap.margin.perScenario.map((sc) => (
                    <div key={sc.name}>
                      <div className="flex justify-between text-[9px] font-mono">
                        <span className="text-neutral-500">{sc.name}</span>
                        <span className={sc.loss > snap.margin.scanningLoss * 0.85 ? "text-amber-300" : "text-neutral-400"}>{fmt(sc.loss, 1)}</span>
                      </div>
                      <div className="h-1 rounded bg-neutral-800 overflow-hidden">
                        <div
                          className="h-full"
                          style={{
                            width: `${Math.min(100, (sc.loss / Math.max(snap.margin.scanningLoss, 1)) * 100)}%`,
                            background: sc.loss > snap.margin.scanningLoss * 0.85 ? "#f59e0b" : "#52525b",
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </Panel>

            <Panel title="funding — premium-index pin · carry-aware quotes">
              <div className="space-y-2.5">
                <div className="grid grid-cols-2 gap-2">
                  <StatChip label="funding rate" value={`${vp(snap.funding.rate, 4)}%`} tone={snap.funding.rate > 0 ? "neg" : "pos"} sub="per 8h interval" />
                  <StatChip label="premium index" value={`${vp(snap.funding.premiumIndex, 3)}%`} tone={snap.funding.premiumIndex > 0 ? "neg" : "pos"} sub="perp vs index" />
                  <StatChip label="next payment" value={`${Math.floor(snap.funding.nextIn / 3600)}h ${Math.floor((snap.funding.nextIn % 3600) / 60)}m`} sub="longs pay when +" />
                  <StatChip label="funding paid" value={fmt(snap.funding.paid, 1)} tone={snap.funding.paid > 0 ? "neg" : "pos"} sub="cumulative" />
                  <StatChip label="index" value={fmt(snap.funding.index)} sub="funding anchor" />
                  <StatChip label="perp mid" value={fmt(snap.spot)} sub={`basis ${vp(snap.funding.premiumIndex, 2)}%`} />
                </div>
                <div>
                  <div className="flex justify-between text-[10px] font-mono mb-1">
                    <span className="text-neutral-500">funding carry skew on quotes</span>
                    <span className={signed(snap.funding.carrySkew)}>{fmt(snap.funding.carrySkew, 3)}</span>
                  </div>
                  <SignedBar value={snap.funding.carrySkew} max={10 * 0.5} colorPos="#38bdf8" colorNeg="#38bdf8" />
                  <div className="text-[8px] font-mono text-neutral-600 mt-0.5">
                    reservation price leans against the carry side — long inventory pays positive funding
                  </div>
                </div>
              </div>
            </Panel>

            <Panel title="adaptive engine — learned intensity · per-side toxicity">
              <div className="space-y-2.5">
                <div className="grid grid-cols-2 gap-2">
                  <StatChip label="κ assumed" value="0.50" sub="prior" />
                  <StatChip label="κ learned" value={fmt(snap.learned.kappaUsed, 2)} tone="accent" sub={`raw ${fmt(snap.learned.kappaRaw, 2)} · ${snap.learned.fills.toFixed(0)} fills`} />
                  <StatChip label="A assumed" value="0.12" sub="per second" />
                  <StatChip label="A learned" value={fmt(snap.learned.aUsed, 3)} tone="accent" sub={`raw ${fmt(snap.learned.aRaw, 3)}`} />
                </div>
                <div className="text-[9px] font-mono text-neutral-600 leading-relaxed">
                  online Cox MLE with shrinkage — the desk re-fits its fill model from realized
                  (non-swept) fills and re-optimizes AS/HJB/queue distances against the estimate.
                </div>
                <div className="grid grid-cols-3 gap-2 pt-1 border-t border-neutral-800">
                  <StatChip label="markout × bid" value={fmt(snap.markouts.multBid)} tone={snap.markouts.multBid > 1.3 ? "warn" : "neutral"} sub="toxic side widens" />
                  <StatChip label="markout × ask" value={fmt(snap.markouts.multAsk)} tone={snap.markouts.multAsk > 1.3 ? "warn" : "neutral"} sub="independently" />
                  <StatChip label="markout × net" value={fmt(snap.markouts.multiplier)} sub="aggregate" />
                </div>
                <div className="text-[9px] font-mono text-neutral-600 leading-relaxed">
                  Albers et al. (2025): fill likelihood and post-fill returns trade off per side —
                  the spread response is directional, widening only the toxic side.
                </div>
              </div>
            </Panel>
          </TabsContent>

          {/* ============================================ TAB: strategy */}
          <TabsContent value="strategy" className="mt-3 grid gap-3 lg:grid-cols-3">
            <Panel title="quoting strategies — 8 policies" className="lg:col-span-2">
              <div className="grid gap-2 sm:grid-cols-2">
                {eng.strategies.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => eng.setStrategy(s.id as StrategyId)}
                    className={`text-left rounded-md border p-2.5 transition-colors ${snap.strategy === s.id ? "border-emerald-500/50 bg-emerald-500/10" : "border-neutral-800 bg-neutral-900/50 hover:border-neutral-700"}`}
                  >
                    <div className={`text-[11px] font-medium ${snap.strategy === s.id ? "text-emerald-300" : "text-neutral-200"}`}>{s.name}</div>
                    <div className="text-[9px] text-neutral-500 mt-0.5 leading-relaxed">{s.blurb}</div>
                  </button>
                ))}
              </div>
            </Panel>

            <Panel title="live parameters">
              <div className="space-y-3.5">
                {([
                  ["gamma", "risk aversion γ*", 0.02, 1, 0.01, 2],
                  ["kappa", "fill decay κ", 0.2, 3, 0.05, 2],
                  ["a", "fill intensity A", 0.2, 3, 0.05, 2],
                  ["horizon", "horizon (s)", 60, 3600, 30, 0],
                  ["sigmaVolAnnual", "vol-of-vol (ann.)", 0.3, 2, 0.05, 2],
                  ["kappaV", "client decay κ_v", 200, 1200, 25, 0],
                  ["aV", "client intensity a_v", 0.005, 0.08, 0.0025, 4],
                  ["fillCostVol", "hedge cost floor (vol)", 0.001, 0.012, 0.0005, 4],
                  ["vegaPerFill", "vega per fill", 10, 60, 5, 0],
                ] as const).map(([key, label, min, max, step, prec]) => (
                  <ParamSlider
                    key={key}
                    eng={eng}
                    pkey={key}
                    label={label}
                    min={min}
                    max={max}
                    step={step}
                    prec={prec}
                    value={(snap.params as unknown as Record<string, number>)[key]}
                  />
                ))}
                <div>
                  <div className="text-[10px] font-mono mb-1 text-neutral-500">hedge mode</div>
                  <div className="flex gap-1.5">
                    {(["off", "every", "ww"] as const).map((m) => (
                      <button
                        key={m}
                        onClick={() => eng.setParam("hedgeMode", m)}
                        className={`rounded px-2 py-1 font-mono text-[10px] border ${m === snap.params.hedgeMode ? "border-amber-500/50 bg-amber-500/10 text-amber-300" : "border-neutral-800 bg-neutral-900/60 text-neutral-400"}`}
                      >
                        {m === "ww" ? "WW band" : m === "every" ? "every step" : "off"}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </Panel>

            <Panel title="current perp quotes — strategy output" className="lg:col-span-3">
              <div className="grid gap-3 md:grid-cols-2">
                <DepthLadder bid={snap.quotes.bid} ask={snap.quotes.ask} mid={snap.spot} />
                <div className="font-mono text-[10px] text-neutral-500 space-y-1">
                  <div>γ scaling: γ_perp = γ* · γ_opt = γ*·30000 · γ_ww = γ*·2e-5</div>
                  <div>quotes re-solved every step; HJB re-solved every 120 steps (RK4, 240 nodes, |q| ≤ 10).</div>
                  <div>spread multiplier from markout toxicity widens all levels: ×{fmt(snap.markouts.multiplier, 2)}</div>
                </div>
              </div>
            </Panel>
          </TabsContent>

          {/* ============================================ TAB: micro / risk / pnl */}
          <TabsContent value="risk" className="mt-3 grid gap-3 lg:grid-cols-3">
            <Panel title="microstructure estimators">
              <div className="space-y-2.5">
                {([
                  ["imbalance (bid share)", snap.micro.imbalance, 1, fmt(snap.micro.imbalance, 3)],
                  ["OFI (rolling lots)", snap.micro.ofi, 40, fmt(snap.micro.ofi, 1)],
                  ["OFI impact fit", snap.micro.ofiImpact, 2, fmt(snap.micro.ofiImpact, 3)],
                ] as const).map(([label, v, max, txt]) => (
                  <div key={label}>
                    <div className="flex justify-between text-[10px] font-mono mb-1">
                      <span className="text-neutral-500">{label}</span>
                      <span className="text-neutral-200">{txt}</span>
                    </div>
                    <SignedBar value={v} max={max} />
                  </div>
                ))}
                <div className="pt-2 border-t border-neutral-800 grid grid-cols-2 gap-3">
                  <StatChip label="micro-price" value={fmt(snap.micro.microPrice)} tone="accent" />
                  <StatChip label="mid" value={fmt(snap.spot)} />
                  <StatChip label="σ fast / slow" value={`${(snap.micro.sigmaFast * 100).toFixed(3)} / ${(snap.micro.sigmaSlow * 100).toFixed(3)}`} sub="% per √s" />
                  <StatChip label="roll spread" value={fmt(snap.micro.rollSpread, 3)} sub="effective" />
                  <StatChip label="jump z" value={fmt(snap.micro.jumpStat, 2)} tone={snap.micro.jumpStat > 3.5 ? "neg" : "neutral"} />
                  <StatChip label="hurst" value={isFinite(snap.micro.hurst) ? fmt(snap.micro.hurst, 2) : "…"} />
                </div>
              </div>
            </Panel>

            <Panel title="risk engine v2 — net-delta limits · directional gating">
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-[10px] font-mono mb-1">
                    <span className="text-neutral-500">net delta (unhedged combined)</span>
                    <span className={signed(snap.risk.netDelta)}>{fmt(snap.risk.netDelta, 1)} / ±{snap.risk.netDeltaLimit}</span>
                  </div>
                  <SignedBar value={snap.risk.netDelta} max={snap.risk.netDeltaLimit} />
                </div>
                <div>
                  <div className="flex justify-between text-[10px] font-mono mb-1">
                    <span className="text-neutral-500">gross hedge leg (sanity cap)</span>
                    <span className="text-neutral-300">{fmt(snap.risk.grossHedge, 0)} / {snap.risk.grossHedgeCap}</span>
                  </div>
                  <SignedBar value={snap.risk.grossHedge} max={snap.risk.grossHedgeCap} colorPos="#38bdf8" colorNeg="#38bdf8" />
                </div>
                <div>
                  <div className="flex justify-between text-[10px] font-mono mb-1">
                    <span className="text-neutral-500">vega usage</span>
                    <span className="text-amber-300">{fmt(snap.risk.vegaUsed, 0)} / {snap.risk.vegaLimit}</span>
                  </div>
                  <SignedBar value={snap.risk.vegaUsed} max={snap.risk.vegaLimit} colorPos="#f59e0b" colorNeg="#f59e0b" />
                </div>
                <div>
                  <div className="flex justify-between text-[10px] font-mono mb-1">
                    <span className="text-neutral-500">drawdown</span>
                    <span className={snap.risk.drawdown > 50 ? "text-rose-400" : "text-neutral-300"}>{fmt(snap.risk.drawdown, 1)} / 80</span>
                  </div>
                  <SignedBar value={snap.risk.drawdown} max={80} colorPos="#fb7185" colorNeg="#fb7185" />
                </div>
                <div
                  className={`rounded border p-2 font-mono text-[10px] ${
                    snap.risk.halted
                      ? "border-rose-600/40 bg-rose-500/10 text-rose-300"
                      : snap.risk.state === "breach"
                        ? "border-orange-600/40 bg-orange-500/10 text-orange-300"
                        : snap.risk.state === "gated"
                          ? "border-amber-600/40 bg-amber-500/10 text-amber-300"
                          : "border-emerald-600/30 bg-emerald-500/5 text-emerald-400"
                  }`}
                >
                  {snap.risk.halted
                    ? `HALTED: ${snap.risk.haltReason} — quotes off, hedge + marks live`
                    : snap.risk.state === "breach"
                      ? `BREACH (${snap.risk.gatedOn}) — unwind-only quoting at the touch; risk-adding side pulled`
                      : snap.risk.state === "gated"
                        ? `GATED (${snap.risk.gatedOn}) — risk-adding side pulled, unwind side tightened · self-recovers`
                        : "ALL SYSTEMS NOMINAL — quoting live both sides"}
                </div>
                <div className="rounded border border-neutral-800 bg-neutral-900/50 p-2 font-mono text-[9px] space-y-0.5">
                  <div className="text-neutral-500 uppercase text-[9px] mb-1">risk events</div>
                  <div className="max-h-24 overflow-y-auto space-y-0.5">
                    {[...snap.risk.events].reverse().map((ev, i) => (
                      <div key={i} className="flex gap-2">
                        <span className="text-neutral-600 shrink-0">{ev.t}s</span>
                        <span className="text-neutral-400">{ev.msg}</span>
                      </div>
                    ))}
                    {!snap.risk.events.length && <div className="text-neutral-600">—</div>}
                  </div>
                </div>
              </div>
            </Panel>

            <Panel title="PnL attribution">
              <div className="space-y-2">
                {([
                  ["spread capture", snap.pnl.spreadCapture, "#34d399"],
                  ["inventory drift", snap.pnl.inventory, "#a3e635"],
                  ["fees", snap.pnl.fees, "#fb7185"],
                  ["hedge cost", snap.pnl.hedgeCost, "#f43f5e"],
                  ["funding", snap.pnl.funding, "#38bdf8"],
                  ["option premium", snap.pnl.optionPremium, "#f59e0b"],
                  ["option mark", snap.pnl.optionMark, "#fbbf24"],
                ] as const).map(([label, v, color]) => {
                  const max = Math.max(...[snap.pnl.spreadCapture, snap.pnl.inventory, snap.pnl.fees, snap.pnl.hedgeCost, snap.pnl.funding, snap.pnl.optionPremium, snap.pnl.optionMark].map(Math.abs), 1);
                  return (
                    <div key={label}>
                      <div className="flex justify-between text-[10px] font-mono mb-1">
                        <span className="text-neutral-500">{label}</span>
                        <span style={{ color }} className={signed(v)}>{fmt(v, 2)}</span>
                      </div>
                      <SignedBar value={v} max={max} colorPos={color} colorNeg={color} />
                    </div>
                  );
                })}
                <div className="pt-2 border-t border-neutral-800">
                  <StatChip label="total equity" value={fmt(snap.pnl.total, 2)} tone={pnlTone} sub={`pnl ${pnl > 0 ? "+" : ""}${fmt(pnl, 1)} (${pnlPct > 0 ? "+" : ""}${pnlPct.toFixed(1)}%) · perp ${fmt(snap.equity - INITIAL_CAPITAL - snap.pnl.optionPremium - snap.pnl.optionMark, 1)} · options ${fmt(snap.pnl.optionPremium + snap.pnl.optionMark, 1)}`} />
                </div>
              </div>
            </Panel>
          </TabsContent>
        </Tabs>
      </main>

      <footer className="mt-3 border-t border-neutral-800 bg-neutral-950 px-3 py-2">
        <div className="max-w-[1440px] mx-auto flex flex-wrap items-center justify-between gap-2 font-mono text-[9px] text-neutral-600">
          <span>perp-mm · 9-crate Rust workspace + TS engine port · SSVI/GLFT/HJB/WW/vanna–volga · 8 strategies · {snap.steps} steps · {snap.perf.ops} fills</span>
          <span>tick p50 {snap.perf.p50 > 0 ? `${(snap.perf.p50 / 1000).toFixed(0)}µs` : "<1µs"} · p99 {(snap.perf.p99 / 1000).toFixed(0)}µs · CU {snap.perf.cu}/{snap.perf.cuBudget} ({snap.perf.cuPct.toFixed(1)}%)</span>
        </div>
      </footer>
    </div>
  );
}

function ParamSlider({
  eng,
  pkey,
  label,
  min,
  max,
  step,
  prec,
  value,
}: {
  eng: ReturnType<typeof useEngine>;
  pkey: string;
  label: string;
  min: number;
  max: number;
  step: number;
  prec: number;
  value: number;
}) {
  return (
    <div>
      <div className="flex justify-between text-[10px] font-mono mb-1">
        <span className="text-neutral-500">{label}</span>
        <span className="text-neutral-200">{isFinite(value) ? value.toFixed(prec) : "—"}</span>
      </div>
      <Slider
        value={[isFinite(value) ? value : min]}
        min={min}
        max={max}
        step={step}
        onValueChange={(nv) => {
          (eng as unknown as { setParam: (k: string, v: number) => void }).setParam(pkey, nv[0]);
        }}
        aria-label={label}
      />
    </div>
  );
}
