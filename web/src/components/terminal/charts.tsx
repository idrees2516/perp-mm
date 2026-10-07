"use client";

// Hand-rolled SVG chart primitives for the terminal: sparklines, line
// charts with baseline, vol-surface heatmaps, horizontal greek bars and
// depth ladder columns. Zero external chart dependencies.

import React from "react";

export function Sparkline({
  data,
  w = 220,
  h = 48,
  color = "#34d399",
  fill = true,
}: {
  data: number[];
  w?: number;
  h?: number;
  color?: string;
  fill?: boolean;
}) {
  if (!data || data.length < 2) {
    return <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} />;
  }
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - 3 - ((v - min) / range) * (h - 6);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="overflow-visible">
      {fill && (
        <polygon points={`0,${h} ${pts.join(" ")} ${w},${h}`} fill={color} opacity={0.12} />
      )}
      <polyline points={pts.join(" ")} fill="none" stroke={color} strokeWidth={1.4} strokeLinejoin="round" />
    </svg>
  );
}

export function LineChart({
  data,
  w = 560,
  h = 160,
  color = "#34d399",
  baseline,
  label,
}: {
  data: number[];
  w?: number;
  h?: number;
  color?: string;
  baseline?: number;
  label?: string;
}) {
  if (!data || data.length < 2) {
    return (
      <div className="flex items-center justify-center text-[10px] text-neutral-500" style={{ width: w, height: h }}>
        collecting…
      </div>
    );
  }
  const min = Math.min(...data, baseline ?? Infinity);
  const max = Math.max(...data, baseline ?? -Infinity);
  const range = max - min || 1;
  const yOf = (v: number) => h - 8 - ((v - min) / range) * (h - 16);
  const pts = data.map((v, i) => `${((i / (data.length - 1)) * w).toFixed(1)},${yOf(v).toFixed(1)}`);
  const baseY = baseline !== undefined ? yOf(baseline) : null;
  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="block">
      {[0.25, 0.5, 0.75].map((f) => (
        <line key={f} x1={0} x2={w} y1={h * f} y2={h * f} stroke="#27272a" strokeWidth={0.5} />
      ))}
      {baseY !== null && (
        <line x1={0} x2={w} y1={baseY} y2={baseY} stroke="#52525b" strokeDasharray="3 3" strokeWidth={0.8} />
      )}
      <polygon points={`0,${h} ${pts.join(" ")} ${w},${h}`} fill={color} opacity={0.1} />
      <polyline points={pts.join(" ")} fill="none" stroke={color} strokeWidth={1.6} strokeLinejoin="round" />
      {label && (
        <text x={4} y={12} fill="#71717a" fontSize={9} fontFamily="monospace">
          {label}
        </text>
      )}
    </svg>
  );
}

/** Vol-surface heatmap: rows = expiries, cols = moneyness. */
export function Heatmap({
  grid,
  rowLabels,
  colLabels,
  format = (v: number) => v.toFixed(1),
  unit = "",
}: {
  grid: number[][];
  rowLabels: string[];
  colLabels: string[];
  format?: (v: number) => string;
  unit?: string;
}) {
  if (!grid.length) return null;
  const flat = grid.flat();
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const span = max - min || 1;
  // color ramp: low = deep emerald, high = amber/red
  const colorOf = (v: number) => {
    const t = (v - min) / span;
    // h: 160 (emerald) → 45 (amber) → 0 (red)
    const h = 160 - 160 * Math.min(1, t * 1.25);
    const s = 62 + 18 * t;
    const l = 16 + 22 * t;
    return `hsl(${h.toFixed(0)} ${s.toFixed(0)}% ${l.toFixed(0)}%)`;
  };
  return (
    <div className="w-full overflow-x-auto">
      <table className="w-full border-separate border-spacing-[2px] text-[10px] font-mono">
        <thead>
          <tr>
            <th className="text-left pr-1 text-neutral-500 font-normal">T \ k</th>
            {colLabels.map((c) => (
              <th key={c} className="text-center font-normal text-neutral-400">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {grid.map((row, ri) => (
            <tr key={ri}>
              <td className="pr-1 text-neutral-400 whitespace-nowrap">{rowLabels[ri]}</td>
              {row.map((v, ci) => (
                <td
                  key={ci}
                  className="text-center py-1 rounded-[3px] text-neutral-100 tabular-nums"
                  style={{ background: colorOf(v) }}
                  title={`${rowLabels[ri]} @ k=${colLabels[ci]}: ${format(v)}${unit}`}
                >
                  {format(v)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Signed horizontal bar centered at zero (greeks). */
export function SignedBar({
  value,
  max,
  colorPos = "#34d399",
  colorNeg = "#fb7185",
  height = 8,
}: {
  value: number;
  max: number;
  colorPos?: string;
  colorNeg?: string;
  height?: number;
}) {
  const m = Math.max(max, 1e-9);
  const frac = Math.min(Math.abs(value) / m, 1);
  const w = frac * 50;
  return (
    <div className="relative w-full bg-neutral-800/60 rounded-sm overflow-hidden" style={{ height }}>
      <div className="absolute left-1/2 top-0 bottom-0 w-px bg-neutral-600" />
      {value >= 0 ? (
        <div className="absolute left-1/2 top-0 bottom-0 rounded-sm" style={{ width: `${w}%`, background: colorPos }} />
      ) : (
        <div className="absolute right-1/2 top-0 bottom-0 rounded-sm" style={{ width: `${w}%`, background: colorNeg }} />
      )}
    </div>
  );
}

/** Depth ladder column pair around the mid. */
export function DepthLadder({
  bid,
  ask,
  mid,
  tickLabel = "",
}: {
  bid: Array<{ p: number; sz: number }>;
  ask: Array<{ p: number; sz: number }>;
  mid: number;
  tickLabel?: string;
}) {
  const all = [...bid.map((l) => l.p), ...ask.map((l) => l.p)];
  if (!all.length) return <div className="text-[10px] text-neutral-500">no quotes</div>;
  const maxSz = Math.max(...bid.map((l) => l.sz), ...ask.map((l) => l.sz), 1);
  const sorted = [...new Set(all)].sort((a, b) => b - a);
  return (
    <div className="w-full font-mono text-[10px]">
      {sorted.map((p) => {
        const isBid = bid.some((l) => l.p === p);
        const lvl = isBid ? bid.find((l) => l.p === p)! : ask.find((l) => l.p === p)!;
        const szFrac = (lvl.sz / maxSz) * 100;
        const dist = Math.abs(p - mid);
        return (
          <div key={p} className="flex items-center gap-1 h-[15px] leading-[15px]">
            <div className="w-14 text-right tabular-nums text-neutral-400">{dist.toFixed(1)}</div>
            <div className="relative flex-1 h-[13px] bg-neutral-800/40 rounded-sm overflow-hidden">
              <div
                className={isBid ? "absolute right-0 top-0 bottom-0 bg-emerald-500/30" : "absolute left-0 top-0 bottom-0 bg-rose-500/30 rounded-sm"}
                style={{ width: `${szFrac}%` }}
              />
              <span className={isBid ? "absolute right-1 text-emerald-300" : "absolute left-1 text-rose-300"}>
                {isBid ? `${lvl.sz}` : `${lvl.sz}`}
              </span>
            </div>
            <div className="w-14 tabular-nums text-neutral-200">{p.toFixed(2)}</div>
            <div className="w-8 text-neutral-500">{isBid ? "BID" : "ASK"}</div>
          </div>
        );
      })}
      {tickLabel && <div className="text-[9px] text-neutral-600 mt-1">{tickLabel}</div>}
    </div>
  );
}

export function StatChip({
  label,
  value,
  tone = "neutral",
  sub,
}: {
  label: string;
  value: string;
  tone?: "neutral" | "pos" | "neg" | "warn" | "accent";
  sub?: string;
}) {
  const tones: Record<string, string> = {
    neutral: "text-neutral-100",
    pos: "text-emerald-400",
    neg: "text-rose-400",
    warn: "text-amber-400",
    accent: "text-teal-300",
  };
  return (
    <div className="flex flex-col gap-0.5 min-w-[86px]">
      <span className="text-[9px] uppercase tracking-wider text-neutral-500">{label}</span>
      <span className={`font-mono text-sm tabular-nums leading-none ${tones[tone]}`}>{value}</span>
      {sub && <span className="text-[9px] text-neutral-600 font-mono">{sub}</span>}
    </div>
  );
}

export function Panel({
  title,
  children,
  right,
  className = "",
}: {
  title: string;
  children: React.ReactNode;
  right?: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-md border border-neutral-800 bg-neutral-900/60 ${className}`}>
      <header className="flex items-center justify-between px-3 py-1.5 border-b border-neutral-800">
        <h2 className="text-[10px] uppercase tracking-[0.14em] text-neutral-400 font-medium">{title}</h2>
        {right}
      </header>
      <div className="p-3">{children}</div>
    </section>
  );
}
