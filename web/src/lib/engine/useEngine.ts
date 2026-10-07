"use client";

// Browser simulation loop for the market-making engine: fixed-timestep
// sim steps batched per animation frame, snapshots throttled to ~9 Hz
// for React re-renders. The engine is an imperative external store held
// in a ref (created in the effect, never read during render) — snapshots
// flow into React state.

import { useEffect, useState, useCallback, useRef } from "react";
import { MarketMaker, EngineSnapshot, EngineParams, StrategyId, STRATEGIES } from "./engine";

export interface EngineControls {
  snap: EngineSnapshot | null;
  running: boolean;
  speed: number;
  setSpeed: (s: number) => void;
  play: () => void;
  pause: () => void;
  reset: (seed?: number) => void;
  resume: () => void;
  setStrategy: (id: StrategyId) => void;
  setParam: <K extends keyof EngineParams>(k: K, v: EngineParams[K]) => void;
  setSelectedExp: (i: number) => void;
  requestRfq: (templateId: string, lots: number) => number | null;
  executeRfq: (id: number, side: "desk-buys" | "desk-sells") => { ok: boolean; reason?: string };
  cancelRfq: (id: number) => void;
  strategyName: string;
  strategies: typeof STRATEGIES;
}

export function useEngine(seed?: number): EngineControls {
  const mmRef = useRef<MarketMaker | null>(null);
  const [snap, setSnap] = useState<EngineSnapshot | null>(null);
  const [running, setRunning] = useState(true);
  const [speed, setSpeedState] = useState(8); // sim steps per frame
  const speedRef = useRef(8);
  const runningRef = useRef(true);

  const setSpeed = useCallback((s: number) => {
    speedRef.current = s;
    setSpeedState(s);
  }, []);
  const play = useCallback(() => {
    runningRef.current = true;
    setRunning(true);
  }, []);
  const pause = useCallback(() => {
    runningRef.current = false;
    setRunning(false);
  }, []);
  const reset = useCallback((newSeed?: number) => {
    const mm = mmRef.current;
    if (!mm) return;
    Object.assign(mm, new MarketMaker(newSeed ?? Math.floor(Math.random() * 1e9)));
    setSnap(mm.snapshot());
  }, []);
  const resume = useCallback(() => {
    const mm = mmRef.current;
    if (!mm) return;
    mm.resume();
    setSnap(mm.snapshot());
  }, []);
  const setStrategy = useCallback((id: StrategyId) => {
    const mm = mmRef.current;
    if (mm) mm.strategy = id;
  }, []);
  const setParam = useCallback(<K extends keyof EngineParams>(k: K, v: EngineParams[K]) => {
    const mm = mmRef.current;
    if (mm) (mm.params as Record<string, unknown>)[k as string] = v;
  }, []);
  const setSelectedExp = useCallback((i: number) => {
    const mm = mmRef.current;
    if (mm) mm.selectedExp = i;
  }, []);
  const requestRfq = useCallback((templateId: string, lots: number) => {
    const mm = mmRef.current;
    return mm ? mm.requestRfq(templateId, lots) : null;
  }, []);
  const executeRfq = useCallback((id: number, side: "desk-buys" | "desk-sells") => {
    const mm = mmRef.current;
    return mm ? mm.executeRfq(id, side) : { ok: false, reason: "engine not ready" };
  }, []);
  const cancelRfq = useCallback((id: number) => {
    const mm = mmRef.current;
    if (mm) mm.cancelRfq(id);
  }, []);

  useEffect(() => {
    if (!mmRef.current) mmRef.current = new MarketMaker(seed ?? Math.floor(Math.random() * 1e9));
    const mm = mmRef.current;
    let raf = 0;
    let last = performance.now();
    let pubAcc = 0;
    const tick = () => {
      const now = performance.now();
      const elapsed = now - last;
      last = now;
      if (runningRef.current) {
        const n = speedRef.current;
        for (let i = 0; i < n; i++) {
          mm.step(1.0);
          if (mm.halted) break;
        }
        pubAcc += elapsed;
        if (pubAcc > 110 || mm.halted) {
          pubAcc = 0;
          setSnap(mm.snapshot());
        }
      } else if (!snap) {
        setSnap(mm.snapshot());
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const strategyName = STRATEGIES.find((s) => s.id === (snap?.strategy ?? "volsurf"))?.name ?? "Vol-Surface MM";

  return { snap, running, speed, setSpeed, play, pause, reset, resume, setStrategy, setParam, setSelectedExp, requestRfq, executeRfq, cancelRfq, strategyName, strategies: STRATEGIES };
}
