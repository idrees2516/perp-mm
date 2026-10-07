# ARCHITECTURE.md

## Data flow

```text
                       ┌──────────────────────────────────────────────┐
                       │                SimVenue                      │
                       │  mid model (GBM / RFSV rough)               │
                       │  MQH-lite book dynamics (LO/CO/MO flows)    │
                       │  our fills: Cox intensity lambda(delta)     │
                       │            + public sweeps (FIFO, queue)    │
                       │  adverse selection, fees, funding, ADL      │
                       └───────┬──────────────────────────┬──────────┘
                     frames   │                          │ venue events
                    (codec)   │                          v
                       ┌──────v──────┐          ┌─────────────────┐
                       │ OrderBook   │          │ RiskEngine      │
                       │ (glass trie │          │ inventory gates │
                       │  + L3 FIFO) │          │ throttle, halts │
                       └──────┬──────┘          │ kill-switch     │
                              │                 └────────┬────────┘
                       ┌──────v───────────────────────────┴─────────┐
                       │            EstimatorStack                 │
                       │  EWMA vol (fast/slow) · LM08 jump flags    │
                       │  Roll+serial spread · micro-price (RLS)   │
                       │  OFI + impact · CLF · RFSV forecast        │
                       └──────┬────────────────────────────────────┘
                              │ MarketState snapshot
                       ┌──────v──────────┐
                       │ QuotingStrategy │  static | as | hjb | queue
                       │                 │  micro  | options (+delta hedge)
                       └──────┬──────────┘
                              │ desired quotes (ticks, lots)
                       ┌──────v──────────┐        ┌──────────────┐
                       │  quote diffing  │───────>│ venue orders │
                       └─────────────────┘        └──────────────┘
                              │
                       ┌──────v──────────┐
                       │ RunMetrics      │  PnL attribution, CSV
                       └─────────────────┘
```

The same frame stream can be driven over a real transport: `SimVenue::
drain_frames()` produces the codec's wire format, and `FeedHandler`
consumes any `Transport` (UDP socket, channel, replay, or the io_uring
reader) into the sequence-validated `DeltaStream`.

## Complexity table

| Operation | Complexity | Notes |
|---|---|---|
| `glass` insert / erase / find | O(13) worst, ~O(1) amortized with cached path | depth bounded by key width, not by n |
| `glass` best bid/ask | O(1) | cached extrema |
| `glass` next/prev (successor) | O(13) | sibling walk with parent pointers |
| `glass` ladder scan (k levels) | O(k · 13) | cached-path jumps between near levels |
| L3 add / cancel | O(1) | intrusive doubly-linked FIFO |
| L3 execute_fifo (sweep) | O(fills) | consumes front of queue |
| codec decode/encode | O(1) per frame | fixed-width layouts |
| estimator tick | O(1) amortized | ring buffers; slow cadences gated |
| rough-vol refresh | O(m³) m≤128 | Gaussian elimination on the conditioning system, every 128 steps |
| HJB solve | O(steps · q_dim) | RK4 over the q-lattice; 300 steps × 41 states ≈ sub-ms |
| HJB quote | O(1) | interpolation of the precomputed policy |
| ADL capped pro-rata | O(n log n) | sort by caps + water-fill |
| queue fill probability | O(1) | closed form `(mu/(mu+nu))^m` |

## The CU (compute-unit) budget

Solana programs run under a hard compute budget per transaction,
metered per instruction. We apply the same discipline to feed events:

```rust
budget.begin_event();
budget.charge(OpClass::CodecDecode);       // 8 CU
budget.charge(OpClass::BookLevelUpdate);   // 3 CU
budget.charge(OpClass::EstimatorTick);     // 14 CU
if budget.charge(OpClass::QuoteCompute) {  // 25 CU
    budget.charge(OpClass::RiskFilter);    // 4 CU
} else {
    // over budget: degrade gracefully — hold current quotes,
    // skip cold-path estimators, never drop the book update
}
budget.end_event();
```

Calibrated to measured ns on this machine (the ratios mirror Solana's
cheap-arithmetic / memory / full-reprice tiers), a typical event costs
**58 CU of a 200 CU budget — 71% headroom** — and the histogram in
`bench_report` shows the degradation path is never hit in normal
operation (0.00% of events above 192 CU).

## Numerical-methods inventory

| Method | Where | Validation |
|---|---|---|
| RK4 backward integration of the v-system ODE | `models::hjb` | terminal FOC = AS closed form exactly; grid convergence 200→1600 steps within 5e-3; MC entropic-CE optimality |
| Closed-form FOC quotes with impact offsets | `models::hjb` | impact widens quotes against inventory (directional test) |
| Davies-Harte circulant embedding fBm (radix-2 FFT) | `micro::rough` | variance scaling `Var(B_{t+Delta}-B_t) = Delta^{2H}` within 25% |
| Gaussian conditional expectation (Riccati-free) | `micro::rough::RfsvModel::forecast` | walk-forward MSE beats climatology |
| Gauss-Jordan matrix inversion | `micro::microprice` | chain micro-price ordering test |
| Kahan-free Poisson tail | `models::queue` | MC within 2% |
| CRR binomial American options | `models::options` | parity, IV roundtrip, FD greeks |
| Water-filling with caps (KKT) | `perps::adl` | budget balance to 1e-6; order-stability |
| Quadratic-in-z exact CS estimator | `micro::spread` | MC recovery within 35% |
| BNS variance re-derivation | `micro::jump` | recovers `theta = pi²/4+pi-5` exactly; size within 10% at alpha=5% |

## Error budget and honest caveats

- The v-system solver's quote levels agree with the AS closed form to
  <15% in the low-intensity regime; at high fill intensity the exact
  solution legitimately skews less (the MC CE test shows it dominates).
- The GLFT survey asymptotics are structural anchors (same sign, same
  order); the exact solver is the ground truth.
- The queue-race "level death" is degenerate under refill protection
  (P=1, MC-confirmed); we implement the non-degenerate races instead.
- Thm 6's `gamma_2(t)` (eq. 32) was not transcribed by our extraction;
  the identity-payoff exact form and the substitution rule cover the
  regimes where it applies.
- The options strategy is dormant in the perp-only simulation (no
  option series in the venue); its quote/hedge paths are unit-tested.
- The MQH simulator is a reduced form (top-of-book meta-queues), not
  the full 12-event Hawkes model.

## The gateway & frontend (stage 2)

```text
                         Unix-domain socket (4-byte length framing)
   ┌────────────────────┐   snapshots ~20 Hz (567 B)     ┌──────────────────┐
   │   mm-daemon        │ ─────────────────────────────▶ │   mm-tui         │
   │ ┌────────────────┐ │   trade prints / acks          │ ┌──────────────┐ │
   │ │ MarketMaker    │ │                                │ │ App          │ │
   │ │ venue+est+strat│ │ ◀───────────────────────────── │ │ 7 pages     │ │
   │ │ option market  │ │   commands (strategy switch,   │ │ order entry │ │
   │ │ markout+risk   │ │    params, pause/kill, orders) │ └──────┬───────┘ │
   │ └───────┬────────┘ │                                │        │         │
   │         │          │                                │ ┌──────▼───────┐ │
   │ ┌───────▼────────┐ │                                │ │ Screen       │ │
   │ │ gateway.rs     │ │                                │ │ cell grid + │ │
   │ │ snapshot build │ │                                │ │ diff render │ │
   │ │ cmd apply      │ │                                │ │ (272 B/frame)│ │
   │ │ latency ring   │ │                                │ └──────┬───────┘ │
   │ └────────────────┘ │                                │        │ ANSI    │
   └────────────────────┘                                │   poll(stdin,sock,33ms)
                                                         └────────┬─────────┘
                                                                  │ write(1)
                                                              terminal
```

- **Protocol** (`feed::proto`): tag-prefixed little-endian frames —
  `Snapshot` (book ladder, our orders, PnL attribution, markout
  multiplier, option-leg greeks + per-strike IV quotes, engine-step
  latency percentiles, drops), `Trade`, `Ack`, and ten `Command`
  variants. Encode 262 ns / decode 280 ns; garbage input rejected with
  trailing-byte checks.
- **Transport** (`feed::uds`): `UdsGateway` fans out non-blocking writes
  and drops stalled clients (never blocks the engine); `UdsClient`
  exposes `send`/`recv`/`fd()` for the TUI's poll loop.
- **Daemon** (`engine/src/bin/mm-daemon.rs`): single thread — engine
  batches paced to `--speed`, command intake + acks, ~20 Hz snapshot
  publish, per-step latency percentiles via a 512-slot ring.
- **TUI** (`tui` crate): raw-mode via direct `ioctl(TCGETS/TCSETS)`
  syscalls (no libc crate), double-buffered cell grid with a run-coalescing
  diff renderer (cursor moves only between dirty runs, SGR only on style
  change — 272 bytes/frame vs ~11 KB full repaint, 41×), braille
  sparklines (2×4 sub-pixels/cell), resize handling via per-frame
  `TIOCGWINSZ` (no signal handling needed), one `write(1)` syscall per
  frame. Full client tick 18.6 µs p50 = 0.06% of the 30 Hz budget.

## Option-market data flow (stage 2)

```text
 SSVI surface (atm_iv ← vol-of-vol GBM; θ = atm²·t)
   │  fair IV per strike (moneyness ladder)
   ▼
 OptMm (GLFT in vega space + per-fill hedge-cost floor + inventory skew)
   │  iv_bid / iv_ask per strike
   ▼
 OptionMarket.step — client requests at a_v·e^{−κ_v·δ}  →  fills at our IVs
   │  positions per leg                     ┌──────────────────────┐
   ▼                                        │ perp CLOB (SimVenue) │
 book greeks (delta in PERP lots, vega, γ)  │ passive volmm quotes │
   │                                        │ around combined Δ    │
   ▼                                        │ SMP-protected takers │
 hedge executor: |unhedged| > band?         │ (rebalance to band   │
   → venue.take(excess)                     │  edge, not to zero)  │
                                            └──────────────────────┘
```
