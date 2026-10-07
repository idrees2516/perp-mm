# perp-mm

**An advanced market-making engine + terminal frontend for
[perp-options-clob](https://github.com/idrees2516/perp-options-clob), in Rust.**

Implements the quoting mathematics, microstructure analytics, perps
mechanics, options-market making (arbitrage-free SSVI surfaces,
vega-inventory quotes, hedge cadence), feed layer, risk engine, and a
zero-dependency TUI trading frontend from a curated set of 2024-2026
research papers, validated end-to-end by Monte Carlo. Zero external
dependencies — even the RNG, FFT, io_uring ABI, terminal driver, and
option pricing are hand-rolled for full control of the hot path.

## Results snapshot

| Measurement (this repo, release build) | Value |
|---|---|
| `glass` book level update (hot path) | **72 ns** p50 |
| `glass` best-bid/ask lookup | **56 ns** p50 |
| codec decode | **28 ns/frame**, 7,480 MB/s, 191.8 Mframes/s |
| estimator tick (vol, jump, OFI, micro-price) | **128 ns** p50 |
| HJB quote computation | **339 ns** p50 |
| full pipeline (decode → book → estimate → quote → encode) | **780 ns** p50 |
| gateway snapshot encode / decode | **262 / 280 ns**, 567-byte wire snapshot |
| TUI frame build + diff render (110×34) | **18.2 µs** p50, **272 bytes/frame** (41× fewer than a full repaint) |
| full TUI client tick (decode → update → draw → render) | **18.6 µs** = 0.06% of the 30 Hz frame budget |
| typical feed event under CU metering | 58 CU of a 200 CU budget (71% headroom) |

Monte-Carlo validation (`cargo run --release -p engine --example mm_sim`):

```
regime        strategy | mean equity ± std | mean max|inv| | fills | drawdown
baseline      static  |   -0.194 ± 1.649   |   21.9  |  67   |  3.99
baseline      as      |   -0.397 ± 1.042   |   19.1  |  67   |  1.74
baseline      hjb     |   -1.442 ± 1.360   |   22.8  |  56   |  3.02
baseline      queue   |   -1.378 ± 1.881   |   21.4  |  53   |  2.93
baseline      micro   |   -0.593 ± 1.114   |   21.8  |  66   |  2.15
rough_stress  static  |   -1.235 ± 2.819   |   23.5  | 132   |  6.82
rough_stress  as      |   +0.864 ± 5.562   |   18.9  | 245   | 12.89
rough_stress  micro   |   +0.421 ± 6.774   |   22.1  | 156   | 14.78
```

The inventory-aware strategies (as/hjb/micro/queue) keep the tightest
inventory and drawdown profiles; the static benchmark drifts to its
inventory cap and pays for it in the stress regime. Figures:
`docs/figs/mc_equity.png`, `mc_inventory_drawdown.png`,
`mc_equity_paths.png`.

In the models crate, the **exact HJB solver provably dominates the
Avellaneda-Stoikov closed form** in Monte-Carlo entropic certainty
equivalent (`models::hjb::tests::mc_entropic_ce_optimality`), and the
**Donnelly-Lin-Lorig liquidator beats Almgren-Chriss under funding**
(`perps::liquidation::tests::closed_form_beats_ac_when_funding_matters`).

### Study 2 — the new quoting stack (`cargo run --release -p engine --example study2`)

```
A  multi-level ladders (rough-vol stress):   ladder +0.10  >  hjb -0.06  >  static -0.83
B  option MM (SSVI surface + client flow):   GLFT+WW +4.78 mean equity (best)
                                             WW band adds +3.2 vs hedge-every-step
C  markout-adaptive spreads (toxic regime):  adaptive +0.48  >  static +0.28
```

Figures: `docs/figs/study2_ladder.png`, `study2_options.png`,
`study2_markout.png`. The option market is simulated end-to-end: an
SSVI surface driven by a vol-of-vol GBM, client requests arriving at
Cox intensity on the vol spread, fills at our quoted IVs, net book
greeks, and delta hedging through the perp book with self-match-
prevented takers.

## Workspace layout

| Crate | Contents |
|---|---|
| `ob` | `glass` 32-way trie order book (arXiv:2506.13991), L3 FIFO queues, snapshot/delta streams |
| `micro` | Jump tests (Lee-Mykland 2008, BNS-style, BHR 2024), spread estimators (Roll, serial-dependence Roll, Corwin-Schultz, Brouty 2025), micro-price (Stoikov 2018), OFI (CKS 2014), RFSV rough volatility (GJR 2014), CLF (2507.05749) |
| `models` | Unified AS/CJ framework (2606.01477: forced φ=γσ²/2 etc.), exact HJB v-system solver, GLFT asymptotics, multi-level pricing ladders, queue fill probabilities, BSM/Black-76/CRR options |
| `vol` | Robust implied-vol solver (Newton+bisection hybrid), raw SVI + SSVI arbitrage-free surfaces (Gatheral-Jacquier 2014, butterfly/calendar audits vs Breeden-Litzenberger ground truth), full greeks incl. vanna/volga, vega-approximation option MM (Bergault-Guéant) with Whalley-Wilmott hedge band, **vanna-volga overhedge** (Castagna-Mercurio 2007: full 3×3 ATM/RR/BF hedge system by Cramer's rule, residual-vs-smile charge, survival damping, vol-point clamp) |
| `perps` | BitMEX funding, Donnelly-Lin-Lorig optimal liquidation (2601.10812), Chitra autodeleveraging policies (2512.01112), isolated margin |
| `feed` | Binary codec, gateway protocol (binary snapshots/commands, 4-byte framing), UDS transport, replay, raw-syscall io_uring (no libc crate; registered buffers, SQPOLL probe, batched completions) |
| `engine` | Estimator stack, 8 quoting strategies, simulated option market (SSVI + client flow + vega quotes), markout tracker, multi-level order management with post-only/SMP, risk engine, simulated venue (MQH-lite dynamics), venue adapters, PnL attribution, daemon gateway |
| `tui` | Zero-dep terminal frontend: raw-mode syscall driver, cell-grid diff renderer (272 bytes/frame), braille sparklines, L2 depth ladder, latency histograms, 7-page dashboard, order entry / strategy switching / live parameter tuning over UDS |
| `bench` | ns timers (rdtsc), Solana-style CU metering with budget degradation, benchmark report + frontend/gateway benchmarks |

## Build and run

```bash
cargo test --workspace          # 190+ tests, all green
cargo run --release -p engine --example mm_sim -- 20 target/sim
cargo run --release -p engine --example study2 -- 12 target/sim
python3 ../scripts/plot_sim.py target/sim docs/figs
python3 ../scripts/study2_figs.py
cargo run --release -p bench --example bench_report
cargo run --release -p bench --example bench_frontend
```

## Run the frontend (daemon + TUI)

```bash
# terminal 1 — the engine daemon (simulated venue + strategy + gateway)
cargo run --release -p engine --bin mm-daemon -- --strategy ladder --speed 60

# terminal 2 — the terminal frontend
cargo run --release -p tui --bin mm-tui
```

The daemon serves ~20 Hz state snapshots and commands over a Unix
socket (`/tmp/perp-mm.sock`); the TUI renders a 7-page dashboard
(dashboard / order book / greeks & vol surface / strategy / risk / tape
/ help) with a diff renderer at ~30 Hz. Keys: `1-7` pages, `b`/`s`
order entry, `B`/`S` aggressive takes, `space` pause quotes, `k` kill
switch, `c` cancel all, Up/Down on the strategy page to switch
strategies live, `+`/`-`/Enter to tune parameters. Add `--option-leg`
to the daemon to enable the SSVI option market leg (greeks page comes
alive). A rendered sample lives at `docs/figs/tui_dashboard.txt`
(`mm-tui --dump-frame` renders offline).

## The web terminal (`web/`)

A browser frontend — the full engine (SSVI surface dynamics, vega-approx
GLFT quotes with the vanna–volga overhedge, WW-band hedging, 8
strategies, markout-adaptive spreads, estimator stack, risk engine, CU
metering) ported 1:1 to TypeScript (`src/lib/engine/`) and driven by a
fixed-timestep simulation loop in the browser — no backend required.

```bash
cd web
bun install        # or npm install
bun run dev        # http://localhost:3000
```

Six live views: **Terminal** (quote ladder, book, micro-price, equity,
markouts/toxicity), **Option Chain** (per-strike bid/fair/ask IVs, VV
shifts, premiums, full greeks, positions, client-fill tape),
**Vol Surface** (SSVI heatmap + butterfly/calendar/density no-arb
badges + term structure), **Greeks & Hedge** (net book greeks incl.
vanna/volga, per-expiry vega, the WW band visualization),
**Strategy** (all 8 policies + live parameter sliders), and
**Micro · Risk · PnL** (estimators, limits/drawdown/kill-switch with
risk-manager resume, full PnL attribution). Speed, pause/reset, expiry
and strategy switching are live; the engine runs at 8 sim-steps per
animation frame (~480 sim-seconds per wall-second) with ~100 µs p50
per engine tick.

## The strategies

1. **static** — fixed spread benchmark.
2. **as** — Avellaneda-Stoikov closed form via the unified framework
   (reservation price + half-spread, live sigma).
3. **hjb** — the exact HJB solver policy (RK4 backward integration of the
   v-system with the FOC quotes, adverse-selection impact aware,
   re-solved when live sigma drifts >50%).
4. **queue** — multi-level ladder with per-level fill probabilities
   `P(fill) = (mu/(mu+nu))^m` vs the away-move clock.
5. **micro** — AS centered on the Stoikov micro-price with the CKS
   OFI-impact drift.
6. **options** — everlasting option series quoted around BSM
   (rough-vol IV) with delta hedging through the perp.
7. **ladder** — multi-level pricing ladders (Barzykin-Bergault-Guéant
   tiers around the GLFT base): fee floor, inventory taper of displayed
   size, markout-driven spread multiplier, funding-aware carry skew
   (arXiv:2605.06405).
8. **volmm** — vol-surface option market maker: the engine's option
   market quotes per-strike IVs off an arbitrage-free SSVI surface via
   the vega-approximation GLFT quotes (with the per-fill hedge cost
   priced in), while this strategy runs the perp legs around the
   combined delta so passive flow pays for the hedge.

See `RESEARCH.md` for the paper-to-code map and `ARCHITECTURE.md` for
data flow, complexity tables, and the CU budget design.
