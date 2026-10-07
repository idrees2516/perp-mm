//! Study 2 — Monte-Carlo validation of the new market-making pieces:
//!
//! - **A. Multi-level ladders**: `ladder` (multi-level pricing ladders
//!   with fee floor, taper, funding skew) vs `hjb` (single-level exact
//!   solver) vs `static` across baseline and stress regimes.
//! - **B. Option market making**: the SSVI-surface option leg with
//!   {naive fixed vol spread, GLFT vega-approximation quotes} x
//!   {hedge every step, fixed band, Whalley–Wilmott band} — expected:
//!   GLFT + WW dominates in certainty-equivalent terms.
//! - **C. Markout-adaptive spreads**: `ladder` with the adaptive spread
//!   multiplier on vs off under a toxic-flow regime.
//!
//! Output: `study2_summary.csv` (all metrics), `study2_options_paths.csv`
//! and `study2_markout_paths.csv` (equity series for the figures).

use engine::config::{EngineConfig, MidModel};
use engine::metrics::RunMetrics;
use engine::mm::run_backtest;
use engine::options_market::{enable_options, HedgeMode, QuoterMode};
use engine::strategy::StrategyKind;
use std::io::Write;

fn entropic_ce(samples: &[f64], gamma: f64) -> f64 {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    mean - 0.5 * gamma * var
}

fn main() {
    let n_paths: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let out_dir = std::path::PathBuf::from(
        std::env::args().nth(2).unwrap_or_else(|| "target/sim".into()),
    );
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let t0 = std::time::Instant::now();

    let header: String = format!(
        "study,config,{},",
        RunMetrics::csv_header().trim_end().split(',').skip(1).collect::<Vec<_>>().join(",")
    );
    let mut csv = String::new();
    csv.push_str(&header);
    csv.push('\n');
    let mut opt_paths: Vec<String> = Vec::new();
    let mut mk_paths: Vec<String> = Vec::new();

    // ------------------------------------------------------------------
    // Study A: multi-level ladders
    // ------------------------------------------------------------------
    let mut cfg_a = EngineConfig::default();
    cfg_a.horizon = 1800.0;
    cfg_a.dt = 0.1;
    let mut cfg_s = EngineConfig {
        mid: MidModel::Rough { h: 0.08, nu: 0.5, sigma0: 0.03, refresh_every: 64 },
        ..EngineConfig::default()
    };
    cfg_s.horizon = 1800.0;
    cfg_s.dt = 0.1;
    cfg_s.adverse_ticks = 1.4;
    cfg_s.mo_rate = 1.4;

    for (regime, cfg) in [("baseline", cfg_a), ("stress", cfg_s)] {
        for kind in [StrategyKind::MultiLevel, StrategyKind::HjbPolicy, StrategyKind::Static] {
            let mut eqs = Vec::new();
            for p in 0..n_paths {
                let seed = 10_000 + p as u64;
                let m = run_backtest(cfg.clone(), kind, seed);
                let regime_kind = format!("{regime}_{}", kind.label());
                csv.push_str(&format!("A_ladder,{regime_kind},{},\n", m.csv_row().trim_end()));
                eqs.push(m.equity);
            }
            println!(
                "A {regime} {:>7}: mean equity {:+8.3} CE {:+8.3}",
                kind.label(),
                eqs.iter().sum::<f64>() / n_paths as f64,
                entropic_ce(&eqs, 0.01)
            );
        }
    }

    // ------------------------------------------------------------------
    // Study B: option market making (SSVI + vega-approx quotes + hedge cadence)
    // ------------------------------------------------------------------
    let mut cfg_o = EngineConfig::default();
    cfg_o.horizon = 1800.0;
    cfg_o.dt = 0.25;
    enable_options(&mut cfg_o, QuoterMode::Glft, HedgeMode::WwBand);
    cfg_o.option_market.kappa_v = 25.0;
    cfg_o.option_market.a_v = 0.08;
    cfg_o.option_market.gamma = 5e-3;
    cfg_o.option_market.lots_per_request = 0.01;
    let configs: [(&str, QuoterMode, HedgeMode); 4] = [
        ("fixed_volspread_every_step", QuoterMode::Fixed { vol_spread: 0.10 }, HedgeMode::EveryStep),
        ("fixed_volspread_ww_band", QuoterMode::Fixed { vol_spread: 0.10 }, HedgeMode::WwBand),
        ("glft_quotes_every_step", QuoterMode::Glft, HedgeMode::EveryStep),
        ("glft_quotes_ww_band", QuoterMode::Glft, HedgeMode::WwBand),
    ];
    for (name, quoter, hedge) in configs {
        let mut eqs = Vec::new();
        for p in 0..n_paths {
            let mut cfg = cfg_o.clone();
            cfg.option_market.quoter = quoter;
            cfg.option_market.hedge = hedge;
            let seed = 20_000 + p as u64;
            let m = run_backtest(cfg, StrategyKind::VolSurface, seed);
            csv.push_str(&format!(
                "{},{},{},\n",
                "B_options",
                name,
                m.csv_row().trim_end()
            ));
            eqs.push(m.equity);
            for (i, &e) in m.equity_path.iter().enumerate() {
                opt_paths.push(format!("{},{},{},{},{}\n", name, p, i, i * 60, e));
            }
        }
        println!(
            "B {name:>28}: mean equity {:+8.3} CE {:+8.3}",
            eqs.iter().sum::<f64>() / n_paths as f64,
            entropic_ce(&eqs, 0.01)
        );
    }

    // ------------------------------------------------------------------
    // Study C: markout-adaptive spreads under toxic flow
    // ------------------------------------------------------------------
    let mut cfg_t = EngineConfig::default();
    cfg_t.horizon = 1800.0;
    cfg_t.dt = 0.1;
    cfg_t.adverse_ticks = 1.2; // toxic fills (the optimal widening
                                // stays within the multiplier cap)
    cfg_t.mo_rate = 1.8;
    for adaptive in [true, false] {
        let mut cfg = cfg_t.clone();
        cfg.markout_adaptive = adaptive;
        let mut eqs = Vec::new();
        for p in 0..n_paths {
            let seed = 30_000 + p as u64;
            let m = run_backtest(cfg.clone(), StrategyKind::MultiLevel, seed);
            csv.push_str(&format!(
                "{},{},{},\n",
                "C_markout",
                if adaptive { "adaptive_on" } else { "adaptive_off" },
                m.csv_row().trim_end()
            ));
            eqs.push(m.equity);
            for (i, &e) in m.equity_path.iter().enumerate() {
                mk_paths.push(format!("{},{},{},{},{}\n", adaptive, p, i, i * 60, e));
            }
        }
        println!(
            "C adaptive={}: mean equity {:+8.3} CE {:+8.3}",
            adaptive,
            eqs.iter().sum::<f64>() / n_paths as f64,
            entropic_ce(&eqs, 0.01)
        );
    }

    let mut f = std::fs::File::create(out_dir.join("study2_summary.csv")).unwrap();
    f.write_all(csv.as_bytes()).unwrap();
    let mut f = std::fs::File::create(out_dir.join("study2_options_paths.csv")).unwrap();
    f.write_all(b"config,path,idx,sim_time,equity\n").unwrap();
    f.write_all(opt_paths.concat().as_bytes()).unwrap();
    let mut f = std::fs::File::create(out_dir.join("study2_markout_paths.csv")).unwrap();
    f.write_all(b"adaptive,path,idx,sim_time,equity\n").unwrap();
    f.write_all(mk_paths.concat().as_bytes()).unwrap();
    println!("study2 done in {:.1}s -> {:?}", t0.elapsed().as_secs_f64(), out_dir);
}
