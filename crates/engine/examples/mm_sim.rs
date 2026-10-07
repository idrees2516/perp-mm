//! Monte-Carlo strategy comparison: 6 strategies x N paths, CSV metrics +
//! PNG-ready equity/inventory series.

use engine::config::{EngineConfig, MidModel};
use engine::metrics::RunMetrics;
use engine::mm::run_backtest;
use engine::strategy::StrategyKind;
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_paths: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let out_dir = std::path::PathBuf::from(
        args.get(2).cloned().unwrap_or_else(|| "target/sim".into()),
    );
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    // Regime A: baseline GBM mid.
    let mut cfg_a = EngineConfig::default();
    cfg_a.horizon = 3600.0; // 1h session
    cfg_a.dt = 0.2;
    // Regime B: rough volatility + jumpier book + higher adverse selection.
    let mut cfg_b = EngineConfig {
        mid: MidModel::Rough {
            h: 0.1,
            nu: 0.35,
            sigma0: 0.025,
            refresh_every: 64,
        },
        ..EngineConfig::default()
    };
    cfg_b.horizon = 3600.0;
    cfg_b.dt = 0.2;
    cfg_b.adverse_ticks = 1.6;
    cfg_b.mo_rate = 1.6;

    let regimes: [(&str, EngineConfig); 2] = [("baseline", cfg_a), ("rough_stress", cfg_b)];

    let mut csv = String::new();
    csv.push_str(&format!(
        "regime,{},",
        RunMetrics::csv_header()
            .trim_end()
            .split(',')
            .skip(1)
            .collect::<Vec<_>>()
            .join(",")
    ));
    csv.push('\n');
    let mut series_rows: Vec<String> = Vec::new();

    let t0 = std::time::Instant::now();
    for (regime_name, cfg) in &regimes {
        for kind in StrategyKind::all() {
            let mut equities: Vec<f64> = Vec::with_capacity(n_paths);
            let mut inv_abs: Vec<i64> = Vec::with_capacity(n_paths);
            let mut draws: Vec<f64> = Vec::with_capacity(n_paths);
            let mut fills: Vec<u64> = Vec::with_capacity(n_paths);
            let mut spread_caps: Vec<f64> = Vec::with_capacity(n_paths);
            let mut adverse: Vec<f64> = Vec::with_capacity(n_paths);
            let mut sharpe: Vec<f64> = Vec::with_capacity(n_paths);
            for p in 0..n_paths {
                let seed = 1000 + p as u64;
                let m = run_backtest(cfg.clone(), kind, seed);
                equities.push(m.equity);
                inv_abs.push(m.inventory_abs_max);
                draws.push(m.max_drawdown);
                fills.push(m.our_fills);
                spread_caps.push(m.spread_capture);
                adverse.push(m.adverse_cost);
                sharpe.push(m.sharpe_per_hour());
                if p == 0 {
                    // record the sample equity path for plotting
                    for (i, e) in m.equity_path.iter().enumerate() {
                        series_rows.push(format!(
                            "{},{},{},{},{}\n",
                            regime_name,
                            kind.label(),
                            p,
                            i,
                            e
                        ));
                    }
                }
            }
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let std = |v: &[f64]| {
                let m = mean(v);
                (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt()
            };
            let inv_mean = inv_abs.iter().map(|&x| x as f64).sum::<f64>() / n_paths as f64;
            let row = format!(
                "{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
                regime_name,
                kind.label(),
                mean(&equities),
                std(&equities),
                mean(&spread_caps),
                mean(&adverse),
                0.0, // fees folded into equity
                0.0, // funding folded into equity
                0.0, // inv_final varies per path
                inv_mean,
                *inv_abs.iter().max().unwrap_or(&0) as f64,
                fills.iter().sum::<u64>() as f64 / n_paths as f64,
                mean(&draws),
                mean(&sharpe),
                mean(&sharpe),
                mean(&draws)
            );
            csv.push_str(&row);
            println!(
                "  {:>13} {:>8}: equity {:+9.3} +/- {:.3} | inv|max| {:.1} | fills {:.0} | dd {:.2}",
                regime_name,
                kind.label(),
                mean(&equities),
                std(&equities),
                inv_mean,
                fills.iter().sum::<u64>() as f64 / n_paths as f64,
                mean(&draws)
            );
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    println!(
        "\nMonte Carlo: {} strategies x {} regimes x {} paths in {:.1}s",
        StrategyKind::all().len(),
        regimes.len(),
        n_paths,
        elapsed
    );

    // write CSVs
    let mut f = std::fs::File::create(out_dir.join("mc_summary.csv")).unwrap();
    f.write_all(csv.as_bytes()).unwrap();
    let mut f = std::fs::File::create(out_dir.join("mc_equity_paths.csv")).unwrap();
    f.write_all("regime,strategy,path,step,equity\n".as_bytes()).unwrap();
    for r in &series_rows {
        f.write_all(r.as_bytes()).unwrap();
    }
    println!(
        "wrote {}/mc_summary.csv and mc_equity_paths.csv",
        out_dir.display()
    );
}
