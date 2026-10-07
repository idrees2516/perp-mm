//! `mm-daemon` — the market-making engine daemon.
//!
//! Runs the simulated venue + estimator stack + quoting strategy +
//! option market leg, and serves frontends over a Unix-domain socket
//! (the gateway protocol): ~20 Hz state snapshots, command intake
//! (strategy switching, live parameters, pause/kill-switch, manual
//! orders), plus per-step latency percentiles.
//!
//! ```text
//! mm-daemon [--path /tmp/perp-mm.sock] [--strategy ladder]
//!           [--speed 60] [--horizon 23400] [--seed 7]
//!           [--headless-steps 2000] [--option-leg]
//! ```

use std::time::{Duration, Instant};

use engine::config::EngineConfig;
use engine::gateway::{apply_command, build_snapshot, LatencyRing};
use engine::mm::MarketMaker;
use engine::options_market::{enable_options, HedgeMode, QuoterMode};
use engine::strategy::StrategyKind;
use feed::proto::{decode_msg, encode_msg, GwMsg};
use feed::uds::UdsGateway;
use micro::Rng;

fn kind_by_label(label: &str) -> StrategyKind {
    StrategyKind::all()
        .into_iter()
        .find(|k| k.label() == label)
        .unwrap_or(StrategyKind::HjbPolicy)
}

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn has_arg(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

fn main() {
    let path = arg("--path").unwrap_or_else(|| "/tmp/perp-mm.sock".into());
    let strategy = arg("--strategy").unwrap_or_else(|| "ladder".into());
    let speed: f64 = arg("--speed").and_then(|s| s.parse().ok()).unwrap_or(60.0);
    let seed: u64 = arg("--seed").and_then(|s| s.parse().ok()).unwrap_or(7);
    let headless_steps: Option<u64> = arg("--headless-steps").and_then(|s| s.parse().ok());
    let quiet = has_arg("--quiet");

    let mut cfg = EngineConfig::default();
    if let Some(h) = arg("--horizon").and_then(|s| s.parse().ok()) {
        cfg.horizon = h;
    }
    if has_arg("--option-leg") {
        enable_options(&mut cfg, QuoterMode::Glft, HedgeMode::WwBand);
    }
    if has_arg("--stress") {
        // rough-vol stress regime (same as the MC study)
        cfg.mid = engine::config::MidModel::Rough {
            h: 0.05,
            nu: 0.8,
            sigma0: 0.03,
            refresh_every: 64,
        };
        cfg.adverse_ticks = 1.6;
    }

    let kind = kind_by_label(&strategy);
    let mut mm = MarketMaker::new(cfg, kind, seed);
    let mut rng = Rng::new(seed ^ 0x9E3779B9);
    let mut gw = match UdsGateway::bind(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("mm-daemon: cannot bind {path}: {e}");
            std::process::exit(1);
        }
    };
    let mut lat = LatencyRing::new(512);
    let mut seq: u64 = 0;
    let mut last_pub = Instant::now() - Duration::from_secs(1);
    let pub_period = Duration::from_millis(50); // ~20 Hz
    let mut scratch = Vec::new();
    if !quiet {
        println!("mm-daemon: strategy={} path={} speed={}x", kind.label(), path, speed);
    }

    // steps per real second (speed = simulated seconds per real second)
    let steps_per_sec = (speed / mm.cfg.dt).max(1.0);
    let mut steps_done: u64 = 0;
    let frame_start = Instant::now();

    loop {
        // ---- engine batch ----
        let target_steps = (steps_per_sec * frame_start.elapsed().as_secs_f64()) as u64;
        while steps_done < target_steps || headless_steps.is_some() {
            let t0 = Instant::now();
            mm.step(&mut rng);
            lat.push(t0.elapsed().as_nanos() as u64);
            steps_done += 1;
            if let Some(n) = headless_steps {
                if steps_done >= n {
                    // final publish + exit
                    gw.accept();
                    let state = mm.state();
                    seq += 1;
                    let snap = build_snapshot(&mut mm, &state, seq, &lat, gw.drops);
                    gw.broadcast(&encode_msg(&GwMsg::Snapshot(snap)));
                    if !quiet {
                        let m = mm.metrics();
                        println!(
                            "mm-daemon: headless done: equity={:.4} fills={} taker={} inv={}",
                            m.equity, m.our_fills, m.taker_fills, m.inventory_final
                        );
                    }
                    let _ = std::fs::remove_file(&path);
                    return;
                }
            }
            if headless_steps.is_none() && steps_done > target_steps + steps_per_sec as u64 * 4 {
                break; // runaway guard
            }
        }
        if headless_steps.is_none() {
            // idle-wait pacing: sleep a little if we're ahead of realtime
            let expected = Duration::from_secs_f64(steps_done as f64 / steps_per_sec);
            if frame_start.elapsed() < expected {
                std::thread::sleep((expected - frame_start.elapsed()).min(Duration::from_millis(5)));
            }
        }

        // ---- gateway ----
        gw.accept();
        let frames = gw.poll_commands(&mut scratch);
        for f in frames {
            match decode_msg(&f) {
                Ok(GwMsg::Command(cmd)) => {
                    let ack = apply_command(&mut mm, &cmd);
                    gw.broadcast(&encode_msg(&GwMsg::Ack(ack)));
                }
                Ok(_) => {}
                Err(_) => {
                    gw.broadcast(&encode_msg(&GwMsg::Ack(
                        feed::proto::GwAck::UnknownCommand,
                    )));
                }
            }
        }

        // ---- publish ----
        if last_pub.elapsed() >= pub_period {
            last_pub = Instant::now();
            let state = mm.state();
            seq += 1;
            let snap = build_snapshot(&mut mm, &state, seq, &lat, gw.drops);
            gw.broadcast(&encode_msg(&GwMsg::Snapshot(snap)));
        }

        if headless_steps.is_none() && steps_done as f64 * mm.cfg.dt >= mm.cfg.horizon {
            if !quiet {
                let m = mm.metrics();
                println!(
                    "mm-daemon: horizon reached: equity={:.4} fills={}",
                    m.equity, m.our_fills
                );
            }
            std::thread::sleep(Duration::from_millis(300)); // flush to clients
            let _ = std::fs::remove_file(&path);
            return;
        }
    }
}
