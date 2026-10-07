//! End-to-end integration: the engine daemon core + gateway transport +
//! a real Unix-domain client, all in-process. Verifies the full loop:
//! snapshots flow, commands apply, acks return, and a TUI-side App
//! consumes the stream.

use std::time::Duration;

use engine::config::EngineConfig;
use engine::gateway::{apply_command, build_snapshot, LatencyRing};
use engine::mm::MarketMaker;
use engine::options_market::{enable_options, HedgeMode, QuoterMode};
use engine::strategy::StrategyKind;
use feed::proto::{decode_msg, encode_msg, GwAck, GwCommand, GwMsg};
use feed::uds::{UdsClient, UdsGateway};
use micro::Rng;

fn tmp_path(tag: u32) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("perp-mm-e2e-{}-{}.sock", tag, std::process::id()));
    p
}

/// Drive the daemon core: steps the engine, pumps the gateway, publishes
/// one snapshot.
fn pump(
    mm: &mut MarketMaker,
    gw: &mut UdsGateway,
    lat: &mut LatencyRing,
    rng: &mut Rng,
    steps: usize,
    seq: &mut u64,
) {
    let mut scratch = Vec::new();
    for _ in 0..steps {
        let t0 = std::time::Instant::now();
        mm.step(rng);
        lat.push(t0.elapsed().as_nanos() as u64);
    }
    gw.accept();
    for f in gw.poll_commands(&mut scratch) {
        if let Ok(GwMsg::Command(cmd)) = decode_msg(&f) {
            let ack = apply_command(mm, &cmd);
            gw.broadcast(&encode_msg(&GwMsg::Ack(ack)));
        }
    }
    *seq += 1;
    let state = mm.state();
    let snap = build_snapshot(mm, &state, *seq, lat, gw.drops);
    gw.broadcast(&encode_msg(&GwMsg::Snapshot(snap)));
}

#[test]
fn full_gateway_loop() {
    let path = tmp_path(1);
    let mut cfg = EngineConfig::default();
    cfg.horizon = 600.0;
    cfg.dt = 0.05;
    enable_options(&mut cfg, QuoterMode::Glft, HedgeMode::WwBand);
    let mut mm = MarketMaker::new(cfg, StrategyKind::MultiLevel, 31337);
    let mut rng = Rng::new(31337);
    let mut gw = UdsGateway::bind(&path).expect("bind");
    let mut lat = LatencyRing::new(256);
    let mut seq = 0u64;

    // warm up the engine before the client connects
    pump(&mut mm, &mut gw, &mut lat, &mut rng, 50, &mut seq);

    // client connects
    let mut client = UdsClient::connect(&path).expect("connect");
    for _ in 0..100 {
        if gw.accept() > 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(gw.client_count(), 1);

    // snapshots flow to the client
    for round in 0..20 {
        pump(&mut mm, &mut gw, &mut lat, &mut rng, 10, &mut seq);
        let frames = client.recv_timeout(Duration::from_millis(200));
        assert!(!frames.is_empty(), "no frames in round {round}");
        let mut got_snap = false;
        for f in frames {
            if let Ok(GwMsg::Snapshot(s)) = decode_msg(&f) {
                got_snap = true;
                assert!(s.mid > 0.0);
                assert!(s.equity.is_finite());
                assert!(s.step_ns_p50 > 0);
                // option leg is live
                let o = s.option.expect("option leg");
                assert!(o.atm_iv > 0.0);
                assert!(!o.leg_quotes.is_empty());
            }
        }
        assert!(got_snap);
        std::thread::sleep(Duration::from_millis(1));
    }

    // commands round-trip: strategy switch, pause, manual order, take
    client
        .send(&encode_msg(&GwMsg::Command(GwCommand::SelectStrategy(0))))
        .unwrap();
    client
        .send(&encode_msg(&GwMsg::Command(GwCommand::ManualPlace {
            side: ob::Side::Bid,
            price_ticks: 190,
            lots: 2,
            post_only: true,
        })))
        .unwrap();
    client
        .send(&encode_msg(&GwMsg::Command(GwCommand::ManualTake {
            side: ob::Side::Ask,
            lots: 1,
        })))
        .unwrap();
    client.send(&encode_msg(&GwMsg::Command(GwCommand::Ping))).unwrap();
    std::thread::sleep(Duration::from_millis(20));

    pump(&mut mm, &mut gw, &mut lat, &mut rng, 5, &mut seq);
    // acks come back
    let frames = client.recv_timeout(Duration::from_millis(300));
    let mut acks = Vec::new();
    for f in frames {
        if let Ok(GwMsg::Ack(a)) = decode_msg(&f) {
            acks.push(a);
        }
    }
    assert!(acks.contains(&GwAck::Pong), "acks: {:?}", acks);
    assert!(acks.contains(&GwAck::Ok), "acks: {:?}", acks);
    // the strategy switch + manual take actually took effect
    assert_eq!(mm.strategy_current_index(), 0);
    assert!(mm.venue.taker_fills >= 1);

    // the switch persisted through more pumping
    pump(&mut mm, &mut gw, &mut lat, &mut rng, 20, &mut seq);
    let frames = client.recv_timeout(Duration::from_millis(200));
    assert!(!frames.is_empty());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn tui_app_consumes_daemon_stream() {
    let path = tmp_path(2);
    let mut cfg = EngineConfig::default();
    cfg.horizon = 300.0;
    cfg.dt = 0.05;
    enable_options(&mut cfg, QuoterMode::Glft, HedgeMode::WwBand);
    let mut mm = MarketMaker::new(cfg, StrategyKind::VolSurface, 4242);
    let mut rng = Rng::new(4242);
    let mut gw = UdsGateway::bind(&path).expect("bind");
    let _path_keep = path.clone();
    let mut lat = LatencyRing::new(128);
    let mut seq = 0u64;
    pump(&mut mm, &mut gw, &mut lat, &mut rng, 30, &mut seq);

    // a TUI-side app drives a client: every page must render with live
    // data coming off the socket.
    let tui = std::thread::spawn(move || {
        let mut client = UdsClient::connect(&path).expect("connect client");
        let mut app = tui::App::new();
        let mut screen = tui::Screen::new(100, 30);
        for _ in 0..10 {
            for frame in client.recv_timeout(Duration::from_millis(200)) {
                if let Ok(msg) = decode_msg(&frame) {
                    app.on_msg(msg);
                }
            }
            app.draw(&mut screen);
            screen.render(false);
        }
        // rendered dashboard text contains live fields
        let text: String = (0..100).map(|c| screen.front[c].ch).collect();
        assert!(text.contains("perp-mm") || text.contains("mid"));
        assert!(!app.equity_hist.is_empty());
        // simulate trading keys end-to-end
        app.on_key(tui::Key::Char('b'));
        app.on_key(tui::Key::Enter);
        let cmds = app.take_commands();
        assert_eq!(cmds.len(), 1);
        client.send(&encode_msg(&GwMsg::Command(cmds[0].clone()))).unwrap();
        std::thread::sleep(Duration::from_millis(10));
        // drain the ack
        for frame in client.recv_timeout(Duration::from_millis(100)) {
            if let Ok(GwMsg::Ack(a)) = decode_msg(&frame) {
                assert_eq!(a, GwAck::Ok);
            }
        }
    });
    // daemon side keeps pumping while the TUI runs
    for _ in 0..12 {
        pump(&mut mm, &mut gw, &mut lat, &mut rng, 10, &mut seq);
        std::thread::sleep(Duration::from_millis(8));
    }
    tui.join().expect("tui thread");
    let _ = std::fs::remove_file(&_path_keep);
}
