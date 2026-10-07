//! Frontend & gateway hot-path benchmarks:
//! - TUI frame build + diff render (the terminal frontend's per-frame
//!   cost at 30 Hz), including the byte savings of the diff renderer
//!   vs a full repaint,
//! - gateway snapshot encode/decode,
//! - the full client loop: decode -> App update -> draw -> render.

use bench::timer::{measure_ns, Stats};
use feed::proto::{encode_msg, decode_msg, GwMsg, GwSnapshot, PnlParts};
use tui::app::App;
use tui::buf::Screen;

fn row(name: &str, s: &Stats, unit: &str) {
    println!(
        "{:<34} p50 {:>9} p99 {:>9} max {:>9} {}  (n={})",
        name, s.p50 as u64, s.p99 as u64, s.max as u64, unit, s.n
    );
}

fn demo_snapshot(seq: u64, t: usize) -> GwSnapshot {
    let mid = 100.0 + 0.5 * (t as f64 * 0.03).sin();
    let mid_tick = (mid / 0.5) as u64;
    GwSnapshot {
        seq,
        step: t as u64,
        sim_time: t as f64 * 0.05,
        mid,
        micro_price: mid + 0.02,
        spread: 0.5,
        sigma_fast: 0.02 + 0.002 * (t as f64 * 0.11).cos(),
        sigma_slow: 0.019,
        sigma_rough: 0.02,
        hurst: 0.13,
        imbalance: 0.5 + 0.2 * (t as f64 * 0.07).sin(),
        ofi: (t as f64 * 0.3).sin() * 8.0,
        jump_flag: false,
        book_bids: (0..10).map(|i| (mid_tick - i, 40 - i)).collect(),
        book_asks: (0..10).map(|i| (mid_tick + 1 + i, 38 - i)).collect(),
        our_orders: vec![(1, mid_tick, 6), (2, mid_tick + 1, 5), (1, mid_tick - 1, 8)],
        inventory: (t as i64 / 7) % 11 - 5,
        equity: 100.0 + 0.3 * (t as f64 * 0.05).sin() + t as f64 * 0.001,
        pnl: PnlParts {
            spread_capture: 1.0 + t as f64 * 0.01,
            adverse_cost: 0.2,
            fees: 0.1,
            funding: 0.0,
            hedge_cost: 0.05,
        },
        markout_mult: 1.0 + 0.1 * (t as f64 * 0.02).sin(),
        halted: false,
        paused: false,
        strategy: 6,
        option: None,
        step_ns_p50: 700 + (t % 300) as u64,
        step_ns_p99: 1500 + (t % 900) as u64,
        drops: 0,
    }
}

fn main() {
    println!("== perp-mm frontend / gateway benchmarks ==\n");

    // 1) snapshot encode / decode
    let snap = demo_snapshot(1, 100);
    let s = measure_ns(200, 20_000, || {
        let bytes = encode_msg(&GwMsg::Snapshot(snap.clone()));
        std::hint::black_box(&bytes);
    });
    row("gateway.snapshot.encode", &s, "ns");
    let bytes = encode_msg(&GwMsg::Snapshot(snap.clone()));
    let s = measure_ns(200, 20_000, || {
        let msg = decode_msg(&bytes);
        std::hint::black_box(&msg);
    });
    row("gateway.snapshot.decode", &s, "ns");
    println!("  snapshot wire size: {} bytes\n", bytes.len());

    // 2) TUI frame build + diff render at 110x34 (the standard layout)
    let mut app = App::new();
    let mut screen = Screen::new(110, 34);
    // prime with history so charts are full
    for t in 0..240 {
        app.on_msg(GwMsg::Snapshot(demo_snapshot(t as u64, t)));
    }
    screen.render(true);
    let mut t = 240usize;
    let mut diff_bytes = 0usize;
    let mut frames = 0usize;
    let s = measure_ns(50, 2_000, || {
        app.on_msg(GwMsg::Snapshot(demo_snapshot(t as u64, t)));
        t += 1;
        app.draw(&mut screen);
        let bytes = screen.render(false);
        diff_bytes += bytes.len();
        frames += 1;
        std::hint::black_box(&bytes);
    });
    row("tui.frame(build+diff render)", &s, "ns");
    println!(
        "  diff renderer: {:.1} bytes/frame average vs ~{} full repaint\n",
        diff_bytes as f64 / frames as f64,
        110 * 34 * 3
    );

    // 3) full repaint for comparison
    let s = measure_ns(20, 500, || {
        app.draw(&mut screen);
        let bytes = screen.render(true);
        std::hint::black_box(&bytes);
    });
    row("tui.frame(full repaint)", &s, "ns");

    // 4) input parsing
    use tui::input::InputParser;
    let mut parser = InputParser::new();
    let burst = b"\x1b[A\x1b[B\x1b[C\x1b[Dq\r\x1b[5~";
    let s = measure_ns(200, 20_000, || {
        parser.feed(burst);
        while let Some(k) = parser.next_key() {
            std::hint::black_box(&k);
        }
    });
    row("tui.input.parse(7-key burst)", &s, "ns\n");

    // 5) end-to-end client tick: decode + app update + draw + render
    let mut app = App::new();
    let mut screen = Screen::new(110, 34);
    for t in 0..240 {
        app.on_msg(GwMsg::Snapshot(demo_snapshot(t as u64, t)));
    }
    screen.render(true);
    let mut t = 480usize;
    let s = measure_ns(50, 2_000, || {
        let bytes = encode_msg(&GwMsg::Snapshot(demo_snapshot(t as u64, t)));
        if let Ok(msg) = decode_msg(&bytes) {
            app.on_msg(msg);
        }
        app.draw(&mut screen);
        let out = screen.render(false);
        std::hint::black_box(&out);
        t += 1;
    });
    row("client.tick(decode+draw+render)", &s, "ns");
    println!("\nAt 30 Hz the client tick budget is 33 ms; the measured cost");
    println!("leaves >99.9% of the frame budget idle.\n");
}
