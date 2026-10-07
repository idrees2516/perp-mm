//! `mm-tui` — the perp-mm terminal frontend.
//!
//! Connects to the engine daemon's gateway socket and renders the
//! 7-page dashboard with a minimal-byte diff renderer at ~30 Hz.
//! `--dump-frame` renders one synthetic frame to stdout (docs/tests,
//! no terminal required).
//!
//! ```text
//! mm-tui [--path /tmp/perp-mm.sock] [--dump-frame] [--cols 120] [--rows 40]
//! ```

use std::time::{Duration, Instant};

use feed::proto::{decode_msg, encode_msg, GwMsg, GwSnapshot, PnlParts};
use feed::uds::UdsClient;

use tui::app::{render_frame_text, App};
use tui::buf::Screen;
use tui::input::InputParser;
use tui::sys::{self, RawMode, POLLIN};

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn has_arg(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

/// A synthetic snapshot for `--dump-frame` (offline demo).
fn demo_snapshot() -> GwSnapshot {
    GwSnapshot {
        seq: 42,
        step: 9_876,
        sim_time: 493.8,
        mid: 101.25,
        micro_price: 101.31,
        spread: 0.5,
        sigma_fast: 0.0231,
        sigma_slow: 0.0192,
        sigma_rough: 0.0201,
        hurst: 0.14,
        imbalance: 0.62,
        ofi: 18.0,
        jump_flag: false,
        book_bids: vec![(203, 48), (202, 21), (201, 9), (200, 40), (199, 12)],
        book_asks: vec![(204, 35), (205, 17), (206, 9), (207, 40), (208, 12)],
        our_orders: vec![(1, 202, 6), (1, 201, 8), (2, 204, 5), (2, 205, 7)],
        inventory: 4,
        equity: 318.4421,
        pnl: PnlParts {
            spread_capture: 41.2,
            adverse_cost: 8.1,
            fees: 1.9,
            funding: 0.2,
            hedge_cost: 2.4,
        },
        markout_mult: 1.25,
        halted: false,
        paused: false,
        strategy: 6,
        option: Some(feed::proto::OptionSnapshot {
            atm_iv: 0.342,
            net_delta_lots: 2.4,
            net_vega: 612.0,
            net_gamma: 0.014,
            mark: 21.7,
            fills: 57,
            leg_quotes: vec![
                (0.361, 0.385),
                (0.352, 0.372),
                (0.340, 0.364),
                (0.335, 0.351),
                (0.328, 0.348),
            ],
        }),
        step_ns_p50: 812,
        step_ns_p99: 1740,
        drops: 0,
    }
}

fn main() {
    if has_arg("--dump-frame") {
        let cols: usize = arg("--cols").and_then(|s| s.parse().ok()).unwrap_or(110);
        let rows: usize = arg("--rows").and_then(|s| s.parse().ok()).unwrap_or(34);
        let mut app = App::new();
        // feed a plausible history so charts are populated
        let demo = demo_snapshot();
        for i in 0..120 {
            let mut s = demo.clone();
            s.equity = 300.0 + 0.9 * (i as f64 * 0.37).sin() + i as f64 * 0.13;
            s.spread = 0.6 + 0.2 * (i as f64 * 0.11).cos();
            s.step_ns_p50 = 600 + (90.0 * (i as f64 * 0.21).sin()) as u64;
            app.on_msg(GwMsg::Snapshot(s));
        }
        for (p, l, a) in [
            (203u64, 12u64, 1u8),
            (202, 3, 1),
            (204, 8, 2),
            (205, 21, 2),
            (203, 5, 1),
        ] {
            app.on_msg(GwMsg::Trade { price_ticks: p, lots: l, aggressor: a });
        }
        print!("{}", render_frame_text(&mut app, cols, rows));
        return;
    }

    let path = arg("--path").unwrap_or_else(|| "/tmp/perp-mm.sock".into());

    // ---- terminal setup ----
    let raw = match RawMode::enter(0) {
        Some(r) => r,
        None => {
            eprintln!("mm-tui: cannot enter raw mode (is stdin a tty?)");
            std::process::exit(1);
        }
    };
    let mut setup: Vec<u8> = Vec::with_capacity(32);
    setup.extend_from_slice(b"\x1b[?1049h\x1b[?25l"); // alt screen, hide cursor
    let _ = sys::write_fd(1, &setup);

    let ws = sys::window_size(0);
    let mut screen = Screen::new(ws.cols as usize, ws.rows as usize);
    let mut app = App::new();
    let mut parser = InputParser::new();
    let mut input_buf = [0u8; 1024];
    let mut full_repaint = true;

    // ---- connect (with retries so the TUI can start before the daemon) ----
    let mut client: Option<UdsClient> = None;
    let connect_deadline = Instant::now() + Duration::from_secs(3);
    while client.is_none() && Instant::now() < connect_deadline {
        if let Ok(c) = UdsClient::connect(&path) {
            client = Some(c);
        } else {
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    // ---- main loop ----
    let stdin_fd = 0i32;
    loop {
        // poll stdin + socket
        let sock_fd = client.as_ref().map(|c| c.fd());
        let mut fds2 = [(stdin_fd, POLLIN), (sock_fd.unwrap_or(-1), POLLIN)];
        {
            let poll_slice = if sock_fd.is_some() {
                &mut fds2[..2]
            } else {
                &mut fds2[..1]
            };
            let _ = sys::poll_fds(poll_slice, 33); // ~30 Hz
        }
        let fds = &fds2;

        // stdin
        if fds[0].1 & POLLIN != 0 {
            let n = sys::read_fd(0, &mut input_buf);
            if n > 0 {
                parser.feed(&input_buf[..n as usize]);
                while let Some(k) = parser.next_key() {
                    app.on_key(k);
                }
            } else if n == 0 {
                app.quit = true; // EOF
            }
        } else if let Some(k) = parser.flush_esc() {
            app.on_key(k);
        }

        // socket
        if let Some(c) = client.as_mut() {
            for frame in c.recv() {
                if let Ok(msg) = decode_msg(&frame) {
                    app.on_msg(msg);
                }
            }
            // send commands
            let cmds = app.take_commands();
            for cmd in cmds {
                let _ = c.send(&encode_msg(&GwMsg::Command(cmd)));
            }
        } else {
            // retry connect in the background
            if let Ok(c) = UdsClient::connect(&path) {
                client = Some(c);
            }
        }

        // resize check (cheap ioctl each frame; avoids signal handling)
        let ws = sys::window_size(0);
        if ws.cols as usize != screen.cols || ws.rows as usize != screen.rows {
            screen.resize(ws.cols as usize, ws.rows as usize);
            full_repaint = true;
        }

        // render
        let t0 = Instant::now();
        app.draw(&mut screen);
        let mut out = screen.render(full_repaint);
        full_repaint = false;
        out.extend_from_slice(b"\x1b[?25l"); // keep the cursor hidden
        let _ = sys::write_fd(1, &out);
        app.last_render_ns = t0.elapsed().as_nanos() as u64;

        if app.quit {
            break;
        }
    }

    // ---- teardown ----
    let mut teardown: Vec<u8> = Vec::with_capacity(32);
    teardown.extend_from_slice(b"\x1b[?25h\x1b[?1049l"); // show cursor, leave alt screen
    let _ = sys::write_fd(1, &teardown);
    drop(raw);
}
