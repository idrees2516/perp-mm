//! The frontend application: connection state, pages, keybindings,
//! order entry, parameter editing, and frame rendering. Pure logic —
//! the terminal I/O loop lives in the binary; everything here is
//! testable against an in-memory [`Screen`].

use std::collections::VecDeque;

use feed::proto::{GwAck, GwCommand, GwMsg, GwSnapshot};
use ob::Side;

use crate::buf::{Color, Screen, Style};
use crate::input::Key;
use crate::widgets::{depth_ladder, hbar, histogram, kv_table, mini_chart, pnl_color, readout, tag};

/// Strategy labels (wire ordering — engine's `StrategyKind::all()`).
pub const STRATEGIES: [&str; 8] = [
    "static  fixed spread benchmark",
    "as      Avellaneda-Stoikov closed form",
    "hjb     exact HJB solver policy",
    "queue   queue-aware ladder",
    "micro   micro-price centered",
    "options everlasting BSM + hedge",
    "ladder  multi-level pricing ladders",
    "volmm   SSVI vol-surface option MM",
];

/// Tunable parameter slots (mirrors engine::gateway).
pub const PARAMS: [&str; 7] = [
    "gamma (risk aversion)",
    "intensity A",
    "intensity kappa",
    "max inventory",
    "ladder levels",
    "markout theta",
    "vol-of-vol",
];

const MAX_HISTORY: usize = 240;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Page {
    Dashboard,
    Book,
    Greeks,
    Strategy,
    Risk,
    Tape,
    Help,
}

impl Page {
    pub const ALL: [Page; 7] = [
        Page::Dashboard,
        Page::Book,
        Page::Greeks,
        Page::Strategy,
        Page::Risk,
        Page::Tape,
        Page::Help,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Page::Dashboard => "DASHBOARD",
            Page::Book => "ORDER BOOK",
            Page::Greeks => "GREEKS / VOL SURFACE",
            Page::Strategy => "STRATEGY",
            Page::Risk => "RISK",
            Page::Tape => "TAPE",
            Page::Help => "HELP",
        }
    }

    pub fn index(self) -> usize {
        Page::ALL.iter().position(|&p| p == self).unwrap_or(0)
    }
}

/// Order entry scratch.
#[derive(Clone, Debug, Default)]
pub struct OrderEntry {
    pub active: bool,
    /// 1 = buy, 2 = sell.
    pub side: u8,
    pub price: String,
    pub lots: String,
    /// 0 = price field, 1 = lots field.
    pub field: u8,
}

impl OrderEntry {
    fn prefill(side: u8, touch: f64, _tick: f64) -> OrderEntry {
        OrderEntry {
            active: true,
            side,
            price: format!("{:.2}", touch),
            lots: "1".into(),
            field: 0,
        }
    }

    fn label(&self) -> &'static str {
        if self.side == 1 { "BUY " } else { "SELL" }
    }
}

/// The application state.
pub struct App {
    pub page: Page,
    pub connected: bool,
    pub quit: bool,
    pub snap: Option<GwSnapshot>,
    /// Recent trade prints (price_ticks, lots, aggressor).
    pub tape: VecDeque<(u64, u64, u8)>,
    pub equity_hist: Vec<f64>,
    pub spread_hist: Vec<f64>,
    pub latency_hist: Vec<f64>,
    /// Commands produced by key handling, drained by the main loop.
    pub outbox: Vec<GwCommand>,
    /// Last ack received.
    pub last_ack: Option<GwAck>,
    /// Selected strategy row (Strategy page).
    pub strategy_sel: u8,
    /// Selected parameter row.
    pub param_sel: u8,
    /// Pending parameter value being typed.
    pub param_entry: Option<String>,
    pub order: OrderEntry,
    /// Frame/latency diagnostics.
    pub frames: u64,
    pub last_render_ns: u64,
    pub tick_size: f64,
}

impl Default for App {
    fn default() -> Self {
        App::new()
    }
}

impl App {
    pub fn new() -> App {
        App {
            page: Page::Dashboard,
            connected: false,
            quit: false,
            snap: None,
            tape: VecDeque::with_capacity(256),
            equity_hist: Vec::with_capacity(MAX_HISTORY),
            spread_hist: Vec::with_capacity(MAX_HISTORY),
            latency_hist: Vec::with_capacity(MAX_HISTORY),
            outbox: Vec::with_capacity(8),
            last_ack: None,
            strategy_sel: 0,
            param_sel: 0,
            param_entry: None,
            order: OrderEntry::default(),
            frames: 0,
            last_render_ns: 0,
            tick_size: 0.5,
        }
    }

    // ------------------------------------------------------------------
    // Inbound messages
    // ------------------------------------------------------------------

    pub fn on_msg(&mut self, msg: GwMsg) {
        match msg {
            GwMsg::Snapshot(s) => {
                self.connected = true;
                self.equity_hist.push(s.equity);
                if self.equity_hist.len() > MAX_HISTORY {
                    self.equity_hist.remove(0);
                }
                self.spread_hist.push(s.spread);
                if self.spread_hist.len() > MAX_HISTORY {
                    self.spread_hist.remove(0);
                }
                self.latency_hist.push(s.step_ns_p50 as f64);
                if self.latency_hist.len() > MAX_HISTORY {
                    self.latency_hist.remove(0);
                }
                if let Some(o) = &s.option {
                    let _ = o;
                }
                self.snap = Some(s);
            }
            GwMsg::Trade { price_ticks, lots, aggressor } => {
                self.tape.push_front((price_ticks, lots, aggressor));
                if self.tape.len() > 200 {
                    self.tape.pop_back();
                }
            }
            GwMsg::Ack(a) => self.last_ack = Some(a),
            GwMsg::Command(_) => {}
        }
    }

    // ------------------------------------------------------------------
    // Key handling
    // ------------------------------------------------------------------

    pub fn on_key(&mut self, key: Key) {
        // Order entry intercepts everything while active.
        if self.order.active {
            self.on_key_order_entry(key);
            return;
        }
        // Parameter entry intercepts digits.
        if self.param_entry.is_some() {
            self.on_key_param_entry(key);
            return;
        }
        match key {
            Key::Char('q') => self.quit = true,
            Key::Esc => {
                if self.page == Page::Help {
                    self.page = Page::Dashboard;
                }
            }
            Key::Char('h') => self.page = Page::Help,
            Key::Tab => {
                let idx = (self.page.index() + 1) % Page::ALL.len();
                self.page = Page::ALL[idx];
            }
            Key::Char(c @ '1'..='7') => {
                let idx = (c as u8 - b'1') as usize;
                if idx < Page::ALL.len() {
                    self.page = Page::ALL[idx];
                }
            }
            Key::Char(' ') => {
                let paused = self.snap.as_ref().map(|s| s.paused).unwrap_or(false);
                self.outbox.push(GwCommand::PauseQuotes(!paused));
            }
            Key::Char('k') => {
                let halted = self.snap.as_ref().map(|s| s.halted).unwrap_or(false);
                self.outbox.push(GwCommand::KillSwitch(!halted));
            }
            Key::Char('c') => self.outbox.push(GwCommand::CancelAll),
            Key::Char('r') => self.outbox.push(GwCommand::ResetSession),
            Key::Char('p') => self.outbox.push(GwCommand::Ping),
            // Manual trading: b/s prefill a limit order at the touch;
            // B/S is an aggressive take.
            Key::Char('b') => {
                let touch = self.snap
                    .as_ref()
                    .map(|s| s.book_bids.first().map(|(p, _)| *p as f64).unwrap_or(s.mid / self.tick_size))
                    .unwrap_or(200.0);
                self.order = OrderEntry::prefill(1, touch * self.tick_size, self.tick_size);
            }
            Key::Char('s') => {
                let touch = self.snap
                    .as_ref()
                    .map(|s| s.book_asks.first().map(|(p, _)| *p as f64).unwrap_or(s.mid / self.tick_size))
                    .unwrap_or(202.0);
                self.order = OrderEntry::prefill(2, touch * self.tick_size, self.tick_size);
            }
            Key::Char('B') => self.outbox.push(GwCommand::ManualTake { side: Side::Bid, lots: 1 }),
            Key::Char('S') => self.outbox.push(GwCommand::ManualTake { side: Side::Ask, lots: 1 }),
            Key::Up | Key::Down if self.page == Page::Strategy => {
                let n = STRATEGIES.len() as u8;
                if key == Key::Up {
                    self.strategy_sel = (self.strategy_sel + n - 1) % n;
                } else {
                    self.strategy_sel = (self.strategy_sel + 1) % n;
                }
                self.outbox.push(GwCommand::SelectStrategy(self.strategy_sel));
            }
            Key::Up | Key::Down | Key::Left | Key::Right if self.page == Page::Strategy => {
                let n = PARAMS.len() as u8;
                if key == Key::Up {
                    self.param_sel = (self.param_sel + n - 1) % n;
                } else if key == Key::Down {
                    self.param_sel = (self.param_sel + 1) % n;
                }
            }
            Key::Char('+') | Key::Char('=') if self.page == Page::Strategy => {
                self.outbox.push(GwCommand::SetParam(self.param_sel, self.bump_param(1.0)));
            }
            Key::Char('-') if self.page == Page::Strategy => {
                self.outbox.push(GwCommand::SetParam(self.param_sel, self.bump_param(-1.0)));
            }
            Key::Enter if self.page == Page::Strategy => {
                self.param_entry = Some(String::new());
            }
            _ => {}
        }
    }

    /// Current value of the selected parameter slot (from the snapshot
    /// proxy fields or defaults).
    fn param_value(&self, id: u8) -> f64 {
        match id {
            3 => self.snap.as_ref().map(|s| s.inventory.abs() as f64 + 20.0).unwrap_or(20.0),
            _ => 0.5,
        }
    }

    /// Bump the selected parameter multiplicatively.
    fn bump_param(&self, dir: f64) -> f64 {
        let v = self.param_value(self.param_sel);
        if v.abs() < 1e-9 {
            return 0.1;
        }
        (v * (1.0 + 0.25 * dir)).max(0.0001)
    }

    fn on_key_order_entry(&mut self, key: Key) {
        match key {
            Key::Esc => self.order.active = false,
            Key::Enter => {
                let price_f: f64 = self.order.price.trim().replace(',', "").parse().unwrap_or(0.0);
                let price_ticks = (price_f / self.tick_size).round() as u64;
                let lots: u64 = self.order.lots.trim().parse().unwrap_or(0);
                if price_ticks > 0 && lots > 0 {
                    let side = if self.order.side == 1 { Side::Bid } else { Side::Ask };
                    self.outbox.push(GwCommand::ManualPlace {
                        side,
                        price_ticks,
                        lots,
                        post_only: true,
                    });
                }
                self.order.active = false;
            }
            Key::Tab => self.order.field = 1 - self.order.field,
            Key::Char('a') | Key::Left => {
                if let Ok(p) = self.order.price.trim().parse::<f64>() {
                    self.order.price = format!("{:.2}", (p - self.tick_size).max(self.tick_size));
                }
            }
            Key::Char('d') | Key::Right => {
                if let Ok(p) = self.order.price.trim().parse::<f64>() {
                    self.order.price = format!("{:.2}", p + self.tick_size);
                }
            }
            Key::Char('w') => {
                if let Ok(l) = self.order.lots.trim().parse::<u64>() {
                    self.order.lots = format!("{}", l + 1);
                }
            }
            Key::Char('e') => {
                if let Ok(l) = self.order.lots.trim().parse::<u64>() {
                    self.order.lots = format!("{}", l.saturating_sub(1));
                }
            }
            Key::Backspace => {
                let field = if self.order.field == 0 {
                    &mut self.order.price
                } else {
                    &mut self.order.lots
                };
                field.pop();
            }
            Key::Char(c) if c.is_ascii_digit() || c == '.' => {
                let field = if self.order.field == 0 {
                    &mut self.order.price
                } else {
                    &mut self.order.lots
                };
                if field.len() < 12 {
                    field.push(c);
                }
            }
            _ => {}
        }
    }

    fn on_key_param_entry(&mut self, key: Key) {
        let entry = self.param_entry.as_mut().unwrap();
        match key {
            Key::Esc => self.param_entry = None,
            Key::Enter => {
                if let Ok(v) = entry.trim().parse::<f64>() {
                    self.outbox.push(GwCommand::SetParam(self.param_sel, v));
                }
                self.param_entry = None;
            }
            Key::Backspace => {
                entry.pop();
            }
            Key::Char(c) if (c.is_ascii_digit() || c == '.' || c == '-')
                && entry.len() < 16 => {
                    entry.push(c);
                }
            _ => {}
        }
    }

    /// Drain pending commands.
    pub fn take_commands(&mut self) -> Vec<GwCommand> {
        std::mem::take(&mut self.outbox)
    }

    // ------------------------------------------------------------------
    // Rendering
    // ------------------------------------------------------------------

    pub fn draw(&mut self, screen: &mut Screen) {
        self.frames += 1;
        self.draw_header(screen);
        match self.page {
            Page::Dashboard => self.draw_dashboard(screen),
            Page::Book => self.draw_book(screen),
            Page::Greeks => self.draw_greeks(screen),
            Page::Strategy => self.draw_strategy(screen),
            Page::Risk => self.draw_risk(screen),
            Page::Tape => self.draw_tape(screen),
            Page::Help => self.draw_help(screen),
        }
        self.draw_footer(screen);
    }

    fn draw_header(&mut self, screen: &mut Screen) {
        let st = Style::fg(Color::White).bold().reverse();
        screen.text(0, 0, " perp-mm ", st);
        let mut col = 9;
        if let Some(s) = &self.snap {
            let t = format!(" t={:7.1}s ", s.sim_time);
            screen.text(0, col, &t, Style::fg(Color::Cyan));
            col += t.chars().count() + 1;
            let m = format!(" mid {:8.2} ", s.mid);
            screen.text(0, col, &m, Style::fg(Color::Yellow).bold());
            let _ = col + m.chars().count() + 1;
        } else {
            let m = if self.connected { " waiting for snapshots " } else { " connecting... " };
            screen.text(0, col, m, Style::fg(Color::BrightBlack));
        }
        // page tabs on the right
        let mut tabs = String::new();
        for (i, p) in Page::ALL.iter().enumerate() {
            tabs.push_str(&format!("{}:{} ", i + 1, p.title()));
        }
        screen.text_right(0, 0, screen.cols, &tabs, Style::fg(Color::BrightBlack));
        // status tags row
        let mut c = 1;
        if let Some(s) = &self.snap {
            c += tag(screen, 1, c, "HALTED", Color::BrightRed, s.halted);
            c += tag(screen, 1, c, "PAUSED", Color::Yellow, s.paused);
            c += tag(screen, 1, c, "JUMP", Color::Magenta, s.jump_flag);
            let markout = s.markout_mult > 1.05;
            c += tag(screen, 1, c, "TOXIC", Color::BrightMagenta, markout);
            let opt = s.option.is_some();
            let _ = tag(screen, 1, c, "OPT", Color::BrightCyan, opt);
        } else {
            tag(screen, 1, c, "OFFLINE", Color::BrightRed, !self.connected);
        }
        // latency right
        if let Some(s) = &self.snap {
            let lat = format!("engine p50 {:>5}ns p99 {:>5}ns", s.step_ns_p50, s.step_ns_p99);
            screen.text_right(1, 0, screen.cols, &lat, Style::fg(Color::BrightBlack));
        }
    }

    fn draw_dashboard(&mut self, screen: &mut Screen) {
        let Some(s) = self.snap.clone() else {
            screen.text(3, 2, "no snapshot yet — is mm-daemon running?", Style::fg(Color::BrightBlack));
            return;
        };
        let left_w = (screen.cols / 2).min(44);
        // left column: readouts
        let r = 3;
        readout(screen, r, 1, left_w, "micro-price", &format!("{:.3}", s.micro_price), Color::Cyan);
        readout(screen, r + 1, 1, left_w, "spread", &format!("{:.3}", s.spread), Color::Default);
        readout(screen, r + 2, 1, left_w, "sigma fast", &format!("{:.5}", s.sigma_fast), Color::Green);
        readout(screen, r + 3, 1, left_w, "sigma slow", &format!("{:.5}", s.sigma_slow), Color::Green);
        readout(screen, r + 4, 1, left_w, "sigma rough", &format!("{:.5}", s.sigma_rough), Color::Green);
        readout(screen, r + 5, 1, left_w, "hurst", &format!("{:.3}", s.hurst), Color::Magenta);
        readout(screen, r + 6, 1, left_w, "OFI", &format!("{:+.1}", s.ofi), Color::Yellow);
        readout(screen, r + 7, 1, left_w, "markout mult", &format!("x{:.2}", s.markout_mult), Color::Magenta);
        // imbalance bar
        screen.text(r + 8, 1, "imbalance", Style::default());
        hbar(screen, r + 9, 1, left_w - 2, s.imbalance, Color::Cyan, Some(0.5));
        // inventory bar centered at 0
        let inv_frac = 0.5 * (1.0 + s.inventory as f64 / 50.0);
        screen.text(r + 10, 1, "inventory", Style::default());
        let inv_col = if s.inventory >= 0 { Color::Green } else { Color::Red };
        hbar(screen, r + 11, 1, left_w - 2, inv_frac.clamp(0.0, 1.0), inv_col, Some(0.5));
        screen.text(
            r + 12,
            1,
            &format!("{} lots", s.inventory),
            Style::fg(inv_col).bold(),
        );
        // right column: charts
        let col0 = left_w + 2;
        let w = screen.cols.saturating_sub(col0 + 1);
        if w > 24 {
            mini_chart(screen, 3, col0, w, "equity", &self.equity_hist, pnl_color(s.equity));
            mini_chart(screen, 7, col0, w, "spread", &self.spread_hist, Color::Cyan);
            mini_chart(screen, 11, col0, w, "engine latency p50 (ns)", &self.latency_hist, Color::Magenta);
            // latency histogram buckets
            let mut buckets = [0f64; 8];
            for v in &self.latency_hist {
                let b = (*v / 2000.0).floor().clamp(0.0, 7.0) as usize;
                buckets[b] += 1.0;
            }
            screen.text(15, col0, "latency histogram (0-2us..14-16us)", Style::fg(Color::Cyan));
            histogram(screen, 16, col0, w.min(40), 4, &buckets, Color::Magenta);
        }
        // bottom band: our quotes
        screen.text(screen.rows.saturating_sub(4), 1, "our live orders:", Style::fg(Color::Cyan));
        let mut col = 17;
        for (i, &(sd, p, l)) in s.our_orders.iter().take(12).enumerate() {
            if col + 12 >= screen.cols {
                break;
            }
            let side_s = if sd == 1 { "B" } else { "S" };
            let st = if sd == 1 { Style::fg(Color::BrightGreen) } else { Style::fg(Color::BrightRed) };
            screen.text(
                screen.rows.saturating_sub(4),
                col,
                &format!("{}{:>5}x{}", side_s, p, l),
                st,
            );
            col += 12;
            let _ = i;
        }
    }

    fn draw_book(&mut self, screen: &mut Screen) {
        let Some(s) = self.snap.clone() else {
            screen.text(3, 2, "no snapshot yet", Style::fg(Color::BrightBlack));
            return;
        };
        let ladder_rows = screen.rows.saturating_sub(7).min(12);
        let width = screen.cols.saturating_sub(2).min(60);
        depth_ladder(
            screen,
            3,
            1,
            width,
            ladder_rows,
            &s.book_bids,
            &s.book_asks,
            &s.our_orders,
            self.tick_size,
        );
        // right side: last trades
        let col0 = width + 4;
        screen.text(3, col0, "LAST TRADES", Style::fg(Color::Cyan));
        for (i, &(p, l, ag)) in self.tape.iter().take(screen.rows.saturating_sub(6)).enumerate() {
            let st = if ag == 1 { Style::fg(Color::Green) } else { Style::fg(Color::Red) };
            screen.text(
                4 + i,
                col0,
                &format!("{:>7.2} x{:>3} {}", p as f64 * self.tick_size, l, if ag == 1 { "BUY" } else { "SELL" }),
                st,
            );
        }
        // bottom: our orders detail
        screen.text(screen.rows.saturating_sub(3), 1, "orders:", Style::fg(Color::Cyan));
        let mut col = 9;
        for &(sd, p, l) in s.our_orders.iter().take(10) {
            if col + 14 >= screen.cols {
                break;
            }
            let st = if sd == 1 { Style::fg(Color::BrightGreen) } else { Style::fg(Color::BrightRed) };
            screen.text(
                screen.rows.saturating_sub(3),
                col,
                &format!("{} #{:>6} {:>3}x{}", if sd == 1 { "BID" } else { "ASK" }, p, l, ""),
                st,
            );
            col += 16;
        }
    }

    fn draw_greeks(&mut self, screen: &mut Screen) {
        let Some(s) = self.snap.clone() else {
            screen.text(3, 2, "no snapshot yet", Style::fg(Color::BrightBlack));
            return;
        };
        let Some(o) = s.option.clone() else {
            screen.text(3, 2, "option leg disabled (enable in engine config)", Style::fg(Color::BrightBlack));
            return;
        };
        let w = screen.cols.saturating_sub(2);
        kv_table(
            screen,
            3,
            1,
            w.min(46),
            &[
                ("ATM IV", format!("{:.4} ({:.2}%)", o.atm_iv, o.atm_iv * 100.0), Color::Cyan),
                ("net delta (perp lots)", format!("{:+.2}", o.net_delta_lots), pnl_color(o.net_delta_lots)),
                ("net vega ($/vol)", format!("{:+.1}", o.net_vega), pnl_color(o.net_vega)),
                ("net gamma", format!("{:+.5}", o.net_gamma), pnl_color(o.net_gamma)),
                ("book mark", format!("{:+.4}", o.mark), pnl_color(o.mark)),
                ("client fills", format!("{}", o.fills), Color::Default),
            ],
        );
        // leg quotes table
        let col0 = w.min(46) + 3;
        screen.text(3, col0, "LEG IV QUOTES (bid/ask)", Style::fg(Color::Cyan));
        for (i, &(b, a)) in o.leg_quotes.iter().enumerate() {
            if 4 + i >= screen.rows - 2 {
                break;
            }
            let mid = 0.5 * (b + a);
            screen.text(4 + i, col0, &format!("leg{}  {:.4} / {:.4}   mid {:.4}", i + 1, b, a, mid), Style::default());
        }
        // hedging summary
        screen.text(11, 1, "delta hedge: unhedged exposure drives perp takers (SMP-protected)", Style::fg(Color::BrightBlack));
        screen.text(12, 1, &format!("perp inventory: {} lots (hedge leg)", s.inventory), Style::fg(Color::Yellow));
        let hedge_cost = s.pnl.hedge_cost;
        readout(screen, 13, 1, 40, "hedge cost", &format!("{:.4}", hedge_cost), Color::Magenta);
    }

    fn draw_strategy(&mut self, screen: &mut Screen) {
        // strategy list
        screen.text(3, 1, "STRATEGIES (Up/Down to switch)", Style::fg(Color::Cyan));
        let current = self.snap.as_ref().map(|s| s.strategy).unwrap_or(0);
        for (i, name) in STRATEGIES.iter().enumerate() {
            let r = 4 + i;
            if r >= screen.rows {
                break;
            }
            let sel = i as u8 == self.strategy_sel;
            let cur = i as u8 == current;
            let mut st = Style::default();
            if sel {
                st = st.reverse();
            }
            if cur {
                st = st.bold().bg(Color::Blue);
            }
            let marker = if cur { "> " } else { "  " };
            screen.text(r, 1, &format!("{}{}", marker, name), st);
        }
        // params
        let col0 = 40.min(screen.cols - 30);
        screen.text(3, col0, "PARAMS (+/- bump, Enter to type)", Style::fg(Color::Cyan));
        for (i, name) in PARAMS.iter().enumerate() {
            let r = 4 + i;
            if r >= screen.rows {
                break;
            }
            let sel = i as u8 == self.param_sel;
            let st = if sel { Style::fg(Color::Yellow).bold() } else { Style::default() };
            let value = if sel {
                self.param_entry.clone().unwrap_or_else(|| format!("{:.4}", self.param_value(i as u8)))
            } else {
                format!("{:.4}", self.param_value(i as u8))
            };
            screen.text(r, col0, &format!("{:<2} {:<22} {}", if sel { ">" } else { " " }, name, value), st);
        }
        if self.param_entry.is_some() {
            screen.text(12, col0, "type value + Enter, Esc cancels", Style::fg(Color::BrightBlack));
        }
    }

    fn draw_risk(&mut self, screen: &mut Screen) {
        let Some(s) = self.snap.clone() else {
            screen.text(3, 2, "no snapshot yet", Style::fg(Color::BrightBlack));
            return;
        };
        let w = screen.cols.saturating_sub(2).min(50);
        kv_table(
            screen,
            3,
            1,
            w,
            &[
                ("equity", format!("{:+.4}", s.equity), pnl_color(s.equity)),
                ("spread capture", format!("{:+.4}", s.pnl.spread_capture), pnl_color(s.pnl.spread_capture)),
                ("adverse cost", format!("{:+.4}", s.pnl.adverse_cost), Color::Red),
                ("fees", format!("{:.4}", s.pnl.fees), Color::Magenta),
                ("funding", format!("{:+.4}", s.pnl.funding), Color::Magenta),
                ("hedge cost", format!("{:.4}", s.pnl.hedge_cost), Color::Magenta),
                ("markout multiplier", format!("x{:.2}", s.markout_mult), Color::Yellow),
                ("inventory", format!("{} lots", s.inventory), if s.inventory >= 0 { Color::Green } else { Color::Red }),
                ("halted / paused", format!("{} / {}", s.halted, s.paused), Color::BrightRed),
                ("gateway drops", format!("{}", s.drops), Color::BrightBlack),
            ],
        );
        screen.text(15, 1, "kill-switch [k]   pause [space]   cancel-all [c]   reset [r]", Style::fg(Color::BrightBlack));
    }

    fn draw_tape(&mut self, screen: &mut Screen) {
        screen.text(3, 1, "PRICE     SIZE  SIDE", Style::fg(Color::Cyan));
        let n = screen.rows.saturating_sub(6);
        for (i, &(p, l, ag)) in self.tape.iter().take(n).enumerate() {
            let st = if ag == 1 { Style::fg(Color::Green) } else { Style::fg(Color::Red) };
            screen.text(
                4 + i,
                1,
                &format!("{:>8.2} {:>5} {}", p as f64 * self.tick_size, l, if ag == 1 { "BUY" } else { "SELL" }),
                st,
            );
        }
    }

    fn draw_help(&mut self, screen: &mut Screen) {
        let lines = [
            "KEYS",
            "  1-7 / Tab     switch pages",
            "  q             quit",
            "  h             this page",
            "TRADING",
            "  b / s         order entry: buy/sell limit at the touch (post-only)",
            "  B / S         aggressive take (market order, 1 lot)",
            "  Enter         submit order;  Tab toggles price/size field",
            "  a/d w/e       price -/+ tick, size -/+",
            "  Esc           cancel entry",
            "QUOTING CONTROL",
            "  space         pause/resume quotes",
            "  k             kill-switch toggle",
            "  c             cancel all orders",
            "  r             reset session accounting",
            "  p             ping the daemon",
            "STRATEGY PAGE",
            "  Up/Down       select strategy (applies immediately)",
            "  +/-           bump selected parameter",
            "  Enter         type an exact parameter value",
        ];
        for (i, l) in lines.iter().enumerate() {
            let st = if l.starts_with("  ") {
                Style::default()
            } else {
                Style::fg(Color::Cyan).bold()
            };
            screen.text(3 + i, 2, l, st);
        }
    }

    fn draw_footer(&mut self, screen: &mut Screen) {
        let r = screen.rows.saturating_sub(2);
        let col = 1;
        if self.order.active {
            let st = if self.order.side == 1 {
                Style::fg(Color::BrightGreen).bold().reverse()
            } else {
                Style::fg(Color::BrightRed).bold().reverse()
            };
            let f = if self.order.field == 0 { "price" } else { "size" };
            screen.text(
                r,
                col,
                &format!(
                    " ORDER {} price[{}] {}  lots {}  Enter=send Tab=field a/d/w/e Esc=cancel ",
                    self.order.label(),
                    f,
                    self.order.price,
                    self.order.lots
                ),
                st,
            );
        } else if let Some(a) = self.last_ack {
            let st = match a {
                GwAck::Pong | GwAck::Ok => Style::fg(Color::Green),
                _ => Style::fg(Color::BrightRed),
            };
            screen.text(r, col, &format!(" ack: {:?} ", a), st);
        }
        // frames + render time
        let stats = format!("frames {} render {:>6}ns", self.frames, self.last_render_ns);
        screen.text_right(r, 0, screen.cols, &stats, Style::fg(Color::BrightBlack));
        // bottom hint line
        let hint = " 1:dash 2:book 3:greeks 4:strategy 5:risk 6:tape 7:help | b/s trade | space pause | k kill | q quit ";
        screen.text(screen.rows.saturating_sub(1), 0, hint, Style::fg(Color::BrightBlack));
    }
}

/// Render one frame into a plain-text string (for `--dump-frame`,
/// tests and documentation).
pub fn render_frame_text(app: &mut App, cols: usize, rows: usize) -> String {
    let mut screen = Screen::new(cols, rows);
    app.draw(&mut screen);
    screen.render(true);
    let mut out = String::with_capacity(cols * rows);
    for r in 0..rows {
        let row: String = (0..cols).map(|c| screen.front[r * cols + c].ch).collect();
        out.push_str(row.trim_end());
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use feed::proto::{GwSnapshot, PnlParts};

    fn snap() -> GwSnapshot {
        GwSnapshot {
            seq: 1,
            step: 10,
            sim_time: 0.5,
            mid: 100.25,
            micro_price: 100.3,
            spread: 0.5,
            sigma_fast: 0.02,
            sigma_slow: 0.019,
            sigma_rough: 0.02,
            hurst: 0.12,
            imbalance: 0.55,
            ofi: 4.0,
            jump_flag: false,
            book_bids: vec![(200, 40), (199, 12)],
            book_asks: vec![(201, 35), (202, 8)],
            our_orders: vec![(1, 199, 5), (2, 201, 5)],
            inventory: -3,
            equity: 12.5,
            pnl: PnlParts {
                spread_capture: 2.0,
                adverse_cost: 0.5,
                fees: 0.1,
                funding: 0.0,
                hedge_cost: 0.0,
            },
            markout_mult: 1.0,
            halted: false,
            paused: false,
            strategy: 6,
            option: None,
            step_ns_p50: 780,
            step_ns_p99: 1500,
            drops: 0,
        }
    }

    #[test]
    fn pages_render_without_snapshot() {
        let mut app = App::new();
        for p in Page::ALL {
            app.page = p;
            let mut s = Screen::new(90, 30);
            app.draw(&mut s);
            let bytes = s.render(true);
            assert!(bytes.len() > 10, "{:?} empty frame", p);
        }
    }

    #[test]
    fn pages_render_with_snapshot() {
        let mut app = App::new();
        app.on_msg(GwMsg::Snapshot(snap()));
        app.on_msg(GwMsg::Trade { price_ticks: 200, lots: 3, aggressor: 1 });
        for p in Page::ALL {
            app.page = p;
            let mut s = Screen::new(90, 30);
            app.draw(&mut s);
            s.render(true);
        }
        // greeks page renders the "disabled" hint
        app.page = Page::Greeks;
        let txt = render_frame_text(&mut app, 90, 30);
        assert!(txt.contains("option leg disabled"));
    }

    #[test]
    fn keys_switch_pages_and_emit_commands() {
        let mut app = App::new();
        app.on_msg(GwMsg::Snapshot(snap()));
        app.on_key(Key::Char('3'));
        assert_eq!(app.page, Page::Greeks);
        app.on_key(Key::Char(' '));
        app.on_key(Key::Char('k'));
        app.on_key(Key::Char('c'));
        app.on_key(Key::Char('r'));
        let cmds = app.take_commands();
        assert_eq!(cmds.len(), 4);
        assert_eq!(cmds[0], GwCommand::PauseQuotes(true));
        assert_eq!(cmds[1], GwCommand::KillSwitch(true));
        assert_eq!(cmds[2], GwCommand::CancelAll);
        assert_eq!(cmds[3], GwCommand::ResetSession);
        app.on_key(Key::Char('q'));
        assert!(app.quit);
    }

    #[test]
    fn order_entry_flow() {
        let mut app = App::new();
        app.on_msg(GwMsg::Snapshot(snap()));
        app.on_key(Key::Char('b'));
        assert!(app.order.active);
        assert_eq!(app.order.side, 1);
        // pre-filled at the best bid (200 ticks * 0.5 = 100.00)
        assert!(app.order.price.contains("100.00"));
        // type more lots
        app.on_key(Key::Tab);
        app.on_key(Key::Char('2'));
        app.on_key(Key::Enter);
        let cmds = app.take_commands();
        assert_eq!(
            cmds,
            vec![GwCommand::ManualPlace {
                side: Side::Bid,
                price_ticks: 200,
                lots: 12,
                post_only: true
            }]
        );
        assert!(!app.order.active);
        // esc cancels
        app.on_key(Key::Char('s'));
        app.on_key(Key::Esc);
        assert!(!app.order.active);
        assert!(app.take_commands().is_empty());
    }

    #[test]
    fn strategy_page_selection_and_params() {
        let mut app = App::new();
        app.on_msg(GwMsg::Snapshot(snap()));
        app.page = Page::Strategy;
        app.on_key(Key::Down);
        let cmds = app.take_commands();
        assert_eq!(cmds, vec![GwCommand::SelectStrategy(1)]);
        app.on_key(Key::Char('+'));
        let cmds = app.take_commands();
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], GwCommand::SetParam(0, _)));
        // typed value
        app.on_key(Key::Enter);
        app.on_key(Key::Char('0'));
        app.on_key(Key::Char('.'));
        app.on_key(Key::Char('5'));
        app.on_key(Key::Enter);
        let cmds = app.take_commands();
        assert_eq!(cmds, vec![GwCommand::SetParam(0, 0.5)]);
    }

    #[test]
    fn history_bounded() {
        let mut app = App::new();
        for i in 0..1000 {
            let mut s = snap();
            s.equity = i as f64;
            app.on_msg(GwMsg::Snapshot(s));
        }
        assert_eq!(app.equity_hist.len(), 240);
        assert_eq!(app.tape.len(), 0);
    }

    #[test]
    fn dump_frame_text() {
        let mut app = App::new();
        app.on_msg(GwMsg::Snapshot(snap()));
        let txt = render_frame_text(&mut app, 100, 28);
        assert!(txt.contains("perp-mm"));
        assert!(txt.contains("DASHBOARD"));
        assert!(txt.lines().count() == 28);
    }
}
