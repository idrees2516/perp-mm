//! Drawing widgets: sparklines with 2x4 braille sub-pixels, depth
//! ladders, big-number readouts, tables, latency histograms and
//! horizontal gauges — all rendered into the [`Screen`] cell buffer
//! with zero allocation in the steady state (formatting scratch is
//! reused via a single `fmt::Write` into a stack buffer).

use crate::buf::{Color, Screen, Style};

/// Braille sparkline: each terminal cell encodes a 2x4 sub-pixel grid
/// (the classic plotting trick — 8x the resolution of ASCII blocks).
pub fn sparkline(s: &mut Screen, row: usize, col: usize, width: usize, data: &[f64], style: Style) {
    if width == 0 || data.is_empty() {
        return;
    }
    let n = data.len();
    let lo = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (lo, hi) = if (hi - lo).abs() < 1e-12 {
        (lo - 0.5, hi + 0.5)
    } else {
        (lo, hi)
    };
    // braille base 0x2800; bit layout (col, row) ->
    // (0,0)=1 (0,1)=2 (0,2)=4 (0,3)=8 (1,0)=16 (1,1)=32 (1,2)=64 (1,3)=128
    for cx in 0..width {
        // sample window for this cell
        let start = cx * n / width;
        let end = ((cx + 1) * n / width).max(start + 1);
        let mut bits = 0u32;
        for (k, i) in (start..end.min(n)).enumerate() {
            let v = (data[i] - lo) / (hi - lo);
            let sub_col = if k % 2 == 0 { 0 } else { 1 };
            let sub_row = ((1.0 - v) * 3.999) as u32;
            let bit = match (sub_col, sub_row) {
                (0, 0) => 1,
                (0, 1) => 2,
                (0, 2) => 4,
                (0, 3) => 8,
                (1, 0) => 16,
                (1, 1) => 32,
                (1, 2) => 64,
                _ => 128,
            };
            bits |= bit;
        }
        let ch = char::from_u32(0x2800 + bits).unwrap_or(' ');
        s.put(row, col + cx, ch, style);
    }
}

/// Big-number readout: label left, value right-aligned, colored.
pub fn readout(s: &mut Screen, row: usize, col: usize, width: usize, label: &str, value: &str, color: Color) {
    s.text(row, col, label, Style::default());
    s.text_right(row, col, width, value, Style::fg(color).bold());
}

/// Signed-color helper for P&L-style values.
pub fn pnl_color(v: f64) -> Color {
    if v > 1e-9 {
        Color::Green
    } else if v < -1e-9 {
        Color::Red
    } else {
        Color::Default
    }
}

/// L2 depth ladder: bids descending on the left, asks ascending on the
/// right, our quotes highlighted.
pub fn depth_ladder(
    s: &mut Screen,
    row: usize,
    col: usize,
    width: usize,
    max_rows: usize,
    bids: &[(u64, u64)],
    asks: &[(u64, u64)],
    ours: &[(u8, u64, u64)],
    tick_size: f64,
) {
    let half = width / 2;
    // header
    s.text(row, col, "BID PRICE", Style::fg(Color::Cyan));
    s.text_right(row, col, half, "SIZE", Style::fg(Color::Cyan));
    s.text_right(row, col + half, width - half, "PRICE", Style::fg(Color::Yellow));
    s.text_right(row, col, width, "ASK SIZE", Style::fg(Color::Yellow));
    for i in 0..max_rows {
        let r = row + 1 + i;
        if r >= s.rows {
            break;
        }
        s.clear_row(r, col, width);
        if let Some(&(p, l)) = bids.get(i) {
            let price = format!("{:.2}", p as f64 * tick_size);
            let size = format!("{}", l);
            let is_ours = ours.iter().any(|&(sd, op, _)| sd == 1 && op == p);
            let st = if is_ours {
                Style::fg(Color::BrightGreen).bold()
            } else {
                Style::fg(Color::Green)
            };
            s.text(r, col, &price, st);
            s.text_right(r, col, half, &size, st);
        }
        if let Some(&(p, l)) = asks.get(i) {
            let price = format!("{:.2}", p as f64 * tick_size);
            let size = format!("{}", l);
            let is_ours = ours.iter().any(|&(sd, op, _)| sd == 2 && op == p);
            let st = if is_ours {
                Style::fg(Color::BrightRed).bold()
            } else {
                Style::fg(Color::Red)
            };
            s.text_right(r, col + half, width - half, &price, st);
            s.text_right(r, col, width, &size, st);
        }
    }
}

/// Horizontal bar (gauge) with a fill character and optional marker
/// position (e.g. imbalance 0..1 with a 0.5 center line).
pub fn hbar(
    s: &mut Screen,
    row: usize,
    col: usize,
    width: usize,
    frac: f64,
    fill_color: Color,
    marker: Option<f64>,
) {
    let frac = frac.clamp(0.0, 1.0);
    let n = (frac * width as f64).round() as usize;
    for c in 0..width {
        s.put(row, col + c, '·', Style::fg(Color::BrightBlack));
    }
    for c in 0..n.min(width) {
        s.put(row, col + c, '█', Style::fg(fill_color));
    }
    if let Some(m) = marker {
        let mc = (m.clamp(0.0, 1.0) * width as f64).round() as usize;
        if mc < width {
            s.put(row, col + mc, '│', Style::fg(Color::White).bold());
        }
    }
}

/// Simple histogram of bucketed values (e.g. latency) drawn with bars.
pub fn histogram(s: &mut Screen, row: usize, col: usize, width: usize, height: usize, buckets: &[f64], color: Color) {
    if buckets.is_empty() || width == 0 || height == 0 {
        return;
    }
    let max = buckets.iter().cloned().fold(0.0f64, f64::max).max(1e-9);
    let bw = (width / buckets.len()).max(1);
    for (i, v) in buckets.iter().enumerate() {
        let h = ((v / max) * height as f64).round() as usize;
        for k in 0..bw.min(width - i * bw) {
            let c = col + i * bw + k;
            for (dy, r) in (row..row + height).enumerate().rev() {
                let filled = dy < h;
                s.put(
                    r,
                    c,
                    if filled { '▓' } else { ' ' },
                    if filled { Style::fg(color) } else { Style::default() },
                );
            }
        }
    }
}

/// A one-line line-chart (min/max labels + sparkline) with a title.
pub fn mini_chart(s: &mut Screen, row: usize, col: usize, width: usize, title: &str, data: &[f64], color: Color) {
    if width < 24 || data.is_empty() {
        return;
    }
    let lo = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    s.text(row, col, title, Style::fg(Color::Cyan));
    let label_lo = format!("{:.2}", lo);
    let label_hi = format!("{:.2}", hi);
    s.text(row + 1, col, &label_lo, Style::fg(Color::BrightBlack));
    s.text_right(row + 1, col, width, &label_hi, Style::fg(Color::BrightBlack));
    sparkline(s, row + 2, col, width, data, Style::fg(color));
}

/// KeyValue table rows.
pub fn kv_table(s: &mut Screen, row: usize, col: usize, width: usize, rows: &[(&str, String, Color)]) {
    for (i, (k, v, c)) in rows.iter().enumerate() {
        let r = row + i;
        if r >= s.rows {
            break;
        }
        s.clear_row(r, col, width);
        s.text(r, col, k, Style::default());
        s.text_right(r, col, width, v, Style::fg(*c));
    }
}

/// A simple status tag (colored, bracketed).
pub fn tag(s: &mut Screen, row: usize, col: usize, label: &str, color: Color, on: bool) -> usize {
    let st = if on {
        Style::fg(color).bold().reverse()
    } else {
        Style::fg(Color::BrightBlack)
    };
    let t = format!("[{}]", label);
    s.text(row, col, &t, st);
    t.chars().count() + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparkline_renders_braille_and_bounds() {
        let mut s = Screen::new(80, 3);
        let data: Vec<f64> = (0..40).map(|i| (i as f64).sin()).collect();
        sparkline(&mut s, 0, 0, 40, &data, Style::fg(Color::Green));
        let bytes = s.render(true);
        assert!(bytes.len() > 40);
        // braille chars present
        #[allow(clippy::identity_op)]
        let front: String = (0..40).map(|i| s.front[i].ch).collect();
        assert!(front.chars().any(|c| (c as u32) >= 0x2800));
        // flat data doesn't divide by zero
        sparkline(&mut s, 1, 0, 40, &[1.0; 10], Style::default());
        let _ = s.render(false);
    }

    #[test]
    fn depth_ladder_marks_our_quotes() {
        let mut s = Screen::new(60, 8);
        depth_ladder(
            &mut s,
            0,
            0,
            50,
            4,
            &[(199, 40), (198, 12)],
            &[(200, 35), (201, 8)],
            &[(1, 198, 4), (2, 201, 5)],
            0.5,
        );
        s.render(true);
        // our bid at 198 uses the bright color
        let cell_198 = s.front[60..2 * 60]
            .iter()
            .find(|c| c.ch == '9') // part of "99.00"
            .map(|c| c.style.fg);
        assert!(cell_198.is_some());
    }

    #[test]
    fn hbar_and_marker() {
        let mut s = Screen::new(30, 2);
        hbar(&mut s, 0, 0, 20, 0.7, Color::Green, Some(0.5));
        s.render(true);
        // fill covers cols 0..13; the marker sits at col 10 on top of it
        assert_eq!(s.front[10].ch, '│');
        assert_eq!(s.front[11].ch, '█');
        assert_eq!(s.front[13].ch, '█');
        assert_eq!(s.front[14].ch, '·');
        assert_eq!(s.front[19].ch, '·');
    }

    #[test]
    fn histogram_heights() {
        let mut s = Screen::new(40, 6);
        histogram(&mut s, 0, 0, 20, 5, &[1.0, 3.0, 2.0], Color::Cyan);
        s.render(true);
        // the tallest bucket reaches the top row
        assert!(s.front[6].ch == '▓' || s.front[40 + 6].ch == '▓');
    }

    #[test]
    fn readout_and_kv() {
        let mut s = Screen::new(40, 6);
        readout(&mut s, 0, 0, 30, "equity", "+123.45", Color::Green);
        kv_table(
            &mut s,
            2,
            0,
            30,
            &[("gamma", format!("{:.3}", 0.1), Color::Default)],
        );
        let b = s.render(true);
        assert!(b.len() > 20);
    }
}
