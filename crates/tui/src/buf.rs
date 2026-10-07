//! Double-buffered cell grid with a minimal-byte ANSI diff renderer.
//!
//! The renderer walks the back buffer row by row and emits only the
//! runs that differ from the front buffer, moving the cursor with
//! absolute positioning when the gap between dirty runs is large (and
//! letting runs flow when small), emitting SGR sequences only when the
//! style changes. A typical 20 Hz dashboard update touches a few
//! hundred cells and costs a few KB of escape codes instead of a full
//! repaint (~10x-50x fewer bytes — asserted in the tests).

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Color {
    Default,
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White,
    BrightBlack,
    BrightRed,
    BrightGreen,
    BrightYellow,
    BrightBlue,
    BrightMagenta,
    BrightCyan,
    BrightWhite,
}

impl Color {
    fn sgr(self, fg: bool) -> String {
        let base = match self {
            Color::Default => return "0".into(),
            Color::Black => 0,
            Color::Red => 1,
            Color::Green => 2,
            Color::Yellow => 3,
            Color::Blue => 4,
            Color::Magenta => 5,
            Color::Cyan => 6,
            Color::White => 7,
            Color::BrightBlack => 8,
            Color::BrightRed => 9,
            Color::BrightGreen => 10,
            Color::BrightYellow => 11,
            Color::BrightBlue => 12,
            Color::BrightMagenta => 13,
            Color::BrightCyan => 14,
            Color::BrightWhite => 15,
        };
        if fg {
            if base >= 8 {
                format!("9{}", base - 8)
            } else {
                format!("3{}", base)
            }
        } else if base >= 8 {
            format!("10{}", base - 8)
        } else {
            format!("4{}", base)
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Style {
    pub fg: Color,
    pub bg: Color,
    pub bold: bool,
    pub reverse: bool,
}

impl Default for Style {
    fn default() -> Self {
        Style { fg: Color::Default, bg: Color::Default, bold: false, reverse: false }
    }
}

impl Style {
    pub fn fg(color: Color) -> Style {
        Style { fg: color, ..Style::default() }
    }

    pub fn bold(mut self) -> Style {
        self.bold = true;
        self
    }

    pub fn reverse(mut self) -> Style {
        self.reverse = true;
        self
    }

    pub fn bg(mut self, color: Color) -> Style {
        self.bg = color;
        self
    }

    fn sgr(&self) -> String {
        let mut parts: Vec<String> = Vec::with_capacity(4);
        if self.fg != Color::Default {
            parts.push(self.fg.sgr(true));
        }
        if self.bg != Color::Default {
            parts.push(self.bg.sgr(false));
        }
        if self.bold {
            parts.push("1".into());
        }
        if self.reverse {
            parts.push("7".into());
        }
        if parts.is_empty() {
            "0".into()
        } else {
            parts.join(";")
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Cell {
    pub ch: char,
    pub style: Style,
}

impl Default for Cell {
    fn default() -> Self {
        Cell { ch: ' ', style: Style::default() }
    }
}

/// A fixed-size back buffer plus the front buffer state.
pub struct Screen {
    pub cols: usize,
    pub rows: usize,
    pub cells: Vec<Cell>,
    pub front: Vec<Cell>,
    /// Bytes emitted by the last render (diagnostics/bench).
    pub last_bytes: usize,
    /// Cells changed by the last render.
    pub last_dirty: usize,
}

impl Screen {
    pub fn new(cols: usize, rows: usize) -> Screen {
        let n = cols * rows;
        Screen {
            cols: cols.max(1),
            rows: rows.max(1),
            cells: vec![Cell::default(); n],
            front: vec![
                Cell {
                    ch: '\u{0}',
                    style: Style {
                        fg: Color::Black,
                        bg: Color::White,
                        bold: true,
                        reverse: true
                    }
                };
                n
            ],
            last_bytes: 0,
            last_dirty: 0,
        }
    }

    pub fn resize(&mut self, cols: usize, rows: usize) {
        let cols = cols.max(1);
        let rows = rows.max(1);
        if cols == self.cols && rows == self.rows {
            return;
        }
        *self = Screen::new(cols, rows);
    }

    #[inline]
    fn idx(&self, row: usize, col: usize) -> Option<usize> {
        if row < self.rows && col < self.cols {
            Some(row * self.cols + col)
        } else {
            None
        }
    }

    /// Write one styled char.
    pub fn put(&mut self, row: usize, col: usize, ch: char, style: Style) {
        if let Some(i) = self.idx(row, col) {
            self.cells[i] = Cell { ch, style };
        }
    }

    /// Write a string clipped to the width; returns the end column.
    pub fn text(&mut self, row: usize, col: usize, s: &str, style: Style) -> usize {
        let mut c = col;
        for ch in s.chars() {
            if c >= self.cols {
                break;
            }
            self.put(row, c, ch, style);
            c += 1;
        }
        c
    }

    /// Write a right-aligned string in [col, col+width).
    pub fn text_right(&mut self, row: usize, col: usize, width: usize, s: &str, style: Style) {
        let n = s.chars().count().min(width);
        let start = col + width - n;
        self.text(row, start, &s.chars().take(n).collect::<String>(), style);
    }

    /// Clear a row segment.
    pub fn clear_row(&mut self, row: usize, col: usize, width: usize) {
        let end = (col + width).min(self.cols);
        for c in col..end {
            self.put(row, c, ' ', Style::default());
        }
    }

    /// Fill a rect with a style.
    pub fn fill(&mut self, row: usize, col: usize, w: usize, h: usize, style: Style) {
        for r in row..(row + h).min(self.rows) {
            for c in col..(col + w).min(self.cols) {
                self.put(r, c, ' ', style);
            }
        }
    }

    /// Draw a single-line box border.
    pub fn box_border(&mut self, row: usize, col: usize, w: usize, h: usize, style: Style) {
        let (r2, c2) = (row + h.saturating_sub(1), col + w.saturating_sub(1));
        for c in col..=c2.min(self.cols.saturating_sub(1)) {
            self.put(row, c, '─', style);
            self.put(r2, c, '─', style);
        }
        for r in row..=r2.min(self.rows.saturating_sub(1)) {
            self.put(r, col, '│', style);
            self.put(r, c2, '│', style);
        }
        // corners (bounds-checked by put)
        self.put(row, col, '┌', style);
        self.put(row, c2, '┐', style);
        self.put(r2, col, '└', style);
        self.put(r2, c2, '┘', style);
    }

    /// Render the diff into ANSI bytes. On `full`, emit a clear-screen
    /// repaint (used after resize or first frame).
    pub fn render(&mut self, full: bool) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::with_capacity(4096);
        if full {
            out.extend_from_slice(b"\x1b[2J");
            // force full repaint by invalidating the front buffer
            for f in self.front.iter_mut() {
                *f = Cell {
                    ch: '\u{0}',
                    style: Style {
                        fg: Color::Black,
                        bg: Color::White,
                        bold: true,
                        reverse: true
                    },
                };
            }
        }
        let mut cur_style: Option<Style> = None;
        let mut dirty = 0usize;
        for r in 0..self.rows {
            let mut c = 0usize;
            let mut cursor_col: Option<usize> = None;
            while c < self.cols {
                let i = r * self.cols + c;
                if self.cells[i] == self.front[i] {
                    if cursor_col.is_some() {
                        cursor_col = None;
                    }
                    c += 1;
                    continue;
                }
                // start of a dirty run: position the cursor
                match cursor_col {
                    None => {
                        out.extend_from_slice(format!("\x1b[{};{}H", r + 1, c + 1).as_bytes());
                    }
                    Some(prev) if c > prev + 1 => {
                        out.extend_from_slice(format!("\x1b[{};{}H", r + 1, c + 1).as_bytes());
                    }
                    _ => {}
                }
                // style change?
                if cur_style != Some(self.cells[i].style) {
                    out.extend_from_slice(format!("\x1b[{}m", self.cells[i].style.sgr()).as_bytes());
                    cur_style = Some(self.cells[i].style);
                }
                let mut buf = [0u8; 4];
                out.extend_from_slice(self.cells[i].ch.encode_utf8(&mut buf).as_bytes());
                self.front[i] = self.cells[i];
                dirty += 1;
                cursor_col = Some(c);
                c += 1;
            }
        }
        // reset style at the end
        if cur_style.is_some() {
            out.extend_from_slice(b"\x1b[0m");
        }
        self.last_bytes = out.len();
        self.last_dirty = dirty;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_render_repaints_everything() {
        let mut s = Screen::new(20, 5);
        s.text(0, 0, "hello", Style::fg(Color::Green).bold());
        let bytes = s.render(true);
        assert!(s.last_dirty >= 5);
        assert!(bytes.len() > 10);
        // second full render with no changes: only the clear + zero cells
        s.render(false);
        assert_eq!(s.last_dirty, 0);
    }

    #[test]
    fn diff_render_is_cheap() {
        let mut s = Screen::new(100, 30);
        s.render(true);
        // big static screen
        for r in 0..30 {
            s.text(r, 0, &"x".repeat(80), Style::default());
        }
        let full = s.render(true);
        let full_bytes = full.len();
        assert!(full_bytes > 2000, "full repaint {full_bytes}");
        // change only a few cells
        s.text(0, 0, "Y", Style::fg(Color::Red));
        s.text(15, 40, "Z", Style::fg(Color::Red));
        let diff = s.render(false);
        assert!(s.last_dirty <= 2, "dirty {}", s.last_dirty);
        assert!(
            diff.len() < full_bytes / 20,
            "diff {} vs full {}",
            diff.len(),
            full_bytes
        );
    }

    #[test]
    fn clipping_stays_in_bounds() {
        let mut s = Screen::new(10, 3);
        s.text(0, 8, "abcdefghij", Style::default());
        s.text(5, 0, "out of range", Style::default());
        s.box_border(2, 8, 100, 100, Style::default());
        let bytes = s.render(true);
        assert!(!bytes.is_empty());
    }

    #[test]
    fn text_right_aligns() {
        let mut s = Screen::new(20, 2);
        s.text_right(0, 0, 10, "123456", Style::default());
        s.render(true);
        // "123456" right-aligned in width 10 starts at col 4
        let i = 4;
        assert_eq!(s.front[i].ch, '1');
        assert_eq!(s.front[9].ch, '6');
    }

    #[test]
    fn style_sgr_codes() {
        assert_eq!(Style::default().sgr(), "0");
        assert_eq!(Style::fg(Color::Red).sgr(), "31");
        assert_eq!(Style::fg(Color::BrightRed).sgr(), "91");
        let st = Style::fg(Color::Green).bold();
        assert_eq!(st.sgr(), "32;1");
        let st2 = Style::default().bg(Color::Blue).reverse();
        assert_eq!(st2.sgr(), "44;7");
    }
}
