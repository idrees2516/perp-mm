//! Terminal input decoding: raw bytes -> [`Key`] events, covering the
//! common xterm sequences (arrows, Home/End/PgUp/PgDn, F1-F4, Delete)
//! plus plain ASCII, Ctrl-<letter>, Enter, Backspace, Tab, Esc.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    Char(char),
    Ctrl(char),
    Enter,
    Backspace,
    Tab,
    Esc,
    Up,
    Down,
    Left,
    Right,
    Home,
    End,
    PgUp,
    PgDn,
    Delete,
    Insert,
    F(u8),
    Unknown,
}

/// Incremental parser: feed bytes, pull keys.
#[derive(Default)]
pub struct InputParser {
    buf: Vec<u8>,
}

impl InputParser {
    pub fn new() -> InputParser {
        InputParser { buf: Vec::with_capacity(32) }
    }

    /// Feed raw bytes.
    pub fn feed(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
        if self.buf.len() > 256 {
            self.buf.clear(); // abuse guard
        }
    }

    /// Pull the next complete key if one is buffered.
    pub fn next_key(&mut self) -> Option<Key> {
        if self.buf.is_empty() {
            return None;
        }
        let b = self.buf[0];
        // UTF-8 multibyte
        if b >= 0x80 {
            let len = utf8_len(b);
            if self.buf.len() < len {
                return None; // wait for more bytes
            }
            let s: Vec<u8> = self.buf.drain(..len).collect();
            if let Ok(st) = std::str::from_utf8(&s) {
                if let Some(ch) = st.chars().next() {
                    return Some(Key::Char(ch));
                }
            }
            return Some(Key::Unknown);
        }
        match b {
            0x1b => {
                if self.buf.len() == 1 {
                    // Lone ESC: could be the start of a sequence; peek
                    // with a small wait by returning None until more
                    // bytes arrive or the caller flushes.
                    return None;
                }
                let b1 = self.buf[1];
                if b1 == b'[' || b1 == b'O' {
                    // find terminator
                    let end = self.buf[2..]
                        .iter()
                        .position(|&c| (0x40..=0x7e).contains(&c))
                        .map(|p| p + 2);
                    let Some(end) = end else {
                        if self.buf.len() > 16 {
                            self.buf.clear();
                            return Some(Key::Unknown);
                        }
                        return None;
                    };
                    let seq: Vec<u8> = self.buf.drain(..=end).collect();
                    Some(parse_csi(&seq))
                } else {
                    // ESC + char = Alt-<char>; treat as Esc
                    self.buf.drain(..2);
                    Some(Key::Esc)
                }
            }
            b'\r' | b'\n' => {
                self.buf.remove(0);
                Some(Key::Enter)
            }
            b'\t' => {
                self.buf.remove(0);
                Some(Key::Tab)
            }
            0x7f | 0x08 => {
                self.buf.remove(0);
                Some(Key::Backspace)
            }
            0x01..=0x1a => {
                self.buf.remove(0);
                Some(Key::Ctrl((b - 1 + b'a') as char))
            }
            _ => {
                self.buf.remove(0);
                Some(Key::Char(b as char))
            }
        }
    }

    /// Flush a pending lone ESC (call when no more bytes arrive this
    /// cycle).
    pub fn flush_esc(&mut self) -> Option<Key> {
        if self.buf.len() == 1 && self.buf[0] == 0x1b {
            self.buf.clear();
            return Some(Key::Esc);
        }
        None
    }

    /// Buffered bytes pending (diagnostics).
    pub fn pending(&self) -> usize {
        self.buf.len()
    }
}

fn utf8_len(b: u8) -> usize {
    if b >= 0xf0 {
        4
    } else if b >= 0xe0 {
        3
    } else {
        2
    }
}

fn parse_csi(seq: &[u8]) -> Key {
    // seq[0] = ESC, seq[1] = '[' or 'O', last = final byte
    let body: &[u8] = &seq[2..seq.len() - 1];
    let final_byte = *seq.last().unwrap();
    match final_byte {
        b'A' => Key::Up,
        b'B' => Key::Down,
        b'C' => Key::Right,
        b'D' => Key::Left,
        b'H' => Key::Home,
        b'F' => Key::End,
        b'Z' => Key::Tab,
        b'~' => {
            let num: u32 = std::str::from_utf8(body)
                .ok()
                .and_then(|s| s.split(';').next().and_then(|x| x.parse().ok()))
                .unwrap_or(0);
            match num {
                1 | 7 => Key::Home,
                2 => Key::Insert,
                3 => Key::Delete,
                4 | 8 => Key::End,
                5 => Key::PgUp,
                6 => Key::PgDn,
                11..=15 => Key::F((num - 10) as u8),
                17..=21 => Key::F((num - 11) as u8),
                23..=24 => Key::F((num - 12) as u8),
                _ => Key::Unknown,
            }
        }
        b'P' => Key::F(1),
        b'Q' => Key::F(2),
        b'R' => Key::F(3),
        b'S' => Key::F(4),
        _ => Key::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(bytes: &[u8]) -> Vec<Key> {
        let mut p = InputParser::new();
        p.feed(bytes);
        let mut out = Vec::new();
        while let Some(k) = p.next_key() {
            out.push(k);
        }
        out
    }

    #[test]
    fn plain_chars() {
        assert_eq!(parse(b"abc"), vec![Key::Char('a'), Key::Char('b'), Key::Char('c')]);
        assert_eq!(parse(b"\r"), vec![Key::Enter]);
        assert_eq!(parse(b"\x7f"), vec![Key::Backspace]);
        assert_eq!(parse(b"\t"), vec![Key::Tab]);
        assert_eq!(parse(b"\x01"), vec![Key::Ctrl('a')]);
        assert_eq!(parse(b"\x03"), vec![Key::Ctrl('c')]);
    }

    #[test]
    fn arrows_and_specials() {
        assert_eq!(parse(b"\x1b[A"), vec![Key::Up]);
        assert_eq!(parse(b"\x1b[B"), vec![Key::Down]);
        assert_eq!(parse(b"\x1b[C"), vec![Key::Right]);
        assert_eq!(parse(b"\x1b[D"), vec![Key::Left]);
        assert_eq!(parse(b"\x1b[H"), vec![Key::Home]);
        assert_eq!(parse(b"\x1b[F"), vec![Key::End]);
        assert_eq!(parse(b"\x1b[5~"), vec![Key::PgUp]);
        assert_eq!(parse(b"\x1b[6~"), vec![Key::PgDn]);
        assert_eq!(parse(b"\x1b[3~"), vec![Key::Delete]);
        assert_eq!(parse(b"\x1bOP"), vec![Key::F(1)]);
        assert_eq!(parse(b"\x1b[15~"), vec![Key::F(5)]);
    }

    #[test]
    fn sequences_split_correctly() {
        assert_eq!(
            parse(b"\x1b[A\x1b[Bx"),
            vec![Key::Up, Key::Down, Key::Char('x')]
        );
    }

    #[test]
    fn utf8_chars() {
        assert_eq!(parse("é".as_bytes()), vec![Key::Char('é')]);
        assert_eq!(parse("λ".as_bytes()), vec![Key::Char('λ')]);
    }

    #[test]
    fn partial_sequences_wait() {
        let mut p = InputParser::new();
        p.feed(b"\x1b[");
        assert_eq!(p.next_key(), None);
        p.feed(b"A");
        assert_eq!(p.next_key(), Some(Key::Up));
    }

    #[test]
    fn lone_esc_flushes() {
        let mut p = InputParser::new();
        p.feed(b"\x1b");
        assert_eq!(p.next_key(), None);
        assert_eq!(p.flush_esc(), Some(Key::Esc));
        assert_eq!(p.pending(), 0);
    }

    #[test]
    fn mouse_like_sequences_do_not_hang() {
        // SGR mouse sequence: ESC [ < 0 ; 1 ; 1 M
        let mut p = InputParser::new();
        p.feed(b"\x1b[<0;1;1M");
        let k = p.next_key();
        assert!(k.is_some());
        assert_eq!(p.pending(), 0);
    }
}
