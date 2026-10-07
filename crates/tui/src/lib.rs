//! # tui — the perp-mm terminal frontend
//!
//! A zero-dependency (workspace-internal) high-efficiency terminal UI:
//! raw-mode via direct syscalls, a double-buffered cell grid with a
//! minimal-byte ANSI diff renderer, braille sparklines, an L2 depth
//! ladder, latency histograms, a full keyboard-driven trading flow
//! (order entry, strategy switching, live parameter tuning, kill
//! switch) and a 7-page dashboard fed by the engine daemon over the
//! gateway protocol (Unix-domain socket).

pub mod app;
pub mod buf;
pub mod input;
pub mod sys;
pub mod widgets;

pub use app::{render_frame_text, App, Page};
pub use buf::{Cell, Color, Screen, Style};
pub use input::{InputParser, Key};
