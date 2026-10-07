//! Deterministic replay of recorded frames.

use std::io;
use crate::codec::Frame;

/// Replays a recorded sequence of encoded frames.
pub struct ReplayTransport {
    frames: Vec<Vec<u8>>,
    pos: usize,
}

impl ReplayTransport {
    pub fn new(frames: Vec<Vec<u8>>) -> ReplayTransport {
        ReplayTransport { frames, pos: 0 }
    }

    /// Build from decoded frames (convenience for tests/sims).
    pub fn from_frames(frames: &[Frame]) -> ReplayTransport {
        let mut enc = Vec::with_capacity(frames.len());
        let mut buf = vec![0u8; 8192];
        for f in frames {
            let n = crate::codec::encode_frame(f, &mut buf).expect("encode");
            enc.push(buf[..n].to_vec());
        }
        ReplayTransport { frames: enc, pos: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.frames.len() - self.pos
    }
}

impl super::Transport for ReplayTransport {
    fn recv(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.frames.len() {
            return Ok(0);
        }
        let data = &self.frames[self.pos];
        self.pos += 1;
        let n = data.len().min(buf.len());
        buf[..n].copy_from_slice(&data[..n]);
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{Frame, MsgType, Payload};
    use crate::transport::Transport;

    #[test]
    fn replay_is_deterministic_and_exhausts() {
        let frames = vec![
            Frame {
                msg_type: MsgType::Heartbeat,
                seq: 1,
                ts_ns: 1,
                payload: Payload::Heartbeat,
            },
            Frame {
                msg_type: MsgType::LevelDelta,
                seq: 2,
                ts_ns: 2,
                payload: Payload::LevelDelta {
                    side: ob::Side::Bid,
                    price_ticks: 100,
                    delta_lots: 3,
                },
            },
        ];
        let mut t = ReplayTransport::from_frames(&frames);
        assert_eq!(t.remaining(), 2);
        let mut buf = [0u8; 256];
        assert_eq!(t.recv(&mut buf).unwrap(), 22); // heartbeat: header only
        assert_eq!(t.recv(&mut buf).unwrap(), 39); // level delta
        assert_eq!(t.recv(&mut buf).unwrap(), 0); // EOF
        assert_eq!(t.remaining(), 0);
    }
}
