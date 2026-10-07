//! Compact binary frame codec.
//!
//! Layout (little-endian, fixed widths, no alignment padding):
//! ```text
//! header (22 bytes):
//!   u16 magic = 0xCF11 | u8 version = 1 | u8 msg_type | u64 seq |
//!   i64 ts_ns | u16 payload_len
//! payloads:
//!   SNAPSHOT:   u32 bid_levels, u32 ask_levels, then per level
//!               (u64 price_ticks, u64 lots)
//!   NEW_ORDER:  u64 order_id, u8 side, u64 price_ticks, u64 lots
//!   CANCEL:     u64 order_id
//!   LEVEL:      u8 side, u64 price_ticks, i64 delta_lots
//!   TRADE:      u64 price_ticks, u64 lots, u8 aggressor
//!   FILL:       u64 order_id, u64 filled_lots, u64 remaining_lots,
//!               u64 price_ticks
//!   HEARTBEAT:  (empty)
//! ```

use ob::Side;

pub const CODEC_MAGIC: u16 = 0xCF11;
pub const CODEC_VERSION: u8 = 1;
pub const HEADER_LEN: usize = 22;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MsgType {
    Snapshot = 1,
    NewOrder = 2,
    CancelOrder = 3,
    LevelDelta = 4,
    Trade = 5,
    Fill = 6,
    Heartbeat = 7,
}

impl MsgType {
    fn from_u8(v: u8) -> Option<MsgType> {
        Some(match v {
            1 => MsgType::Snapshot,
            2 => MsgType::NewOrder,
            3 => MsgType::CancelOrder,
            4 => MsgType::LevelDelta,
            5 => MsgType::Trade,
            6 => MsgType::Fill,
            7 => MsgType::Heartbeat,
            _ => return None,
        })
    }
}

/// Decoded payload variants.
#[derive(Clone, Debug, PartialEq)]
pub enum Payload {
    Snapshot {
        bids: Vec<(u64, u64)>,
        asks: Vec<(u64, u64)>,
    },
    NewOrder {
        order_id: u64,
        side: Side,
        price_ticks: u64,
        lots: u64,
    },
    CancelOrder {
        order_id: u64,
    },
    LevelDelta {
        side: Side,
        price_ticks: u64,
        delta_lots: i64,
    },
    Trade {
        price_ticks: u64,
        lots: u64,
        aggressor: Side,
    },
    Fill {
        order_id: u64,
        filled_lots: u64,
        remaining_lots: u64,
        price_ticks: u64,
    },
    Heartbeat,
}

/// One decoded frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub msg_type: MsgType,
    pub seq: u64,
    pub ts_ns: i64,
    pub payload: Payload,
}

fn side_to_u8(s: Side) -> u8 {
    match s {
        Side::Bid => 0,
        Side::Ask => 1,
    }
}

fn side_from_u8(v: u8) -> Option<Side> {
    Some(match v {
        0 => Side::Bid,
        1 => Side::Ask,
        _ => return None,
    })
}

/// Encode a frame into `buf`; returns the total bytes written.
pub fn encode_frame(frame: &Frame, buf: &mut [u8]) -> Result<usize, String> {
    let plen = payload_len(&frame.payload);
    let total = HEADER_LEN + plen;
    if buf.len() < total {
        return Err(format!("buffer too small: {} < {}", buf.len(), total));
    }
    buf[0..2].copy_from_slice(&CODEC_MAGIC.to_le_bytes());
    buf[2] = CODEC_VERSION;
    buf[3] = frame.msg_type as u8;
    buf[4..12].copy_from_slice(&frame.seq.to_le_bytes());
    buf[12..20].copy_from_slice(&frame.ts_ns.to_le_bytes());
    buf[20..22].copy_from_slice(&(plen as u16).to_le_bytes());
    #[allow(unused_mut)]
    let mut p = &mut buf[HEADER_LEN..];
    encode_payload(&frame.payload, p)?;
    Ok(total)
}

fn payload_len(p: &Payload) -> usize {
    match p {
        Payload::Snapshot { bids, asks } => 8 + (bids.len() + asks.len()) * 16,
        Payload::NewOrder { .. } => 25,
        Payload::CancelOrder { .. } => 8,
        Payload::LevelDelta { .. } => 17,
        Payload::Trade { .. } => 17,
        Payload::Fill { .. } => 32,
        Payload::Heartbeat => 0,
    }
}

#[allow(unused_assignments)] // macro advances the cursor then returns
fn encode_payload(p: &Payload, mut buf: &mut [u8]) -> Result<(), String> {
    macro_rules! w {
        ($n:expr) => {{
            if buf.len() < $n {
                return Err("encode buffer exhausted".into());
            }
            let (head, rest) = buf.split_at_mut($n);
            buf = rest;
            head
        }};
    }
    match p {
        Payload::Snapshot { bids, asks } => {
            if bids.len() + asks.len() > u32::MAX as usize {
                return Err("too many levels".into());
            }
            w!(4).copy_from_slice(&(bids.len() as u32).to_le_bytes());
            w!(4).copy_from_slice(&(asks.len() as u32).to_le_bytes());
            for &(price, lots) in bids.iter().chain(asks.iter()) {
                w!(8).copy_from_slice(&price.to_le_bytes());
                w!(8).copy_from_slice(&lots.to_le_bytes());
            }
        }
        Payload::NewOrder {
            order_id,
            side,
            price_ticks,
            lots,
        } => {
            w!(8).copy_from_slice(&order_id.to_le_bytes());
            w!(1)[0] = side_to_u8(*side);
            w!(8).copy_from_slice(&price_ticks.to_le_bytes());
            w!(8).copy_from_slice(&lots.to_le_bytes());
        }
        Payload::CancelOrder { order_id } => {
            w!(8).copy_from_slice(&order_id.to_le_bytes());
        }
        Payload::LevelDelta {
            side,
            price_ticks,
            delta_lots,
        } => {
            w!(1)[0] = side_to_u8(*side);
            w!(8).copy_from_slice(&price_ticks.to_le_bytes());
            w!(8).copy_from_slice(&delta_lots.to_le_bytes());
        }
        Payload::Trade {
            price_ticks,
            lots,
            aggressor,
        } => {
            w!(8).copy_from_slice(&price_ticks.to_le_bytes());
            w!(8).copy_from_slice(&lots.to_le_bytes());
            w!(1)[0] = side_to_u8(*aggressor);
        }
        Payload::Fill {
            order_id,
            filled_lots,
            remaining_lots,
            price_ticks,
        } => {
            w!(8).copy_from_slice(&order_id.to_le_bytes());
            w!(8).copy_from_slice(&filled_lots.to_le_bytes());
            w!(8).copy_from_slice(&remaining_lots.to_le_bytes());
            w!(8).copy_from_slice(&price_ticks.to_le_bytes());
        }
        Payload::Heartbeat => {}
    }
    Ok(())
}

/// Decode one frame from `buf` (must contain at least a full frame).
#[allow(unused_assignments)] // the read macro advances the cursor then returns
pub fn decode_frame(buf: &[u8]) -> Result<(Frame, usize), String> {
    if buf.len() < HEADER_LEN {
        return Err("short header".into());
    }
    let magic = u16::from_le_bytes([buf[0], buf[1]]);
    if magic != CODEC_MAGIC {
        return Err(format!("bad magic {magic:#x}"));
    }
    if buf[2] != CODEC_VERSION {
        return Err(format!("bad version {}", buf[2]));
    }
    let msg_type = MsgType::from_u8(buf[3]).ok_or_else(|| format!("bad msg type {}", buf[3]))?;
    let seq = u64::from_le_bytes(buf[4..12].try_into().unwrap());
    let ts_ns = i64::from_le_bytes(buf[12..20].try_into().unwrap());
    let plen = u16::from_le_bytes([buf[20], buf[21]]) as usize;
    if buf.len() < HEADER_LEN + plen {
        return Err("truncated payload".into());
    }
    let mut p = &buf[HEADER_LEN..HEADER_LEN + plen];
    macro_rules! r {
        ($n:expr) => {{
            if p.len() < $n {
                return Err("payload underrun".into());
            }
            let (head, rest) = p.split_at($n);
            p = rest;
            head
        }};
    }
    macro_rules! u64r {
        () => {{
            u64::from_le_bytes(r!(8).try_into().unwrap())
        }};
    }
    let payload = match msg_type {
        MsgType::Snapshot => {
            let nb = u32::from_le_bytes(r!(4).try_into().unwrap()) as usize;
            let na = u32::from_le_bytes(r!(4).try_into().unwrap()) as usize;
            if p.len() < (nb + na) * 16 {
                return Err("snapshot levels truncated".into());
            }
            let read_level = |p: &mut &[u8]| -> (u64, u64) {
                let price = u64::from_le_bytes(p[0..8].try_into().unwrap());
                let lots = u64::from_le_bytes(p[8..16].try_into().unwrap());
                *p = &p[16..];
                (price, lots)
            };
            let mut rest = p;
            let mut bids = Vec::with_capacity(nb);
            for _ in 0..nb {
                bids.push(read_level(&mut rest));
            }
            let mut asks = Vec::with_capacity(na);
            for _ in 0..na {
                asks.push(read_level(&mut rest));
            }
            Payload::Snapshot { bids, asks }
        }
        MsgType::NewOrder => Payload::NewOrder {
            order_id: u64r!(),
            side: side_from_u8(r!(1)[0]).ok_or("bad side")?,
            price_ticks: u64r!(),
            lots: u64r!(),
        },
        MsgType::CancelOrder => Payload::CancelOrder {
            order_id: u64r!(),
        },
        MsgType::LevelDelta => Payload::LevelDelta {
            side: side_from_u8(r!(1)[0]).ok_or("bad side")?,
            price_ticks: u64r!(),
            delta_lots: i64::from_le_bytes(r!(8).try_into().unwrap()),
        },
        MsgType::Trade => Payload::Trade {
            price_ticks: u64r!(),
            lots: u64r!(),
            aggressor: side_from_u8(r!(1)[0]).ok_or("bad side")?,
        },
        MsgType::Fill => Payload::Fill {
            order_id: u64r!(),
            filled_lots: u64r!(),
            remaining_lots: u64r!(),
            price_ticks: u64r!(),
        },
        MsgType::Heartbeat => Payload::Heartbeat,
    };
    Ok((
        Frame {
            msg_type,
            seq,
            ts_ns,
            payload,
        },
        HEADER_LEN + plen,
    ))
}

/// Decode a batch of concatenated frames.
pub fn decode_batch(buf: &[u8]) -> Result<Vec<Frame>, String> {
    let mut out = Vec::new();
    let mut off = 0;
    while off < buf.len() {
        let (f, n) = decode_frame(&buf[off..])?;
        out.push(f);
        off += n;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(f: Frame) -> Frame {
        let mut buf = [0u8; 4096];
        let n = encode_frame(&f, &mut buf).unwrap();
        let (g, m) = decode_frame(&buf[..n]).unwrap();
        assert_eq!(n, m);
        g
    }

    #[test]
    fn roundtrips() {
        let cases = vec![
            Frame {
                msg_type: MsgType::NewOrder,
                seq: 42,
                ts_ns: 1_700_000_000_000_000_000,
                payload: Payload::NewOrder {
                    order_id: 7,
                    side: Side::Bid,
                    price_ticks: 99,
                    lots: 12,
                },
            },
            Frame {
                msg_type: MsgType::LevelDelta,
                seq: 43,
                ts_ns: 2,
                payload: Payload::LevelDelta {
                    side: Side::Ask,
                    price_ticks: 101,
                    delta_lots: -5,
                },
            },
            Frame {
                msg_type: MsgType::Snapshot,
                seq: 1,
                ts_ns: 0,
                payload: Payload::Snapshot {
                    bids: vec![(99, 10), (98, 20)],
                    asks: vec![(101, 5)],
                },
            },
            Frame {
                msg_type: MsgType::Trade,
                seq: 44,
                ts_ns: 3,
                payload: Payload::Trade {
                    price_ticks: 100,
                    lots: 3,
                    aggressor: Side::Ask,
                },
            },
            Frame {
                msg_type: MsgType::Fill,
                seq: 45,
                ts_ns: 4,
                payload: Payload::Fill {
                    order_id: 7,
                    filled_lots: 2,
                    remaining_lots: 10,
                    price_ticks: 99,
                },
            },
            Frame {
                msg_type: MsgType::Heartbeat,
                seq: 46,
                ts_ns: 5,
                payload: Payload::Heartbeat,
            },
        ];
        for f in cases {
            let g = roundtrip(f.clone());
            assert_eq!(f, g);
        }
    }

    #[test]
    fn rejects_garbage() {
        let mut rng = micro_like_rng(7);
        for _ in 0..2000 {
            let n = 40;
            let mut buf = vec![0u8; n];
            for b in buf.iter_mut() {
                *b = (rng.next() % 256) as u8;
            }
            // nearly all garbage must be rejected
            if decode_frame(&buf).is_ok() {
                // only valid if magic/version/type/length all line up
                let ok_magic = u16::from_le_bytes([buf[0], buf[1]]) == CODEC_MAGIC;
                assert!(ok_magic, "accepted non-magic garbage");
            }
        }
    }

    #[test]
    fn batch_decode() {
        let frames = vec![
            Frame {
                msg_type: MsgType::Heartbeat,
                seq: 1,
                ts_ns: 0,
                payload: Payload::Heartbeat,
            },
            Frame {
                msg_type: MsgType::LevelDelta,
                seq: 2,
                ts_ns: 1,
                payload: Payload::LevelDelta {
                    side: Side::Bid,
                    price_ticks: 98,
                    delta_lots: 7,
                },
            },
        ];
        let mut buf = Vec::new();
        let mut tmp = [0u8; 512];
        for f in &frames {
            let n = encode_frame(f, &mut tmp).unwrap();
            buf.extend_from_slice(&tmp[..n]);
        }
        let got = decode_batch(&buf).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[1].seq, 2);
    }

    // tiny deterministic rng for the garbage test (no deps)
    struct R(u64);
    fn micro_like_rng(seed: u64) -> R {
        R(seed | 1)
    }
    impl R {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }
    }
}
