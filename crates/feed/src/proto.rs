//! Gateway protocol: compact binary snapshots and commands between the
//! engine daemon and frontends (the TUI). Little-endian, tag-prefixed
//! frames; vectors are length-prefixed with a `u8` and bounded. Every
//! message round-trips through [`encode_msg`]/[`decode_msg`] and
//! malformed input is rejected.

use ob::Side;

pub const GW_MAGIC: u16 = 0xC0DE;
pub const GW_VERSION: u8 = 2;
/// Maximum book levels / quote rows per snapshot.
pub const MAX_LEVELS: usize = 10;
/// Maximum option legs per snapshot.
pub const MAX_LEGS: usize = 8;

// ---------------------------------------------------------------------------
// Message model
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct PnlParts {
    pub spread_capture: f64,
    pub adverse_cost: f64,
    pub fees: f64,
    pub funding: f64,
    pub hedge_cost: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OptionSnapshot {
    pub atm_iv: f64,
    pub net_delta_lots: f64,
    pub net_vega: f64,
    pub net_gamma: f64,
    pub mark: f64,
    pub fills: u64,
    /// Per-leg IV quotes (bid, ask) aligned with the venue's moneyness
    /// ladder.
    pub leg_quotes: Vec<(f64, f64)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GwSnapshot {
    pub seq: u64,
    pub step: u64,
    pub sim_time: f64,
    pub mid: f64,
    pub micro_price: f64,
    pub spread: f64,
    pub sigma_fast: f64,
    pub sigma_slow: f64,
    pub sigma_rough: f64,
    pub hurst: f64,
    pub imbalance: f64,
    pub ofi: f64,
    pub jump_flag: bool,
    /// Book bids best-first `(price_ticks, lots)`.
    pub book_bids: Vec<(u64, u64)>,
    /// Book asks best-first.
    pub book_asks: Vec<(u64, u64)>,
    /// Our live orders `(side, price_ticks, lots)`.
    pub our_orders: Vec<(u8, u64, u64)>,
    pub inventory: i64,
    pub equity: f64,
    pub pnl: PnlParts,
    pub markout_mult: f64,
    pub halted: bool,
    pub paused: bool,
    /// Strategy index (see `strategy::StrategyKind` ordering in the
    /// engine; labels ride the snapshot as a tag).
    pub strategy: u8,
    pub option: Option<OptionSnapshot>,
    /// Engine step latency (nanoseconds).
    pub step_ns_p50: u64,
    pub step_ns_p99: u64,
    /// Gateway drops (snapshots dropped when a client stalls).
    pub drops: u64,
}

/// Client -> daemon commands.
#[derive(Clone, Debug, PartialEq)]
pub enum GwCommand {
    Ping,
    SelectStrategy(u8),
    /// Tunable parameter slots (gamma, intensity A, kappa, max inventory,
    /// levels, markout theta, vol-of-vol).
    SetParam(u8, f64),
    PauseQuotes(bool),
    KillSwitch(bool),
    ManualPlace { side: Side, price_ticks: u64, lots: u64, post_only: bool },
    ManualTake { side: Side, lots: u64 },
    ManualCancel { order_id: u64 },
    CancelAll,
    ResetSession,
}

/// Daemon -> client replies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GwAck {
    Pong,
    Ok,
    UnknownCommand,
    BadParam,
}

#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::large_enum_variant)] // Snapshot dominates; commands are tiny and rare
pub enum GwMsg {
    Snapshot(GwSnapshot),
    /// A public trade print (for the tape view).
    Trade { price_ticks: u64, lots: u64, aggressor: u8 },
    Ack(GwAck),
    Command(GwCommand),
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

struct Enc {
    buf: Vec<u8>,
}

impl Enc {
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn levels(&mut self, lv: &[(u64, u64)], cap: usize) {
        let n = lv.len().min(cap);
        self.u8(n as u8);
        for &(p, l) in lv.iter().take(n) {
            self.u64(p);
            self.u64(l);
        }
    }
    fn orders(&mut self, lv: &[(u8, u64, u64)]) {
        let n = lv.len().min(2 * MAX_LEVELS);
        self.u8(n as u8);
        for &(s, p, l) in lv.iter().take(n) {
            self.u8(s);
            self.u64(p);
            self.u64(l);
        }
    }
}

struct Dec<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Dec<'a> {
    fn need(&self, n: usize) -> Result<(), String> {
        if self.pos + n > self.buf.len() {
            Err("truncated frame".into())
        } else {
            Ok(())
        }
    }
    fn u8(&mut self) -> Result<u8, String> {
        self.need(1)?;
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }
    fn u16(&mut self) -> Result<u16, String> {
        self.need(2)?;
        let v = u16::from_le_bytes([self.buf[self.pos], self.buf[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }
    fn u32(&mut self) -> Result<u32, String> {
        self.need(4)?;
        let mut b = [0u8; 4];
        b.copy_from_slice(&self.buf[self.pos..self.pos + 4]);
        self.pos += 4;
        Ok(u32::from_le_bytes(b))
    }
    fn u64(&mut self) -> Result<u64, String> {
        self.need(8)?;
        let mut b = [0u8; 8];
        b.copy_from_slice(&self.buf[self.pos..self.pos + 8]);
        self.pos += 8;
        Ok(u64::from_le_bytes(b))
    }
    fn i64(&mut self) -> Result<i64, String> {
        Ok(self.u64()? as i64)
    }
    fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_bits(self.u64()?))
    }
    fn levels(&mut self, cap: usize) -> Result<Vec<(u64, u64)>, String> {
        let n = self.u8()? as usize;
        if n > cap {
            return Err("level count over cap".into());
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let p = self.u64()?;
            let l = self.u64()?;
            out.push((p, l));
        }
        Ok(out)
    }
}

/// Encode one message (no framing).
pub fn encode_msg(msg: &GwMsg) -> Vec<u8> {
    let mut e = Enc { buf: Vec::with_capacity(256) };
    e.u16(GW_MAGIC);
    e.u8(GW_VERSION);
    match msg {
        GwMsg::Snapshot(s) => {
            e.u8(1);
            e.u64(s.seq);
            e.u64(s.step);
            e.f64(s.sim_time);
            e.f64(s.mid);
            e.f64(s.micro_price);
            e.f64(s.spread);
            e.f64(s.sigma_fast);
            e.f64(s.sigma_slow);
            e.f64(s.sigma_rough);
            e.f64(s.hurst);
            e.f64(s.imbalance);
            e.f64(s.ofi);
            e.u8(s.jump_flag as u8);
            e.levels(&s.book_bids, MAX_LEVELS);
            e.levels(&s.book_asks, MAX_LEVELS);
            e.orders(&s.our_orders);
            e.i64(s.inventory);
            e.f64(s.equity);
            e.f64(s.pnl.spread_capture);
            e.f64(s.pnl.adverse_cost);
            e.f64(s.pnl.fees);
            e.f64(s.pnl.funding);
            e.f64(s.pnl.hedge_cost);
            e.f64(s.markout_mult);
            e.u8(s.halted as u8);
            e.u8(s.paused as u8);
            e.u8(s.strategy);
            if let Some(o) = &s.option {
                e.u8(1);
                e.f64(o.atm_iv);
                e.f64(o.net_delta_lots);
                e.f64(o.net_vega);
                e.f64(o.net_gamma);
                e.f64(o.mark);
                e.u32(o.fills as u32);
                let n = o.leg_quotes.len().min(MAX_LEGS);
                e.u8(n as u8);
                for &(b, a) in o.leg_quotes.iter().take(n) {
                    e.f64(b);
                    e.f64(a);
                }
            } else {
                e.u8(0);
            }
            e.u64(s.step_ns_p50);
            e.u64(s.step_ns_p99);
            e.u64(s.drops);
        }
        GwMsg::Trade { price_ticks, lots, aggressor } => {
            e.u8(2);
            e.u64(*price_ticks);
            e.u64(*lots);
            e.u8(*aggressor);
        }
        GwMsg::Ack(a) => {
            e.u8(3);
            e.u8(match a {
                GwAck::Pong => 1,
                GwAck::Ok => 2,
                GwAck::UnknownCommand => 3,
                GwAck::BadParam => 4,
            });
        }
        GwMsg::Command(c) => {
            e.u8(4);
            match c {
                GwCommand::Ping => e.u8(1),
                GwCommand::SelectStrategy(k) => {
                    e.u8(2);
                    e.u8(*k);
                }
                GwCommand::SetParam(id, v) => {
                    e.u8(3);
                    e.u8(*id);
                    e.f64(*v);
                }
                GwCommand::PauseQuotes(p) => {
                    e.u8(4);
                    e.u8(*p as u8);
                }
                GwCommand::KillSwitch(k) => {
                    e.u8(5);
                    e.u8(*k as u8);
                }
                GwCommand::ManualPlace { side, price_ticks, lots, post_only } => {
                    e.u8(6);
                    e.u8(side_byte(*side));
                    e.u64(*price_ticks);
                    e.u64(*lots);
                    e.u8(*post_only as u8);
                }
                GwCommand::ManualTake { side, lots } => {
                    e.u8(7);
                    e.u8(side_byte(*side));
                    e.u64(*lots);
                }
                GwCommand::ManualCancel { order_id } => {
                    e.u8(8);
                    e.u64(*order_id);
                }
                GwCommand::CancelAll => e.u8(9),
                GwCommand::ResetSession => e.u8(10),
            }
        }
    }
    e.buf
}

fn side_byte(s: Side) -> u8 {
    match s {
        Side::Bid => 1,
        Side::Ask => 2,
    }
}

fn side_from(b: u8) -> Result<Side, String> {
    match b {
        1 => Ok(Side::Bid),
        2 => Ok(Side::Ask),
        _ => Err("bad side byte".into()),
    }
}

/// Decode one message (no framing). Errors on malformed input.
pub fn decode_msg(buf: &[u8]) -> Result<GwMsg, String> {
    let mut d = Dec { buf, pos: 0 };
    if d.u16()? != GW_MAGIC {
        return Err("bad magic".into());
    }
    if d.u8()? != GW_VERSION {
        return Err("bad version".into());
    }
    let tag = d.u8()?;
    let msg = match tag {
        1 => {
            let seq = d.u64()?;
            let step = d.u64()?;
            let sim_time = d.f64()?;
            let mid = d.f64()?;
            let micro_price = d.f64()?;
            let spread = d.f64()?;
            let sigma_fast = d.f64()?;
            let sigma_slow = d.f64()?;
            let sigma_rough = d.f64()?;
            let hurst = d.f64()?;
            let imbalance = d.f64()?;
            let ofi = d.f64()?;
            let jump_flag = d.u8()? != 0;
            let book_bids = d.levels(MAX_LEVELS)?;
            let book_asks = d.levels(MAX_LEVELS)?;
            let n_orders = d.u8()? as usize;
            if n_orders > 2 * MAX_LEVELS {
                return Err("order count over cap".into());
            }
            let mut our_orders = Vec::with_capacity(n_orders);
            for _ in 0..n_orders {
                let s = d.u8()?;
                let p = d.u64()?;
                let l = d.u64()?;
                our_orders.push((s, p, l));
            }
            let inventory = d.i64()?;
            let equity = d.f64()?;
            let spread_capture = d.f64()?;
            let adverse_cost = d.f64()?;
            let fees = d.f64()?;
            let funding = d.f64()?;
            let hedge_cost = d.f64()?;
            let markout_mult = d.f64()?;
            let halted = d.u8()? != 0;
            let paused = d.u8()? != 0;
            let strategy = d.u8()?;
            let option = if d.u8()? != 0 {
                let atm_iv = d.f64()?;
                let net_delta_lots = d.f64()?;
                let net_vega = d.f64()?;
                let net_gamma = d.f64()?;
                let mark = d.f64()?;
                let fills = d.u32()? as u64;
                let n = d.u8()? as usize;
                if n > MAX_LEGS {
                    return Err("leg count over cap".into());
                }
                let mut leg_quotes = Vec::with_capacity(n);
                for _ in 0..n {
                    let b = d.f64()?;
                    let a = d.f64()?;
                    leg_quotes.push((b, a));
                }
                Some(OptionSnapshot {
                    atm_iv,
                    net_delta_lots,
                    net_vega,
                    net_gamma,
                    mark,
                    fills,
                    leg_quotes,
                })
            } else {
                None
            };
            let step_ns_p50 = d.u64()?;
            let step_ns_p99 = d.u64()?;
            let drops = d.u64()?;
            GwMsg::Snapshot(GwSnapshot {
                seq,
                step,
                sim_time,
                mid,
                micro_price,
                spread,
                sigma_fast,
                sigma_slow,
                sigma_rough,
                hurst,
                imbalance,
                ofi,
                jump_flag,
                book_bids,
                book_asks,
                our_orders,
                inventory,
                equity,
                pnl: PnlParts {
                    spread_capture,
                    adverse_cost,
                    fees,
                    funding,
                    hedge_cost,
                },
                markout_mult,
                halted,
                paused,
                strategy,
                option,
                step_ns_p50,
                step_ns_p99,
                drops,
            })
        }
        2 => {
            let price_ticks = d.u64()?;
            let lots = d.u64()?;
            let aggressor = d.u8()?;
            GwMsg::Trade { price_ticks, lots, aggressor }
        }
        3 => {
            let code = d.u8()?;
            GwMsg::Ack(match code {
                1 => GwAck::Pong,
                2 => GwAck::Ok,
                3 => GwAck::UnknownCommand,
                4 => GwAck::BadParam,
                _ => return Err("bad ack code".into()),
            })
        }
        4 => {
            let code = d.u8()?;
            GwMsg::Command(match code {
                1 => GwCommand::Ping,
                2 => GwCommand::SelectStrategy(d.u8()?),
                3 => {
                    let id = d.u8()?;
                    let v = d.f64()?;
                    GwCommand::SetParam(id, v)
                }
                4 => GwCommand::PauseQuotes(d.u8()? != 0),
                5 => GwCommand::KillSwitch(d.u8()? != 0),
                6 => {
                    let side = side_from(d.u8()?)?;
                    let price_ticks = d.u64()?;
                    let lots = d.u64()?;
                    let post_only = d.u8()? != 0;
                    GwCommand::ManualPlace { side, price_ticks, lots, post_only }
                }
                7 => {
                    let side = side_from(d.u8()?)?;
                    let lots = d.u64()?;
                    GwCommand::ManualTake { side, lots }
                }
                8 => GwCommand::ManualCancel { order_id: d.u64()? },
                9 => GwCommand::CancelAll,
                10 => GwCommand::ResetSession,
                _ => return Err("bad command code".into()),
            })
        }
        _ => return Err("bad tag".into()),
    };
    if d.pos != buf.len() {
        return Err("trailing bytes".into());
    }
    Ok(msg)
}

// ---------------------------------------------------------------------------
// Stream framing: 4-byte little-endian length prefix + payload.
// ---------------------------------------------------------------------------

/// Frame a payload.
pub fn frame(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + payload.len());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Append one framed payload to a byte stream.
pub fn write_framed(stream: &mut Vec<u8>, payload: &[u8]) {
    stream.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    stream.extend_from_slice(payload);
}

/// Incremental frame parser: pulls zero or more complete frames off the
/// head of `buf`, returning the decoded payloads and consuming their
/// bytes.
pub fn read_framed(buf: &mut Vec<u8>) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        if buf.len() < 4 {
            break;
        }
        let mut lb = [0u8; 4];
        lb.copy_from_slice(&buf[..4]);
        let len = u32::from_le_bytes(lb) as usize;
        if len == 0 || len > 1 << 20 {
            // corrupt stream: drop everything
            buf.clear();
            break;
        }
        if buf.len() < 4 + len {
            break;
        }
        let payload = buf[4..4 + len].to_vec();
        buf.drain(..4 + len);
        out.push(payload);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot() -> GwSnapshot {
        GwSnapshot {
            seq: 42,
            step: 1234,
            sim_time: 61.7,
            mid: 100.25,
            micro_price: 100.31,
            spread: 0.5,
            sigma_fast: 0.021,
            sigma_slow: 0.019,
            sigma_rough: 0.02,
            hurst: 0.13,
            imbalance: 0.55,
            ofi: 12.0,
            jump_flag: false,
            book_bids: vec![(200, 40), (199, 12)],
            book_asks: vec![(201, 35), (202, 8)],
            our_orders: vec![(1, 200, 5), (2, 201, 5), (1, 199, 4)],
            inventory: -3,
            equity: 128.5,
            pnl: PnlParts {
                spread_capture: 30.0,
                adverse_cost: 10.0,
                fees: 2.0,
                funding: 0.5,
                hedge_cost: 1.2,
            },
            markout_mult: 1.4,
            halted: false,
            paused: false,
            strategy: 6,
            option: Some(OptionSnapshot {
                atm_iv: 0.32,
                net_delta_lots: 2.5,
                net_vega: 800.0,
                net_gamma: 0.01,
                mark: 12.0,
                fills: 34,
                leg_quotes: vec![(0.30, 0.34), (0.31, 0.33)],
            }),
            step_ns_p50: 780,
            step_ns_p99: 2100,
            drops: 0,
        }
    }

    #[test]
    fn snapshot_roundtrip() {
        let s = sample_snapshot();
        let bytes = encode_msg(&GwMsg::Snapshot(s.clone()));
        match decode_msg(&bytes) {
            Ok(GwMsg::Snapshot(g)) => {
                assert_eq!(g.seq, s.seq);
                assert_eq!(g.step, s.step);
                assert!((g.mid - s.mid).abs() < 1e-12);
                assert_eq!(g.book_bids, s.book_bids);
                assert_eq!(g.our_orders.len(), 3);
                assert_eq!(g.inventory, s.inventory);
                assert!((g.pnl.hedge_cost - s.pnl.hedge_cost).abs() < 1e-12);
                let o = g.option.unwrap();
                assert!((o.atm_iv - 0.32).abs() < 1e-12);
                assert_eq!(o.leg_quotes.len(), 2);
                assert_eq!(g.step_ns_p99, 2100);
            }
            other => panic!("bad decode: {:?}", other.err()),
        }
    }

    #[test]
    fn commands_roundtrip() {
        let cmds = vec![
            GwCommand::Ping,
            GwCommand::SelectStrategy(3),
            GwCommand::SetParam(1, 0.25),
            GwCommand::PauseQuotes(true),
            GwCommand::KillSwitch(false),
            GwCommand::ManualPlace { side: Side::Bid, price_ticks: 199, lots: 4, post_only: true },
            GwCommand::ManualTake { side: Side::Ask, lots: 2 },
            GwCommand::ManualCancel { order_id: 12345 },
            GwCommand::CancelAll,
            GwCommand::ResetSession,
        ];
        for c in cmds {
            let bytes = encode_msg(&GwMsg::Command(c.clone()));
            match decode_msg(&bytes) {
                Ok(GwMsg::Command(g)) => assert_eq!(g, c),
                other => panic!("bad decode: {:?}", other.err()),
            }
        }
    }

    #[test]
    fn acks_roundtrip() {
        for a in [GwAck::Pong, GwAck::Ok, GwAck::UnknownCommand, GwAck::BadParam] {
            let bytes = encode_msg(&GwMsg::Ack(a));
            assert_eq!(decode_msg(&bytes), Ok(GwMsg::Ack(a)));
        }
    }

    #[test]
    fn garbage_rejected() {
        assert!(decode_msg(&[]).is_err());
        assert!(decode_msg(&[0x01, 0x02]).is_err());
        assert!(decode_msg(&[0xDE, 0xC0, 2, 1]).is_err());
        let good = encode_msg(&GwMsg::Ack(GwAck::Ok));
        assert!(decode_msg(&good[..good.len() - 1]).is_err());
        let mut trailing = good.clone();
        trailing.push(0);
        assert!(decode_msg(&trailing).is_err());
    }

    #[test]
    fn framing_roundtrip() {
        let mut stream = Vec::new();
        let a = encode_msg(&GwMsg::Ack(GwAck::Ok));
        let b = encode_msg(&GwMsg::Trade { price_ticks: 200, lots: 5, aggressor: 1 });
        write_framed(&mut stream, &a);
        write_framed(&mut stream, &b);
        // partial first read
        let mut buf = stream[..6].to_vec();
        assert!(read_framed(&mut buf).is_empty());
        buf.extend_from_slice(&stream[6..]);
        let frames = read_framed(&mut buf);
        assert_eq!(frames.len(), 2);
        assert_eq!(decode_msg(&frames[0]), Ok(GwMsg::Ack(GwAck::Ok)));
        assert!(buf.is_empty());
        // corrupt length drops the stream
        let mut bad = vec![0xFF, 0xFF, 0xFF, 0xFF, 1, 2];
        assert!(read_framed(&mut bad).is_empty());
        assert!(bad.is_empty());
    }

    #[test]
    fn caps_enforced() {
        let mut s = sample_snapshot();
        s.book_bids = (0..40).map(|i| (200 - i, 10u64)).collect();
        let bytes = encode_msg(&GwMsg::Snapshot(s));
        match decode_msg(&bytes) {
            Ok(GwMsg::Snapshot(g)) => assert_eq!(g.book_bids.len(), MAX_LEVELS),
            _ => panic!(),
        }
    }
}
