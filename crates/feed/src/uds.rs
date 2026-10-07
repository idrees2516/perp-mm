//! Unix-domain-socket gateway transport with length-prefixed framing.
//!
//! The daemon side ([`UdsGateway`]) fans snapshots out to all connected
//! frontends (non-blocking writes; a stalled client is dropped rather
//! than allowed to slow the engine) and collects command frames. The
//! client side ([`UdsClient`]) connects and exchanges framed messages
//! with a poll-friendly, non-blocking API. Zero external dependencies:
//! `std::os::unix::net` + `std::io`.

use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::time::Duration;

use crate::proto::{read_framed, write_framed};

/// Server side: bound listener + connected clients.
pub struct UdsGateway {
    listener: UnixListener,
    clients: Vec<UnixStream>,
    /// Connected client count served so far.
    pub total_clients: u64,
    /// Snapshots dropped on stalled clients.
    pub drops: u64,
}

impl UdsGateway {
    /// Bind `path` (removing a stale socket file first).
    pub fn bind<P: AsRef<Path>>(path: P) -> std::io::Result<UdsGateway> {
        let path = path.as_ref();
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        let listener = UnixListener::bind(path)?;
        listener.set_nonblocking(true)?;
        Ok(UdsGateway {
            listener,
            clients: Vec::new(),
            total_clients: 0,
            drops: 0,
        })
    }

    /// Number of connected clients.
    pub fn client_count(&self) -> usize {
        self.clients.len()
    }

    /// Accept pending connections (non-blocking).
    pub fn accept(&mut self) -> usize {
        let mut n = 0;
        while let Ok((stream, _)) = self.listener.accept() {
            let _ = stream.set_nonblocking(true);
            let _ = stream.set_write_timeout(Some(Duration::from_millis(5)));
            self.clients.push(stream);
            self.total_clients += 1;
            n += 1;
        }
        n
    }

    /// Broadcast one framed payload to all clients. Stalled/disconnected
    /// clients are dropped (counted in `drops`).
    pub fn broadcast(&mut self, payload: &[u8]) {
        let mut dead: Vec<usize> = Vec::new();
        for (i, c) in self.clients.iter_mut().enumerate() {
            let mut framed = Vec::with_capacity(4 + payload.len());
            write_framed(&mut framed, payload);
            match c.write_all(&framed) {
                Ok(_) => {}
                Err(e) => {
                    match e.kind() {
                        std::io::ErrorKind::WouldBlock => {
                            // stalled: drop this snapshot (and the client)
                            self.drops += 1;
                            dead.push(i);
                        }
                        _ => {
                            self.drops += 1;
                            dead.push(i);
                        }
                    }
                }
            }
        }
        for &i in dead.iter().rev() {
            self.clients.remove(i);
        }
    }

    /// Read complete command frames from all clients.
    pub fn poll_commands(&mut self, scratch: &mut Vec<u8>) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        let mut dead: Vec<usize> = Vec::new();
        let mut bufs: Vec<Vec<u8>> = self.clients.iter_mut().map(|_| Vec::new()).collect();
        let _ = scratch;
        for (i, c) in self.clients.iter_mut().enumerate() {
            let mut tmp = [0u8; 4096];
            loop {
                match c.read(&mut tmp) {
                    Ok(0) => {
                        dead.push(i);
                        break;
                    }
                    Ok(n) => {
                        bufs[i].extend_from_slice(&tmp[..n]);
                        if n < tmp.len() {
                            break;
                        }
                    }
                    Err(e) => match e.kind() {
                        std::io::ErrorKind::WouldBlock => break,
                        _ => {
                            dead.push(i);
                            break;
                        }
                    },
                }
            }
        }
        for b in bufs.iter_mut() {
            out.extend(read_framed(b));
        }
        for &i in dead.iter().rev() {
            self.clients.remove(i);
        }
        out
    }
}

/// Client side.
pub struct UdsClient {
    stream: UnixStream,
    rbuf: Vec<u8>,
}

impl UdsClient {
    pub fn connect<P: AsRef<Path>>(path: P) -> std::io::Result<UdsClient> {
        let stream = UnixStream::connect(path)?;
        stream.set_nonblocking(true)?;
        stream.set_read_timeout(Some(Duration::from_millis(50)))?;
        stream.set_write_timeout(Some(Duration::from_millis(200)))?;
        Ok(UdsClient { stream, rbuf: Vec::with_capacity(4096) })
    }

    /// Raw fd of the underlying stream (for poll()-based event loops).
    pub fn fd(&self) -> i32 {
        use std::os::unix::io::AsRawFd;
        self.stream.as_raw_fd()
    }

    /// Send one framed payload (blocking-ish with a write timeout).
    pub fn send(&mut self, payload: &[u8]) -> std::io::Result<()> {
        let mut framed = Vec::with_capacity(4 + payload.len());
        write_framed(&mut framed, payload);
        self.stream.write_all(&framed)
    }

    /// Receive complete frames (non-blocking-ish; returns empty when
    /// nothing is ready).
    pub fn recv(&mut self) -> Vec<Vec<u8>> {
        let mut tmp = [0u8; 8192];
        loop {
            match self.stream.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => {
                    self.rbuf.extend_from_slice(&tmp[..n]);
                    if n < tmp.len() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        read_framed(&mut self.rbuf)
    }

    /// Receive with a blocking wait up to `timeout`.
    pub fn recv_timeout(&mut self, timeout: Duration) -> Vec<Vec<u8>> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let frames = self.recv();
            if !frames.is_empty() {
                return frames;
            }
            if std::time::Instant::now() >= deadline {
                return Vec::new();
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::{encode_msg, GwAck, GwCommand, GwMsg};

    fn tmp_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("perp-mm-uds-{}-{}.sock", tag, std::process::id()));
        p
    }

    #[test]
    fn gateway_client_roundtrip() {
        let path = tmp_path("rt");
        let mut gw = UdsGateway::bind(&path).expect("bind");
        let mut cl = UdsClient::connect(&path).expect("connect");
        // the client connects asynchronously; accept may need a retry
        for _ in 0..50 {
            if gw.accept() > 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(gw.client_count(), 1);
        // client -> server command
        cl.send(&encode_msg(&GwMsg::Command(GwCommand::Ping))).unwrap();
        std::thread::sleep(Duration::from_millis(10));
        let mut scratch = Vec::new();
        let cmds = gw.poll_commands(&mut scratch);
        assert_eq!(cmds.len(), 1);
        assert_eq!(
            crate::proto::decode_msg(&cmds[0]),
            Ok(GwMsg::Command(GwCommand::Ping))
        );
        // server -> client snapshot-ish ack
        gw.broadcast(&encode_msg(&GwMsg::Ack(GwAck::Pong)));
        let frames = cl.recv_timeout(Duration::from_millis(500));
        assert_eq!(frames.len(), 1);
        assert_eq!(crate::proto::decode_msg(&frames[0]), Ok(GwMsg::Ack(GwAck::Pong)));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn dropped_clients_are_reaped() {
        let path = tmp_path("drop");
        let mut gw = UdsGateway::bind(&path).expect("bind");
        {
            let mut cl = UdsClient::connect(&path).unwrap();
            for _ in 0..50 {
                if gw.accept() > 0 {
                    break;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            assert_eq!(gw.client_count(), 1);
            gw.broadcast(&encode_msg(&GwMsg::Ack(GwAck::Ok)));
            let frames = cl.recv_timeout(Duration::from_millis(300));
            assert_eq!(frames.len(), 1);
        } // client dropped here
        std::thread::sleep(Duration::from_millis(5));
        let mut scratch = Vec::new();
        let _ = gw.poll_commands(&mut scratch);
        // after the disconnect is noticed, broadcasting reaps it
        gw.broadcast(&encode_msg(&GwMsg::Ack(GwAck::Ok)));
        assert_eq!(gw.client_count(), 0);
        let _ = std::fs::remove_file(&path);
    }
}
