//! Transports: the byte-source abstraction behind the feed handler.

use std::io;
use std::net::UdpSocket;

/// A source of raw datagram/frames.
pub trait Transport {
    /// Receive up to `buf.len()` bytes; `Ok(0)` means EOF/closed.
    fn recv(&mut self, buf: &mut [u8]) -> io::Result<usize>;
}

/// Blocking UDP transport (multicast or unicast), with optional
/// read timeout.
pub struct UdpTransport {
    sock: UdpSocket,
}

impl UdpTransport {
    pub fn bind(addr: &str, timeout_ms: Option<u64>) -> io::Result<UdpTransport> {
        let sock = UdpSocket::bind(addr)?;
        if let Some(ms) = timeout_ms {
            sock.set_read_timeout(Some(std::time::Duration::from_millis(ms)))?;
        }
        Ok(UdpTransport { sock })
    }

    pub fn join_multicast(&self, group: &str, iface: &str) -> io::Result<()> {
        use std::net::Ipv4Addr;
        let g: Ipv4Addr = group.parse().map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "bad multicast group")
        })?;
        let i: Ipv4Addr = iface.parse().unwrap_or(Ipv4Addr::UNSPECIFIED);
        self.sock
            .join_multicast_v4(&g, &i)
    }
}

impl Transport for UdpTransport {
    fn recv(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.sock.recv(buf)
    }
}

/// In-process channel transport: the venue simulator pushes encoded
/// frames; the handler pulls them.
pub struct ChanTransport {
    rx: std::sync::mpsc::Receiver<Vec<u8>>,
}

impl ChanTransport {
    pub fn new(rx: std::sync::mpsc::Receiver<Vec<u8>>) -> ChanTransport {
        ChanTransport { rx }
    }
}

impl Transport for ChanTransport {
    fn recv(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self.rx.recv() {
            Ok(data) => {
                let n = data.len().min(buf.len());
                buf[..n].copy_from_slice(&data[..n]);
                Ok(n)
            }
            Err(_) => Ok(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn channel_transport_roundtrip() {
        let (tx, rx) = mpsc::channel();
        let mut t = ChanTransport::new(rx);
        tx.send(vec![1, 2, 3, 4, 5]).unwrap();
        let mut buf = [0u8; 64];
        assert_eq!(t.recv(&mut buf).unwrap(), 5);
        assert_eq!(&buf[..5], &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn udp_loopback() {
        // bind an ephemeral port and send to ourselves
        let sender = UdpSocket::bind("127.0.0.1:0").unwrap();
        let addr = sender.local_addr().unwrap();
        let mut t = UdpTransport::bind("127.0.0.1:0", Some(500)).unwrap();
        let t_addr = t.sock.local_addr().unwrap();
        let _ = t_addr;
        // send from sender to transport's socket
        let probe = UdpSocket::bind("127.0.0.1:0").unwrap();
        let _ = addr;
        probe.send_to(b"hello", t.sock.local_addr().unwrap()).unwrap();
        let mut buf = [0u8; 64];
        assert_eq!(t.recv(&mut buf).unwrap(), 5);
        assert_eq!(&buf[..5], b"hello");
    }
}
