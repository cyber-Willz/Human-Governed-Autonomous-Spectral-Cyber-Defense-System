//! Low-interaction multi-port TCP honeypot.
//!
//! One listener per configured decoy port. On connect, optionally sends a
//! banner (e.g. fake FTP/SMTP/Redis banner), then reads whatever the peer
//! sends and records it as a `RawPayload` event with a best-effort
//! protocol guess. This deliberately does not attempt full protocol
//! emulation (that's what the SSH module is for) -- it exists to cover
//! the long tail of scanned ports cheaply, matching the `portlurker`
//! pattern: catch everything, understand what you can, never crash on
//! malformed input.

use crate::config::TcpPortConfig;
use crate::event::{Confidence, EventKind, HoneypotEvent, Severity};
use anyhow::Result;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::mpsc::Sender;
use tokio::time::{timeout, Duration};
use tracing::{info, warn};

const READ_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_READ_BYTES: usize = 4096;

pub async fn run_all(
    ports: Vec<TcpPortConfig>,
    host_id: String,
    tx: Sender<HoneypotEvent>,
) -> Result<()> {
    let mut handles = Vec::new();
    for port_cfg in ports {
        let host_id = host_id.clone();
        let tx = tx.clone();
        handles.push(tokio::spawn(async move {
            if let Err(e) = run_single(port_cfg.clone(), host_id, tx).await {
                warn!(port = port_cfg.port, error = %e, "tcp honeypot listener exited with error");
            }
        }));
    }
    // Listeners run for the process lifetime; nothing further to await here
    // beyond letting them run. Errors are logged inside each task.
    for h in handles {
        let _ = h.await;
    }
    Ok(())
}

async fn run_single(
    cfg: TcpPortConfig,
    host_id: String,
    tx: Sender<HoneypotEvent>,
) -> Result<()> {
    let addr = format!("0.0.0.0:{}", cfg.port);
    let listener = TcpListener::bind(&addr).await?;
    info!(port = cfg.port, label = %cfg.label, "tcp honeypot listening");

    loop {
        let (mut socket, peer) = match listener.accept().await {
            Ok(v) => v,
            Err(e) => {
                warn!(port = cfg.port, error = %e, "accept failed");
                continue;
            }
        };

        let host_id = host_id.clone();
        let tx = tx.clone();
        let cfg = cfg.clone();

        tokio::spawn(async move {
            let open_event = HoneypotEvent::new(
                &host_id,
                "tcp_honeypot",
                Some(peer.ip()),
                Some(peer.port()),
                Confidence::Certain,
                Severity::Low,
                EventKind::ConnectionOpened {
                    local_port: cfg.port,
                    protocol_guess: cfg.banner.as_ref().map(|_| cfg.label.clone()),
                },
            );
            let _ = tx.try_send(open_event);

            if let Some(banner) = &cfg.banner {
                if let Err(e) = socket.write_all(banner.as_bytes()).await {
                    warn!(port = cfg.port, error = %e, "failed writing banner");
                    return;
                }
            }

            let mut buf = vec![0u8; MAX_READ_BYTES];
            match timeout(READ_TIMEOUT, socket.read(&mut buf)).await {
                Ok(Ok(n)) if n > 0 => {
                    let sample = printable_sample(&buf[..n]);
                    let guess = guess_protocol(&buf[..n], cfg.port);

                    let event = HoneypotEvent::new(
                        &host_id,
                        "tcp_honeypot",
                        Some(peer.ip()),
                        Some(peer.port()),
                        Confidence::Certain,
                        Severity::Medium,
                        EventKind::RawPayload {
                            local_port: cfg.port,
                            byte_len: n,
                            printable_sample: sample,
                            protocol_guess: guess,
                        },
                    );
                    let _ = tx.try_send(event);
                }
                Ok(Ok(_)) => { /* zero-byte read: peer closed immediately */ }
                Ok(Err(e)) => warn!(port = cfg.port, error = %e, "read error"),
                Err(_) => { /* idle timeout, no payload sent */ }
            }
        });
    }
}

/// Renders bytes as a safe, bounded, printable string for logging --
/// never echo raw attacker-controlled bytes into logs unescaped, since
/// downstream log viewers (terminal, dashboards) could be exploited by
/// embedded control sequences.
fn printable_sample(bytes: &[u8]) -> String {
    const MAX_SAMPLE: usize = 256;
    bytes
        .iter()
        .take(MAX_SAMPLE)
        .map(|&b| {
            if b.is_ascii_graphic() || b == b' ' {
                b as char
            } else {
                '.'
            }
        })
        .collect()
}

fn guess_protocol(bytes: &[u8], port: u16) -> Option<String> {
    let head = &bytes[..bytes.len().min(16)];
    if head.starts_with(b"SSH-") {
        return Some("ssh".to_string());
    }
    if head.starts_with(b"GET ") || head.starts_with(b"POST ") || head.starts_with(b"HEAD ") {
        return Some("http".to_string());
    }
    if head.starts_with(&[0x16, 0x03]) {
        return Some("tls".to_string());
    }
    if head.starts_with(b"\x00\x00\x00") {
        return Some("possible-length-prefixed-binary".to_string());
    }
    match port {
        21 => Some("ftp".to_string()),
        23 => Some("telnet".to_string()),
        25 | 587 => Some("smtp".to_string()),
        445 => Some("smb".to_string()),
        3389 => Some("rdp".to_string()),
        6379 => Some("redis".to_string()),
        _ => None,
    }
}
