//! Live packet capture via `pnet`'s datalink layer.
//!
//! `pnet` abstracts the OS backend for us: raw `AF_PACKET` sockets on
//! Linux, BPF on macOS/BSD, and Npcap/WinPcap on Windows -- the same
//! `datalink::channel()` / `Channel::Ethernet` API is used on every
//! platform, so `parse_ethernet_frame` and everything downstream is
//! already portable. What differs across platforms is how interfaces
//! are named and what privilege the OS requires, both handled below.

use crate::packet::{parse_ethernet_frame, PacketMeta};
use anyhow::{anyhow, Result};
use pnet::datalink::{self, Channel, NetworkInterface};
use tokio::sync::mpsc::Sender;

pub fn list_interfaces() -> Vec<NetworkInterface> {
    datalink::interfaces()
}

/// Pretty-prints available interfaces. On Linux/macOS `name` (e.g.
/// `eth0`, `en0`) is what you pass to `--interface`. On Windows the
/// underlying device name is an opaque `\Device\NPF_{GUID}`, so the
/// human-readable `description` and the numeric `index` are shown too
/// -- either can be passed to `--interface` instead of the raw name.
pub fn print_interfaces() {
    for iface in list_interfaces() {
        let ips: Vec<String> = iface.ips.iter().map(|n| n.to_string()).collect();
        println!(
            "[{}] {}{}\n      ips: {}",
            iface.index,
            iface.name,
            if iface.description.is_empty() || iface.description == iface.name {
                String::new()
            } else {
                format!("  ({})", iface.description)
            },
            if ips.is_empty() { "-".to_string() } else { ips.join(", ") }
        );
    }
    #[cfg(windows)]
    println!("\nTip: on Windows, pass the number in [brackets] to --interface instead of the raw device name.");
}

/// Resolves `--interface` against index, exact name, or a
/// case-insensitive substring of the description -- so the same CLI
/// works whether the platform calls it "eth0" or
/// "\\Device\\NPF_{9B2E...}" / "Ethernet 2".
pub fn find_interface(selector: &str) -> Result<NetworkInterface> {
    let interfaces = datalink::interfaces();

    if let Ok(idx) = selector.parse::<u32>() {
        if let Some(iface) = interfaces.iter().find(|i| i.index == idx) {
            return Ok(iface.clone());
        }
    }
    if let Some(iface) = interfaces.iter().find(|i| i.name == selector) {
        return Ok(iface.clone());
    }
    let needle = selector.to_lowercase();
    if let Some(iface) = interfaces
        .iter()
        .find(|i| i.description.to_lowercase().contains(&needle))
    {
        return Ok(iface.clone());
    }

    Err(anyhow!(
        "interface '{selector}' not found. Run --list-interfaces to see valid names/indices."
    ))
}

/// Runs the blocking capture loop on a dedicated OS thread (the
/// datalink receiver is synchronous on every backend) and forwards
/// parsed packets to an async consumer via a channel.
pub fn spawn_capture_thread(iface: NetworkInterface, tx: Sender<PacketMeta>) -> Result<()> {
    let cfg = datalink::Config::default();
    let channel = datalink::channel(&iface, cfg).map_err(|e| anyhow!("{}", capture_open_error(&iface, &e)))?;
    let mut rx = match channel {
        Channel::Ethernet(_tx, rx) => rx,
        _ => return Err(anyhow!("unsupported channel type for interface {}", iface.name)),
    };

    std::thread::Builder::new()
        .name("nsm-capture".into())
        .spawn(move || loop {
            match rx.next() {
                Ok(frame) => {
                    if let Some(meta) = parse_ethernet_frame(frame) {
                        if tx.blocking_send(meta).is_err() {
                            tracing::warn!("analysis channel closed, stopping capture");
                            break;
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("capture read error: {e}");
                }
            }
        })
        .map_err(|e| anyhow!("failed to spawn capture thread: {e}"))?;
    Ok(())
}

/// Builds a platform-appropriate hint when opening the capture device
/// fails -- almost always a privilege or missing-driver issue, and the
/// fix is different on each OS.
fn capture_open_error(iface: &NetworkInterface, source: &std::io::Error) -> String {
    #[cfg(unix)]
    let hint = "run this binary as root (or grant it CAP_NET_RAW, e.g. `sudo setcap cap_net_raw,cap_net_admin=eip ./nsm`)";
    #[cfg(windows)]
    let hint = "run this binary from an elevated (Administrator) terminal, and make sure Npcap is installed with \"WinPcap API-compatible Mode\" enabled";
    #[cfg(not(any(unix, windows)))]
    let hint = "check that this platform's packet-capture driver is installed and that the process has sufficient privileges";

    format!("failed to open capture on '{}': {source} -- {hint}", iface.name)
}
