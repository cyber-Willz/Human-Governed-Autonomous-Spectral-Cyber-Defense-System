mod alert;
mod capture;
mod detect;
mod flow;
mod packet;
mod sim;
#[cfg(all(target_os = "linux", feature = "xdp"))]
mod xdp;

use anyhow::Result;
use clap::Parser;
use detect::DetectionEngine;
use flow::FlowTable;
use std::collections::HashSet;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

/// nsm -- a lightweight Network Security Monitor.
#[derive(Parser, Debug)]
#[command(name = "nsm", version, about = "Lightweight Network Security Monitor")]
struct Cli {
    /// Network interface to capture on: name (Linux/macOS, e.g. eth0),
    /// numeric index, or a substring of the description (useful on
    /// Windows where device names are opaque GUIDs). Requires
    /// root/CAP_NET_RAW on Unix or an elevated terminal + Npcap on Windows.
    #[arg(short, long)]
    interface: Option<String>,

    /// List available network interfaces and exit.
    #[arg(long)]
    list_interfaces: bool,

    /// Run with synthetic traffic instead of a live capture -- no root needed.
    #[arg(long)]
    simulate: bool,

    /// Idle flow eviction timeout in seconds.
    #[arg(long, default_value_t = 300)]
    flow_idle_secs: u64,

    /// Capture via the XDP fast path instead of pnet/AF_PACKET.
    /// Linux only; requires root/CAP_BPF+CAP_NET_ADMIN and a compiled
    /// nsm-ebpf object (see `scripts/build-ebpf.sh` / README).
    #[cfg(all(target_os = "linux", feature = "xdp"))]
    #[arg(long)]
    xdp: bool,

    /// Path to the compiled nsm-ebpf object. Defaults to the path
    /// `scripts/build-ebpf.sh` produces.
    #[cfg(all(target_os = "linux", feature = "xdp"))]
    #[arg(long, default_value = "xdp/nsm-ebpf/target/bpfel-unknown-none/release/nsm-ebpf")]
    xdp_obj: std::path::PathBuf,

    /// XDP attach mode. `native` is fastest but needs driver support;
    /// fall back to `skb` if attaching fails.
    #[cfg(all(target_os = "linux", feature = "xdp"))]
    #[arg(long, value_enum, default_value = "native")]
    xdp_mode: xdp::XdpMode,

    /// When set, source IPv4 addresses behind a Critical port-scan or
    /// SYN-flood alert are pushed into the kernel's XDP blocklist
    /// (auto-expiring), so confirmed attackers get dropped before the
    /// network stack rather than merely logged.
    #[cfg(all(target_os = "linux", feature = "xdp"))]
    #[arg(long)]
    xdp_auto_block: bool,

    /// How long an auto-block entry stays in the kernel blocklist.
    #[cfg(all(target_os = "linux", feature = "xdp"))]
    #[arg(long, default_value_t = 300)]
    xdp_block_secs: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("nsm=info".parse()?))
        .with_writer(std::io::stderr) // keep stdout clean for NDJSON alerts
        .init();

    let cli = Cli::parse();

    if cli.list_interfaces {
        capture::print_interfaces();
        return Ok(());
    }

    let flow_table = Arc::new(FlowTable::new());
    let (tx, rx) = mpsc::channel(4096);

    // The port-scan detector needs to know which addresses are "us"
    // to tell inbound scans apart from ordinary outbound browsing.
    let mut local_ips: HashSet<IpAddr> = HashSet::new();
    local_ips.insert(IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
    local_ips.insert(IpAddr::V6(std::net::Ipv6Addr::LOCALHOST));

    if cli.simulate {
        local_ips.insert(sim::DEMO_LOCAL_IP);
        let engine = Arc::new(DetectionEngine::new(local_ips));
        tokio::spawn(sim::run(tx));
        #[cfg(all(target_os = "linux", feature = "xdp"))]
        {
            run_event_loop(flow_table, engine, rx, cli.flow_idle_secs, None, false, cli.xdp_block_secs).await
        }
        #[cfg(not(all(target_os = "linux", feature = "xdp")))]
        {
            run_event_loop(flow_table, engine, rx, cli.flow_idle_secs).await
        }
    } else {
        let iface_name = cli
            .interface
            .ok_or_else(|| anyhow::anyhow!("--interface is required (or pass --simulate / --list-interfaces)"))?;
        let iface = capture::find_interface(&iface_name)?;
        for net in &iface.ips {
            local_ips.insert(net.ip());
        }
        // Also fold in every other interface's addresses (e.g. Wi-Fi +
        // Hyper-V + Ethernet all resolve to "us" for routing purposes).
        for other in capture::list_interfaces() {
            for net in &other.ips {
                local_ips.insert(net.ip());
            }
        }
        tracing::info!("capturing on {} ({})", iface.name, iface.description);

        #[cfg(all(target_os = "linux", feature = "xdp"))]
        let xdp_handle = if cli.xdp {
            tracing::info!("using XDP fast-path capture (--xdp-mode {:?})", cli.xdp_mode);
            let handle = xdp::start(&cli.xdp_obj, &iface.name, cli.xdp_mode, tx).await?;
            Some(Arc::new(handle))
        } else {
            #[cfg(unix)]
            tracing::info!("if this fails, re-run as root or grant CAP_NET_RAW");
            capture::spawn_capture_thread(iface, tx)?;
            None
        };
        #[cfg(not(all(target_os = "linux", feature = "xdp")))]
        {
            #[cfg(unix)]
            tracing::info!("if this fails, re-run as root or grant CAP_NET_RAW");
            #[cfg(windows)]
            tracing::info!("if this fails, re-run from an elevated (Administrator) terminal with Npcap installed");
            capture::spawn_capture_thread(iface, tx)?;
        }

        let engine = Arc::new(DetectionEngine::new(local_ips));
        #[cfg(all(target_os = "linux", feature = "xdp"))]
        {
            run_event_loop(flow_table, engine, rx, cli.flow_idle_secs, xdp_handle, cli.xdp_auto_block, cli.xdp_block_secs).await
        }
        #[cfg(not(all(target_os = "linux", feature = "xdp")))]
        {
            run_event_loop(flow_table, engine, rx, cli.flow_idle_secs).await
        }
    }
}

#[cfg(not(all(target_os = "linux", feature = "xdp")))]
async fn run_event_loop(
    flow_table: Arc<FlowTable>,
    engine: Arc<DetectionEngine>,
    mut rx: mpsc::Receiver<packet::PacketMeta>,
    flow_idle_secs: u64,
) -> Result<()> {
    spawn_flow_reaper(flow_table.clone(), flow_idle_secs);

    tracing::info!("nsm running, alerts stream as NDJSON on stdout (Ctrl+C to stop)");
    loop {
        tokio::select! {
            maybe_meta = rx.recv() => {
                match maybe_meta {
                    Some(meta) => {
                        flow_table.update(&meta);
                        for a in engine.analyze(&meta) {
                            a.emit();
                        }
                    }
                    None => break, // capture side hung up
                }
            }
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("received Ctrl+C, shutting down ({} active flows)", flow_table.active_count());
                break;
            }
        }
    }

    Ok(())
}

#[cfg(all(target_os = "linux", feature = "xdp"))]
#[allow(clippy::too_many_arguments)]
async fn run_event_loop(
    flow_table: Arc<FlowTable>,
    engine: Arc<DetectionEngine>,
    mut rx: mpsc::Receiver<packet::PacketMeta>,
    flow_idle_secs: u64,
    xdp_handle: Option<Arc<xdp::XdpCapture>>,
    xdp_auto_block: bool,
    xdp_block_secs: u64,
) -> Result<()> {
    spawn_flow_reaper(flow_table.clone(), flow_idle_secs);

    let block_ttl = Duration::from_secs(xdp_block_secs);

    tracing::info!("nsm running, alerts stream as NDJSON on stdout (Ctrl+C to stop)");
    loop {
        tokio::select! {
            maybe_meta = rx.recv() => {
                match maybe_meta {
                    Some(meta) => {
                        flow_table.update(&meta);
                        for a in engine.analyze(&meta) {
                            // Kernel-level enforcement: a Critical
                            // port-scan/SYN-flood alert against a known
                            // source IPv4 gets pushed into BLOCKLIST_V4,
                            // so the *next* packet from that source is
                            // XDP_DROP'd before it costs any more CPU
                            // anywhere in the stack.
                            if xdp_auto_block {
                                if let (Some(handle), Some(IpAddr::V4(src))) = (&xdp_handle, a.src_ip) {
                                    if a.severity == alert::Severity::Critical {
                                        if let Err(e) = handle.block_ipv4(src, block_ttl) {
                                            tracing::warn!("xdp: failed to auto-block {src}: {e}");
                                        }
                                    }
                                }
                            }
                            a.emit();
                        }
                    }
                    None => break, // capture side hung up
                }
            }
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("received Ctrl+C, shutting down ({} active flows)", flow_table.active_count());
                break;
            }
        }
    }

    Ok(())
}

fn spawn_flow_reaper(flow_table: Arc<FlowTable>, flow_idle_secs: u64) {
    let idle = Duration::from_secs(flow_idle_secs);
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(30));
        loop {
            ticker.tick().await;
            let reaped = flow_table.reap_idle(idle);
            if reaped > 0 {
                tracing::debug!("reaped {reaped} idle flows, {} active", flow_table.active_count());
            }
        }
    });
}
