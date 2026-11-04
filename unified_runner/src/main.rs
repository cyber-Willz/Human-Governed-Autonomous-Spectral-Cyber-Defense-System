//! End-to-end live run of the unified NDR/IPS/SIEM detection +
//! containment path:
//!
//! ```text
//! synthetic packets -> ingestion::IngestionPipeline (broadcast fan-out)
//!                    -> hg_builder::HypergraphPipeline (windowed hyperedges,
//!                       HMM-belief-weighted, Fiedler/rho_B/Ramanujan scoring)
//!                    -> correlation_engine::CorrelationEngine (per-host fusion,
//!                       multi-lane corroboration / single-lane-critical gating)
//!                    -> ips_control_plane::IpsControlPlane (review_queue's
//!                       confidence/margin trigger: auto-resolve or queue)
//!                    -> rustwall::quarantine::Quarantine (actual containment)
//! ```
//!
//! Replays four scenarios back to back against the live pipeline:
//! 1. A DDoS-style destination fan-in (many sources, one victim) --
//!    moderate confidence, ends up queued for human review.
//! 2. A honeytoken touch, submitted the way `honeypot_engine` would --
//!    confidence 1.0 by construction, auto-contained immediately.
//! 3. A port-scan-style source fan-out (one attacker, many destination
//!    ports) -- also queued, demonstrating the review path is not just a
//!    honeytoken special case.
//! 4. `nsm --simulate` run as a real, separate child process -- its own
//!    independent capture/detector stack, generating actual NDJSON
//!    alerts on its own stdout (not synthetic events built in-process
//!    like scenarios 1-3), bridged into the same live correlation engine
//!    via `correlation_engine::NsmSender`.

use std::collections::VecDeque;
use std::net::{IpAddr, Ipv4Addr};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use bytes::Bytes;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use correlation_engine::{CorrelationConfig, CorrelationEngine};
use hg_builder::belief::StaticBeliefSource;
use hg_builder::pipeline::HypergraphPipeline;
use ingestion::{
    Direction, FlowKey, IngestedPacket, IngestionConfig, IngestionError, IngestionPipeline,
    PacketSource, Protocol,
};
use ips_control_plane::IpsControlPlane;
use review_queue::prelude::{SlaPolicy, TriggerConfig};
use rustwall::quarantine::Quarantine;

mod advanced;
mod rl;

/// Replays a fixed queue of pre-built packets, then blocks forever (so the
/// ingestion pipeline stays alive for `shutdown` to end it cleanly rather
/// than erroring out once the synthetic traffic is exhausted).
struct DemoSource {
    queue: VecDeque<IngestedPacket>,
}

#[async_trait::async_trait]
impl PacketSource for DemoSource {
    async fn recv(&mut self) -> Result<IngestedPacket, IngestionError> {
        match self.queue.pop_front() {
            Some(pkt) => Ok(pkt),
            None => {
                std::future::pending::<()>().await;
                unreachable!("pending future never resolves")
            }
        }
    }
}

fn pkt(src: IpAddr, dst: IpAddr, dst_port: u16) -> IngestedPacket {
    IngestedPacket {
        flow: FlowKey {
            src_ip: src,
            dst_ip: dst,
            src_port: 44321,
            dst_port,
            protocol: Protocol::Tcp,
        },
        captured_at: SystemTime::now(),
        ifindex: 1,
        direction: Direction::Ingress,
        payload: Bytes::from_static(b"synthetic-demo-payload"),
        seq: 0, // overwritten by IngestionPipeline::run
    }
}

fn v4(a: u8, b: u8, c: u8, d: u8) -> IpAddr {
    IpAddr::V4(Ipv4Addr::new(a, b, c, d))
}

/// `nsm` and `unified_runner` land in the same `target/debug/` directory
/// when built via `cargo build --workspace`, so resolving the sibling
/// binary through `current_exe()` works regardless of the caller's CWD --
/// no PATH lookup, no hardcoded `../target/...` relative path.
fn sibling_nsm_binary() -> std::io::Result<std::path::PathBuf> {
    let exe = std::env::current_exe()?;
    let dir = exe.parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound, "current_exe() has no parent directory")
    })?;
    let name = if cfg!(windows) { "nsm.exe" } else { "nsm" };
    Ok(dir.join(name))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // -- containment + control plane -------------------------------------
    let quarantine = Arc::new(Quarantine::new(1024));
    let plane = Arc::new(IpsControlPlane::new(
        TriggerConfig::default(),
        SlaPolicy::FailSafe,
        quarantine.clone(),
    )?);

    // Hosts autonomous containment must never ban. In a real deployment this
    // is configuration (gateway, resolvers, the protected servers, authorised
    // scanners); here it is the demo victim (the nsm --simulate "local" host).
    plane.protect_hosts([nsm::sim::DEMO_LOCAL_IP]);

    // RL decision layer, attached in Shadow mode from the start: it scores
    // every verdict of scenarios 1-5 and learns from their outcomes while
    // changing nothing. Scenario 6 shows what it earned.
    let rl_layer = rl::attach(&plane);

    // -- correlation engine ------------------------------------------------
    let (out_tx, out_rx) = tokio::sync::mpsc::channel(64);
    let (engine, senders, _corr_metrics) =
        CorrelationEngine::new(CorrelationConfig::default(), out_tx, None);
    tokio::spawn(engine.run());

    let plane_for_run = plane.clone();
    tokio::spawn(async move { plane_for_run.run(out_rx).await });

    // -- hypergraph builder (hg_builder) -----------------------------------
    // Lower thresholds than the crate defaults (50 / 30) so a small demo
    // burst closes a window immediately instead of waiting for real
    // production-scale volume.
    let window_cfg = hg_builder::builder::WindowConfig {
        fast_window: Duration::from_secs(5),
        slow_window: Duration::from_secs(60),
        fan_in_count_threshold: 20,
        fan_out_count_threshold: 15,
        session_window: Duration::from_secs(30),
        honeytoken_campaign_window: Duration::from_secs(60),
    };
    let belief_source = Arc::new(StaticBeliefSource::default());
    let hg_pipeline = HypergraphPipeline::new(window_cfg, belief_source, senders.hypergraph.clone());

    // -- ingestion -----------------------------------------------------
    let (ingestion_pipeline, ingest_metrics) = IngestionPipeline::new(IngestionConfig::default());
    let lanes = ingestion_pipeline.subscribe_lanes();
    // Demo doesn't run the ClamAV or autoencoder (siem-ml) lanes; hg_builder
    // consumes the "spectral" broadcast lane, same as the guide's real
    // wiring (`nsm`/`clamav`/`spectral` are all clones of one broadcast
    // channel -- consuming one doesn't affect the others).
    let (_touch_tx, touch_rx) = tokio::sync::mpsc::channel(8);
    tokio::spawn(hg_pipeline.run(lanes.spectral, touch_rx, Duration::from_millis(250)));

    let victim = v4(10, 0, 0, 50);
    let scan_target = v4(10, 0, 0, 60);
    let attacker = v4(10, 0, 2, 7);
    let honeytoken_host = v4(10, 0, 3, 99);

    let mut queue = VecDeque::new();
    // A little benign warm-up traffic first.
    for i in 1..=3u8 {
        queue.push_back(pkt(v4(10, 0, 9, i), v4(10, 0, 9, 200), 80));
    }
    // Scenario 1: destination fan-in (DDoS-style) -- 25 distinct sources
    // hitting one victim:port, past the 20-source threshold above.
    for i in 1..=25u8 {
        queue.push_back(pkt(v4(10, 0, 1, i), victim, 443));
    }
    // Scenario 3: source fan-out (port-scan-style) -- one attacker probing
    // 20 distinct destination ports on one target, past the 15-port
    // threshold above.
    for port in 20000..20000 + 20u16 {
        queue.push_back(pkt(attacker, scan_target, port));
    }

    let source = DemoSource { queue };
    let shutdown = CancellationToken::new();
    let shutdown_for_run = shutdown.clone();
    tokio::spawn(async move {
        if let Err(err) = ingestion_pipeline.run(source, shutdown_for_run).await {
            tracing::error!(%err, "ingestion pipeline exited with error");
        }
    });

    // Let the fan-in/fan-out bursts propagate through hg_builder's
    // count-threshold closure and scoring.
    tokio::time::sleep(Duration::from_millis(400)).await;

    println!("\n=== after DDoS fan-in + port-scan fan-out (hypergraph lane only) ===");
    let stats = plane.queue().stats();
    println!(
        "review_queue: total={} auto_resolved={} pending={} under_review={} resolved={}",
        stats.total, stats.auto_resolved, stats.pending, stats.under_review, stats.resolved
    );
    println!(
        "quarantine: victim {victim} banned={}, scan target {scan_target} banned={}",
        quarantine.is_active(victim),
        quarantine.is_active(scan_target)
    );
    println!(
        "(expected: still 0/false -- correlation_engine only emits a verdict on either\n \
         multi-lane corroboration or a single lane at Critical severity; a lone\n \
         moderate-confidence hypergraph signal is deliberately held back, see\n \
         `CorrelationEngine::maybe_emit`.)"
    );

    // Now a second, independent lane corroborates the same hosts -- e.g.
    // `siem-ml`'s autoencoder flagging the same victim/attacker flow on its
    // own. This is the realistic multi-signal case the fusion design exists
    // for: two moderate-confidence lanes on the same host should outweigh
    // either alone.
    senders.spectral.submit(
        victim,
        correlation_engine::FlowKey { src_ip: v4(10, 0, 1, 1), dst_ip: victim, src_port: 44321, dst_port: 443, protocol: correlation_engine::Protocol::Tcp },
        0.6,
        0.0,
    );
    senders.spectral.submit(
        attacker,
        correlation_engine::FlowKey { src_ip: attacker, dst_ip: scan_target, src_port: 44321, dst_port: 20000, protocol: correlation_engine::Protocol::Tcp },
        0.55,
        0.0,
    );
    tokio::time::sleep(Duration::from_millis(300)).await;

    println!("\n=== after a corroborating spectral-lane signal on the same hosts ===");
    let stats = plane.queue().stats();
    println!(
        "review_queue: total={} auto_resolved={} pending={} under_review={} resolved={}",
        stats.total, stats.auto_resolved, stats.pending, stats.under_review, stats.resolved
    );
    println!(
        "quarantine: victim {victim} banned={}, scan target {scan_target} banned={}",
        quarantine.is_active(victim),
        quarantine.is_active(scan_target)
    );
    println!(
        "(expected: pending>=1 -- two moderate-confidence lanes on the same host now\n \
         clear `TriggerConfig`'s multi-lane-corroboration reason and queue for human\n \
         review; confidence stays under the 0.90 auto-resolve floor, so it's queued,\n \
         not auto-banned.)"
    );

    // Scenario 2: a honeytoken touch, submitted exactly the way
    // `honeypot_engine`'s sink would (see `correlation_engine::HoneytokenSender`
    // docs: confidence 1.0, always Critical -- "no spectral scoring needed").
    senders
        .honeytoken
        .submit(honeytoken_host, "tok-canary-01", None);
    tokio::time::sleep(Duration::from_millis(200)).await;

    println!("\n=== after honeytoken touch ===");
    let stats = plane.queue().stats();
    println!(
        "review_queue: total={} auto_resolved={} pending={} under_review={} resolved={}",
        stats.total, stats.auto_resolved, stats.pending, stats.under_review, stats.resolved
    );
    println!(
        "quarantine: honeytoken host {honeytoken_host} banned={}",
        quarantine.is_active(honeytoken_host)
    );
    println!("quarantine: total bans issued so far = {}", quarantine.bans_total());

    let ingest_snap = ingest_metrics.snapshot();
    println!(
        "\ningestion: captured={} bytes={} spectral_lane_delivered={}",
        ingest_snap.captured, ingest_snap.bytes, ingest_snap.spectral_delivered
    );

    // Scenario 4: `nsm --simulate` as a genuinely separate process -- its
    // own capture loop, its own five detectors, its own flow table --
    // bridged in over its real stdout NDJSON stream via `NsmSender`,
    // exactly the interface `nsm`'s own README describes ("pipe into...
    // a SIEM"). Not a synthetic in-process event like scenarios 1-3.
    println!("\n=== spawning `nsm --simulate` as a child process ===");
    match run_nsm_bridge(&senders.nsm).await {
        Ok(fed) => println!("nsm bridge: parsed and forwarded {fed} real alerts from nsm's own stdout"),
        Err(err) => println!("nsm bridge: couldn't run ({err}) -- skipping this scenario"),
    }
    tokio::time::sleep(Duration::from_millis(200)).await;

    println!("\n=== after the nsm bridge ===");
    let stats = plane.queue().stats();
    println!(
        "review_queue: total={} auto_resolved={} pending={} under_review={} resolved={}",
        stats.total, stats.auto_resolved, stats.pending, stats.under_review, stats.resolved
    );
    println!("quarantine: total bans issued so far = {}", quarantine.bans_total());

    // Scenario 5 (advanced build): connectome-LSTM lane + burn-llm bridge.
    advanced::run(&plane, &senders, &quarantine).await;

    // Scenario 6: RL decision layer (sandbox -> risk -> policy -> shielded
    // action selection -> reward -> guarded learning).
    rl::run(&plane, &senders, &quarantine, &rl_layer).await;

    shutdown.cancel();
    tokio::time::sleep(Duration::from_millis(50)).await;
    Ok(())
}

/// Spawns `nsm --simulate`, reads its NDJSON alert stream off real stdout,
/// parses each line as a `nsm::alert::Alert`, and forwards it into the
/// live correlation engine via `sender.submit_alert`. Stops after either
/// `MAX_ALERTS` alerts or `TIME_BUDGET` elapses (`--simulate` runs
/// indefinitely otherwise), then kills the child. Returns the number of
/// alerts successfully parsed and forwarded.
async fn run_nsm_bridge(sender: &correlation_engine::NsmSender) -> std::io::Result<usize> {
    const MAX_ALERTS: usize = 60;
    const TIME_BUDGET: Duration = Duration::from_secs(3);

    let nsm_path = sibling_nsm_binary()?;
    let mut child = Command::new(&nsm_path)
        .arg("--simulate")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()?;

    let stdout = child.stdout.take().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::Other, "nsm child process has no stdout pipe")
    })?;
    let mut lines = BufReader::new(stdout).lines();

    let mut fed = 0usize;
    let deadline = tokio::time::Instant::now() + TIME_BUDGET;
    loop {
        if fed >= MAX_ALERTS {
            break;
        }
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        let next_line = tokio::time::timeout(remaining, lines.next_line()).await;
        let line = match next_line {
            Ok(Ok(Some(line))) => line,
            Ok(Ok(None)) => break, // nsm's stdout closed
            Ok(Err(err)) => {
                tracing::warn!(%err, "error reading nsm stdout");
                break;
            }
            Err(_elapsed) => break, // time budget hit
        };
        match serde_json::from_str::<NsmAlertWire>(&line) {
            Ok(wire) => {
                let Some(host) = wire.src_ip.or(wire.dst_ip) else {
                    tracing::warn!(detector = %wire.detector, "nsm alert has neither src_ip nor dst_ip, skipping");
                    continue;
                };
                let attribution =
                    correlation_engine::Attribution::from_alert_fields(wire.src_ip, &wire.extra);
                sender.submit_attributed(host, wire.detector, wire.message, wire.severity, attribution);
                fed += 1;
            }
            Err(err) => tracing::warn!(%err, %line, "couldn't parse an nsm alert line"),
        }
    }

    let _ = child.kill().await;
    Ok(fed)
}

/// Mirrors the wire shape of `nsm::alert::Alert` (see that struct's docs
/// for why `Alert` itself isn't `Deserialize`) with owned `String`s
/// instead of `&'static str`, so a transient NDJSON line borrowed from a
/// child process's stdout can actually be parsed into it.
#[derive(serde::Deserialize)]
struct NsmAlertWire {
    severity: nsm::alert::Severity,
    detector: String,
    message: String,
    src_ip: Option<IpAddr>,
    dst_ip: Option<IpAddr>,
    #[serde(default)]
    extra: serde_json::Value,
}
