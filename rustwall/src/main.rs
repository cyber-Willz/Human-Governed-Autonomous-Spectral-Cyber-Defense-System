mod chan;
mod config;
mod conntrack;
mod engine;
mod log_gate;
mod logging;
mod metrics;
mod nfqueue;
mod os_firewall;
mod packet;
mod quarantine;
mod rate_limit;
mod reject;
mod sync_worker;
mod threshold;

use arc_swap::ArcSwap;
use clap::Parser;
use config::Config;
use conntrack::ConnTrack;
use engine::Engine;
use log_gate::LogGate;
use metrics::Metrics;
use signal_hook::consts::signal::{SIGHUP, SIGINT, SIGTERM};
use signal_hook::iterator::Signals;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info, warn};

#[derive(Parser, Debug)]
#[command(name = "rustwall", about = "Inline stateful firewall backed by NFQUEUE")]
struct Args {
    /// Path to the TOML rule configuration.
    #[arg(short, long, default_value = "/etc/rustwall/rustwall.toml")]
    config: PathBuf,

    /// Validate the config and exit without binding to NFQUEUE (useful for
    /// pre-deployment CI checks, analogous to `checkpoint_conf` dry-run
    /// tooling or `nft -c` config validation).
    #[arg(long)]
    check_only: bool,
}

/// Builds the Quarantine, wiring in an OS-native firewall sync backend if
/// `sync_to_os_firewall` is set. Deliberately fails soft, not hard: if the
/// backend can't initialize (e.g. `nft` isn't installed, or this isn't
/// actually Linux), rustwall still runs with its own in-process quarantine
/// enforcement -- OS sync is defense in depth on top of that, never a
/// prerequisite for rustwall to function at all.
///
/// The backend itself is never called directly from here (or from
/// Quarantine::ban/sweep_expired) -- it's driven entirely by a dedicated
/// worker thread (sync_worker::spawn) reading off a bounded channel. An
/// earlier version of this code called the backend's `nft`/`netsh`
/// subprocess synchronously from the packet-processing and maintenance
/// threads; a hung subprocess call had no bounded latency and could stall
/// packet processing indefinitely. That's fixed now: `ban()`/`sweep_expired`
/// just enqueue and return immediately, and only this dedicated thread ever
/// blocks on a subprocess call.
fn build_quarantine(
    cfg: &Config,
    metrics: Arc<Metrics>,
    running: Arc<AtomicBool>,
) -> quarantine::Quarantine {
    if !cfg.sync_to_os_firewall {
        return quarantine::Quarantine::new(cfg.quarantine_max_entries);
    }

    #[cfg(target_os = "linux")]
    {
        match os_firewall::NftablesSync::new() {
            Ok(sync) => {
                info!("OS firewall sync enabled: nftables (rustwall_dynamic table)");
                let tx = sync_worker::spawn(Arc::new(sync), metrics, running);
                return quarantine::Quarantine::with_os_sync(cfg.quarantine_max_entries, tx);
            }
            Err(e) => {
                warn!(
                    error = %e,
                    "sync_to_os_firewall is enabled but nftables initialization failed; \
                     continuing with rustwall's own in-process quarantine only. Check that \
                     `nft` is installed and this process has CAP_NET_ADMIN."
                );
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        match os_firewall::WindowsFirewallSync::new() {
            Ok(sync) => {
                info!("OS firewall sync enabled: Windows Firewall (netsh)");
                let tx = sync_worker::spawn(Arc::new(sync), metrics, running);
                return quarantine::Quarantine::with_os_sync(cfg.quarantine_max_entries, tx);
            }
            Err(e) => {
                warn!(error = %e, "sync_to_os_firewall is enabled but Windows Firewall sync failed to initialize");
            }
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        warn!(
            "sync_to_os_firewall is enabled but no OS firewall backend exists for this platform; \
             continuing with rustwall's own in-process quarantine only"
        );
    }

    quarantine::Quarantine::new(cfg.quarantine_max_entries)
}

fn main() -> anyhow::Result<()> {
    logging::init();
    let args = Args::parse();

    let cfg = Config::load(&args.config)?;
    info!(
        rules = cfg.rules.len(),
        queue_num = cfg.queue_num,
        queue_workers = cfg.queue_workers,
        "configuration loaded"
    );

    if args.check_only {
        println!(
            "config OK: {} rules loaded, {} queue worker(s) starting at queue {}",
            cfg.rules.len(),
            cfg.queue_workers,
            cfg.queue_num
        );
        return Ok(());
    }

    let running = Arc::new(AtomicBool::new(true));
    let metrics = Metrics::new();
    let conntrack = Arc::new(ConnTrack::new(cfg.conntrack.clone()));
    let quarantine = Arc::new(build_quarantine(&cfg, metrics.clone(), running.clone()));
    let engine = Arc::new(ArcSwap::from_pointee(Engine::new(&cfg)));
    let log_gate = Arc::new(LogGate::new(cfg.log_max_per_sec));

    // Signal handling: SIGINT/SIGTERM stop the process; SIGHUP reloads the
    // config file and hot-swaps the rule engine without dropping conntrack
    // state or restarting NFQUEUE workers. This matters operationally --
    // restarting to pick up a rule change means every active connection has
    // to re-handshake, which is a real (if brief) disruption on a box
    // sitting inline on production traffic.
    {
        let running = running.clone();
        let engine = engine.clone();
        let config_path = args.config.clone();
        let mut signals = Signals::new([SIGHUP, SIGINT, SIGTERM])?;
        std::thread::spawn(move || {
            for sig in signals.forever() {
                match sig {
                    SIGHUP => match Config::load(&config_path) {
                        Ok(new_cfg) => {
                            let rule_count = new_cfg.rules.len();
                            engine.store(Arc::new(Engine::new(&new_cfg)));
                            info!(rule_count, "SIGHUP: rules reloaded, conntrack state preserved");
                        }
                        Err(e) => {
                            warn!(error = %e, "SIGHUP: reload failed, keeping previous rules active");
                        }
                    },
                    SIGINT | SIGTERM => {
                        info!("shutdown signal received");
                        running.store(false, Ordering::Relaxed);
                    }
                    _ => {}
                }
            }
        });
    }

    // Background maintenance: expires stale conntrack entries and
    // rate-limiter buckets so long-running deployments have bounded memory
    // instead of growing until an OOM kill takes the firewall down, and
    // flushes a summary of any log lines the log-rate-limiter suppressed
    // this window so a flood is visible in aggregate even when individual
    // lines were dropped.
    {
        let conntrack = conntrack.clone();
        let quarantine = quarantine.clone();
        let engine = engine.clone();
        let log_gate = log_gate.clone();
        let running = running.clone();
        std::thread::spawn(move || {
            while running.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_secs(10));
                conntrack.sweep_expired();
                quarantine.sweep_expired();
                engine.load().periodic_maintenance();
                let suppressed = log_gate.take_suppressed_count();
                if suppressed > 0 {
                    warn!(
                        suppressed_log_lines = suppressed,
                        "log rate limit engaged this window; some policy decisions were not logged individually"
                    );
                }
                info!(
                    active_flows = conntrack.len(),
                    quarantined_hosts = quarantine.active_count(),
                    "maintenance sweep"
                );
            }
        });
    }

    // Optional metrics endpoint.
    if let Some(addr_str) = &cfg.metrics_listen {
        match addr_str.parse::<std::net::SocketAddr>() {
            Ok(addr) => {
                let metrics = metrics.clone();
                let conntrack = conntrack.clone();
                let quarantine = quarantine.clone();
                let running = running.clone();
                let engine_for_metrics = engine.clone();
                let auth_token = cfg.metrics_auth_token.clone();
                std::thread::spawn(move || {
                    if let Err(e) = metrics::serve(
                        addr,
                        metrics,
                        move || conntrack.len() as u64,
                        move || engine_for_metrics.load().rule_count() as u64,
                        quarantine,
                        auth_token,
                        running,
                    ) {
                        error!(error = %e, "metrics server exited with error");
                    }
                });
            }
            Err(e) => {
                warn!(error = %e, addr = %addr_str, "invalid metrics_listen address, metrics disabled");
            }
        }
    }

    // Spawn one NFQUEUE worker thread per configured queue number
    // (queue_num..queue_num+queue_workers). This requires a matching
    // `queue num START-END fanout` rule in nft/iptables so the kernel
    // actually distributes packets across the range -- see README.
    let mut handles = Vec::new();
    for i in 0..cfg.queue_workers {
        let queue_num = cfg.queue_num + i;
        let engine = engine.clone();
        let conntrack = conntrack.clone();
        let quarantine = quarantine.clone();
        let metrics = metrics.clone();
        let log_gate = log_gate.clone();
        let running = running.clone();
        handles.push(std::thread::spawn(move || {
            if let Err(e) = nfqueue::run(
                i, queue_num, engine, conntrack, quarantine, metrics, log_gate, running,
            ) {
                error!(worker_id = i, error = %e, "nfqueue worker exited with error");
            }
        }));
    }

    for h in handles {
        let _ = h.join();
    }

    Ok(())
}
