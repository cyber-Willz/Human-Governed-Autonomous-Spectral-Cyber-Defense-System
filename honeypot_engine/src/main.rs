mod config;
mod event;
mod honeytoken;
mod sink;
mod ssh_honeypot;
mod tcp_honeypot;

use anyhow::{Context, Result};
use config::Config;
use event::HoneypotEvent;
use std::env;
use tokio::signal;
use tokio::sync::mpsc;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .json()
        .init();

    let config_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "honeypot.toml".to_string());
    let cfg = Config::load(&config_path)
        .with_context(|| format!("loading config from {}", config_path))?;

    info!(host_id = %cfg.host_id, "honeypot_engine starting");

    // Single bounded mpsc event bus, fan-in from every honeypot subsystem
    // -- mirrors the typed channel fan-in pattern used by the SIEM
    // correlation engine. Bounded so a burst of adversarial traffic
    // applies backpressure to producers rather than growing memory
    // unbounded; producers use try_send and drop-and-log on a full
    // channel rather than blocking the accept loop.
    let (tx, rx) = mpsc::channel::<HoneypotEvent>(cfg.sinks.channel_capacity);

    let mut tasks = tokio::task::JoinSet::new();

    // Sink runner: drains the bus to JSONL / Postgres / Unix socket.
    {
        let sink_cfg = cfg.sinks.clone();
        tasks.spawn(async move {
            sink::SinkRunner::new(sink_cfg)
                .run(rx)
                .await
                .context("sink runner exited")
        });
    }

    // SSH honeypot.
    {
        let ssh_cfg = cfg.ssh.clone();
        let host_id = cfg.host_id.clone();
        let tx = tx.clone();
        tasks.spawn(async move {
            ssh_honeypot::SshHoneypot::new(ssh_cfg, host_id, tx)
                .run()
                .await
                .context("ssh honeypot exited")
        });
    }

    // Multi-port low-interaction TCP honeypots.
    if !cfg.tcp_ports.is_empty() {
        let ports = cfg.tcp_ports.clone();
        let host_id = cfg.host_id.clone();
        let tx = tx.clone();
        tasks.spawn(async move {
            tcp_honeypot::run_all(ports, host_id, tx)
                .await
                .context("tcp honeypot fan-out exited")
        });
    }

    // Honeytoken provisioning + watcher.
    {
        let ht_cfg = cfg.honeytokens.clone();
        let host_id = cfg.host_id.clone();
        let tx = tx.clone();
        tasks.spawn(async move {
            honeytoken::provision_and_watch(ht_cfg, host_id, tx)
                .await
                .context("honeytoken watcher exited")
        });
    }

    // Drop the original sender once all subsystems hold their own clone,
    // so the sink's rx.recv() loop can terminate on shutdown once every
    // producer task has finished.
    drop(tx);

    tokio::select! {
        _ = signal::ctrl_c() => {
            info!("shutdown signal received, stopping honeypot_engine");
        }
        _ = async {
            // Loop on task completions rather than returning after the
            // first one. Subsystems can legitimately finish early (e.g.
            // the honeytoken watcher returns immediately when nothing is
            // configured) without that being a reason to tear down the
            // whole daemon -- only stop looping once every task has
            // finished, at which point there's nothing left running
            // anyway.
            while let Some(result) = tasks.join_next().await {
                match result {
                    Ok(Ok(())) => info!("a honeypot subsystem exited cleanly"),
                    Ok(Err(e)) => error!(error = %e, "a honeypot subsystem exited with error"),
                    Err(e) => error!(error = %e, "a honeypot subsystem task panicked"),
                }
            }
            info!("all honeypot subsystems have exited, shutting down");
        } => {}
    }

    Ok(())
}
