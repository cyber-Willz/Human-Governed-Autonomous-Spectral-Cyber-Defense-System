//! Event sinks.
//!
//! The honeypot's internal event bus is a single bounded `tokio::mpsc`
//! channel (see `main.rs`). This module drains that channel and fans each
//! event out to whichever sinks are configured:
//!
//! - JSONL file (rotation-aware, at-least-once, matches the existing log
//!   shipper's harvesting expectations so no shipper changes are needed).
//! - PostgreSQL (batched insert, matches the ACID / concurrent-writer
//!   rationale already used for the review queue).
//! - Unix domain socket (best-effort, real-time push for processes that
//!   want sub-second ingestion into SIEM correlation without waiting on
//!   the log shipper's poll interval).
//!
//! Sink failures are logged and do not block the bus: a slow or down
//! Postgres instance must never cause the honeypot listeners themselves to
//! stall, since that would be an easy DoS/blind-spot vector for an
//! attacker who notices timing changes.

use crate::config::SinkConfig;
use crate::event::HoneypotEvent;
use anyhow::{Context, Result};
use chrono::Utc;
use std::path::PathBuf;
use tokio::fs::{self, File, OpenOptions};
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;
use tokio::sync::mpsc::Receiver;
use tracing::{debug, error, info, warn};

pub struct SinkRunner {
    cfg: SinkConfig,
}

impl SinkRunner {
    pub fn new(cfg: SinkConfig) -> Self {
        Self { cfg }
    }

    /// Drains `rx` until the channel closes, fanning each event to all
    /// configured sinks. Runs as a dedicated tokio task.
    pub async fn run(self, mut rx: Receiver<HoneypotEvent>) -> Result<()> {
        fs::create_dir_all(&self.cfg.jsonl_dir)
            .await
            .with_context(|| format!("creating jsonl_dir {}", self.cfg.jsonl_dir.display()))?;

        let mut jsonl = JsonlSink::new(self.cfg.jsonl_dir.clone()).await?;

        let mut pg = match &self.cfg.postgres {
            Some(pgcfg) => match PostgresSink::connect(pgcfg.clone()).await {
                Ok(sink) => {
                    info!("honeypot: postgres sink connected");
                    Some(sink)
                }
                Err(e) => {
                    error!(error = %e, "honeypot: postgres sink failed to connect; continuing without it");
                    None
                }
            },
            None => None,
        };

        let unix_sock_path = self.cfg.unix_socket_path.clone();

        while let Some(event) = rx.recv().await {
            if let Err(e) = jsonl.write(&event).await {
                error!(error = %e, "honeypot: jsonl sink write failed");
            }

            if let Some(pg) = pg.as_mut() {
                if let Err(e) = pg.push(event.clone()).await {
                    warn!(error = %e, "honeypot: postgres sink push failed");
                }
            }

            if let Some(path) = unix_sock_path.as_ref() {
                if let Err(e) = forward_unix(path, &event).await {
                    debug!(error = %e, "honeypot: unix socket forward failed (best-effort, ignoring)");
                }
            }
        }

        if let Some(mut pg) = pg {
            let _ = pg.flush().await;
        }

        Ok(())
    }
}

/// Rotation-aware JSONL writer. Rotates daily by filename
/// (`events_YYYYMMDD.jsonl`), matching the convention already used by the
/// SSH auth/command logs so a single harvester config covers both.
struct JsonlSink {
    dir: PathBuf,
    current_date: chrono::NaiveDate,
    file: File,
}

impl JsonlSink {
    async fn new(dir: PathBuf) -> Result<Self> {
        let today = Utc::now().date_naive();
        let file = Self::open_for_date(&dir, today).await?;
        Ok(Self {
            dir,
            current_date: today,
            file,
        })
    }

    async fn open_for_date(dir: &PathBuf, date: chrono::NaiveDate) -> Result<File> {
        let path = dir.join(format!("events_{}.jsonl", date.format("%Y%m%d")));
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await
            .with_context(|| format!("opening jsonl sink file {}", path.display()))
    }

    async fn write(&mut self, event: &HoneypotEvent) -> Result<()> {
        let today = event.timestamp.date_naive();
        if today != self.current_date {
            self.file = Self::open_for_date(&self.dir, today).await?;
            self.current_date = today;
        }
        let mut line = serde_json::to_string(event)?;
        line.push('\n');
        self.file.write_all(line.as_bytes()).await?;
        // Flush per-line: honeypot event volume is low relative to the
        // XDP/eBPF NSM path, so durability matters more than batching here.
        self.file.flush().await?;
        Ok(())
    }
}

#[derive(Clone)]
struct PostgresSinkHandle;

struct PostgresSink {
    pool: sqlx::PgPool,
    batch: Vec<HoneypotEvent>,
    batch_size: usize,
    flush_interval: std::time::Duration,
    last_flush: tokio::time::Instant,
}

impl PostgresSink {
    async fn connect(cfg: crate::config::PostgresSinkConfig) -> Result<Self> {
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(5)
            .connect(&cfg.connection_string)
            .await
            .context("connecting to postgres")?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS honeypot_events (
                event_id UUID PRIMARY KEY,
                ts TIMESTAMPTZ NOT NULL,
                source_ip TEXT,
                source_port INTEGER,
                host_id TEXT NOT NULL,
                sensor TEXT NOT NULL,
                confidence TEXT NOT NULL,
                severity TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload JSONB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_honeypot_events_ts ON honeypot_events (ts);
            CREATE INDEX IF NOT EXISTS idx_honeypot_events_source_ip ON honeypot_events (source_ip);
            "#,
        )
        .execute(&pool)
        .await
        .context("creating honeypot_events table")?;

        Ok(Self {
            pool,
            batch: Vec::with_capacity(cfg.batch_size),
            batch_size: cfg.batch_size,
            flush_interval: std::time::Duration::from_millis(cfg.flush_interval_ms),
            last_flush: tokio::time::Instant::now(),
        })
    }

    async fn push(&mut self, event: HoneypotEvent) -> Result<()> {
        self.batch.push(event);
        if self.batch.len() >= self.batch_size || self.last_flush.elapsed() >= self.flush_interval
        {
            self.flush().await?;
        }
        Ok(())
    }

    async fn flush(&mut self) -> Result<()> {
        if self.batch.is_empty() {
            self.last_flush = tokio::time::Instant::now();
            return Ok(());
        }
        let mut tx = self.pool.begin().await?;
        for event in self.batch.drain(..) {
            let kind_label = kind_label(&event.kind);
            let payload = serde_json::to_value(&event)?;
            sqlx::query(
                r#"
                INSERT INTO honeypot_events
                    (event_id, ts, source_ip, source_port, host_id, sensor, confidence, severity, kind, payload)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (event_id) DO NOTHING
                "#,
            )
            .bind(event.event_id)
            .bind(event.timestamp)
            .bind(event.source_ip.map(|ip| ip.to_string()))
            .bind(event.source_port.map(|p| p as i32))
            .bind(&event.host_id)
            .bind(&event.sensor)
            .bind(format!("{:?}", event.confidence))
            .bind(format!("{:?}", event.severity))
            .bind(kind_label)
            .bind(payload)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        self.last_flush = tokio::time::Instant::now();
        Ok(())
    }
}

fn kind_label(kind: &crate::event::EventKind) -> &'static str {
    use crate::event::EventKind::*;
    match kind {
        ConnectionOpened { .. } => "connection_opened",
        SshAuthAttempt { .. } => "ssh_auth_attempt",
        SshCommandExecuted { .. } => "ssh_command_executed",
        RawPayload { .. } => "raw_payload",
        HoneytokenTriggered { .. } => "honeytoken_triggered",
        SessionClosed { .. } => "session_closed",
    }
}

/// Best-effort real-time forward over a Unix domain socket. Each call opens
/// a short-lived connection and writes one newline-delimited JSON event;
/// this keeps the sink stateless and safe to call from a hot path without
/// holding a long-lived connection open across sink restarts on the
/// receiving end (e.g. the correlation engine restarting).
async fn forward_unix(path: &PathBuf, event: &HoneypotEvent) -> Result<()> {
    let mut stream = UnixStream::connect(path).await?;
    let mut line = serde_json::to_string(event)?;
    line.push('\n');
    stream.write_all(line.as_bytes()).await?;
    Ok(())
}
