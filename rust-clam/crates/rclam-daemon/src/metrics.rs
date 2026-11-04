//! Minimal `/healthz` + Prometheus-text `/metrics` endpoint.
//!
//! Without this, an operator has logs but no signal an orchestrator can act
//! on programmatically: a crash gets noticed and restarted, but "the scan
//! queue is growing and requests are queuing up" produces no log line to
//! page on. This is a deliberately small hand-rolled HTTP/1.x responder
//! (parse the request line, ignore everything else, write a fixed response,
//! close the connection) rather than pulling in a full HTTP server
//! dependency -- a metrics/health endpoint has exactly two routes and is
//! polled by tools (Prometheus, `curl`, an orchestrator's liveness probe)
//! that all speak plain HTTP/1.0-compatible requests, so there is no
//! keep-alive, chunked encoding, or routing complexity actually needed
//! here.
//!
//! Bound to loopback by default and on a separate port from the scan
//! protocol -- metrics/health should stay reachable (and cheap to serve)
//! even under scan-side load, and shouldn't share a port with a protocol
//! that has nothing to do with HTTP.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[derive(Default)]
pub struct Metrics {
    pub connections_total: AtomicU64,
    pub connections_rejected_busy: AtomicU64,
    pub connections_rejected_rate_limited: AtomicU64,
    pub connections_rejected_concurrency: AtomicU64,
    pub connections_rejected_path: AtomicU64,
    pub scans_total: AtomicU64,
    pub files_scanned_total: AtomicU64,
    pub files_infected_total: AtomicU64,
    pub scan_errors_total: AtomicU64,
}

pub struct MetricsHandle {
    pub metrics: Arc<Metrics>,
    pub start: Instant,
    pub queue_capacity: usize,
    pub queue_available: Arc<tokio::sync::Semaphore>,
}

impl Metrics {
    fn render_prometheus(
        &self,
        uptime_secs: u64,
        queue_capacity: usize,
        queue_available: usize,
    ) -> String {
        let g = |name: &str, help: &str, value: u64, out: &mut String| {
            out.push_str(&format!(
                "# HELP {name} {help}\n# TYPE {name} counter\n{name} {value}\n"
            ));
        };
        let mut out = String::new();
        g(
            "rclamd_connections_total",
            "Total connections accepted",
            self.connections_total.load(Ordering::Relaxed),
            &mut out,
        );
        g(
            "rclamd_connections_rejected_busy_total",
            "Connections rejected because the global connection cap was reached",
            self.connections_rejected_busy.load(Ordering::Relaxed),
            &mut out,
        );
        g(
            "rclamd_connections_rejected_rate_limited_total",
            "Connections rejected by per-peer rate limiting",
            self.connections_rejected_rate_limited
                .load(Ordering::Relaxed),
            &mut out,
        );
        g(
            "rclamd_connections_rejected_concurrency_total",
            "Connections rejected by the per-peer concurrency cap",
            self.connections_rejected_concurrency
                .load(Ordering::Relaxed),
            &mut out,
        );
        g(
            "rclamd_connections_rejected_path_total",
            "SCAN requests rejected because the path was not in an allowed root",
            self.connections_rejected_path.load(Ordering::Relaxed),
            &mut out,
        );
        g(
            "rclamd_scans_total",
            "Total SCAN/CONTSCAN requests handled",
            self.scans_total.load(Ordering::Relaxed),
            &mut out,
        );
        g(
            "rclamd_files_scanned_total",
            "Total individual files scanned (including archive members)",
            self.files_scanned_total.load(Ordering::Relaxed),
            &mut out,
        );
        g(
            "rclamd_files_infected_total",
            "Total individual files that produced a signature match",
            self.files_infected_total.load(Ordering::Relaxed),
            &mut out,
        );
        g(
            "rclamd_scan_errors_total",
            "Total scan requests that errored",
            self.scan_errors_total.load(Ordering::Relaxed),
            &mut out,
        );
        out.push_str(&format!(
            "# HELP rclamd_uptime_seconds Seconds since daemon start\n# TYPE rclamd_uptime_seconds gauge\nrclamd_uptime_seconds {uptime_secs}\n"
        ));
        out.push_str(&format!(
            "# HELP rclamd_connection_queue_capacity Configured max_connections\n# TYPE rclamd_connection_queue_capacity gauge\nrclamd_connection_queue_capacity {queue_capacity}\n"
        ));
        out.push_str(&format!(
            "# HELP rclamd_connection_queue_available Free connection slots right now\n# TYPE rclamd_connection_queue_available gauge\nrclamd_connection_queue_available {queue_available}\n"
        ));
        out
    }
}

/// Runs the metrics/health HTTP listener until the process exits. Errors
/// binding the listener are logged and treated as non-fatal to the daemon
/// as a whole -- losing the metrics endpoint shouldn't take down scanning.
pub async fn serve(addr: (std::net::IpAddr, u16), handle: Arc<MetricsHandle>) {
    let listener = match TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!(error = %e, ?addr, "failed to bind metrics/health listener, continuing without it");
            return;
        }
    };
    tracing::info!(?addr, "metrics/health listening (/healthz, /metrics)");

    loop {
        let (mut stream, _) = match listener.accept().await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "metrics listener accept error");
                continue;
            }
        };
        let handle = Arc::clone(&handle);
        tokio::spawn(async move {
            let mut buf = [0u8; 2048];
            // Read once; a request line comfortably fits in one read for
            // any well-behaved client, and this endpoint doesn't need to
            // handle bodies or pipelining.
            let n = match stream.read(&mut buf).await {
                Ok(n) => n,
                Err(_) => return,
            };
            let req = String::from_utf8_lossy(&buf[..n]);
            let path = req
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .unwrap_or("/");

            let (status, body) = match path {
                "/healthz" => ("200 OK", "ok\n".to_string()),
                "/metrics" => {
                    let uptime = handle.start.elapsed().as_secs();
                    let available = handle.queue_available.available_permits();
                    (
                        "200 OK",
                        handle
                            .metrics
                            .render_prometheus(uptime, handle.queue_capacity, available),
                    )
                }
                _ => ("404 Not Found", "not found\n".to_string()),
            };

            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes()).await;
            let _ = stream.shutdown().await;
        });
    }
}
