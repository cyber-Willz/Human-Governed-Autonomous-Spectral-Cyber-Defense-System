use crate::quarantine::Quarantine;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Process-wide counters. Deliberately minimal -- a handful of atomics, not a
/// metrics framework -- because the point is operational visibility (can you
/// tell whether this box is dropping way more than it used to, without
/// grepping logs) not full observability tooling. A real deployment would
/// scrape this into Prometheus/Datadog/whatever already exists in the
/// environment rather than rely on this exporter long-term.
#[derive(Default)]
pub struct Metrics {
    pub accepted: AtomicU64,
    pub dropped: AtomicU64,
    pub rejected: AtomicU64,
    pub conntrack_established_hits: AtomicU64,
    pub conntrack_table_full: AtomicU64,
    pub parse_failures: AtomicU64,
    /// Non-first IP fragments dropped because they lack an L4 header to
    /// classify -- see packet::ParseOutcome::UnclassifiableFragment.
    /// Tracked separately from parse_failures because it usually means a
    /// netfilter-wiring problem (missing kernel-side defrag), not hostile
    /// or corrupt traffic.
    pub fragment_drops: AtomicU64,
    pub logs_suppressed: AtomicU64,
    /// Packets blocked because their source was already quarantined (the
    /// pre-rule-evaluation "quick block" check), as opposed to a fresh
    /// packet that matched a `drop`/`reject` rule.
    pub quarantine_blocks: AtomicU64,
    /// Number of times a rule's `auto_block_secs` triggered a new (or
    /// extended) quarantine ban.
    pub quarantine_bans: AtomicU64,
    /// Times a sync job (ban or unban) failed when the sync worker thread
    /// actually ran it against the OS firewall backend (e.g. `nft`/`netsh`
    /// returned an error). Distinct from jobs dropped for being enqueued
    /// while the sync queue was full -- see quarantine.sync_dropped(),
    /// exposed via the live quarantine reference in render(), not as its own
    /// atomic here.
    pub quarantine_sync_failures: AtomicU64,
    /// Number of manual unban requests received via the admin endpoint,
    /// regardless of whether the target IP was actually banned.
    pub quarantine_manual_unbans: AtomicU64,
    /// Requests to the metrics/control endpoint rejected for missing or
    /// incorrect bearer token, when `metrics_auth_token` is configured.
    pub auth_failures: AtomicU64,
}

impl Metrics {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    #[allow(clippy::too_many_arguments)]
    fn render(
        &self,
        active_flows: u64,
        rule_count: u64,
        quarantine_active: u64,
        quarantine_bans_total: u64,
        quarantine_capacity_rejections_total: u64,
        quarantine_sync_dropped_total: u64,
    ) -> String {
        let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
        format!(
            "# HELP rustwall_packets_accepted_total Packets accepted.\n\
             # TYPE rustwall_packets_accepted_total counter\n\
             rustwall_packets_accepted_total {}\n\
             # HELP rustwall_packets_dropped_total Packets dropped (silent).\n\
             # TYPE rustwall_packets_dropped_total counter\n\
             rustwall_packets_dropped_total {}\n\
             # HELP rustwall_packets_rejected_total Packets actively rejected (RST/ICMP).\n\
             # TYPE rustwall_packets_rejected_total counter\n\
             rustwall_packets_rejected_total {}\n\
             # HELP rustwall_conntrack_established_hits_total Packets fast-pathed via conntrack.\n\
             # TYPE rustwall_conntrack_established_hits_total counter\n\
             rustwall_conntrack_established_hits_total {}\n\
             # HELP rustwall_conntrack_table_full_total Times a new flow was dropped due to a full conntrack table.\n\
             # TYPE rustwall_conntrack_table_full_total counter\n\
             rustwall_conntrack_table_full_total {}\n\
             # HELP rustwall_parse_failures_total Packets that failed to parse and were fail-closed dropped.\n\
             # TYPE rustwall_parse_failures_total counter\n\
             rustwall_parse_failures_total {}\n\
             # HELP rustwall_fragment_drops_total Non-first IP fragments dropped for lacking a classifiable L4 header (see README Fragmentation section).\n\
             # TYPE rustwall_fragment_drops_total counter\n\
             rustwall_fragment_drops_total {}\n\
             # HELP rustwall_logs_suppressed_total Policy-decision log lines suppressed by the log rate limiter.\n\
             # TYPE rustwall_logs_suppressed_total counter\n\
             rustwall_logs_suppressed_total {}\n\
             # HELP rustwall_quarantine_blocks_total Packets blocked because their source was already quarantined.\n\
             # TYPE rustwall_quarantine_blocks_total counter\n\
             rustwall_quarantine_blocks_total {}\n\
             # HELP rustwall_quarantine_bans_total Times a rule's auto_block_secs triggered a new or extended ban.\n\
             # TYPE rustwall_quarantine_bans_total counter\n\
             rustwall_quarantine_bans_total {}\n\
             # HELP rustwall_quarantine_capacity_rejections_total Times a new IP could not be quarantined because the table was at capacity.\n\
             # TYPE rustwall_quarantine_capacity_rejections_total counter\n\
             rustwall_quarantine_capacity_rejections_total {}\n\
             # HELP rustwall_quarantine_sync_failures_total Times the OS firewall sync worker failed to apply a ban/unban (e.g. nft/netsh returned an error).\n\
             # TYPE rustwall_quarantine_sync_failures_total counter\n\
             rustwall_quarantine_sync_failures_total {}\n\
             # HELP rustwall_quarantine_sync_dropped_total Times a ban/unban sync job was dropped because the sync worker's queue was full.\n\
             # TYPE rustwall_quarantine_sync_dropped_total counter\n\
             rustwall_quarantine_sync_dropped_total {}\n\
             # HELP rustwall_quarantine_manual_unbans_total Manual unban requests received via the admin endpoint.\n\
             # TYPE rustwall_quarantine_manual_unbans_total counter\n\
             rustwall_quarantine_manual_unbans_total {}\n\
             # HELP rustwall_auth_failures_total Requests rejected for missing/incorrect bearer token.\n\
             # TYPE rustwall_auth_failures_total counter\n\
             rustwall_auth_failures_total {}\n\
             # HELP rustwall_conntrack_active_flows Current number of tracked flows.\n\
             # TYPE rustwall_conntrack_active_flows gauge\n\
             rustwall_conntrack_active_flows {}\n\
             # HELP rustwall_quarantine_active_hosts Current number of quarantined source IPs.\n\
             # TYPE rustwall_quarantine_active_hosts gauge\n\
             rustwall_quarantine_active_hosts {}\n\
             # HELP rustwall_rules_loaded Number of policy rules currently active.\n\
             # TYPE rustwall_rules_loaded gauge\n\
             rustwall_rules_loaded {}\n",
            g(&self.accepted),
            g(&self.dropped),
            g(&self.rejected),
            g(&self.conntrack_established_hits),
            g(&self.conntrack_table_full),
            g(&self.parse_failures),
            g(&self.fragment_drops),
            g(&self.logs_suppressed),
            g(&self.quarantine_blocks),
            quarantine_bans_total,
            quarantine_capacity_rejections_total,
            g(&self.quarantine_sync_failures),
            quarantine_sync_dropped_total,
            g(&self.quarantine_manual_unbans),
            g(&self.auth_failures),
            active_flows,
            quarantine_active,
            rule_count,
        )
    }
}

/// A minimal parsed HTTP/1.x request: method, path, and bearer token if an
/// `Authorization: Bearer <token>` header was present. Deliberately not a
/// general-purpose HTTP parser -- this endpoint serves exactly two routes,
/// so we only extract what routing and auth actually need.
struct ParsedRequest {
    method: String,
    path: String,
    bearer_token: Option<String>,
}

fn parse_request(raw: &str) -> Option<ParsedRequest> {
    let mut lines = raw.split("\r\n");
    let request_line = lines.next()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();

    let mut bearer_token = None;
    for line in lines {
        if line.is_empty() {
            break; // end of headers
        }
        if let Some((name, value)) = line.split_once(':') {
            if name.trim().eq_ignore_ascii_case("authorization") {
                let value = value.trim();
                if let Some(token) = value.strip_prefix("Bearer ") {
                    bearer_token = Some(token.trim().to_string());
                }
            }
        }
    }

    Some(ParsedRequest {
        method,
        path,
        bearer_token,
    })
}

/// Constant-time comparison so token checking doesn't leak timing
/// information about how many leading bytes matched -- a short-circuiting
/// `==` on a bearer token is exactly the kind of thing that looks harmless
/// and isn't, on an endpoint whose entire job is gatekeeping.
fn tokens_match(provided: &str, expected: &str) -> bool {
    let a = provided.as_bytes();
    let b = expected.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn write_response(stream: &mut impl Write, status: &str, body: &str) {
    let response = format!(
        "HTTP/1.1 {}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status,
        body.len(),
        body
    );
    let _ = stream.write_all(response.as_bytes());
}

/// Serves the metrics/control endpoint on `addr` until `running` flips
/// false. Blocking, single-threaded accept loop -- scraping and occasional
/// admin unban calls are low-frequency and low-concurrency, so this doesn't
/// need the complexity of an async runtime or thread pool.
///
/// Routes:
///   GET  /metrics                     -- Prometheus-format counters
///   POST /quarantine/unban/<ip>       -- removes <ip> from quarantine now
///
/// If `auth_token` is `Some`, both routes require a matching
/// `Authorization: Bearer <token>` header; missing/incorrect tokens get 401.
#[allow(clippy::too_many_arguments)]
pub fn serve(
    addr: std::net::SocketAddr,
    metrics: Arc<Metrics>,
    active_flows_fn: impl Fn() -> u64 + Send + 'static,
    rule_count_fn: impl Fn() -> u64 + Send + 'static,
    quarantine: Arc<Quarantine>,
    auth_token: Option<String>,
    running: Arc<std::sync::atomic::AtomicBool>,
) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    listener.set_nonblocking(true)?;
    if auth_token.is_none() {
        tracing::warn!(
            %addr,
            "metrics/control endpoint listening WITHOUT authentication -- set metrics_auth_token \
             if this is reachable from anywhere beyond a fully trusted loopback"
        );
    } else {
        tracing::info!(%addr, "metrics/control endpoint listening (bearer token required)");
    }

    while running.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((mut stream, _)) => {
                let mut buf = [0u8; 4096];
                let n = stream.read(&mut buf).unwrap_or(0);
                let raw = String::from_utf8_lossy(&buf[..n]);

                let Some(req) = parse_request(&raw) else {
                    write_response(&mut stream, "400 Bad Request", "malformed request\n");
                    continue;
                };

                if let Some(expected) = &auth_token {
                    let authorized = req
                        .bearer_token
                        .as_deref()
                        .map(|t| tokens_match(t, expected))
                        .unwrap_or(false);
                    if !authorized {
                        metrics.auth_failures.fetch_add(1, Ordering::Relaxed);
                        write_response(&mut stream, "401 Unauthorized", "missing or invalid bearer token\n");
                        continue;
                    }
                }

                match (req.method.as_str(), req.path.as_str()) {
                    ("GET", "/metrics") => {
                        let body = metrics.render(
                            active_flows_fn(),
                            rule_count_fn(),
                            quarantine.active_count() as u64,
                            quarantine.bans_total(),
                            quarantine.capacity_rejections(),
                            quarantine.sync_dropped(),
                        );
                        write_response(&mut stream, "200 OK", &body);
                    }
                    ("POST", path) if path.starts_with("/quarantine/unban/") => {
                        let ip_str = &path["/quarantine/unban/".len()..];
                        match ip_str.parse::<std::net::IpAddr>() {
                            Ok(ip) => {
                                metrics
                                    .quarantine_manual_unbans
                                    .fetch_add(1, Ordering::Relaxed);
                                if quarantine.manual_unban(ip) {
                                    tracing::info!(%ip, "quarantine ban removed via admin endpoint");
                                    write_response(&mut stream, "200 OK", "unbanned\n");
                                } else {
                                    write_response(&mut stream, "404 Not Found", "that IP was not quarantined\n");
                                }
                            }
                            Err(_) => {
                                write_response(&mut stream, "400 Bad Request", "invalid IP address\n");
                            }
                        }
                    }
                    _ => {
                        write_response(&mut stream, "404 Not Found", "no such route\n");
                    }
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            Err(e) => {
                tracing::warn!(error = %e, "metrics listener accept error");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_get_metrics_with_no_auth_header() {
        let req = parse_request("GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        assert_eq!(req.method, "GET");
        assert_eq!(req.path, "/metrics");
        assert_eq!(req.bearer_token, None);
    }

    #[test]
    fn parses_bearer_token_case_insensitive_header_name() {
        let req = parse_request(
            "GET /metrics HTTP/1.1\r\nauthorization: Bearer secret123\r\n\r\n",
        )
        .unwrap();
        assert_eq!(req.bearer_token.as_deref(), Some("secret123"));
    }

    #[test]
    fn parses_post_unban_path() {
        let req = parse_request(
            "POST /quarantine/unban/203.0.113.5 HTTP/1.1\r\nAuthorization: Bearer tok\r\n\r\n",
        )
        .unwrap();
        assert_eq!(req.method, "POST");
        assert_eq!(req.path, "/quarantine/unban/203.0.113.5");
        assert_eq!(req.bearer_token.as_deref(), Some("tok"));
    }

    #[test]
    fn malformed_request_line_fails_to_parse() {
        assert!(parse_request("").is_none());
        // A single token with no path at all should fail -- there's no
        // second whitespace-separated part for `path`.
        assert!(parse_request("GET\r\n").is_none());
    }

    #[test]
    fn tokens_match_requires_exact_equality() {
        assert!(tokens_match("abc123", "abc123"));
        assert!(!tokens_match("abc123", "abc124"));
        assert!(!tokens_match("abc123", "abc12"));
        assert!(!tokens_match("", "x"));
        assert!(tokens_match("", ""));
    }
}
