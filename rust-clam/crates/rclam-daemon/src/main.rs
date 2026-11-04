//! rclamd: async scanning daemon speaking a clamd-compatible protocol
//! subset (PING / VERSION / SCAN / CONTSCAN / SHUTDOWN / STATS).
//!
//! Transport is platform-dependent: a Unix domain socket on Unix (where
//! `tokio::net::UnixListener` exists), and a TCP loopback socket on Windows
//! (where it doesn't). Both paths share one generic connection handler --
//! `handle_conn` is written against `AsyncRead + AsyncWrite`, not against
//! either concrete stream type, so there's exactly one implementation of
//! the protocol regardless of transport.
//!
//! Unlike clamd's thread-per-connection model with a global signature-db
//! lock, each connection here gets its own tokio task and scans read
//! `Arc<Scanner>` concurrently without any lock at all -- the signature
//! engine is immutable after load, so concurrent readers need no
//! synchronization beyond the `Arc` refcount.
//!
//! Resilience note: the workspace profile uses `panic = "unwind"`
//! deliberately. If a single scan request somehow panics (e.g. an
//! unexpected bug in a signature or PE parser), tokio catches the panic at
//! the task boundary -- only that one connection is lost, the daemon
//! process keeps serving every other connection. `panic = "abort"` would
//! trade that resilience for a smaller binary, which is the wrong trade for
//! a long-running service that should stay up.
//!
//! Trust boundary note: socket permissions (`--socket-mode`) gate *who* can
//! connect; they say nothing about *what* a connected client may then ask
//! the daemon to scan. `allowlist::PathAllowlist` (populated from
//! `--allow-root`) is the layer that answers that second question, and
//! fails closed -- with no `--allow-root` configured, every SCAN/CONTSCAN
//! is refused rather than defaulting to "scan anything the daemon's own
//! user can read". `limits::PeerLimits` adds a second per-peer layer on top
//! of the existing global `--max-connections` semaphore so one noisy or
//! malicious client sharing a daemon with other tenants can't consume the
//! whole connection budget by itself.

mod allowlist;
mod limits;
mod metrics;

use allowlist::PathAllowlist;
use clap::Parser;
use limits::{Admission, PeerKey, PeerLimits};
use metrics::{Metrics, MetricsHandle};
use scanner_core::{GuardLimits, Scanner, Verdict};
use sig_engine::SignatureEngine;
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::Semaphore;

/// Hard ceiling on a single command line read from a client connection.
/// Without this, a client that never sends `\n` could make the daemon
/// buffer an unbounded amount of data per connection (a classic
/// slowloris-style resource-exhaustion vector against a local daemon).
const MAX_COMMAND_LINE_BYTES: usize = 8192;

#[derive(Parser, Debug)]
#[command(name = "rclamd")]
struct Args {
    #[arg(long)]
    ndb: Vec<PathBuf>,
    #[arg(long)]
    hdb: Vec<PathBuf>,

    /// Unix domain socket path (Unix only; ignored on Windows).
    #[cfg(unix)]
    #[arg(long, default_value = "/tmp/rclamd.sock")]
    socket: PathBuf,

    /// Permission bits (octal) applied to the Unix domain socket after
    /// binding. Defaults to owner-only (0600) -- clamd's own default of
    /// group-writable sockets has been a recurring source of local
    /// privilege-escalation footguns in deployments that got the
    /// surrounding group membership wrong, so the safer default is chosen
    /// here and left to the operator to loosen deliberately if needed.
    #[cfg(unix)]
    #[arg(long, default_value = "0600")]
    socket_mode: String,

    /// TCP port on 127.0.0.1 (Windows; also available on Unix if you'd
    /// rather use TCP there, via --tcp).
    #[arg(long, default_value_t = 3310)]
    port: u16,

    /// Force TCP transport even on Unix (default there is the domain socket).
    #[cfg(unix)]
    #[arg(long)]
    tcp: bool,

    /// Maximum number of scan requests handled concurrently across *all*
    /// clients. Additional connections are accepted but immediately told
    /// the server is busy and closed, rather than queued indefinitely --
    /// bounding worst-case latency and memory/fd use under load instead of
    /// degrading silently.
    #[arg(long, default_value_t = 64)]
    max_connections: usize,

    /// Maximum size, in bytes, of a single top-level file scanned directly.
    #[arg(long, default_value_t = 200 * 1024 * 1024)]
    max_file_size: u64,

    /// Maximum nested-archive recursion depth.
    #[arg(long, default_value_t = 16)]
    max_depth: u32,

    /// Root directory a client is allowed to ask the daemon to SCAN.
    /// Repeatable. A requested path is permitted only if it resolves
    /// (after canonicalization, so symlink escapes are caught) underneath
    /// one of these roots. If none are given, every SCAN/CONTSCAN is
    /// refused -- this is a fail-closed default, not an oversight, since a
    /// shared daemon with no configured roots has no safe default other
    /// than "scan nothing".
    #[arg(long = "allow-root")]
    allow_root: Vec<PathBuf>,

    /// Sustained new-connections-per-second budget for a single peer
    /// (identified by uid over a Unix socket, or by IP over TCP), before
    /// per-peer rate limiting kicks in. Independent of --max-connections,
    /// which bounds the daemon as a whole.
    #[arg(long, default_value_t = 5.0)]
    peer_rate_per_sec: f64,

    /// Burst capacity for the per-peer rate limiter -- how many
    /// connections a single peer may open back-to-back before the
    /// sustained rate above starts throttling it.
    #[arg(long, default_value_t = 20.0)]
    peer_burst: f64,

    /// Maximum connections a single peer may have open at once.
    #[arg(long, default_value_t = 8)]
    peer_max_concurrent: usize,

    /// Address to serve /healthz and /metrics (Prometheus text format) on.
    /// Bound to loopback by default; set to 0.0.0.0 deliberately if the
    /// scraper lives off-host. Pass an empty string to disable.
    #[arg(long, default_value = "127.0.0.1")]
    metrics_addr: String,

    #[arg(long, default_value_t = 9310)]
    metrics_port: u16,
}

fn init_logging() {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    init_logging();
    let args = Args::parse();

    let mut builder = SignatureEngine::builder();
    for p in &args.ndb {
        let text = std::fs::read_to_string(p).map_err(|e| {
            tracing::error!(file = %p.display(), error = %e, "cannot read ndb file");
            e
        })?;
        builder = builder.load_ndb(&text).map_err(|e| {
            tracing::error!(file = %p.display(), error = %e, "failed to load ndb signatures");
            std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
        })?;
    }
    for p in &args.hdb {
        let text = std::fs::read_to_string(p).map_err(|e| {
            tracing::error!(file = %p.display(), error = %e, "cannot read hdb file");
            e
        })?;
        builder = builder.load_hdb(&text).map_err(|e| {
            tracing::error!(file = %p.display(), error = %e, "failed to load hdb signatures");
            std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
        })?;
    }
    let engine = builder.build();
    tracing::info!(
        wildcard_sigs = engine.hex_sig_count(),
        hash_sigs = engine.hash_sig_count(),
        "signature database loaded"
    );

    let limits = GuardLimits {
        max_depth: args.max_depth,
        max_file_size: args.max_file_size,
        ..GuardLimits::default()
    };
    let scanner = Arc::new(Scanner::new(engine).with_limits(limits));
    let concurrency = Arc::new(Semaphore::new(args.max_connections));

    let allowlist = Arc::new(PathAllowlist::new(&args.allow_root).map_err(|e| {
        tracing::error!(error = %e, "invalid --allow-root configuration");
        e
    })?);
    if allowlist.is_empty() {
        tracing::warn!(
            "no --allow-root configured: every SCAN/CONTSCAN request will be refused until at least one is set"
        );
    } else {
        tracing::info!(roots = ?args.allow_root, "scan path allowlist active");
    }

    let peer_limits = Arc::new(PeerLimits::new(
        args.peer_rate_per_sec,
        args.peer_burst,
        args.peer_max_concurrent,
    ));

    let metrics = Arc::new(Metrics::default());
    if !args.metrics_addr.is_empty() {
        let ip: IpAddr = args.metrics_addr.parse().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("invalid --metrics-addr {}: {e}", args.metrics_addr),
            )
        })?;
        let handle = Arc::new(MetricsHandle {
            metrics: Arc::clone(&metrics),
            start: Instant::now(),
            queue_capacity: args.max_connections,
            queue_available: Arc::clone(&concurrency),
        });
        tokio::spawn(metrics::serve((ip, args.metrics_port), handle));
    }

    let ctx = Arc::new(ConnCtx {
        scanner,
        allowlist,
        peer_limits,
        metrics,
    });

    run_server(args, ctx, concurrency).await
}

/// Everything a connection handler needs, bundled so `spawn_conn` doesn't
/// grow an ever-longer parameter list as more cross-cutting concerns
/// (allowlisting, rate limiting, metrics) get added.
struct ConnCtx {
    scanner: Arc<Scanner>,
    allowlist: Arc<PathAllowlist>,
    peer_limits: Arc<PeerLimits>,
    metrics: Arc<Metrics>,
}

/// Waits for either Ctrl+C or (on Unix) SIGTERM, whichever comes first.
/// Used to break the accept loop for an orderly shutdown instead of the
/// process being killed mid-request by an unhandled signal.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => tracing::warn!(error = %e, "failed to install SIGTERM handler"),
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}

#[cfg(unix)]
async fn run_server(
    args: Args,
    ctx: Arc<ConnCtx>,
    concurrency: Arc<Semaphore>,
) -> std::io::Result<()> {
    use tokio::net::UnixListener;

    if args.tcp {
        return run_tcp(args.port, ctx, concurrency).await;
    }

    let _ = std::fs::remove_file(&args.socket);
    let listener = UnixListener::bind(&args.socket)?;

    let mode =
        u32::from_str_radix(args.socket_mode.trim_start_matches("0o"), 8).unwrap_or_else(|e| {
            tracing::warn!(
                value = %args.socket_mode,
                error = %e,
                "invalid --socket-mode, falling back to 0600"
            );
            0o600
        });
    if let Err(e) = std::fs::set_permissions(
        &args.socket,
        std::os::unix::fs::PermissionsExt::from_mode(mode),
    ) {
        tracing::warn!(error = %e, "failed to set socket permissions");
    }

    tracing::info!(socket = %args.socket.display(), mode = format!("{mode:o}"), "listening on unix socket");

    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let (stream, _addr) = accepted?;
                // SO_PEERCRED-derived uid identifies *which local user*
                // connected, independent of the fact that Unix sockets are
                // already access-controlled by filesystem permissions --
                // that's a coarser control (can this user open the socket
                // at all) than per-uid rate limiting (is this user, among
                // possibly several who can open it, currently flooding).
                let peer = stream
                    .peer_cred()
                    .map(|c| PeerKey::Uid(c.uid()))
                    .unwrap_or(PeerKey::Unknown);
                spawn_conn(stream, peer, Arc::clone(&ctx), Arc::clone(&concurrency));
            }
            _ = shutdown_signal() => {
                tracing::info!("shutdown signal received, stopping listener");
                let _ = std::fs::remove_file(&args.socket);
                return Ok(());
            }
        }
    }
}

#[cfg(windows)]
async fn run_server(
    args: Args,
    ctx: Arc<ConnCtx>,
    concurrency: Arc<Semaphore>,
) -> std::io::Result<()> {
    run_tcp(args.port, ctx, concurrency).await
}

async fn run_tcp(port: u16, ctx: Arc<ConnCtx>, concurrency: Arc<Semaphore>) -> std::io::Result<()> {
    use tokio::net::TcpListener;

    let addr = ("127.0.0.1", port);
    let listener = TcpListener::bind(addr).await?;
    tracing::info!(port, "listening on tcp 127.0.0.1");
    serve_tcp(listener, ctx, concurrency).await
}

/// The actual accept loop for the TCP transport, split out from `run_tcp`
/// so integration tests can bind an ephemeral port (`port = 0`), read back
/// the OS-assigned address, and drive real `TcpStream` connections through
/// exactly the same code path Windows uses exclusively (see the
/// `tcp_roundtrip` tests module at the bottom of this file) -- rather than
/// that path only ever being compiled on Windows CI and never functionally
/// exercised there.
async fn serve_tcp(
    listener: tokio::net::TcpListener,
    ctx: Arc<ConnCtx>,
    concurrency: Arc<Semaphore>,
) -> std::io::Result<()> {
    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let (stream, addr) = accepted?;
                stream.set_nodelay(true).ok();
                let peer = PeerKey::Ip(addr.ip());
                spawn_conn(stream, peer, Arc::clone(&ctx), Arc::clone(&concurrency));
            }
            _ = shutdown_signal() => {
                tracing::info!("shutdown signal received, stopping listener");
                return Ok(());
            }
        }
    }
}

/// Admits a connection under both the per-peer limiter and the global
/// concurrency semaphore, or -- if either is exhausted -- spawns a minimal
/// task that tells the client why and closes. Load-shedding immediately
/// like this keeps worst-case latency and resource use bounded under a
/// connection flood (global) or a single misbehaving tenant (per-peer),
/// instead of an unbounded backlog of parked tasks each holding a socket
/// fd.
fn spawn_conn<S>(stream: S, peer: PeerKey, ctx: Arc<ConnCtx>, concurrency: Arc<Semaphore>)
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    ctx.metrics
        .connections_total
        .fetch_add(1, Ordering::Relaxed);
    tokio::spawn(async move {
        let mut stream = stream;

        let peer_permit = match ctx.peer_limits.admit(peer).await {
            Admission::Allowed(permit) => permit,
            Admission::RateLimited => {
                ctx.metrics
                    .connections_rejected_rate_limited
                    .fetch_add(1, Ordering::Relaxed);
                let _ = stream
                    .write_all(b"ERROR rate limit exceeded, slow down\n")
                    .await;
                let _ = stream.flush().await;
                return;
            }
            Admission::TooManyConcurrent => {
                ctx.metrics
                    .connections_rejected_concurrency
                    .fetch_add(1, Ordering::Relaxed);
                let _ = stream
                    .write_all(b"ERROR too many concurrent connections from this peer\n")
                    .await;
                let _ = stream.flush().await;
                return;
            }
        };

        let global_permit = match concurrency.try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => {
                ctx.metrics
                    .connections_rejected_busy
                    .fetch_add(1, Ordering::Relaxed);
                let _ = stream
                    .write_all(b"ERROR server busy, try again later\n")
                    .await;
                let _ = stream.flush().await;
                return;
            }
        };

        if let Err(e) = handle_conn(stream, &ctx).await {
            tracing::warn!(error = %e, "connection error");
        }
        drop(global_permit);
        drop(peer_permit);
    });
}

/// One protocol implementation shared by every transport: reads a single
/// command line, dispatches it, writes a single response. Generic over any
/// `AsyncRead + AsyncWrite` stream so Unix sockets and TCP sockets go
/// through identical logic.
async fn handle_conn<S>(stream: S, ctx: &ConnCtx) -> std::io::Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let (read_half, mut write_half) = tokio::io::split(stream);
    let mut reader = BufReader::new(read_half);
    let mut line = String::new();
    // Cap the read at MAX_COMMAND_LINE_BYTES + 1: if we come back with
    // exactly that many bytes and still no trailing newline, the line was
    // too long rather than the peer having genuinely run out of data.
    let n = (&mut reader)
        .take(MAX_COMMAND_LINE_BYTES as u64 + 1)
        .read_line(&mut line)
        .await?;
    if n == 0 {
        return Ok(()); // client disconnected without sending anything
    }
    if !line.ends_with('\n') && line.len() > MAX_COMMAND_LINE_BYTES {
        write_half
            .write_all(b"ERROR command line too long\n")
            .await?;
        write_half.flush().await?;
        return Ok(());
    }
    let cmd = line.trim();

    let response = if cmd.eq_ignore_ascii_case("PING") {
        "PONG\n".to_string()
    } else if cmd.eq_ignore_ascii_case("VERSION") {
        format!(
            "rclamd {} / unified-hex-hash-engine\n",
            env!("CARGO_PKG_VERSION")
        )
    } else if cmd.eq_ignore_ascii_case("STATS") {
        render_stats(ctx)
    } else if cmd.eq_ignore_ascii_case("SHUTDOWN") {
        write_half.write_all(b"OK\n").await?;
        write_half.flush().await?;
        tracing::info!("SHUTDOWN command received");
        std::process::exit(0);
    } else if let Some(path) = cmd
        .strip_prefix("SCAN ")
        .or_else(|| cmd.strip_prefix("CONTSCAN "))
    {
        run_scan(ctx, path)
    } else {
        format!("ERROR unknown command: {cmd}\n")
    };

    write_half.write_all(response.as_bytes()).await?;
    write_half.flush().await?;
    Ok(())
}

fn render_stats(ctx: &ConnCtx) -> String {
    format!(
        "SCANS {}\nFILES {}\nINFECTED {}\nERRORS {}\nEND\n",
        ctx.metrics.scans_total.load(Ordering::Relaxed),
        ctx.metrics.files_scanned_total.load(Ordering::Relaxed),
        ctx.metrics.files_infected_total.load(Ordering::Relaxed),
        ctx.metrics.scan_errors_total.load(Ordering::Relaxed),
    )
}

fn run_scan(ctx: &ConnCtx, path: &str) -> String {
    ctx.metrics.scans_total.fetch_add(1, Ordering::Relaxed);

    let requested = PathBuf::from(path);
    let allowed_path = match ctx.allowlist.check(&requested) {
        Ok(canon) => canon,
        Err(_) => {
            ctx.metrics
                .connections_rejected_path
                .fetch_add(1, Ordering::Relaxed);
            // Deliberately generic: doesn't distinguish "outside the
            // allowlist" from "doesn't exist" from "no --allow-root
            // configured at all", to avoid handing an unprivileged client
            // a filesystem existence oracle. Operators can see the real
            // reason in the daemon's own logs / the rejected-path metric.
            return format!("{path}: ERROR access denied\n");
        }
    };

    let p = allowed_path;
    let reports = if p.is_dir() {
        ctx.scanner.scan_directory(&p)
    } else {
        match ctx.scanner.scan_path(&p) {
            Ok(r) => r,
            Err(e) => {
                ctx.metrics
                    .scan_errors_total
                    .fetch_add(1, Ordering::Relaxed);
                return format!("{path}: ERROR {e}\n");
            }
        }
    };

    let mut out = String::new();
    let mut infected = 0usize;
    for r in &reports {
        ctx.metrics
            .files_scanned_total
            .fetch_add(1, Ordering::Relaxed);
        match &r.verdict {
            Verdict::Clean => out.push_str(&format!("{}: OK\n", r.logical_path)),
            Verdict::Infected(dets) => {
                infected += 1;
                ctx.metrics
                    .files_infected_total
                    .fetch_add(1, Ordering::Relaxed);
                for d in dets {
                    out.push_str(&format!("{}: {} FOUND\n", r.logical_path, d.name));
                }
            }
            Verdict::Skipped { reason } => {
                out.push_str(&format!("{}: SKIPPED({reason})\n", r.logical_path));
            }
            Verdict::LimitExceeded { reason, .. } => {
                out.push_str(&format!("{}: LIMIT EXCEEDED({reason})\n", r.logical_path));
            }
        }
    }
    out.push_str(if infected > 0 {
        "SCAN SUMMARY: INFECTED\n"
    } else {
        "SCAN SUMMARY: OK\n"
    });
    out
}

/// Integration tests over the TCP transport -- the code path Windows uses
/// exclusively (there's no Unix-domain-socket equivalent there), and one
/// this codebase's own history notes as "compiled but never functionally
/// exercised." These bind an ephemeral port, drive real `TcpStream`
/// connections through `serve_tcp`/`spawn_conn`/`handle_conn` exactly as a
/// real client would, and run on every OS in CI (including
/// `windows-latest`), rather than the TCP branch only ever being compiled
/// there and never actually connected to.
#[cfg(test)]
mod tcp_roundtrip_tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    async fn test_ctx(allow_root: Option<&std::path::Path>) -> Arc<ConnCtx> {
        let engine = SignatureEngine::builder()
            .load_ndb("Sig.Test:74657374\n") // matches the bytes "test"
            .unwrap()
            .build();
        let scanner = Arc::new(Scanner::new(engine));
        let allowlist = Arc::new(
            PathAllowlist::new(
                &allow_root
                    .map(|p| vec![p.to_path_buf()])
                    .unwrap_or_default(),
            )
            .unwrap(),
        );
        Arc::new(ConnCtx {
            scanner,
            allowlist,
            peer_limits: Arc::new(PeerLimits::new(1000.0, 1000.0, 100)),
            metrics: Arc::new(Metrics::default()),
        })
    }

    /// Binds an ephemeral TCP port, spawns the real accept loop against it,
    /// and returns the address a test client should connect to.
    async fn spawn_test_server(ctx: Arc<ConnCtx>) -> std::net::SocketAddr {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        let concurrency = Arc::new(Semaphore::new(64));
        tokio::spawn(async move {
            let _ = serve_tcp(listener, ctx, concurrency).await;
        });
        addr
    }

    async fn roundtrip(addr: std::net::SocketAddr, command: &str) -> String {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream.write_all(command.as_bytes()).await.unwrap();
        stream.write_all(b"\n").await.unwrap();
        let mut buf = Vec::new();
        // The server closes the connection after one response, so reading
        // to EOF is the correct way to collect the full reply here.
        stream.read_to_end(&mut buf).await.unwrap();
        String::from_utf8(buf).unwrap()
    }

    #[tokio::test]
    async fn ping_over_real_tcp_socket() {
        let ctx = test_ctx(None).await;
        let addr = spawn_test_server(ctx).await;
        assert_eq!(roundtrip(addr, "PING").await, "PONG\n");
    }

    #[tokio::test]
    async fn version_over_real_tcp_socket() {
        let ctx = test_ctx(None).await;
        let addr = spawn_test_server(ctx).await;
        let resp = roundtrip(addr, "VERSION").await;
        assert!(
            resp.starts_with("rclamd "),
            "unexpected VERSION reply: {resp}"
        );
    }

    #[tokio::test]
    async fn scan_over_real_tcp_socket_finds_signature_when_path_allowed() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("payload.txt");
        std::fs::write(&file, b"contains test marker").unwrap();

        let ctx = test_ctx(Some(dir.path())).await;
        let addr = spawn_test_server(ctx).await;

        let resp = roundtrip(addr, &format!("SCAN {}", file.display())).await;
        assert!(
            resp.contains("Sig.Test FOUND"),
            "unexpected SCAN reply: {resp}"
        );
        assert!(resp.contains("SCAN SUMMARY: INFECTED"));
    }

    #[tokio::test]
    async fn scan_over_real_tcp_socket_denied_outside_allowlist() {
        let allowed_dir = tempfile::tempdir().unwrap();
        let other_dir = tempfile::tempdir().unwrap();
        let file = other_dir.path().join("payload.txt");
        std::fs::write(&file, b"contains test marker").unwrap();

        let ctx = test_ctx(Some(allowed_dir.path())).await;
        let addr = spawn_test_server(ctx).await;

        let resp = roundtrip(addr, &format!("SCAN {}", file.display())).await;
        assert_eq!(resp, format!("{}: ERROR access denied\n", file.display()));
    }

    #[tokio::test]
    async fn scan_over_real_tcp_socket_denied_with_no_allow_root_configured() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("payload.txt");
        std::fs::write(&file, b"contains test marker").unwrap();

        // No allow_root at all -- fail-closed default.
        let ctx = test_ctx(None).await;
        let addr = spawn_test_server(ctx).await;

        let resp = roundtrip(addr, &format!("SCAN {}", file.display())).await;
        assert_eq!(resp, format!("{}: ERROR access denied\n", file.display()));
    }

    #[tokio::test]
    async fn stats_over_real_tcp_socket_reflects_completed_scans() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("payload.txt");
        std::fs::write(&file, b"contains test marker").unwrap();

        let ctx = test_ctx(Some(dir.path())).await;
        let addr = spawn_test_server(ctx).await;

        roundtrip(addr, &format!("SCAN {}", file.display())).await;
        let stats = roundtrip(addr, "STATS").await;
        assert!(stats.contains("SCANS 1"), "unexpected STATS reply: {stats}");
        assert!(
            stats.contains("INFECTED 1"),
            "unexpected STATS reply: {stats}"
        );
    }

    #[tokio::test]
    async fn unknown_command_over_real_tcp_socket() {
        let ctx = test_ctx(None).await;
        let addr = spawn_test_server(ctx).await;
        let resp = roundtrip(addr, "BOGUS").await;
        assert!(resp.starts_with("ERROR unknown command"), "got: {resp}");
    }

    #[tokio::test]
    async fn oversized_command_line_is_rejected_not_hung() {
        let ctx = test_ctx(None).await;
        let addr = spawn_test_server(ctx).await;
        let huge = "SCAN ".to_string() + &"a".repeat(MAX_COMMAND_LINE_BYTES + 100);
        let resp = roundtrip(addr, &huge).await;
        assert_eq!(resp, "ERROR command line too long\n");
    }
}
