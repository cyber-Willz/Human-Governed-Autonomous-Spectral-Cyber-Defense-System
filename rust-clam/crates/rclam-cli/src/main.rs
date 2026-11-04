use clap::Parser;
use scanner_core::{GuardLimits, Scanner, Verdict};
use sig_engine::SignatureEngine;
use std::path::PathBuf;
use std::process::ExitCode;

/// Exit code conventions follow ClamAV's `clamscan`: 0 clean, 1 infected
/// (or a limit was hit while detections were already found), 2 for any
/// operational error (bad signature file, unreadable path, etc). Scripts
/// and CI pipelines rely on this distinction, so it's treated as part of
/// the CLI's stable contract, not just a human-readable detail.
const EXIT_CLEAN: u8 = 0;
const EXIT_INFECTED: u8 = 1;
const EXIT_ERROR: u8 = 2;

#[derive(Parser, Debug)]
#[command(
    name = "rclam",
    about = "Rust malware scanner: hash + wildcard signature engine"
)]
struct Args {
    /// Path to a .ndb wildcard-signature database file
    #[arg(long)]
    ndb: Vec<PathBuf>,

    /// Path to a .hdb hash-signature database file
    #[arg(long)]
    hdb: Vec<PathBuf>,

    /// Maximum size, in bytes, of a single top-level file that will be
    /// scanned directly. Files larger than this are skipped (not scanned)
    /// rather than causing the whole invocation to hang or exhaust memory.
    #[arg(long, default_value_t = 200 * 1024 * 1024)]
    max_file_size: u64,

    /// Maximum nested-archive recursion depth (zip-in-zip, gzip-in-gzip, ...).
    #[arg(long, default_value_t = 16)]
    max_depth: u32,

    /// File or directory to scan
    path: PathBuf,
}

fn init_logging() {
    use tracing_subscriber::EnvFilter;
    // Diagnostics go to stderr via `tracing`, controllable with RUST_LOG
    // (e.g. `RUST_LOG=debug`); scan results themselves are printed to
    // stdout via `println!` below since that's the tool's actual output
    // and needs to stay stable/parseable independent of log verbosity.
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}

fn main() -> ExitCode {
    init_logging();
    let args = Args::parse();

    let mut builder = SignatureEngine::builder();
    for p in &args.ndb {
        match std::fs::read_to_string(p) {
            Ok(text) => match builder.load_ndb(&text) {
                Ok(b) => builder = b,
                Err(e) => {
                    tracing::error!(file = %p.display(), error = %e, "failed to load ndb signatures");
                    return ExitCode::from(EXIT_ERROR);
                }
            },
            Err(e) => {
                tracing::error!(file = %p.display(), error = %e, "cannot read ndb file");
                return ExitCode::from(EXIT_ERROR);
            }
        }
    }
    for p in &args.hdb {
        match std::fs::read_to_string(p) {
            Ok(text) => match builder.load_hdb(&text) {
                Ok(b) => builder = b,
                Err(e) => {
                    tracing::error!(file = %p.display(), error = %e, "failed to load hdb signatures");
                    return ExitCode::from(EXIT_ERROR);
                }
            },
            Err(e) => {
                tracing::error!(file = %p.display(), error = %e, "cannot read hdb file");
                return ExitCode::from(EXIT_ERROR);
            }
        }
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
    let scanner = Scanner::new(engine).with_limits(limits);

    let reports = if args.path.is_dir() {
        scanner.scan_directory(&args.path)
    } else {
        match scanner.scan_path(&args.path) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(path = %args.path.display(), error = %e, "scan error");
                return ExitCode::from(EXIT_ERROR);
            }
        }
    };

    let mut infected_count = 0usize;
    let mut error_count = 0usize;
    for report in &reports {
        match &report.verdict {
            Verdict::Clean => {
                println!("{}: OK", report.logical_path);
            }
            Verdict::Infected(dets) => {
                infected_count += 1;
                for d in dets {
                    println!(
                        "{}: FOUND {} ({:?}{})",
                        report.logical_path,
                        d.name,
                        d.kind,
                        d.offset.map(|o| format!(" @0x{o:x}")).unwrap_or_default()
                    );
                }
            }
            Verdict::Skipped { reason } => {
                println!("{}: SKIPPED ({reason})", report.logical_path);
            }
            Verdict::LimitExceeded { detections, reason } => {
                println!("{}: LIMIT EXCEEDED ({reason})", report.logical_path);
                if !detections.is_empty() {
                    infected_count += 1;
                } else {
                    error_count += 1;
                }
            }
        }
        if let Some(h) = &report.pe_heuristics {
            if h.score > 0 {
                println!(
                    "{}: HEURISTIC score={} high_entropy={:?} rwx_sections={:?} ep_outside={} few_sections={}",
                    report.logical_path,
                    h.score,
                    h.high_entropy_sections,
                    h.writable_and_executable_sections,
                    h.entry_point_outside_sections,
                    h.suspiciously_few_sections
                );
            }
        }
    }

    println!(
        "----------- SCAN SUMMARY -----------\nfiles scanned: {}\ninfected: {}\nerrors: {}",
        reports.len(),
        infected_count,
        error_count
    );

    if infected_count > 0 {
        ExitCode::from(EXIT_INFECTED)
    } else if error_count > 0 {
        ExitCode::from(EXIT_ERROR)
    } else {
        ExitCode::from(EXIT_CLEAN)
    }
}
