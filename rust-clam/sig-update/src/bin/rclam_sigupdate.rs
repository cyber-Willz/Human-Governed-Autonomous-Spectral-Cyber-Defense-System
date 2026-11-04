//! `rclam-sigupdate`: a small, `freshclam`-equivalent CLI. Fetches a
//! manifest (optionally checking an Ed25519 signature over it), downloads
//! every listed file, verifies each one's SHA-256 against the manifest,
//! stages them all in a temp directory, and only then atomically swaps
//! that staging directory into place as the live signature directory --
//! see `sig_update::apply` for why the swap is structured that way.
//!
//! This intentionally does not run as a daemon/timer itself; run it from
//! cron/systemd-timer, same as `freshclam` traditionally is. It exits
//! non-zero on any verification failure, so a scheduler treats a bad update
//! as a failed job rather than silently leaving stale-but-safe signatures
//! in place (which is what happens automatically anyway, since nothing is
//! touched in `sig_dir` until every check passes).

use clap::Parser;
use sig_update::{apply_update, parse_manifest, verify_manifest_signature, verify_sha256};
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "rclam-sigupdate")]
struct Args {
    /// URL of the JSON manifest listing signature files and their SHA-256s.
    #[arg(long)]
    manifest_url: String,

    /// Live signature directory rclamd reads from. Replaced atomically on
    /// success; the prior generation is kept as `<sig_dir>.previous`.
    #[arg(long)]
    sig_dir: PathBuf,

    /// Hex-encoded Ed25519 public key. If given, the manifest fetch must
    /// also find a signature at `<manifest_url>.sig` (64 raw bytes,
    /// hex-encoded) and it must verify, or the update is refused. Without
    /// this flag, integrity relies on TLS-on-the-manifest-fetch plus the
    /// per-file SHA-256 checks alone -- fine against a flaky mirror,
    /// weaker against a fully compromised origin.
    #[arg(long)]
    public_key: Option<String>,

    /// Print what would happen without touching sig_dir.
    #[arg(long)]
    dry_run: bool,
}

fn fetch(url: &str) -> Result<Vec<u8>, String> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("GET {url} failed: {e}"))?;
    let mut buf = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut buf)
        .map_err(|e| format!("reading response body from {url} failed: {e}"))?;
    Ok(buf)
}

fn run(args: Args) -> Result<(), String> {
    let manifest_bytes = fetch(&args.manifest_url)?;

    if let Some(pubkey) = &args.public_key {
        let sig_url = format!("{}.sig", args.manifest_url);
        let sig_bytes = fetch(&sig_url)?;
        let sig_hex = String::from_utf8_lossy(&sig_bytes).trim().to_string();
        verify_manifest_signature(&manifest_bytes, pubkey, &sig_hex)
            .map_err(|e| format!("manifest signature verification failed: {e}"))?;
        println!("manifest signature OK ({sig_url})");
    } else {
        eprintln!(
            "warning: no --public-key given; manifest integrity relies on TLS alone, not on an independent signature"
        );
    }

    let manifest_text =
        String::from_utf8(manifest_bytes).map_err(|_| "manifest is not valid UTF-8".to_string())?;
    let manifest = parse_manifest(&manifest_text).map_err(|e| e.to_string())?;
    println!(
        "manifest version {} lists {} file(s)",
        manifest.version,
        manifest.files.len()
    );

    if args.dry_run {
        for f in &manifest.files {
            println!("  would fetch {} <- {}", f.name, f.url);
        }
        println!("dry run: sig_dir not touched");
        return Ok(());
    }

    let staging = tempfile::Builder::new()
        .prefix(".rclam-sigupdate-staging-")
        .tempdir_in(
            args.sig_dir
                .parent()
                .unwrap_or_else(|| std::path::Path::new(".")),
        )
        .map_err(|e| format!("creating staging directory failed: {e}"))?;

    for f in &manifest.files {
        let data = fetch(&f.url)?;
        verify_sha256(&data, &f.sha256)
            .map_err(|e| format!("{}: integrity check failed: {e}", f.name))?;
        let out_path = staging.path().join(&f.name);
        let mut file = std::fs::File::create(&out_path)
            .map_err(|e| format!("writing {}: {e}", out_path.display()))?;
        file.write_all(&data)
            .map_err(|e| format!("writing {}: {e}", out_path.display()))?;
        println!("  verified and staged {}", f.name);
    }

    // Record the version alongside the files themselves so it moves
    // atomically with everything else in the same swap, rather than being
    // written to sig_dir in a separate step that could observe a
    // half-updated state if this process is killed mid-way.
    std::fs::write(
        staging.path().join(".rclam-sigversion"),
        format!("{}\n", manifest.version),
    )
    .map_err(|e| format!("writing version file: {e}"))?;

    // `apply_update` renames (moves) the staging directory into place, so
    // hand it the path and let the TempDir guard go out of scope without
    // trying to clean up a directory that's already been moved elsewhere.
    let staging_path = staging.keep();
    apply_update(&staging_path, &args.sig_dir).map_err(|e| e.to_string())?;

    println!(
        "signature update applied: {} now at version {}",
        args.sig_dir.display(),
        manifest.version
    );
    Ok(())
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(args) {
        eprintln!("rclam-sigupdate: error: {e}");
        std::process::exit(1);
    }
}
