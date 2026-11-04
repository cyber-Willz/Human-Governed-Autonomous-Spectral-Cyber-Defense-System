//! Honeytoken watcher.
//!
//! Provisions decoy files (fake credentials, fake backups, fake configs)
//! at startup, then watches them -- and any additional configured
//! directories -- for access via `notify` (inotify on Linux). Any access
//! to a decoy file is inherently anomalous: no legitimate process in
//! `active_spectral_defense` has a reason to ever read these paths, so
//! this is the highest signal-to-noise source in the whole honeypot
//! suite and should be wired to the `neural_hmm` belief filter as a
//! near-instant high-confidence emission (see architecture notes in
//! sink.rs's module doc).
//!
//! Note: inotify reports OPEN/ACCESS/CLOSE events, not "read" specifically
//! at the byte level -- that's a kernel-level limitation, not a gap in
//! this implementation. For byte-level read auditing you'd need eBPF
//! (which you already have infrastructure for in `nsm-xdp`); this module
//! intentionally stays in userspace/inotify for simplicity and portability.

use crate::config::HoneytokenConfig;
use crate::event::{Confidence, EventKind, HoneypotEvent, HoneytokenKind, Severity};
use anyhow::{Context, Result};
use notify::{Event, EventKind as NotifyEventKind, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::sync::mpsc::Sender;
use tracing::{info, warn};

pub async fn provision_and_watch(
    cfg: HoneytokenConfig,
    host_id: String,
    tx: Sender<HoneypotEvent>,
) -> Result<()> {
    let mut token_paths: HashMap<PathBuf, String> = HashMap::new();

    for decoy in &cfg.decoy_files {
        if let Some(parent) = decoy.path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("creating parent dir for decoy file {}", decoy.path.display())
            })?;
        }
        if !decoy.path.exists() {
            std::fs::write(&decoy.path, &decoy.content)
                .with_context(|| format!("writing decoy file {}", decoy.path.display()))?;
            // Restrict permissions so the file *looks* like a real secret
            // worth protecting (0600), which also makes an attacker's
            // successful read a slightly stronger signal of privilege.
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let perms = std::fs::Permissions::from_mode(0o600);
                std::fs::set_permissions(&decoy.path, perms).ok();
            }
            info!(path = %decoy.path.display(), token_id = %decoy.token_id, "provisioned decoy file");
        }
        token_paths.insert(decoy.path.clone(), decoy.token_id.clone());
    }

    if token_paths.is_empty() && cfg.watch_paths.is_empty() {
        info!("honeytoken watcher: nothing configured, skipping");
        return Ok(());
    }

    let (notify_tx, mut notify_rx) = tokio::sync::mpsc::channel::<notify::Result<Event>>(256);

    let mut watcher = notify::recommended_watcher(move |res| {
        // notify's callback runs on its own thread; forward into async
        // land via a std blocking_send-equivalent (try_send is fine since
        // this channel only needs to carry filesystem event bursts, and
        // drops here are acceptable -- we still catch repeated access).
        let _ = notify_tx.blocking_send(res);
    })
    .context("creating filesystem watcher")?;

    for path in token_paths.keys() {
        watcher
            .watch(path, RecursiveMode::NonRecursive)
            .with_context(|| format!("watching decoy file {}", path.display()))?;
    }
    for dir in &cfg.watch_paths {
        watcher
            .watch(dir, RecursiveMode::Recursive)
            .with_context(|| format!("watching directory {}", dir.display()))?;
        info!(dir = %dir.display(), "honeytoken watcher: watching directory recursively");
    }

    info!(count = token_paths.len(), "honeytoken watcher active");

    // Keep the watcher alive for the process lifetime by moving it into
    // this loop's scope.
    let _watcher = watcher;

    while let Some(res) = notify_rx.recv().await {
        match res {
            Ok(event) => handle_fs_event(event, &token_paths, &host_id, &tx),
            Err(e) => warn!(error = %e, "honeytoken watcher: notify error"),
        }
    }

    Ok(())
}

fn handle_fs_event(
    event: Event,
    token_paths: &HashMap<PathBuf, String>,
    host_id: &str,
    tx: &Sender<HoneypotEvent>,
) {
    let interesting = matches!(
        event.kind,
        NotifyEventKind::Access(_) | NotifyEventKind::Modify(_) | NotifyEventKind::Other
    );
    if !interesting {
        return;
    }

    for path in &event.paths {
        let token_id = token_paths
            .get(path)
            .cloned()
            .unwrap_or_else(|| format!("watched_dir:{}", path.display()));

        let honeypot_event = HoneypotEvent::new(
            host_id,
            "honeytoken_watcher",
            None,
            None,
            Confidence::Certain,
            Severity::Critical,
            EventKind::HoneytokenTriggered {
                token_id,
                token_kind: HoneytokenKind::DecoyFile,
                access_detail: format!("{:?} on {}", event.kind, path.display()),
            },
        );
        if let Err(e) = tx.try_send(honeypot_event) {
            warn!(error = %e, "honeytoken watcher: event bus full or closed, dropping event");
        }
    }
}
