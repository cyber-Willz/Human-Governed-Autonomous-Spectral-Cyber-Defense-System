use crate::chan::{self, Sender};
use crate::metrics::Metrics;
use crate::os_firewall::OsFirewallSync;
use std::net::IpAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tracing::warn;

/// A unit of work for the OS-firewall sync worker thread. Deliberately tiny
/// and owned (no borrows) so it can cross the channel cleanly.
pub enum SyncJob {
    Ban(IpAddr, Duration),
    Unban(IpAddr),
}

/// How many pending sync jobs we'll buffer before dropping new ones. This is
/// a deliberate, bounded queue, not an unbounded one -- an unbounded queue
/// backed by a stuck subprocess is just the same memory-exhaustion problem
/// as an unbounded quarantine table, one layer down. Capacity is generous
/// relative to expected ban rates (a real attack triggering thousands of
/// *new* auto-blocks per second is already a bigger problem than this queue
/// can solve), so drops here should be rare in practice and are counted via
/// metrics when they do happen.
const SYNC_QUEUE_CAPACITY: usize = 4096;

/// Spawns the dedicated sync worker thread and returns a `Sender` for
/// enqueueing ban/unban jobs. The worker owns `backend` and `recv()`s in a
/// blocking loop -- blocking here is exactly correct, since this thread's
/// only job is to serialize subprocess calls to `nft`/`netsh` one at a time,
/// off the NFQUEUE packet path and off the conntrack/quarantine maintenance
/// sweep entirely. A hung `nft` call now stalls only this one thread, not
/// packet processing.
pub fn spawn(
    backend: Arc<dyn OsFirewallSync>,
    metrics: Arc<Metrics>,
    running: Arc<AtomicBool>,
) -> Sender<SyncJob> {
    let (tx, rx) = chan::bounded(SYNC_QUEUE_CAPACITY);

    std::thread::spawn(move || {
        // rx.recv() blocks until a job arrives or every Sender is dropped
        // (which happens at process shutdown when Quarantine and the
        // maintenance thread's clones are gone) -- no polling, no busy loop.
        while let Ok(job) = rx.recv() {
            if !running.load(Ordering::Relaxed) {
                break;
            }
            let result = match job {
                SyncJob::Ban(ip, duration) => backend.sync_ban(ip, duration),
                SyncJob::Unban(ip) => backend.sync_unban(ip),
            };
            if let Err(e) = result {
                metrics
                    .quarantine_sync_failures
                    .fetch_add(1, Ordering::Relaxed);
                warn!(backend = backend.name(), error = %e, "OS firewall sync job failed");
            }
        }
    });

    tx
}
