use crate::chan::{Sender, TrySendError};
use crate::sync_worker::SyncJob;
use dashmap::DashMap;
use std::net::IpAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tracing::warn;

/// Canonical ban key: IPv4-mapped IPv6 -> IPv4; other IPv6 -> its /64 (an
/// attacker owning a /64 can rotate through 2^64 addresses, so a per-address
/// ban is meaningless); IPv4 unchanged. Must stay identical to
/// `nsm::detect::bounded::canon_src` (parity test in `ips_control_plane`).
/// Note: OS-firewall sync still receives the *exact* address; the prefix
/// semantics apply to rustwall's own in-process quarantine check.
pub fn ban_key(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V4(_) => ip,
        IpAddr::V6(a) => {
            if let Some(v4) = a.to_ipv4_mapped() {
                return IpAddr::V4(v4);
            }
            let mut o = a.octets();
            for b in &mut o[8..] {
                *b = 0;
            }
            IpAddr::V6(o.into())
        }
    }
}

/// Exact key: only unwraps IPv4-mapped IPv6.
fn exact_key(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V6(a) => a.to_ipv4_mapped().map(IpAddr::V4).unwrap_or(ip),
        v4 => v4,
    }
}

/// A dynamic, TTL-based ban list checked *before* normal rule evaluation --
/// the same "quick block, evaluated first" pattern as pfSense's floating
/// quick rules, and functionally what Untangle's behavioral quarantine and
/// pfBlocker-NG's dynamic tables do on top of pfSense/OPNsense. A source
/// that trips a rule with `auto_block_secs` set gets cut off wholesale for
/// the ban duration, not just refused on that one connection attempt.
///
/// Deliberately separate from `Engine`: Engine is rebuilt wholesale on every
/// SIGHUP config reload, but quarantine state is *runtime* state earned by
/// observed behavior, not configuration -- it must survive a rule reload the
/// same way the conntrack table does. This mirrors the existing
/// engine/conntrack split rather than inventing a new pattern.
///
/// Bounded the same way ConnTrack is: without a cap, a source that can
/// trigger `auto_block_secs` on demand (see the spoofing caveat in
/// rustwall.example.toml) could grow this table without limit and turn a
/// feature meant to protect the box into a memory-exhaustion vector against
/// it. Capacity is checked on insert of a *new* IP only -- extending an
/// existing ban never fails, since that doesn't grow the table.
///
/// Optionally pushes bans into the host's own firewall (nftables on Linux,
/// Windows Firewall via netsh where applicable) by enqueueing onto a bounded
/// channel (see chan.rs / sync_worker.rs) read by a dedicated background
/// thread. This is deliberately NOT a direct call to the OS firewall backend
/// from here: `ban()` and `sweep_expired()` run on the packet-processing and
/// maintenance threads respectively, and a `nft`/`netsh` subprocess call has
/// no bounded latency -- blocking either of those threads on it (as an
/// earlier version of this code did) turns a defense-in-depth feature into
/// an availability bug. `try_send` is used, never blocking `send`: if the
/// sync worker's queue is momentarily full, the sync job is dropped (counted
/// via metrics) rather than blocking here -- rustwall's own in-process
/// quarantine check is unaffected either way, since OS sync is additive, not
/// load-bearing for correctness.
pub struct Quarantine {
    table: DashMap<IpAddr, Instant>,
    max_entries: usize,
    bans_total: AtomicU64,
    capacity_rejections: AtomicU64,
    sync_dropped: AtomicU64,
    os_sync: Option<Sender<SyncJob>>,
}

impl Quarantine {
    pub fn new(max_entries: usize) -> Self {
        Self {
            table: DashMap::new(),
            max_entries,
            bans_total: AtomicU64::new(0),
            capacity_rejections: AtomicU64::new(0),
            sync_dropped: AtomicU64::new(0),
            os_sync: None,
        }
    }

    pub fn with_os_sync(max_entries: usize, os_sync: Sender<SyncJob>) -> Self {
        Self {
            table: DashMap::new(),
            max_entries,
            bans_total: AtomicU64::new(0),
            capacity_rejections: AtomicU64::new(0),
            sync_dropped: AtomicU64::new(0),
            os_sync: Some(os_sync),
        }
    }

    fn enqueue_sync(&self, job: SyncJob) {
        let Some(tx) = &self.os_sync else { return };
        if let Err(TrySendError::Full(_)) = tx.try_send(job) {
            self.sync_dropped.fetch_add(1, Ordering::Relaxed);
            warn!(
                "OS firewall sync queue full, dropping a ban/unban sync job \
                 (rustwall's own in-process quarantine is unaffected)"
            );
        }
        // A `Disconnected` try_send error means the sync worker thread has
        // exited (e.g. during shutdown) -- nothing to log per-event for
        // that; it would just be noise on every ban during teardown.
    }

    /// Bans `ip` for `duration` from now. If the IP is already banned, this
    /// extends the ban rather than shortening it -- repeated bad behavior
    /// during an existing ban should not reset to a shorter remaining time.
    ///
    /// Returns `true` if the ban was recorded, `false` if the table is at
    /// capacity and this is a new IP that couldn't be admitted. A `false`
    /// return does NOT mean the packet that triggered this call was allowed
    /// through -- the rule that set `auto_block_secs` already produced a
    /// `drop`/`reject` verdict for that packet independently. It only means
    /// *future* packets from this source won't get the fast pre-rule
    /// quarantine block; they'll still be evaluated against the normal
    /// ruleset, which most likely drops them again anyway.
    pub fn ban(&self, ip: IpAddr, duration: Duration) -> bool {
        self.ban_key_inner(ban_key(ip), ip, duration)
    }

    /// Bans exactly this address (IPv6: the /128, not the /64). For the rare
    /// human-approved ban of one host inside an operator-protected prefix.
    pub fn ban_exact(&self, ip: IpAddr, duration: Duration) -> bool {
        self.ban_key_inner(exact_key(ip), ip, duration)
    }

    fn ban_key_inner(&self, ip_key: IpAddr, orig: IpAddr, duration: Duration) -> bool {
        let ip = ip_key;
        let new_expiry = Instant::now() + duration;

        if !self.table.contains_key(&ip) && self.table.len() >= self.max_entries {
            // Full. Before rejecting, (1) reclaim expired bans, then (2) make
            // room by displacing the ban that expires soonest -- but only if
            // the new ban clearly outlasts it. Rejecting outright let an attacker who
            // could provoke bans on forged addresses fill the table and then
            // be immune (red-team scenario B04).
            self.sweep_expired();
            if self.table.len() >= self.max_entries {
                let soonest = self.table.iter().min_by_key(|e| *e.value()).map(|e| (*e.key(), *e.value()));
                match soonest {
                    // "Outlasts" needs a real margin (5 % of the new ban, min 1 s),
                    // otherwise back-to-back bans of equal length would displace
                    // each other in FIFO order and a ban flood could churn out
                    // bans that are still doing their job.
                    Some((victim_ip, victim_expiry))
                        if new_expiry > victim_expiry + (duration / 20).max(Duration::from_secs(1)) =>
                    {
                        if self.table.remove(&victim_ip).is_some() {
                            self.enqueue_sync(SyncJob::Unban(victim_ip));
                        }
                    }
                    _ => {
                        self.capacity_rejections.fetch_add(1, Ordering::Relaxed);
                        return false;
                    }
                }
            }
        }

        self.table
            .entry(ip)
            .and_modify(|expiry| {
                if new_expiry > *expiry {
                    *expiry = new_expiry;
                }
            })
            .or_insert(new_expiry);
        self.bans_total.fetch_add(1, Ordering::Relaxed);

        self.enqueue_sync(SyncJob::Ban(orig, duration));

        true
    }

    /// Fast path: is this source currently quarantined? Checked on every
    /// packet before conntrack lookup or rule evaluation, so it needs to
    /// stay O(1).
    pub fn is_active(&self, ip: IpAddr) -> bool {
        let now = Instant::now();
        let live = |k: IpAddr| self.table.get(&k).map_or(false, |e| now < *e);
        live(ban_key(ip)) || live(exact_key(ip))
    }

    /// Removes `ip` from quarantine immediately, on demand -- the admin
    /// unban path (see the `/quarantine/unban/<ip>` endpoint in metrics.rs),
    /// as opposed to the automatic TTL expiry `sweep_expired` handles.
    /// Returns `true` if the IP was actually banned (and is now removed),
    /// `false` if it wasn't banned in the first place -- callers use this to
    /// distinguish "unbanned" from "nothing to unban" for the HTTP response.
    pub fn manual_unban(&self, ip: IpAddr) -> bool {
        let a = self.table.remove(&ban_key(ip)).is_some();
        let b = self.table.remove(&exact_key(ip)).is_some();
        let was_present = a || b;
        if was_present {
            self.enqueue_sync(SyncJob::Unban(ip));
        }
        was_present
    }

    /// Sweep expired entries. Run periodically from the same maintenance
    /// tick that sweeps conntrack. Expired entries are also enqueued for OS
    /// firewall unban if a sync worker is configured -- this matters most
    /// for the Windows backend, which has no native rule-expiry mechanism
    /// and relies entirely on this call to actually remove a stale block;
    /// the Linux nftables backend's sets use the `timeout` flag and expire
    /// themselves kernel-side regardless, so this is a courtesy there.
    pub fn sweep_expired(&self) {
        let now = Instant::now();
        let mut expired = Vec::new();
        self.table.retain(|ip, expiry| {
            let keep = now < *expiry;
            if !keep {
                expired.push(*ip);
            }
            keep
        });

        for ip in expired {
            self.enqueue_sync(SyncJob::Unban(ip));
        }
    }

    pub fn active_count(&self) -> usize {
        self.table.len()
    }

    pub fn bans_total(&self) -> u64 {
        self.bans_total.load(Ordering::Relaxed)
    }

    pub fn capacity_rejections(&self) -> u64 {
        self.capacity_rejections.load(Ordering::Relaxed)
    }

    pub fn sync_dropped(&self) -> u64 {
        self.sync_dropped.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn ip(a: u8, b: u8, c: u8, d: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(a, b, c, d))
    }

    #[test]
    fn unbanned_ip_is_not_active() {
        let q = Quarantine::new(1000);
        assert!(!q.is_active(ip(10, 0, 0, 1)));
    }

    #[test]
    fn manual_unban_removes_an_active_ban_immediately() {
        let q = Quarantine::new(1000);
        q.ban(ip(10, 0, 0, 1), Duration::from_secs(3600));
        assert!(q.is_active(ip(10, 0, 0, 1)));

        assert!(q.manual_unban(ip(10, 0, 0, 1)));
        assert!(!q.is_active(ip(10, 0, 0, 1)));
    }

    #[test]
    fn manual_unban_of_a_non_banned_ip_returns_false() {
        let q = Quarantine::new(1000);
        assert!(!q.manual_unban(ip(10, 0, 0, 1)));
    }

    #[test]
    fn manual_unban_enqueues_a_sync_job_when_os_sync_configured() {
        let (tx, rx) = crate::chan::bounded(8);
        let q = Quarantine::with_os_sync(1000, tx);
        q.ban(ip(10, 0, 0, 1), Duration::from_secs(3600));
        let _ = rx.try_recv(); // drain the Ban job

        assert!(q.manual_unban(ip(10, 0, 0, 1)));
        match rx.try_recv().expect("an unban sync job should have been enqueued") {
            crate::sync_worker::SyncJob::Unban(unbanned_ip) => {
                assert_eq!(unbanned_ip, ip(10, 0, 0, 1));
            }
            _ => panic!("expected an Unban job"),
        }
    }

    #[test]
    fn banned_ip_is_active_until_expiry() {
        let q = Quarantine::new(1000);
        assert!(q.ban(ip(10, 0, 0, 1), Duration::from_secs(60)));
        assert!(q.is_active(ip(10, 0, 0, 1)));
        assert!(!q.is_active(ip(10, 0, 0, 2)));
    }

    #[test]
    fn re_ban_extends_rather_than_shortens() {
        let q = Quarantine::new(1000);
        q.ban(ip(10, 0, 0, 1), Duration::from_secs(60));
        // A second, shorter ban should not shrink the remaining time.
        q.ban(ip(10, 0, 0, 1), Duration::from_millis(1));
        assert!(q.is_active(ip(10, 0, 0, 1)));
    }

    #[test]
    fn sweep_removes_expired_entries() {
        let q = Quarantine::new(1000);
        q.ban(ip(10, 0, 0, 1), Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(10));
        assert!(!q.is_active(ip(10, 0, 0, 1)));
        q.sweep_expired();
        assert_eq!(q.active_count(), 0);
    }

    #[test]
    fn bans_total_counts_every_ban_call_not_unique_ips() {
        let q = Quarantine::new(1000);
        q.ban(ip(10, 0, 0, 1), Duration::from_secs(60));
        q.ban(ip(10, 0, 0, 1), Duration::from_secs(60));
        q.ban(ip(10, 0, 0, 2), Duration::from_secs(60));
        assert_eq!(q.bans_total(), 3);
        assert_eq!(q.active_count(), 2);
    }

    #[test]
    fn new_ip_rejected_once_table_is_at_capacity() {
        let q = Quarantine::new(2);
        assert!(q.ban(ip(10, 0, 0, 1), Duration::from_secs(60)));
        assert!(q.ban(ip(10, 0, 0, 2), Duration::from_secs(60)));
        // Table is full at 2/2 -- a third, never-seen IP must be rejected.
        assert!(!q.ban(ip(10, 0, 0, 3), Duration::from_secs(60)));
        assert!(!q.is_active(ip(10, 0, 0, 3)));
        assert_eq!(q.active_count(), 2);
        assert_eq!(q.capacity_rejections(), 1);
    }

    #[test]
    fn extending_an_existing_ban_never_fails_on_capacity() {
        let q = Quarantine::new(1);
        assert!(q.ban(ip(10, 0, 0, 1), Duration::from_secs(60)));
        // Table is at its cap of 1, but re-banning the SAME ip must still
        // succeed since it doesn't grow the table.
        assert!(q.ban(ip(10, 0, 0, 1), Duration::from_secs(120)));
        assert_eq!(q.active_count(), 1);
        assert_eq!(q.capacity_rejections(), 0);
    }

    #[test]
    fn ban_enqueues_a_sync_job_when_os_sync_configured() {
        let (tx, rx) = crate::chan::bounded(8);
        let q = Quarantine::with_os_sync(1000, tx);

        assert!(q.ban(ip(10, 0, 0, 1), Duration::from_secs(60)));

        match rx.try_recv().expect("a sync job should have been enqueued") {
            crate::sync_worker::SyncJob::Ban(banned_ip, duration) => {
                assert_eq!(banned_ip, ip(10, 0, 0, 1));
                assert_eq!(duration, Duration::from_secs(60));
            }
            _ => panic!("expected a Ban job"),
        }
    }

    #[test]
    fn sweep_expired_enqueues_unban_job_when_os_sync_configured() {
        let (tx, rx) = crate::chan::bounded(8);
        let q = Quarantine::with_os_sync(1000, tx);

        q.ban(ip(10, 0, 0, 1), Duration::from_millis(1));
        // Drain the Ban job so the next try_recv definitely gets the Unban.
        let _ = rx.try_recv();

        std::thread::sleep(Duration::from_millis(10));
        q.sweep_expired();

        match rx.try_recv().expect("an unban sync job should have been enqueued") {
            crate::sync_worker::SyncJob::Unban(unbanned_ip) => {
                assert_eq!(unbanned_ip, ip(10, 0, 0, 1));
            }
            _ => panic!("expected an Unban job"),
        }
    }

    #[test]
    fn full_sync_queue_never_blocks_or_fails_the_ban_itself() {
        // Capacity 1, and we never drain it -- every enqueue attempt after
        // the first must find the queue full. This is the core guarantee
        // this whole channel wiring exists for: rustwall's own quarantine
        // enforcement must be completely unaffected by OS-sync backpressure.
        let (tx, _rx) = crate::chan::bounded(1);
        let q = Quarantine::with_os_sync(1000, tx);

        assert!(q.ban(ip(10, 0, 0, 1), Duration::from_secs(60)));
        assert!(q.ban(ip(10, 0, 0, 2), Duration::from_secs(60)));
        assert!(q.ban(ip(10, 0, 0, 3), Duration::from_secs(60)));

        // All three bans succeeded at the Quarantine level regardless of
        // the 1-slot sync queue filling up after the first.
        assert!(q.is_active(ip(10, 0, 0, 1)));
        assert!(q.is_active(ip(10, 0, 0, 2)));
        assert!(q.is_active(ip(10, 0, 0, 3)));
        assert!(q.sync_dropped() >= 1);
    }

    #[test]
    fn full_table_reclaims_expired_bans_before_rejecting() {
        let q = Quarantine::new(2);
        assert!(q.ban(ip(10, 0, 0, 1), Duration::from_millis(1)));
        assert!(q.ban(ip(10, 0, 0, 2), Duration::from_secs(3600)));
        std::thread::sleep(Duration::from_millis(10));
        // table is "full" but one entry is dead: the new ban must fit
        assert!(q.ban(ip(10, 0, 0, 3), Duration::from_secs(60)));
        assert!(q.is_active(ip(10, 0, 0, 3)));
        assert!(q.is_active(ip(10, 0, 0, 2)));
        assert_eq!(q.capacity_rejections(), 0);
    }

    #[test]
    fn full_table_displaces_soonest_expiring_for_a_longer_ban() {
        let q = Quarantine::new(2);
        assert!(q.ban(ip(10, 0, 0, 1), Duration::from_secs(60)));
        assert!(q.ban(ip(10, 0, 0, 2), Duration::from_secs(7200)));
        assert!(q.ban(ip(10, 0, 0, 3), Duration::from_secs(86400)));
        assert!(!q.is_active(ip(10, 0, 0, 1)), "shortest ban should have been displaced");
        assert!(q.is_active(ip(10, 0, 0, 2)));
        assert!(q.is_active(ip(10, 0, 0, 3)));
        assert_eq!(q.active_count(), 2);
    }

    #[test]
    fn full_table_rejects_a_ban_shorter_than_everything_held() {
        let q = Quarantine::new(2);
        assert!(q.ban(ip(10, 0, 0, 1), Duration::from_secs(86400)));
        assert!(q.ban(ip(10, 0, 0, 2), Duration::from_secs(86400)));
        assert!(!q.ban(ip(10, 0, 0, 3), Duration::from_secs(60)));
        assert_eq!(q.capacity_rejections(), 1);
        assert!(q.is_active(ip(10, 0, 0, 1)) && q.is_active(ip(10, 0, 0, 2)));
    }
}
