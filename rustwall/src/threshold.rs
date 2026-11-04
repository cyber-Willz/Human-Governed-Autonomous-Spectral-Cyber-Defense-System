use dashmap::DashMap;
use std::net::IpAddr;
use std::time::{Duration, Instant};

/// Tracks, per (rule, source IP), how many times a quarantine-triggering
/// rule has matched within a rolling window, and reports whether that count
/// has reached the configured threshold.
///
/// This exists because a single-shot "one match = instant ban" design (the
/// original `auto_block_secs` behavior) is trivially triggered by exactly
/// one spoofed packet -- there is no way for rustwall to distinguish a real
/// attacker from a forged packet on a single observation, because the
/// ban-triggering event IS the first (and typically only) packet seen for
/// that flow. Requiring repeated matches within a window doesn't eliminate
/// spoofing (an attacker can still send N spoofed packets instead of one),
/// but it closes the trivial single-packet case and matches how real
/// systems in this space actually handle it -- fail2ban's `maxretry`,
/// Suricata's `threshold` rules -- rather than pretending a cleverer
/// single-packet heuristic (e.g. "only trust packets with ACK set") would
/// help, which it wouldn't: an attacker spoofing one packet controls every
/// flag on it.
pub struct ThresholdTracker {
    // Keyed by (rule index, source IP) so different rules track separately
    // even if they'd otherwise see the same source.
    counts: DashMap<(usize, IpAddr), (u32, Instant)>,
}

impl ThresholdTracker {
    pub fn new() -> Self {
        Self {
            counts: DashMap::new(),
        }
    }

    /// Records one match for (rule_idx, ip) and returns true if the running
    /// count within `window` has now reached `threshold`. A `threshold` of
    /// 1 preserves the original single-shot behavior exactly (every call
    /// returns true), so this is backward compatible by default.
    pub fn record_and_check(
        &self,
        rule_idx: usize,
        ip: IpAddr,
        threshold: u32,
        window: Duration,
    ) -> bool {
        if threshold <= 1 {
            return true; // fast path: no window tracking needed at all
        }

        let now = Instant::now();
        let mut entry = self
            .counts
            .entry((rule_idx, ip))
            .or_insert((0, now));

        if now.duration_since(entry.1) > window {
            // Window expired since the last match; start a fresh window.
            entry.0 = 0;
            entry.1 = now;
        }

        entry.0 += 1;
        let reached = entry.0 >= threshold;
        if reached {
            // Reset so a sustained attacker has to build back up to
            // threshold again for the next ban, rather than firing on
            // every single subsequent packet once past the line.
            entry.0 = 0;
            entry.1 = now;
        }
        reached
    }

    /// Bounds memory the same way rate_limit.rs's buckets do: drop entries
    /// that haven't been touched in a while so a scan-and-move-on attacker
    /// doesn't leave permanent per-IP state behind.
    pub fn sweep(&self, max_idle: Duration) {
        let now = Instant::now();
        self.counts
            .retain(|_, (_, last_touched)| now.duration_since(*last_touched) < max_idle);
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
    fn threshold_of_one_fires_immediately_preserving_old_behavior() {
        let t = ThresholdTracker::new();
        assert!(t.record_and_check(0, ip(10, 0, 0, 1), 1, Duration::from_secs(60)));
    }

    #[test]
    fn threshold_of_three_requires_three_matches() {
        let t = ThresholdTracker::new();
        let addr = ip(10, 0, 0, 1);
        assert!(!t.record_and_check(0, addr, 3, Duration::from_secs(60)));
        assert!(!t.record_and_check(0, addr, 3, Duration::from_secs(60)));
        assert!(t.record_and_check(0, addr, 3, Duration::from_secs(60)));
    }

    #[test]
    fn a_single_spoofed_packet_cannot_trigger_a_threshold_greater_than_one() {
        // The exact scenario this exists to close: one packet, one match.
        let t = ThresholdTracker::new();
        assert!(!t.record_and_check(0, ip(203, 0, 113, 5), 5, Duration::from_secs(60)));
    }

    #[test]
    fn different_rules_track_the_same_source_independently() {
        let t = ThresholdTracker::new();
        let addr = ip(10, 0, 0, 1);
        assert!(!t.record_and_check(0, addr, 2, Duration::from_secs(60)));
        // A different rule (index 1) matching the same source starts its
        // own count from zero, not inheriting rule 0's count of 1.
        assert!(!t.record_and_check(1, addr, 2, Duration::from_secs(60)));
        assert!(t.record_and_check(0, addr, 2, Duration::from_secs(60)));
    }

    #[test]
    fn window_expiry_resets_the_count_instead_of_accumulating_forever() {
        let t = ThresholdTracker::new();
        let addr = ip(10, 0, 0, 1);
        assert!(!t.record_and_check(0, addr, 3, Duration::from_millis(20)));
        std::thread::sleep(Duration::from_millis(30));
        // Window has expired; this should start a fresh count of 1, not 2.
        assert!(!t.record_and_check(0, addr, 3, Duration::from_millis(20)));
        assert!(!t.record_and_check(0, addr, 3, Duration::from_millis(20)));
        assert!(t.record_and_check(0, addr, 3, Duration::from_millis(20)));
    }

    #[test]
    fn reaching_threshold_resets_count_so_next_ban_requires_full_threshold_again() {
        let t = ThresholdTracker::new();
        let addr = ip(10, 0, 0, 1);
        assert!(!t.record_and_check(0, addr, 2, Duration::from_secs(60)));
        assert!(t.record_and_check(0, addr, 2, Duration::from_secs(60)));
        // Immediately after firing, count should be back to 0 -- one more
        // match should NOT fire again until threshold is rebuilt.
        assert!(!t.record_and_check(0, addr, 2, Duration::from_secs(60)));
    }

    #[test]
    fn sweep_removes_idle_entries() {
        let t = ThresholdTracker::new();
        t.record_and_check(0, ip(10, 0, 0, 1), 5, Duration::from_secs(60));
        std::thread::sleep(Duration::from_millis(10));
        t.sweep(Duration::from_millis(1));
        assert_eq!(t.counts.len(), 0);
    }
}
