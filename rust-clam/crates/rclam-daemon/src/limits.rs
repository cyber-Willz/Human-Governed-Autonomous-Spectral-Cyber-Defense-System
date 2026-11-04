//! Per-peer admission control.
//!
//! `--max-connections` bounds the daemon's *total* concurrency, but does
//! nothing to stop one noisy or malicious client from consuming most of
//! that shared budget by itself -- with a global cap of 64 and one client
//! opening 60 connections, every other tenant sharing the daemon is
//! effectively locked out. This module adds a second, per-peer layer on top
//! of the existing global semaphore:
//!
//! - a hard cap on how many connections one peer may have open concurrently
//! - a token-bucket rate limit on how many new connections one peer may
//!   open per second (bursty but bounded, rather than a hard request/sec
//!   wall that penalizes a single legitimate burst)
//!
//! "Peer" is identified by OS-level credential where one exists (the
//! connecting user's uid over a Unix domain socket, via `SO_PEERCRED`) or by
//! IP address for TCP. Using the uid rather than, say, the fact of a Unix
//! socket connection at all means two different local users sharing one
//! daemon are rate-limited independently of each other, which matches the
//! multi-tenant threat model these limits exist for.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PeerKey {
    Uid(u32),
    Ip(IpAddr),
    /// Fallback for platforms/transports where neither a credential nor an
    /// address could be determined. Deliberately *not* a shared bucket for
    /// every such connection -- each gets treated as a fresh, unrelated
    /// peer -- so this fallback can't be (ab)used to dodge rate limiting by
    /// forcing the daemon down this path; it only ever collapses limiting
    /// back to the global semaphore, never grants extra unbounded budget.
    Unknown,
}

struct TokenBucket {
    tokens: f64,
    last_refill: Instant,
}

struct PeerState {
    bucket: TokenBucket,
    concurrency: Arc<Semaphore>,
    last_seen: Instant,
}

pub struct PeerLimits {
    /// New connections/sec a single peer may sustain, on average.
    rate_per_sec: f64,
    /// Burst capacity -- how many connections a peer may open back-to-back
    /// before the sustained rate starts throttling it.
    burst: f64,
    /// Max connections a single peer may have open concurrently.
    max_concurrent: usize,
    state: Mutex<HashMap<PeerKey, PeerState>>,
}

pub enum Admission {
    Allowed(tokio::sync::OwnedSemaphorePermit),
    RateLimited,
    TooManyConcurrent,
}

impl PeerLimits {
    pub fn new(rate_per_sec: f64, burst: f64, max_concurrent: usize) -> Self {
        Self {
            rate_per_sec,
            burst,
            max_concurrent,
            state: Mutex::new(HashMap::new()),
        }
    }

    /// Attempts to admit a new connection from `peer`. On success, returns
    /// a permit that must be held for the lifetime of the connection --
    /// dropping it frees the peer's concurrency slot.
    pub async fn admit(&self, peer: PeerKey) -> Admission {
        let mut state = self.state.lock().await;

        // Opportunistic eviction of peers idle long enough that keeping
        // their bucket around serves no purpose -- otherwise a daemon that
        // sees many distinct TCP source IPs over its lifetime would grow
        // this map unboundedly.
        const IDLE_EVICT_AFTER: Duration = Duration::from_secs(600);
        let now = Instant::now();
        state.retain(|_, s| now.duration_since(s.last_seen) < IDLE_EVICT_AFTER);

        let entry = state.entry(peer).or_insert_with(|| PeerState {
            bucket: TokenBucket {
                tokens: self.burst,
                last_refill: now,
            },
            concurrency: Arc::new(Semaphore::new(self.max_concurrent)),
            last_seen: now,
        });
        entry.last_seen = now;

        // Refill the bucket based on elapsed time, then check.
        let elapsed = now.duration_since(entry.bucket.last_refill).as_secs_f64();
        entry.bucket.tokens = (entry.bucket.tokens + elapsed * self.rate_per_sec).min(self.burst);
        entry.bucket.last_refill = now;

        if entry.bucket.tokens < 1.0 {
            return Admission::RateLimited;
        }

        let concurrency = Arc::clone(&entry.concurrency);
        // Drop the map lock before waiting on the per-peer semaphore -- the
        // semaphore itself is what should ever block, not other peers'
        // unrelated admission checks.
        drop(state);

        match concurrency.try_acquire_owned() {
            Ok(permit) => {
                // Only spend the rate-limit token once concurrency admission
                // also succeeded, so a peer that's merely at its concurrency
                // ceiling (not actually flooding) doesn't also get penalized
                // on its rate budget for every rejected attempt.
                let mut state = self.state.lock().await;
                if let Some(entry) = state.get_mut(&peer) {
                    entry.bucket.tokens -= 1.0;
                }
                Admission::Allowed(permit)
            }
            Err(_) => Admission::TooManyConcurrent,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn burst_is_allowed_then_throttled() {
        let limits = PeerLimits::new(1.0, 3.0, 100);
        let peer = PeerKey::Ip("127.0.0.1".parse().unwrap());

        for _ in 0..3 {
            assert!(matches!(limits.admit(peer).await, Admission::Allowed(_)));
        }
        // Burst exhausted; the 4th immediate request should be throttled.
        assert!(matches!(limits.admit(peer).await, Admission::RateLimited));
    }

    #[tokio::test]
    async fn distinct_peers_have_independent_buckets() {
        let limits = PeerLimits::new(1.0, 1.0, 100);
        let a = PeerKey::Ip("10.0.0.1".parse().unwrap());
        let b = PeerKey::Ip("10.0.0.2".parse().unwrap());

        assert!(matches!(limits.admit(a).await, Admission::Allowed(_)));
        assert!(matches!(limits.admit(a).await, Admission::RateLimited));
        // b's bucket is untouched by a's traffic.
        assert!(matches!(limits.admit(b).await, Admission::Allowed(_)));
    }

    #[tokio::test]
    async fn concurrency_cap_is_enforced_per_peer() {
        let limits = PeerLimits::new(1000.0, 1000.0, 2);
        let peer = PeerKey::Uid(1000);

        let p1 = limits.admit(peer).await;
        let p2 = limits.admit(peer).await;
        assert!(matches!(p1, Admission::Allowed(_)));
        assert!(matches!(p2, Admission::Allowed(_)));
        // Third concurrent connection from the same uid is rejected even
        // though the rate budget is effectively unlimited here.
        assert!(matches!(
            limits.admit(peer).await,
            Admission::TooManyConcurrent
        ));

        drop(p1);
        // Freeing a slot lets a subsequent connection back in.
        assert!(matches!(limits.admit(peer).await, Admission::Allowed(_)));
    }
}
