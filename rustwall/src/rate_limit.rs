use dashmap::DashMap;
use std::net::IpAddr;
use std::time::Instant;

/// Classic token bucket: refills at `rate` tokens/sec up to `burst` capacity.
/// One bucket per (source IP, rule) so a single rule's rate_limit_pps applies
/// independently of other rules -- this is what stops a single scanning host
/// from exhausting the connection table (a cheap analogue of FortiGate's
/// per-policy "session rate limiting" / PAN-OS zone protection profiles).
struct Bucket {
    tokens: f64,
    last_refill: Instant,
}

pub struct RateLimiter {
    buckets: DashMap<(IpAddr, u32), Bucket>,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            buckets: DashMap::new(),
        }
    }

    /// Returns true if the packet should be allowed under the given rule's
    /// rate limit; false if it should be dropped as excess.
    pub fn allow(&self, src: IpAddr, rule_id: u32, rate_pps: u32) -> bool {
        if rate_pps == 0 {
            return true; // no limit configured
        }
        let now = Instant::now();
        let burst = (rate_pps as f64).max(1.0);
        let mut bucket = self
            .buckets
            .entry((src, rule_id))
            .or_insert_with(|| Bucket {
                tokens: burst,
                last_refill: now,
            });

        let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * rate_pps as f64).min(burst);
        bucket.last_refill = now;

        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Periodic cleanup so long-idle source IPs don't accumulate forever.
    pub fn sweep(&self, max_idle: std::time::Duration) {
        let now = Instant::now();
        self.buckets
            .retain(|_, b| now.duration_since(b.last_refill) < max_idle);
    }
}
