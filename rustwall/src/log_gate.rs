use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Caps how many "policy decision" lines we emit per second. Without this, a
/// SYN flood or port scan turns every dropped packet into a JSON log line,
/// which can pin a CPU core on serialization/IO and fill disk faster than
/// the actual attack would otherwise matter -- the rate limiter in
/// rate_limit.rs protects the conntrack table, but nothing protected the
/// logger itself in the first pass of this engine. This does.
pub struct LogGate {
    max_per_sec: u64,
    window_start: Mutex<Instant>,
    emitted_this_window: AtomicU64,
    suppressed_this_window: AtomicU64,
}

pub enum GateResult {
    Emit,
    /// Caller should skip logging this event. The running suppressed count
    /// for the window is available via `take_suppressed_count`, so callers
    /// can periodically flush a "N events suppressed" summary line.
    Suppress,
}

impl LogGate {
    pub fn new(max_per_sec: u64) -> Self {
        Self {
            max_per_sec,
            window_start: Mutex::new(Instant::now()),
            emitted_this_window: AtomicU64::new(0),
            suppressed_this_window: AtomicU64::new(0),
        }
    }

    pub fn check(&self) -> GateResult {
        if self.max_per_sec == 0 {
            return GateResult::Emit; // 0 means "unlimited"
        }

        let mut window_start = self.window_start.lock().unwrap();
        if window_start.elapsed() >= Duration::from_secs(1) {
            *window_start = Instant::now();
            self.emitted_this_window.store(0, Ordering::Relaxed);
            self.suppressed_this_window.store(0, Ordering::Relaxed);
        }
        drop(window_start);

        if self.emitted_this_window.fetch_add(1, Ordering::Relaxed) < self.max_per_sec {
            GateResult::Emit
        } else {
            self.suppressed_this_window.fetch_add(1, Ordering::Relaxed);
            GateResult::Suppress
        }
    }

    /// Returns how many events were suppressed in the window that just
    /// elapsed, for periodic summary logging (call this from the same
    /// maintenance tick that sweeps conntrack).
    pub fn take_suppressed_count(&self) -> u64 {
        self.suppressed_this_window.swap(0, Ordering::Relaxed)
    }
}
