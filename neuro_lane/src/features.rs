//! Per-host, per-time-bin traffic features and their rolling history.

use std::collections::{HashMap, HashSet, VecDeque};
use std::net::IpAddr;
use std::time::SystemTime;

/// Channel order matches `llm_bridge::FEATURE_NAMES`.
pub const N_FEAT: usize = 6;
pub type FeatVec = [f64; N_FEAT];

#[derive(Default)]
struct BinAcc {
    pkts: u32,
    bytes: u64,
    empty: u32,
    dsts: HashSet<IpAddr>,
    ports: HashSet<u16>,
    last: Option<SystemTime>,
    gaps: Vec<f64>,
}

fn lnorm(n: usize, cap: f64) -> f64 {
    ((1.0 + n as f64).ln() / (1.0 + cap).ln()).min(1.0)
}

fn finalize(a: &BinAcc) -> FeatVec {
    if a.pkts == 0 {
        return [0.0; N_FEAT];
    }
    let n = a.pkts as f64;
    let regular = if a.gaps.len() >= 3 {
        let m = a.gaps.iter().sum::<f64>() / a.gaps.len() as f64;
        if m > 1e-9 {
            let var = a.gaps.iter().map(|g| (g - m).powi(2)).sum::<f64>() / a.gaps.len() as f64;
            (1.0 - var.sqrt() / m).clamp(0.0, 1.0)
        } else {
            0.0
        }
    } else {
        0.0
    };
    [
        lnorm(a.pkts as usize, 512.0),
        lnorm(a.dsts.len(), 64.0),
        lnorm(a.ports.len(), 64.0),
        (a.bytes as f64 / n / 1500.0).min(1.0),
        a.empty as f64 / n,
        regular,
    ]
}

/// Accumulates packets for the open bin and keeps the last `seq_len`
/// closed bins per source host. Memory is bounded by `max_hosts`.
pub struct HostBinner {
    cur: HashMap<IpAddr, BinAcc>,
    hist: HashMap<IpAddr, VecDeque<FeatVec>>,
    seq_len: usize,
    max_hosts: usize,
    pub dropped_new_hosts: u64,
}

impl HostBinner {
    pub fn new(seq_len: usize, max_hosts: usize) -> Self {
        Self { cur: HashMap::new(), hist: HashMap::new(), seq_len: seq_len.max(2), max_hosts: max_hosts.max(1), dropped_new_hosts: 0 }
    }

    pub fn observe(&mut self, src: IpAddr, dst: IpAddr, dst_port: u16, payload_len: usize, at: SystemTime) {
        if !self.cur.contains_key(&src) && self.cur.len() >= self.max_hosts {
            self.dropped_new_hosts += 1;
            return;
        }
        let a = self.cur.entry(src).or_default();
        a.pkts = a.pkts.saturating_add(1);
        a.bytes += payload_len as u64;
        if payload_len == 0 {
            a.empty += 1;
        }
        if a.dsts.len() < 4096 {
            a.dsts.insert(dst);
        }
        if a.ports.len() < 4096 {
            a.ports.insert(dst_port);
        }
        if let Some(prev) = a.last {
            if let Ok(d) = at.duration_since(prev) {
                if a.gaps.len() < 256 {
                    a.gaps.push(d.as_secs_f64());
                }
            }
        }
        a.last = Some(at);
    }

    /// Closes the open bin. Every tracked host gets one new entry (zeros if
    /// idle) so sequences stay time-aligned; hosts idle for a whole window
    /// are forgotten. Returns the hosts that were active in this bin.
    pub fn close_bin(&mut self) -> Vec<IpAddr> {
        let cur = std::mem::take(&mut self.cur);
        let active: Vec<IpAddr> = cur.keys().copied().collect();
        for (h, acc) in &cur {
            if !self.hist.contains_key(h) && self.hist.len() >= self.max_hosts {
                continue;
            }
            let q = self.hist.entry(*h).or_default();
            q.push_back(finalize(acc));
            while q.len() > self.seq_len {
                q.pop_front();
            }
        }
        let seq_len = self.seq_len;
        self.hist.retain(|h, q| {
            if !cur.contains_key(h) {
                q.push_back([0.0; N_FEAT]);
                while q.len() > seq_len {
                    q.pop_front();
                }
            }
            q.iter().any(|f| f.iter().any(|x| *x != 0.0))
        });
        active
    }

    pub fn sequence(&self, host: IpAddr) -> Option<Vec<FeatVec>> {
        self.hist.get(&host).map(|q| q.iter().copied().collect())
    }

    pub fn tracked_hosts(&self) -> usize {
        self.hist.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;
    use std::time::Duration;

    fn ip(n: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(10, 0, 0, n))
    }

    #[test]
    fn features_reflect_behaviour() {
        let mut b = HostBinner::new(8, 100);
        let t0 = SystemTime::UNIX_EPOCH + Duration::from_secs(1000);
        // host 1: periodic, one destination, small payload
        for i in 0..10 {
            b.observe(ip(1), ip(200), 443, 40, t0 + Duration::from_secs(i * 5));
        }
        // host 2: many ports, empty payloads
        for p in 0..40u16 {
            b.observe(ip(2), ip(201), 1000 + p, 0, t0 + Duration::from_millis(p as u64 * 37 % 11));
        }
        let act = b.close_bin();
        assert_eq!(act.len(), 2);
        let f1 = b.sequence(ip(1)).unwrap()[0];
        let f2 = b.sequence(ip(2)).unwrap()[0];
        assert!(f1[5] > 0.95, "periodic timing -> regularity {:?}", f1);
        assert!(f2[2] > f1[2] + 0.5, "port diversity");
        assert!((f2[4] - 1.0).abs() < 1e-9 && f1[4] == 0.0, "empty-payload share");
        assert!(f1.iter().chain(f2.iter()).all(|x| (0.0..=1.0).contains(x)));
    }

    #[test]
    fn idle_hosts_are_padded_then_forgotten_and_memory_is_bounded() {
        let mut b = HostBinner::new(4, 3);
        let t = SystemTime::UNIX_EPOCH;
        for h in 1..=10u8 {
            b.observe(ip(h), ip(9), 80, 10, t);
        }
        assert_eq!(b.dropped_new_hosts, 7);
        b.close_bin();
        assert_eq!(b.tracked_hosts(), 3);
        for _ in 0..4 {
            b.close_bin();
        }
        assert_eq!(b.tracked_hosts(), 0, "idle for a full window -> forgotten");
    }
}
