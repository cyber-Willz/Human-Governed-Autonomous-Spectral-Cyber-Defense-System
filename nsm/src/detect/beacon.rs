//! C2 beacon detector.
//!
//! Implants call home at a near-constant interval, usually with some
//! configured jitter. The original test (coefficient of variation <= 0.15
//! over >= 6 intervals, keyed per destination *port*) missed three evasions
//! the red-team harness reproduces: +/-30 % jitter, port-hopping, and a
//! single late/missed beat inflating the variance. Intervals are now judged
//! with three complementary tests over a per-(src,dst) history (ports are
//! deliberately *not* part of the key):
//!
//! 1. classic: cv <= 0.15 (>= 6 samples);
//! 2. robust: >= 10 samples, MAD/median <= 0.08 and >= 70 % of intervals
//!    within +/-15 % of the median -- tolerant of outlier beats;
//! 3. bounded jitter: >= 16 samples, p90/p10 <= 2.0 and cv <= 0.30. Uniform
//!    +/-30 % jitter has p90/p10 ~ 1.6; human or app-driven (Poisson-like)
//!    arrivals have a heavy tail with p90/p10 in the tens, so they do not
//!    qualify.
//!
//! Round-2 additions (adaptive adversary):
//!
//! 4. **wide jitter**: >= 32 samples, every interval within [0.12, 2.2] x the
//!    median and cv <= 0.55. A Cobalt-Strike-style +/-80 % uniform jitter has
//!    hard bounds ([0.2, 1.8] x base) that no human-driven arrival process has
//!    (log-normal / Poisson revisit times have tails far beyond 2.2 x median
//!    within 32 samples). Calibrated by `wide_jitter_calibration` below:
//!    ~99 % detection at +/-80 %, <1 % per-sequence false positives for
//!    log-normal (sigma = 1) and exponential revisit times.
//! 5. **destination pools**: history is *also* kept per (src, destination
//!    /24 | /48). An implant that rotates over a pool of addresses in one
//!    network (CDN / cloud front) never builds a per-address history; per-network
//!    it beats on schedule. Only evaluated once >= 3 distinct addresses of the
//!    network were seen, so a single stable server is not double-reported.
//!    Rotation across *unrelated* networks remains out of scope (harness K03).
//!
//! Connections closer than 1 s are one burst (a browser opening parallel
//! connections), not separate beats. Legitimate periodic software
//! (health checks, NTP, mail polling) will still match; that is why the
//! alert is Medium and only meaningful with corroboration.

use super::bounded::{make_room, DEFAULT_CAP};
use crate::alert::{Alert, Severity};
use crate::packet::{is_syn_only, PacketMeta};
use dashmap::DashMap;
use std::net::IpAddr;
use std::time::Duration;

const MIN_SAMPLES: usize = 6;
const MAX_HISTORY: usize = 32;
const CV_THRESHOLD: f64 = 0.15;
const MIN_GAP_S: f64 = 1.0;
const MIN_MEAN_S: f64 = 3.0;
const ALERT_COOLDOWN: Duration = Duration::from_secs(300);
const WIDE_MIN_SAMPLES: usize = 32;
const WIDE_MIN_FRAC: f64 = 0.12;
const WIDE_MAX_FRAC: f64 = 2.2;
const WIDE_CV: f64 = 0.55;

struct History {
    last_conn: Option<Duration>,
    intervals: Vec<f64>,
    last_alert: Option<Duration>,
    last_seen: Duration,
    /// Distinct destinations seen (network-keyed history only; capped).
    dsts: Vec<IpAddr>,
}

pub struct BeaconDetector {
    per_pair: DashMap<(IpAddr, IpAddr), History>,
    per_net: DashMap<(IpAddr, IpAddr), History>,
}

/// Destination network: IPv4 /24, IPv6 /48.
fn net_of(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V4(a) => {
            let o = a.octets();
            IpAddr::V4([o[0], o[1], o[2], 0].into())
        }
        IpAddr::V6(a) => {
            let mut o = a.octets();
            for b in &mut o[6..] {
                *b = 0;
            }
            IpAddr::V6(o.into())
        }
    }
}

const MIN_ROTATION_DSTS: usize = 3;
const DST_CAP: usize = 8;

struct Finding {
    test: &'static str,
    median: f64,
    mean: f64,
    cv: f64,
    n: usize,
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

impl BeaconDetector {
    pub fn new() -> Self {
        Self { per_pair: DashMap::new(), per_net: DashMap::new() }
    }

    pub fn observe(&self, pkt: &PacketMeta) -> Option<Alert> {
        if !is_syn_only(pkt.tcp_flags) {
            return None;
        }
        let now = pkt.ts_dur();
        // Both histories are always updated; the exact-pair verdict wins.
        let pair = Self::step(&self.per_pair, (pkt.src_ip, pkt.dst_ip), pkt.dst_ip, now, false);
        let net_key = net_of(pkt.dst_ip);
        let net = Self::step(&self.per_net, (pkt.src_ip, net_key), pkt.dst_ip, now, true);
        if let Some(f) = pair {
            return Some(Self::alert(pkt, &f, None));
        }
        net.map(|f| Self::alert(pkt, &f, Some(net_key)))
    }

    fn alert(pkt: &PacketMeta, f: &Finding, net: Option<IpAddr>) -> Alert {
        let target = match net {
            Some(n) => format!("{n} (network, rotating addresses)"),
            None => pkt.dst_ip.to_string(),
        };
        Alert::new(
            Severity::Medium,
            "beacon",
            format!(
                "{} -> {} reconnects every ~{:.1}s (cv={:.3}, {} samples, test={}): possible C2 beacon",
                pkt.src_ip, target, f.median, f.cv, f.n, f.test
            ),
            Some(pkt.src_ip),
            Some(pkt.dst_ip),
            serde_json::json!({
                "median_interval_s": f.median, "mean_interval_s": f.mean, "cv": f.cv, "samples": f.n,
                "test": f.test, "src_spoofable": true, "network_keyed": net.is_some()
            }),
        )
    }

    /// Records one connection attempt in `map[key]` and judges the history.
    /// `require_rotation`: only report when >= MIN_ROTATION_DSTS distinct
    /// destinations contributed (network-keyed history).
    fn step(
        map: &DashMap<(IpAddr, IpAddr), History>,
        key: (IpAddr, IpAddr),
        dst: IpAddr,
        now: Duration,
        require_rotation: bool,
    ) -> Option<Finding> {
        if !map.contains_key(&key) {
            // The network-keyed history is a second index over the same traffic;
            // keep it to a quarter of the budget so a spoofed-source flood
            // cannot double detector memory.
            make_room(
                map,
                if require_rotation { DEFAULT_CAP / 4 } else { DEFAULT_CAP },
                |h| now.saturating_sub(h.last_seen) > Duration::from_secs(3600),
                |h| h.intervals.is_empty(),
            );
        }
        let mut h = map.entry(key).or_insert_with(|| History {
            last_conn: None,
            intervals: Vec::new(),
            last_alert: None,
            last_seen: now,
            dsts: Vec::new(),
        });
        h.last_seen = now;
        if require_rotation && h.dsts.len() < DST_CAP && !h.dsts.contains(&dst) {
            h.dsts.push(dst);
        }
        if let Some(last) = h.last_conn {
            let dt = now.saturating_sub(last).as_secs_f64();
            if dt >= MIN_GAP_S {
                h.intervals.push(dt);
                if h.intervals.len() > MAX_HISTORY {
                    h.intervals.remove(0);
                }
                h.last_conn = Some(now);
            }
            // else: same burst; keep the earlier timestamp as the beat time
        } else {
            h.last_conn = Some(now);
        }

        let n = h.intervals.len();
        if n < MIN_SAMPLES || (require_rotation && h.dsts.len() < MIN_ROTATION_DSTS) {
            return None;
        }
        if h.last_alert.map_or(false, |t| now.saturating_sub(t) < ALERT_COOLDOWN) {
            return None;
        }
        let f = Self::judge(&h.intervals)?;
        h.last_alert = Some(now);
        Some(f)
    }

    fn judge(intervals: &[f64]) -> Option<Finding> {
        let n = intervals.len();
        let mean: f64 = intervals.iter().sum::<f64>() / n as f64;
        if mean < MIN_MEAN_S {
            return None;
        }
        let variance: f64 = intervals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let cv = variance.sqrt() / mean;

        let mut sorted = intervals.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = percentile(&sorted, 0.5);
        let mut devs: Vec<f64> = sorted.iter().map(|v| (v - median).abs()).collect();
        devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = percentile(&devs, 0.5);
        let near = sorted.iter().filter(|v| (**v - median).abs() <= 0.15 * median).count() as f64 / n as f64;
        let p10 = percentile(&sorted, 0.10);
        let p90 = percentile(&sorted, 0.90);
        let (min, max) = (sorted[0], sorted[n - 1]);

        let classic = cv <= CV_THRESHOLD;
        let robust = n >= 10 && median > 0.0 && mad / median <= 0.08 && near >= 0.70;
        let bounded = n >= 16 && p10 > 0.0 && p90 / p10 <= 2.0 && cv <= 0.30;
        let wide = n >= WIDE_MIN_SAMPLES && median > 0.0 && min >= WIDE_MIN_FRAC * median && max <= WIDE_MAX_FRAC * median && cv <= WIDE_CV;
        if !(classic || robust || bounded || wide) {
            return None;
        }
        let test = if classic {
            "low-cv"
        } else if robust {
            "robust-median"
        } else if bounded {
            "bounded-jitter"
        } else {
            "wide-jitter"
        };
        Some(Finding { test, median, mean, cv, n })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn f64(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.f64().max(1e-12);
            let u2 = self.f64();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    fn any_window_flags(series: &[f64]) -> bool {
        // slide exactly as the detector does: judge the trailing <= 32 intervals
        // after every new sample once MIN_SAMPLES are available.
        let mut hist: Vec<f64> = Vec::new();
        for &d in series {
            hist.push(d);
            if hist.len() > MAX_HISTORY {
                hist.remove(0);
            }
            if hist.len() >= MIN_SAMPLES && BeaconDetector::judge(&hist).is_some() {
                return true;
            }
        }
        false
    }

    /// Measured, not assumed: the wide-jitter test must catch randomised
    /// implants and stay quiet on human-like arrival processes.
    #[test]
    fn wide_jitter_calibration() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let trials = 2000;
        let rate = |gen: &mut dyn FnMut(&mut Rng) -> f64, rng: &mut Rng| -> f64 {
            let mut hits = 0;
            for _ in 0..trials {
                let series: Vec<f64> = (0..48).map(|_| gen(rng)).collect();
                if any_window_flags(&series) {
                    hits += 1;
                }
            }
            hits as f64 / trials as f64
        };
        // implants: 60 s base, uniform jitter
        let tpr80 = rate(&mut |r| 60.0 * (1.0 + 0.8 * (2.0 * r.f64() - 1.0)), &mut rng);
        let tpr70 = rate(&mut |r| 60.0 * (1.0 + 0.7 * (2.0 * r.f64() - 1.0)), &mut rng);
        // humans / apps
        let fp_ln10 = rate(&mut |r| 60.0 * r.normal().exp(), &mut rng);
        let fp_ln15 = rate(&mut |r| 60.0 * (1.5 * r.normal()).exp(), &mut rng);
        let fp_exp = rate(&mut |r| -60.0 * (1.0 - r.f64()).ln(), &mut rng);
        eprintln!("TPR +-80%: {tpr80:.3}  +-70%: {tpr70:.3}  FPR lognormal(1.0): {fp_ln10:.4}  lognormal(1.5): {fp_ln15:.4}  exponential: {fp_exp:.4}");
        assert!(tpr80 >= 0.95, "tpr80 {tpr80}");
        assert!(tpr70 >= 0.95, "tpr70 {tpr70}");
        assert!(fp_ln10 <= 0.01, "fp lognormal(1.0) {fp_ln10}");
        assert!(fp_ln15 <= 0.01, "fp lognormal(1.5) {fp_ln15}");
        assert!(fp_exp <= 0.01, "fp exponential {fp_exp}");
    }

    #[test]
    fn network_key_groups_a_slash24_and_ignores_other_networks() {
        let a: IpAddr = "203.0.113.7".parse().unwrap();
        let b: IpAddr = "203.0.113.200".parse().unwrap();
        let c: IpAddr = "203.0.114.7".parse().unwrap();
        assert_eq!(net_of(a), net_of(b));
        assert_ne!(net_of(a), net_of(c));
    }
}
