//! Port-scan detection (vertical: many ports on one host; horizontal: many
//! hosts), hardened against the evasions the red-team harness exercises:
//!
//! * **Stealth probes.** Original code counted only SYN-only segments, so
//!   NULL / FIN / XMAS scans (and illegal SYN+FIN / SYN+RST probes) were
//!   invisible. Any TCP segment that cannot belong to an established
//!   connection and carries scan-style flags now counts as a touch.
//! * **Low-and-slow.** A second, hour-long window per remote source (inbound
//!   only -- outbound browsing legitimately touches hundreds of hosts/hour)
//!   catches scanners that pace themselves under the 10 s thresholds.
//! * **Bounded memory + eviction that protects accumulated evidence**
//!   (see `bounded.rs`), and a per-source alert cooldown so one scan produces
//!   one alert, not one per extra packet.
//!
//! Round-2 additions (adaptive adversary):
//!
//! * **Distributed scans.** Every counter above is per *source*, so a botnet of
//!   N hosts that each probe a few ports is invisible. A per-*target* view now
//!   counts distinct ports and distinct sources against each local address
//!   (`observe_distributed`): many sources fanning over many ports is a scan
//!   whatever each source individually did. The alert is victim-only (the
//!   sources of a distributed or decoy scan are not a safe ban target).
//! * **Address rotation.** Sources are keyed by `canon_src`: an IPv6 /64 is one
//!   actor, and `::ffff:a.b.c.d` is the same actor as `a.b.c.d`.
//!
//! Original design notes still apply: direction matters (our SYNs vs theirs)
//! and outbound thresholds are far looser, because one web page can touch a
//! dozen-plus CDN hosts in a second.

use super::bounded::{canon_src, fnv64, make_room, src_hash, DEFAULT_CAP};
use crate::alert::{Alert, Severity};
use crate::packet::{is_probe_flags, is_syn_only, L4Proto, PacketMeta};
use dashmap::DashMap;
use std::collections::HashSet;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

/// Sets never need to grow past this: every threshold is below it.
const SET_CAP: usize = 128;
const ALERT_COOLDOWN: Duration = Duration::from_secs(30);

#[derive(Default)]
struct Distinct {
    ports: Vec<u16>,
    hosts: Vec<u64>,
}

impl Distinct {
    fn clear(&mut self) {
        self.ports.clear();
        self.hosts.clear();
    }
    fn add(&mut self, port: u16, host: u64) {
        if self.ports.len() < SET_CAP && !self.ports.contains(&port) {
            self.ports.push(port);
        }
        if self.hosts.len() < SET_CAP && !self.hosts.contains(&host) {
            self.hosts.push(host);
        }
    }
}

/// Per-target view for `observe_distributed`.
struct DstWindow {
    started: Duration,
    ports: Vec<u16>,
    srcs: Vec<u64>,
    sample: Vec<IpAddr>,
    last_alert: Option<Duration>,
}

/// Distributed-scan thresholds: a target probed on this many distinct ports by
/// this many distinct sources inside the window. Ordinary servers expose a
/// handful of ports; 64 distinct ports from 16 distinct clients is not
/// clients, it is enumeration.
const DIST_WINDOW: Duration = Duration::from_secs(900);
const DIST_PORTS: usize = 64;
const DIST_SOURCES: usize = 16;
const DIST_COOLDOWN: Duration = Duration::from_secs(600);
const DIST_TARGET_CAP: usize = 4096;

struct Window {
    started: Duration,
    fast: Distinct,
    slow_started: Duration,
    slow: Distinct,
    touches: u32,
    last_alert: Option<Duration>,
    last_slow_alert: Option<Duration>,
    last_seen: Duration,
}

pub struct Thresholds {
    pub port_threshold: usize,
    pub host_threshold: usize,
}

pub struct PortScanDetector {
    local_ips: Arc<HashSet<IpAddr>>,
    per_src: DashMap<IpAddr, Window>,
    window: Duration,
    slow_window: Duration,
    inbound: Thresholds,
    outbound: Thresholds,
    slow_inbound: Thresholds,
    cap: usize,
    per_dst: DashMap<IpAddr, DstWindow>,
}

impl PortScanDetector {
    pub fn new(local_ips: Arc<HashSet<IpAddr>>, window: Duration, inbound: Thresholds, outbound: Thresholds) -> Self {
        Self {
            local_ips,
            per_src: DashMap::new(),
            window,
            slow_window: Duration::from_secs(3600),
            inbound,
            outbound,
            // A remote source touching 40 distinct ports (or 30 hosts) of
            // ours within an hour is not a client; ordinary clients use 1-3.
            slow_inbound: Thresholds { port_threshold: 40, host_threshold: 30 },
            cap: DEFAULT_CAP,
            per_dst: DashMap::new(),
        }
    }

    /// Per-target distributed-scan detection (see module docs).
    pub fn observe_distributed(&self, pkt: &PacketMeta) -> Option<Alert> {
        if pkt.proto != L4Proto::Tcp || pkt.dst_port == 0 {
            return None;
        }
        if !(is_syn_only(pkt.tcp_flags) || is_probe_flags(pkt.tcp_flags)) || !self.local_ips.contains(&pkt.dst_ip) {
            return None;
        }
        if !self.per_dst.contains_key(&pkt.dst_ip) && self.per_dst.len() >= DIST_TARGET_CAP {
            return None;
        }
        let now = pkt.ts_dur();
        let src = canon_src(pkt.src_ip);
        let mut w = self.per_dst.entry(pkt.dst_ip).or_insert_with(|| DstWindow {
            started: now,
            ports: Vec::new(),
            srcs: Vec::new(),
            sample: Vec::new(),
            last_alert: None,
        });
        if now.saturating_sub(w.started) > DIST_WINDOW {
            w.started = now;
            w.ports.clear();
            w.srcs.clear();
            w.sample.clear();
        }
        if w.ports.len() < SET_CAP && !w.ports.contains(&pkt.dst_port) {
            w.ports.push(pkt.dst_port);
        }
        let sh = src_hash(src);
        if w.srcs.len() < SET_CAP && !w.srcs.contains(&sh) {
            w.srcs.push(sh);
            if w.sample.len() < 8 {
                w.sample.push(src);
            }
        }
        let (n_ports, n_srcs) = (w.ports.len(), w.srcs.len());
        let cooled = w.last_alert.map_or(true, |t| now.saturating_sub(t) >= DIST_COOLDOWN);
        if cooled && n_ports >= DIST_PORTS && n_srcs >= DIST_SOURCES {
            w.last_alert = Some(now);
            let sample: Vec<String> = w.sample.iter().map(|a| a.to_string()).collect();
            return Some(Alert::new(
                Severity::Medium,
                "portscan",
                format!(
                    "{} probed on {} distinct ports by {} distinct sources within {:?} (distributed scan: each source stays below per-source thresholds)",
                    pkt.dst_ip, n_ports, n_srcs, DIST_WINDOW
                ),
                None,
                Some(pkt.dst_ip),
                serde_json::json!({
                    "distinct_ports": n_ports, "distinct_sources": n_srcs, "distributed": true,
                    "sample_sources": sample, "victim_only": true, "src_spoofable": true
                }),
            ));
        }
        None
    }

    pub fn observe(&self, pkt: &PacketMeta) -> Option<Alert> {
        if pkt.proto != L4Proto::Tcp || pkt.dst_port == 0 {
            return None;
        }
        let syn = is_syn_only(pkt.tcp_flags);
        let probe = is_probe_flags(pkt.tcp_flags);
        if !syn && !probe {
            return None;
        }

        let dst_is_local = self.local_ips.contains(&pkt.dst_ip);
        let src_is_local = self.local_ips.contains(&pkt.src_ip);
        if !dst_is_local && !src_is_local {
            return None;
        }
        let (thresholds, direction) =
            if dst_is_local { (&self.inbound, "inbound") } else { (&self.outbound, "outbound") };
        let scan_type = if probe && !syn { "stealth" } else { "syn" };

        let now = pkt.ts_dur();
        // Remote inbound sources are keyed per actor (/64 for IPv6); our own
        // addresses are exact -- merging distinct local clients of one /64
        // would add their counters together.
        let key = if dst_is_local { canon_src(pkt.src_ip) } else { pkt.src_ip };
        let host_h = match pkt.dst_ip {
            IpAddr::V4(a) => fnv64(&a.octets()),
            IpAddr::V6(a) => fnv64(&a.octets()),
        };

        if !self.per_src.contains_key(&key) {
            let window = self.window.max(self.slow_window);
            make_room(
                &self.per_src,
                self.cap,
                |w| now.saturating_sub(w.last_seen) > window,
                |w| w.touches <= 1,
            );
        }
        let mut entry = self.per_src.entry(key).or_insert_with(|| Window {
            started: now,
            fast: Distinct::default(),
            slow_started: now,
            slow: Distinct::default(),
            touches: 0,
            last_alert: None,
            last_slow_alert: None,
            last_seen: now,
        });

        if now.saturating_sub(entry.started) > self.window {
            entry.started = now;
            entry.fast.clear();
        }
        if now.saturating_sub(entry.slow_started) > self.slow_window {
            entry.slow_started = now;
            entry.slow.clear();
        }
        entry.last_seen = now;
        entry.touches = entry.touches.saturating_add(1);
        entry.fast.add(pkt.dst_port, host_h);
        if dst_is_local {
            entry.slow.add(pkt.dst_port, host_h);
        }
        let (n_ports, n_hosts) = (entry.fast.ports.len(), entry.fast.hosts.len());
        let (s_ports, s_hosts) = (entry.slow.ports.len(), entry.slow.hosts.len());
        let cooled = entry.last_alert.map_or(true, |t| now.saturating_sub(t) >= ALERT_COOLDOWN);
        let slow_cooled = entry.last_slow_alert.map_or(true, |t| now.saturating_sub(t) >= ALERT_COOLDOWN * 20);

        let mut alert = None;
        if cooled && n_ports >= thresholds.port_threshold {
            entry.last_alert = Some(now);
            alert = Some(Alert::new(
                Severity::High,
                "portscan",
                format!(
                    "{} touched {} distinct ports within {:?} ({direction} vertical {scan_type} scan)",
                    key, n_ports, self.window
                ),
                Some(key),
                if dst_is_local { Some(pkt.dst_ip) } else { None },
                serde_json::json!({
                    "distinct_ports": n_ports, "direction": direction, "scan_type": scan_type,
                    "src_spoofable": true
                }),
            ));
        } else if cooled && n_hosts >= thresholds.host_threshold {
            entry.last_alert = Some(now);
            let severity = if direction == "inbound" { Severity::Medium } else { Severity::Low };
            alert = Some(Alert::new(
                severity,
                "portscan",
                format!(
                    "{} touched {} distinct hosts within {:?} ({direction} horizontal {scan_type} scan)",
                    key, n_hosts, self.window
                ),
                Some(key),
                None,
                serde_json::json!({
                    "distinct_hosts": n_hosts, "direction": direction, "scan_type": scan_type,
                    "src_spoofable": true
                }),
            ));
        } else if dst_is_local
            && slow_cooled
            && (s_ports >= self.slow_inbound.port_threshold || s_hosts >= self.slow_inbound.host_threshold)
        {
            entry.last_slow_alert = Some(now);
            alert = Some(Alert::new(
                Severity::Medium,
                "portscan",
                format!(
                    "{} touched {} distinct ports / {} hosts within {:?} (low-and-slow {scan_type} scan)",
                    key, s_ports, s_hosts, self.slow_window
                ),
                Some(key),
                Some(pkt.dst_ip),
                serde_json::json!({
                    "distinct_ports": s_ports, "distinct_hosts": s_hosts, "direction": direction,
                    "scan_type": scan_type, "slow": true, "src_spoofable": true
                }),
            ));
        }
        alert
    }
}
