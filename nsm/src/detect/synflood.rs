//! SYN-flood detection: SYN-only segments aimed at one destination IP:port
//! within a window, compared against a threshold.
//!
//! Hardening notes (red-team findings):
//! * The alert is about the **victim**. Its sources are, by the nature of a
//!   flood, overwhelmingly forged, so the alert carries `src_ip = None`,
//!   `victim_only = true` and `src_spoofable = true`. Downstream response
//!   logic must never treat the destination of this alert as an attacker to
//!   ban (the original pipeline did, auto-banning the host under attack).
//! * State is bounded: the per-destination table has a hard cap and the
//!   distinct-source set stops growing once it can no longer change the
//!   verdict.
//! * One alert per destination per cooldown instead of one per packet.
//!
//! Round-2 finding: the counter is keyed on (dst ip, dst port), so a flood that
//! *randomises the destination port* never lifts any one counter (4000 SYN/2 s
//! spread over 4000 ports = ~1 per counter). A second, per-destination-IP view
//! now counts all SYNs to an address and how many distinct ports they hit; a
//! flood is declared when the total is a multiple of the per-port threshold
//! *and* spread over many ports. A busy server's SYNs concentrate on a few
//! ports (which the per-port counter already judges), so the extra view does
//! not make ordinary load alarm.

use super::bounded::{make_room, DEFAULT_CAP};
use crate::alert::{Alert, Severity};
use crate::packet::{is_syn_only, PacketMeta};
use dashmap::DashMap;
use std::collections::HashSet;
use std::net::IpAddr;
use std::time::Duration;

/// Distinct sources only matter up to this point (`> 20` => Critical).
const SOURCE_CAP: usize = 64;
const ALERT_COOLDOWN: Duration = Duration::from_secs(10);

/// Spread flood: total SYNs to one IP (any port) >= SPREAD_FACTOR x the
/// per-port threshold, across at least SPREAD_PORTS distinct ports.
const SPREAD_FACTOR: u32 = 4;
const SPREAD_PORTS: usize = 64;
const PORT_SET_CAP: usize = 128;

struct SpreadCounter {
    started: Duration,
    syns: u32,
    ports: Vec<u16>,
    sources: HashSet<IpAddr>,
    last_alert: Option<Duration>,
}

struct Counter {
    started: Duration,
    syns: u32,
    sources: HashSet<IpAddr>,
    last_alert: Option<Duration>,
}

pub struct SynFloodDetector {
    per_dst: DashMap<(IpAddr, u16), Counter>,
    per_ip: DashMap<IpAddr, SpreadCounter>,
    window: Duration,
    syn_threshold: u32,
    cap: usize,
}

impl SynFloodDetector {
    pub fn new(window: Duration, syn_threshold: u32) -> Self {
        Self { per_dst: DashMap::new(), per_ip: DashMap::new(), window, syn_threshold, cap: DEFAULT_CAP }
    }

    pub fn observe(&self, pkt: &PacketMeta) -> Option<Alert> {
        if !is_syn_only(pkt.tcp_flags) {
            return None;
        }
        let key = (pkt.dst_ip, pkt.dst_port);
        let now = pkt.ts_dur();
        let spread_alert = self.observe_spread(pkt, now);
        if !self.per_dst.contains_key(&key) {
            let window = self.window;
            make_room(&self.per_dst, self.cap, |c| now.saturating_sub(c.started) > window, |c| c.syns <= 1);
        }
        let mut c = self.per_dst.entry(key).or_insert_with(|| Counter {
            started: now,
            syns: 0,
            sources: HashSet::new(),
            last_alert: None,
        });

        if now.saturating_sub(c.started) > self.window {
            c.started = now;
            c.syns = 0;
            c.sources.clear();
        }
        c.syns = c.syns.saturating_add(1);
        if c.sources.len() < SOURCE_CAP {
            c.sources.insert(pkt.src_ip);
        }
        let (syns, n_src) = (c.syns, c.sources.len());
        let cooled = c.last_alert.map_or(true, |t| now.saturating_sub(t) >= ALERT_COOLDOWN);

        if syns >= self.syn_threshold && cooled {
            c.last_alert = Some(now);
            drop(c);
            let severity = if n_src > 20 { Severity::Critical } else { Severity::High };
            return Some(Alert::new(
                severity,
                "synflood",
                format!(
                    "{}:{} received {} SYNs from {}{} sources within {:?}",
                    pkt.dst_ip,
                    pkt.dst_port,
                    syns,
                    n_src,
                    if n_src >= SOURCE_CAP { "+" } else { "" },
                    self.window
                ),
                None,
                Some(pkt.dst_ip),
                serde_json::json!({
                    "syn_count": syns, "distinct_sources": n_src,
                    "victim_only": true, "src_spoofable": true
                }),
            ));
        }
        spread_alert
    }

    fn observe_spread(&self, pkt: &PacketMeta, now: Duration) -> Option<Alert> {
        if !self.per_ip.contains_key(&pkt.dst_ip) {
            let window = self.window;
            make_room(&self.per_ip, self.cap, |c| now.saturating_sub(c.started) > window, |c| c.syns <= 1);
        }
        let mut c = self.per_ip.entry(pkt.dst_ip).or_insert_with(|| SpreadCounter {
            started: now,
            syns: 0,
            ports: Vec::new(),
            sources: HashSet::new(),
            last_alert: None,
        });
        if now.saturating_sub(c.started) > self.window {
            c.started = now;
            c.syns = 0;
            c.ports.clear();
            c.sources.clear();
        }
        c.syns = c.syns.saturating_add(1);
        if c.ports.len() < PORT_SET_CAP && !c.ports.contains(&pkt.dst_port) {
            c.ports.push(pkt.dst_port);
        }
        if c.sources.len() < SOURCE_CAP {
            c.sources.insert(pkt.src_ip);
        }
        let (syns, n_ports, n_src) = (c.syns, c.ports.len(), c.sources.len());
        let cooled = c.last_alert.map_or(true, |t| now.saturating_sub(t) >= ALERT_COOLDOWN);
        if cooled && syns >= self.syn_threshold.saturating_mul(SPREAD_FACTOR) && n_ports >= SPREAD_PORTS {
            c.last_alert = Some(now);
            drop(c);
            let severity = if n_src > 20 { Severity::Critical } else { Severity::High };
            return Some(Alert::new(
                severity,
                "synflood",
                format!(
                    "{} received {} SYNs across {}{} distinct ports from {}{} sources within {:?} (port-randomised flood)",
                    pkt.dst_ip,
                    syns,
                    n_ports,
                    if n_ports >= PORT_SET_CAP { "+" } else { "" },
                    n_src,
                    if n_src >= SOURCE_CAP { "+" } else { "" },
                    self.window
                ),
                None,
                Some(pkt.dst_ip),
                serde_json::json!({
                    "syn_count": syns, "distinct_ports": n_ports, "distinct_sources": n_src,
                    "spread": true, "victim_only": true, "src_spoofable": true
                }),
            ));
        }
        None
    }
}
