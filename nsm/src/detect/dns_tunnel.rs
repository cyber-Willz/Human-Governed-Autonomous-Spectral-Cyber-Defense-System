//! DNS tunnelling / exfiltration detection.
//!
//! The original heuristic (entropy of the first 256 payload bytes incl. the
//! DNS header, >=5 "long high-entropy" labels in 30 s, or 60 queries / 30 s)
//! had three problems the red-team harness reproduced:
//!
//! * **False positives** on ordinary CDN-style names: a few long, random-
//!   looking hostnames queried repeatedly look "high entropy" one at a time.
//! * **Evasion** by pacing (1 query / 25 s), by splitting data over many
//!   short labels, and by encoding with dictionary words -- none of which
//!   trip a per-label entropy test.
//! * Counting DNS *responses* (src port 53) as if the resolver were the
//!   tunnelling client.
//!
//! The invariant that actually separates a tunnel from browsing is
//! **cardinality under one parent zone**: a tunnel must put its payload in
//! the *name*, so it generates many *distinct* subdomains of the attacker's
//! zone, each carrying a chunk of encoded data. Repeated lookups of the same
//! few names do not. This detector tracks, per (client, parent zone), the
//! first-seen times of distinct subdomains and fires when enough distinct,
//! "payload-sized" subdomains appear in any of three windows (fast / medium /
//! slow), which also defeats rate pacing. Entropy and length only refine
//! severity. Known CDN/infrastructure zones that legitimately mint huge
//! numbers of subdomains are exempted: a tunnel needs the attacker's own
//! authoritative zone, which is never one of these.
//!
//! The legacy per-source query-rate rule is kept as a Medium-severity
//! backstop. Out of scope: DoH/DoT (payload not visible).
//!
//! Round-2 additions (adaptive adversary):
//!
//! * **Zone parsing.** The registrable zone was guessed from label lengths, so
//!   `<payload>.<payload>.abc.io` (a 3-letter name under a 2-letter ccTLD) was
//!   parsed as zone `<payload>.abc.io` with an empty subdomain part: the tunnel
//!   was skipped entirely, or keyed on a zone that changes with every query.
//!   Now only genuine second-level-under-ccTLD suffixes (`co.uk`, `com.au`,
//!   `ne.jp` ...) take three labels.
//! * **DNS over TCP/53.** The detector looked at UDP only; a tunnel simply used
//!   TCP. Length-prefixed (RFC 1035 sec 4.2.2) queries are now parsed too.
//! * **Fleet view.** All per-client thresholds are per client, so N hosts that
//!   each send a few names to one attacker zone stay under every one. A per-zone
//!   view counts distinct subdomains and distinct clients across the whole
//!   network. Its alert names no host (it is analyst-facing and cannot reach
//!   containment): cloud-AV / EDR reputation lookups look identical on the wire
//!   and an operator must be able to allow-list them without risking a ban.

use super::bounded::{fnv64, make_room};
use crate::alert::{Alert, Severity};
use crate::packet::{L4Proto, PacketMeta};
use dashmap::DashMap;
use std::net::IpAddr;
use std::time::Duration;

/// Zones whose subdomain fan-out is legitimately huge (and which an attacker
/// cannot use as a tunnel endpoint because they do not control the zone).
const BENIGN_ZONES: &[&str] = &[
    "cloudfront.net",
    "amazonaws.com",
    "akamaiedge.net",
    "akamai.net",
    "akadns.net",
    "edgekey.net",
    "googleusercontent.com",
    "googlevideo.com",
    "gstatic.com",
    "fbcdn.net",
    "azureedge.net",
    "cloudflare.net",
    "fastly.net",
    "in-addr.arpa",
    "ip6.arpa",
    "local",
];

/// Subdomains shorter than this carry too little data to be a tunnel chunk.
const MIN_MEAN_SUB_LEN: f64 = 10.0;
/// (window, distinct-subdomain threshold)
const TIERS: [(u64, usize); 3] = [(60, 8), (600, 12), (3600, 30)];
const HIST_CAP: usize = 48;

/// Second-level labels that form a public suffix together with a two-letter
/// country code (`co.uk`, `com.au`, `ne.jp`, `ac.nz` ...).
const SLD_UNDER_CCTLD: &[&str] = &[
    "co", "com", "org", "net", "gov", "edu", "ac", "or", "ne", "go", "gob", "mil", "ltd", "plc", "sch", "nom", "id", "web",
];

// Fleet view: distinct subdomains of one zone across >= FLEET_CLIENTS clients.
const FLEET_WINDOW_S: u64 = 3600;
const FLEET_SUBS: usize = 200;
const FLEET_CLIENTS: usize = 5;
const FLEET_SUBS_CAP: usize = 512;
const FLEET_CLIENT_CAP: usize = 64;
const FLEET_KEYS_CAP: usize = 20_000;
const FLEET_COOLDOWN: Duration = Duration::from_secs(900);
const TUNNEL_KEYS_CAP: usize = 20_000;
const SRC_KEYS_CAP: usize = 50_000;
const ALERT_COOLDOWN: Duration = Duration::from_secs(120);

struct ZoneState {
    /// (first-seen time, hash of subdomain, subdomain length, entropy*100)
    seen: Vec<(Duration, u64, u16, u16)>,
    last_alert: Option<Duration>,
    last_seen: Duration,
}

struct FleetZone {
    zone: String,
    subs: Vec<(Duration, u64, u16, u16)>,
    clients: Vec<u64>,
    sample: Vec<IpAddr>,
    last_alert: Option<Duration>,
    last_seen: Duration,
}

struct RateWindow {
    started: Duration,
    queries: u32,
    last_alert: Option<Duration>,
}

pub struct DnsTunnelDetector {
    per_zone: DashMap<(IpAddr, u64), ZoneState>,
    per_src: DashMap<IpAddr, RateWindow>,
    per_fleet: DashMap<u64, FleetZone>,
    window: Duration,
    query_rate_threshold: u32,
}

struct Parsed {
    labels: Vec<String>,
}

fn parse_qname(payload: &[u8]) -> Option<Parsed> {
    if payload.len() < 13 {
        return None;
    }
    // QR bit must be 0 (query) and QDCOUNT >= 1.
    if payload[2] & 0x80 != 0 || (payload[4] == 0 && payload[5] == 0) {
        return None;
    }
    let mut i = 12usize;
    let mut labels = Vec::new();
    while i < payload.len() {
        let len = payload[i] as usize;
        if len == 0 {
            return if labels.is_empty() { None } else { Some(Parsed { labels }) };
        }
        if len & 0xC0 != 0 || i + 1 + len > payload.len() {
            break; // compression pointer or truncated capture: use what we have
        }
        labels.push(String::from_utf8_lossy(&payload[i + 1..i + 1 + len]).to_ascii_lowercase());
        i += 1 + len;
    }
    if labels.is_empty() { None } else { Some(Parsed { labels }) }
}

/// Splits labels into (registered-zone key, subdomain labels). Uses the last
/// two labels, or three for the common `xx.co.uk` shape.
fn split_zone(labels: &[String]) -> (String, &[String]) {
    let n = labels.len();
    let zone_len = if n >= 3 && labels[n - 1].len() == 2 && SLD_UNDER_CCTLD.contains(&labels[n - 2].as_str()) { 3 } else { 2 };
    let zone_len = zone_len.min(n);
    (labels[n - zone_len..].join("."), &labels[..n - zone_len])
}

/// A DNS-over-TCP segment carries a 2-byte big-endian length before the
/// message. Returns the first complete message in the segment, if any.
fn tcp_dns_message(p: &[u8]) -> Option<&[u8]> {
    if p.len() < 14 {
        return None;
    }
    let l = u16::from_be_bytes([p[0], p[1]]) as usize;
    if l < 12 || l + 2 > p.len() {
        return None;
    }
    Some(&p[2..2 + l])
}

fn shannon_entropy(bytes: &[u8]) -> f64 {
    if bytes.is_empty() {
        return 0.0;
    }
    let mut counts = [0u32; 256];
    for &b in bytes {
        counts[b as usize] += 1;
    }
    let len = bytes.len() as f64;
    counts.iter().filter(|&&c| c > 0).map(|&c| {
        let p = c as f64 / len;
        -p * p.log2()
    }).sum()
}

impl DnsTunnelDetector {
    pub fn new(window: Duration, query_rate_threshold: u32) -> Self {
        Self { per_zone: DashMap::new(), per_src: DashMap::new(), per_fleet: DashMap::new(), window, query_rate_threshold }
    }

    pub fn observe(&self, pkt: &PacketMeta) -> Option<Alert> {
        // Queries only: responses echo the question and come *from* resolvers.
        if pkt.dst_port != 53 {
            return None;
        }
        let msg: &[u8] = match pkt.proto {
            L4Proto::Udp => &pkt.payload_head,
            L4Proto::Tcp => tcp_dns_message(&pkt.payload_head)?,
            _ => return None,
        };
        let now = pkt.ts_dur();

        // ---- legacy per-source query-rate backstop (Medium) ----
        let mut rate_alert = None;
        {
            if !self.per_src.contains_key(&pkt.src_ip) {
                let window = self.window;
                make_room(&self.per_src, SRC_KEYS_CAP, |w| now.saturating_sub(w.started) > window, |w| w.queries <= 1);
            }
            let mut w = self.per_src.entry(pkt.src_ip).or_insert_with(|| RateWindow {
                started: now,
                queries: 0,
                last_alert: None,
            });
            if now.saturating_sub(w.started) > self.window {
                w.started = now;
                w.queries = 0;
            }
            w.queries = w.queries.saturating_add(1);
            let cooled = w.last_alert.map_or(true, |t| now.saturating_sub(t) >= ALERT_COOLDOWN);
            if w.queries >= self.query_rate_threshold && cooled {
                w.last_alert = Some(now);
                rate_alert = Some((w.queries, pkt));
            }
        }

        // ---- cardinality-under-one-zone tunnel detection ----
        let parsed = parse_qname(msg)?;
        let (zone, sub_labels) = split_zone(&parsed.labels);
        let exempt = sub_labels.is_empty() || BENIGN_ZONES.iter().any(|z| zone == *z || zone.ends_with(&format!(".{z}")));
        let tunnel_alert = if exempt { None } else { self.observe_zone(pkt, now, &zone, sub_labels) };
        let fleet_alert = if exempt { None } else { self.observe_fleet(pkt, now, &zone, sub_labels) };

        if tunnel_alert.is_some() {
            return tunnel_alert;
        }
        if fleet_alert.is_some() {
            return fleet_alert;
        }
        rate_alert.map(|(queries, p)| {
            Alert::new(
                Severity::Medium,
                "dns_tunnel",
                format!("{} issued {} DNS queries within {:?} (abnormal query rate)", p.src_ip, queries, self.window),
                Some(p.src_ip),
                Some(p.dst_ip),
                serde_json::json!({ "query_count": queries, "src_spoofable": true }),
            )
        })
    }

    fn observe_fleet(&self, pkt: &PacketMeta, now: Duration, zone: &str, sub_labels: &[String]) -> Option<Alert> {
        let sub = sub_labels.join(".");
        let sub_h = fnv64(sub.as_bytes());
        let zone_h = fnv64(zone.as_bytes());
        if !self.per_fleet.contains_key(&zone_h) {
            let horizon = Duration::from_secs(FLEET_WINDOW_S);
            make_room(&self.per_fleet, FLEET_KEYS_CAP, |z| now.saturating_sub(z.last_seen) > horizon, |z| z.subs.len() <= 1);
        }
        let mut z = self.per_fleet.entry(zone_h).or_insert_with(|| FleetZone {
            zone: zone.to_string(),
            subs: Vec::new(),
            clients: Vec::new(),
            sample: Vec::new(),
            last_alert: None,
            last_seen: now,
        });
        z.last_seen = now;
        let horizon = Duration::from_secs(FLEET_WINDOW_S);
        z.subs.retain(|(t, ..)| now.saturating_sub(*t) <= horizon);
        if !z.subs.iter().any(|(_, h, ..)| *h == sub_h) {
            if z.subs.len() >= FLEET_SUBS_CAP {
                z.subs.remove(0);
            }
            let ent = (shannon_entropy(sub.as_bytes()) * 100.0) as u16;
            z.subs.push((now, sub_h, sub.len().min(u16::MAX as usize) as u16, ent));
        }
        let ch = match pkt.src_ip {
            IpAddr::V4(a) => fnv64(&a.octets()),
            IpAddr::V6(a) => fnv64(&a.octets()),
        };
        if z.clients.len() < FLEET_CLIENT_CAP && !z.clients.contains(&ch) {
            z.clients.push(ch);
            if z.sample.len() < 8 {
                z.sample.push(pkt.src_ip);
            }
        }
        let cooled = z.last_alert.map_or(true, |t| now.saturating_sub(t) >= FLEET_COOLDOWN);
        let (n, n_clients) = (z.subs.len(), z.clients.len());
        if !cooled || n < FLEET_SUBS || n_clients < FLEET_CLIENTS {
            return None;
        }
        let mean_len = z.subs.iter().map(|r| r.2 as f64).sum::<f64>() / n as f64;
        if mean_len < MIN_MEAN_SUB_LEN {
            return None;
        }
        let mean_ent = z.subs.iter().map(|r| r.3 as f64).sum::<f64>() / n as f64 / 100.0;
        z.last_alert = Some(now);
        let sample: Vec<String> = z.sample.iter().map(|a| a.to_string()).collect();
        let zname = z.zone.clone();
        drop(z);
        Some(Alert::new(
            if mean_ent >= 3.0 { Severity::Medium } else { Severity::Low },
            "dns_tunnel",
            format!(
                "{n} distinct subdomains of {zname} queried by {n_clients} clients within {FLEET_WINDOW_S}s (mean length {mean_len:.0}, entropy {mean_ent:.2} bits/char): possible distributed DNS exfiltration, or cloud-reputation lookups (allow-list the zone if so)"
            ),
            None,
            None,
            serde_json::json!({
                "zone": zname, "fleet": true, "distinct_subdomains": n, "distinct_clients": n_clients,
                "mean_subdomain_len": mean_len, "mean_entropy": mean_ent, "sample_clients": sample
            }),
        ))
    }

    fn observe_zone(&self, pkt: &PacketMeta, now: Duration, zone: &str, sub_labels: &[String]) -> Option<Alert> {
        let sub = sub_labels.join(".");
        let sub_h = fnv64(sub.as_bytes());
        let key = (pkt.src_ip, fnv64(zone.as_bytes()));
        if !self.per_zone.contains_key(&key) {
            let horizon = Duration::from_secs(TIERS[2].0);
            make_room(&self.per_zone, TUNNEL_KEYS_CAP, |z| now.saturating_sub(z.last_seen) > horizon, |z| z.seen.len() <= 1);
        }
        let mut z = self.per_zone.entry(key).or_insert_with(|| ZoneState {
            seen: Vec::new(),
            last_alert: None,
            last_seen: now,
        });
        z.last_seen = now;
        let horizon = Duration::from_secs(TIERS[2].0);
        z.seen.retain(|(t, ..)| now.saturating_sub(*t) <= horizon);
        if !z.seen.iter().any(|(_, h, ..)| *h == sub_h) {
            if z.seen.len() >= HIST_CAP {
                z.seen.remove(0);
            }
            let ent = (shannon_entropy(sub.as_bytes()) * 100.0) as u16;
            z.seen.push((now, sub_h, sub.len().min(u16::MAX as usize) as u16, ent));
        }
        let cooled = z.last_alert.map_or(true, |t| now.saturating_sub(t) >= ALERT_COOLDOWN);
        if !cooled {
            return None;
        }
        for (win_s, need) in TIERS {
            let win = Duration::from_secs(win_s);
            let recent: Vec<&(Duration, u64, u16, u16)> =
                z.seen.iter().filter(|(t, ..)| now.saturating_sub(*t) <= win).collect();
            if recent.len() < need {
                continue;
            }
            let mean_len = recent.iter().map(|r| r.2 as f64).sum::<f64>() / recent.len() as f64;
            if mean_len < MIN_MEAN_SUB_LEN {
                continue;
            }
            let mean_ent = recent.iter().map(|r| r.3 as f64).sum::<f64>() / recent.len() as f64 / 100.0;
            let n = recent.len();
            let severity = if mean_ent >= 3.2 || mean_len >= 24.0 { Severity::High } else { Severity::Medium };
            z.last_alert = Some(now);
            drop(z);
            return Some(Alert::new(
                severity,
                "dns_tunnel",
                format!(
                    "{} queried {} distinct subdomains of {} within {}s (mean length {:.0}, entropy {:.2} bits/char): possible DNS tunnelling",
                    pkt.src_ip, n, zone, win_s, mean_len, mean_ent
                ),
                Some(pkt.src_ip),
                Some(pkt.dst_ip),
                serde_json::json!({
                    "zone": zone, "distinct_subdomains": n, "window_s": win_s,
                    "mean_subdomain_len": mean_len, "mean_entropy": mean_ent, "src_spoofable": true
                }),
            ));
        }
        None
    }
}
