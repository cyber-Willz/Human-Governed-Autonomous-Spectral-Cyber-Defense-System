//! Synthetic packet generator used by `--simulate`. Lets the whole
//! pipeline (flow table + every detector + alerting) be exercised
//! without root privileges or a live NIC -- handy for CI, demos, and
//! for developing new detectors offline.

use crate::packet::{L4Proto, PacketMeta};
use std::net::{IpAddr, Ipv4Addr};
use std::time::{Duration, SystemTime};
use tokio::sync::mpsc::Sender;

/// The synthetic "this machine" address used throughout the demo
/// traffic below. Exposed so `main.rs` can seed the detectors' notion
/// of "local" addresses to match -- otherwise every inbound/outbound
/// directionality check in the port-scan detector would find neither
/// endpoint local and silently drop all simulated traffic.
pub const DEMO_LOCAL_IP: IpAddr = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 10));

fn pkt(src: [u8; 4], dst: [u8; 4], sport: u16, dport: u16, proto: L4Proto, flags: u8, payload: &[u8]) -> PacketMeta {
    PacketMeta {
        ts: SystemTime::now(),
        src_ip: IpAddr::V4(Ipv4Addr::from(src)),
        dst_ip: IpAddr::V4(Ipv4Addr::from(dst)),
        src_port: sport,
        dst_port: dport,
        proto,
        tcp_flags: flags,
        length: 60 + payload.len(),
        payload_head: payload.to_vec(),
    }
}

const SYN: u8 = 0x02;

pub async fn run(tx: Sender<PacketMeta>) {
    tracing::info!("simulate mode: generating synthetic port-scan, SYN-flood, DNS-tunnel, and beacon traffic");
    let attacker = [10, 0, 0, 66];
    let victim = [10, 0, 0, 10];
    let dns_server = [8, 8, 8, 8];
    let c2 = [203, 0, 113, 9];
    let normal_client = [10, 0, 0, 42];

    // 1. Vertical port scan: one source hammers many ports on one host.
    for port in 1..=30u16 {
        let _ = tx.send(pkt(attacker, victim, 40000 + port, port, L4Proto::Tcp, SYN, &[])).await;
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // 2. SYN flood: many spoofed-looking sources hit one port.
    for i in 0..150u32 {
        let src = [172, 16, ((i / 256) % 256) as u8, (i % 256) as u8];
        let _ = tx.send(pkt(src, victim, 50000, 443, L4Proto::Tcp, SYN, &[])).await;
        if i % 20 == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    // 3. DNS tunneling: long, high-entropy labels queried rapidly.
    let encoded_labels = [
        "aGVsbG93b3JsZHRoaXNpc2FsbG9uZ2Jhc2U2NHN0cmluZw",
        "eJx7z9NHqYWG5aFnMxOEyf6zVw2LrIeYuT5vfDFdG9pbG",
        "cXVpY2tzaWx2ZXJmb3hqdW1wZWRvdmVydGhlbGF6eWRvZw",
    ];
    for round in 0..3 {
        for label in encoded_labels.iter() {
            let mut payload = vec![0u8; 12]; // fake DNS header
            payload.push(label.len() as u8);
            payload.extend_from_slice(label.as_bytes());
            payload.push(0); // root label
            let _ = tx
                .send(pkt(attacker, dns_server, 51000 + round, 53, L4Proto::Udp, 0, &payload))
                .await;
            tokio::time::sleep(Duration::from_millis(30)).await;
        }
    }

    // 4. C2 beacon: victim reconnects to the same host:port at a very
    //    regular interval -- classic low-jitter check-in behavior.
    for _ in 0..8 {
        let _ = tx.send(pkt(victim, c2, 55000, 4444, L4Proto::Tcp, SYN, &[])).await;
        tokio::time::sleep(Duration::from_millis(700)).await; // "seconds" compressed for the demo
    }

    // 5. Benign background traffic with a signature hit (cleartext basic auth).
    let http_req = b"GET /login HTTP/1.1\r\nAuthorization: Basic dXNlcjpwYXNz\r\n\r\n";
    let _ = tx.send(pkt(normal_client, victim, 33000, 80, L4Proto::Tcp, SYN, http_req)).await;

    tracing::info!("simulate mode: finished generating traffic");
}
