//! 5-tuple flow tracking, modeled loosely on Zeek's `conn.log`.
//! Flows are kept in a concurrent map keyed by a direction-normalized
//! tuple so that both sides of a connection accumulate into one record.

use crate::packet::{L4Proto, PacketMeta};
use dashmap::DashMap;
use serde::Serialize;
use std::net::IpAddr;
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FlowKey {
    pub lo_ip: IpAddr,
    pub hi_ip: IpAddr,
    pub lo_port: u16,
    pub hi_port: u16,
    pub proto: u8,
}

impl FlowKey {
    /// Normalizes direction so A->B and B->A map to the same key.
    fn new(a_ip: IpAddr, a_port: u16, b_ip: IpAddr, b_port: u16, proto_id: u8) -> Self {
        if (a_ip, a_port) <= (b_ip, b_port) {
            FlowKey { lo_ip: a_ip, hi_ip: b_ip, lo_port: a_port, hi_port: b_port, proto: proto_id }
        } else {
            FlowKey { lo_ip: b_ip, hi_ip: a_ip, lo_port: b_port, hi_port: a_port, proto: proto_id }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct FlowRecord {
    pub orig_ip: IpAddr,
    pub resp_ip: IpAddr,
    pub orig_port: u16,
    pub resp_port: u16,
    pub proto: String,
    pub packets: u64,
    pub bytes: u64,
    #[serde(skip)]
    pub first_seen: SystemTime,
    #[serde(skip)]
    pub last_seen: SystemTime,
    pub syn_count: u32,
    pub fin_or_rst: bool,
}

impl FlowRecord {
    /// Wall-clock lifetime of this flow so far. Exposed for future
    /// reporting/export; not consumed internally yet.
    #[allow(dead_code)]
    pub fn duration(&self) -> Duration {
        self.last_seen.duration_since(self.first_seen).unwrap_or_default()
    }
}

pub struct FlowTable {
    inner: DashMap<FlowKey, FlowRecord>,
}

fn proto_id(p: L4Proto) -> u8 {
    match p {
        L4Proto::Tcp => 6,
        L4Proto::Udp => 17,
        L4Proto::Icmp => 1,
        L4Proto::Other(n) => n,
    }
}

fn proto_name(p: L4Proto) -> &'static str {
    match p {
        L4Proto::Tcp => "tcp",
        L4Proto::Udp => "udp",
        L4Proto::Icmp => "icmp",
        L4Proto::Other(_) => "other",
    }
}

impl FlowTable {
    pub fn new() -> Self {
        Self { inner: DashMap::new() }
    }

    /// Updates (or creates) the flow this packet belongs to and returns
    /// a snapshot of the record after the update.
    pub fn update(&self, pkt: &PacketMeta) -> FlowRecord {
        let key = FlowKey::new(pkt.src_ip, pkt.src_port, pkt.dst_ip, pkt.dst_port, proto_id(pkt.proto));
        let mut entry = self.inner.entry(key).or_insert_with(|| FlowRecord {
            orig_ip: pkt.src_ip,
            resp_ip: pkt.dst_ip,
            orig_port: pkt.src_port,
            resp_port: pkt.dst_port,
            proto: proto_name(pkt.proto).to_string(),
            packets: 0,
            bytes: 0,
            first_seen: pkt.ts,
            last_seen: pkt.ts,
            syn_count: 0,
            fin_or_rst: false,
        });
        entry.packets += 1;
        entry.bytes += pkt.length as u64;
        entry.last_seen = pkt.ts;
        if crate::packet::is_syn_only(pkt.tcp_flags) {
            entry.syn_count += 1;
        }
        if pkt.tcp_flags & 0x05 != 0 {
            // FIN (0x01) or RST (0x04)
            entry.fin_or_rst = true;
        }
        entry.clone()
    }

    /// Evicts flows that have been idle longer than `idle_timeout`,
    /// keeping memory bounded on long-running captures.
    pub fn reap_idle(&self, idle_timeout: Duration) -> usize {
        let now = SystemTime::now();
        let before = self.inner.len();
        self.inner.retain(|_, rec| {
            now.duration_since(rec.last_seen).map(|d| d < idle_timeout).unwrap_or(true)
        });
        before - self.inner.len()
    }

    pub fn active_count(&self) -> usize {
        self.inner.len()
    }
}
