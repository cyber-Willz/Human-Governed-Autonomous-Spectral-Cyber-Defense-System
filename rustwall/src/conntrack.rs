use crate::config::ConntrackConfig;
use crate::packet::{L4Proto, ParsedPacket};
use dashmap::DashMap;
use std::net::IpAddr;
use std::time::{Duration, Instant};

/// Simplified TCP state machine. We don't need full RFC 793 fidelity for a
/// firewall (we're not a TCP stack) -- we only need enough state to (a) let
/// return traffic for connections we approved flow, and (b) tear down state
/// promptly on FIN/RST so the table doesn't fill with zombie entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcpState {
    SynSent,
    SynAckSeen,
    Established,
    FinWait,
    /// Reserved for a future explicit close-confirmation state (tracking the
    /// final ACK of a FIN/FIN-ACK exchange); currently entries transition
    /// straight from FinWait to expiry via the transitory timeout.
    #[allow(dead_code)]
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FlowKey {
    pub proto: u8, // 6=TCP 17=UDP 1/58=ICMP
    pub ip_lo: IpAddr,
    pub port_lo: u16,
    pub ip_hi: IpAddr,
    pub port_hi: u16,
}

impl FlowKey {
    /// Build a direction-agnostic key so both legs of a flow hash to the same
    /// bucket. "lo"/"hi" is just a total order on (ip, port), not client/server.
    pub fn new(a_ip: IpAddr, a_port: u16, b_ip: IpAddr, b_port: u16, proto: u8) -> Self {
        if (a_ip, a_port) <= (b_ip, b_port) {
            FlowKey {
                proto,
                ip_lo: a_ip,
                port_lo: a_port,
                ip_hi: b_ip,
                port_hi: b_port,
            }
        } else {
            FlowKey {
                proto,
                ip_lo: b_ip,
                port_lo: b_port,
                ip_hi: a_ip,
                port_hi: a_port,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct FlowEntry {
    pub state: TcpState,
    pub last_seen: Instant,
    pub packets: u64,
    pub bytes: u64,
}

pub struct ConnTrack {
    table: DashMap<FlowKey, FlowEntry>,
    cfg: ConntrackConfig,
}

pub enum LookupResult {
    /// No existing flow -- this is a candidate "new connection", subject to
    /// full policy evaluation.
    New,
    /// Matches an existing tracked flow -- fast-path accept without walking
    /// the rule list, mirroring how ASIC/NPU-backed firewalls short-circuit
    /// established flows.
    Established,
    /// Table is full; caller should apply backpressure (drop) rather than
    /// let an attacker exhaust memory via connection-table flooding.
    TableFull,
}

impl ConnTrack {
    pub fn new(cfg: ConntrackConfig) -> Self {
        Self {
            table: DashMap::new(),
            cfg,
        }
    }

    fn proto_num(proto: L4Proto) -> u8 {
        match proto {
            L4Proto::Tcp => 6,
            L4Proto::Udp => 17,
            L4Proto::Icmp => 1,
            L4Proto::Other(n) => n,
        }
    }

    pub fn lookup_or_admit(&self, pkt: &ParsedPacket) -> LookupResult {
        let proto = Self::proto_num(pkt.proto);
        let key = FlowKey::new(pkt.src_ip, pkt.src_port, pkt.dst_ip, pkt.dst_port, proto);

        if let Some(mut entry) = self.table.get_mut(&key) {
            entry.last_seen = Instant::now();
            entry.packets += 1;
            entry.bytes += pkt.payload_len as u64;
            if let Some(flags) = pkt.tcp_flags {
                if flags.rst || flags.fin {
                    entry.state = TcpState::FinWait;
                } else if flags.syn && flags.ack {
                    entry.state = TcpState::SynAckSeen;
                } else if entry.state == TcpState::SynAckSeen {
                    entry.state = TcpState::Established;
                }
            }
            return LookupResult::Established;
        }

        if self.table.len() >= self.cfg.max_entries {
            return LookupResult::TableFull;
        }
        LookupResult::New
    }

    /// Called after a rule-based ACCEPT decision on a genuinely new flow, so
    /// subsequent packets (including the reply direction) short-circuit
    /// straight to LookupResult::Established.
    pub fn record_new(&self, pkt: &ParsedPacket) {
        let proto = Self::proto_num(pkt.proto);
        let key = FlowKey::new(pkt.src_ip, pkt.src_port, pkt.dst_ip, pkt.dst_port, proto);
        let state = match pkt.tcp_flags {
            Some(f) if f.syn && !f.ack => TcpState::SynSent,
            _ => TcpState::Established, // UDP/ICMP have no handshake
        };
        self.table.insert(
            key,
            FlowEntry {
                state,
                last_seen: Instant::now(),
                packets: 1,
                bytes: pkt.payload_len as u64,
            },
        );
    }

    /// Sweep expired entries. Run periodically from a background thread.
    /// Timeout depends on protocol/state: short-lived for handshakes and UDP
    /// pseudo-flows, long for confirmed-established TCP -- this mirrors the
    /// tiered conntrack timeouts every NGFW uses to bound memory without
    /// killing long-lived connections.
    pub fn sweep_expired(&self) {
        let now = Instant::now();
        let est_timeout = Duration::from_secs(self.cfg.tcp_established_timeout_secs);
        let transitory_timeout = Duration::from_secs(self.cfg.tcp_transitory_timeout_secs);
        let udp_timeout = Duration::from_secs(self.cfg.udp_timeout_secs);

        self.table.retain(|key, entry| {
            let age = now.duration_since(entry.last_seen);
            let timeout = if key.proto == 6 {
                if entry.state == TcpState::Established {
                    est_timeout
                } else {
                    transitory_timeout
                }
            } else {
                udp_timeout
            };
            age < timeout
        });
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }
}
