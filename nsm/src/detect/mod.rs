pub mod beacon;
pub mod bounded;
pub mod dns_tunnel;
pub mod portscan;
pub mod signature;
pub mod synflood;

use crate::alert::Alert;
use crate::packet::PacketMeta;
use std::collections::HashSet;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

/// Bundles every detector so the capture loop only needs to call one
/// `analyze()` per packet.
pub struct DetectionEngine {
    pub portscan: portscan::PortScanDetector,
    pub synflood: synflood::SynFloodDetector,
    pub dns_tunnel: dns_tunnel::DnsTunnelDetector,
    pub beacon: beacon::BeaconDetector,
    pub signatures: signature::SignatureEngine,
}

impl DetectionEngine {
    /// `local_ips` should contain every address this host owns
    /// (loopback + all interface IPs) so detectors can tell "someone
    /// connecting to us" apart from "us connecting out" -- the two
    /// need very different port-scan thresholds to avoid flagging
    /// ordinary browsing as an attack.
    pub fn new(local_ips: HashSet<IpAddr>) -> Self {
        let local_ips = Arc::new(local_ips);
        Self {
            portscan: portscan::PortScanDetector::new(
                local_ips,
                Duration::from_secs(10),
                portscan::Thresholds { port_threshold: 20, host_threshold: 15 },
                // Outbound thresholds are deliberately looser: loading a
                // single modern webpage can legitimately touch a dozen-plus
                // distinct hosts (CDNs, ad/tracking networks) in seconds.
                portscan::Thresholds { port_threshold: 60, host_threshold: 40 },
            ),
            synflood: synflood::SynFloodDetector::new(Duration::from_secs(5), 100),
            dns_tunnel: dns_tunnel::DnsTunnelDetector::new(Duration::from_secs(30), 60),
            beacon: beacon::BeaconDetector::new(),
            signatures: signature::SignatureEngine::with_default_ruleset(),
        }
    }

    pub fn analyze(&self, pkt: &PacketMeta) -> Vec<Alert> {
        let mut alerts = Vec::new();
        if let Some(a) = self.portscan.observe(pkt) {
            alerts.push(a);
        }
        if let Some(a) = self.portscan.observe_distributed(pkt) {
            alerts.push(a);
        }
        if let Some(a) = self.synflood.observe(pkt) {
            alerts.push(a);
        }
        if let Some(a) = self.dns_tunnel.observe(pkt) {
            alerts.push(a);
        }
        if let Some(a) = self.beacon.observe(pkt) {
            alerts.push(a);
        }
        alerts.extend(self.signatures.scan(pkt));
        alerts
    }
}
