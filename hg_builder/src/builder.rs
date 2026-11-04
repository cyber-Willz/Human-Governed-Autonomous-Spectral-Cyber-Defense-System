//! Rolling windowed hyperedge construction (guide §3).
//!
//! `spectral_hypergraph::hypergraph::SpectralHypergraph` is immutable and
//! build-once (`HypergraphBuilder -> build() -> Clone + Send + Sync`, no
//! removal API), so this doesn't try to maintain one long-lived hypergraph
//! -- it maintains lightweight per-family trackers keyed by whatever the
//! guide's table (§3) says the hyperedge's vertices are, and hands off a
//! [`ClosedWindow`] the moment a tracker closes (time boundary or count
//! threshold, whichever fires first). [`crate::scoring::score_window`] is
//! what actually builds and scores the `SpectralHypergraph` for a closed
//! window.

use std::collections::HashMap;
use std::net::IpAddr;
use std::time::{Duration, Instant};

use correlation_engine::HyperedgeKind;
use ingestion::IngestedPacket;

/// A hyperedge family window that has closed and is ready for scoring.
#[derive(Debug, Clone)]
pub struct ClosedWindow {
    pub kind: HyperedgeKind,
    pub window_id: String,
    /// Vertex labels to feed `HypergraphBuilder` -- not necessarily host
    /// IPs (guide §3: ports/subnets/hostnames are legitimate vertex types
    /// too; `SessionCluster` windows below use flow identifiers).
    pub hyperedge_members: Vec<String>,
    /// Which host(s) this window's eventual `CorrelationVerdict` should be
    /// attributed to -- the correlation join key, kept separate from
    /// `hyperedge_members` since a window's graph vertices and its
    /// correlation hosts aren't always the same set (a destination fan-in
    /// window's vertices are the *sources*, but guide §2 also wants the
    /// *destination* host correlated, since it's the one under attack).
    pub correlation_hosts: Vec<IpAddr>,
    /// Flow count or byte volume or duration, per the guide's per-family
    /// weight column -- fed to `HypergraphBuilder::add_hyperedge`'s weight
    /// parameter.
    pub weight: f64,
}

#[derive(Debug, Clone)]
pub struct WindowConfig {
    /// Burst window: near-simultaneous floods (HOIC/LOIC-style).
    pub fast_window: Duration,
    /// Campaign window: slower fan-in/fan-out patterns the fast window
    /// alone would miss (guide §8, "Window sizing").
    pub slow_window: Duration,
    /// Count threshold that closes a fan-in tracker early, before its
    /// window elapses, if a burst is large enough to be worth scoring
    /// immediately rather than waiting out the rest of the window.
    pub fan_in_count_threshold: usize,
    pub fan_out_count_threshold: usize,
    /// Session-cluster window (proxy for an app-layer session -- see
    /// module docs on why this degrades to a `(src_ip, dst_ip)` key rather
    /// than a real TLS/HTTP session id, since `IngestedPacket` doesn't
    /// carry one).
    pub session_window: Duration,
    /// How long a honeytoken co-touch campaign window stays open,
    /// accumulating additional touches from other hosts before closing.
    pub honeytoken_campaign_window: Duration,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            fast_window: Duration::from_secs(10),
            slow_window: Duration::from_secs(300),
            fan_in_count_threshold: 50,
            fan_out_count_threshold: 30,
            session_window: Duration::from_secs(200),
            honeytoken_campaign_window: Duration::from_secs(600),
        }
    }
}

struct FanInTracker {
    first_seen: Instant,
    last_seen: Instant,
    srcs: Vec<IpAddr>,
    byte_volume: u64,
}

struct FanOutTracker {
    first_seen: Instant,
    last_seen: Instant,
    dsts: Vec<IpAddr>,
    distinct_ports: Vec<u16>,
}

struct SessionTracker {
    first_seen: Instant,
    last_seen: Instant,
    flow_ports: Vec<u16>,
    src_ip: IpAddr,
    dst_ip: IpAddr,
}

struct HoneytokenTracker {
    first_seen: Instant,
    last_seen: Instant,
    touching_hosts: Vec<IpAddr>,
}

fn push_unique<T: PartialEq>(v: &mut Vec<T>, item: T) {
    if !v.contains(&item) {
        v.push(item);
    }
}

/// The rolling windowed hyperedge builder. One instance sits between
/// `ingestion`'s flow stream and `hg_builder::scoring` / the SIEM.
pub struct WindowedHyperedgeBuilder {
    config: WindowConfig,
    fan_in_fast: HashMap<(IpAddr, u16), FanInTracker>,
    fan_in_slow: HashMap<(IpAddr, u16), FanInTracker>,
    fan_out_fast: HashMap<IpAddr, FanOutTracker>,
    fan_out_slow: HashMap<IpAddr, FanOutTracker>,
    session: HashMap<(IpAddr, IpAddr), SessionTracker>,
    honeytoken: HashMap<String, HoneytokenTracker>,
    window_seq: u64,
}

impl WindowedHyperedgeBuilder {
    pub fn new(config: WindowConfig) -> Self {
        Self {
            config,
            fan_in_fast: HashMap::new(),
            fan_in_slow: HashMap::new(),
            fan_out_fast: HashMap::new(),
            fan_out_slow: HashMap::new(),
            session: HashMap::new(),
            honeytoken: HashMap::new(),
            window_seq: 0,
        }
    }

    fn next_window_id(&mut self, prefix: &str) -> String {
        self.window_seq += 1;
        format!("{prefix}-{}", self.window_seq)
    }

    /// Called per-flow from `ingestion`'s spectral (or a dedicated fourth)
    /// broadcast subscriber. Returns every window that closed as a result
    /// of this packet -- usually none, occasionally one or more.
    pub fn ingest(&mut self, pkt: &IngestedPacket) -> Vec<ClosedWindow> {
        let now = Instant::now();
        let mut closed = Vec::new();

        self.ingest_fan_in(pkt, now, &mut closed);
        self.ingest_fan_out(pkt, now, &mut closed);
        self.ingest_session(pkt, now, &mut closed);

        closed
    }

    /// A `honeypot_engine` honeytoken touch. Not part of `ingestion`'s flow
    /// stream -- fed separately from the honeypot daemon's own event sink.
    pub fn ingest_honeytoken_touch(
        &mut self,
        campaign_id: &str,
        host: IpAddr,
    ) -> Option<ClosedWindow> {
        let now = Instant::now();
        let closed_early;
        {
            let tracker = self.honeytoken.entry(campaign_id.to_string()).or_insert_with(|| {
                HoneytokenTracker { first_seen: now, last_seen: now, touching_hosts: Vec::new() }
            });
            tracker.last_seen = now;
            push_unique(&mut tracker.touching_hosts, host);
            closed_early = tracker.touching_hosts.len() >= 2
                && now.duration_since(tracker.first_seen) >= self.config.honeytoken_campaign_window;
        }
        if closed_early {
            self.close_honeytoken(campaign_id)
        } else {
            None
        }
    }

    /// Periodic sweep for windows that should close purely on elapsed
    /// time even if no new packet arrives to trigger the check inline
    /// (an idle attacker doesn't get an indefinitely-open window). Call
    /// this on a timer, same cadence as `correlation_engine`'s own sweep.
    pub fn tick(&mut self) -> Vec<ClosedWindow> {
        let now = Instant::now();
        let mut closed = Vec::new();

        let expired_fan_in_fast: Vec<_> = self
            .fan_in_fast
            .iter()
            .filter(|(_, t)| now.duration_since(t.first_seen) >= self.config.fast_window)
            .map(|(k, _)| *k)
            .collect();
        for key in expired_fan_in_fast {
            if let Some(w) = self.close_fan_in(key, true) {
                closed.push(w);
            }
        }

        let expired_fan_in_slow: Vec<_> = self
            .fan_in_slow
            .iter()
            .filter(|(_, t)| now.duration_since(t.first_seen) >= self.config.slow_window)
            .map(|(k, _)| *k)
            .collect();
        for key in expired_fan_in_slow {
            if let Some(w) = self.close_fan_in(key, false) {
                closed.push(w);
            }
        }

        let expired_fan_out_fast: Vec<_> = self
            .fan_out_fast
            .iter()
            .filter(|(_, t)| now.duration_since(t.first_seen) >= self.config.fast_window)
            .map(|(k, _)| *k)
            .collect();
        for key in expired_fan_out_fast {
            if let Some(w) = self.close_fan_out(key, true) {
                closed.push(w);
            }
        }

        let expired_fan_out_slow: Vec<_> = self
            .fan_out_slow
            .iter()
            .filter(|(_, t)| now.duration_since(t.first_seen) >= self.config.slow_window)
            .map(|(k, _)| *k)
            .collect();
        for key in expired_fan_out_slow {
            if let Some(w) = self.close_fan_out(key, false) {
                closed.push(w);
            }
        }

        let expired_sessions: Vec<_> = self
            .session
            .iter()
            .filter(|(_, t)| now.duration_since(t.first_seen) >= self.config.session_window)
            .map(|(k, _)| *k)
            .collect();
        for key in expired_sessions {
            if let Some(w) = self.close_session(key) {
                closed.push(w);
            }
        }

        let expired_honeytoken: Vec<_> = self
            .honeytoken
            .iter()
            .filter(|(_, t)| now.duration_since(t.first_seen) >= self.config.honeytoken_campaign_window)
            .map(|(k, _)| k.clone())
            .collect();
        for campaign in expired_honeytoken {
            if let Some(w) = self.close_honeytoken(&campaign) {
                closed.push(w);
            }
        }

        closed
    }

    // -- fan-in -----------------------------------------------------

    fn ingest_fan_in(&mut self, pkt: &IngestedPacket, now: Instant, closed: &mut Vec<ClosedWindow>) {
        let key = (pkt.flow.dst_ip, pkt.flow.dst_port);
        let bytes = pkt.payload.len() as u64;
        let count_threshold = self.config.fan_in_count_threshold;

        let mut to_close: Vec<bool> = Vec::new();
        for (map, window, is_fast) in [
            (&mut self.fan_in_fast, self.config.fast_window, true),
            (&mut self.fan_in_slow, self.config.slow_window, false),
        ] {
            let tracker = map.entry(key).or_insert_with(|| FanInTracker {
                first_seen: now,
                last_seen: now,
                srcs: Vec::new(),
                byte_volume: 0,
            });
            tracker.last_seen = now;
            tracker.byte_volume += bytes;
            push_unique(&mut tracker.srcs, pkt.flow.src_ip);

            let should_close = tracker.srcs.len() >= count_threshold
                || now.duration_since(tracker.first_seen) >= window;
            if should_close {
                to_close.push(is_fast);
            }
        }

        for is_fast in to_close {
            if let Some(w) = self.close_fan_in(key, is_fast) {
                closed.push(w);
            }
        }
    }

    fn close_fan_in(&mut self, key: (IpAddr, u16), fast: bool) -> Option<ClosedWindow> {
        let map = if fast { &mut self.fan_in_fast } else { &mut self.fan_in_slow };
        let tracker = map.remove(&key)?;
        if tracker.srcs.len() < 2 {
            // Degenerate hyperedge (<2 members): nothing to score. Mirrors
            // `HypergraphBuilder`'s own "correctness first" guarantee --
            // dropped here rather than handed downstream for free.
            return None;
        }
        let (dst_ip, _dst_port) = key;
        let window_id = self.next_window_id(if fast { "fan-in-fast" } else { "fan-in-slow" });
        Some(ClosedWindow {
            kind: HyperedgeKind::DestinationFanIn,
            window_id,
            hyperedge_members: tracker.srcs.iter().map(|ip| ip.to_string()).collect(),
            correlation_hosts: {
                // Attribute to both the destination under attack and every
                // source in the burst -- containment may need to act on
                // either side depending on policy.
                let mut hosts = tracker.srcs.clone();
                push_unique(&mut hosts, dst_ip);
                hosts
            },
            weight: tracker.byte_volume.max(tracker.srcs.len() as u64) as f64,
        })
    }

    // -- fan-out ------------------------------------------------------

    fn ingest_fan_out(&mut self, pkt: &IngestedPacket, now: Instant, closed: &mut Vec<ClosedWindow>) {
        let key = pkt.flow.src_ip;
        let count_threshold = self.config.fan_out_count_threshold;

        let mut to_close: Vec<bool> = Vec::new();
        for (map, window, is_fast) in [
            (&mut self.fan_out_fast, self.config.fast_window, true),
            (&mut self.fan_out_slow, self.config.slow_window, false),
        ] {
            let tracker = map.entry(key).or_insert_with(|| FanOutTracker {
                first_seen: now,
                last_seen: now,
                dsts: Vec::new(),
                distinct_ports: Vec::new(),
            });
            tracker.last_seen = now;
            push_unique(&mut tracker.dsts, pkt.flow.dst_ip);
            push_unique(&mut tracker.distinct_ports, pkt.flow.dst_port);

            let should_close = tracker.dsts.len() >= count_threshold
                || now.duration_since(tracker.first_seen) >= window;
            if should_close {
                to_close.push(is_fast);
            }
        }

        for is_fast in to_close {
            if let Some(w) = self.close_fan_out(key, is_fast) {
                closed.push(w);
            }
        }
    }

    fn close_fan_out(&mut self, src_ip: IpAddr, fast: bool) -> Option<ClosedWindow> {
        let map = if fast { &mut self.fan_out_fast } else { &mut self.fan_out_slow };
        let tracker = map.remove(&src_ip)?;
        if tracker.dsts.len() < 2 {
            return None;
        }
        let window_id = self.next_window_id(if fast { "fan-out-fast" } else { "fan-out-slow" });
        Some(ClosedWindow {
            kind: HyperedgeKind::SourceFanOut,
            window_id,
            hyperedge_members: tracker.dsts.iter().map(|ip| ip.to_string()).collect(),
            correlation_hosts: {
                let mut hosts = tracker.dsts.clone();
                push_unique(&mut hosts, src_ip);
                hosts
            },
            weight: tracker.distinct_ports.len() as f64,
        })
    }

    // -- session cluster ------------------------------------------------

    fn ingest_session(&mut self, pkt: &IngestedPacket, now: Instant, closed: &mut Vec<ClosedWindow>) {
        let key = (pkt.flow.src_ip, pkt.flow.dst_ip);
        let window = self.config.session_window;
        let tracker = self.session.entry(key).or_insert_with(|| SessionTracker {
            first_seen: now,
            last_seen: now,
            flow_ports: Vec::new(),
            src_ip: key.0,
            dst_ip: key.1,
        });
        tracker.last_seen = now;
        push_unique(&mut tracker.flow_ports, pkt.flow.src_port);

        if now.duration_since(tracker.first_seen) >= window {
            if let Some(w) = self.close_session(key) {
                closed.push(w);
            }
        }
    }

    fn close_session(&mut self, key: (IpAddr, IpAddr)) -> Option<ClosedWindow> {
        let tracker = self.session.remove(&key)?;
        if tracker.flow_ports.len() < 2 {
            // A single flow isn't a "cluster" -- nothing distinguishes it
            // from ordinary per-flow autoencoder scoring.
            return None;
        }
        let duration = tracker.last_seen.duration_since(tracker.first_seen).as_secs_f64();
        let window_id = self.next_window_id("session-cluster");
        Some(ClosedWindow {
            kind: HyperedgeKind::SessionCluster,
            window_id,
            // Vertices here are flow instances (src_ip:src_port), not
            // hosts -- guide §3's note that vertices don't have to be IPs.
            // This is also the family's known simplification: without an
            // app-layer session id from a real TLS/HTTP/keep-alive parse,
            // "same (src_ip, dst_ip) pair, many short-lived flows" is the
            // closest proxy `IngestedPacket`'s 5-tuple gives us.
            hyperedge_members: tracker
                .flow_ports
                .iter()
                .map(|p| format!("{}:{}", tracker.src_ip, p))
                .collect(),
            correlation_hosts: vec![tracker.src_ip, tracker.dst_ip],
            weight: duration,
        })
    }

    // -- honeytoken co-touch ---------------------------------------------

    fn close_honeytoken(&mut self, campaign_id: &str) -> Option<ClosedWindow> {
        let tracker = self.honeytoken.remove(campaign_id)?;
        if tracker.touching_hosts.len() < 2 {
            return None;
        }
        let window_id = self.next_window_id("honeytoken-co-touch");
        Some(ClosedWindow {
            kind: HyperedgeKind::HoneytokenCoTouch,
            window_id,
            hyperedge_members: tracker.touching_hosts.iter().map(|ip| ip.to_string()).collect(),
            correlation_hosts: tracker.touching_hosts.clone(),
            weight: 1.0, // binary (touched=1) per the guide's table
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use ingestion::{Direction, FlowKey, Protocol};
    use std::net::Ipv4Addr;
    use std::time::SystemTime;

    fn pkt(src: u8, dst_ip: Ipv4Addr, dst_port: u16) -> IngestedPacket {
        IngestedPacket {
            flow: FlowKey {
                src_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, src)),
                dst_ip: IpAddr::V4(dst_ip),
                src_port: 40000 + src as u16,
                dst_port,
                protocol: Protocol::Tcp,
            },
            captured_at: SystemTime::now(),
            ifindex: 1,
            direction: Direction::Ingress,
            payload: Bytes::from_static(b"x"),
            seq: 0,
        }
    }

    #[test]
    fn destination_fan_in_closes_on_count_threshold() {
        let mut cfg = WindowConfig::default();
        cfg.fan_in_count_threshold = 3;
        let mut b = WindowedHyperedgeBuilder::new(cfg);

        let dst = Ipv4Addr::new(192, 168, 1, 1);
        let mut all_closed = Vec::new();
        for src in 1..=3u8 {
            all_closed.extend(b.ingest(&pkt(src, dst, 80)));
        }

        let fan_ins: Vec<_> =
            all_closed.iter().filter(|w| w.kind == HyperedgeKind::DestinationFanIn).collect();
        assert!(!fan_ins.is_empty(), "expected at least one fan-in window to close");
        let w = fan_ins[0];
        assert_eq!(w.hyperedge_members.len(), 3);
        assert!(w.correlation_hosts.iter().any(|h| *h == IpAddr::V4(dst)));
    }

    #[test]
    fn source_fan_out_closes_on_count_threshold() {
        let mut cfg = WindowConfig::default();
        cfg.fan_out_count_threshold = 2;
        let mut b = WindowedHyperedgeBuilder::new(cfg);

        let mut all_closed = Vec::new();
        all_closed.extend(b.ingest(&pkt(9, Ipv4Addr::new(10, 0, 0, 100), 445)));
        all_closed.extend(b.ingest(&pkt(9, Ipv4Addr::new(10, 0, 0, 101), 445)));

        let fan_outs: Vec<_> =
            all_closed.iter().filter(|w| w.kind == HyperedgeKind::SourceFanOut).collect();
        assert!(!fan_outs.is_empty());
        assert_eq!(fan_outs[0].hyperedge_members.len(), 2);
    }

    #[test]
    fn single_flow_never_produces_a_degenerate_hyperedge() {
        let mut b = WindowedHyperedgeBuilder::new(WindowConfig::default());
        let closed = b.ingest(&pkt(1, Ipv4Addr::new(10, 0, 0, 50), 22));
        assert!(closed.is_empty(), "a single flow has <2 members in every family, so nothing should close yet");
    }

    #[test]
    fn honeytoken_co_touch_requires_at_least_two_hosts() {
        let mut b = WindowedHyperedgeBuilder::new(WindowConfig::default());
        let h1: IpAddr = "10.0.0.5".parse().unwrap();
        assert!(b.ingest_honeytoken_touch("campaign-1", h1).is_none());
    }
}
