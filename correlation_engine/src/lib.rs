//! Correlation fan-in — SIEM correlation engine.
//!
//! Connects the analysis lanes — XDP/`rustwall` enforcement, ClamAV
//! quarantine, spectral-autoencoder anomaly scoring, honeytoken touches,
//! and (new in this integration) `hg_builder`'s hypergraph-window /
//! NBSC-cycle / HMM-belief signal — to the correlation engine. Each lane
//! gets its own typed, non-blocking sender; the engine is the single
//! consumer that fuses events per host into a windowed correlation record
//! and emits a [`CorrelationVerdict`] downstream to `ips_control_plane`.
//!
//! # What changed from the original single-SIEM design
//!
//! - **`LaneSource::Hypergraph`** carries the hyperedge-window signal from
//!   `hg_builder`: the Fiedler-eigenvalue drop / algebraic-connectivity
//!   spike (§4 of the guide), NBSC's `rho_B` non-backtracking spectral
//!   radius, the Ramanujan-bound deviation, and the mean HMM belief of the
//!   window's member hosts, all as first-class fields on one
//!   [`LaneDetail::Hypergraph`] variant — "a single incident record can
//!   carry all four signals" (guide §4), correlated by the same `host`
//!   key the autoencoder's `Spectral` lane already uses.
//! - **`LaneSource::Honeytoken`** carries `honeypot_engine` honeytoken
//!   touches straight through as near-ground-truth, `Critical`-severity,
//!   confidence-1.0 evidence — guide §2: "no spectral scoring needed."
//! - **[`CorrelationVerdict::nearest_known_incident`]**: at emission time
//!   the engine builds the fused 4-D signal vector (autoencoder
//!   reconstruction error, hypergraph-window score, `rho_B`, HMM belief)
//!   from a host's accumulated evidence and queries an optional
//!   [`siem_event_index::PersistentEventIndex`] for the nearest previously
//!   *confirmed* incident — guide §4's fifth signal. Passing `None` for
//!   the index at construction time reproduces the original engine's
//!   behavior exactly (this field is then always `None`), so this is a
//!   strictly additive change, not a breaking one.
//!
//! Everything else — fan-in over `try_send`, host-keyed windowed
//! correlation state, bounded-memory eviction, the two-distinct-lanes /
//! single-critical-lane emission rule — is unchanged from the original
//! design; see the module-level doc comments retained below.
//!
//! - **Fan-in, not fan-out.** Unlike ingestion (one producer, many
//!   consumers), this is many producers, one consumer — a natural fit for
//!   `tokio::sync::mpsc`, which supports multiple cloned senders into a
//!   single receiver.
//! - **Lane sends never block.** Each typed sender uses `try_send`. If the
//!   correlation engine is overloaded and its inbound buffer is full, the
//!   event is dropped and counted — no lane may ever stall waiting on the
//!   SIEM.
//! - **Correlation key is the host**, not the flow.
//! - **Emission rule:** a verdict is emitted the moment two distinct lanes
//!   corroborate the same host within the correlation window, or
//!   immediately for a single `Critical` event.
//! - **Bounded memory under adversarial load.**

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::time::Instant;

use siem_event_index::{EventVector, NearestIncident, PersistentEventIndex, AUTOENCODER_ERROR_IDX, HMM_BELIEF_IDX, LAPLACIAN_WINDOW_SCORE_IDX, RHO_B_IDX};

// ---------------------------------------------------------------------
// Shared flow identity
//
// In the real crate this is `ingestion::FlowKey` — redefined here,
// minimally, so this module compiles and tests standalone, matching the
// pattern the original correlation.rs already used.
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Protocol {
    Tcp,
    Udp,
    Icmp,
    Other(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FlowKey {
    pub src_ip: IpAddr,
    pub dst_ip: IpAddr,
    pub src_port: u16,
    pub dst_port: u16,
    pub protocol: Protocol,
}

// ---------------------------------------------------------------------
// Severity, ordered Low < Medium < High < Critical
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

// ---------------------------------------------------------------------
// Lane-specific event detail
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LaneSource {
    Xdp,
    ClamAv,
    Spectral,
    /// New: `hg_builder`'s hyperedge-window scoring.
    Hypergraph,
    /// New: `honeypot_engine` honeytoken touches.
    Honeytoken,
    /// New: `nsm`'s standalone detectors (port scan, SYN flood, DNS
    /// tunneling, beaconing, signature match) -- see `NsmSender`.
    Nsm,
    /// New (advanced build): `neuro_lane`'s connectome-LSTM temporal
    /// sequence detector. Purely numeric -- derived from per-host traffic
    /// feature sequences, never from attacker-controlled text -- so it is
    /// a legitimate independent corroborating lane.
    Connectome,
}

/// Which of the guide's §3 hyperedge families produced a
/// [`LaneDetail::Hypergraph`] event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HyperedgeKind {
    DestinationFanIn,
    SourceFanOut,
    SessionCluster,
    HoneytokenCoTouch,
}

#[derive(Debug, Clone)]
pub enum LaneDetail {
    Xdp {
        flow: FlowKey,
        reason: String,
        human_approved: bool,
    },
    ClamAv {
        file_name: String,
        signature: String,
        quarantined: bool,
    },
    Spectral {
        flow: FlowKey,
        anomaly_score: f64,
        fiedler_shift: f64,
    },
    /// The hypergraph-window signal (guide §4): all three new scores as
    /// first-class fields on one event, correlated by `host`, so a single
    /// `CorrelationVerdict` can carry autoencoder + hypergraph + NBSC + HMM
    /// evidence together.
    Hypergraph {
        window_id: String,
        hyperedge_kind: HyperedgeKind,
        /// Algebraic-connectivity (Fiedler eigenvalue) drop relative to the
        /// window's own baseline; larger = more anomalous ("these hosts
        /// all started acting as one" per guide §4).
        fiedler_shift: f64,
        /// NBSC's `rho_B`, if computed for this window (fan-out windows
        /// only, per guide §4 -- fan-in windows don't run the NBSC pass).
        rho_b: Option<f64>,
        /// Ramanujan-bound deviation (guide §5), if computed for this
        /// window.
        ramanujan_deviation: Option<f64>,
        /// Mean neural-HMM belief across the window's member hosts (the
        /// vertex-weight input to the scoring, surfaced here too since a
        /// reviewer/analyst wants it alongside the hypergraph scores, not
        /// just as an opaque internal weighting).
        mean_hmm_belief: f64,
        member_count: usize,
    },
    /// A `honeypot_engine` honeytoken touch. Deliberately carries no
    /// spectral score -- guide §2: a touched honeytoken is close to ground
    /// truth on its own.
    Honeytoken {
        token_id: String,
        campaign_window_id: Option<String>,
    },
    /// One `nsm::alert::Alert`, carried through close to verbatim: `nsm`
    /// already does its own severity scoring per detector (port scan,
    /// SYN flood, DNS tunneling, beaconing, signature match), so there's
    /// nothing to recompute here -- see `NsmSender::submit`.
    Nsm {
        detector: String,
        message: String,
    },
    /// A connectome-LSTM sequence score for one host.
    Connectome {
        /// Model output in [0, 1] (calibrated against synthetic data only
        /// -- see `neuro_lane`'s docs).
        score: f64,
        /// Score change over the most recent bins (positive = rising).
        trend: f64,
        /// Number of time bins of history the score is based on.
        bins: usize,
        /// Whether an advisory (LLM bridge or other lane) had raised this
        /// host's scrutiny, i.e. lowered the alert threshold, when the
        /// score was produced.
        scrutiny_elevated: bool,
    },
}

/// A single fused-in event, produced by one of the lanes.
#[derive(Debug, Clone)]
pub struct CorrelationEvent {
    pub host: IpAddr,
    pub source: LaneSource,
    pub severity: Severity,
    /// 0.0-1.0. Reflects the producing lane's own confidence.
    pub confidence: f32,
    pub observed_at: Instant,
    pub detail: LaneDetail,
}

// ---------------------------------------------------------------------
// Correlated output, handed to containment
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrelationReason {
    MultiLaneCorroboration,
    SingleLaneCritical,
}

#[derive(Debug, Clone)]
pub struct CorrelationVerdict {
    pub host: IpAddr,
    pub severity: Severity,
    pub confidence: f32,
    pub sources: Vec<LaneSource>,
    pub evidence: Vec<CorrelationEvent>,
    pub reason: CorrelationReason,
    pub decided_at: Instant,
    /// Guide §4's fifth signal: the nearest previously-*confirmed*
    /// incident to this verdict's fused signal vector, if an
    /// [`siem_event_index::PersistentEventIndex`] was supplied to the
    /// engine and it has any confirmed incidents recorded yet.
    pub nearest_known_incident: Option<NearestIncident>,
}

// ---------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct CorrelationMetrics {
    pub events_received: AtomicU64,
    pub verdicts_emitted: AtomicU64,
    pub verdicts_dropped: AtomicU64,
    pub hosts_evicted_ttl: AtomicU64,
    pub hosts_evicted_capacity: AtomicU64,
    pub xdp_dropped: AtomicU64,
    pub clamav_dropped: AtomicU64,
    pub spectral_dropped: AtomicU64,
    pub hypergraph_dropped: AtomicU64,
    pub honeytoken_dropped: AtomicU64,
    pub nsm_dropped: AtomicU64,
    pub connectome_dropped: AtomicU64,
}

#[derive(Clone)]
pub struct CorrelationMetricsHandle(Arc<CorrelationMetrics>);

impl CorrelationMetricsHandle {
    pub fn snapshot(&self) -> CorrelationMetricsSnapshot {
        let m = &self.0;
        CorrelationMetricsSnapshot {
            events_received: m.events_received.load(Ordering::Relaxed),
            verdicts_emitted: m.verdicts_emitted.load(Ordering::Relaxed),
            verdicts_dropped: m.verdicts_dropped.load(Ordering::Relaxed),
            hosts_evicted_ttl: m.hosts_evicted_ttl.load(Ordering::Relaxed),
            hosts_evicted_capacity: m.hosts_evicted_capacity.load(Ordering::Relaxed),
            xdp_dropped: m.xdp_dropped.load(Ordering::Relaxed),
            clamav_dropped: m.clamav_dropped.load(Ordering::Relaxed),
            spectral_dropped: m.spectral_dropped.load(Ordering::Relaxed),
            hypergraph_dropped: m.hypergraph_dropped.load(Ordering::Relaxed),
            honeytoken_dropped: m.honeytoken_dropped.load(Ordering::Relaxed),
            nsm_dropped: m.nsm_dropped.load(Ordering::Relaxed),
            connectome_dropped: m.connectome_dropped.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CorrelationMetricsSnapshot {
    pub events_received: u64,
    pub verdicts_emitted: u64,
    pub verdicts_dropped: u64,
    pub hosts_evicted_ttl: u64,
    pub hosts_evicted_capacity: u64,
    pub xdp_dropped: u64,
    pub clamav_dropped: u64,
    pub spectral_dropped: u64,
    pub hypergraph_dropped: u64,
    pub honeytoken_dropped: u64,
    pub nsm_dropped: u64,
    pub connectome_dropped: u64,
}

// ---------------------------------------------------------------------
// Priority-aware event channel
//
// Red-team finding: with a single bounded `try_send` channel, an attacker
// who floods low-severity noise (e.g. forged-source signature hits) fills
// the channel and the *real* Critical event -- a honeytoken touch -- is
// dropped with the rest (harness B06: 375,711 of 400,000 events dropped,
// the intrusion never banned). High/Critical events now travel on their own
// channel, which the engine always drains first, so noise can no longer
// crowd out the events that matter.
// ---------------------------------------------------------------------

/// Capacity of the high-priority (>= `Severity::High`) channel.
const PRIORITY_CHANNEL_CAPACITY: usize = 4096;

#[derive(Clone)]
pub struct EventTx {
    normal: mpsc::Sender<CorrelationEvent>,
    prio: mpsc::Sender<CorrelationEvent>,
}

impl EventTx {
    fn try_send(&self, event: CorrelationEvent) -> Result<(), ()> {
        let tx = if event.severity >= Severity::High { &self.prio } else { &self.normal };
        tx.try_send(event).map_err(|_| ())
    }
}

// ---------------------------------------------------------------------
// Typed, non-blocking lane senders
// ---------------------------------------------------------------------

#[derive(Clone)]
pub struct XdpSender {
    tx: EventTx,
    metrics: Arc<CorrelationMetrics>,
}

impl XdpSender {
    pub fn submit(&self, host: IpAddr, flow: FlowKey, reason: impl Into<String>, human_approved: bool) {
        let severity = if human_approved { Severity::Critical } else { Severity::High };
        let event = CorrelationEvent {
            host,
            source: LaneSource::Xdp,
            severity,
            confidence: if human_approved { 1.0 } else { 0.6 },
            observed_at: Instant::now(),
            detail: LaneDetail::Xdp { flow, reason: reason.into(), human_approved },
        };
        if self.tx.try_send(event).is_err() {
            self.metrics.xdp_dropped.fetch_add(1, Ordering::Relaxed);
            tracing::warn!("correlation engine backpressured: dropped XDP event");
        }
    }
}

#[derive(Clone)]
pub struct ClamAvSender {
    tx: EventTx,
    metrics: Arc<CorrelationMetrics>,
}

impl ClamAvSender {
    pub fn submit(&self, host: IpAddr, file_name: impl Into<String>, signature: impl Into<String>, quarantined: bool) {
        let event = CorrelationEvent {
            host,
            source: LaneSource::ClamAv,
            severity: if quarantined { Severity::High } else { Severity::Medium },
            confidence: 1.0,
            observed_at: Instant::now(),
            detail: LaneDetail::ClamAv { file_name: file_name.into(), signature: signature.into(), quarantined },
        };
        if self.tx.try_send(event).is_err() {
            self.metrics.clamav_dropped.fetch_add(1, Ordering::Relaxed);
            tracing::warn!("correlation engine backpressured: dropped ClamAV event");
        }
    }
}

#[derive(Clone)]
pub struct SpectralSender {
    tx: EventTx,
    metrics: Arc<CorrelationMetrics>,
}

impl SpectralSender {
    pub fn submit(&self, host: IpAddr, flow: FlowKey, anomaly_score: f64, fiedler_shift: f64) {
        let severity = if anomaly_score >= 0.9 {
            Severity::Critical
        } else if anomaly_score >= 0.7 {
            Severity::High
        } else if anomaly_score >= 0.4 {
            Severity::Medium
        } else {
            Severity::Low
        };
        let event = CorrelationEvent {
            host,
            source: LaneSource::Spectral,
            severity,
            confidence: anomaly_score.clamp(0.0, 1.0) as f32,
            observed_at: Instant::now(),
            detail: LaneDetail::Spectral { flow, anomaly_score, fiedler_shift },
        };
        if self.tx.try_send(event).is_err() {
            self.metrics.spectral_dropped.fetch_add(1, Ordering::Relaxed);
            tracing::warn!("correlation engine backpressured: dropped spectral event");
        }
    }
}

/// Sender handed to `hg_builder`'s window-scoring task.
#[derive(Clone)]
pub struct HypergraphSender {
    tx: EventTx,
    metrics: Arc<CorrelationMetrics>,
}

impl HypergraphSender {
    /// `member_hosts` are every vertex in the window's hyperedge set that
    /// is a plain IP vertex (per guide §3, some vertices are ports /
    /// subnets / hostnames and don't map to a single `host` correlation
    /// key -- those are still counted in `member_count` but don't each get
    /// their own `CorrelationEvent`; scoring the *window* means the
    /// correlation record naturally centers on whichever host(s) the
    /// window flagged rather than needing one event per member).
    #[allow(clippy::too_many_arguments)]
    pub fn submit(
        &self,
        member_hosts: &[IpAddr],
        window_id: impl Into<String>,
        hyperedge_kind: HyperedgeKind,
        fiedler_shift: f64,
        rho_b: Option<f64>,
        ramanujan_deviation: Option<f64>,
        mean_hmm_belief: f64,
        member_count: usize,
    ) {
        let window_id = window_id.into();
        // Anomaly strength drives severity the same way the autoencoder's
        // score does; fiedler_shift and ramanujan_deviation are both
        // "larger = more anomalous, unbounded above" so squash through the
        // same logistic-ish clamp the rest of the pipeline already uses
        // for anomaly_score, rather than inventing a second scale.
        let raw = fiedler_shift.max(ramanujan_deviation.unwrap_or(0.0));
        let confidence = (raw / (raw + 1.0)).clamp(0.0, 1.0);
        let severity = if confidence >= 0.9 {
            Severity::Critical
        } else if confidence >= 0.7 {
            Severity::High
        } else if confidence >= 0.4 {
            Severity::Medium
        } else {
            Severity::Low
        };

        for &host in member_hosts {
            let event = CorrelationEvent {
                host,
                source: LaneSource::Hypergraph,
                severity,
                confidence: confidence as f32,
                observed_at: Instant::now(),
                detail: LaneDetail::Hypergraph {
                    window_id: window_id.clone(),
                    hyperedge_kind,
                    fiedler_shift,
                    rho_b,
                    ramanujan_deviation,
                    mean_hmm_belief,
                    member_count,
                },
            };
            if self.tx.try_send(event).is_err() {
                self.metrics.hypergraph_dropped.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(%window_id, "correlation engine backpressured: dropped hypergraph event");
            }
        }
    }
}

/// Sender handed to `honeypot_engine`. Honeytoken touches are
/// high-confidence by construction (guide §2) and always `Critical` on
/// their own -- a single touch is sufficient for `SingleLaneCritical`
/// emission, matching how a human-approved XDP block behaves today.
#[derive(Clone)]
pub struct HoneytokenSender {
    tx: EventTx,
    metrics: Arc<CorrelationMetrics>,
}

impl HoneytokenSender {
    pub fn submit(&self, host: IpAddr, token_id: impl Into<String>, campaign_window_id: Option<String>) {
        let event = CorrelationEvent {
            host,
            source: LaneSource::Honeytoken,
            severity: Severity::Critical,
            confidence: 1.0,
            observed_at: Instant::now(),
            detail: LaneDetail::Honeytoken { token_id: token_id.into(), campaign_window_id },
        };
        if self.tx.try_send(event).is_err() {
            self.metrics.honeytoken_dropped.fetch_add(1, Ordering::Relaxed);
            tracing::warn!("correlation engine backpressured: dropped honeytoken event");
        }
    }
}

/// Sender for `nsm`'s standalone detectors (port scan, SYN flood, DNS
/// tunneling, beaconing, signature match -- see `nsm::detect`). `nsm` runs
/// as its own independent sensor (own capture, own flow table, own
/// per-detector state) and already assigns each alert a severity; this
/// lane doesn't recompute anomaly scores the way the spectral/hypergraph
/// lanes do, it maps `nsm`'s existing severity straight across.
///
/// A `nsm::alert::Alert` carries `src_ip` and/or `dst_ip` as `Option`
/// (some detectors, e.g. a destination-keyed SYN-flood alert, may only
/// have one side populated); `submit` takes the already-resolved `host`
/// this event should correlate against, leaving that choice to the
/// caller (see `NsmSender::submit_alert` for the common case of "use
/// `src_ip`, falling back to `dst_ip`").
#[derive(Clone)]
pub struct NsmSender {
    tx: EventTx,
    metrics: Arc<CorrelationMetrics>,
}

impl NsmSender {
    /// Maps `nsm::alert::Severity` onto this crate's four-level
    /// `Severity` plus a confidence scalar placed just inside the
    /// matching bucket of `SpectralSender`/`HypergraphSender`'s own
    /// `severity(confidence)` cutoffs (0.90/0.70/0.40), so recomputing
    /// severity from the confidence this lane reports would reproduce
    /// the same severity, not a different one.
    fn map_severity(s: nsm::alert::Severity) -> (Severity, f32) {
        use nsm::alert::Severity as NsmSeverity;
        match s {
            NsmSeverity::Critical => (Severity::Critical, 0.92),
            NsmSeverity::High => (Severity::High, 0.75),
            NsmSeverity::Medium => (Severity::Medium, 0.5),
            NsmSeverity::Low => (Severity::Low, 0.25),
            NsmSeverity::Info => (Severity::Low, 0.1),
        }
    }

    pub fn submit(&self, host: IpAddr, detector: impl Into<String>, message: impl Into<String>, severity: nsm::alert::Severity) {
        self.submit_attributed(host, detector, message, severity, Attribution::Verified);
    }

    /// Like [`submit`](Self::submit), but states how much the alert's host
    /// can be trusted as *the attacker*.
    ///
    /// Red-team findings this closes: (1) a SYN-flood alert is keyed on the
    /// victim, and the response path auto-banned the victim for 24 h (B01);
    /// (2) UDP/ICMP/bare-SYN source addresses are forgeable, so one forged
    /// packet carrying a Log4Shell string got the default gateway and the DNS
    /// resolver banned (B08/B09) and a few hundred forged packets filled the
    /// ban table (B04). Anything not `Verified` is capped below the
    /// auto-containment floor (High / 0.75): it still reaches a human via
    /// corroboration, but can never trigger an autonomous ban on its own.
    pub fn submit_attributed(
        &self,
        host: IpAddr,
        detector: impl Into<String>,
        message: impl Into<String>,
        severity: nsm::alert::Severity,
        attribution: Attribution,
    ) {
        let (mut severity, mut confidence) = Self::map_severity(severity);
        if attribution != Attribution::Verified && severity > Severity::High {
            severity = Severity::High;
            confidence = confidence.min(0.75);
        }
        let event = CorrelationEvent {
            host,
            source: LaneSource::Nsm,
            severity,
            confidence,
            observed_at: Instant::now(),
            detail: LaneDetail::Nsm { detector: detector.into(), message: message.into() },
        };
        if self.tx.try_send(event).is_err() {
            self.metrics.nsm_dropped.fetch_add(1, Ordering::Relaxed);
            tracing::warn!("correlation engine backpressured: dropped nsm event");
        }
    }

    /// Convenience wrapper over a whole `nsm::alert::Alert`: correlates
    /// against `src_ip` if present, else `dst_ip`, else drops the alert
    /// (logged) since `CorrelationEvent` requires a host to key on. Every
    /// alert `nsm`'s own detectors currently produce sets at least one of
    /// the two, so the drop path is a defensive fallback, not the
    /// common case.
    pub fn submit_alert(&self, alert: &nsm::alert::Alert) {
        let Some(host) = alert.src_ip.or(alert.dst_ip) else {
            tracing::warn!(detector = alert.detector, "nsm alert has neither src_ip nor dst_ip, dropping");
            return;
        };
        let attribution = Attribution::from_alert_fields(alert.src_ip, &alert.extra);
        self.submit_attributed(host, alert.detector, alert.message.clone(), alert.severity, attribution);
    }
}

/// How far an [`NsmSender`] event's `host` can be trusted as the attacker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attribution {
    /// Source address is established-flow evidence (e.g. TCP data with ACK).
    Verified,
    /// Source address is forgeable (UDP, ICMP, bare SYN): do not auto-ban.
    Spoofable,
    /// The alert describes the *victim* (e.g. SYN flood); `host` is the
    /// victim, never an attacker.
    VictimOnly,
}

impl Attribution {
    /// Derives attribution from an `nsm` alert's fields (used both for
    /// in-process alerts and for the NDJSON child-process bridge). Missing
    /// flags default to the *conservative* reading: no source address means
    /// victim-only; an unflagged alert with a source is `Verified`.
    pub fn from_alert_fields(src_ip: Option<IpAddr>, extra: &serde_json::Value) -> Self {
        let flag = |k: &str| extra.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
        if flag("victim_only") || src_ip.is_none() {
            Attribution::VictimOnly
        } else if flag("src_spoofable") {
            Attribution::Spoofable
        } else {
            Attribution::Verified
        }
    }
}

/// Sender for `neuro_lane`'s connectome-LSTM detector.
#[derive(Clone)]
pub struct ConnectomeSender {
    tx: EventTx,
    metrics: Arc<CorrelationMetrics>,
}

impl ConnectomeSender {
    /// Highest confidence this lane may report on its own. Kept strictly
    /// under `review_queue`'s 0.90 auto-resolve floor.
    pub const MAX_CONFIDENCE: f32 = 0.85;

    /// Severity buckets reuse the 0.90/0.70/0.40 cutoffs the other scored
    /// lanes use, **but are capped at `High`**: this lane has only been
    /// validated on synthetic behaviours, so it must never be able to
    /// trigger `SingleLaneCritical` (and thus an automatic ban) by itself.
    /// It earns enforcement the way every new lane does: corroboration by
    /// an independent lane, or a human decision in the review queue.
    pub fn severity_for(score: f64) -> Severity {
        if score >= 0.7 {
            Severity::High
        } else if score >= 0.4 {
            Severity::Medium
        } else {
            Severity::Low
        }
    }

    pub fn submit(&self, host: IpAddr, score: f64, trend: f64, bins: usize, scrutiny_elevated: bool) {
        let score = if score.is_finite() { score.clamp(0.0, 1.0) } else { 0.0 };
        let event = CorrelationEvent {
            host,
            source: LaneSource::Connectome,
            severity: Self::severity_for(score),
            confidence: (score as f32).min(Self::MAX_CONFIDENCE),
            observed_at: Instant::now(),
            detail: LaneDetail::Connectome { score, trend, bins, scrutiny_elevated },
        };
        if self.tx.try_send(event).is_err() {
            self.metrics.connectome_dropped.fetch_add(1, Ordering::Relaxed);
            tracing::warn!("correlation engine backpressured: dropped connectome event");
        }
    }
}

// ---------------------------------------------------------------------
// Analyst-feedback hook (verdict -> labels), consumed by `neuro_lane`
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedbackKind {
    Confirmed,
    Dismissed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedbackSource {
    /// A human reviewer recorded the decision.
    Human,
    /// Auto-resolved, with a ground-truth-by-construction lane
    /// (honeytoken) among the evidence.
    GroundTruthLane,
}

/// Receives the outcome of a verdict so learning components can use it as
/// a label. Implementations must be cheap and non-blocking.
pub trait FeedbackObserver: Send + Sync {
    fn on_feedback(&self, host: IpAddr, kind: FeedbackKind, source: FeedbackSource, lanes: &[LaneSource]);
}

// ---------------------------------------------------------------------
// Per-host correlation state
// ---------------------------------------------------------------------

/// A finalized host may be re-judged, but only on strict severity
/// escalation, and at most this many times (bounds review-queue spam).
const MAX_REEMITS: u8 = 3;

struct HostRecord {
    evidence: Vec<CorrelationEvent>,
    first_seen: Instant,
    last_seen: Instant,
    finalized: bool,
    /// Severity of the most recent verdict emitted for this host.
    emitted_severity: Option<Severity>,
    reemits: u8,
}

impl HostRecord {
    fn new(event: CorrelationEvent) -> Self {
        let now = event.observed_at;
        Self {
            evidence: vec![event],
            first_seen: now,
            last_seen: now,
            finalized: false,
            emitted_severity: None,
            reemits: 0,
        }
    }

    fn push(&mut self, event: CorrelationEvent, max_evidence: usize) {
        self.last_seen = event.observed_at;
        if self.evidence.len() >= max_evidence {
            // Evict the *least significant* evidence (lowest severity, then
            // oldest), never blindly the oldest: an attacker must not be able
            // to push a Critical signal out by emitting 16 noise events.
            if let Some(idx) = self
                .evidence
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| (e.severity, e.observed_at))
                .map(|(i, _)| i)
            {
                self.evidence.remove(idx);
            }
        }
        self.evidence.push(event);
    }

    /// Fused confidence of everything seen for this host.
    ///
    /// The original used the plain mean over all retained events, which let an
    /// attacker *dilute* a ground-truth signal: ten Info events (0.1) followed
    /// by a honeytoken touch (1.0) averaged to 0.18 and was queued instead of
    /// contained (harness B03). Now: take each lane's strongest reading, average
    /// across lanes (many events from one lane count once), and if any evidence
    /// is Critical the verdict is at least as confident as that evidence.
    fn fused_confidence(&self) -> f32 {
        if self.evidence.is_empty() {
            return 0.0;
        }
        let mut per_lane: Vec<(LaneSource, f32)> = Vec::new();
        for e in &self.evidence {
            match per_lane.iter_mut().find(|(l, _)| *l == e.source) {
                Some((_, c)) => *c = c.max(e.confidence),
                None => per_lane.push((e.source, e.confidence)),
            }
        }
        let mean = per_lane.iter().map(|(_, c)| *c).sum::<f32>() / per_lane.len() as f32;
        let critical = self
            .evidence
            .iter()
            .filter(|e| e.severity == Severity::Critical)
            .map(|e| e.confidence)
            .fold(0.0f32, f32::max);
        mean.max(critical)
    }

    fn distinct_sources(&self) -> Vec<LaneSource> {
        let mut sources = Vec::new();
        for e in &self.evidence {
            if !sources.contains(&e.source) {
                sources.push(e.source);
            }
        }
        sources
    }

    fn max_severity(&self) -> Severity {
        self.evidence.iter().map(|e| e.severity).max().unwrap_or(Severity::Low)
    }

    /// Builds the fused 4-D signal vector for `siem_event_index` from
    /// whatever evidence has accumulated so far: max autoencoder
    /// reconstruction-proxy (`anomaly_score`) from `Spectral` events, and
    /// the hypergraph/`rho_B`/HMM fields from the most recent `Hypergraph`
    /// event. Dimensions with no contributing evidence yet default to
    /// `0.0` -- a verdict that fires on `SingleLaneCritical` (e.g. a bare
    /// honeytoken touch) legitimately has no spectral or hypergraph
    /// reading yet, and `0.0` is the correct "no anomaly observed on this
    /// axis" value for the nearest-neighbor query, not a sentinel to
    /// special-case.
    fn signal_vector(&self) -> EventVector {
        let mut v: EventVector = [0.0; 4];
        for e in &self.evidence {
            match &e.detail {
                LaneDetail::Spectral { anomaly_score, .. } => {
                    v[AUTOENCODER_ERROR_IDX] = v[AUTOENCODER_ERROR_IDX].max(*anomaly_score);
                }
                LaneDetail::Hypergraph { fiedler_shift, rho_b, mean_hmm_belief, .. } => {
                    v[LAPLACIAN_WINDOW_SCORE_IDX] = v[LAPLACIAN_WINDOW_SCORE_IDX].max(*fiedler_shift);
                    if let Some(rho) = rho_b {
                        v[RHO_B_IDX] = v[RHO_B_IDX].max(*rho);
                    }
                    v[HMM_BELIEF_IDX] = v[HMM_BELIEF_IDX].max(*mean_hmm_belief);
                }
                _ => {}
            }
        }
        v
    }
}

// ---------------------------------------------------------------------
// Engine configuration
// ---------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CorrelationConfig {
    pub window: Duration,
    /// How long a host that has shown Medium-or-worse evidence stays tracked
    /// after its last event. An attacker who paces stages further apart than
    /// `window` must not get a fresh slate each time (round-2 C15). Evidence
    /// below Medium still expires after `window`.
    pub campaign_window: Duration,
    pub sweep_interval: Duration,
    pub channel_capacity: usize,
    pub max_evidence_per_host: usize,
    pub max_tracked_hosts: usize,
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            window: Duration::from_secs(60),
            campaign_window: Duration::from_secs(6 * 3600),
            sweep_interval: Duration::from_secs(5),
            channel_capacity: 4096,
            max_evidence_per_host: 16,
            max_tracked_hosts: 100_000,
        }
    }
}

// ---------------------------------------------------------------------
// The correlation engine
// ---------------------------------------------------------------------

pub struct CorrelationEngine {
    rx: mpsc::Receiver<CorrelationEvent>,
    prio_rx: mpsc::Receiver<CorrelationEvent>,
    out_tx: mpsc::Sender<CorrelationVerdict>,
    state: HashMap<IpAddr, HostRecord>,
    config: CorrelationConfig,
    metrics: Arc<CorrelationMetrics>,
    event_index: Option<Arc<PersistentEventIndex>>,
}

/// Every lane sender the engine hands back, bundled so call sites don't
/// need a five-tuple destructure.
pub struct LaneSenders {
    pub xdp: XdpSender,
    pub clamav: ClamAvSender,
    pub spectral: SpectralSender,
    pub hypergraph: HypergraphSender,
    pub honeytoken: HoneytokenSender,
    pub nsm: NsmSender,
    pub connectome: ConnectomeSender,
}

impl CorrelationEngine {
    /// Builds the engine plus its lane senders. `out_tx` is supplied by the
    /// caller (`ips_control_plane`'s inbound channel). `event_index` is
    /// optional: pass `None` to run the engine exactly as before this
    /// integration (verdicts' `nearest_known_incident` is then always
    /// `None`).
    pub fn new(
        config: CorrelationConfig,
        out_tx: mpsc::Sender<CorrelationVerdict>,
        event_index: Option<Arc<PersistentEventIndex>>,
    ) -> (Self, LaneSenders, CorrelationMetricsHandle) {
        let (ntx, rx) = mpsc::channel(config.channel_capacity);
        let (ptx, prio_rx) = mpsc::channel(PRIORITY_CHANNEL_CAPACITY);
        let tx = EventTx { normal: ntx, prio: ptx };
        let metrics = Arc::new(CorrelationMetrics::default());

        let senders = LaneSenders {
            xdp: XdpSender { tx: tx.clone(), metrics: metrics.clone() },
            clamav: ClamAvSender { tx: tx.clone(), metrics: metrics.clone() },
            spectral: SpectralSender { tx: tx.clone(), metrics: metrics.clone() },
            hypergraph: HypergraphSender { tx: tx.clone(), metrics: metrics.clone() },
            honeytoken: HoneytokenSender { tx: tx.clone(), metrics: metrics.clone() },
            nsm: NsmSender { tx: tx.clone(), metrics: metrics.clone() },
            connectome: ConnectomeSender { tx, metrics: metrics.clone() },
        };

        let engine =
            Self { rx, prio_rx, out_tx, state: HashMap::new(), config, metrics: metrics.clone(), event_index };

        (engine, senders, CorrelationMetricsHandle(metrics))
    }

    pub async fn run(mut self) {
        let mut sweep = tokio::time::interval(self.config.sweep_interval);
        loop {
            tokio::select! {
                biased;
                _ = sweep.tick() => {
                    self.sweep_expired();
                }
                maybe_prio = self.prio_rx.recv() => {
                    match maybe_prio {
                        Some(event) => self.handle_event(event),
                        None => {
                            // every sender is gone; drain whatever is still queued
                            while let Ok(event) = self.rx.try_recv() {
                                self.handle_event(event);
                            }
                            tracing::info!("correlation engine: all lane senders dropped, shutting down");
                            return;
                        }
                    }
                }
                maybe_event = self.rx.recv() => {
                    match maybe_event {
                        Some(event) => self.handle_event(event),
                        None => {
                            tracing::info!("correlation engine: all lane senders dropped, shutting down");
                            return;
                        }
                    }
                }
            }
        }
    }

    fn handle_event(&mut self, event: CorrelationEvent) {
        use std::collections::hash_map::Entry;

        self.metrics.events_received.fetch_add(1, Ordering::Relaxed);
        // One actor, one record: IPv4-mapped IPv6 -> IPv4, IPv6 -> its /64.
        let mut event = event;
        event.host = nsm::detect::bounded::canon_src(event.host);
        let host = event.host;
        let max_evidence = self.config.max_evidence_per_host;

        match self.state.entry(host) {
            Entry::Vacant(v) => {
                v.insert(HostRecord::new(event));
            }
            Entry::Occupied(mut o) => {
                o.get_mut().push(event, max_evidence);
            }
        }

        // Always re-evaluate: a host that already produced a verdict must
        // still be able to *escalate* (harness B02: a moderate verdict was
        // queued for review, then the same host touched a honeytoken and the
        // engine ignored it because the record was already "finalized").
        self.maybe_emit(host);

        self.enforce_capacity();
    }

    fn maybe_emit(&mut self, host: IpAddr) {
        let Some(record) = self.state.get_mut(&host) else { return };

        let sources = record.distinct_sources();
        let severity = record.max_severity();

        // Distinct NSM detectors (portscan, signature, beacon, dns...) are
        // independent observations even though they share one lane: an attacker
        // who evades the ML lanes but trips three NSM detectors on a verified
        // host must still reach a human (round-2 C16). Counts only Medium+.
        let nsm_detectors = {
            let mut names: Vec<&str> = record
                .evidence
                .iter()
                .filter(|e| e.severity >= Severity::Medium)
                .filter_map(|e| match &e.detail {
                    LaneDetail::Nsm { detector, .. } => Some(detector.as_str()),
                    _ => None,
                })
                .collect();
            names.sort_unstable();
            names.dedup();
            names.len()
        };
        let effective_lanes = sources.len() + nsm_detectors.saturating_sub(1);

        let reason = if effective_lanes >= 2 {
            Some(CorrelationReason::MultiLaneCorroboration)
        } else if severity == Severity::Critical {
            Some(CorrelationReason::SingleLaneCritical)
        } else {
            None
        };

        let Some(reason) = reason else { return };

        // Already judged: only speak again on strict severity escalation.
        if record.finalized {
            let escalated = record.emitted_severity.map_or(true, |prev| severity > prev);
            if !escalated || record.reemits >= MAX_REEMITS {
                return;
            }
            record.reemits += 1;
        }

        let nearest_known_incident = self.event_index.as_ref().and_then(|idx| {
            let vec = record.signal_vector();
            idx.nearest(&vec)
        });

        let verdict = CorrelationVerdict {
            host,
            severity,
            confidence: record.fused_confidence(),
            sources,
            evidence: record.evidence.clone(),
            reason,
            decided_at: Instant::now(),
            nearest_known_incident,
        };

        record.finalized = true;
        record.emitted_severity = Some(severity);

        match self.out_tx.try_send(verdict) {
            Ok(()) => {
                self.metrics.verdicts_emitted.fetch_add(1, Ordering::Relaxed);
            }
            Err(mpsc::error::TrySendError::Full(v)) if v.severity >= Severity::High => {
                // A High/Critical verdict is never silently dropped because the
                // control plane is momentarily busy: hand it to a task that
                // waits for room.
                let tx = self.out_tx.clone();
                let metrics = self.metrics.clone();
                tokio::spawn(async move {
                    if tx.send(v).await.is_ok() {
                        metrics.verdicts_emitted.fetch_add(1, Ordering::Relaxed);
                    } else {
                        metrics.verdicts_dropped.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
            Err(_) => {
                self.metrics.verdicts_dropped.fetch_add(1, Ordering::Relaxed);
                tracing::error!(%host, "containment channel backpressured or closed: verdict dropped");
            }
        }
    }

    fn sweep_expired(&mut self) {
        let now = Instant::now();
        let window = self.config.window;
        let campaign = self.config.campaign_window.max(window);
        let before = self.state.len();
        self.state.retain(|_, record| {
            let keep = if record.max_severity() >= Severity::Medium { campaign } else { window };
            now.duration_since(record.last_seen) < keep
        });
        let evicted = before - self.state.len();
        if evicted > 0 {
            self.metrics.hosts_evicted_ttl.fetch_add(evicted as u64, Ordering::Relaxed);
        }
    }

    fn enforce_capacity(&mut self) {
        let max = self.config.max_tracked_hosts;
        if self.state.len() <= max {
            return;
        }
        // The original scanned the whole table for the single oldest host on
        // *every* event once full -- O(n) per event, which collapsed
        // throughput to ~10k events/s at 20k hosts (harness B07) and so turned
        // the host cap into a CPU-exhaustion lever. Now evict a 5 % batch in
        // one O(n) pass (amortised O(1) per event), weakest first: hosts with
        // no verdict, lowest severity, then least recently seen.
        let batch = (max / 20).max(1);
        let mut keys: Vec<((bool, Severity, Instant), IpAddr)> = self
            .state
            .iter()
            .map(|(h, r)| ((r.finalized, r.max_severity(), r.last_seen), *h))
            .collect();
        let n = batch.min(keys.len().saturating_sub(1));
        if n == 0 {
            return;
        }
        keys.select_nth_unstable_by_key(n - 1, |(k, _)| *k);
        for (_, host) in keys.into_iter().take(n) {
            self.state.remove(&host);
        }
        self.metrics.hosts_evicted_capacity.fetch_add(n as u64, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn host() -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(10, 0, 0, 5))
    }

    fn flow() -> FlowKey {
        FlowKey {
            src_ip: host(),
            dst_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
            src_port: 5555,
            dst_port: 443,
            protocol: Protocol::Tcp,
        }
    }

    #[tokio::test]
    async fn single_sub_critical_lane_does_not_emit() {
        let (out_tx, mut out_rx) = mpsc::channel(8);
        let (engine, senders, _metrics) = CorrelationEngine::new(CorrelationConfig::default(), out_tx, None);
        tokio::spawn(engine.run());

        senders.spectral.submit(host(), flow(), 0.5, 0.1);

        let got = tokio::time::timeout(Duration::from_millis(100), out_rx.recv()).await;
        assert!(got.is_err(), "expected no verdict yet, insufficient evidence");
    }

    #[tokio::test]
    async fn two_lanes_corroborate_and_emit() {
        let (out_tx, mut out_rx) = mpsc::channel(8);
        let (engine, senders, metrics) = CorrelationEngine::new(CorrelationConfig::default(), out_tx, None);
        tokio::spawn(engine.run());

        senders.xdp.submit(host(), flow(), "syn flood", false);
        senders.clamav.submit(host(), "invoice.exe", "Win.Trojan.Generic", true);

        let verdict = tokio::time::timeout(Duration::from_millis(200), out_rx.recv())
            .await
            .expect("verdict should arrive")
            .expect("channel open");

        assert_eq!(verdict.host, host());
        assert_eq!(verdict.reason, CorrelationReason::MultiLaneCorroboration);
        assert_eq!(verdict.sources.len(), 2);
        assert!(verdict.nearest_known_incident.is_none());

        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(metrics.snapshot().verdicts_emitted, 1);
    }

    #[tokio::test]
    async fn honeytoken_touch_emits_immediately_as_single_lane_critical() {
        let (out_tx, mut out_rx) = mpsc::channel(8);
        let (engine, senders, _metrics) = CorrelationEngine::new(CorrelationConfig::default(), out_tx, None);
        tokio::spawn(engine.run());

        senders.honeytoken.submit(host(), "tok-abc123", None);

        let verdict = tokio::time::timeout(Duration::from_millis(200), out_rx.recv())
            .await
            .expect("verdict should arrive")
            .expect("channel open");

        assert_eq!(verdict.reason, CorrelationReason::SingleLaneCritical);
        assert_eq!(verdict.severity, Severity::Critical);
        assert_eq!(verdict.sources, vec![LaneSource::Honeytoken]);
    }

    #[tokio::test]
    async fn hypergraph_lane_carries_all_new_signal_fields_and_annotates_nearest_incident() {
        let index = Arc::new(PersistentEventIndex::in_memory());
        index
            .insert(siem_event_index::IndexedIncident {
                incident_id: "confirmed-hoic-1".into(),
                label: "HOIC".into(),
                vector: [0.9, 0.0, 0.0, 0.8],
                confirmed_at_unix_ms: 0,
            })
            .unwrap();

        let (out_tx, mut out_rx) = mpsc::channel(8);
        let (engine, senders, _metrics) =
            CorrelationEngine::new(CorrelationConfig::default(), out_tx, Some(index));
        tokio::spawn(engine.run());

        // A bare hypergraph reading alone is sub-critical/single-lane, so
        // pair it with a spectral reading to trigger multi-lane emission
        // and populate both signal-vector dimensions.
        senders.spectral.submit(host(), flow(), 0.8, 0.0);
        senders.hypergraph.submit(
            &[host()],
            "win-42",
            HyperedgeKind::DestinationFanIn,
            0.85,
            Some(0.3),
            Some(0.6),
            0.7,
            215,
        );

        let verdict = tokio::time::timeout(Duration::from_millis(200), out_rx.recv())
            .await
            .expect("verdict should arrive")
            .expect("channel open");

        assert!(verdict.sources.contains(&LaneSource::Hypergraph));
        let has_hg_detail = verdict.evidence.iter().any(|e| matches!(e.detail, LaneDetail::Hypergraph { .. }));
        assert!(has_hg_detail);
        let nearest = verdict.nearest_known_incident.expect("index was supplied and non-empty");
        assert_eq!(nearest.incident_id, "confirmed-hoic-1");
    }

    #[tokio::test]
    async fn nsm_critical_alert_is_single_lane_critical() {
        let (out_tx, mut out_rx) = mpsc::channel(8);
        let (engine, senders, _metrics) = CorrelationEngine::new(CorrelationConfig::default(), out_tx, None);
        tokio::spawn(engine.run());

        // nsm's own signature detector already decided this is Critical
        // (e.g. an EICAR-string byte match); the lane shouldn't need a
        // second corroborating lane to raise a verdict, same as honeytoken.
        senders.nsm.submit(host(), "signature", "EICAR test string matched", nsm::alert::Severity::Critical);

        let verdict = tokio::time::timeout(Duration::from_millis(200), out_rx.recv())
            .await
            .expect("verdict should arrive")
            .expect("channel open");

        assert_eq!(verdict.reason, CorrelationReason::SingleLaneCritical);
        assert!(verdict.sources.contains(&LaneSource::Nsm));
        let has_nsm_detail = verdict
            .evidence
            .iter()
            .any(|e| matches!(&e.detail, LaneDetail::Nsm { detector, .. } if detector == "signature"));
        assert!(has_nsm_detail);
    }

    #[tokio::test]
    async fn nsm_medium_alert_alone_does_not_emit_but_corroborates() {
        let (out_tx, mut out_rx) = mpsc::channel(8);
        let (engine, senders, _metrics) = CorrelationEngine::new(CorrelationConfig::default(), out_tx, None);
        tokio::spawn(engine.run());

        // A single Medium port-scan alert (nsm's own default severity for
        // that detector) shouldn't fire alone...
        senders.nsm.submit(host(), "portscan", "20 distinct ports in 10s", nsm::alert::Severity::High);
        let got = tokio::time::timeout(Duration::from_millis(100), out_rx.recv()).await;
        assert!(got.is_err(), "expected no verdict yet, single sub-critical lane");

        // ...but corroborates with a second, independent lane on the same
        // host, same as every other lane in this file's fusion tests.
        senders.spectral.submit(host(), flow(), 0.6, 0.0);
        let verdict = tokio::time::timeout(Duration::from_millis(200), out_rx.recv())
            .await
            .expect("verdict should arrive")
            .expect("channel open");
        assert_eq!(verdict.reason, CorrelationReason::MultiLaneCorroboration);
        assert!(verdict.sources.contains(&LaneSource::Nsm));
        assert!(verdict.sources.contains(&LaneSource::Spectral));
    }

    #[test]
    fn nsm_alert_with_only_dst_ip_still_correlates() {
        let (ntx, _nrx) = mpsc::channel(8);
        let (ptx, mut rx) = mpsc::channel(8); // High severity travels on the priority channel
        let sender = NsmSender { tx: EventTx { normal: ntx, prio: ptx }, metrics: Arc::new(CorrelationMetrics::default()) };
        let alert = nsm::alert::Alert::new(
            nsm::alert::Severity::High,
            "synflood",
            "150 SYNs at one destination",
            None,
            Some(host()),
            serde_json::json!({}),
        );
        sender.submit_alert(&alert);
        let event = rx.try_recv().expect("event should be queued");
        assert_eq!(event.host, host());
        assert_eq!(event.source, LaneSource::Nsm);
    }

    #[tokio::test]
    async fn connectome_lane_alone_can_never_trigger_a_single_lane_verdict() {
        let (out_tx, mut out_rx) = mpsc::channel(8);
        let (engine, senders, _metrics) = CorrelationEngine::new(CorrelationConfig::default(), out_tx, None);
        tokio::spawn(engine.run());
        senders.connectome.submit(host(), 1.0, 0.5, 24, false);
        let got = tokio::time::timeout(Duration::from_millis(100), out_rx.recv()).await;
        assert!(got.is_err(), "a lone connectome score, even 1.0, must not emit");
        // ...but it does corroborate another lane.
        senders.spectral.submit(host(), flow(), 0.5, 0.1);
        let v = tokio::time::timeout(Duration::from_millis(200), out_rx.recv()).await.unwrap().unwrap();
        assert!(v.sources.contains(&LaneSource::Connectome));
        assert!(v.evidence.iter().filter(|e| e.source == LaneSource::Connectome).all(|e| e.confidence <= ConnectomeSender::MAX_CONFIDENCE && e.severity <= Severity::High));
    }

    #[tokio::test]
    async fn escalation_after_a_verdict_is_re_judged() {
        let (out_tx, mut out_rx) = mpsc::channel(8);
        let (engine, senders, _m) = CorrelationEngine::new(CorrelationConfig::default(), out_tx, None);
        tokio::spawn(engine.run());

        senders.spectral.submit(host(), flow(), 0.5, 0.0);
        senders.nsm.submit(host(), "portscan", "scan", nsm::alert::Severity::Medium);
        let first = tokio::time::timeout(Duration::from_millis(200), out_rx.recv()).await.unwrap().unwrap();
        assert_eq!(first.severity, Severity::Medium);

        senders.honeytoken.submit(host(), "tok-1", None);
        let second = tokio::time::timeout(Duration::from_millis(200), out_rx.recv())
            .await
            .expect("escalation must produce a second verdict")
            .unwrap();
        assert_eq!(second.severity, Severity::Critical);
        assert!(second.confidence >= 0.99);
    }

    #[tokio::test]
    async fn same_severity_evidence_does_not_re_emit() {
        let (out_tx, mut out_rx) = mpsc::channel(8);
        let (engine, senders, _m) = CorrelationEngine::new(CorrelationConfig::default(), out_tx, None);
        tokio::spawn(engine.run());
        senders.spectral.submit(host(), flow(), 0.5, 0.0);
        senders.nsm.submit(host(), "portscan", "scan", nsm::alert::Severity::Medium);
        let _ = tokio::time::timeout(Duration::from_millis(200), out_rx.recv()).await.unwrap().unwrap();
        for _ in 0..5 {
            senders.nsm.submit(host(), "portscan", "scan", nsm::alert::Severity::Medium);
        }
        assert!(tokio::time::timeout(Duration::from_millis(100), out_rx.recv()).await.is_err());
    }

    #[tokio::test]
    async fn noise_cannot_dilute_a_critical_signal() {
        let (out_tx, mut out_rx) = mpsc::channel(8);
        let (engine, senders, _m) = CorrelationEngine::new(CorrelationConfig::default(), out_tx, None);
        tokio::spawn(engine.run());
        for _ in 0..12 {
            senders.nsm.submit(host(), "signature", "noise", nsm::alert::Severity::Info);
        }
        senders.honeytoken.submit(host(), "tok-1", None);
        let v = tokio::time::timeout(Duration::from_millis(200), out_rx.recv()).await.unwrap().unwrap();
        assert!(v.confidence >= 0.9, "confidence was diluted to {}", v.confidence);
    }

    #[tokio::test]
    async fn priority_events_survive_a_flooded_normal_channel() {
        let (out_tx, mut out_rx) = mpsc::channel(8);
        let cfg = CorrelationConfig { channel_capacity: 4, ..CorrelationConfig::default() };
        // no engine running yet: the normal channel fills and starts dropping
        let (engine, senders, metrics) = CorrelationEngine::new(cfg, out_tx, None);
        for i in 0..100u8 {
            senders.nsm.submit(IpAddr::V4(Ipv4Addr::new(100, 0, 0, i)), "signature", "noise", nsm::alert::Severity::Info);
        }
        assert!(metrics.snapshot().nsm_dropped > 0, "flood should have overflowed the normal channel");
        senders.honeytoken.submit(host(), "tok-1", None);
        tokio::spawn(engine.run());
        let v = tokio::time::timeout(Duration::from_millis(300), out_rx.recv()).await.unwrap().unwrap();
        assert_eq!(v.host, host());
        assert_eq!(v.severity, Severity::Critical);
    }

    #[tokio::test]
    async fn spoofable_or_victim_only_critical_is_capped_below_auto_containment() {
        let (out_tx, mut out_rx) = mpsc::channel(8);
        let (engine, senders, _m) = CorrelationEngine::new(CorrelationConfig::default(), out_tx, None);
        tokio::spawn(engine.run());
        senders.nsm.submit_attributed(host(), "signature", "forged", nsm::alert::Severity::Critical, Attribution::Spoofable);
        // capped to High / 0.75: alone it is single-lane sub-critical => no verdict
        assert!(tokio::time::timeout(Duration::from_millis(100), out_rx.recv()).await.is_err());
        // ...but with a second lane it corroborates and is judged at High, not Critical
        senders.spectral.submit(host(), flow(), 0.5, 0.0);
        let v = tokio::time::timeout(Duration::from_millis(200), out_rx.recv()).await.unwrap().unwrap();
        assert_eq!(v.severity, Severity::High);
        assert!(v.confidence < 0.9);
    }

    #[tokio::test]
    async fn host_table_eviction_is_batched_and_bounded() {
        let (out_tx, _out_rx) = mpsc::channel(8);
        let cfg = CorrelationConfig { max_tracked_hosts: 100, ..CorrelationConfig::default() };
        let (engine, senders, metrics) = CorrelationEngine::new(cfg, out_tx, None);
        tokio::spawn(engine.run());
        for i in 0..1000u32 {
            let ip = IpAddr::V4(Ipv4Addr::new(100, 1, (i / 256) as u8, (i % 256) as u8));
            senders.nsm.submit(ip, "signature", "noise", nsm::alert::Severity::Info);
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        let m = metrics.snapshot();
        assert_eq!(m.events_received + m.nsm_dropped, 1000);
        assert!(m.hosts_evicted_capacity > 0);
    }
}
