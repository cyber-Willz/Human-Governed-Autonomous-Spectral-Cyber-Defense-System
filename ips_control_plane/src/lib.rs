//! Guide §6: the closed feedback loop.
//!
//! `correlation_engine::CorrelationVerdict` -> `review_queue::FlowPrediction`
//! (human-in-the-loop gate) -> on an approved `ExecuteContainment` decision
//! -> `rustwall::quarantine::Quarantine::ban`. The correlation engine never
//! calls `Quarantine` directly -- guide §2's core design point, restated
//! here in code rather than just in a diagram: *nothing* reaches
//! enforcement without going through `review_queue`'s audit graph first,
//! whether that's an instant auto-resolve (a honeytoken touch, or a
//! long-tenured, well-calibrated detector) or a human decision.
//!
//! # Why this doubles as the guide's §6 "observe-only rollout" without a
//! separate feature flag
//!
//! `review_queue::TriggerConfig::default()` auto-resolves an attack call
//! only above 90% confidence with a >=15-point margin over any runner-up.
//! [`correlation_engine::HypergraphSender`]'s confidence formula
//! (`raw / (raw + 1)`, `raw` = the larger of the Fiedler shift and the
//! Ramanujan deviation) only clears 0.90 when `raw >= 9` -- a strong,
//! unambiguous structural anomaly. In practice that means brand-new
//! detection lanes (hypergraph, Ramanujan) start out queuing for human
//! review by default, purely because their scores haven't yet demonstrated
//! the kind of separation a well-tenured, well-calibrated lane (a
//! honeytoken touch at confidence `1.0`, or an XDP block a human already
//! approved) has. That is exactly the "observe-only, then earn
//! enforcement" rollout the guide calls for -- it falls out of confidence
//! calibration rather than needing its own gate to remember to remove
//! later.

use std::collections::{HashMap, HashSet};
use std::net::IpAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use correlation_engine::{
    CorrelationVerdict, FeedbackKind, FeedbackObserver, FeedbackSource, HyperedgeKind, LaneDetail, LaneSource,
    Severity,
};
use parking_lot::RwLock;
use review_queue::prelude::*;
use rl_decision::RlDecisionLayer;
use rustwall::quarantine::Quarantine;
use tokio::sync::mpsc;

/// Bans are keyed by host, not by the (unique, per-verdict) `flow_id` --
/// `RustwallContainmentAction::execute` receives only a `flow_id` string
/// (per `review_queue::containment::ContainmentAction`'s deliberately
/// infrastructure-agnostic signature), so this records what to actually do
/// for each `flow_id` at the point the `FlowPrediction` was built, and
/// looks it back up when (if) containment fires.
struct PlannedBan {
    host: IpAddr,
    duration: Duration,
}

/// Turns a [`CorrelationVerdict`] into a human-readable predicted label for
/// `review_queue`'s (label-set-is-just-strings) trigger evaluation, and a
/// short rationale string for the audit trail.
fn classify(verdict: &CorrelationVerdict) -> (String, String) {
    if verdict.sources.contains(&LaneSource::Honeytoken) {
        return ("HoneytokenTouch".to_string(), "honeytoken touched -- near-ground-truth".to_string());
    }

    // Prefer the most specific label available; fall back through the
    // lanes in roughly "most certain first" order.
    for e in &verdict.evidence {
        if let LaneDetail::Xdp { reason, human_approved: true, .. } = &e.detail {
            return ("XdpHumanApprovedBlock".to_string(), reason.clone());
        }
    }
    for e in &verdict.evidence {
        if let LaneDetail::ClamAv { signature, quarantined: true, .. } = &e.detail {
            return ("MalwareQuarantined".to_string(), format!("ClamAV signature match: {signature}"));
        }
    }
    for e in &verdict.evidence {
        if let LaneDetail::Hypergraph { hyperedge_kind, window_id, fiedler_shift, ramanujan_deviation, .. } =
            &e.detail
        {
            let label = match hyperedge_kind {
                HyperedgeKind::DestinationFanIn => "HypergraphDestinationFanIn",
                HyperedgeKind::SourceFanOut => "HypergraphSourceFanOut",
                HyperedgeKind::SessionCluster => "HypergraphSessionCluster",
                HyperedgeKind::HoneytokenCoTouch => "HypergraphHoneytokenCoTouch",
            };
            return (
                label.to_string(),
                format!(
                    "window {window_id}: fiedler_shift={fiedler_shift:.3}, ramanujan_deviation={:.3}",
                    ramanujan_deviation.unwrap_or(0.0)
                ),
            );
        }
    }
    for e in &verdict.evidence {
        if let LaneDetail::Spectral { anomaly_score, .. } = &e.detail {
            return ("SpectralAnomaly".to_string(), format!("autoencoder anomaly_score={anomaly_score:.3}"));
        }
    }
    for e in &verdict.evidence {
        if let LaneDetail::Connectome { score, trend, bins, scrutiny_elevated } = &e.detail {
            return (
                "ConnectomeSequenceAnomaly".to_string(),
                format!(
                    "connectome-LSTM score={score:.3} trend={trend:+.3} over {bins} bins (scrutiny_elevated={scrutiny_elevated})"
                ),
            );
        }
    }
    for e in &verdict.evidence {
        if let LaneDetail::Nsm { detector, message } = &e.detail {
            return (format!("Nsm{}", capitalize(detector)), message.clone());
        }
    }
    ("MultiLaneCorrelation".to_string(), "multiple lanes corroborated without a single dominant signal".to_string())
}

/// `nsm`'s own detector names (`"portscan"`, `"synflood"`, `"dns_tunnel"`,
/// `"beacon"`, `"signature"`) are lowercase, snake_case identifiers meant
/// for machine consumption (see `nsm::detect`'s module docs) -- this just
/// gives them the same `UpperCamelCase` label shape every other lane's
/// classification returns (`SpectralAnomaly`, `HypergraphSourceFanOut`,
/// ...), without hand-listing all five here and needing an update if
/// `nsm` adds a sixth.
fn capitalize(snake_case: &str) -> String {
    snake_case
        .split('_')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

/// Whether any evidence on this verdict represents a structural pattern the
/// hypergraph/Ramanujan pass flagged as deviating from its own expected
/// bound -- the closest analog this pipeline has to "out of distribution"
/// for `review_queue::types::FlowPrediction::is_out_of_distribution`.
fn is_structurally_novel(verdict: &CorrelationVerdict) -> bool {
    verdict.evidence.iter().any(|e| {
        matches!(&e.detail, LaneDetail::Hypergraph { ramanujan_deviation: Some(d), .. } if *d > 0.0)
    })
}

/// Default ban length per severity.
///
/// `Medium`/`Low` used to map to `None`, which meant a verdict of that
/// severity that a human *approved for containment* (or that cleared the
/// auto-resolve floor) hit "no containment plan recorded" and nothing was
/// banned -- the approval silently did nothing. They now get the shortest
/// tier (15 min) so an approved decision always has an effect, while a
/// Medium/Low verdict still needs the same review/confidence gate as any
/// other before it can reach enforcement.
fn ban_duration_for(severity: Severity) -> Option<Duration> {
    match severity {
        Severity::Critical => Some(Duration::from_secs(24 * 3600)),
        Severity::High => Some(Duration::from_secs(2 * 3600)),
        Severity::Medium | Severity::Low => Some(Duration::from_secs(15 * 60)),
    }
}

/// [`review_queue::containment::ContainmentAction`] wired to
/// `rustwall::quarantine::Quarantine`.
/// Addresses that must never be banned, whatever the evidence: banning them
/// is either meaningless or an outage (loopback, unspecified, multicast,
/// broadcast, link-local).
fn never_bannable(ip: IpAddr) -> bool {
    // Judge the canonical spelling: ::ffff:127.0.0.1 is loopback, and an
    // IPv6 link-local address (fe80::/10) is how on-link routing and neighbour
    // discovery work -- banning the gateway's is an outage.
    let ip = match ip {
        IpAddr::V6(a) => a.to_ipv4_mapped().map(IpAddr::V4).unwrap_or(ip),
        v4 => v4,
    };
    match ip {
        IpAddr::V4(a) => a.is_loopback() || a.is_unspecified() || a.is_multicast() || a.is_broadcast() || a.is_link_local(),
        IpAddr::V6(a) => a.is_loopback() || a.is_unspecified() || a.is_multicast() || (a.segments()[0] & 0xffc0) == 0xfe80,
    }
}

pub struct RustwallContainmentAction {
    quarantine: Arc<Quarantine>,
    planned: Arc<RwLock<HashMap<String, PlannedBan>>>,
    /// Operator-declared infrastructure (gateway, DNS, the protected servers
    /// themselves, authorised scanners). Enforced for autonomous containment;
    /// a human reviewer who explicitly approves may override it.
    protected: Arc<RwLock<HashSet<IpAddr>>>,
    enforce_protected: bool,
    suppressed: Arc<AtomicU64>,
}

impl review_queue::containment::ContainmentAction for RustwallContainmentAction {
    fn execute(&self, flow_id: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let plan = {
            let table = self.planned.read();
            table.get(flow_id).map(|p| (p.host, p.duration))
        };
        match plan {
            Some((host, _)) if never_bannable(host) => {
                self.suppressed.fetch_add(1, Ordering::Relaxed);
                Err(format!("refusing to ban non-routable/special address {host}").into())
            }
            Some((host, _)) if self.enforce_protected && self.protected.read().contains(&rustwall::quarantine::ban_key(host)) => {
                self.suppressed.fetch_add(1, Ordering::Relaxed);
                tracing::error!(%flow_id, %host, "containment suppressed: host is on the protected list (needs explicit human approval)");
                Err(format!("host {host} is protected; autonomous ban suppressed").into())
            }
            Some((host, duration)) => {
                // A human-approved ban of a host inside a protected prefix must not
                // take the whole prefix (e.g. the LAN's /64) with it.
                let banned = if self.protected.read().contains(&rustwall::quarantine::ban_key(host)) {
                    self.quarantine.ban_exact(host, duration)
                } else {
                    self.quarantine.ban(host, duration)
                };
                if banned {
                    tracing::warn!(%flow_id, %host, ?duration, "containment: banned host via rustwall quarantine");
                } else {
                    tracing::error!(%flow_id, %host, "containment: quarantine table at capacity, ban NOT recorded");
                }
                Ok(())
            }
            None => {
                // Shouldn't happen -- every `flow_id` this crate hands to
                // `review_queue` has a plan recorded first -- but a
                // containment action must never silently no-op on a
                // decision it was told to execute.
                Err(format!("no containment plan recorded for flow_id '{flow_id}'").into())
            }
        }
    }
}

/// The control plane itself: owns the `review_queue::ReviewQueue`, the
/// shared `rustwall::Quarantine`, and the verdict -> prediction mapping.
pub struct IpsControlPlane {
    queue: Arc<ReviewQueue>,
    quarantine: Arc<Quarantine>,
    planned: Arc<RwLock<HashMap<String, PlannedBan>>>,
    flow_seq: AtomicU64,
    /// flow_id -> (host, lanes, honeytoken-ground-truth), kept so a later
    /// human decision on that flow_id can be turned into a labelled
    /// feedback event for learning lanes. Bounded (see `remember_flow`).
    flow_info: RwLock<HashMap<String, (IpAddr, Vec<LaneSource>, bool)>>,
    protected: Arc<RwLock<HashSet<IpAddr>>>,
    suppressed: Arc<AtomicU64>,
    observers: RwLock<Vec<Arc<dyn FeedbackObserver>>>,
    /// Optional RL decision layer (see `rl_decision`). `None` = the legacy
    /// severity-tier behaviour, exactly as before.
    decision: RwLock<Option<Arc<RlDecisionLayer>>>,
}

const MAX_REMEMBERED_FLOWS: usize = 50_000;

impl IpsControlPlane {
    pub fn new(
        trigger_cfg: TriggerConfig,
        sla_policy: SlaPolicy,
        quarantine: Arc<Quarantine>,
    ) -> review_queue::error::Result<Self> {
        Ok(Self {
            queue: Arc::new(ReviewQueue::new(trigger_cfg, sla_policy)?),
            quarantine,
            planned: Arc::new(RwLock::new(HashMap::new())),
            flow_seq: AtomicU64::new(0),
            flow_info: RwLock::new(HashMap::new()),
            protected: Arc::new(RwLock::new(HashSet::new())),
            suppressed: Arc::new(AtomicU64::new(0)),
            observers: RwLock::new(Vec::new()),
            decision: RwLock::new(None),
        })
    }

    /// Attaches the RL decision layer and subscribes it to labelled
    /// outcomes. What it may actually change depends on its `Mode`
    /// (default `Shadow`: decisions are logged and scored, nothing changes).
    /// It can never bypass `review_queue`, raise confidence, or ban a
    /// protected host -- see `rl_decision::layer`.
    pub fn set_decision_layer(&self, layer: Arc<RlDecisionLayer>) {
        self.add_feedback_observer(layer.clone());
        *self.decision.write() = Some(layer);
    }

    pub fn decision_layer(&self) -> Option<Arc<RlDecisionLayer>> {
        self.decision.read().clone()
    }

    /// Registers a learner that receives labelled outcomes (human
    /// decisions, and auto-resolved containment backed by a
    /// ground-truth-by-construction lane such as a honeytoken).
    /// Declares hosts that autonomous containment must never ban: the
    /// gateway, DNS resolvers, the servers being protected, authorised
    /// scanners. A verdict naming one is capped below the auto-resolve floor,
    /// so it goes to a human, and the ban itself is refused unless a human
    /// explicitly approves it.
    pub fn protect_hosts(&self, hosts: impl IntoIterator<Item = IpAddr>) {
        // Stored as ban keys: protecting one address of an IPv6 /64 protects
        // the /64, because that is the unit a ban applies to.
        self.protected.write().extend(hosts.into_iter().map(rustwall::quarantine::ban_key));
    }

    /// Number of autonomous bans refused because the target was protected or
    /// non-routable.
    pub fn suppressed_bans(&self) -> u64 {
        self.suppressed.load(Ordering::Relaxed)
    }

    fn is_protected(&self, ip: IpAddr) -> bool {
        never_bannable(ip) || self.protected.read().contains(&rustwall::quarantine::ban_key(ip))
    }

    pub fn add_feedback_observer(&self, observer: Arc<dyn FeedbackObserver>) {
        self.observers.write().push(observer);
    }

    fn remember_flow(&self, flow_id: &str, host: IpAddr, lanes: Vec<LaneSource>, ground_truth: bool) {
        let mut m = self.flow_info.write();
        if m.len() >= MAX_REMEMBERED_FLOWS {
            m.clear(); // crude bound; losing labels is safe, unbounded growth is not
        }
        m.insert(flow_id.to_string(), (host, lanes, ground_truth));
    }

    fn notify(&self, host: IpAddr, kind: FeedbackKind, source: FeedbackSource, lanes: &[LaneSource]) {
        for o in self.observers.read().iter() {
            o.on_feedback(host, kind, source, lanes);
        }
    }

    pub fn queue(&self) -> Arc<ReviewQueue> {
        self.queue.clone()
    }

    fn action(&self, enforce_protected: bool) -> RustwallContainmentAction {
        RustwallContainmentAction {
            quarantine: self.quarantine.clone(),
            planned: self.planned.clone(),
            protected: self.protected.clone(),
            enforce_protected,
            suppressed: self.suppressed.clone(),
        }
    }

    /// Drives the loop: consumes `correlation_engine::CorrelationVerdict`s
    /// until the sender side closes.
    pub async fn run(&self, mut verdicts: mpsc::Receiver<CorrelationVerdict>) {
        // Expired bans used to be removed only by rustwall's own main loop; in
        // this pipeline nothing swept them, so dead bans accumulated until the
        // table was full and new bans were rejected.
        let mut sweep = tokio::time::interval(Duration::from_secs(30));
        loop {
            tokio::select! {
                _ = sweep.tick() => self.quarantine.sweep_expired(),
                maybe = verdicts.recv() => match maybe {
                    Some(verdict) => self.handle_verdict(verdict),
                    None => break,
                },
            }
        }
        tracing::info!("ips_control_plane: verdict channel closed, shutting down");
    }

    fn handle_verdict(&self, verdict: CorrelationVerdict) {
        let (predicted_label, rationale) = classify(&verdict);
        let seq = self.flow_seq.fetch_add(1, Ordering::Relaxed);
        // `host` embedded in `flow_id` only for operator legibility in
        // logs/audit UIs; `RustwallContainmentAction` looks the host up
        // from `self.planned`, not by parsing this string.
        let flow_id = format!("corr-{}-{seq}", verdict.host);

        // RL decision layer (if attached): may force human review and/or
        // pick the ban tier, subject to its mode and shield. Everything
        // below -- the confidence gate, review_queue, protected-host
        // refusal -- still applies to whatever it chooses.
        let directive = self.decision.read().as_ref().map(|layer| {
            let d = layer.decide(&verdict, self.is_protected(verdict.host));
            tracing::info!(
                host = %verdict.host, severity = ?verdict.severity, confidence = verdict.confidence, mode = ?d.mode, action = d.action.name(), legacy = d.legacy_action.name(),
                p_malicious = format!("{:.3}", d.p_malicious), "rl_decision"
            );
            d.directive()
        });
        let planned_duration = directive
            .and_then(|d| d.ban_duration)
            .or_else(|| ban_duration_for(verdict.severity));
        if let Some(duration) = planned_duration {
            self.planned.write().insert(flow_id.clone(), PlannedBan { host: verdict.host, duration });
        }
        let force_review = directive.map_or(false, |d| d.force_review);

        let ground_truth = verdict.sources.contains(&LaneSource::Honeytoken);
        self.remember_flow(&flow_id, verdict.host, verdict.sources.clone(), ground_truth);
        let verdict_host = verdict.host;
        let verdict_lanes = verdict.sources.clone();

        let prediction = FlowPrediction {
            flow_id: flow_id.clone(),
            predicted_label,
            // A protected host must never be auto-resolved into containment.
            confidence: if force_review || self.is_protected(verdict.host) {
                (verdict.confidence as f64).min(0.89)
            } else {
                verdict.confidence as f64
            },
            runner_up_label: None,
            runner_up_confidence: None,
            is_out_of_distribution: is_structurally_novel(&verdict),
            observed_at: now(),
        };

        let outcome = match self.queue.ingest(prediction) {
            Ok(outcome) => outcome,
            Err(err) => {
                tracing::error!(%flow_id, %err, "failed to ingest verdict into review queue");
                return;
            }
        };

        tracing::debug!(%flow_id, ?outcome, %rationale, "verdict routed through review_queue");

        match outcome {
            IngestOutcome::AutoResolved { verdict: Verdict::ExecuteContainment, .. } => {
                self.try_contain(&flow_id, false);
                if ground_truth {
                    self.notify(verdict_host, FeedbackKind::Confirmed, FeedbackSource::GroundTruthLane, &verdict_lanes);
                }
            }
            IngestOutcome::AutoResolved { .. } => {
                // Dismissed automatically (e.g. auto-resolved benign) --
                // nothing to do. `EscalateFurther` cannot occur from
                // auto-resolution (see `queue.rs::ingest`), only from a
                // human's `record_decision`.
            }
            IngestOutcome::QueuedForReview { reason } => {
                tracing::info!(%flow_id, %reason, "queued for human review, no containment action taken");
            }
        }
    }

    fn try_contain(&self, flow_id: &str, human_approved: bool) {
        match self.queue.execute_containment_if_approved(flow_id, self.action(!human_approved)) {
            Ok(decision) => tracing::info!(%flow_id, %decision, "containment executed"),
            Err(err) => tracing::error!(%flow_id, %err, "containment did not execute"),
        }
    }

    /// Called once a human records a verdict on a previously-queued
    /// prediction (e.g. from a review UI). Wraps
    /// `ReviewQueue::record_decision` and, if the verdict was
    /// `ExecuteContainment`, drives containment the same way an
    /// auto-resolved verdict does.
    pub fn record_human_decision(
        &self,
        flow_id: &str,
        reviewer: &str,
        verdict: Verdict,
        rationale: &str,
    ) -> review_queue::error::Result<()> {
        self.queue.record_decision(flow_id, reviewer, verdict, rationale)?;
        if verdict == Verdict::ExecuteContainment {
            self.try_contain(flow_id, true);
        }
        let info = self.flow_info.read().get(flow_id).cloned();
        if let Some((host, lanes, _)) = info {
            match verdict {
                Verdict::ExecuteContainment => self.notify(host, FeedbackKind::Confirmed, FeedbackSource::Human, &lanes),
                Verdict::Dismiss => self.notify(host, FeedbackKind::Dismissed, FeedbackSource::Human, &lanes),
                Verdict::EscalateFurther => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use correlation_engine::{CorrelationConfig, CorrelationEngine, FlowKey, HyperedgeKind};
    use std::net::Ipv4Addr;

    fn host() -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(10, 0, 0, 77))
    }

    #[tokio::test]
    async fn honeytoken_touch_auto_contains_without_human_review() {
        let quarantine = Arc::new(Quarantine::new(1024));
        let plane = Arc::new(
            IpsControlPlane::new(TriggerConfig::default(), SlaPolicy::FailSafe, quarantine.clone()).unwrap(),
        );

        let (out_tx, out_rx) = mpsc::channel(8);
        let (engine, senders, _metrics) = CorrelationEngine::new(CorrelationConfig::default(), out_tx, None);
        tokio::spawn(engine.run());

        let plane_clone = plane.clone();
        tokio::spawn(async move { plane_clone.run(out_rx).await });

        senders.honeytoken.submit(host(), "tok-xyz", None);

        // Give both async tasks a moment to process.
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert!(quarantine.is_active(host()), "honeytoken touch should auto-contain via rustwall");
    }

    #[tokio::test]
    async fn low_confidence_hypergraph_signal_queues_instead_of_banning() {
        let quarantine = Arc::new(Quarantine::new(1024));
        let plane = Arc::new(
            IpsControlPlane::new(TriggerConfig::default(), SlaPolicy::FailSafe, quarantine.clone()).unwrap(),
        );

        let (out_tx, out_rx) = mpsc::channel(8);
        let (engine, senders, _metrics) = CorrelationEngine::new(CorrelationConfig::default(), out_tx, None);
        tokio::spawn(engine.run());

        let plane_clone = plane.clone();
        tokio::spawn(async move { plane_clone.run(out_rx).await });

        // A single Medium-severity, single-lane event never reaches
        // `review_queue` at all -- `correlation_engine::maybe_emit` only
        // fires a verdict on `sources.len() >= 2` (multi-lane
        // corroboration) or a single lane at `Severity::Critical`
        // (confirmed by reading `correlation_engine::CorrelationEngine::
        // maybe_emit`, not assumed). So exercising "queued, not banned"
        // needs two corroborating lanes, each moderate on its own --
        // fiedler_shift=1.0 -> confidence = 1/(1+1) = 0.5, and a matching
        // spectral anomaly_score of 0.5 -- both well under the 0.90
        // auto-resolve floor even after they're combined into
        // `mean_confidence()`.
        let flow = FlowKey {
            src_ip: host(),
            dst_ip: host(),
            src_port: 1234,
            dst_port: 443,
            protocol: correlation_engine::Protocol::Tcp,
        };
        senders.spectral.submit(host(), flow, 0.5, 0.0);
        senders.hypergraph.submit(
            &[host()],
            "win-1",
            HyperedgeKind::DestinationFanIn,
            1.0,
            None,
            None,
            0.6,
            5,
        );

        tokio::time::sleep(Duration::from_millis(100)).await;

        assert!(!quarantine.is_active(host()), "a modest, unproven signal should be queued, not auto-contained");
        assert_eq!(plane.queue().stats().pending, 1);
    }

    #[tokio::test]
    async fn protected_host_is_never_auto_banned_but_a_human_can_override() {
        let quarantine = Arc::new(Quarantine::new(1024));
        let plane = Arc::new(
            IpsControlPlane::new(TriggerConfig::default(), SlaPolicy::FailSafe, quarantine.clone()).unwrap(),
        );
        let gateway = IpAddr::V4(std::net::Ipv4Addr::new(10, 0, 0, 1));
        plane.protect_hosts([gateway]);

        let (out_tx, out_rx) = mpsc::channel(8);
        let (engine, senders, _metrics) = CorrelationEngine::new(CorrelationConfig::default(), out_tx, None);
        tokio::spawn(engine.run());
        let p2 = plane.clone();
        tokio::spawn(async move { p2.run(out_rx).await });

        senders.honeytoken.submit(gateway, "tok-1", None);
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(!quarantine.is_active(gateway), "protected host must not be auto-banned");

        let pending = plane.queue().list_pending();
        assert_eq!(pending.len(), 1, "it must go to a human instead");
        let flow_id = pending[0].prediction.flow_id.clone();
        let _ = plane.queue().claim(&flow_id, "alice");
        plane.record_human_decision(&flow_id, "alice", Verdict::ExecuteContainment, "confirmed").unwrap();
        assert!(quarantine.is_active(gateway), "explicit human approval may override the protected list");
    }

    #[tokio::test]
    async fn human_approved_containment_of_a_moderate_verdict_actually_bans() {
        // Regression: Medium/Low severity used to have no planned ban, so a
        // human's "ExecuteContainment" logged "no containment plan recorded"
        // and nothing happened.
        let quarantine = Arc::new(Quarantine::new(1024));
        let plane = Arc::new(
            IpsControlPlane::new(TriggerConfig::default(), SlaPolicy::FailSafe, quarantine.clone()).unwrap(),
        );
        let (out_tx, out_rx) = mpsc::channel(8);
        let (engine, senders, _m) = CorrelationEngine::new(CorrelationConfig::default(), out_tx, None);
        tokio::spawn(engine.run());
        let p2 = plane.clone();
        tokio::spawn(async move { p2.run(out_rx).await });
        let flow = FlowKey { src_ip: host(), dst_ip: host(), src_port: 1, dst_port: 443, protocol: correlation_engine::Protocol::Tcp };
        senders.spectral.submit(host(), flow, 0.5, 0.0);
        senders.hypergraph.submit(&[host()], "w", HyperedgeKind::DestinationFanIn, 1.0, None, None, 0.6, 5);
        tokio::time::sleep(Duration::from_millis(100)).await;
        let pending = plane.queue().list_pending();
        assert_eq!(pending.len(), 1);
        let id = pending[0].prediction.flow_id.clone();
        let _ = plane.queue().claim(&id, "bob");
        plane.record_human_decision(&id, "bob", Verdict::ExecuteContainment, "confirmed").unwrap();
        assert!(quarantine.is_active(host()), "an approved containment must take effect");
    }

    fn rl_plane(mode: rl_decision::Mode) -> (Arc<IpsControlPlane>, Arc<Quarantine>, Arc<RlDecisionLayer>) {
        let quarantine = Arc::new(Quarantine::new(1024));
        let plane = Arc::new(
            IpsControlPlane::new(TriggerConfig::default(), SlaPolicy::FailSafe, quarantine.clone()).unwrap(),
        );
        let layer = Arc::new(RlDecisionLayer::new(rl_decision::RlConfig { mode, ..Default::default() }));
        layer.pretrain_in_sandbox(2000, 3);
        plane.set_decision_layer(layer.clone());
        (plane, quarantine, layer)
    }

    #[tokio::test]
    async fn rl_layer_active_still_contains_honeytoken_and_receives_the_label() {
        let (plane, quarantine, layer) = rl_plane(rl_decision::Mode::Active);
        let (out_tx, out_rx) = mpsc::channel(8);
        let (engine, senders, _m) = CorrelationEngine::new(CorrelationConfig::default(), out_tx, None);
        tokio::spawn(engine.run());
        let p2 = plane.clone();
        tokio::spawn(async move { p2.run(out_rx).await });

        senders.honeytoken.submit(host(), "tok-rl", None);
        tokio::time::sleep(Duration::from_millis(150)).await;

        assert!(quarantine.is_active(host()), "honeytoken must still be contained with the RL layer in Active mode");
        let st = layer.stats();
        assert_eq!(st.decisions, 1);
        assert_eq!(st.labels, 1, "the honeytoken-backed containment must come back as a labelled outcome");
    }

    #[tokio::test]
    async fn rl_layer_cannot_ban_a_protected_host_even_in_active_mode() {
        let (plane, quarantine, layer) = rl_plane(rl_decision::Mode::Active);
        let gateway = IpAddr::V4(std::net::Ipv4Addr::new(10, 0, 0, 1));
        plane.protect_hosts([gateway]);
        let (out_tx, out_rx) = mpsc::channel(8);
        let (engine, senders, _m) = CorrelationEngine::new(CorrelationConfig::default(), out_tx, None);
        tokio::spawn(engine.run());
        let p2 = plane.clone();
        tokio::spawn(async move { p2.run(out_rx).await });

        senders.honeytoken.submit(gateway, "tok-gw", None);
        tokio::time::sleep(Duration::from_millis(150)).await;

        assert!(!quarantine.is_active(gateway));
        assert_eq!(plane.queue().stats().pending, 1, "goes to a human");
        let st = layer.stats();
        assert_eq!(st.action_counts[rl_decision::Action::Escalate.index()], 1, "layer itself chose Escalate");
    }

    #[tokio::test]
    async fn rl_layer_in_shadow_mode_changes_nothing_but_still_scores() {
        let (plane, quarantine, layer) = rl_plane(rl_decision::Mode::Shadow);
        let (out_tx, out_rx) = mpsc::channel(8);
        let (engine, senders, _m) = CorrelationEngine::new(CorrelationConfig::default(), out_tx, None);
        tokio::spawn(engine.run());
        let p2 = plane.clone();
        tokio::spawn(async move { p2.run(out_rx).await });
        senders.honeytoken.submit(host(), "tok-shadow", None);
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(quarantine.is_active(host()));
        assert_eq!(layer.stats().decisions, 1);
    }

    #[test]
    fn ban_key_matches_detector_canon_src() {
        for a in ["198.51.100.7", "::ffff:198.51.100.7", "2001:db8:bad:1::1", "2001:db8:bad:1:aaaa:bbbb:cccc:dddd", "fe80::1", "::1"] {
            let ip: IpAddr = a.parse().unwrap();
            assert_eq!(rustwall::quarantine::ban_key(ip), nsm::detect::bounded::canon_src(ip), "{a}");
        }
    }

    #[test]
    fn mapped_loopback_and_v6_link_local_are_never_bannable() {
        assert!(never_bannable("::ffff:127.0.0.1".parse().unwrap()));
        assert!(never_bannable("fe80::1".parse().unwrap()));
        assert!(!never_bannable("2001:db8::1".parse().unwrap()));
    }

    #[test]
    fn special_addresses_are_never_bannable() {
        assert!(never_bannable(IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)));
        assert!(never_bannable(IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED)));
        assert!(never_bannable(IpAddr::V4(std::net::Ipv4Addr::BROADCAST)));
        assert!(!never_bannable(IpAddr::V4(std::net::Ipv4Addr::new(198, 51, 100, 7))));
    }
}
