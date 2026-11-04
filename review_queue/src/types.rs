//! Core domain types: a classifier's per-flow prediction, the reason a
//! prediction was (or wasn't) routed to a human, the resulting audit
//! decision, and the state machine that connects them.
//!
//! Design intent: the classifier's output alone must never be sufficient to
//! trigger containment. Every [`FlowPrediction`] either (a) auto-resolves
//! because it is unambiguous and low-stakes (confidently benign, or a
//! confidently-classified attack that cleared the trigger thresholds with a
//! wide margin), or (b) is queued as [`ReviewState::Pending`] until a human
//! records a [`Verdict`]. [`ReviewState::Resolved`] is the only state from
//! which [`crate::containment::ContainmentExecutor`] will act.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

/// Unix seconds. Kept as a plain integer (rather than pulling in `chrono`)
/// since `ontology_engine::PropertyType` only has `Integer`, and this keeps
/// the audit graph properties directly representable without a shim layer.
pub type Timestamp = i64;

pub fn now() -> Timestamp {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before UNIX_EPOCH")
        .as_secs() as i64
}

/// A single classifier output for one network flow. `predicted_label` is
/// deliberately a plain `String` rather than a closed Rust enum: label sets
/// evolve (new attack categories get added) and the ontology layer this
/// feeds into models categorical data as validated strings, not compiled
/// variants. See the categorical-data discussion this module grew out of.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlowPrediction {
    pub flow_id: String,
    pub predicted_label: String,
    pub confidence: f64,
    /// Second-most-likely label and its confidence, if the classifier
    /// exposes it. Used to catch cases where top-1 confidence looks fine in
    /// isolation but the model was nearly torn between two classes.
    pub runner_up_label: Option<String>,
    pub runner_up_confidence: Option<f64>,
    /// True if the classifier (or an upstream novelty/OOD detector) flagged
    /// this flow as unlike anything in its training distribution.
    pub is_out_of_distribution: bool,
    pub observed_at: Timestamp,
}

impl FlowPrediction {
    /// Confidence margin between the top prediction and the runner-up.
    /// `None` if no runner-up was supplied (treated as "wide margin" by the
    /// trigger, since there is nothing to be confused with).
    pub fn margin(&self) -> Option<f64> {
        self.runner_up_confidence.map(|r| self.confidence - r)
    }
}

/// Why a prediction was routed to a human, or why an automatic decision was
/// allowed to stand without one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TriggerReason {
    /// Predicted a non-benign class below the confidence floor.
    LowConfidenceAttack,
    /// Top-1 vs runner-up confidence gap was too small, regardless of the
    /// absolute confidence of either.
    NarrowMargin,
    /// Flagged out-of-distribution by an upstream novelty detector.
    NovelPattern,
    /// Explicitly requested by an operator (e.g. re-review after a rule
    /// change), independent of the automatic thresholds.
    ManualFlag,
}

impl fmt::Display for TriggerReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            TriggerReason::LowConfidenceAttack => "low_confidence_attack",
            TriggerReason::NarrowMargin => "narrow_margin",
            TriggerReason::NovelPattern => "novel_pattern",
            TriggerReason::ManualFlag => "manual_flag",
        };
        write!(f, "{s}")
    }
}

/// A human (or, for SLA timeouts, the system acting under a pre-declared
/// fallback policy) reviewer's decision on a flagged flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    /// Confirmed malicious: containment is authorized to act.
    ExecuteContainment,
    /// Reviewed and confirmed benign (or a tolerable false positive): no
    /// action, prediction is closed out.
    Dismiss,
    /// Reviewer could not make the call themselves: bump to a
    /// higher-tier/out-of-band process. Containment does *not* act on this.
    EscalateFurther,
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Verdict::ExecuteContainment => "execute_containment",
            Verdict::Dismiss => "dismiss",
            Verdict::EscalateFurther => "escalate_further",
        };
        write!(f, "{s}")
    }
}

impl Verdict {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "contain" | "execute_containment" | "containment" => Some(Verdict::ExecuteContainment),
            "dismiss" | "benign" => Some(Verdict::Dismiss),
            "escalate" | "escalate_further" => Some(Verdict::EscalateFurther),
            _ => None,
        }
    }
}

/// Who or what produced a [`Verdict`]. Kept distinct from a free-text
/// reviewer id so automated SLA-timeout resolutions are unambiguously
/// distinguishable from human ones in the audit trail.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Reviewer {
    Human(String),
    SystemSlaTimeout,
    SystemAutoResolve,
}

impl fmt::Display for Reviewer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reviewer::Human(id) => write!(f, "human:{id}"),
            Reviewer::SystemSlaTimeout => write!(f, "system:sla_timeout"),
            Reviewer::SystemAutoResolve => write!(f, "system:auto_resolve"),
        }
    }
}

/// A recorded decision on a [`FlowPrediction`]. Immutable once created by
/// convention: the queue never exposes an update path for these, only
/// `record_decision`, which creates a new one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditDecision {
    pub decision_id: String,
    pub flow_id: String,
    pub reviewer: Reviewer,
    pub verdict: Verdict,
    pub rationale: String,
    pub decided_at: Timestamp,
}

/// SLA fallback policy applied when a [`ReviewState::Pending`] item ages out
/// without a human decision. See module docs on the review-queue design:
/// fail-safe never auto-contains; fail-secure will, above a confidence
/// floor, treating an unreviewed high-confidence attack call as actionable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlaPolicy {
    FailSafe,
    FailSecure { auto_contain_above_confidence: u32 }, // stored as confidence * 1000 to keep this Eq
}

/// Lifecycle state of one queued [`FlowPrediction`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReviewState {
    /// Auto-resolved without human involvement because it cleared the
    /// trigger thresholds cleanly (confidently benign, or a confidently and
    /// unambiguously classified attack).
    AutoResolved { decision_id: String },
    /// Waiting for a human.
    Pending { trigger_reason: TriggerReason },
    /// A human has claimed it and is actively looking at it.
    UnderReview {
        trigger_reason: TriggerReason,
        reviewer: String,
        claimed_at: Timestamp,
    },
    /// Resolved, either by a human or by an SLA-timeout fallback.
    Resolved { decision_id: String },
}

impl fmt::Display for ReviewState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReviewState::AutoResolved { .. } => write!(f, "auto_resolved"),
            ReviewState::Pending { .. } => write!(f, "pending"),
            ReviewState::UnderReview { .. } => write!(f, "under_review"),
            ReviewState::Resolved { .. } => write!(f, "resolved"),
        }
    }
}

/// One item in the queue: the prediction plus its current lifecycle state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueItem {
    pub prediction: FlowPrediction,
    pub state: ReviewState,
    pub enqueued_at: Timestamp,
}

/// The outcome of resolving one flow via SLA fallback, returned from
/// [`crate::queue::ReviewQueue::sweep_expired`] so callers get the full
/// decision (verdict, rationale, reviewer) inline rather than having to
/// look it back up.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlaResolution {
    pub prediction: FlowPrediction,
    pub decision: AuditDecision,
}
