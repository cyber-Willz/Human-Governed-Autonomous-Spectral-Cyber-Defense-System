//! Alerting for SLA-fallback resolutions.
//!
//! `sweep_expired` resolves flows nobody reviewed in time. Under
//! [`crate::types::SlaPolicy::FailSafe`] that always means `Dismiss` --
//! which is the safe choice against false-positive-driven outages, but it
//! also means a possibly-real attack that never got human eyes is now
//! sitting closed out, unreviewed, indistinguishable in a casual scan from
//! a flow a human actually looked at and cleared. That gap is exactly what
//! this module exists to close: every SLA-fallback resolution goes through
//! an [`SlaBreachAlertSink`], not just the audit graph.

use crate::types::{AuditDecision, FlowPrediction, Verdict};

/// One SLA-fallback resolution, for handing to an alert sink.
#[derive(Debug, Clone, PartialEq)]
pub struct SlaBreach {
    pub prediction: FlowPrediction,
    pub decision: AuditDecision,
}

impl SlaBreach {
    /// True for the case that actually needs a human's attention urgently:
    /// a non-benign call that timed out and was dismissed unreviewed under
    /// fail-safe. A fail-secure auto-containment is also worth a look (risk
    /// of a false-positive outage) but is less time-critical -- something
    /// is already blocked, whereas a fail-safe dismissal means nothing is.
    pub fn is_unreviewed_attack_dismissal(&self) -> bool {
        self.decision.verdict == Verdict::Dismiss
            && self.prediction.predicted_label != "Benign"
    }
}

/// Receives every SLA-fallback resolution as it happens. Implement this to
/// wire paging, Slack, a SIEM webhook, etc. in production; [`sweep_expired`]
/// always calls a sink (defaulting to [`LoggingAlertSink`]) so this can
/// never be silently skipped.
///
/// [`sweep_expired`]: crate::queue::ReviewQueue::sweep_expired
pub trait SlaBreachAlertSink: Send + Sync {
    fn notify(&self, breach: &SlaBreach);
}

/// Default sink: structured `tracing` output. `error!` for a fail-safe (or
/// otherwise unreviewed non-benign) dismissal -- an attack call is now
/// running with no eyes on it and needs urgent follow-up. `warn!` for
/// everything else (confident auto-containments, benign dismissals), which
/// still merits a look but is not itself an active risk.
///
/// This guarantees *something* lands in logs; it does not page anyone.
/// Production deployments should wrap or replace this with a sink that
/// actually reaches an on-call human.
pub struct LoggingAlertSink;

impl SlaBreachAlertSink for LoggingAlertSink {
    fn notify(&self, breach: &SlaBreach) {
        if breach.is_unreviewed_attack_dismissal() {
            tracing::error!(
                flow_id = %breach.prediction.flow_id,
                predicted_label = %breach.prediction.predicted_label,
                confidence = breach.prediction.confidence,
                decision_id = %breach.decision.decision_id,
                "SLA BREACH: non-benign flow dismissed unreviewed under fail-safe policy -- \
                 requires urgent human follow-up, this attack call is not contained and no \
                 one has looked at it"
            );
        } else {
            tracing::warn!(
                flow_id = %breach.prediction.flow_id,
                predicted_label = %breach.prediction.predicted_label,
                confidence = breach.prediction.confidence,
                verdict = %breach.decision.verdict,
                decision_id = %breach.decision.decision_id,
                "SLA fallback resolved a flow with no human review; recommend follow-up audit"
            );
        }
    }
}

/// Discards breaches. Only appropriate for tests, or when a caller is
/// deliberately layering its own sink via [`ReviewQueue::sweep_expired_with_sink`]
/// and wants no duplicate logging.
///
/// [`ReviewQueue::sweep_expired_with_sink`]: crate::queue::ReviewQueue::sweep_expired_with_sink
pub struct NullAlertSink;

impl SlaBreachAlertSink for NullAlertSink {
    fn notify(&self, _breach: &SlaBreach) {}
}

/// Fans one breach out to several sinks (e.g. log *and* page). Order is
/// call order; a panic in one sink is not caught, so keep sinks infallible.
pub struct FanOutAlertSink {
    sinks: Vec<Box<dyn SlaBreachAlertSink>>,
}

impl FanOutAlertSink {
    pub fn new(sinks: Vec<Box<dyn SlaBreachAlertSink>>) -> Self {
        Self { sinks }
    }
}

impl SlaBreachAlertSink for FanOutAlertSink {
    fn notify(&self, breach: &SlaBreach) {
        for sink in &self.sinks {
            sink.notify(breach);
        }
    }
}
