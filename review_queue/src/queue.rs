//! Orchestrates the full lifecycle: classifier prediction in, either an
//! automatic resolution or a human-reviewed one out, all recorded on the
//! `ontology_engine` audit graph.

use crate::engine_bridge;
use crate::error::{ReviewQueueError, Result};
use crate::schema;
use crate::trigger::{ReviewTrigger, TriggerConfig};
use crate::types::{
    now, AuditDecision, FlowPrediction, QueueItem, ReviewState, Reviewer, SlaPolicy,
    SlaResolution, TriggerReason, Verdict,
};
use ontology_engine::engine::OntologyEngine;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Thread-safe. All state-changing operations go through the single
/// internal `Mutex` (matching the "one lock over a multi-step mutation"
/// pattern `ontology_engine::engine` itself uses) so that, e.g., a
/// `record_decision` and a concurrent `sweep_expired` on the same flow can
/// never both succeed.
pub struct ReviewQueue {
    inner: Mutex<Inner>,
    decision_seq: AtomicU64,
}

struct Inner {
    engine: OntologyEngine,
    items: HashMap<String, QueueItem>,
    trigger_cfg: TriggerConfig,
    sla_policy: SlaPolicy,
}

/// Outcome of ingesting one prediction, returned to the caller so a CLI or
/// upstream pipeline can log/alert appropriately without re-deriving it.
#[derive(Debug, Clone, PartialEq)]
pub enum IngestOutcome {
    AutoResolved { verdict: Verdict, decision_id: String },
    QueuedForReview { reason: TriggerReason },
}

impl ReviewQueue {
    pub fn new(trigger_cfg: TriggerConfig, sla_policy: SlaPolicy) -> Result<Self> {
        let engine = OntologyEngine::new();
        schema::register(&engine)?;
        Ok(Self {
            inner: Mutex::new(Inner {
                engine,
                items: HashMap::new(),
                trigger_cfg,
                sla_policy,
            }),
            decision_seq: AtomicU64::new(0),
        })
    }

    fn next_decision_id(&self, flow_id: &str) -> String {
        let seq = self.decision_seq.fetch_add(1, Ordering::Relaxed);
        format!("dec-{flow_id}-{seq}")
    }

    /// Feed one classifier output through the trigger and either resolve it
    /// automatically or queue it for a human. Errors if `flow_id` was
    /// already ingested -- callers must not silently re-submit predictions,
    /// since that could be used to paper over a flagged flow with a fresh
    /// "clean" one.
    pub fn ingest(&self, prediction: FlowPrediction) -> Result<IngestOutcome> {
        let mut inner = self.inner.lock().unwrap();

        if inner.items.contains_key(&prediction.flow_id) {
            return Err(ReviewQueueError::DuplicateFlow(prediction.flow_id));
        }

        engine_bridge::record_prediction(&inner.engine, &prediction)?;

        let trigger = ReviewTrigger::evaluate(&prediction, &inner.trigger_cfg);
        let enqueued_at = now();

        let outcome = match trigger {
            None => {
                let is_benign = prediction.predicted_label == inner.trigger_cfg.benign_label;
                let verdict = if is_benign {
                    Verdict::Dismiss
                } else {
                    Verdict::ExecuteContainment
                };
                let decision_id = self.next_decision_id(&prediction.flow_id);
                let decision = AuditDecision {
                    decision_id: decision_id.clone(),
                    flow_id: prediction.flow_id.clone(),
                    reviewer: Reviewer::SystemAutoResolve,
                    verdict,
                    rationale: format!(
                        "auto-resolved: predicted '{}' at {:.3} confidence cleared all review \
                         thresholds (min_attack_confidence={:.2}, min_margin={:.2}, \
                         min_benign_confidence={:.2})",
                        prediction.predicted_label,
                        prediction.confidence,
                        inner.trigger_cfg.min_attack_confidence,
                        inner.trigger_cfg.min_margin,
                        inner.trigger_cfg.min_benign_confidence
                    ),
                    decided_at: enqueued_at,
                };
                engine_bridge::record_decision(&inner.engine, &decision)?;
                let item = QueueItem {
                    prediction: prediction.clone(),
                    state: ReviewState::AutoResolved {
                        decision_id: decision_id.clone(),
                    },
                    enqueued_at,
                };
                inner.items.insert(prediction.flow_id.clone(), item);
                IngestOutcome::AutoResolved { verdict, decision_id }
            }
            Some(reason) => {
                let item = QueueItem {
                    prediction: prediction.clone(),
                    state: ReviewState::Pending {
                        trigger_reason: reason,
                    },
                    enqueued_at,
                };
                inner.items.insert(prediction.flow_id.clone(), item);
                IngestOutcome::QueuedForReview { reason }
            }
        };

        Ok(outcome)
    }

    /// A human claims a pending item so two reviewers don't duplicate work.
    pub fn claim(&self, flow_id: &str, reviewer: &str) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        let item = inner
            .items
            .get_mut(flow_id)
            .ok_or_else(|| ReviewQueueError::FlowNotFound(flow_id.to_string()))?;

        let trigger_reason = match item.state {
            ReviewState::Pending { trigger_reason } => trigger_reason,
            ref other => {
                return Err(ReviewQueueError::NotPending {
                    flow_id: flow_id.to_string(),
                    state: other.to_string(),
                })
            }
        };

        item.state = ReviewState::UnderReview {
            trigger_reason,
            reviewer: reviewer.to_string(),
            claimed_at: now(),
        };
        Ok(())
    }

    /// Record a human's verdict on a `Pending` or `UnderReview` item. This
    /// is the only path (besides `sweep_expired`) that can move an item
    /// into `Resolved`, and therefore the only path that can make
    /// [`crate::containment::ContainmentExecutor`] willing to act.
    pub fn record_decision(
        &self,
        flow_id: &str,
        reviewer: &str,
        verdict: Verdict,
        rationale: &str,
    ) -> Result<AuditDecision> {
        if reviewer.trim().is_empty() {
            return Err(ReviewQueueError::EmptyReviewer(flow_id.to_string()));
        }
        if rationale.trim().is_empty() {
            return Err(ReviewQueueError::EmptyRationale(flow_id.to_string()));
        }

        let mut inner = self.inner.lock().unwrap();
        {
            let item = inner
                .items
                .get(flow_id)
                .ok_or_else(|| ReviewQueueError::FlowNotFound(flow_id.to_string()))?;
            match item.state {
                ReviewState::Pending { .. } | ReviewState::UnderReview { .. } => {}
                ref other => {
                    return Err(ReviewQueueError::NotPending {
                        flow_id: flow_id.to_string(),
                        state: other.to_string(),
                    })
                }
            }
        }

        let decision_id = self.next_decision_id(flow_id);
        let decision = AuditDecision {
            decision_id: decision_id.clone(),
            flow_id: flow_id.to_string(),
            reviewer: Reviewer::Human(reviewer.to_string()),
            verdict,
            rationale: rationale.to_string(),
            decided_at: now(),
        };
        engine_bridge::record_decision(&inner.engine, &decision)?;

        let item = inner.items.get_mut(flow_id).unwrap();
        item.state = ReviewState::Resolved {
            decision_id: decision_id.clone(),
        };

        Ok(decision)
    }

    /// Move every `Pending`/`UnderReview` item older than `sla_seconds`
    /// (measured from `enqueued_at`) to `Resolved`, applying the queue's
    /// configured [`SlaPolicy`]. Every resolution is reported to a
    /// [`crate::alert::LoggingAlertSink`] as it happens -- see
    /// [`sweep_expired_with_sink`](Self::sweep_expired_with_sink) to wire a
    /// custom sink (paging, Slack, a SIEM) instead of/in addition to logs.
    pub fn sweep_expired(&self, sla_seconds: i64) -> Result<Vec<SlaResolution>> {
        self.sweep_expired_with_sink(sla_seconds, &crate::alert::LoggingAlertSink)
    }

    /// Same as [`sweep_expired`](Self::sweep_expired), but reports every
    /// resolution to the given `alert_sink` instead of the default logging
    /// sink. Under [`SlaPolicy::FailSafe`] this can only ever produce
    /// [`Verdict::Dismiss`] decisions -- containment is never authorized by
    /// a timeout under that policy, only by a human via
    /// [`record_decision`](Self::record_decision).
    pub fn sweep_expired_with_sink(
        &self,
        sla_seconds: i64,
        alert_sink: &dyn crate::alert::SlaBreachAlertSink,
    ) -> Result<Vec<SlaResolution>> {
        let mut inner = self.inner.lock().unwrap();
        let cutoff = now() - sla_seconds;

        let expired_flow_ids: Vec<String> = inner
            .items
            .iter()
            .filter(|(_, item)| {
                matches!(
                    item.state,
                    ReviewState::Pending { .. } | ReviewState::UnderReview { .. }
                ) && item.enqueued_at <= cutoff
            })
            .map(|(id, _)| id.clone())
            .collect();

        let sla_policy = inner.sla_policy;
        let benign_label = inner.trigger_cfg.benign_label.clone();
        let mut resolutions = Vec::with_capacity(expired_flow_ids.len());

        for flow_id in &expired_flow_ids {
            let item = inner.items.get(flow_id).unwrap();
            let prediction = item.prediction.clone();

            // `verdict` here is the single point that decides whether a
            // timeout can ever authorize containment. Under `FailSafe` it
            // is hard-coded to `Dismiss` in every branch below -- there is
            // no code path in this match arm that can produce
            // `ExecuteContainment`, which is what
            // `sla_fail_safe_never_auto_contains_property` (in
            // tests/categorical_isolation_review.rs) exists to pin down.
            let (verdict, rationale) = match sla_policy {
                SlaPolicy::FailSafe => (
                    Verdict::Dismiss,
                    format!(
                        "SLA breach ({sla_seconds}s with no human decision); fail-safe policy \
                         applied: defaulting to dismiss rather than auto-containing. Requires \
                         urgent follow-up review -- this flow was never seen by a human."
                    ),
                ),
                SlaPolicy::FailSecure {
                    auto_contain_above_confidence,
                } => {
                    let confidence_millis = schema::to_millis(prediction.confidence) as u32;
                    if prediction.predicted_label != benign_label
                        && confidence_millis >= auto_contain_above_confidence
                    {
                        (
                            Verdict::ExecuteContainment,
                            format!(
                                "SLA breach ({sla_seconds}s with no human decision); fail-secure \
                                 policy applied: confidence {:.3} >= threshold, auto-containing. \
                                 Requires follow-up review to confirm this wasn't a false \
                                 positive.",
                                prediction.confidence
                            ),
                        )
                    } else {
                        (
                            Verdict::Dismiss,
                            format!(
                                "SLA breach ({sla_seconds}s with no human decision); fail-secure \
                                 policy applied: confidence below auto-contain threshold, \
                                 dismissing. Requires follow-up review."
                            ),
                        )
                    }
                }
            };

            let decision_id = self.next_decision_id(flow_id);
            let decision = AuditDecision {
                decision_id: decision_id.clone(),
                flow_id: flow_id.clone(),
                reviewer: Reviewer::SystemSlaTimeout,
                verdict,
                rationale,
                decided_at: now(),
            };
            engine_bridge::record_decision(&inner.engine, &decision)?;

            let item = inner.items.get_mut(flow_id).unwrap();
            item.state = ReviewState::Resolved { decision_id };

            let breach = crate::alert::SlaBreach {
                prediction,
                decision,
            };
            alert_sink.notify(&breach);
            resolutions.push(SlaResolution {
                prediction: breach.prediction,
                decision: breach.decision,
            });
        }

        Ok(resolutions)
    }

    pub fn get(&self, flow_id: &str) -> Option<QueueItem> {
        self.inner.lock().unwrap().items.get(flow_id).cloned()
    }

    pub fn list_pending(&self) -> Vec<QueueItem> {
        self.inner
            .lock()
            .unwrap()
            .items
            .values()
            .filter(|i| matches!(i.state, ReviewState::Pending { .. } | ReviewState::UnderReview { .. }))
            .cloned()
            .collect()
    }

    pub fn list_all(&self) -> Vec<QueueItem> {
        self.inner.lock().unwrap().items.values().cloned().collect()
    }

    pub fn decisions_for_flow(&self, flow_id: &str) -> Vec<AuditDecision> {
        let inner = self.inner.lock().unwrap();
        engine_bridge::decisions_for_flow(&inner.engine, flow_id)
    }

    /// Checks whether containment is authorized for `flow_id`, per the
    /// audit graph. See [`crate::containment::ContainmentExecutor`] --
    /// this is a thin pass-through so callers don't need to reach into the
    /// engine directly (it stays private to `Inner`).
    pub fn containment_decision(&self, flow_id: &str) -> crate::containment::ContainmentDecision {
        let inner = self.inner.lock().unwrap();
        crate::containment::ContainmentExecutor::new(crate::containment::LoggingContainmentAction)
            .check(&inner.engine, flow_id)
    }

    /// Runs `action` for `flow_id` iff the audit graph shows an approved,
    /// unambiguous containment decision.
    pub fn execute_containment_if_approved<A: crate::containment::ContainmentAction>(
        &self,
        flow_id: &str,
        action: A,
    ) -> std::result::Result<crate::containment::ContainmentDecision, crate::containment::ContainmentError> {
        let inner = self.inner.lock().unwrap();
        crate::containment::ContainmentExecutor::new(action).execute_if_approved(&inner.engine, flow_id)
    }

    /// Total counts by state, for a quick operational summary.
    pub fn stats(&self) -> QueueStats {
        let inner = self.inner.lock().unwrap();
        let mut stats = QueueStats::default();
        for item in inner.items.values() {
            match item.state {
                ReviewState::AutoResolved { .. } => stats.auto_resolved += 1,
                ReviewState::Pending { .. } => stats.pending += 1,
                ReviewState::UnderReview { .. } => stats.under_review += 1,
                ReviewState::Resolved { .. } => stats.resolved += 1,
            }
        }
        stats.total = inner.items.len();
        stats
    }

    /// Used only by [`crate::store`] to rebuild a queue from persisted
    /// state without re-running trigger evaluation against already-decided
    /// items (which could disagree with the original decision if
    /// thresholds were tuned in between runs).
    /// Rebuilds a queue (and replays its full audit graph) from persisted
    /// state. `decisions` must contain every [`AuditDecision`] referenced by
    /// a `decision_id` anywhere in `items`'s states -- [`crate::store`] is
    /// the only caller and guarantees this.
    pub(crate) fn rehydrate(
        trigger_cfg: TriggerConfig,
        sla_policy: SlaPolicy,
        items: HashMap<String, QueueItem>,
        decisions: HashMap<String, AuditDecision>,
        next_decision_seq: u64,
    ) -> Result<Self> {
        let engine = OntologyEngine::new();
        schema::register(&engine)?;

        // Predictions first (link targets must exist before links do), then
        // decisions in the order they were made, so replay produces exactly
        // the graph a live run would have produced.
        let mut ordered_items: Vec<&QueueItem> = items.values().collect();
        ordered_items.sort_by_key(|i| i.enqueued_at);
        for item in ordered_items {
            engine_bridge::record_prediction(&engine, &item.prediction)?;
        }

        let mut ordered_decisions: Vec<&AuditDecision> = decisions.values().collect();
        ordered_decisions.sort_by_key(|d| d.decided_at);
        for decision in ordered_decisions {
            engine_bridge::record_decision(&engine, decision)?;
        }

        Ok(Self {
            inner: Mutex::new(Inner {
                engine,
                items,
                trigger_cfg,
                sla_policy,
            }),
            decision_seq: AtomicU64::new(next_decision_seq),
        })
    }

    pub(crate) fn trigger_cfg(&self) -> TriggerConfig {
        self.inner.lock().unwrap().trigger_cfg.clone()
    }

    pub(crate) fn sla_policy(&self) -> SlaPolicy {
        self.inner.lock().unwrap().sla_policy
    }

    pub(crate) fn next_decision_seq(&self) -> u64 {
        self.decision_seq.load(Ordering::Relaxed)
    }

    pub(crate) fn snapshot_items(&self) -> HashMap<String, QueueItem> {
        self.inner.lock().unwrap().items.clone()
    }

    pub(crate) fn snapshot_decisions(&self) -> HashMap<String, AuditDecision> {
        let inner = self.inner.lock().unwrap();
        engine_bridge::all_decisions(&inner.engine)
            .into_iter()
            .map(|d| (d.decision_id.clone(), d))
            .collect()
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct QueueStats {
    pub total: usize,
    pub auto_resolved: usize,
    pub pending: usize,
    pub under_review: usize,
    pub resolved: usize,
}
