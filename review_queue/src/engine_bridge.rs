//! Translates [`FlowPrediction`]/[`AuditDecision`] to and from
//! `ontology_engine` instances, and implements the graph query that
//! [`crate::containment::ContainmentExecutor`] relies on.

use crate::schema::{self, AUDIT_DECISION_TYPE, FLOW_PREDICTION_TYPE, REVIEW_OF_LINK};
use crate::types::{AuditDecision, FlowPrediction, Reviewer, Verdict};
use ontology_engine::prelude::*;
use std::collections::HashMap;

pub fn record_prediction(engine: &OntologyEngine, pred: &FlowPrediction) -> ontology_engine::error::Result<()> {
    let properties = HashMap::from([
        ("flow_id".to_string(), PropertyValue::String(pred.flow_id.clone())),
        (
            "predicted_label".to_string(),
            PropertyValue::String(pred.predicted_label.clone()),
        ),
        (
            "confidence_millis".to_string(),
            PropertyValue::Integer(schema::to_millis(pred.confidence)),
        ),
        (
            "observed_at".to_string(),
            PropertyValue::Integer(pred.observed_at),
        ),
    ]);
    engine.create_object_instance(ObjectInstance::new(
        pred.flow_id.clone(),
        FLOW_PREDICTION_TYPE,
        properties,
    ))
}

pub fn record_decision(engine: &OntologyEngine, decision: &AuditDecision) -> ontology_engine::error::Result<()> {
    let properties = HashMap::from([
        (
            "decision_id".to_string(),
            PropertyValue::String(decision.decision_id.clone()),
        ),
        ("flow_id".to_string(), PropertyValue::String(decision.flow_id.clone())),
        (
            "reviewer".to_string(),
            PropertyValue::String(decision.reviewer.to_string()),
        ),
        (
            "verdict".to_string(),
            PropertyValue::String(decision.verdict.to_string()),
        ),
        (
            "rationale".to_string(),
            PropertyValue::String(decision.rationale.clone()),
        ),
        (
            "decided_at".to_string(),
            PropertyValue::Integer(decision.decided_at),
        ),
    ]);
    engine.create_object_instance(ObjectInstance::new(
        decision.decision_id.clone(),
        AUDIT_DECISION_TYPE,
        properties,
    ))?;
    engine.create_link(LinkInstance::new(
        REVIEW_OF_LINK,
        decision.decision_id.clone(),
        decision.flow_id.clone(),
    ))
}

/// Every [`AuditDecision`] recorded against `flow_id`, read back through the
/// graph (not the queue's own JSON cache) -- this is the query
/// [`crate::containment::ContainmentExecutor`] uses to authorize action, so
/// it deliberately goes through the same traversal path an independent
/// auditor would use.
pub fn decisions_for_flow(engine: &OntologyEngine, flow_id: &str) -> Vec<AuditDecision> {
    engine
        .traverse(flow_id, REVIEW_OF_LINK, Direction::Incoming)
        .into_iter()
        .filter_map(|instance| decision_from_instance(&instance))
        .collect()
}

/// Every [`AuditDecision`] ever recorded, read back through the graph.
/// Used by [`crate::store`] to persist the full audit trail alongside the
/// queue's own item map.
pub fn all_decisions(engine: &OntologyEngine) -> Vec<AuditDecision> {
    engine
        .list_instances_by_type(AUDIT_DECISION_TYPE)
        .iter()
        .filter_map(decision_from_instance)
        .collect()
}

fn decision_from_instance(instance: &ObjectInstance) -> Option<AuditDecision> {
    let get_str = |k: &str| -> Option<String> {
        match instance.properties.get(k)? {
            PropertyValue::String(s) => Some(s.clone()),
            _ => None,
        }
    };
    let get_int = |k: &str| -> Option<i64> {
        match instance.properties.get(k)? {
            PropertyValue::Integer(i) => Some(*i),
            _ => None,
        }
    };

    let reviewer_raw = get_str("reviewer")?;
    let reviewer = if let Some(id) = reviewer_raw.strip_prefix("human:") {
        Reviewer::Human(id.to_string())
    } else if reviewer_raw == "system:sla_timeout" {
        Reviewer::SystemSlaTimeout
    } else {
        Reviewer::SystemAutoResolve
    };

    Some(AuditDecision {
        decision_id: get_str("decision_id")?,
        flow_id: get_str("flow_id")?,
        reviewer,
        verdict: Verdict::parse(&get_str("verdict")?)?,
        rationale: get_str("rationale")?,
        decided_at: get_int("decided_at")?,
    })
}
