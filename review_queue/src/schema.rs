//! Registers the audit graph's schema on an [`OntologyEngine`].
//!
//! Two object types and one link type carry the whole audit trail:
//!
//! ```text
//! AuditDecision --ReviewOf--> FlowPrediction
//! ```
//!
//! Putting this in the ontology engine (rather than just the queue's own
//! JSON state) is what lets [`crate::containment::ContainmentExecutor`]
//! prove, via a graph traversal, that a specific approved decision node
//! exists for a flow -- not just that some boolean flag was set somewhere.

use ontology_engine::prelude::*;
use ontology_engine::error::OntologyError;

pub const FLOW_PREDICTION_TYPE: &str = "FlowPrediction";
pub const AUDIT_DECISION_TYPE: &str = "AuditDecision";
pub const REVIEW_OF_LINK: &str = "ReviewOf";

/// Registers the object/link types used by the review queue. Idempotent:
/// "already registered" errors are swallowed so this can be called safely
/// every time a [`crate::queue::ReviewQueue`] is constructed, including
/// against an engine that was already set up in a previous process.
pub fn register(engine: &OntologyEngine) -> Result<()> {
    let flow_prediction = ObjectTypeBuilder::new(FLOW_PREDICTION_TYPE)
        .primary_key("flow_id")
        .property("flow_id", PropertyType::String)
        .property("predicted_label", PropertyType::String)
        .property("confidence_millis", PropertyType::Integer) // confidence * 1000, see note below
        .property("observed_at", PropertyType::Integer)
        .build()
        .expect("static schema is well-formed");
    ignore_already_registered(engine.register_object_type(flow_prediction))?;

    let audit_decision = ObjectTypeBuilder::new(AUDIT_DECISION_TYPE)
        .primary_key("decision_id")
        .property("decision_id", PropertyType::String)
        .property("flow_id", PropertyType::String)
        .property("reviewer", PropertyType::String)
        .property("verdict", PropertyType::String)
        .property("rationale", PropertyType::String)
        .property("decided_at", PropertyType::Integer)
        .build()
        .expect("static schema is well-formed");
    ignore_already_registered(engine.register_object_type(audit_decision))?;

    ignore_already_registered(engine.register_link_type(LinkType::new(
        REVIEW_OF_LINK,
        AUDIT_DECISION_TYPE,
        FLOW_PREDICTION_TYPE,
    )))?;

    Ok(())
}

fn ignore_already_registered(result: ontology_engine::error::Result<()>) -> Result<()> {
    match result {
        Ok(()) => Ok(()),
        Err(OntologyError::ObjectTypeAlreadyRegistered(_))
        | Err(OntologyError::LinkTypeAlreadyRegistered(_)) => Ok(()),
        Err(other) => Err(other),
    }
}

// `ontology_engine::PropertyType` has no `Enum`/constrained-string variant
// (see the categorical-data discussion this crate grew out of) and no
// fixed-point/decimal type either, so `confidence` -- a value in [0, 1] --
// is stored as an integer number of millis (0..=1000) rather than a
// `Float`, purely so exact equality checks in `find_by_property` behave
// predictably. Floats are converted at the crate boundary; nothing outside
// `engine_bridge.rs` should need to know about this representation.
pub fn to_millis(confidence: f64) -> i64 {
    (confidence.clamp(0.0, 1.0) * 1000.0).round() as i64
}

pub type Result<T> = ontology_engine::error::Result<T>;
