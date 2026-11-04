//! `review_queue`: a human-in-the-loop audit/review gate that sits between
//! an NDR classifier's per-flow predictions and containment execution.
//!
//! Predictions are auto-resolved when they clearly clear confidence/margin
//! thresholds, and queued for a human otherwise. Every automatic and human
//! decision is recorded as an `AuditDecision` node on an
//! [`ontology_engine::engine::OntologyEngine`] graph, linked back to the
//! `FlowPrediction` it decided. [`containment::ContainmentExecutor`] will
//! only act on a flow if that graph shows an unambiguous, approved
//! decision -- it never trusts the classifier's raw output directly.
//!
//! ```
//! use review_queue::prelude::*;
//!
//! let queue = ReviewQueue::new(TriggerConfig::default(), SlaPolicy::FailSafe).unwrap();
//!
//! // A confidently benign flow auto-resolves with no human involved.
//! let outcome = queue
//!     .ingest(FlowPrediction {
//!         flow_id: "flow-1".into(),
//!         predicted_label: "Benign".into(),
//!         confidence: 0.99,
//!         runner_up_label: None,
//!         runner_up_confidence: None,
//!         is_out_of_distribution: false,
//!         observed_at: now(),
//!     })
//!     .unwrap();
//! assert!(matches!(outcome, IngestOutcome::AutoResolved { .. }));
//!
//! // A borderline attack call (like the 0.856-confidence Infiltration case
//! // this crate was built to gate) is queued instead of acted on.
//! let outcome = queue
//!     .ingest(FlowPrediction {
//!         flow_id: "flow-2".into(),
//!         predicted_label: "Infiltration".into(),
//!         confidence: 0.856,
//!         runner_up_label: None,
//!         runner_up_confidence: None,
//!         is_out_of_distribution: false,
//!         observed_at: now(),
//!     })
//!     .unwrap();
//! assert!(matches!(outcome, IngestOutcome::QueuedForReview { .. }));
//!
//! // Containment refuses to act until a human resolves it.
//! assert!(matches!(
//!     queue.containment_decision("flow-2"),
//!     ContainmentDecision::NotApproved
//! ));
//!
//! queue
//!     .record_decision("flow-2", "analyst_priya", Verdict::ExecuteContainment, "confirmed via pcap")
//!     .unwrap();
//! assert!(matches!(
//!     queue.containment_decision("flow-2"),
//!     ContainmentDecision::Approved { .. }
//! ));
//! ```

pub mod alert;
pub mod containment;
pub mod engine_bridge;
pub mod error;
pub mod queue;
pub mod schema;
pub mod store;
pub mod trigger;
pub mod types;

pub mod prelude {
    pub use crate::alert::{FanOutAlertSink, LoggingAlertSink, NullAlertSink, SlaBreach, SlaBreachAlertSink};
    pub use crate::containment::{
        ContainmentAction, ContainmentDecision, ContainmentError, ContainmentExecutor,
        LoggingContainmentAction,
    };
    pub use crate::error::{ReviewQueueError, Result};
    pub use crate::queue::{IngestOutcome, QueueStats, ReviewQueue};
    pub use crate::trigger::{ReviewTrigger, TriggerConfig};
    pub use crate::types::{
        now, AuditDecision, FlowPrediction, QueueItem, ReviewState, Reviewer, SlaPolicy,
        SlaResolution, Timestamp, TriggerReason, Verdict,
    };
}
