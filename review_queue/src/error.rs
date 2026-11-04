//! Error types for the review queue.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReviewQueueError {
    #[error("flow prediction '{0}' was already ingested")]
    DuplicateFlow(String),

    #[error("flow prediction '{0}' was not found in the queue")]
    FlowNotFound(String),

    #[error("flow '{flow_id}' is not pending review (current state: {state}); a decision cannot be recorded")]
    NotPending { flow_id: String, state: String },

    #[error("verdict rationale must not be empty for flow '{0}' (audit trail requires a reason)")]
    EmptyRationale(String),

    #[error("reviewer id must not be empty for flow '{0}'")]
    EmptyReviewer(String),

    #[error("underlying ontology graph error: {0}")]
    Ontology(#[from] ontology_engine::error::OntologyError),

    #[error("persistence I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("persistence (de)serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, ReviewQueueError>;
