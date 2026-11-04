//! The containment side of the gate. Deliberately kept tiny and
//! dependency-free of `queue.rs`'s in-memory state: authorization is
//! decided *only* by walking the `ontology_engine` audit graph for
//! `flow_id`, so an approval can be verified independently of (and by a
//! different process than) whatever produced it.
//!
//! This module does not perform any actual network/host containment
//! action -- that integration point (firewall rule push, EDR isolate host,
//! etc.) is deliberately left to the caller via [`ContainmentAction`], since
//! it depends on infrastructure this crate has no knowledge of.

use crate::types::Verdict;
use ontology_engine::engine::OntologyEngine;
use std::fmt;

pub trait ContainmentAction {
    /// Actually perform containment for `flow_id` (e.g. push a firewall
    /// rule, isolate a host via EDR). Implementations should be idempotent:
    /// this may be called more than once for the same flow.
    fn execute(&self, flow_id: &str) -> std::result::Result<(), Box<dyn std::error::Error>>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContainmentDecision {
    /// Approved: exactly one resolved decision authorizing containment was
    /// found, and no resolved decision on the same flow contradicts it.
    Approved { decision_id: String },
    /// No decision authorizing containment exists yet (still pending, or
    /// resolved with a different verdict).
    NotApproved,
    /// More than one resolved decision on this flow disagrees about
    /// containment. This should not happen under normal operation --
    /// `ReviewQueue` only allows one active decision per flow -- but a
    /// containment executor must not guess in the face of contradictory
    /// audit records, so this is surfaced rather than silently resolved.
    Conflicting { decision_ids: Vec<String> },
}

impl fmt::Display for ContainmentDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContainmentDecision::Approved { decision_id } => {
                write!(f, "approved (decision {decision_id})")
            }
            ContainmentDecision::NotApproved => write!(f, "not approved"),
            ContainmentDecision::Conflicting { decision_ids } => {
                write!(f, "conflicting decisions: {}", decision_ids.join(", "))
            }
        }
    }
}

pub struct ContainmentExecutor<A: ContainmentAction> {
    action: A,
}

impl<A: ContainmentAction> ContainmentExecutor<A> {
    pub fn new(action: A) -> Self {
        Self { action }
    }

    /// Reads the audit graph directly (not any cached queue state) to
    /// determine whether containment is authorized for `flow_id`.
    pub fn check(&self, engine: &OntologyEngine, flow_id: &str) -> ContainmentDecision {
        let contain_decisions: Vec<String> = crate::engine_bridge::decisions_for_flow(engine, flow_id)
            .into_iter()
            .filter(|d| d.verdict == Verdict::ExecuteContainment)
            .map(|d| d.decision_id)
            .collect();

        match contain_decisions.len() {
            0 => ContainmentDecision::NotApproved,
            1 => ContainmentDecision::Approved {
                decision_id: contain_decisions.into_iter().next().unwrap(),
            },
            _ => ContainmentDecision::Conflicting {
                decision_ids: contain_decisions,
            },
        }
    }

    /// Executes containment for `flow_id` if and only if `check` returns
    /// `Approved`. Returns the decision that authorized it.
    pub fn execute_if_approved(
        &self,
        engine: &OntologyEngine,
        flow_id: &str,
    ) -> std::result::Result<ContainmentDecision, ContainmentError> {
        let decision = self.check(engine, flow_id);
        match &decision {
            ContainmentDecision::Approved { .. } => {
                self.action
                    .execute(flow_id)
                    .map_err(ContainmentError::ActionFailed)?;
                Ok(decision)
            }
            ContainmentDecision::NotApproved => Err(ContainmentError::NotApproved(flow_id.to_string())),
            ContainmentDecision::Conflicting { decision_ids } => {
                Err(ContainmentError::Conflicting(flow_id.to_string(), decision_ids.clone()))
            }
        }
    }
}

#[derive(Debug)]
pub enum ContainmentError {
    NotApproved(String),
    Conflicting(String, Vec<String>),
    ActionFailed(Box<dyn std::error::Error>),
}

impl fmt::Display for ContainmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContainmentError::NotApproved(flow_id) => {
                write!(f, "refusing to contain '{flow_id}': no approved decision on the audit graph")
            }
            ContainmentError::Conflicting(flow_id, ids) => write!(
                f,
                "refusing to contain '{flow_id}': conflicting decisions {}",
                ids.join(", ")
            ),
            ContainmentError::ActionFailed(e) => write!(f, "containment action failed: {e}"),
        }
    }
}

impl std::error::Error for ContainmentError {}

/// A no-op [`ContainmentAction`] that logs instead of acting. Useful for
/// dry-run CLI invocations and for tests.
pub struct LoggingContainmentAction;

impl ContainmentAction for LoggingContainmentAction {
    fn execute(&self, flow_id: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
        tracing::warn!(flow_id, "CONTAINMENT (dry-run): would isolate/block flow");
        Ok(())
    }
}
