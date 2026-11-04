//! Persists [`ReviewQueue`] state to a JSON file between CLI invocations.
//!
//! `ontology_engine::OntologyEngine` itself is in-memory only and not
//! `Serialize`, so persistence works one layer up: the queue's own
//! `items`/`decisions` are the source of truth on disk, and on load a fresh
//! engine is rebuilt by replaying them (see [`crate::queue::ReviewQueue::rehydrate`]).
//! This also means the on-disk format never depends on ontology_engine's
//! internal representation.

use crate::error::Result;
use crate::queue::ReviewQueue;
use crate::trigger::TriggerConfig;
use crate::types::{AuditDecision, QueueItem, SlaPolicy};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
struct PersistedState {
    /// Bumped if the on-disk schema changes shape, so a future version can
    /// detect and migrate (or refuse to load) an older file instead of
    /// silently misinterpreting it.
    format_version: u32,
    items: HashMap<String, QueueItem>,
    decisions: HashMap<String, AuditDecision>,
    next_decision_seq: u64,
    trigger_cfg: TriggerConfig,
    sla_policy: SlaPolicy,
}

const FORMAT_VERSION: u32 = 1;

/// Loads a queue from `path` if it exists, otherwise creates a fresh one
/// with the given defaults (used only on first run at a given path).
pub fn load_or_new(
    path: impl AsRef<Path>,
    default_trigger_cfg: TriggerConfig,
    default_sla_policy: SlaPolicy,
) -> Result<ReviewQueue> {
    let path = path.as_ref();
    if !path.exists() {
        return ReviewQueue::new(default_trigger_cfg, default_sla_policy);
    }

    let raw = std::fs::read_to_string(path)?;
    let persisted: PersistedState = serde_json::from_str(&raw)?;

    ReviewQueue::rehydrate(
        persisted.trigger_cfg,
        persisted.sla_policy,
        persisted.items,
        persisted.decisions,
        persisted.next_decision_seq,
    )
}

pub fn save(queue: &ReviewQueue, path: impl AsRef<Path>) -> Result<()> {
    let persisted = PersistedState {
        format_version: FORMAT_VERSION,
        items: queue.snapshot_items(),
        decisions: queue.snapshot_decisions(),
        next_decision_seq: queue.next_decision_seq(),
        trigger_cfg: queue.trigger_cfg(),
        sla_policy: queue.sla_policy(),
    };

    let json = serde_json::to_string_pretty(&persisted)?;

    // Write to a temp file and rename, so a crash mid-write (or a
    // concurrent reader) never observes a half-written state file.
    let path = path.as_ref();
    let tmp_path = path.with_extension("json.tmp");
    std::fs::write(&tmp_path, json)?;
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}
