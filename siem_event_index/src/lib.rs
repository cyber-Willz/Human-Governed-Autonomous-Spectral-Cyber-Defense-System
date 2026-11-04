//! Persistent index of confirmed-incident event embeddings, standing up the
//! fifth signal the guide calls for in §4: *distance to the nearest
//! previously-confirmed incident*, computed over the full multi-signal
//! event vector (hypergraph-Laplacian window score, NBSC `rho_B`, HMM
//! belief, spectral-autoencoder reconstruction error) rather than any one
//! pipeline's internal features.
//!
//! # Why this is not `spectral_jacobi_ds::HnswIndex` reused directly
//!
//! `spectral_jacobi_ds` already has a working `HnswIndex`, but it operates
//! on 2-D Jacobi-embedded [`SpectralNode`]s scoped to that crate's own node
//! anomaly pipeline (see its module docs). This index is deliberately
//! separate, as the guide specifies: it indexes a different, 4-D vector
//! space (the SIEM's fused signal, not a per-node spectral embedding), and
//! it needs to survive process restarts and grow across the lifetime of a
//! deployment, not just a single run.
//!
//! # Why this is exact (linear-scan) nearest neighbor, not a graph-based ANN
//!
//! The guide's own framing is that persistence is "the new part... rather
//! than the algorithm itself" -- the corpus this indexes is *confirmed
//! incidents*, which even a busy SOC accumulates at a rate of maybe
//! thousands per year, not per second. Exact nearest neighbor over a few
//! thousand 4-D points is sub-millisecond; there is no accuracy/latency
//! tradeoff to make at this scale, and no case for a second, differently-
//! bugged approximate index alongside `spectral_jacobi_ds::HnswIndex`
//! (whose layered-graph construction is real, tested code to reach for if
//! this index's corpus ever grows past the point where a linear scan is
//! the bottleneck -- the honest scope note applies here exactly as it does
//! for `rho_B` streaming elsewhere in this codebase).
//!
//! # Persistence
//!
//! Append-only JSON Lines file: one [`IndexedIncident`] per line. Loaded
//! fully into memory on [`PersistentEventIndex::open`] (again: thousands of
//! points, this is fine), appended to on [`PersistentEventIndex::insert`].
//! `fsync`-on-write is deliberately not attempted here — an incident
//! missing from the index after a crash degrades gracefully (one fewer
//! comparison point for the fifth signal), whereas losing an
//! `AuditDecision` in `review_queue`'s graph would not, so that crate's
//! durability story is the one worth hardening first.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The full multi-signal event vector, in a fixed, documented field order
/// so callers can't transpose fields by accident:
/// `[laplacian_window_score, rho_b, hmm_belief, autoencoder_reconstruction_error]`.
pub type EventVector = [f64; 4];

pub const LAPLACIAN_WINDOW_SCORE_IDX: usize = 0;
pub const RHO_B_IDX: usize = 1;
pub const HMM_BELIEF_IDX: usize = 2;
pub const AUTOENCODER_ERROR_IDX: usize = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedIncident {
    /// Free-form identifier -- e.g. the `CorrelationVerdict`'s host and
    /// decision id, or a hyperedge window id -- whatever the caller uses
    /// to look the full incident record back up elsewhere.
    pub incident_id: String,
    pub label: String, // e.g. "HOIC", "LOIC-HTTP", "lateral-movement", ...
    pub vector: EventVector,
    pub confirmed_at_unix_ms: u64,
}

#[derive(Debug, Clone)]
pub struct NearestIncident {
    pub incident_id: String,
    pub label: String,
    pub distance: f64,
}

#[derive(Debug, Error)]
pub enum EventIndexError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, EventIndexError>;

struct State {
    incidents: Vec<IndexedIncident>,
}

pub struct PersistentEventIndex {
    path: PathBuf,
    state: RwLock<State>,
}

impl PersistentEventIndex {
    /// Opens (creating if absent) the JSONL-backed index at `path`,
    /// replaying every line into memory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut incidents = Vec::new();

        if path.exists() {
            let file = File::open(&path)?;
            for line in BufReader::new(file).lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                incidents.push(serde_json::from_str::<IndexedIncident>(&line)?);
            }
        } else {
            // Touch the file so subsequent opens/appends don't need to
            // special-case "doesn't exist yet".
            OpenOptions::new().create(true).write(true).open(&path)?;
        }

        Ok(Self { path, state: RwLock::new(State { incidents }) })
    }

    /// In-memory-only index, for tests or a caller that doesn't want disk
    /// persistence (e.g. an offline forensic replay over a synthetic set).
    pub fn in_memory() -> Self {
        Self { path: PathBuf::new(), state: RwLock::new(State { incidents: Vec::new() }) }
    }

    /// Records a newly-confirmed incident: appends to the backing file
    /// (best-effort -- see module docs on the durability tradeoff) and adds
    /// it to the in-memory corpus immediately, so it's queryable without
    /// requiring a re-open.
    pub fn insert(&self, incident: IndexedIncident) -> Result<()> {
        if !self.path.as_os_str().is_empty() {
            let mut file = OpenOptions::new().create(true).append(true).open(&self.path)?;
            let line = serde_json::to_string(&incident)?;
            writeln!(file, "{line}")?;
        }
        self.state.write().incidents.push(incident);
        Ok(())
    }

    /// The confirmed incident whose vector is nearest `query` under
    /// Euclidean distance, or `None` if the index is empty. Each of the
    /// four signal dimensions carries a different natural scale (a
    /// Laplacian window score and an NBSC `rho_B` are not directly
    /// comparable), so callers should normalize/whiten `query` and the
    /// stored vectors consistently upstream -- this index deliberately
    /// stays opinion-free about that normalization, since the right
    /// scaling is a property of the four upstream scorers, not of the
    /// index itself.
    pub fn nearest(&self, query: &EventVector) -> Option<NearestIncident> {
        let state = self.state.read();
        state
            .incidents
            .iter()
            .map(|inc| {
                let d = euclidean(query, &inc.vector);
                (inc, d)
            })
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(inc, d)| NearestIncident {
                incident_id: inc.incident_id.clone(),
                label: inc.label.clone(),
                distance: d,
            })
    }

    pub fn len(&self) -> usize {
        self.state.read().incidents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn euclidean(a: &EventVector, b: &EventVector) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_index_returns_none() {
        let idx = PersistentEventIndex::in_memory();
        assert!(idx.nearest(&[0.0, 0.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn nearest_picks_the_closest_confirmed_incident() {
        let idx = PersistentEventIndex::in_memory();
        idx.insert(IndexedIncident {
            incident_id: "inc-1".into(),
            label: "HOIC".into(),
            vector: [0.9, 0.8, 0.9, 0.7],
            confirmed_at_unix_ms: 0,
        })
        .unwrap();
        idx.insert(IndexedIncident {
            incident_id: "inc-2".into(),
            label: "benign-burst".into(),
            vector: [0.1, 0.1, 0.1, 0.1],
            confirmed_at_unix_ms: 0,
        })
        .unwrap();

        let nearest = idx.nearest(&[0.85, 0.75, 0.88, 0.65]).unwrap();
        assert_eq!(nearest.incident_id, "inc-1");
        assert_eq!(nearest.label, "HOIC");
    }

    #[test]
    fn persists_and_reloads_across_open_calls() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("incidents.jsonl");

        {
            let idx = PersistentEventIndex::open(&path).unwrap();
            idx.insert(IndexedIncident {
                incident_id: "inc-1".into(),
                label: "LOIC-HTTP".into(),
                vector: [0.6, 0.2, 0.5, 0.4],
                confirmed_at_unix_ms: 1234,
            })
            .unwrap();
        }

        let reopened = PersistentEventIndex::open(&path).unwrap();
        assert_eq!(reopened.len(), 1);
        let nearest = reopened.nearest(&[0.6, 0.2, 0.5, 0.4]).unwrap();
        assert_eq!(nearest.incident_id, "inc-1");
        assert!(nearest.distance < 1e-9);
    }
}
