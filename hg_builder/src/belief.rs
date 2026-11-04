//! HMM belief as hypergraph vertex weight (guide §4).
//!
//! `neural_hmm::hmm::Belief` exposes `argmax()` (most-likely state and its
//! probability) but nothing that's directly a `[0,1]` "how suspicious is
//! this host" scalar — and `argmax()` alone would be a lossy vertex weight:
//! a belief of `[0.49, 0.51]` and `[0.01, 0.99]` both argmax to state 1,
//! but a hyperedge scorer should weight a host the HMM is barely-more-than-
//! guessing about very differently from one it's all but certain of.

use neural_hmm::hmm::Belief;
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::RwLock;

/// Converts a [`Belief`] into a `[0,1]` suspicion score: total probability
/// mass on every state *except* `benign_state`. For a two-state
/// benign/malicious HMM this is just `belief[1]`; for a multi-state model
/// (distinct attack-family states) it's "probability the host is in *any*
/// non-benign state," which is the right aggregate for a vertex weight —
/// the hyperedge scoring in [`crate::scoring`] cares about "is this host
/// suspicious," not which specific attack state the HMM currently favors.
///
/// Returns `0.0` (maximally-benign prior) if `benign_state` is out of range
/// for `belief`, rather than panicking — a misconfigured state index should
/// degrade to "trust the hyperedge structure alone," not take down the
/// scoring pass for the whole window.
pub fn vertex_prior_from_belief(belief: &Belief, benign_state: usize) -> f64 {
    let slice = belief.as_slice();
    if benign_state >= slice.len() {
        return 0.0;
    }
    (1.0 - slice[benign_state] as f64).clamp(0.0, 1.0)
}

/// What [`crate::scoring::score_window`] needs from the neural HMM layer:
/// a per-host suspicion prior in `[0,1]`. Kept as a trait (rather than a
/// concrete dependency on a live `Hmm<B>` + feature-tensor pipeline) so
/// this crate doesn't need to know about `burn` backends or how raw flow
/// features get vectorized for `Hmm::filter_step` — that wiring lives
/// wherever `active_spectral_defense` already runs the per-host belief
/// tracker (see the crate's module docs on `neural_hmm`), and only needs to
/// hand this crate the resulting scalar per host.
pub trait HostBeliefSource: Send + Sync {
    /// `[0,1]` suspicion prior for `host`; hosts with no belief state yet
    /// (never observed) should return a neutral prior, not `0.0` — a host
    /// this crate has never seen isn't *known* benign, it's *unknown*.
    fn belief_score(&self, host: IpAddr) -> f64;
}

/// A trivial [`HostBeliefSource`] returning a fixed, neutral prior for
/// every host. Useful for wiring/testing `hg_builder` end-to-end before the
/// live HMM belief cache is plumbed in, and as the fallback this crate
/// falls back to if no source is supplied at all — see
/// [`crate::scoring::score_window`].
pub struct StaticBeliefSource {
    pub prior: f64,
}

impl Default for StaticBeliefSource {
    fn default() -> Self {
        Self { prior: 0.5 }
    }
}

impl HostBeliefSource for StaticBeliefSource {
    fn belief_score(&self, _host: IpAddr) -> f64 {
        self.prior
    }
}

/// A concrete [`HostBeliefSource`] backed by a live per-host [`Belief`]
/// cache, for the production integration once the HMM's feature pipeline
/// is wired up to update it (e.g. from the same `IngestedPacket` /
/// `Arc<IngestedPacket>` stream the autoencoder lane already consumes, via
/// `Hmm::filter_step`). `hg_builder` itself never calls `filter_step` --
/// that requires the emission-tensor pipeline this crate has no
/// visibility into -- it only reads the resulting [`Belief`]s.
pub struct HmmBeliefCache {
    benign_state: usize,
    beliefs: RwLock<HashMap<IpAddr, Belief>>,
}

impl HmmBeliefCache {
    pub fn new(benign_state: usize) -> Self {
        Self { benign_state, beliefs: RwLock::new(HashMap::new()) }
    }

    /// Called by whatever drives `Hmm::filter_step` per host, each time a
    /// fresh belief is available.
    pub fn update(&self, host: IpAddr, belief: Belief) {
        self.beliefs.write().expect("belief cache lock poisoned").insert(host, belief);
    }
}

impl HostBeliefSource for HmmBeliefCache {
    fn belief_score(&self, host: IpAddr) -> f64 {
        let guard = self.beliefs.read().expect("belief cache lock poisoned");
        match guard.get(&host) {
            Some(belief) => vertex_prior_from_belief(belief, self.benign_state),
            // Never observed: neutral, not confidently-benign.
            None => 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_state_belief_is_just_the_nonbenign_mass() {
        let belief = Belief::new(vec![0.3, 0.7]).unwrap();
        let score = vertex_prior_from_belief(&belief, 0);
        assert!((score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn narrow_margin_and_wide_margin_are_distinguished_unlike_argmax() {
        let narrow = Belief::new(vec![0.49, 0.51]).unwrap();
        let wide = Belief::new(vec![0.01, 0.99]).unwrap();
        // argmax() alone would call both of these "state 1", losing exactly
        // the distinction a vertex weight needs.
        assert_eq!(narrow.argmax().0, 1);
        assert_eq!(wide.argmax().0, 1);
        assert!(vertex_prior_from_belief(&narrow, 0) < vertex_prior_from_belief(&wide, 0));
    }

    #[test]
    fn unobserved_host_gets_neutral_not_benign_prior() {
        let cache = HmmBeliefCache::new(0);
        let host: IpAddr = "10.0.0.9".parse().unwrap();
        assert_eq!(cache.belief_score(host), 0.5);
    }
}
