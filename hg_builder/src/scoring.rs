//! Window scoring (guide §4-§5): builds the per-window
//! [`SpectralHypergraph`], runs the matrix-free Fiedler-eigenvalue pass,
//! NBSC's `rho_B` for fan-out windows, and the Ramanujan-deviation feature,
//! against a rolling per-hyperedge-kind baseline.

use std::collections::HashMap;

use correlation_engine::HyperedgeKind;
use nbsc::hypergraph_bridge::clique_expand;
use nbsc::spectral::estimate_spectral_radius;
use spectral_hypergraph::hypergraph::HypergraphBuilder;
use spectral_hypergraph::laplacian::HypergraphOperator;
use spectral_hypergraph::operator::LinearOperator;
use spectral_hypergraph::spectral::lanczos_smallest;
use spectral_jacobi_ds::compute_hypergraph_ramanujan_feature;

use crate::belief::HostBeliefSource;
use crate::builder::ClosedWindow;

/// Deterministic seed for the Lanczos/Ramanujan starting vectors. Fixed
/// (not per-window-random) so a given window's score is reproducible for
/// debugging and for the `siem_event_index` regression tests that compare
/// against a recorded vector -- see guide §9's honesty note on
/// reproducibility for exactly this reason.
const SPECTRAL_SEED: u64 = 0x5150_5350; // "SPSP" -- arbitrary, just fixed.
const NBSC_KRYLOV_DIM: usize = 32;
const LANCZOS_MAX_ITER: usize = 200;
const LANCZOS_TOL: f64 = 1e-8;

#[derive(Debug, Clone)]
pub struct ScoredWindow {
    pub closed: ClosedWindow,
    /// Second-smallest eigenvalue of the normalized hypergraph Laplacian
    /// for this window (the Fiedler eigenvalue itself -- algebraic
    /// connectivity), not yet compared to a baseline.
    pub fiedler_eigenvalue: f64,
    /// `(baseline - fiedler_eigenvalue).max(0.0)` for this window's
    /// `HyperedgeKind` -- guide §4's "sharp drop relative to normal."
    /// Larger = more anomalous.
    pub fiedler_shift: f64,
    /// NBSC's `rho_B` on the clique-expanded window graph. Only computed
    /// for [`HyperedgeKind::SourceFanOut`] windows (guide §4: "for
    /// directed/cyclic structure -- beaconing, C2 loops -- run NBSC's
    /// Hashimoto spectrum on the fan-out graph as a second, independent
    /// feature").
    pub rho_b: Option<f64>,
    /// Ramanujan-bound deviation (guide §5), computed for every window
    /// that reaches scoring (cheap relative to the eigendecomposition
    /// already performed, and the guide's build order treats it as a
    /// validation pass worth running broadly before deciding whether it's
    /// independently publishable).
    pub ramanujan_deviation: f64,
    pub mean_hmm_belief: f64,
}

/// Rolling per-`HyperedgeKind` baseline for the Fiedler eigenvalue, so
/// "shift" means "relative to this window family's own recent normal," not
/// an arbitrary global constant that would conflate a naturally-dense
/// session-cluster hypergraph with a naturally-sparse fan-in one.
pub struct ScoringBaselines {
    /// Exponential-moving-average smoothing factor. `0.1` means each new
    /// window's eigenvalue nudges the baseline 10% of the way toward it --
    /// slow enough that one anomalous window doesn't itself redefine
    /// "normal," fast enough to track genuine drift (e.g. a network's
    /// steady-state connectivity changing after a topology change).
    alpha: f64,
    means: HashMap<HyperedgeKind, f64>,
}

impl Default for ScoringBaselines {
    fn default() -> Self {
        Self { alpha: 0.1, means: HashMap::new() }
    }
}

impl ScoringBaselines {
    pub fn new(alpha: f64) -> Self {
        Self { alpha: alpha.clamp(0.0, 1.0), means: HashMap::new() }
    }

    /// Returns the shift for `eigenvalue` against the current baseline for
    /// `kind`, then updates the baseline. The very first window for a
    /// given `kind` has no baseline yet, so it seeds the baseline at its
    /// own eigenvalue and reports a shift of `0.0` -- a lone data point
    /// can't be anomalous relative to itself.
    fn observe(&mut self, kind: HyperedgeKind, eigenvalue: f64) -> f64 {
        match self.means.get_mut(&kind) {
            Some(mean) => {
                let shift = (*mean - eigenvalue).max(0.0);
                *mean = (1.0 - self.alpha) * *mean + self.alpha * eigenvalue;
                shift
            }
            None => {
                self.means.insert(kind, eigenvalue);
                0.0
            }
        }
    }
}

/// Scores one [`ClosedWindow`]: builds its [`SpectralHypergraph`] (vertex
/// weight = HMM belief prior from `belief_source`, hyperedge weight = the
/// window's own weight per the guide's table), runs the matrix-free
/// Fiedler pass, NBSC `rho_B` for fan-out windows, and the Ramanujan
/// feature, then folds the Fiedler eigenvalue through `baselines` for this
/// window's [`HyperedgeKind`].
///
/// **Verified limitation, not an oversight:** `spectral_hypergraph`'s own
/// docs are explicit that a vertex's intrinsic `weight` is "distinct from
/// the *degree* of the vertex" -- `HypergraphOperator`'s `D_v` comes from
/// `vertex_degree()` (sum of incident hyperedge weights only), and
/// `add_hyperedge()` always records a uniform incidence weight of `1.0`
/// for every member. So the HMM belief prior set here does *not* currently
/// perturb the Fiedler eigenvalue, NBSC `rho_B`, or the Ramanujan feature
/// -- confirmed empirically (see `scoring::tests`), not just by reading
/// the doc comment. It still does real work: it's carried through as
/// [`ScoredWindow::mean_hmm_belief`], a first-class fused signal alongside
/// the three spectral ones (see `correlation_engine::LaneDetail::Hypergraph`
/// and the guide's 4-D signal vector for `siem_event_index`). Making belief
/// actually reweight the spectral pass would mean either extending
/// `spectral_hypergraph::HypergraphOperator` to read `vertex_weight()` into
/// `D_v`, or building per-member incidence weights instead of `1.0` here --
/// both are changes to the pre-existing, unmodified `spectral_hypergraph`
/// crate's public contract, which this integration deliberately left alone
/// (see top-level README, "Pre-existing crates: unmodified"). Flagged here
/// rather than silently shipping a doc comment that overclaims.
///
/// Returns `None` if the window's hyperedge is degenerate after all -- this
/// shouldn't happen given [`ClosedWindow`]'s own `<2`-member guard, but a
/// single well-defined "give up gracefully" path here is cheaper than
/// auditing every future `ClosedWindow` producer for the same invariant.
pub fn score_window(
    closed: ClosedWindow,
    belief_source: &dyn HostBeliefSource,
    baselines: &mut ScoringBaselines,
) -> Option<ScoredWindow> {
    if closed.hyperedge_members.len() < 2 {
        tracing::warn!(window_id = %closed.window_id, "degenerate window reached scoring; dropping");
        return None;
    }

    let mut builder = HypergraphBuilder::with_capacity(closed.hyperedge_members.len(), 1, closed.hyperedge_members.len());

    let mut vertex_ids = Vec::with_capacity(closed.hyperedge_members.len());
    let mut belief_sum = 0.0;
    for label in &closed.hyperedge_members {
        // Members are either host-IP strings or (for SessionCluster)
        // `ip:port` flow-instance labels; the belief source only has
        // opinions about hosts, so extract the IP prefix for the belief
        // lookup either way (an `ip:port` label's belief is "the belief
        // about the host that flow instance belongs to").
        let host_part = label.split(':').next().unwrap_or(label);
        let prior = host_part
            .parse()
            .ok()
            .map(|ip| belief_source.belief_score(ip))
            .unwrap_or(0.5);
        belief_sum += prior;

        // Vertex weight must be strictly positive (guide's own
        // `HypergraphOperator` requires nonzero degree, and a zero-weight
        // vertex risks exactly that isolation) -- floor the HMM prior at a
        // small epsilon rather than letting a confidently-benign host (
        // prior == 0.0) zero out its own vertex weight.
        let weight = prior.max(0.01);
        let vid = match builder.add_weighted_vertex(label.clone(), weight) {
            Ok(v) => v,
            Err(err) => {
                tracing::error!(window_id = %closed.window_id, %err, "failed to add vertex; dropping window");
                return None;
            }
        };
        vertex_ids.push(vid);
    }
    let mean_hmm_belief = belief_sum / closed.hyperedge_members.len() as f64;

    if let Err(err) = builder.add_hyperedge(&vertex_ids, closed.weight.max(1.0)) {
        tracing::error!(window_id = %closed.window_id, %err, "failed to add hyperedge; dropping window");
        return None;
    }

    let hg = match builder.build() {
        Ok(hg) => hg,
        Err(err) => {
            tracing::error!(window_id = %closed.window_id, %err, "failed to build hypergraph; dropping window");
            return None;
        }
    };

    // -- Fiedler eigenvalue, matrix-free (guide §4 hot-path recommendation) --
    let operator = match HypergraphOperator::new(&hg) {
        Ok(op) => op,
        Err(err) => {
            tracing::error!(window_id = %closed.window_id, %err, "failed to build hypergraph operator; dropping window");
            return None;
        }
    };
    let k = 2.min(operator.dim());
    let eigen = match lanczos_smallest(&operator, k, LANCZOS_MAX_ITER, LANCZOS_TOL, SPECTRAL_SEED) {
        Ok(e) => e,
        Err(err) => {
            tracing::warn!(window_id = %closed.window_id, %err, "lanczos did not converge; dropping window");
            return None;
        }
    };
    // With k>=2 the Fiedler eigenvalue is index 1 (the "trivial" all-ones
    // eigenvector sits at index 0 with eigenvalue 0 for a connected
    // component); a 2-vertex window only has k=1 available, in which case
    // there is no non-trivial eigenvalue to speak of and 0.0 correctly
    // means "no connectivity structure beyond the single edge itself."
    let fiedler_eigenvalue = if eigen.eigenvalues.len() > 1 { eigen.eigenvalues[1] } else { 0.0 };

    let fiedler_shift = baselines.observe(closed.kind, fiedler_eigenvalue);

    // -- NBSC rho_B, fan-out windows only (guide §4) --
    let rho_b = if closed.kind == HyperedgeKind::SourceFanOut {
        let graph = clique_expand(&hg);
        Some(estimate_spectral_radius(&graph, NBSC_KRYLOV_DIM, SPECTRAL_SEED))
    } else {
        None
    };

    // -- Ramanujan-bound deviation (guide §5) --
    let ramanujan_deviation = match compute_hypergraph_ramanujan_feature(&hg, SPECTRAL_SEED) {
        Ok(feature) => feature.deviation,
        Err(err) => {
            tracing::warn!(window_id = %closed.window_id, %err, "ramanujan feature computation failed; treating as 0.0");
            0.0
        }
    };

    Some(ScoredWindow {
        closed,
        fiedler_eigenvalue,
        fiedler_shift,
        rho_b,
        ramanujan_deviation,
        mean_hmm_belief,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::StaticBeliefSource;
    use std::net::IpAddr;

    fn closed_window(kind: HyperedgeKind, members: &[&str]) -> ClosedWindow {
        ClosedWindow {
            kind,
            window_id: "test-window".to_string(),
            hyperedge_members: members.iter().map(|s| s.to_string()).collect(),
            correlation_hosts: vec!["10.0.0.1".parse::<IpAddr>().unwrap()],
            weight: 10.0,
        }
    }

    #[test]
    fn fan_in_window_scores_without_rho_b() {
        let window = closed_window(
            HyperedgeKind::DestinationFanIn,
            &["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"],
        );
        let belief = StaticBeliefSource::default();
        let mut baselines = ScoringBaselines::default();
        let scored = score_window(window, &belief, &mut baselines).expect("should score");
        assert!(scored.rho_b.is_none(), "fan-in windows don't run the NBSC pass");
        assert!(scored.fiedler_eigenvalue.is_finite());
        assert!(scored.ramanujan_deviation >= 0.0);
    }

    #[test]
    fn fan_out_window_computes_rho_b() {
        let window = closed_window(
            HyperedgeKind::SourceFanOut,
            &["10.0.0.10", "10.0.0.11", "10.0.0.12"],
        );
        let belief = StaticBeliefSource::default();
        let mut baselines = ScoringBaselines::default();
        let scored = score_window(window, &belief, &mut baselines).expect("should score");
        assert!(scored.rho_b.is_some(), "fan-out windows must run the NBSC pass");
    }

    #[test]
    fn first_window_of_a_kind_has_zero_shift_by_definition() {
        let window = closed_window(HyperedgeKind::DestinationFanIn, &["10.0.0.1", "10.0.0.2"]);
        let belief = StaticBeliefSource::default();
        let mut baselines = ScoringBaselines::default();
        let scored = score_window(window, &belief, &mut baselines).expect("should score");
        assert_eq!(scored.fiedler_shift, 0.0);
    }

    #[test]
    fn distinguishable_beliefs_change_mean_belief_but_not_the_spectral_pass() {
        struct TwoTier;
        impl HostBeliefSource for TwoTier {
            fn belief_score(&self, host: IpAddr) -> f64 {
                if host.to_string().ends_with(".99") {
                    0.95
                } else {
                    0.05
                }
            }
        }

        let benign_window =
            closed_window(HyperedgeKind::DestinationFanIn, &["10.0.0.1", "10.0.0.2", "10.0.0.3"]);
        let suspicious_window =
            closed_window(HyperedgeKind::DestinationFanIn, &["10.0.0.99", "10.0.0.2", "10.0.0.3"]);

        let belief = TwoTier;
        let mut b1 = ScoringBaselines::default();
        let mut b2 = ScoringBaselines::default();
        let scored_benign = score_window(benign_window, &belief, &mut b1).unwrap();
        let scored_suspicious = score_window(suspicious_window, &belief, &mut b2).unwrap();

        // The HMM belief prior reaches `mean_hmm_belief` (a first-class
        // fused signal downstream), which is where this test's guarantee
        // actually holds.
        assert!(scored_suspicious.mean_hmm_belief > scored_benign.mean_hmm_belief);

        // It does NOT currently reach the spectral pass: see the
        // "Verified limitation" doc comment on `score_window` for why
        // `spectral_hypergraph`'s own Laplacian definition makes vertex
        // weight inert here (uniform 1.0 incidence weight, degree computed
        // from hyperedge weights only). Asserted explicitly, not just
        // documented, so a future change to either crate that starts
        // wiring belief into the spectral pass is a visible, deliberate
        // test update rather than a silent behavior change.
        assert_eq!(scored_benign.fiedler_eigenvalue, scored_suspicious.fiedler_eigenvalue);
    }
}
