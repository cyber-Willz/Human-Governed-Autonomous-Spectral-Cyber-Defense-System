//! Bridge between this crate's Ramanujan-deviation anomaly pipeline and
//! `nbsc`'s Hashimoto / non-backtracking spectral machinery, gated behind
//! the `nbsc_bridge` feature (off by default — this crate's core anomaly
//! pipeline has no `nbsc`/Burn/`krylov_ds` dependency at all without it).
//!
//! ## Why this pairing, specifically
//!
//! [`crate::compute_ramanujan_feature`] flags a node population whose
//! largest non-trivial **adjacency** eigenvalue exceeds the `2*sqrt(d-1)`
//! Ramanujan bound — that bound is a statement about ordinary
//! (backtracking-walk-inclusive) spectral structure, and is blind to
//! *oriented* cycle structure by construction (the adjacency matrix is
//! symmetric; a directed 3-cycle and its reverse look identical to it).
//! `nbsc`'s `rho_B` (spectral radius of the Hashimoto / non-backtracking
//! matrix) is built from directed edges with a no-immediate-reversal rule
//! specifically *because* it is sensitive to that oriented-cycle structure
//! (see `nbsc`'s own `docs/ihara_zeta.rs`).
//!
//! For the AML/compliance context this pipeline sits next to
//! (`build_demo_hypergraph`'s multi-party counterparty rings), that
//! distinction is not academic: a circular fund flow — A pays B pays C
//! pays A — is exactly the kind of oriented-cycle structure the Hashimoto
//! spectrum is built to see and the plain adjacency-spectrum Ramanujan
//! bound is not. This module does not replace [`crate::RamanujanFeature`]
//! with a Hashimoto-based one; it computes both and hands back a
//! feature vector with the second appended, so a downstream model (or a
//! human) can weigh disagreement between the two as its own signal —
//! a node population that is adjacency-Ramanujan (unremarkable degree
//! spectrum) but Hashimoto-expansive (unremarkable *undirected* structure
//! hiding a real directed cycle) is precisely the case where using only
//! one or the other would miss something.
//!
//! ## What is and is not validated here
//!
//! The two crates independently converged on the same `||A||_2 / rho_B`
//! "expansivity ratio" diagnostic — `nbsc`'s own
//! `docs/results_cora_draft.md` uses it to explain RNBSC's depth-3
//! instability on Cora (ratio `1.594`, "measurably expansive"), and
//! `nbsc::hypergraph_bridge`'s module docs report `1.136` on a synthetic
//! hypergraph. [`HashimotoDeviationFeature::expansivity_ratio`] here reuses
//! that exact quantity as an eighth anomaly feature. That reuse is a
//! plausible hypothesis, not a validated one: nobody has yet checked
//! whether transaction-graph "expansivity" actually correlates with
//! known-fraudulent subgraphs the way it correlates with GNN
//! over-smoothing instability on Cora. Treat
//! [`extend_with_hashimoto`]'s output as an additional feature column to
//! evaluate empirically, not as a pre-validated fraud signal.

use crate::{CsrGraph, NodeAnomalyFeatures, ANOMALY_FEATURE_DIM};
use nbsc::graph::Graph as NbscGraph;
use nbsc::spectral::{adjacency_operator_norm, estimate_spectral_radius};

/// Flattens this crate's [`CsrGraph`] into `nbsc::graph::Graph`, the type
/// `nbsc`'s entire Hashimoto/NBSC pipeline is built around. Undirected,
/// unweighted, index-preserving (vertex `i` here is vertex `i` there) —
/// the same shape of mapping as `nbsc::hypergraph_bridge::clique_expand`,
/// just starting from a plain graph instead of a hypergraph.
pub fn csr_graph_to_nbsc_graph(graph: &CsrGraph) -> NbscGraph {
    let n = graph.num_nodes;
    let mut g = NbscGraph::new(n);
    for u in 0..n {
        for &v in graph.neighbors(u) {
            if v > u {
                g.add_edge(u, v);
            }
        }
    }
    g
}

/// Hashimoto/non-backtracking counterpart to [`crate::RamanujanFeature`]:
/// the spectral radius `rho_B` of the non-backtracking matrix, the plain
/// adjacency operator norm `||A||_2` (via `nbsc`'s own matrix-free
/// `krylov_ds` Arnoldi estimators), and their ratio.
///
/// `expansivity_ratio > 1` mirrors the diagnostic `nbsc`'s
/// `docs/results_cora_draft.md` uses to flag an unstable, expansive
/// propagation regime — see the module docs above for why that reuse is a
/// hypothesis here, not a settled fraud signal.
#[derive(Debug, Clone, Copy)]
pub struct HashimotoDeviationFeature {
    pub rho_b: f64,
    pub adjacency_operator_norm: f64,
    pub expansivity_ratio: f64,
}

/// Computes [`HashimotoDeviationFeature`] for `graph`, at Krylov subspace
/// dimension `krylov_dim` (same knob `nbsc::spectral::estimate_spectral_radius`
/// exposes directly — larger values are more accurate and more expensive;
/// `nbsc`'s own benchmarks use dimensions in the 20-40 range for graphs in
/// the hundreds-to-low-thousands of nodes) and deterministic seed `seed`.
pub fn compute_hashimoto_deviation_feature(
    graph: &CsrGraph,
    krylov_dim: usize,
    seed: u64,
) -> HashimotoDeviationFeature {
    let g = csr_graph_to_nbsc_graph(graph);
    let rho_b = estimate_spectral_radius(&g, krylov_dim, seed);
    let adjacency_operator_norm = adjacency_operator_norm(&g, krylov_dim, seed);
    let expansivity_ratio = if rho_b > 1e-9 {
        adjacency_operator_norm / rho_b
    } else {
        f64::INFINITY
    };
    HashimotoDeviationFeature { rho_b, adjacency_operator_norm, expansivity_ratio }
}

/// One more column than [`crate::ANOMALY_FEATURE_DIM`]: this crate's
/// existing 7 features (Laplacian embedding x/y, relative degree, 3 Jacobi
/// probe evaluations, graph-level Ramanujan deviation) plus the
/// graph-level Hashimoto expansivity ratio, broadcast to every node —
/// exactly the same "graph-level scalar broadcast to all nodes" pattern
/// [`crate::extract_anomaly_features`] already uses for the Ramanujan
/// deviation itself.
pub const ANOMALY_FEATURE_DIM_HASHIMOTO: usize = ANOMALY_FEATURE_DIM + 1;

/// [`NodeAnomalyFeatures`] extended with the Hashimoto expansivity ratio.
/// A separate struct rather than growing [`NodeAnomalyFeatures`] in place
/// (or the `ANOMALY_FEATURE_DIM` const it's sized on) — additive, so
/// everything upstream of this module (dataset construction, the
/// `FeatureNormalizer`/autoencoder pipeline as already written) keeps
/// working unmodified for callers who don't opt into the `nbsc_bridge`
/// feature.
#[derive(Debug, Clone)]
pub struct NodeAnomalyFeaturesHashimoto {
    pub node_id: usize,
    pub values: [f64; ANOMALY_FEATURE_DIM_HASHIMOTO],
}

/// Appends `hashimoto.expansivity_ratio` as an eighth feature to each row
/// of `base` (typically the output of [`crate::extract_anomaly_features`]
/// or [`crate::extract_anomaly_features_hg`] run on the same graph
/// `hashimoto` was computed from).
pub fn extend_with_hashimoto(
    base: &[NodeAnomalyFeatures],
    hashimoto: &HashimotoDeviationFeature,
) -> Vec<NodeAnomalyFeaturesHashimoto> {
    base.iter()
        .map(|f| {
            let mut values = [0.0; ANOMALY_FEATURE_DIM_HASHIMOTO];
            values[..ANOMALY_FEATURE_DIM].copy_from_slice(&f.values);
            values[ANOMALY_FEATURE_DIM] = hashimoto.expansivity_ratio;
            NodeAnomalyFeaturesHashimoto { node_id: f.node_id, values }
        })
        .collect()
}

/// Hypergraph counterpart of [`compute_hashimoto_deviation_feature`]:
/// reuses `nbsc::hypergraph_bridge::clique_expand` (gated on `nbsc`'s own
/// `hypergraph` feature, which this crate's `nbsc_bridge` feature turns on
/// — see `Cargo.toml`) rather than this crate's own
/// [`crate::csr_graph_to_hypergraph`], which goes the opposite direction
/// (plain graph -> hypergraph, for feeding `spectral_hypergraph`'s native
/// solvers). Both clique-expansion implementations exist because they
/// serve different callers — this one produces `nbsc::graph::Graph` for
/// the Hashimoto pipeline, `spectral_hypergraph::laplacian::clique_expansion_adjacency`
/// (used inside `nbsc::hypergraph_bridge::clique_expand` itself) produces a
/// dense adjacency matrix — not because the reduction was reimplemented
/// twice by accident.
pub fn compute_hashimoto_deviation_feature_hg(
    hg: &spectral_hypergraph::SpectralHypergraph,
    krylov_dim: usize,
    seed: u64,
) -> HashimotoDeviationFeature {
    let g = nbsc::hypergraph_bridge::clique_expand(hg);
    let rho_b = estimate_spectral_radius(&g, krylov_dim, seed);
    let adjacency_operator_norm = adjacency_operator_norm(&g, krylov_dim, seed);
    let expansivity_ratio = if rho_b > 1e-9 {
        adjacency_operator_norm / rho_b
    } else {
        f64::INFINITY
    };
    HashimotoDeviationFeature { rho_b, adjacency_operator_norm, expansivity_ratio }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrGraph;

    /// A 4-cycle: no oriented-cycle structure beyond the single undirected
    /// cycle, small enough for both `rho_b` and `||A||_2` to be checkable
    /// against the plain fact that a 4-cycle is 2-regular. Mostly a
    /// wiring smoke test: does the bridge actually run end to end and
    /// produce finite, non-degenerate numbers, not a claim about what
    /// "correct" rho_b/||A||_2 values look like on this graph (nbsc's own
    /// test suite already covers that math).
    #[test]
    fn hashimoto_feature_runs_on_small_graph() {
        let edges = [(0, 1), (1, 2), (2, 3), (3, 0)];
        let graph = CsrGraph::from_edges(4, &edges).unwrap();
        let feature = compute_hashimoto_deviation_feature(&graph, 4, 7);
        assert!(feature.rho_b.is_finite());
        assert!(feature.adjacency_operator_norm.is_finite());
        assert!(feature.rho_b >= 0.0);
    }

    #[test]
    fn extend_with_hashimoto_preserves_base_columns() {
        let base = vec![NodeAnomalyFeatures { node_id: 0, values: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] }];
        let hashimoto = HashimotoDeviationFeature {
            rho_b: 2.0,
            adjacency_operator_norm: 3.0,
            expansivity_ratio: 1.5,
        };
        let extended = extend_with_hashimoto(&base, &hashimoto);
        assert_eq!(extended.len(), 1);
        assert_eq!(&extended[0].values[..ANOMALY_FEATURE_DIM], &base[0].values[..]);
        assert_eq!(extended[0].values[ANOMALY_FEATURE_DIM], 1.5);
    }

    #[test]
    fn hashimoto_feature_hg_runs_on_small_hypergraph() {
        use spectral_hypergraph::HypergraphBuilder;
        let mut builder = HypergraphBuilder::with_capacity(4, 2, 3);
        let ids: Vec<_> = (0..4).map(|i| builder.add_vertex(format!("v{i}"))).collect::<Result<_, _>>().unwrap();
        builder.add_hyperedge(&[ids[0], ids[1], ids[2]], 1.0).unwrap();
        builder.add_hyperedge(&[ids[1], ids[2], ids[3]], 1.0).unwrap();
        let hg = builder.build().unwrap();
        let feature = compute_hashimoto_deviation_feature_hg(&hg, 4, 7);
        assert!(feature.rho_b.is_finite());
        assert!(feature.adjacency_operator_norm.is_finite());
    }
}
