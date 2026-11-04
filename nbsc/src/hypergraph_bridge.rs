//! Bridge between this crate and [`spectral_hypergraph`], gated behind the
//! `hypergraph` feature.
//!
//! The two crates each grew their own notion of "a sparse symmetric
//! operator I can run Krylov methods against": this crate's
//! [`krylov_ds::operator::LinearOperator`] (generic over `T: Float`,
//! slice-in/slice-out, allocation-free per `apply`) and
//! `spectral_hypergraph`'s [`spectral_hypergraph::operator::LinearOperator`]
//! (`f64`-only, `nalgebra::DVector`-in/out, one allocation per `apply`).
//! Neither crate depends on the other, so there is no blanket impl bridging
//! them; this module is the seam.
//!
//! Two independent integration paths are provided:
//!
//! 1. **Clique expansion -> ordinary [`Graph`]** ([`clique_expand`]):
//!    flattens a [`SpectralHypergraph`] into the plain, unweighted
//!    [`crate::graph::Graph`] this crate's whole pipeline (NBSC, GCN, GAT,
//!    SAGE, the Ihara-zeta / Hashimoto spectral radius) is built around.
//!    Every existing filter, baseline, and diagnostic in this crate works
//!    unmodified on the result — this is the "just make it a graph" path,
//!    and [`nbsc_filter_bank_from_hypergraph`] is a one-call convenience
//!    for it.
//!
//! 2. **Matrix-free operator adapter** ([`HypergraphLaplacianOperator`]):
//!    wraps `spectral_hypergraph`'s
//!    [`spectral_hypergraph::laplacian::HypergraphOperator`] (the
//!    matrix-free normalized hypergraph Laplacian, `O(nnz(H))` per matvec)
//!    as a `krylov_ds::operator::LinearOperator<f64>`, so this crate's own
//!    Arnoldi/Lanczos routines — the same ones [`crate::spectral`] uses for
//!    `rho_B` and the adjacency operator norm — can run directly against
//!    the *true* hypergraph Laplacian, without ever forming the clique
//!    expansion. [`hypergraph_laplacian_operator_norm`] and
//!    [`hypergraph_algebraic_connectivity`] are built on top of it,
//!    mirroring [`crate::spectral::adjacency_operator_norm`]'s pattern
//!    exactly (deterministic seeded start vector, `krylov_ds::Lanczos`,
//!    `krylov_ds::eig::lanczos_ritz_pairs`) but pointed at the hypergraph
//!    operator instead of the plain graph adjacency operator.
//!
//! Path (2) allocates one `DVector` per `apply` call (translating in/out of
//! `spectral_hypergraph`'s representation); for hot loops on very large
//! hypergraphs where that allocation matters, path (1) followed by this
//! crate's existing matrix-free adjacency operator is cheaper. Path (2)
//! earns its cost when the hypergraph structure itself — not its clique
//! expansion — is the object of interest, e.g. comparing `rho_B` of the
//! clique-expanded graph (higher-order structure collapsed) against the
//! spectral gap of the *true* hypergraph Laplacian (higher-order structure
//! preserved).

use nalgebra::DVector;

use crate::graph::Graph;
use crate::spectral::NbscFilterBank;
use krylov_ds::eig::lanczos_ritz_pairs;
use krylov_ds::operator::LinearOperator as KrylovOperator;
use krylov_ds::{Lanczos, Reorthogonalization};
use spectral_hypergraph::hypergraph::SpectralHypergraph;
use spectral_hypergraph::laplacian::HypergraphOperator;
use spectral_hypergraph::operator::LinearOperator as SpectralHypergraphOperator;
use spectral_hypergraph::Result as HgResult;

/// Flatten a [`SpectralHypergraph`] into the unweighted [`Graph`] this
/// crate's NBSC/GCN/GAT/SAGE pipeline is built around, via the classical
/// clique expansion: every hyperedge `e` becomes a clique on its member
/// vertices (an edge is added, unweighted, between every pair of members).
///
/// This mirrors [`spectral_hypergraph::laplacian::clique_expansion_adjacency`]
/// structurally (same nonzero pattern — a pair `(u, v)` is connected here
/// iff its clique-expansion weight there is nonzero) but drops the
/// `w(e) / (|e| - 1)` edge weighting, since [`Graph`] and the Hashimoto/
/// non-backtracking construction it feeds are unweighted by design (see
/// `docs/ihara_zeta.rs`). Vertices with no incident hyperedges (isolated in
/// `hg`) become isolated, degree-0 vertices in the result rather than being
/// dropped, so `VertexId(i).0 == ` the resulting `Graph`'s vertex index `i`
/// for every `i` — the mapping is the identity on indices.
pub fn clique_expand(hg: &SpectralHypergraph) -> Graph {
    let n = hg.num_vertices();
    let mut g = Graph::new(n);
    for e in hg.hyperedge_ids() {
        // `e` came from `hg.hyperedge_ids()`, so this can only fail if `hg`
        // violates its own invariants.
        let members = hg
            .hyperedge_members(e)
            .expect("hyperedge id from hg.hyperedge_ids() is always valid");
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                g.add_edge(members[i].0, members[j].0);
            }
        }
    }
    g
}

/// Clique-expand `hg` via [`clique_expand`] and immediately build an
/// [`NbscFilterBank`] on the result, in one call. Returns both the
/// expanded [`Graph`] (needed to call [`NbscFilterBank::apply_taps`]) and
/// the bank itself.
pub fn nbsc_filter_bank_from_hypergraph(
    hg: &SpectralHypergraph,
    krylov_dim: usize,
    seed: u64,
) -> (Graph, NbscFilterBank) {
    let g = clique_expand(hg);
    let bank = NbscFilterBank::build(&g, krylov_dim, seed);
    (g, bank)
}

/// Adapts `spectral_hypergraph`'s matrix-free normalized hypergraph
/// Laplacian operator (`Delta = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}`,
/// symmetric PSD) to this crate's [`krylov_ds::operator::LinearOperator<f64>`]
/// trait, so `krylov_ds::Arnoldi`/`krylov_ds::Lanczos` can run against it
/// directly. Each [`KrylovOperator::apply`] call allocates one
/// `nalgebra::DVector<f64>` to cross into `spectral_hypergraph`'s
/// `DVector`-based representation and back — see the module docs for when
/// that cost matters.
pub struct HypergraphLaplacianOperator {
    inner: HypergraphOperator,
}

impl HypergraphLaplacianOperator {
    /// Build the adapter for `hg`. Fails under the same conditions as
    /// [`HypergraphOperator::new`]: an isolated vertex (undefined
    /// `D_v^{-1/2}`), or no hyperedges at all.
    pub fn new(hg: &SpectralHypergraph) -> HgResult<Self> {
        Ok(Self { inner: HypergraphOperator::new(hg)? })
    }
}

impl KrylovOperator<f64> for HypergraphLaplacianOperator {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        let xv = DVector::from_row_slice(x);
        let yv = SpectralHypergraphOperator::apply(&self.inner, &xv);
        y.copy_from_slice(yv.as_slice());
    }
}

/// Deterministic pseudo-random start vector, byte-for-byte the same
/// xorshift generator [`crate::spectral::estimate_spectral_radius`] and
/// [`crate::spectral::adjacency_operator_norm`] use, so results built on
/// top of this module are reproducible the same way and comparable across
/// the clique-expansion and matrix-free-hypergraph-operator code paths for
/// the same `seed`.
fn xorshift_start_vector(dim: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(2685821657736338717).wrapping_add(1);
    let mut v0 = vec![0.0f64; dim];
    for vi in v0.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *vi = (state as f64 / u64::MAX as f64) - 0.5;
    }
    v0
}

/// The operator (spectral) norm of the true normalized hypergraph
/// Laplacian, `||Delta||_2`, found via `krylov_ds::Lanczos` on the
/// matrix-free [`HypergraphLaplacianOperator`] — the hypergraph-native
/// counterpart of [`crate::spectral::adjacency_operator_norm`]. Since
/// `Delta` is symmetric with spectrum in `[0, 2]`, this is its
/// largest-magnitude eigenvalue.
pub fn hypergraph_laplacian_operator_norm(
    hg: &SpectralHypergraph,
    krylov_dim: usize,
    seed: u64,
) -> HgResult<f64> {
    let op = HypergraphLaplacianOperator::new(hg)?;
    let n = op.dim();
    let m = krylov_dim.min(n).max(1);
    let v0 = xorshift_start_vector(n, seed);

    let lanczos = Lanczos::new(m, 1e-12, Reorthogonalization::Full);
    let result = lanczos
        .run(&op, &v0)
        .expect("Lanczos failed on hypergraph Laplacian operator");
    let pairs = lanczos_ritz_pairs(&result);
    Ok(pairs.iter().map(|p| p.value.abs()).fold(0.0, f64::max))
}

/// The algebraic connectivity of `hg`'s normalized hypergraph Laplacian —
/// its smallest *nonzero* eigenvalue — found via `krylov_ds::Lanczos` on
/// the matrix-free [`HypergraphLaplacianOperator`]. This is the
/// hypergraph-native, matrix-free-via-`krylov_ds` counterpart of
/// `spectral_hypergraph`'s own
/// [`spectral_hypergraph::spectral::fiedler_vector`] (which returns the
/// corresponding eigen*vector* via `spectral_hypergraph`'s own hand-rolled
/// Lanczos); this function instead reuses this crate's Krylov engine and
/// returns the eigen*value*, for direct comparison against
/// [`crate::spectral::estimate_spectral_radius`] applied to the
/// clique-expanded graph.
///
/// Requires `hg` connected (in the hypergraph sense: its bipartite
/// vertex-hyperedge incidence graph is connected) so that `0` is a simple
/// eigenvalue; on a disconnected hypergraph the two smallest Ritz values
/// returned by Lanczos may both be (numerically) the trivial `~0`
/// eigenvalue, in which case this returns a near-zero value rather than a
/// meaningful gap — inspect the full spectrum via
/// `spectral_hypergraph::laplacian::dense_normalized_laplacian` +
/// `spectral_hypergraph::spectral::dense_eigen` if that's a possibility.
pub fn hypergraph_algebraic_connectivity(
    hg: &SpectralHypergraph,
    krylov_dim: usize,
    seed: u64,
) -> HgResult<f64> {
    let op = HypergraphLaplacianOperator::new(hg)?;
    let n = op.dim();
    // Need at least the two smallest Ritz values (trivial 0 + the gap), so
    // the Krylov subspace must be at least 2-dimensional.
    let m = krylov_dim.min(n).max(2);
    let v0 = xorshift_start_vector(n, seed);

    let lanczos = Lanczos::new(m, 1e-12, Reorthogonalization::Full);
    let result = lanczos
        .run(&op, &v0)
        .expect("Lanczos failed on hypergraph Laplacian operator");
    let pairs = lanczos_ritz_pairs(&result);
    // `lanczos_ritz_pairs` sorts ascending; index 0 is the trivial ~0
    // eigenvalue (eigenvector ~ D_v^{1/2} * 1), index 1 is the gap.
    Ok(pairs.get(1).map(|p| p.value).unwrap_or(0.0))
}

/// Community-structured synthetic hypergraph generator, the hypergraph
/// analogue of [`crate::graph::stochastic_block_model`]: `k` blocks of
/// `block_size` vertices each. Within each block, `edges_per_block`
/// hyperedges are drawn, each a uniformly random `hyperedge_card`-subset of
/// the block's vertices; `cross_block_edges` additional hyperedges are
/// drawn with members split roughly evenly across two randomly chosen
/// distinct blocks (the higher-order analogue of the SBM's `p_out` bridge
/// edges — enough to keep the whole hypergraph connected without erasing
/// the block structure). Useful for exercising [`clique_expand`] +
/// [`crate::spectral::NbscFilterBank`] and
/// `spectral_hypergraph::spectral::spectral_cluster` against a common
/// ground truth.
pub fn hypergraph_stochastic_block_model(
    k: usize,
    block_size: usize,
    edges_per_block: usize,
    hyperedge_card: usize,
    cross_block_edges: usize,
    seed: u64,
) -> HgResult<(SpectralHypergraph, Vec<usize>)> {
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::Rng;
    use rand::SeedableRng;
    use spectral_hypergraph::hypergraph::HypergraphBuilder;

    let n = k * block_size;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut b = HypergraphBuilder::with_capacity(n, edges_per_block * k + cross_block_edges, hyperedge_card);

    let mut ids = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for block in 0..k {
        for local in 0..block_size {
            let v = b.add_vertex(format!("b{block}_v{local}"))?;
            ids.push(v);
            labels.push(block);
        }
    }

    let card = hyperedge_card.min(block_size).max(2);
    for block in 0..k {
        let block_ids: Vec<_> = ids[block * block_size..(block + 1) * block_size].to_vec();

        // Coverage pass: partition a shuffled copy of the block into
        // `card`-sized chunks and add each as a hyperedge, so every vertex
        // in the block is guaranteed to be a member of at least one
        // within-block hyperedge regardless of how the random pass below
        // happens to land. Without this, a vertex can end up isolated
        // (present in no sampled hyperedge) with non-negligible
        // probability at small `block_size` / `edges_per_block`, which
        // then fails every downstream routine that requires
        // `is_degree_normalizable`.
        let mut shuffled = block_ids.clone();
        shuffled.shuffle(&mut rng);
        for chunk in shuffled.chunks(card) {
            if chunk.len() >= 2 {
                b.add_hyperedge(chunk, 1.0)?;
            } else if let [lone] = chunk {
                // Leftover singleton from the chunking: pair it with a
                // random other block member so it still satisfies the
                // >=2-member hyperedge invariant while staying covered.
                let partner = block_ids[rng.gen_range(0..block_ids.len())];
                if partner != *lone {
                    b.add_hyperedge(&[*lone, partner], 1.0)?;
                } else if block_ids.len() > 1 {
                    let alt = block_ids.iter().find(|&&v| v != *lone).copied().unwrap();
                    b.add_hyperedge(&[*lone, alt], 1.0)?;
                }
            }
        }

        // Additional random within-block hyperedges on top of the coverage
        // pass, for the density/overlap `edges_per_block` actually asks for.
        for _ in 0..edges_per_block {
            let mut sample = block_ids.clone();
            sample.shuffle(&mut rng);
            sample.truncate(card);
            b.add_hyperedge(&sample, 1.0)?;
        }
    }

    if k >= 2 {
        for _ in 0..cross_block_edges {
            let mut blocks: Vec<usize> = (0..k).collect();
            blocks.shuffle(&mut rng);
            let (b1, b2) = (blocks[0], blocks[1]);
            let half = (card / 2).max(1);
            let mut members = Vec::with_capacity(card);
            let mut b1_ids: Vec<_> = ids[b1 * block_size..(b1 + 1) * block_size].to_vec();
            let mut b2_ids: Vec<_> = ids[b2 * block_size..(b2 + 1) * block_size].to_vec();
            b1_ids.shuffle(&mut rng);
            b2_ids.shuffle(&mut rng);
            members.extend_from_slice(&b1_ids[..half.min(b1_ids.len())]);
            members.extend_from_slice(&b2_ids[..(card - half).max(1).min(b2_ids.len())]);
            if members.len() >= 2 {
                b.add_hyperedge(&members, 0.5)?;
            }
        }
    }

    let hg = b.build()?;
    Ok((hg, labels))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use spectral_hypergraph::hypergraph::HypergraphBuilder;
    use spectral_hypergraph::laplacian::{clique_expansion_adjacency, dense_normalized_laplacian};
    use spectral_hypergraph::spectral::dense_eigen;

    fn triangle_plus_edge() -> SpectralHypergraph {
        let mut b = HypergraphBuilder::new();
        let a = b.add_vertex("a").unwrap();
        let v = b.add_vertex("b").unwrap();
        let c = b.add_vertex("c").unwrap();
        let d = b.add_vertex("d").unwrap();
        b.add_hyperedge(&[a, v, c], 1.0).unwrap();
        b.add_hyperedge(&[c, d], 1.0).unwrap();
        b.build().unwrap()
    }

    /// [`clique_expand`]'s nonzero pattern must exactly match
    /// `spectral_hypergraph::laplacian::clique_expansion_adjacency`'s.
    #[test]
    fn clique_expand_matches_reference_nonzero_pattern() {
        let hg = triangle_plus_edge();
        let g = clique_expand(&hg);
        let dense_adj = clique_expansion_adjacency(&hg).unwrap();

        assert_eq!(g.n, hg.num_vertices());
        for i in 0..g.n {
            for j in 0..g.n {
                let connected_in_g = g.neighbors[i].contains(&j);
                let connected_in_dense = dense_adj[(i, j)] > 0.0;
                assert_eq!(
                    connected_in_g, connected_in_dense,
                    "mismatch at ({i}, {j})"
                );
            }
        }
        // {a,b,c} clique: 3 edges. {c,d}: 1 edge. Total 4.
        assert_eq!(g.m(), 4);
    }

    /// [`HypergraphLaplacianOperator`] must apply identically to
    /// `spectral_hypergraph`'s own `HypergraphOperator` (it's a thin
    /// wrapper, but this pins the translation layer against regressions).
    #[test]
    fn laplacian_operator_adapter_matches_inner() {
        let hg = triangle_plus_edge();
        let inner = HypergraphOperator::new(&hg).unwrap();
        let adapter = HypergraphLaplacianOperator::new(&hg).unwrap();

        for i in 0..hg.num_vertices() {
            let mut e_i = vec![0.0; hg.num_vertices()];
            e_i[i] = 1.0;
            let mut via_adapter = vec![0.0; hg.num_vertices()];
            adapter.apply(&e_i, &mut via_adapter);

            let e_i_dvec = DVector::from_row_slice(&e_i);
            let via_inner = SpectralHypergraphOperator::apply(&inner, &e_i_dvec);

            for j in 0..hg.num_vertices() {
                assert_relative_eq!(via_adapter[j], via_inner[j], epsilon = 1e-12);
            }
        }
    }

    /// The operator norm found via `krylov_ds::Lanczos` on the adapter must
    /// agree with a brute-force dense eigendecomposition of the true
    /// normalized hypergraph Laplacian.
    #[test]
    fn hypergraph_laplacian_operator_norm_matches_dense_ground_truth() {
        let hg = triangle_plus_edge();
        let dense = dense_normalized_laplacian(&hg).unwrap();
        let dense_norm = dense_eigen(&dense)
            .eigenvalues
            .iter()
            .map(|x| x.abs())
            .fold(0.0, f64::max);

        let krylov_norm = hypergraph_laplacian_operator_norm(&hg, 4, 5).unwrap();
        assert_relative_eq!(krylov_norm, dense_norm, epsilon = 1e-6, max_relative = 1e-4);
    }

    /// The algebraic connectivity found via `krylov_ds::Lanczos` on the
    /// adapter must agree with the second-smallest eigenvalue of the dense
    /// normalized hypergraph Laplacian.
    #[test]
    fn hypergraph_algebraic_connectivity_matches_dense_ground_truth() {
        let hg = triangle_plus_edge();
        let dense = dense_normalized_laplacian(&hg).unwrap();
        let mut eigs: Vec<f64> = dense_eigen(&dense).eigenvalues.iter().copied().collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let dense_gap = eigs[1];

        let krylov_gap = hypergraph_algebraic_connectivity(&hg, 4, 5).unwrap();
        assert_relative_eq!(krylov_gap, dense_gap, epsilon = 1e-6, max_relative = 1e-4);
    }

    /// The generator should produce a hypergraph with the requested shape
    /// and a block-structured, degree-normalizable result connected enough
    /// for spectral routines to run.
    #[test]
    fn hypergraph_sbm_shape_and_connectivity() {
        let (hg, labels) =
            hypergraph_stochastic_block_model(3, 12, 8, 4, 4, 42).unwrap();
        assert_eq!(hg.num_vertices(), 36);
        assert_eq!(labels.len(), 36);
        assert!(hg.is_degree_normalizable(), "SBM hypergraph should have no isolated vertices at this density");

        // Clique-expanding should give a connected graph (the cross-block
        // hyperedges act as bridges) with more than just the within-block
        // structure.
        let g = clique_expand(&hg);
        assert!(g.is_connected(), "clique-expanded SBM hypergraph should be connected");
    }

    /// [`nbsc_filter_bank_from_hypergraph`] should produce a finite,
    /// positive `rho_B` (i.e. the clique-expanded SBM hypergraph is a
    /// non-degenerate graph as far as the Hashimoto spectrum is concerned)
    /// and taps of the right shape.
    #[test]
    fn filter_bank_from_hypergraph_runs_end_to_end() {
        let (hg, _labels) =
            hypergraph_stochastic_block_model(3, 10, 6, 3, 3, 7).unwrap();
        let (g, bank) = nbsc_filter_bank_from_hypergraph(&hg, 30, 1);
        assert!(bank.rho_b.is_finite() && bank.rho_b > 0.0);

        let f = 2;
        let x = crate::spectral::FeatureMatrix::zeros(g.n, f);
        let taps = bank.apply_taps(&g, &x, 3);
        assert_eq!(taps.len(), 4);
        for t in &taps {
            assert_eq!(t.n, g.n);
            assert_eq!(t.f, f);
        }
    }
}
