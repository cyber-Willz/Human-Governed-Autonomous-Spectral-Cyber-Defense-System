//! Demonstrates both integration paths in [`nbsc::hypergraph_bridge`] on a
//! single synthetic community-structured hypergraph, and cross-checks each
//! against both crates' own ground truth.
//!
//! No `burn` dependency: this only exercises the pure-CPU spectral/graph
//! machinery (clique expansion, `rho_B`, the matrix-free hypergraph
//! Laplacian operator adapter), not the learnable layers.
//!
//! Run with: `cargo run --release --example hypergraph_bridge_demo --features hypergraph`

use nbsc::hypergraph_bridge::{
    hypergraph_algebraic_connectivity, hypergraph_laplacian_operator_norm,
    hypergraph_stochastic_block_model, nbsc_filter_bank_from_hypergraph,
};
use nbsc::spectral::{adjacency_operator_norm, FeatureMatrix};
use spectral_hypergraph::spectral::spectral_cluster;

const SEED: u64 = 11;

/// Cluster-recovery accuracy up to label permutation: for each predicted
/// cluster, count members matching its plurality ground-truth label, sum
/// across clusters, divide by n. A cheap majority-vote purity metric --
/// good enough to see "did spectral clustering recover the SBM blocks",
/// not a publication-grade clustering metric.
fn purity(assignments: &[usize], labels: &[usize], k: usize) -> f64 {
    let n = assignments.len();
    let mut correct = 0usize;
    for cluster in 0..k {
        let mut counts = vec![0usize; k];
        for i in 0..n {
            if assignments[i] == cluster {
                counts[labels[i]] += 1;
            }
        }
        correct += counts.into_iter().max().unwrap_or(0);
    }
    correct as f64 / n as f64
}

fn main() {
    println!("=== nbsc <-> spectral_hypergraph bridge demo ===\n");

    // A 4-community hypergraph: triples drawn within each block plus a
    // handful of cross-block bridging hyperedges, analogous in spirit to
    // `nbsc::graph::stochastic_block_model` but at the hyperedge level.
    let k = 4;
    let block_size = 25;
    let (hg, labels) =
        hypergraph_stochastic_block_model(k, block_size, 30, 3, 6, SEED)
            .expect("hypergraph SBM construction should succeed at this density");
    println!(
        "hypergraph: {} vertices, {} hyperedges, {} blocks of {}",
        hg.num_vertices(),
        hg.num_hyperedges(),
        k,
        block_size
    );

    // --- Path 1: clique expansion -> ordinary Graph -> full NBSC pipeline ---
    let (graph, bank) = nbsc_filter_bank_from_hypergraph(&hg, 40, SEED);
    println!(
        "\n[path 1: clique expansion] graph: n={}, m={}, connected={}, bipartite={}",
        graph.n,
        graph.m(),
        graph.is_connected(),
        graph.is_bipartite()
    );
    println!("  rho_B (Hashimoto spectral radius)   = {:.6}", bank.rho_b);

    let adj_norm = adjacency_operator_norm(&graph, 40, SEED);
    println!("  ||A||_2 (plain adjacency op norm)    = {:.6}", adj_norm);
    println!(
        "  ||A||_2 / rho_B                      = {:.6}  ({})",
        adj_norm / bank.rho_b,
        if adj_norm / bank.rho_b > 1.0 {
            "A/rho_B tap is expansive"
        } else {
            "A/rho_B tap is non-expansive"
        }
    );

    // Exercise the filter bank itself: three taps applied to a random
    // feature matrix, just to confirm shapes flow through end-to-end on
    // hypergraph-derived structure exactly as they would on a plain graph.
    let f = 4;
    let mut x = FeatureMatrix::zeros(graph.n, f);
    let mut state = SEED ^ 0x9E3779B97F4A7C15;
    for v in x.data.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *v = (state as f64 / u64::MAX as f64) - 0.5;
    }
    let taps = bank.apply_taps(&graph, &x, 3);
    println!(
        "  filter bank taps: {} taps, each {}x{}",
        taps.len(),
        taps[0].n,
        taps[0].f
    );

    // --- Path 2: matrix-free hypergraph Laplacian operator, via krylov_ds ---
    println!("\n[path 2: matrix-free hypergraph Laplacian, via krylov_ds]");
    let lap_norm = hypergraph_laplacian_operator_norm(&hg, 40, SEED)
        .expect("hypergraph Laplacian operator norm should converge");
    let algebraic_connectivity = hypergraph_algebraic_connectivity(&hg, 40, SEED)
        .expect("hypergraph algebraic connectivity should converge");
    println!("  ||Delta||_2 (hypergraph Laplacian op norm) = {:.6}", lap_norm);
    println!("  algebraic connectivity (spectral gap)      = {:.6}", algebraic_connectivity);
    println!(
        "  (both computed with this crate's own krylov_ds::Lanczos engine, \
         not spectral_hypergraph's built-in Lanczos)"
    );

    // --- Cross-validation: does clique-expansion-derived community signal
    // agree with spectral_hypergraph's native hypergraph spectral
    // clustering? Both should recover the same k=4 blocks. ---
    println!("\n[cross-check: community recovery]");
    let hg_clusters = spectral_cluster(&hg, k, false, SEED)
        .expect("spectral clustering on the true hypergraph Laplacian should succeed");
    let hg_purity = purity(&hg_clusters.assignments, &labels, k);
    println!(
        "  spectral_hypergraph::spectral_cluster purity (native hypergraph Laplacian) = {:.3}",
        hg_purity
    );

    // Same k-way spectral clustering, but on the clique-expanded graph's
    // own Laplacian eigenvectors (dense, via nalgebra directly -- this
    // crate doesn't ship a graph spectral-clustering routine, so this is a
    // deliberately small amount of glue code to make the comparison, not a
    // gap in the bridge module itself).
    let ce_purity = clique_expansion_spectral_cluster_purity(&graph, &labels, k, SEED);
    println!(
        "  clique-expansion Laplacian spectral clustering purity                     = {:.3}",
        ce_purity
    );
    println!(
        "\nBoth paths draw on the same underlying hyperedges; agreement here is the \
         end-to-end sanity check that the bridge module's clique expansion and the \
         native hypergraph Laplacian are looking at consistent structure."
    );
}

/// Minimal k-means-on-Laplacian-eigenvectors spectral clustering for a
/// plain [`nbsc::graph::Graph`], used only to give path 1 a directly
/// comparable purity number against `spectral_hypergraph::spectral_cluster`
/// in this demo. Dense and `O(n^3)`; fine at demo scale, not something this
/// bridge module exposes as public API since `nbsc` otherwise has no
/// graph-clustering routine of its own to keep parity with.
fn clique_expansion_spectral_cluster_purity(
    graph: &nbsc::graph::Graph,
    labels: &[usize],
    k: usize,
    seed: u64,
) -> f64 {
    let n = graph.n;
    let degrees = graph.degrees();
    let mut l = nalgebra::DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        l[(i, i)] = if degrees[i] > 0.0 { 1.0 } else { 0.0 };
        for &j in &graph.neighbors[i] {
            if degrees[i] > 0.0 && degrees[j] > 0.0 {
                l[(i, j)] = -1.0 / (degrees[i].sqrt() * degrees[j].sqrt());
            }
        }
    }
    let eig = nalgebra::linalg::SymmetricEigen::new(l);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[a].partial_cmp(&eig.eigenvalues[b]).unwrap());
    let embed_dim = k.min(n);
    let mut embedding = nalgebra::DMatrix::<f64>::zeros(n, embed_dim);
    for (col, &orig) in order.iter().take(embed_dim).enumerate() {
        embedding.set_column(col, &eig.eigenvectors.column(orig));
    }

    // Simple seeded k-means (Lloyd's algorithm), local to this demo.
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = StdRng::seed_from_u64(seed ^ 0xC1057E12);
    let mut centroids: Vec<nalgebra::DVector<f64>> = (0..k)
        .map(|_| embedding.row(rng.gen_range(0..n)).transpose())
        .collect();
    let mut assignments = vec![0usize; n];
    for _ in 0..100 {
        let mut changed = false;
        for i in 0..n {
            let p = embedding.row(i).transpose();
            let (mut best, mut best_dist) = (0usize, f64::INFINITY);
            for (c_idx, c) in centroids.iter().enumerate() {
                let d = (&p - c).norm_squared();
                if d < best_dist {
                    best_dist = d;
                    best = c_idx;
                }
            }
            if assignments[i] != best {
                changed = true;
            }
            assignments[i] = best;
        }
        let mut sums = vec![nalgebra::DVector::<f64>::zeros(embed_dim); k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            sums[assignments[i]] += embedding.row(i).transpose();
            counts[assignments[i]] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                centroids[c] = &sums[c] / counts[c] as f64;
            }
        }
        if !changed {
            break;
        }
    }
    purity(&assignments, labels, k)
}
