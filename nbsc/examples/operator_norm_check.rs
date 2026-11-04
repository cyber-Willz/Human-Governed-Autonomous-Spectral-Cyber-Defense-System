//! Standalone diagnostic (no `burn` feature required — pure linear algebra
//! on the loaded graph). Tests a specific hypothesis raised by the
//! `benchmark_cora` results: NBSC's Dirichlet energy *grows* with depth
//! (10.57 at depth 2, ~flat but with 8x higher cross-seed variance at
//! depth 3) while GCN/GAT/GraphSAGE's energy *shrinks* with depth, the
//! expected over-smoothing direction. GCN's symmetric-normalized
//! `D^-1/2 (A+I) D^-1/2` propagator is non-expansive by construction
//! (operator norm exactly 1); NBSC's `A / rho_B` tap has no such
//! guarantee, since `rho_B` is the spectral radius of the *Hashimoto*
//! (non-backtracking) matrix, a different operator from `A` itself.
//!
//! If `||A||_2 / rho_B > 1`, `A / rho_B` is expansive, which would plausibly
//! explain both the energy growth and the variance blowup at depth 3 as
//! compounding amplification rather than ordinary over-smoothing.
//!
//! Run with: `cargo run --release --example operator_norm_check`
//! (deliberately no `--features burn` needed)

use nbsc::dataset::Dataset;
use nbsc::spectral::{adjacency_operator_norm, estimate_spectral_radius};

fn main() {
    let ds = Dataset::load_cora_default(0).expect(
        "failed to load Cora -- check that nbsc/data/cora/{cora.content,cora.cites} exist",
    );
    println!("Cora: n={}, m={}", ds.graph.n, ds.graph.m());

    // krylov_dim: same default reasoning as NbscFilterBank::build uses
    // internally for rho_B (2 * n.min(40)); kept explicit here so both
    // quantities are computed with comparable Krylov-subspace sizes.
    let krylov_dim = (2 * ds.graph.n).min(80);
    let seed = 0;

    let rho_b = estimate_spectral_radius(&ds.graph, krylov_dim, seed);
    let a_norm = adjacency_operator_norm(&ds.graph, krylov_dim, seed);
    let ratio = a_norm / rho_b;

    println!("rho_B (Hashimoto spectral radius)   = {rho_b:.6}");
    println!("||A||_2 (adjacency operator norm)   = {a_norm:.6}");
    println!("||A||_2 / rho_B                     = {ratio:.6}");
    println!();
    if ratio > 1.0 {
        println!(
            "ratio > 1: A / rho_B IS expansive on this graph. This is consistent with \
             (though does not on its own prove) the hypothesis that NBSC's growing \
             Dirichlet energy and depth-3 variance blowup on Cora come from compounding \
             amplification through stacked layers, unlike GCN's provably non-expansive \
             symmetric-normalized propagator."
        );
    } else {
        println!(
            "ratio <= 1: A / rho_B is NOT expansive on this graph by this measure. The \
             energy-growth anomaly is more likely coming from somewhere else -- e.g. the \
             (D-I)/rho_B^2 term in the 3-term recursion for taps k>=2, the per-tap linear \
             weights W_k themselves (unconstrained, no weight decay in the current \
             benchmark), or an interaction between the two. Worth checking the (D-I) term's \
             contribution separately before ruling the normalization hypothesis out."
        );
    }
}
