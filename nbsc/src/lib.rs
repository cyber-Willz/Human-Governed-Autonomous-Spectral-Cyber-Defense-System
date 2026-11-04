//! Non-Backtracking Spectral Convolution (NBSC).
//!
//! Graph convolutions built from the poles of the Ihara zeta function
//! (equivalently, the eigenvalues of the Hashimoto / non-backtracking
//! matrix `B`) instead of the normalized graph Laplacian. See
//! `ihara_zeta.rs` (the original derivation this crate implements) for the
//! full math; module docs below point at which section each piece covers.
//!
//! - [`graph`] — graph representation + synthetic generators (SBM, tree,
//!   near-regular expander) used for benchmarking.
//! - [`spectral`] — §2-6: Bass reduction, the matrix-free linearized
//!   Hashimoto operator, `rho_B` estimation via `krylov_ds` Arnoldi, and the
//!   sparse `T_k` feature-recursion filter bank.
//! - [`gcn`] — a standard symmetric-normalized-Laplacian GCN propagation
//!   rule, implemented the same way as [`spectral::NbscFilterBank`], used
//!   purely as the baseline comparator.
//! - [`burn_layer`] (feature `burn`) — §7-8: the actual learnable NBSC and
//!   GCN layers as `burn::Module`s, differentiable end-to-end via Burn's
//!   autodiff, plus a Dirichlet-energy over-smoothing metric.
//! - [`gat_layer`] (feature `burn`) — GAT baseline (Velickovic et al. 2018).
//! - [`sage_layer`] (feature `burn`) — GraphSAGE baseline, mean aggregator
//!   (Hamilton, Ying & Leskovec 2017).
//! - [`dataset`] — real-data loading (Cora) with provenance and
//!   train/val/test split documentation; no `burn` dependency, usable
//!   standalone for data inspection/validation.
//! - [`hypergraph_bridge`] (feature `hypergraph`) — bridges the
//!   `spectral_hypergraph` crate into this pipeline: clique-expand a
//!   `SpectralHypergraph` into a [`graph::Graph`] (so every filter/baseline
//!   above runs unmodified on hypergraph data), plus a `krylov_ds`
//!   `LinearOperator` adapter around `spectral_hypergraph`'s matrix-free
//!   normalized hypergraph Laplacian so this crate's own Arnoldi/Lanczos
//!   engine (the one behind `rho_B`) can be run directly against the true
//!   hypergraph structure, not just its clique expansion.

pub mod dataset;
pub mod gcn;
pub mod graph;
pub mod spectral;

#[cfg(feature = "burn")]
pub mod burn_layer;
#[cfg(feature = "burn")]
pub mod gat_layer;
#[cfg(feature = "hypergraph")]
pub mod hypergraph_bridge;
#[cfg(feature = "burn")]
pub mod sage_layer;
