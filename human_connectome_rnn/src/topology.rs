//! Connectome topology and harmonics, built on the attached systems:
//!
//! * `spectral_hypergraph`: the connectome as a 2-uniform weighted hypergraph;
//!   its normalised Laplacian, eigen-harmonics and spectral clustering (modules).
//! * `nbsc`: non-backtracking (Hashimoto) spectral radius rho_B and the adjacency
//!   operator norm of the wiring graph, via `krylov_ds` Arnoldi/Lanczos.

use crate::connectome::Connectome;
use nalgebra::DMatrix;
use spectral_hypergraph::laplacian::dense_normalized_laplacian;
use spectral_hypergraph::spectral::{dense_eigen, spectral_cluster};
use spectral_hypergraph::{HypergraphBuilder, SpectralHypergraph};

/// The connectome as a 2-uniform weighted hypergraph (one hyperedge per tract).
pub fn connectome_hypergraph(c: &Connectome) -> Result<SpectralHypergraph, String> {
    let n = c.n();
    let mut b = HypergraphBuilder::new();
    let ids: Vec<_> = (0..n)
        .map(|i| b.add_vertex(format!("v{i}")).map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    for i in 0..n {
        for j in (i + 1)..n {
            let w = c.weight(i, j);
            if w > 0.0 {
                b.add_hyperedge(&[ids[i], ids[j]], w).map_err(|e| e.to_string())?;
            }
        }
    }
    b.build().map_err(|e| e.to_string())
}

/// Symmetric normalised graph Laplacian L = I - D^-1/2 A D^-1/2 (spectrum in [0, 2]),
/// dense row-major. For a 2-uniform hypergraph the system's Laplacian equals L/2, so
/// we take it from `spectral_hypergraph` whenever it is defined and fall back to a direct
/// computation only when some region has no tract (isolated vertex).
pub fn sym_laplacian(c: &Connectome) -> Vec<f64> {
    let n = c.n();
    if let Ok(hg) = connectome_hypergraph(c) {
        if hg.is_degree_normalizable() {
            if let Ok(l) = dense_normalized_laplacian(&hg) {
                let mut out = vec![0.0; n * n];
                for i in 0..n {
                    for j in 0..n {
                        out[i * n + j] = 2.0 * l[(i, j)];
                    }
                }
                return out;
            }
        }
    }
    let s: Vec<f64> = (0..n).map(|i| c.strength(i)).collect();
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        if s[i] > 0.0 {
            out[i * n + i] = 1.0;
        }
        for j in 0..n {
            if i != j && c.weight(i, j) > 0.0 {
                out[i * n + j] = -c.weight(i, j) / (s[i] * s[j]).sqrt();
            }
        }
    }
    out
}

/// Eigen-decomposition of the normalised Laplacian ("connectome harmonics"),
/// ascending eigenvalues in [0, 2]; `vectors[(region, k)]` is harmonic k.
pub struct Harmonics {
    pub values: Vec<f64>,
    pub vectors: DMatrix<f64>,
}

pub fn harmonics(c: &Connectome) -> Harmonics {
    let n = c.n();
    let l = sym_laplacian(c);
    let m = DMatrix::from_fn(n, n, |i, j| 0.5 * (l[i * n + j] + l[j * n + i]));
    let e = dense_eigen(&m);
    Harmonics { values: e.eigenvalues.iter().copied().collect(), vectors: e.eigenvectors }
}

/// Data-driven modules: spectral clustering of the connectome hypergraph.
pub fn spectral_modules(c: &Connectome, k: usize, seed: u64) -> Result<Vec<usize>, String> {
    let hg = connectome_hypergraph(c)?;
    spectral_cluster(&hg, k, false, seed).map(|a| a.assignments).map_err(|e| e.to_string())
}

#[derive(Clone, Debug)]
pub struct TopologyReport {
    pub n: usize,
    pub edges: usize,
    pub mean_degree: f64,
    /// Perron-Frobenius radius of the Hashimoto (non-backtracking) matrix (nbsc).
    pub rho_b: f64,
    /// Spectral norm of the adjacency matrix (nbsc, Lanczos).
    pub adj_norm: f64,
    /// adj_norm / rho_b; > 1 means the rescaled operator A/rho_B is expansive.
    pub expansion_ratio: f64,
    /// Second-smallest eigenvalue of the normalised Laplacian (algebraic connectivity).
    pub fiedler: f64,
}

pub fn topology_report(c: &Connectome) -> TopologyReport {
    let n = c.n();
    let mut g = nbsc::graph::Graph::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            if c.weight(i, j) > 0.0 {
                g.add_edge(i, j);
            }
        }
    }
    let kd = (2 * n).min(80);
    let rho_b = nbsc::spectral::estimate_spectral_radius(&g, kd, 1);
    let adj_norm = nbsc::spectral::adjacency_operator_norm(&g, n.min(80), 1);
    let h = harmonics(c);
    TopologyReport {
        n,
        edges: g.m(),
        mean_degree: 2.0 * g.m() as f64 / n as f64,
        rho_b,
        adj_norm,
        expansion_ratio: if rho_b > 0.0 { adj_norm / rho_b } else { f64::NAN },
        fiedler: h.values.get(1).copied().unwrap_or(0.0),
    }
}
