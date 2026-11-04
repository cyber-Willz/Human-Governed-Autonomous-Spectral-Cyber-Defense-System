//! Baseline: the standard Kipf & Welling GCN propagation rule,
//!
//! ```text
//! A_hat = D^{-1/2} (A + I) D^{-1/2}
//! H' = sigma(A_hat X W)
//! ```
//!
//! implemented with the same sparse, `O(|E| * f)` style as
//! [`crate::spectral::NbscFilterBank`] so that depth-for-depth,
//! feature-width-for-feature-width comparisons in the benchmark are apples
//! to apples: the only difference under test is which spectrum (Laplacian
//! vs. non-backtracking/Hashimoto) drives the propagation.

use crate::graph::Graph;
use crate::spectral::FeatureMatrix;

/// Precomputed, graph-fixed normalization `D^{-1/2}` for `A_hat`.
pub struct GcnPropagator {
    inv_sqrt_deg: Vec<f64>,
}

impl GcnPropagator {
    pub fn build(graph: &Graph) -> Self {
        let inv_sqrt_deg = graph
            .degrees()
            .iter()
            .map(|&d| 1.0 / (d + 1.0).sqrt()) // +1 for the added self-loop (A+I)
            .collect();
        Self { inv_sqrt_deg }
    }

    /// `Y = A_hat @ X`, sparse, self-loop included.
    pub fn propagate(&self, graph: &Graph, x: &FeatureMatrix) -> FeatureMatrix {
        assert_eq!(x.n, graph.n);
        let mut out = FeatureMatrix::zeros(x.n, x.f);
        for v in 0..graph.n {
            let row_out = out.row_mut(v);
            // self-loop term
            let self_w = self.inv_sqrt_deg[v] * self.inv_sqrt_deg[v];
            for (o, &xv) in row_out.iter_mut().zip(x.row(v).iter()) {
                *o += self_w * xv;
            }
            for &u in &graph.neighbors[v] {
                let w = self.inv_sqrt_deg[v] * self.inv_sqrt_deg[u];
                let row_u = x.row(u);
                let row_out = out.row_mut(v);
                for (o, &xu) in row_out.iter_mut().zip(row_u.iter()) {
                    *o += w * xu;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::random_near_regular;

    #[test]
    fn propagation_preserves_constant_signal_reasonably() {
        // On a d-regular graph, A_hat applied to the all-ones vector should
        // stay close to 1 everywhere (it's exactly the top eigenvector of
        // A_hat with eigenvalue ~1).
        let g = random_near_regular(40, 4, 5);
        let prop = GcnPropagator::build(&g);
        let ones = FeatureMatrix::from_rows(g.n, 1, vec![1.0; g.n]);
        let out = prop.propagate(&g, &ones);
        for v in 0..g.n {
            assert!((out.row(v)[0] - 1.0).abs() < 0.35, "value {} too far from 1", out.row(v)[0]);
        }
    }
}
