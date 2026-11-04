//! §7-8 of `ihara_zeta.rs`: the actual learnable layer.
//!
//! ```text
//! H = sigma( sum_{k=0}^{K} T_k X W_k + b )
//! ```
//!
//! `T_k` is graph-fixed (depends only on `A`, `D`, `rho_B` — computed once
//! per graph by [`crate::spectral::NbscFilterBank`] / here directly as Burn
//! tensors) and *not learnable*; only `{W_k}` and `b` are. Because `T_k` is
//! a fixed linear operator, `T_k X` is implemented here with ordinary Burn
//! tensor ops (`matmul`), so Burn's autodiff produces the correct
//! `d/dX` and `d/dW_k` gradients automatically — no hand-derived VJP is
//! needed for the recursion itself, exactly as argued in §8.
//!
//! Implementation note (also flagged in §8's TODO): `A` and `D-I` are held
//! here as *dense* `n x n` Burn tensors. That's the right tradeoff for the
//! graph sizes in this benchmark (correctness and simplicity, and it keeps
//! every op inside Burn's autodiff graph with zero custom backward code).
//! For large sparse graphs, the dense `n x n` matmuls should be replaced
//! with a custom sparse (CSR) kernel + hand-written backward, since Burn
//! 0.13 does not yet have mature sparse-tensor autodiff — the sparse CPU
//! path in [`crate::spectral::NbscFilterBank`] shows what that kernel's
//! forward pass looks like.

use crate::graph::Graph;
use crate::spectral::estimate_spectral_radius;
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{activation::relu, Tensor};

/// Builds the dense `n x n` adjacency tensor and the length-`n` `(D-I)`
/// vector for a graph, on the given backend/device. Shared by both the
/// NBSC and GCN layers below so the two comparators see byte-identical
/// graph inputs.
fn dense_adjacency_tensor<B: Backend>(graph: &Graph, device: &B::Device) -> Tensor<B, 2> {
    let data: Vec<f32> = graph.dense_adjacency().into_iter().map(|v| v as f32).collect();
    Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([graph.n, graph.n])
}

fn degrees_tensor<B: Backend>(graph: &Graph, device: &B::Device) -> Tensor<B, 1> {
    let d: Vec<f32> = graph.degrees().into_iter().map(|v| v as f32).collect();
    Tensor::<B, 1>::from_floats(d.as_slice(), device)
}

// ---------------------------------------------------------------------
// NBSC layer
// ---------------------------------------------------------------------

#[derive(Config, Debug)]
pub struct NbscLayerConfig {
    pub d_input: usize,
    pub d_output: usize,
    /// Number of non-backtracking filter taps `K` (uses `T_0..T_K`, i.e.
    /// `K+1` learnable projections).
    pub k_taps: usize,
    #[config(default = true)]
    pub bias: bool,
    /// Ablation switch (off by default, so existing configs/results are
    /// unaffected): applies `LayerNorm` to the summed pre-activation output
    /// before the bias+ReLU. Added to test the hypothesis that NBSC's
    /// growing Dirichlet energy and depth-3 variance blowup on Cora (see
    /// `examples/operator_norm_check.rs`) come from the `A / rho_B` tap
    /// lacking GCN's non-expansive-by-construction normalization -- if
    /// enabling this flattens the energy trend and closes some of the
    /// accuracy gap at depth 3, that's triangulating evidence for the
    /// hypothesis, independent of the operator-norm check itself.
    #[config(default = false)]
    pub normalize: bool,
}

#[derive(Module, Debug)]
pub struct NbscLayer<B: Backend> {
    taps: Vec<Linear<B>>,
    bias: Option<burn::module::Param<Tensor<B, 1>>>,
    norm: Option<LayerNorm<B>>,
    adjacency: Tensor<B, 2>,
    d_minus_i: Tensor<B, 1>,
    rho_b: f64,
    k_taps: usize,
}

impl NbscLayerConfig {
    /// Estimates `rho_B` via `krylov_ds` Arnoldi on the matrix-free
    /// Hashimoto linearization (§4-6), then builds the layer's dense `A`
    /// and `(D-I)` buffers and its `K+1` learnable per-tap projections.
    pub fn init<B: Backend>(&self, graph: &Graph, device: &B::Device) -> NbscLayer<B> {
        let rho_b = estimate_spectral_radius(graph, graph.n.min(40), 0).max(1e-6);
        let adjacency = dense_adjacency_tensor::<B>(graph, device);
        let degrees = degrees_tensor::<B>(graph, device);
        let d_minus_i = degrees - 1.0;

        let taps: Vec<Linear<B>> = (0..=self.k_taps)
            .map(|_| {
                LinearConfig::new(self.d_input, self.d_output)
                    .with_bias(false)
                    .init(device)
            })
            .collect();

        let bias = if self.bias {
            Some(burn::module::Param::from_tensor(Tensor::zeros(
                [self.d_output],
                device,
            )))
        } else {
            None
        };

        let norm = if self.normalize {
            Some(LayerNormConfig::new(self.d_output).init(device))
        } else {
            None
        };

        NbscLayer { taps, bias, norm, adjacency, d_minus_i, rho_b, k_taps: self.k_taps }
    }
}

impl<B: Backend> NbscLayer<B> {
    /// Applies the rescaled non-backtracking recursion (§6) directly on
    /// Burn tensors and sums the learned per-tap projections (§7). `x` is
    /// `[n, d_input]`.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // T_0 X = X
        let t0 = x.clone();
        let mut out = self.taps[0].forward(t0.clone());

        if self.k_taps >= 1 {
            // T_1 X = (A / rho_B) X
            let t1 = self.adjacency.clone().matmul(x.clone()) / self.rho_b;
            out = out + self.taps[1].forward(t1.clone());

            let mut prev2 = t0; // T_{k-1} X
            let mut prev = t1; // T_k X
            let rho2 = self.rho_b * self.rho_b;

            for kk in 2..=self.k_taps {
                let a_term = self.adjacency.clone().matmul(prev.clone()) * (2.0 / self.rho_b);
                // (D-I) X, broadcasting the length-n diagonal over feature columns.
                let d_term = prev2.clone() * self.d_minus_i.clone().unsqueeze_dim::<2>(1) / rho2;
                let next = a_term - d_term;
                out = out + self.taps[kk].forward(next.clone());
                prev2 = prev;
                prev = next;
            }
        }

        if let Some(b) = &self.bias {
            out = out + b.val().unsqueeze::<2>();
        }
        if let Some(n) = &self.norm {
            out = n.forward(out);
        }
        relu(out)
    }
}

// ---------------------------------------------------------------------
// GCN baseline layer (same interface, for direct comparison)
// ---------------------------------------------------------------------

#[derive(Config, Debug)]
pub struct GcnLayerConfig {
    pub d_input: usize,
    pub d_output: usize,
    #[config(default = true)]
    pub bias: bool,
}

#[derive(Module, Debug)]
pub struct GcnLayer<B: Backend> {
    linear: Linear<B>,
    a_hat: Tensor<B, 2>, // dense, precomputed D^{-1/2}(A+I)D^{-1/2}
}

impl GcnLayerConfig {
    pub fn init<B: Backend>(&self, graph: &Graph, device: &B::Device) -> GcnLayer<B> {
        let n = graph.n;
        let mut a = graph.dense_adjacency();
        for i in 0..n {
            a[i * n + i] = 1.0; // self-loop
        }
        let degrees = graph.degrees();
        let inv_sqrt: Vec<f64> = degrees.iter().map(|&d| 1.0 / (d + 1.0).sqrt()).collect();
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] *= inv_sqrt[i] * inv_sqrt[j];
            }
        }
        let a_f32: Vec<f32> = a.into_iter().map(|v| v as f32).collect();
        let a_hat = Tensor::<B, 1>::from_floats(a_f32.as_slice(), device).reshape([n, n]);

        let linear = LinearConfig::new(self.d_input, self.d_output)
            .with_bias(self.bias)
            .with_initializer(Initializer::KaimingUniform { gain: 1.0 / (3.0f64).sqrt(), fan_out_only: false })
            .init(device);

        GcnLayer { linear, a_hat }
    }
}

impl<B: Backend> GcnLayer<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let propagated = self.a_hat.clone().matmul(x);
        relu(self.linear.forward(propagated))
    }
}

/// Dirichlet energy of a feature matrix w.r.t. the graph,
/// `E(X) = (1/|E|) * sum_{(u,v) in E} ||x_u - x_v||^2`, normalized by the
/// mean squared row norm so it's comparable across layers/widths. This is
/// the standard over-smoothing diagnostic: as depth increases, a GCN's
/// representations collapse toward a constant vector per connected
/// component and `E(X) -> 0`; a layer that resists over-smoothing keeps
/// `E(X)` from collapsing as quickly.
pub fn dirichlet_energy<B: Backend>(graph: &Graph, x: &Tensor<B, 2>) -> f32 {
    let data = x.to_data().convert::<f32>();
    let n = graph.n;
    let f = data.shape.dims[1];
    let values = data.value;

    let mut sq_diff_sum = 0.0f64;
    for &(u, v) in &graph.edges {
        let mut d = 0.0f64;
        for j in 0..f {
            let diff = values[u * f + j] - values[v * f + j];
            d += (diff * diff) as f64;
        }
        sq_diff_sum += d;
    }
    let energy = sq_diff_sum / graph.m().max(1) as f64;

    let mut norm_sum = 0.0f64;
    for i in 0..n {
        for j in 0..f {
            norm_sum += (values[i * f + j] * values[i * f + j]) as f64;
        }
    }
    let mean_sq_norm = (norm_sum / n as f64).max(1e-12);

    (energy / mean_sq_norm) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::random_near_regular;
    use burn::backend::ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn nbsc_layer_forward_runs_and_shapes_match() {
        let device = Default::default();
        let g = random_near_regular(20, 4, 1);
        let cfg = NbscLayerConfig::new(6, 8, 3);
        let layer = cfg.init::<B>(&g, &device);

        let x = Tensor::<B, 2>::zeros([g.n, 6], &device) + 1.0;
        let out = layer.forward(x);
        assert_eq!(out.dims(), [g.n, 8]);
    }

    #[test]
    fn gcn_layer_forward_runs_and_shapes_match() {
        let device = Default::default();
        let g = random_near_regular(20, 4, 2);
        let cfg = GcnLayerConfig::new(6, 8);
        let layer = cfg.init::<B>(&g, &device);

        let x = Tensor::<B, 2>::zeros([g.n, 6], &device) + 1.0;
        let out = layer.forward(x);
        assert_eq!(out.dims(), [g.n, 8]);
    }

    #[test]
    fn dirichlet_energy_is_zero_for_constant_signal() {
        let device = Default::default();
        let g = random_near_regular(15, 3, 3);
        let x = Tensor::<B, 2>::zeros([g.n, 4], &device) + 2.0;
        let e = dirichlet_energy::<B>(&g, &x);
        assert!(e < 1e-6, "constant signal should have ~0 Dirichlet energy, got {e}");
    }

    #[test]
    fn dirichlet_energy_is_positive_for_random_signal() {
        let device = Default::default();
        let g = random_near_regular(15, 3, 4);
        let x = Tensor::<B, 2>::random(
            [g.n, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let e = dirichlet_energy::<B>(&g, &x);
        assert!(e > 1e-4, "random signal should have non-trivial Dirichlet energy, got {e}");
    }
}
