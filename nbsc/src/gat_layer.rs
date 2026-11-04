//! Graph Attention Network (GAT) layer (Velickovic et al., 2018), baseline
//! #2 alongside [`crate::burn_layer::GcnLayer`]. Where GCN's per-edge
//! weight is fixed by degree normalization, GAT *learns* per-edge
//! attention weights from the node features themselves:
//!
//! ```text
//! e_ij     = LeakyReLU(a^T [W h_i || W h_j]),   j in N(i) union {i}
//! alpha_ij = softmax_j(e_ij)                     (masked to i's neighbors)
//! h_i'     = sigma( sum_j alpha_ij W h_j )
//! ```
//!
//! Implemented in the same dense `n x n` masked-tensor style as the rest of
//! this crate's Burn layers (see the module doc in `burn_layer.rs` for why
//! dense is the deliberate tradeoff at this graph scale, and what would
//! need to change -- a sparse CSR kernel with hand-written backward -- to
//! scale past a few thousand nodes).
//!
//! `a^T [Wh_i || Wh_j]` is factored as `a_src . Wh_i + a_dst . Wh_j`
//! (splitting the length-`2d` attention vector into two length-`d` halves)
//! so the `n x n` score matrix is built as a broadcast outer-sum of two
//! length-`n` vectors, rather than ever materializing an `n x n x 2d`
//! concatenated tensor.
//!
//! Deviations from the original paper, both for implementation-risk
//! reasons (this file could not be compiled/tested in the environment it
//! was written in -- see the crate README) and documented here so they are
//! visible rather than silently different from the citation:
//! - The output nonlinearity is ReLU (matching [`crate::burn_layer::GcnLayer`]
//!   and [`crate::burn_layer::NbscLayer`] elsewhere in this crate), not the
//!   paper's ELU.
//! - No numerically-stabilizing max-subtraction before the softmax
//!   exponential (relies on masked entries being additively driven to a
//!   large negative value, which underflows `exp` to exactly `0.0` in
//!   `f32` without needing the max-subtraction trick). If training
//!   destabilizes with large logits, add a `- max_dim(1)` step before `exp`.
//! - Multi-head concatenation only (no "average heads at the output
//!   layer" variant); configure `n_heads = 1` for an output layer.

use crate::graph::Graph;
use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{activation::relu, Distribution, Tensor};

const NEG_INF: f32 = -1.0e9;

/// Additive attention mask: `0.0` where node `j` is a neighbor of `i` (or
/// `i` itself -- GAT attends over `N(i) union {i}`), `NEG_INF` elsewhere.
/// Added to raw attention scores before softmax so disallowed entries
/// vanish after `exp`.
fn additive_attention_mask<B: Backend>(graph: &Graph, device: &B::Device) -> Tensor<B, 2> {
    let n = graph.n;
    let mut a = graph.dense_adjacency();
    for i in 0..n {
        a[i * n + i] = 1.0;
    }
    let mask: Vec<f32> = a.into_iter().map(|v| if v > 0.5 { 0.0 } else { NEG_INF }).collect();
    Tensor::<B, 1>::from_floats(mask.as_slice(), device).reshape([n, n])
}

/// `LeakyReLU(x) = relu(x) - slope * relu(-x)`, built from `relu` (already
/// proven to exist in this codebase via `burn_layer.rs`) rather than
/// assuming a `leaky_relu` free function is present in this Burn version.
fn leaky_relu<B: Backend, const D: usize>(x: Tensor<B, D>, slope: f64) -> Tensor<B, D> {
    relu(x.clone()) - relu(-x) * slope
}

#[derive(Config, Debug)]
pub struct GatLayerConfig {
    pub d_input: usize,
    pub d_output_per_head: usize,
    pub n_heads: usize,
    #[config(default = 0.2)]
    pub leaky_relu_slope: f64,
}

#[derive(Module, Debug)]
pub struct GatHead<B: Backend> {
    w: Linear<B>,
    a_src: Param<Tensor<B, 1>>,
    a_dst: Param<Tensor<B, 1>>,
}

#[derive(Module, Debug)]
pub struct GatLayer<B: Backend> {
    heads: Vec<GatHead<B>>,
    additive_mask: Tensor<B, 2>,
    leaky_relu_slope: f64,
}

impl GatLayerConfig {
    pub fn init<B: Backend>(&self, graph: &Graph, device: &B::Device) -> GatLayer<B> {
        let additive_mask = additive_attention_mask::<B>(graph, device);
        let heads: Vec<GatHead<B>> = (0..self.n_heads)
            .map(|_| {
                let w = LinearConfig::new(self.d_input, self.d_output_per_head)
                    .with_bias(false)
                    .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                    .init(device);
                let a_src = Param::from_tensor(Tensor::random(
                    [self.d_output_per_head],
                    Distribution::Normal(0.0, 0.1),
                    device,
                ));
                let a_dst = Param::from_tensor(Tensor::random(
                    [self.d_output_per_head],
                    Distribution::Normal(0.0, 0.1),
                    device,
                ));
                GatHead { w, a_src, a_dst }
            })
            .collect();
        GatLayer { heads, additive_mask, leaky_relu_slope: self.leaky_relu_slope }
    }
}

impl<B: Backend> GatLayer<B> {
    /// `x`: `[n, d_input]`. Returns `[n, n_heads * d_output_per_head]`
    /// (heads concatenated -- standard for hidden GAT layers; configure
    /// `n_heads = 1` for an output layer so the returned width equals
    /// `d_output_per_head` exactly).
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let n = x.dims()[0];
        let mut head_outputs = Vec::with_capacity(self.heads.len());

        for head in &self.heads {
            let wh = head.w.forward(x.clone()); // [n, d_out]

            // score_src[i] = a_src . Wh_i, score_dst[j] = a_dst . Wh_j
            let score_src = wh.clone().matmul(head.a_src.val().unsqueeze_dim::<2>(1)); // [n, 1]
            let score_dst = wh.clone().matmul(head.a_dst.val().unsqueeze_dim::<2>(1)); // [n, 1]

            // e[i,j] = score_src[i] + score_dst[j] via broadcast outer-sum:
            // [n,1] reshaped stays a column, [n,1] reshaped to [1,n] becomes
            // a row of the same n values in the same order (both are a
            // contiguous length-n sequence, so reshape just reinterprets
            // orientation, not order).
            let e = score_src.reshape([n, 1]) + score_dst.reshape([1, n]);
            let e = leaky_relu(e, self.leaky_relu_slope);

            let masked = e + self.additive_mask.clone();
            let exp = masked.exp();
            let sum_per_row = exp.clone().sum_dim(1); // [n, 1], keeps rank (reduced dim size 1)
            let alpha = exp / (sum_per_row + 1e-12);

            let out = alpha.matmul(wh); // [n, d_out]
            head_outputs.push(out);
        }

        let combined = Tensor::cat(head_outputs, 1);
        relu(combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::random_near_regular;
    use burn::backend::ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn gat_layer_forward_runs_and_shapes_match_multihead() {
        let device = Default::default();
        let g = random_near_regular(20, 4, 21);
        let cfg = GatLayerConfig::new(6, 8, 4); // 4 heads * 8 = 32-wide output
        let layer = cfg.init::<B>(&g, &device);
        let x = Tensor::<B, 2>::zeros([g.n, 6], &device) + 1.0;
        let out = layer.forward(x);
        assert_eq!(out.dims(), [g.n, 32]);
    }

    #[test]
    fn gat_single_head_output_dim_matches_config() {
        let device = Default::default();
        let g = random_near_regular(15, 4, 22);
        let cfg = GatLayerConfig::new(5, 7, 1);
        let layer = cfg.init::<B>(&g, &device);
        let x = Tensor::<B, 2>::zeros([g.n, 5], &device) + 1.0;
        let out = layer.forward(x);
        assert_eq!(out.dims(), [g.n, 7]);
    }
}
