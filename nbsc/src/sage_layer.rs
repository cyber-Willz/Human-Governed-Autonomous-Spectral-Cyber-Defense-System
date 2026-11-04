//! GraphSAGE layer (Hamilton, Ying & Leskovec, 2017), mean-aggregator
//! variant, baseline #3.
//!
//! Structural contrast with [`crate::burn_layer::GcnLayer`] worth stating
//! explicitly in the thesis methods chapter: GCN mixes a node's own
//! features into the *same* symmetric-normalized sum as its neighbors
//! (self modeled as an added self-loop, normalized the same way as any
//! other edge). GraphSAGE keeps the self-transform separate -- it
//! concatenates `h_i` with a neighbor aggregate `h_N(i)` before a single
//! shared linear projection:
//!
//! ```text
//! h_N(i) = mean_{j in N(i)} h_j
//! h_i'   = sigma( W [h_i || h_N(i)] )
//! ```
//!
//! so the model can learn to weight self-information independently of
//! neighbor information, rather than that weighting being fixed by degree
//! as in GCN.
//!
//! Scope note: this implements full-batch mean aggregation over *all*
//! neighbors, no neighbor sampling. The original paper's headline
//! algorithmic contribution is the sampling scheme for inductive learning
//! on graphs too large to fit in memory; that's out of scope at the graph
//! sizes used in this benchmark (Cora: 2708 nodes). What's compared here is
//! the aggregate-then-concatenate combine rule itself, which is the part
//! that's architecturally distinct from GCN regardless of sampling.

use crate::graph::Graph;
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{activation::relu, Tensor};

/// Row-stochastic (mean) neighbor-aggregation operator, `n x n`, with no
/// self-loop (GraphSAGE keeps self separate via concatenation, unlike
/// GCN's `A + I`).
fn mean_aggregation_tensor<B: Backend>(graph: &Graph, device: &B::Device) -> Tensor<B, 2> {
    let n = graph.n;
    let mut a = graph.dense_adjacency();
    for v in 0..n {
        let deg = graph.degree(v).max(1) as f64; // isolated nodes: row is all-zero, aggregate stays 0
        for u in 0..n {
            a[v * n + u] /= deg;
        }
    }
    let data: Vec<f32> = a.into_iter().map(|x| x as f32).collect();
    Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([n, n])
}

#[derive(Config, Debug)]
pub struct SageLayerConfig {
    pub d_input: usize,
    pub d_output: usize,
    #[config(default = true)]
    pub bias: bool,
}

#[derive(Module, Debug)]
pub struct SageLayer<B: Backend> {
    linear: Linear<B>, // input width = 2 * d_input (self || neighbor-mean, concatenated)
    mean_agg: Tensor<B, 2>,
}

impl SageLayerConfig {
    pub fn init<B: Backend>(&self, graph: &Graph, device: &B::Device) -> SageLayer<B> {
        let mean_agg = mean_aggregation_tensor::<B>(graph, device);
        let linear = LinearConfig::new(2 * self.d_input, self.d_output)
            .with_bias(self.bias)
            .init(device);
        SageLayer { linear, mean_agg }
    }
}

impl<B: Backend> SageLayer<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let neighbor_agg = self.mean_agg.clone().matmul(x.clone());
        let combined = Tensor::cat(vec![x, neighbor_agg], 1);
        relu(self.linear.forward(combined))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::random_near_regular;
    use burn::backend::ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn sage_layer_forward_runs_and_shapes_match() {
        let device = Default::default();
        let g = random_near_regular(20, 4, 11);
        let cfg = SageLayerConfig::new(6, 8);
        let layer = cfg.init::<B>(&g, &device);
        let x = Tensor::<B, 2>::zeros([g.n, 6], &device) + 1.0;
        let out = layer.forward(x);
        assert_eq!(out.dims(), [g.n, 8]);
    }

    #[test]
    fn mean_aggregation_rows_sum_to_one_for_non_isolated_nodes() {
        let device = Default::default();
        let g = random_near_regular(15, 4, 12);
        let agg = mean_aggregation_tensor::<B>(&g, &device);
        let ones = Tensor::<B, 2>::zeros([g.n, 1], &device) + 1.0;
        let row_sums = agg.matmul(ones);
        let data = row_sums.to_data().convert::<f32>().value;
        for &s in data.iter() {
            assert!((s - 1.0).abs() < 1e-4, "row sum {s} should be ~1.0 for a mean aggregator");
        }
    }
}
