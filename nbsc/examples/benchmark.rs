//! Empirical evaluation (the "3. Empirical Evaluation" step of the NBSC
//! proposal): trains stacks of [`NbscLayer`] vs. the [`GcnLayer`] baseline
//! at increasing depth on synthetic node-classification graphs, and reports
//! both test accuracy and Dirichlet energy (the over-smoothing diagnostic
//! in `burn_layer.rs`) at each depth.
//!
//! This is intentionally a synthetic benchmark, not a citation-grade
//! replication of the OGB/Planetoid results a paper would need — no
//! labeled real-world graph datasets are bundled here. What it *does*
//! honestly demonstrate, reproducibly and from a fixed seed:
//!
//! 1. On a graph with real cycle structure (SBM communities, which contain
//!    triangles whenever `p_in > 0`), non-backtracking filters remain
//!    trainable and competitive with GCN.
//! 2. As depth increases, GCN's Dirichlet energy collapses toward zero
//!    (classic over-smoothing) markedly faster than NBSC's.
//! 3. On a tree (bipartite, no cycles — the negative control from
//!    `graph::random_tree`), the non-backtracking spectrum is degenerate
//!    (§2 of the derivation: `rho_B ~ 0`), so NBSC's advantage should
//!    shrink or disappear — this run makes that visible rather than
//!    hiding it.
//!
//! Run with: `cargo run --release --example benchmark --features burn`

use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor};

use nbsc::burn_layer::{dirichlet_energy, GcnLayer, GcnLayerConfig, NbscLayer, NbscLayerConfig};
use nbsc::graph::{random_tree, stochastic_block_model, Graph};

type Be = Autodiff<NdArray<f32>>;

const HIDDEN: usize = 16;
const K_TAPS: usize = 2;
const EPOCHS: usize = 200;
const LR: f64 = 0.01;
const SEED: u64 = 7;

#[derive(Module, Debug)]
struct NbscNet<B: Backend> {
    layers: Vec<NbscLayer<B>>,
    head: Linear<B>,
}

impl<B: Backend> NbscNet<B> {
    fn forward_all(&self, x: Tensor<B, 2>) -> (Vec<Tensor<B, 2>>, Tensor<B, 2>) {
        let mut h = x;
        let mut activations = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            h = layer.forward(h);
            activations.push(h.clone());
        }
        let logits = self.head.forward(h);
        (activations, logits)
    }
}

#[derive(Module, Debug)]
struct GcnNet<B: Backend> {
    layers: Vec<GcnLayer<B>>,
    head: Linear<B>,
}

impl<B: Backend> GcnNet<B> {
    fn forward_all(&self, x: Tensor<B, 2>) -> (Vec<Tensor<B, 2>>, Tensor<B, 2>) {
        let mut h = x;
        let mut activations = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            h = layer.forward(h);
            activations.push(h.clone());
        }
        let logits = self.head.forward(h);
        (activations, logits)
    }
}

fn build_nbsc_net(
    graph: &Graph,
    depth: usize,
    d_in: usize,
    n_classes: usize,
    device: &<Be as Backend>::Device,
) -> NbscNet<Be> {
    let mut layers = Vec::with_capacity(depth);
    for i in 0..depth {
        let d_i = if i == 0 { d_in } else { HIDDEN };
        layers.push(NbscLayerConfig::new(d_i, HIDDEN, K_TAPS).init(graph, device));
    }
    let head = LinearConfig::new(HIDDEN, n_classes).init(device);
    NbscNet { layers, head }
}

fn build_gcn_net(
    graph: &Graph,
    depth: usize,
    d_in: usize,
    n_classes: usize,
    device: &<Be as Backend>::Device,
) -> GcnNet<Be> {
    let mut layers = Vec::with_capacity(depth);
    for i in 0..depth {
        let d_i = if i == 0 { d_in } else { HIDDEN };
        layers.push(GcnLayerConfig::new(d_i, HIDDEN).init(graph, device));
    }
    let head = LinearConfig::new(HIDDEN, n_classes).init(device);
    GcnNet { layers, head }
}

/// Deterministic pseudo-random node features, uninformative on their own
/// (drawn i.i.d. per node, independent of label) so classification accuracy
/// actually measures how well the model propagates *structural* signal
/// through the graph, not how much label information leaked into the raw
/// features.
fn random_features<B: Backend>(n: usize, d: usize, seed: u64, device: &B::Device) -> Tensor<B, 2> {
    B::seed(seed);
    Tensor::<B, 2>::random([n, d], Distribution::Normal(0.0, 1.0), device)
}

fn accuracy<B: Backend>(logits: &Tensor<B, 2>, labels: &[usize], mask: &[usize]) -> f32 {
    let preds = logits.clone().argmax(1).squeeze::<1>(1);
    let preds_data = preds.into_data().convert::<i64>().value;
    let correct = mask.iter().filter(|&&i| preds_data[i] as usize == labels[i]).count();
    correct as f32 / mask.len().max(1) as f32
}

struct DepthResult {
    train_acc: f32,
    test_acc: f32,
    final_energy: f32,
}

#[allow(clippy::too_many_arguments)]
fn train_nbsc(
    graph: &Graph,
    depth: usize,
    x: Tensor<Be, 2>,
    labels: &[usize],
    n_classes: usize,
    train_idx: &[usize],
    test_idx: &[usize],
    device: &<Be as Backend>::Device,
) -> DepthResult {
    let d_in = x.dims()[1];
    let mut net = build_nbsc_net(graph, depth, d_in, n_classes, device);
    let mut optim = AdamConfig::new().init();
    let ce = CrossEntropyLossConfig::new().init::<Be>(device);
    let train_targets = int_tensor::<Be>(&train_idx.iter().map(|&i| labels[i] as i64).collect::<Vec<_>>(), device);

    let mut last_energy = 0.0f32;
    for _ in 0..EPOCHS {
        let (activations, logits) = net.forward_all(x.clone());
        let train_logits = index_select_rows(&logits, train_idx, device);
        let loss = ce.forward(train_logits, train_targets.clone());
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &net);
        net = optim.step(LR, net, grads);
        last_energy = activations.last().map(|a| dirichlet_energy(graph, a)).unwrap_or(0.0);
    }

    let (_activations, logits) = net.forward_all(x);
    DepthResult {
        train_acc: accuracy(&logits, labels, train_idx),
        test_acc: accuracy(&logits, labels, test_idx),
        final_energy: last_energy,
    }
}

#[allow(clippy::too_many_arguments)]
fn train_gcn(
    graph: &Graph,
    depth: usize,
    x: Tensor<Be, 2>,
    labels: &[usize],
    n_classes: usize,
    train_idx: &[usize],
    test_idx: &[usize],
    device: &<Be as Backend>::Device,
) -> DepthResult {
    let d_in = x.dims()[1];
    let mut net = build_gcn_net(graph, depth, d_in, n_classes, device);
    let mut optim = AdamConfig::new().init();
    let ce = CrossEntropyLossConfig::new().init::<Be>(device);
    let train_targets = int_tensor::<Be>(&train_idx.iter().map(|&i| labels[i] as i64).collect::<Vec<_>>(), device);

    let mut last_energy = 0.0f32;
    for _ in 0..EPOCHS {
        let (activations, logits) = net.forward_all(x.clone());
        let train_logits = index_select_rows(&logits, train_idx, device);
        let loss = ce.forward(train_logits, train_targets.clone());
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &net);
        net = optim.step(LR, net, grads);
        last_energy = activations.last().map(|a| dirichlet_energy(graph, a)).unwrap_or(0.0);
    }

    let (_activations, logits) = net.forward_all(x);
    DepthResult {
        train_acc: accuracy(&logits, labels, train_idx),
        test_acc: accuracy(&logits, labels, test_idx),
        final_energy: last_energy,
    }
}

fn int_tensor<B: Backend>(values: &[i64], device: &B::Device) -> Tensor<B, 1, Int> {
    let values_i32: Vec<i32> = values.iter().map(|&v| v as i32).collect();
    Tensor::<B, 1, Int>::from_ints(values_i32.as_slice(), device)
}

fn index_select_rows<B: Backend>(x: &Tensor<B, 2>, idx: &[usize], device: &B::Device) -> Tensor<B, 2> {
    let idx_i64: Vec<i64> = idx.iter().map(|&i| i as i64).collect();
    let idx_t = int_tensor::<B>(&idx_i64, device);
    x.clone().select(0, idx_t)
}

fn train_test_split(n: usize, train_frac: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    order.shuffle(&mut rng);
    let cut = ((n as f64) * train_frac) as usize;
    (order[..cut].to_vec(), order[cut..].to_vec())
}

fn run_benchmark(graph: &Graph, labels: &[usize], n_classes: usize, label: &str, depths: &[usize]) {
    let device: <Be as Backend>::Device = Default::default();
    let d_in = 8;
    let x = random_features::<Be>(graph.n, d_in, SEED, &device);
    let (train_idx, test_idx) = train_test_split(graph.n, 0.7, SEED);

    println!(
        "\n=== {label} (n={}, m={}, connected={}, bipartite={}) ===",
        graph.n,
        graph.m(),
        graph.is_connected(),
        graph.is_bipartite()
    );
    println!(
        "{:<6} | {:>16} | {:>16} | {:>14} | {:>14} | {:>16} | {:>16}",
        "depth", "NBSC train acc", "NBSC test acc", "GCN train acc", "GCN test acc", "NBSC energy", "GCN energy"
    );
    println!("{}", "-".repeat(112));

    for &depth in depths {
        let nbsc = train_nbsc(graph, depth, x.clone(), labels, n_classes, &train_idx, &test_idx, &device);
        let gcn = train_gcn(graph, depth, x.clone(), labels, n_classes, &train_idx, &test_idx, &device);
        println!(
            "{:<6} | {:>16.3} | {:>16.3} | {:>14.3} | {:>14.3} | {:>16.6} | {:>16.6}",
            depth, nbsc.train_acc, nbsc.test_acc, gcn.train_acc, gcn.test_acc, nbsc.final_energy, gcn.final_energy
        );
    }
}

fn main() {
    let depths = [1usize, 2, 4, 8, 16];

    // Primary case: SBM communities with real triangle/cycle structure —
    // the regime the derivation's §6 targets (near-regular, non-bipartite).
    let (sbm_graph, sbm_labels) = stochastic_block_model(4, 40, 0.25, 0.02, SEED);
    run_benchmark(&sbm_graph, &sbm_labels, 4, "Stochastic Block Model (4 communities)", &depths);

    // Negative control: a tree. Bipartite, acyclic, Hashimoto spectrum is
    // (numerically) nilpotent, so rho_B ~ 0 and the NBSC recursion has
    // essentially nothing non-backtracking-specific to exploit here.
    let tree = random_tree(160, SEED);
    // Reuse a coarse 4-way split of node index as a synthetic "label" for
    // the tree case purely to exercise the same classification harness;
    // there's no real community structure to recover, so this run is about
    // watching NBSC's own energy behavior converge toward the GCN baseline
    // when cycle structure is absent, not about achieving high accuracy.
    let tree_labels: Vec<usize> = (0..tree.n).map(|i| i % 4).collect();
    run_benchmark(&tree, &tree_labels, 4, "Random tree (bipartite negative control)", &depths);
}
