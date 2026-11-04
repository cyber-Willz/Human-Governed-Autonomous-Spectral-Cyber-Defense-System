//! Real-data benchmark: NBSC vs. GCN vs. GAT vs. GraphSAGE on the Citeseer
//! citation network — a second real dataset, run after `benchmark_cora`,
//! specifically to check whether the GCN/GAT > RNBSC ranking observed on
//! Cora generalizes or is Cora-specific (Cora is known to be an unusually
//! homophilous, GCN-favorable graph; see `docs/results_cora_draft.md`).
//!
//! IMPORTANT -- read before quoting these numbers anywhere:
//! This uses the bundled `data/citeseer/{citeseer.content,citeseer.cites}`
//! files (3327 nodes, 3703-dim features, 6 classes -- see `dataset.rs` for
//! provenance and validation against published statistics) and the split
//! implemented in `dataset::stratified_split` (`Dataset::load_citeseer_default`),
//! same shape as the standard semi-supervised Planetoid protocol (20
//! labeled/class train, 500 val, 1000 test) but NOT the literature's exact
//! split -- see `dataset.rs` module docs. Do not present these numbers next
//! to published Citeseer leaderboard entries as if directly comparable.
//!
//! METHODOLOGY NOTE -- this run differs from the original Cora baseline:
//! `WEIGHT_DECAY` defaults to 5e-4 here (vs. 0.0 in the original
//! `benchmark_cora` run), since the Cora results already showed evidence of
//! overfitting (accuracy well below the literature's ~81.5%, no
//! regularization in that run at all) and there's no existing Citeseer
//! baseline to preserve reproducibility of. This means Citeseer numbers
//! from this file are not a clean apples-to-apples comparison against the
//! *original* Cora table without also rerunning Cora with the same
//! `WEIGHT_DECAY=5e-4` setting (in `benchmark_cora.rs`) -- do that if you
//! want a controlled two-dataset comparison, rather than comparing this
//! file's output directly to the already-recorded Cora numbers.
//!
//! Expect real wall-clock time: the Cora run showed GAT costing roughly
//! 10-15x more than GCN per seed (500-1700s vs. 30-135s, likely a
//! vectorization gap in the dense attention-score broadcast, not a
//! fundamental cost) -- Citeseer is a larger graph (3327 vs. 2708 nodes)
//! so expect this to run longer than the Cora benchmark, not shorter.
//!
//! Run with: `cargo run --release --example benchmark_citeseer --features burn`

use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{decay::WeightDecayConfig, AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use nbsc::burn_layer::{dirichlet_energy, GcnLayer, GcnLayerConfig, NbscLayer, NbscLayerConfig};
use nbsc::dataset::Dataset;
use nbsc::gat_layer::{GatLayer, GatLayerConfig};
use nbsc::graph::Graph;
use nbsc::sage_layer::{SageLayer, SageLayerConfig};

type Be = Autodiff<NdArray<f32>>;

const HIDDEN: usize = 16;
const K_TAPS: usize = 2;
const GAT_HEADS: usize = 4;
const EPOCHS: usize = 150;
const LR: f64 = 0.01;
/// L2 weight decay, applied to all four architectures identically (fair
/// comparison). Defaults to 5e-4 here (unlike `benchmark_cora.rs`, which
/// defaults to 0.0 to preserve its already-recorded baseline) -- there is
/// no prior Citeseer baseline to preserve, and the Cora results already
/// showed evidence of overfitting without any regularization, so this run
/// starts from that lesson rather than repeating the unregularized setup.
/// 5e-4 is the value from Kipf & Welling's original GCN paper. If you want
/// a controlled Cora-vs-Citeseer comparison, rerun `benchmark_cora` with
/// `WEIGHT_DECAY = 5e-4` too, rather than comparing this file's output
/// directly to the original (`WEIGHT_DECAY = 0.0`) Cora table.
const WEIGHT_DECAY: f64 = 5e-4;
const N_SEEDS: u64 = 5;
const SPLIT_SEED: u64 = 0; // fixed data split shared across all seeds/models; only param-init/training seed varies
/// Ablation switch: set to `true` and rerun to test whether LayerNorm on
/// NBSC's output stabilizes the depth-3 energy/variance blowup and closes
/// any of the accuracy gap. Default `false` reproduces the original run
/// exactly -- flip this, rerun, and diff against your saved depth 1/2/3
/// tables rather than overwriting them.
const NBSC_NORMALIZE: bool = false;

// --- shared helpers (deliberately duplicated from benchmark.rs rather
// than factored into a shared lib module, matching that file's existing
// style of keeping each example self-contained) -----------------------

fn int_tensor<B: Backend>(values: &[i64], device: &B::Device) -> Tensor<B, 1, Int> {
    let values_i32: Vec<i32> = values.iter().map(|&v| v as i32).collect();
    Tensor::<B, 1, Int>::from_ints(values_i32.as_slice(), device)
}

fn index_select_rows<B: Backend>(x: &Tensor<B, 2>, idx: &[usize], device: &B::Device) -> Tensor<B, 2> {
    let idx_i64: Vec<i64> = idx.iter().map(|&i| i as i64).collect();
    let idx_t = int_tensor::<B>(&idx_i64, device);
    x.clone().select(0, idx_t)
}

fn accuracy<B: Backend>(logits: &Tensor<B, 2>, labels: &[usize], idx: &[usize]) -> f32 {
    let preds = logits.clone().argmax(1).squeeze::<1>(1);
    let preds_data = preds.into_data().convert::<i64>().value;
    let correct = idx.iter().filter(|&&i| preds_data[i] as usize == labels[i]).count();
    correct as f32 / idx.len().max(1) as f32
}

fn mean_std(values: &[f32]) -> (f32, f32) {
    let n = values.len().max(1) as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}

/// `None` when `WEIGHT_DECAY == 0.0`, so the default configuration takes
/// the exact same code path as before weight decay was added (rather than
/// relying on `WeightDecayConfig::new(0.0)` being a no-op, which is
/// probably true but unnecessary to depend on).
fn weight_decay_config() -> Option<WeightDecayConfig> {
    if WEIGHT_DECAY > 0.0 {
        Some(WeightDecayConfig::new(WEIGHT_DECAY))
    } else {
        None
    }
}

fn features_tensor<B: Backend>(ds: &Dataset, device: &B::Device) -> Tensor<B, 2> {
    let data: Vec<f32> = ds.features.data.iter().map(|&v| v as f32).collect();
    Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([ds.graph.n, ds.features.f])
}

struct EvalResult {
    val_acc: f32,
    test_acc: f32,
    final_energy: f32,
}

// --- model definitions (one net type per architecture, mirroring
// benchmark.rs's NbscNet/GcnNet pattern rather than a shared generic
// trait -- see the note in the accompanying chat response for why the
// generic-trait version was deliberately not used here) ---------------

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

#[derive(Module, Debug)]
struct GatNet<B: Backend> {
    layers: Vec<GatLayer<B>>,
    head: Linear<B>,
}
impl<B: Backend> GatNet<B> {
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
struct SageNet<B: Backend> {
    layers: Vec<SageLayer<B>>,
    head: Linear<B>,
}
impl<B: Backend> SageNet<B> {
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

// --- builders -----------------------------------------------------------

fn build_nbsc_net(graph: &Graph, depth: usize, d_in: usize, n_classes: usize, device: &<Be as Backend>::Device) -> NbscNet<Be> {
    let mut layers = Vec::with_capacity(depth);
    for i in 0..depth {
        let d_i = if i == 0 { d_in } else { HIDDEN };
        layers.push(NbscLayerConfig::new(d_i, HIDDEN, K_TAPS).with_normalize(NBSC_NORMALIZE).init(graph, device));
    }
    let head = LinearConfig::new(HIDDEN, n_classes).init(device);
    NbscNet { layers, head }
}

fn build_gcn_net(graph: &Graph, depth: usize, d_in: usize, n_classes: usize, device: &<Be as Backend>::Device) -> GcnNet<Be> {
    let mut layers = Vec::with_capacity(depth);
    for i in 0..depth {
        let d_i = if i == 0 { d_in } else { HIDDEN };
        layers.push(GcnLayerConfig::new(d_i, HIDDEN).init(graph, device));
    }
    let head = LinearConfig::new(HIDDEN, n_classes).init(device);
    GcnNet { layers, head }
}

fn build_gat_net(graph: &Graph, depth: usize, d_in: usize, n_classes: usize, device: &<Be as Backend>::Device) -> GatNet<Be> {
    let per_head = (HIDDEN / GAT_HEADS).max(1);
    let mut layers = Vec::with_capacity(depth);
    for i in 0..depth {
        let d_i = if i == 0 { d_in } else { per_head * GAT_HEADS };
        layers.push(GatLayerConfig::new(d_i, per_head, GAT_HEADS).init(graph, device));
    }
    let head = LinearConfig::new(per_head * GAT_HEADS, n_classes).init(device);
    GatNet { layers, head }
}

fn build_sage_net(graph: &Graph, depth: usize, d_in: usize, n_classes: usize, device: &<Be as Backend>::Device) -> SageNet<Be> {
    let mut layers = Vec::with_capacity(depth);
    for i in 0..depth {
        let d_i = if i == 0 { d_in } else { HIDDEN };
        layers.push(SageLayerConfig::new(d_i, HIDDEN).init(graph, device));
    }
    let head = LinearConfig::new(HIDDEN, n_classes).init(device);
    SageNet { layers, head }
}

// --- training loops (one per architecture, same shape; see note above
// on why this isn't collapsed into one generic function) ---------------

#[allow(clippy::too_many_arguments)]
fn train_nbsc(
    graph: &Graph, depth: usize, x: Tensor<Be, 2>, labels: &[usize], n_classes: usize,
    train_idx: &[usize], val_idx: &[usize], test_idx: &[usize], device: &<Be as Backend>::Device, seed: u64,
) -> EvalResult {
    Be::seed(seed);
    let d_in = x.dims()[1];
    let mut net = build_nbsc_net(graph, depth, d_in, n_classes, device);
    let mut optim = AdamConfig::new().with_weight_decay(weight_decay_config()).init();
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
    let (_a, logits) = net.forward_all(x);
    EvalResult {
        val_acc: accuracy(&logits, labels, val_idx),
        test_acc: accuracy(&logits, labels, test_idx),
        final_energy: last_energy,
    }
}

#[allow(clippy::too_many_arguments)]
fn train_gcn(
    graph: &Graph, depth: usize, x: Tensor<Be, 2>, labels: &[usize], n_classes: usize,
    train_idx: &[usize], val_idx: &[usize], test_idx: &[usize], device: &<Be as Backend>::Device, seed: u64,
) -> EvalResult {
    Be::seed(seed);
    let d_in = x.dims()[1];
    let mut net = build_gcn_net(graph, depth, d_in, n_classes, device);
    let mut optim = AdamConfig::new().with_weight_decay(weight_decay_config()).init();
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
    let (_a, logits) = net.forward_all(x);
    EvalResult {
        val_acc: accuracy(&logits, labels, val_idx),
        test_acc: accuracy(&logits, labels, test_idx),
        final_energy: last_energy,
    }
}

#[allow(clippy::too_many_arguments)]
fn train_gat(
    graph: &Graph, depth: usize, x: Tensor<Be, 2>, labels: &[usize], n_classes: usize,
    train_idx: &[usize], val_idx: &[usize], test_idx: &[usize], device: &<Be as Backend>::Device, seed: u64,
) -> EvalResult {
    Be::seed(seed);
    let d_in = x.dims()[1];
    let mut net = build_gat_net(graph, depth, d_in, n_classes, device);
    let mut optim = AdamConfig::new().with_weight_decay(weight_decay_config()).init();
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
    let (_a, logits) = net.forward_all(x);
    EvalResult {
        val_acc: accuracy(&logits, labels, val_idx),
        test_acc: accuracy(&logits, labels, test_idx),
        final_energy: last_energy,
    }
}

#[allow(clippy::too_many_arguments)]
fn train_sage(
    graph: &Graph, depth: usize, x: Tensor<Be, 2>, labels: &[usize], n_classes: usize,
    train_idx: &[usize], val_idx: &[usize], test_idx: &[usize], device: &<Be as Backend>::Device, seed: u64,
) -> EvalResult {
    Be::seed(seed);
    let d_in = x.dims()[1];
    let mut net = build_sage_net(graph, depth, d_in, n_classes, device);
    let mut optim = AdamConfig::new().with_weight_decay(weight_decay_config()).init();
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
    let (_a, logits) = net.forward_all(x);
    EvalResult {
        val_acc: accuracy(&logits, labels, val_idx),
        test_acc: accuracy(&logits, labels, test_idx),
        final_energy: last_energy,
    }
}

fn report(name: &str, depth: usize, results: &[EvalResult]) {
    let test_accs: Vec<f32> = results.iter().map(|r| r.test_acc).collect();
    let val_accs: Vec<f32> = results.iter().map(|r| r.val_acc).collect();
    let energies: Vec<f32> = results.iter().map(|r| r.final_energy).collect();
    let (test_mean, test_std) = mean_std(&test_accs);
    let (val_mean, val_std) = mean_std(&val_accs);
    let (energy_mean, energy_std) = mean_std(&energies);
    println!(
        "{name:<10} | depth {depth:<2} | val {val_mean:.3}+-{val_std:.3} | test {test_mean:.3}+-{test_std:.3} | energy {energy_mean:.4}+-{energy_std:.4} | n_seeds={}",
        results.len()
    );
}

fn main() {
    let ds = Dataset::load_citeseer_default(SPLIT_SEED).expect(
        "failed to load Citeseer -- check that nbsc/data/citeseer/{citeseer.content,citeseer.cites} exist",
    );
    println!(
        "Loaded Citeseer: n={}, m={}, classes={}, train={}, val={}, test={}",
        ds.graph.n,
        ds.graph.m(),
        ds.num_classes(),
        ds.train_indices().len(),
        ds.val_indices().len(),
        ds.test_indices().len(),
    );
    println!(
        "NOTE: this is a stratified split with the same *shape* as the standard Planetoid protocol,\n\
         not the bit-identical published split (see dataset.rs docs). Compare models only within this run."
    );

    let device: <Be as Backend>::Device = Default::default();
    let x = features_tensor::<Be>(&ds, &device);
    let train_idx = ds.train_indices();
    let val_idx = ds.val_indices();
    let test_idx = ds.test_indices();
    let n_classes = ds.num_classes();

    let depths = [1usize, 2, 3];
    eprintln!(
        "\nRunning {} configs ({} depths x {} seeds x 4 architectures). The first layer of every \
         config does a dense 3327x3327x3703 propagation on the raw input width -- Citeseer is both \
         a larger graph and a wider feature vector than Cora, and Cora's own GAT runs already took \
         500-1700s/seed -- so expect this to take noticeably longer than the Cora benchmark, not \
         instant output. Progress prints below as each (depth, seed, model) finishes.\n",
        depths.len() * N_SEEDS as usize * 4,
        depths.len(),
        N_SEEDS,
    );

    for &depth in &depths {
        let mut nbsc_results = Vec::with_capacity(N_SEEDS as usize);
        let mut gcn_results = Vec::with_capacity(N_SEEDS as usize);
        let mut gat_results = Vec::with_capacity(N_SEEDS as usize);
        let mut sage_results = Vec::with_capacity(N_SEEDS as usize);

        for seed in 0..N_SEEDS {
            let t = std::time::Instant::now();
            eprint!("  depth={depth} seed={seed}/{N_SEEDS} NBSC...  ");
            nbsc_results.push(train_nbsc(&ds.graph, depth, x.clone(), &ds.labels, n_classes, &train_idx, &val_idx, &test_idx, &device, seed));
            eprintln!("done ({:.1}s)", t.elapsed().as_secs_f32());

            let t = std::time::Instant::now();
            eprint!("  depth={depth} seed={seed}/{N_SEEDS} GCN...   ");
            gcn_results.push(train_gcn(&ds.graph, depth, x.clone(), &ds.labels, n_classes, &train_idx, &val_idx, &test_idx, &device, seed));
            eprintln!("done ({:.1}s)", t.elapsed().as_secs_f32());

            let t = std::time::Instant::now();
            eprint!("  depth={depth} seed={seed}/{N_SEEDS} GAT...   ");
            gat_results.push(train_gat(&ds.graph, depth, x.clone(), &ds.labels, n_classes, &train_idx, &val_idx, &test_idx, &device, seed));
            eprintln!("done ({:.1}s)", t.elapsed().as_secs_f32());

            let t = std::time::Instant::now();
            eprint!("  depth={depth} seed={seed}/{N_SEEDS} SAGE...  ");
            sage_results.push(train_sage(&ds.graph, depth, x.clone(), &ds.labels, n_classes, &train_idx, &val_idx, &test_idx, &device, seed));
            eprintln!("done ({:.1}s)", t.elapsed().as_secs_f32());
        }

        println!("\n=== depth {depth} ===");
        report("NBSC", depth, &nbsc_results);
        report("GCN", depth, &gcn_results);
        report("GAT", depth, &gat_results);
        report("GraphSAGE", depth, &sage_results);
    }
}
