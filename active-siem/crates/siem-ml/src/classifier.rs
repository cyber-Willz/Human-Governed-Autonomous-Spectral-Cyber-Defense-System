//! siem-ml::classifier - categorical attack-vector isolation.
//!
//! Motivated by analyzing `spec_engine`'s approach (in the uploaded
//! `spectral_homology_tar.gz`): it classifies CIC-IDS2018 labels into MITRE
//! ATT&CK tactics via `classify_mitre_from_label`, giving each attack family
//! a distinct category rather than a single "attack/not-attack" bit. That's
//! the right shape for this crate too - the previous `autoencoder` module
//! only ever answers "does this look like the one benign pattern I trained
//! on" (binary, and per the false-positive test in the repo, badly
//! miscalibrated). A *categorical* system needs to answer "which of several
//! known shapes does this match, if any" - a genuinely different problem,
//! solved here with a small supervised softmax classifier instead of an
//! unsupervised reconstruction threshold.
//!
//! # Why supervised here, when `autoencoder` is deliberately unsupervised
//! The autoencoder's whole point is catching attack shapes nobody has
//! labeled yet - it must only ever see benign data, or it stops meaning
//! anything (this was explicitly flagged and declined earlier: training an
//! anomaly detector on attack-only data flips what "anomalous" means). A
//! *categorical* system asking "which known attack family is this" is a
//! different, legitimate task - it needs labeled examples of every category
//! it claims to distinguish, benign included. Use both: this classifier for
//! "which of these known categories, if any" and `autoencoder` as an
//! open-set fallback for "doesn't match anything I have labels for."
//!
//! # Categories
//! A condensed version of `spec_engine`'s MITRE vocabulary - narrow enough
//! that each category corresponds to a genuinely distinguishable flow shape
//! at this crate's 8-feature resolution (see `FlowFeatures`), rather than
//! the full ATT&CK technique list, most of which isn't distinguishable from
//! flow-level statistics alone (e.g. no flow-level feature set separates
//! T1055 process injection from T1059 command execution - that needs host
//! telemetry, which is `siem-rules`' job, not this classifier's).

use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{ElementConversion, Int, Tensor};

use crate::FlowFeatures;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    /// Ordinary traffic - the class every other class must be isolated from.
    Benign,
    /// Reconnaissance / scanning (MITRE TA0043) - many short, low-byte
    /// connections to different ports/hosts in a short window. Complements
    /// `siem-rules`' `port-scan-vertical` threshold rule with a shape-based
    /// signal that doesn't depend on a fixed count/window.
    Reconnaissance,
    /// Credential access via brute force (MITRE TA0006 / T1110) - many
    /// short, regular connections to one authentication endpoint.
    BruteForce,
    /// Low-and-slow foothold / lateral movement (MITRE TA0008) - long
    /// duration, small and asymmetric (upload-skewed) byte counts, low
    /// packet rate. This is `spec_engine`'s "Infiltration" label and the
    /// same class the autoencoder in this crate was built to catch.
    Infiltration,
    /// Command-and-control beaconing / botnet activity (MITRE TA0011) -
    /// long duration, very regular (low-variance) small-packet exchanges,
    /// often near-symmetric fwd/bwd counts (periodic check-ins).
    CommandAndControl,
    /// DoS/DDoS flood (MITRE TA0040 / T1498-T1499) - extreme packet counts
    /// or byte volume relative to duration; the opposite shape from
    /// Infiltration (loud and fast instead of quiet and slow).
    DenialOfService,
}

pub const NUM_CLASSES: usize = 6;

impl Category {
    pub fn all() -> [Category; NUM_CLASSES] {
        [
            Category::Benign,
            Category::Reconnaissance,
            Category::BruteForce,
            Category::Infiltration,
            Category::CommandAndControl,
            Category::DenialOfService,
        ]
    }

    pub fn index(&self) -> usize {
        match self {
            Category::Benign => 0,
            Category::Reconnaissance => 1,
            Category::BruteForce => 2,
            Category::Infiltration => 3,
            Category::CommandAndControl => 4,
            Category::DenialOfService => 5,
        }
    }

    pub fn from_index(i: usize) -> Option<Category> {
        Category::all().into_iter().find(|c| c.index() == i)
    }

    /// Loosely mirrors `spec_engine::classify_mitre_from_label`'s keyword
    /// matching, condensed to this crate's six categories. Used to turn
    /// CIC-IDS2018-style string labels (as in the datasets provided in this
    /// conversation) into training targets without hand-labeling each row.
    pub fn from_cic_label(label: &str) -> Option<Category> {
        let lower = label.to_lowercase();
        if lower == "benign" {
            return Some(Category::Benign);
        }
        let pairs: &[(&str, Category)] = &[
            ("infilter", Category::Infiltration),
            ("bot", Category::CommandAndControl),
            ("ddos", Category::DenialOfService),
            ("dos", Category::DenialOfService),
            ("hulk", Category::DenialOfService),
            ("goldeneye", Category::DenialOfService),
            ("slowloris", Category::DenialOfService),
            ("slowhttp", Category::DenialOfService),
            ("loic", Category::DenialOfService),
            ("brute", Category::BruteForce),
            ("ftpbrute", Category::BruteForce),
            ("scan", Category::Reconnaissance),
            ("recon", Category::Reconnaissance),
        ];
        for (kw, cat) in pairs {
            if lower.contains(kw) {
                return Some(*cat);
            }
        }
        None
    }

    pub fn name(&self) -> &'static str {
        match self {
            Category::Benign => "Benign",
            Category::Reconnaissance => "Reconnaissance",
            Category::BruteForce => "BruteForce",
            Category::Infiltration => "Infiltration",
            Category::CommandAndControl => "CommandAndControl",
            Category::DenialOfService => "DenialOfService",
        }
    }
}

#[derive(Module, Debug)]
pub struct Classifier<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    relu: Relu,
}

#[derive(Debug, Clone)]
pub struct ClassifierConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub num_classes: usize,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            input_dim: crate::FEATURE_DIM,
            hidden_dim: 16,
            num_classes: NUM_CLASSES,
        }
    }
}

impl ClassifierConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Classifier<B> {
        Classifier {
            fc1: LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            fc2: LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            fc3: LinearConfig::new(self.hidden_dim, self.num_classes).init(device),
            relu: Relu::new(),
        }
    }
}

impl<B: Backend> Classifier<B> {
    /// Raw logits, shape [batch, num_classes]. Use `predict` for a decision.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.relu.forward(self.fc1.forward(input));
        let x = self.relu.forward(self.fc2.forward(x));
        self.fc3.forward(x)
    }
}

/// One labeled training example.
pub struct LabeledFlow {
    pub features: FlowFeatures,
    pub category: Category,
}

fn batch_tensor<B: Backend>(
    device: &B::Device,
    samples: &[LabeledFlow],
) -> (Tensor<B, 2>, Tensor<B, 1, Int>) {
    let n = samples.len();
    let dim = crate::FEATURE_DIM;
    let data: Vec<f32> = samples.iter().flat_map(|s| s.features.0).collect();
    let inputs = Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([n, dim]);
    let labels: Vec<i32> = samples.iter().map(|s| s.category.index() as i32).collect();
    let targets = Tensor::<B, 1, Int>::from_ints(labels.as_slice(), device);
    (inputs, targets)
}

/// Trains the classifier on labeled flows spanning every category it claims
/// to distinguish - unlike `autoencoder::train`, which must only ever see
/// benign data, this needs (and requires) attack examples too.
pub fn train<B: AutodiffBackend>(
    device: &B::Device,
    config: &ClassifierConfig,
    samples: &[LabeledFlow],
    epochs: usize,
    lr: f64,
) -> Classifier<B> {
    let mut model = config.init::<B>(device);
    let mut optim = AdamConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init::<B>(device);
    let (inputs, targets) = batch_tensor::<B>(device, samples);

    for _ in 0..epochs {
        let logits = model.forward(inputs.clone());
        let loss = loss_fn.forward(logits, targets.clone());
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);
    }
    model
}

/// Predicted category plus the softmax confidence for that category, so
/// callers can apply their own confidence floor (e.g. fall back to the
/// autoencoder's anomaly score, or to `siem-rules`, below some threshold)
/// instead of trusting every prediction unconditionally.
pub struct Prediction {
    pub category: Category,
    pub confidence: f32,
}

pub fn predict<B: Backend>(
    model: &Classifier<B>,
    device: &B::Device,
    features: FlowFeatures,
) -> Prediction {
    let input = Tensor::<B, 1>::from_floats(features.0.as_slice(), device)
        .reshape([1, crate::FEATURE_DIM]);
    let logits = model.forward(input);
    let probs = burn::tensor::activation::softmax(logits, 1);
    let probs_data: Vec<f32> = probs
        .clone()
        .into_data()
        .convert::<f32>()
        .value
        .into_iter()
        .collect();
    let (best_idx, best_prob) = probs_data
        .iter()
        .enumerate()
        .fold((0usize, f32::MIN), |acc, (i, &p)| if p > acc.1 { (i, p) } else { acc });
    let _ = probs; // keep tensor alive through the data conversion above
    Prediction {
        category: Category::from_index(best_idx).unwrap_or(Category::Benign),
        confidence: best_prob.elem(),
    }
}
