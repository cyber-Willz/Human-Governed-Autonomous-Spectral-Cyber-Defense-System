//! siem-ml: unsupervised anomaly detection for the "quiet" infiltration cases
//! that signature/threshold rules in siem-rules can't see.
//!
//! Why an autoencoder, and why this is the piece Wazuh/OSSEC/Security Onion
//! genuinely lack: all three are fundamentally signature + threshold engines
//! (OSSEC/Wazuh decoders+rules, Security Onion's Suricata/Sigma). They are
//! excellent at noisy attacks (brute force, scans, known exploit signatures)
//! but structurally blind to *low-and-slow infiltration*: a foothold that
//! trickles small amounts of data out over long, evenly-spaced connections
//! designed specifically to stay under every static threshold. That gap is
//! exactly why CIC-IDS2018's "Infiltration" class is notoriously the hardest
//! to detect with rules (see `spec-engine`'s own unresolved "undetected
//! Infiltration class" finding) - it needs a learned model of "normal"
//! traffic shape, not a bigger threshold.
//!
//! Approach: train a small dense autoencoder on benign `FlowFeatures` only.
//! At inference time, reconstruction error (MSE) above a threshold flags the
//! flow as anomalous. This mirrors the standard unsupervised approach used in
//! the network-anomaly-detection literature and sidesteps the need for
//! labeled attack data, which is the actual bottleneck in practice.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::ElementConversion;
use burn::tensor::Tensor;
use siem_core::EventKind;

pub mod classifier;

pub const FEATURE_DIM: usize = 8;

/// Numeric feature vector derived from a `siem_core::EventKind::Flow`.
/// This is the same feature family CIC-IDS2018-style flow classifiers use,
/// kept small enough to stay useful with limited training data.
#[derive(Debug, Clone, Copy)]
pub struct FlowFeatures(pub [f32; FEATURE_DIM]);

impl FlowFeatures {
    pub fn from_event(kind: &EventKind) -> Option<Self> {
        let EventKind::Flow {
            src_port,
            dst_port,
            proto,
            duration_ms,
            bytes_src_to_dst,
            bytes_dst_to_src,
            packets,
            ..
        } = kind
        else {
            return None;
        };
        let dur = (*duration_ms as f32).max(1.0);
        let total_bytes = (*bytes_src_to_dst + *bytes_dst_to_src) as f32;
        let f = [
            (*src_port as f32) / 65535.0,
            (*dst_port as f32) / 65535.0,
            (*proto as f32) / 255.0,
            (dur / 60_000.0).min(10.0), // minutes, capped so long beacons don't blow up scale
            (*bytes_src_to_dst as f32).ln_1p() / 20.0,
            (*bytes_dst_to_src as f32).ln_1p() / 20.0,
            (*packets as f32).ln_1p() / 15.0,
            if total_bytes > 0.0 {
                (*bytes_src_to_dst as f32) / total_bytes // upload ratio: infiltration exfil skews high
            } else {
                0.0
            },
        ];
        Some(FlowFeatures(f))
    }
}

#[derive(Module, Debug)]
pub struct Autoencoder<B: Backend> {
    enc1: Linear<B>,
    enc2: Linear<B>,
    dec1: Linear<B>,
    dec2: Linear<B>,
    relu: Relu,
}

#[derive(Debug, Clone)]
pub struct AutoencoderConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub latent_dim: usize,
}

impl Default for AutoencoderConfig {
    fn default() -> Self {
        Self {
            input_dim: FEATURE_DIM,
            hidden_dim: 6,
            latent_dim: 3,
        }
    }
}

impl AutoencoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Autoencoder<B> {
        Autoencoder {
            enc1: LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            enc2: LinearConfig::new(self.hidden_dim, self.latent_dim).init(device),
            dec1: LinearConfig::new(self.latent_dim, self.hidden_dim).init(device),
            dec2: LinearConfig::new(self.hidden_dim, self.input_dim).init(device),
            relu: Relu::new(),
        }
    }
}

impl<B: Backend> Autoencoder<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let z = self.relu.forward(self.enc1.forward(input));
        let z = self.relu.forward(self.enc2.forward(z));
        let z = self.relu.forward(self.dec1.forward(z));
        self.dec2.forward(z)
    }

    /// Per-sample reconstruction error (mean squared error across features),
    /// used both as the training loss and as the inference anomaly score.
    pub fn reconstruction_error(&self, input: Tensor<B, 2>) -> Tensor<B, 1> {
        let out = self.forward(input.clone());
        let diff = out - input;
        let sq = diff.clone() * diff;
        sq.mean_dim(1).squeeze(1)
    }
}

/// Train on a batch of benign flows only. Returns the trained model.
/// `epochs`/`lr` are deliberately small-model defaults; retune against real data.
pub fn train<B: AutodiffBackend>(
    device: &B::Device,
    config: &AutoencoderConfig,
    benign_samples: &[FlowFeatures],
    epochs: usize,
    lr: f64,
) -> Autoencoder<B> {
    let mut model = config.init::<B>(device);
    let mut optim = AdamConfig::new().init();

    let data: Vec<f32> = benign_samples.iter().flat_map(|f| f.0).collect();
    let n = benign_samples.len();
    let input = Tensor::<B, 1>::from_floats(data.as_slice(), device)
        .reshape([n, config.input_dim]);

    for _ in 0..epochs {
        let errors = model.reconstruction_error(input.clone());
        let loss = errors.mean();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);
    }
    model
}

/// Inference-time anomaly score for a single flow, plus a decision at a
/// chosen threshold. The threshold should be calibrated as, e.g., the 99th
/// percentile of reconstruction error on held-out benign traffic.
pub fn score<B: Backend>(model: &Autoencoder<B>, device: &B::Device, f: FlowFeatures) -> f32 {
    let input = Tensor::<B, 1>::from_floats(f.0.as_slice(), device).reshape([1, FEATURE_DIM]);
    let err = model.reconstruction_error(input);
    err.into_scalar().elem()
}

pub fn is_anomalous<B: Backend>(
    model: &Autoencoder<B>,
    device: &B::Device,
    f: FlowFeatures,
    threshold: f32,
) -> bool {
    score(model, device, f) > threshold
}
