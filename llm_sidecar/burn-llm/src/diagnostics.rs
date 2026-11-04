//! Cheap numerical health checks for a loaded model.
//!
//! The failure mode this exists for: a wrong-but-loadable checkpoint/config (missing norm,
//! wrong RoPE base, bad dtype) gives *fluent-looking garbage*, not an error. A residual stream
//! that explodes, collapses, or goes non-finite is the earliest visible symptom.

use burn::tensor::{backend::Backend, Int, Tensor};

use crate::model::LlmModel;

#[derive(Debug, Clone)]
pub struct LayerStat {
    pub layer: usize,
    /// Root-mean-square of the residual stream after this block (over tokens and channels).
    pub rms: f32,
    pub all_finite: bool,
}

pub struct ProbeReport {
    pub layers: Vec<LayerStat>,
    pub logits_finite: bool,
}

impl ProbeReport {
    pub fn healthy(&self) -> bool {
        self.logits_finite && self.layers.iter().all(|l| l.all_finite)
    }
}

/// Runs `prompt` through the model and reports per-layer residual RMS and finiteness.
pub fn probe<B: Backend>(model: &LlmModel<B>, prompt: &[u32], device: &B::Device) -> ProbeReport {
    let mut state = model.init_generation(device, prompt.len().max(1));
    let ids: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let ids = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device).reshape([1, prompt.len()]);
    let (logits, taps) = model.forward_last_probe(ids, &mut state);

    let to_vec = |d: burn::tensor::TensorData| d.convert::<f32>().to_vec::<f32>().unwrap();
    let layers = taps
        .into_iter()
        .enumerate()
        .map(|(layer, h)| {
            let v = to_vec(h.into_data());
            let all_finite = v.iter().all(|x| x.is_finite());
            let rms = (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt() as f32;
            LayerStat { layer, rms, all_finite }
        })
        .collect();
    let logits_finite = to_vec(logits.into_data()).iter().all(|x| x.is_finite());
    ProbeReport { layers, logits_finite }
}
