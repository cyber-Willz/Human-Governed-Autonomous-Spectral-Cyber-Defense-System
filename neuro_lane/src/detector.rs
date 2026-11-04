//! The connectome-LSTM sequence scorer, its trainer, and the guarded
//! fine-tuner.

use std::path::Path;

use hcrnn::network::{default_roles, default_threads};
use hcrnn::optim::Adam;
use hcrnn::rng::Rng;
use hcrnn::{Connectome, ConnectomeLstm, LstmConfig, RnnConfig, SequenceModel, Trial};

use crate::features::{FeatVec, N_FEAT};
use crate::synth::{self, Behavior};

#[derive(Debug, Clone)]
pub struct DetectorConfig {
    /// Regions per hemisphere of the synthetic stand-in connectome.
    pub per_hemi: usize,
    pub density: f64,
    pub conn_seed: u64,
    pub model_seed: u64,
    /// Bins of history a sequence window holds.
    pub seq_len: usize,
    /// Leading bins excluded from the loss (and not scored online).
    pub warm: usize,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self { per_hemi: 10, density: 0.3, conn_seed: 7, model_seed: 11, seq_len: 24, warm: 6 }
    }
}

#[derive(Debug, Clone)]
pub struct TrainSummary {
    pub steps: usize,
    pub final_train_loss: f64,
    pub seconds: f64,
}

#[derive(Debug, Clone)]
pub struct FineTuneOutcome {
    pub applied: bool,
    pub examples: usize,
    pub anchor_acc_before: f64,
    pub anchor_acc_after: f64,
    pub reason: String,
}

pub struct ConnectomeDetector {
    pub net: ConnectomeLstm,
    pub cfg: DetectorConfig,
    anchors: Vec<(Vec<FeatVec>, bool, Behavior)>,
}

fn sigmoid_free_score(y: &[f64], t: usize) -> f64 {
    // Mean of the last two outputs, clipped to [0, 1].
    let a = y[t - 1];
    let b = if t >= 2 { y[t - 2] } else { a };
    (0.5 * (a + b)).clamp(0.0, 1.0)
}

impl ConnectomeDetector {
    pub fn new(cfg: DetectorConfig) -> Result<Self, String> {
        let conn = Connectome::synthetic_human(cfg.per_hemi, cfg.density, cfg.conn_seed)?.log_compressed();
        Self::with_connectome(&conn, cfg)
    }

    /// Use any connectome (e.g. `Connectome::from_tvb_dir` for measured wiring).
    pub fn with_connectome(conn: &Connectome, cfg: DetectorConfig) -> Result<Self, String> {
        let (inp, out) = default_roles(conn)?;
        let rcfg = RnnConfig { n_in: N_FEAT, n_out: 1, input_regions: inp, output_regions: out, seed: cfg.model_seed, ..Default::default() };
        let net = ConnectomeLstm::new(conn, rcfg, LstmConfig::default())?;
        net.check_constraints()?;
        let mut d = Self { net, cfg, anchors: Vec::new() };
        d.anchors = d.make_set(60, 0xA11C_0DE, 0.0);
        Ok(d)
    }

    fn make_set(&self, n: usize, seed: u64, shift: f64) -> Vec<(Vec<FeatVec>, bool, Behavior)> {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|i| {
                let b = synth::ALL[i % synth::ALL.len()];
                (synth::sequence(&mut rng, b, self.cfg.seq_len, shift).0, b.hostile(), b)
            })
            .collect()
    }

    /// Independent copy (same wiring, same weights).
    pub fn fork(&self) -> Result<Self, String> {
        let mut d = Self::new(self.cfg.clone())?;
        d.net.params.copy_from_slice(&self.net.params);
        Ok(d)
    }

    fn trial_of(&self, seq: &[FeatVec]) -> Trial {
        synth::to_trial(seq, 0.0, 0, 0)
    }

    /// Score in [0, 1] for the *end* of `seq` (needs >= 1 bin; callers
    /// enforce a minimum history).
    pub fn score(&self, seq: &[FeatVec]) -> f64 {
        if seq.is_empty() {
            return 0.0;
        }
        let y = self.net.outputs(&self.trial_of(seq));
        sigmoid_free_score(&y, seq.len())
    }

    /// Feature indices ordered by how much zeroing that channel across the
    /// whole sequence lowers the score (occlusion attribution).
    pub fn drivers(&self, seq: &[FeatVec]) -> Vec<usize> {
        let base = self.score(seq);
        let mut d: Vec<(usize, f64)> = (0..N_FEAT)
            .map(|c| {
                let occluded: Vec<FeatVec> = seq
                    .iter()
                    .map(|f| {
                        let mut g = *f;
                        g[c] = 0.0;
                        g
                    })
                    .collect();
                (c, base - self.score(&occluded))
            })
            .collect();
        d.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        d.into_iter().filter(|(_, drop)| *drop > 1e-3).map(|(c, _)| c).collect()
    }

    pub fn train(&mut self, steps: usize, batch: usize, lr: f64, seed: u64) -> TrainSummary {
        let t0 = std::time::Instant::now();
        let mut rng = Rng::new(seed);
        let mut opt = Adam::new(self.net.params().len(), lr, 1.0);
        let mut grad = vec![0.0; self.net.params().len()];
        let threads = default_threads();
        let (mut acc, mut n) = (0.0, 0usize);
        for _ in 0..steps {
            let trials: Vec<Trial> = (0..batch).map(|_| synth::sample_trial(&mut rng, self.cfg.seq_len, 0.5, self.cfg.warm).0).collect();
            let loss = self.net.batch_loss_and_grad(&trials, &mut grad, threads);
            opt.step(self.net.params_mut(), &grad);
            acc += loss;
            n += 1;
            if n > 20 {
                acc *= 0.95;
                n = 20;
            }
        }
        TrainSummary { steps, final_train_loss: acc / n.max(1) as f64, seconds: t0.elapsed().as_secs_f64() }
    }

    /// Accuracy at `thr` over labelled sequences, only scoring windows past warm-up.
    pub fn accuracy(&self, set: &[(Vec<FeatVec>, bool, Behavior)], thr: f64) -> f64 {
        let ok = set.iter().filter(|(s, y, _)| (self.score(s) >= thr) == *y).count();
        ok as f64 / set.len().max(1) as f64
    }

    pub fn anchor_accuracy(&self, thr: f64) -> f64 {
        self.accuracy(&self.anchors, thr)
    }

    /// Fresh labelled set (seeded; `shift > 0` = distribution-shifted).
    pub fn eval_set(&self, n: usize, seed: u64, shift: f64) -> Vec<(Vec<FeatVec>, bool, Behavior)> {
        self.make_set(n, seed, shift)
    }

    /// Fine-tunes on analyst-labelled sequences mixed 1:1 with synthetic
    /// replay (to resist forgetting) at a small learning rate, then keeps the
    /// update only if accuracy on the fixed anchor set did not fall by more
    /// than `anchor_tolerance`; otherwise restores the previous parameters
    /// bit-for-bit.
    pub fn fine_tune_guarded(
        &mut self,
        labelled: &[(Vec<FeatVec>, f64)],
        steps: usize,
        lr: f64,
        anchor_tolerance: f64,
        seed: u64,
    ) -> FineTuneOutcome {
        let before = self.anchor_accuracy(0.5);
        if labelled.is_empty() {
            return FineTuneOutcome { applied: false, examples: 0, anchor_acc_before: before, anchor_acc_after: before, reason: "no labelled examples".into() };
        }
        let saved = self.net.params.clone();
        let mut rng = Rng::new(seed);
        let mut opt = Adam::new(saved.len(), lr, 1.0);
        let mut grad = vec![0.0; saved.len()];
        let threads = default_threads();
        for _ in 0..steps {
            let mut trials: Vec<Trial> = Vec::new();
            for _ in 0..8 {
                let (s, y) = &labelled[rng.below(labelled.len())];
                trials.push(synth::to_trial(s, *y, 0, self.cfg.warm.min(s.len().saturating_sub(1))));
                trials.push(synth::sample_trial(&mut rng, self.cfg.seq_len, 0.5, self.cfg.warm).0);
            }
            self.net.batch_loss_and_grad(&trials, &mut grad, threads);
            opt.step(self.net.params_mut(), &grad);
        }
        let after = self.anchor_accuracy(0.5);
        if after + anchor_tolerance < before {
            self.net.params.copy_from_slice(&saved);
            FineTuneOutcome {
                applied: false,
                examples: labelled.len(),
                anchor_acc_before: before,
                anchor_acc_after: after,
                reason: format!("anchor accuracy fell {before:.3} -> {after:.3} (> tolerance {anchor_tolerance}); reverted"),
            }
        } else {
            FineTuneOutcome { applied: true, examples: labelled.len(), anchor_acc_before: before, anchor_acc_after: after, reason: "accepted".into() }
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        self.net.save_params(path)
    }
    pub fn load(&mut self, path: &Path) -> Result<(), String> {
        self.net.load_params(path)
    }
}
