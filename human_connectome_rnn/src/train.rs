//! Training loop, evaluation and a simple per-system analysis of delay activity.

use crate::model::{masked_loss, SequenceModel};
use crate::optim::Adam;
use crate::rng::Rng;
use crate::task::{MemoryGuidedTask, Trial};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub steps: usize,
    pub batch: usize,
    pub lr: f64,
    pub clip: f64,
    pub seed: u64,
    pub log_every: usize,
    pub eval_trials: usize,
    pub threads: usize,
    /// Weight of the Dirichlet-energy (Rayleigh quotient) regulariser; 0 disables it.
    pub energy_lambda: f64,
    /// The regulariser is switched on only from this step (it can silence the network if applied
    /// before the task has been learned).
    pub energy_start: usize,
    /// If set, project recurrent weights so the delayed Arnoldi radius never exceeds this.
    pub radius_cap: Option<f64>,
    pub project_every: usize,
    /// Linear warm-up steps for the learning rate.
    pub warmup: usize,
    /// Cosine-decay the learning rate to 5% of `lr` over `steps`.
    pub cosine: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            steps: 400,
            batch: 24,
            lr: 6e-3,
            clip: 1.0,
            seed: 1,
            log_every: 50,
            eval_trials: 96,
            threads: crate::network::default_threads(),
            energy_lambda: 0.0,
            energy_start: 0,
            radius_cap: None,
            project_every: 10,
            warmup: 0,
            cosine: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EvalMetrics {
    pub loss: f64,
    /// Mean absolute angular error (degrees) of the decoded response; chance is 90.
    pub angular_error_deg: f64,
    /// Fraction of trials decoded within 30 degrees of the cue.
    pub acc_30deg: f64,
}

#[derive(Clone, Debug)]
pub struct LogLine {
    pub step: usize,
    pub train_loss: f64,
    pub grad_norm: f64,
    /// Arnoldi radius of the delayed linearised dynamics at this step.
    pub radius: f64,
    pub eval: EvalMetrics,
    pub seconds: f64,
}

#[derive(Clone, Debug)]
pub struct TrainReport {
    pub log: Vec<LogLine>,
    pub final_eval: EvalMetrics,
    pub seconds: f64,
}

/// Learning rate at a 1-based step: linear warm-up then optional cosine decay to 5%.
pub fn lr_at(cfg: &TrainConfig, step: usize) -> f64 {
    let mut lr = cfg.lr;
    if cfg.warmup > 0 && step <= cfg.warmup {
        lr *= step as f64 / cfg.warmup as f64;
    } else if cfg.cosine && cfg.steps > cfg.warmup {
        let p = (step - cfg.warmup) as f64 / (cfg.steps - cfg.warmup) as f64;
        lr *= 0.05 + 0.95 * 0.5 * (1.0 + (std::f64::consts::PI * p.min(1.0)).cos());
    }
    lr
}

pub fn make_trials(task: &MemoryGuidedTask, n: usize, seed: u64) -> Vec<Trial> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| task.sample(&mut rng)).collect()
}

fn wrap_angle(a: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut d = (a + std::f64::consts::PI).rem_euclid(two_pi) - std::f64::consts::PI;
    if d < -std::f64::consts::PI {
        d += two_pi;
    }
    d
}

pub fn evaluate<M: SequenceModel>(net: &M, trials: &[Trial]) -> EvalMetrics {
    let (mut loss, mut err, mut hit) = (0.0, 0.0, 0usize);
    for tr in trials {
        let y = net.outputs(tr);
        loss += masked_loss(&y, tr);
        // Decode from the last steps of the response window.
        let win = 8.min(tr.t - tr.go);
        let (mut cx, mut cy) = (0.0, 0.0);
        for t in (tr.t - win)..tr.t {
            cx += y[t * 2];
            cy += y[t * 2 + 1];
        }
        let pred = cy.atan2(cx);
        let e = wrap_angle(pred - tr.angle).abs().to_degrees();
        err += e;
        if e < 30.0 {
            hit += 1;
        }
    }
    let n = trials.len().max(1) as f64;
    EvalMetrics { loss: loss / n, angular_error_deg: err / n, acc_30deg: hit as f64 / n }
}

pub fn train<M: SequenceModel>(
    net: &mut M,
    task: &MemoryGuidedTask,
    cfg: &TrainConfig,
    mut on_log: impl FnMut(&LogLine),
) -> TrainReport {
    let mut rng = Rng::new(cfg.seed);
    let eval_set = make_trials(task, cfg.eval_trials, cfg.seed ^ 0xE7A1_5EED);
    let mut opt = Adam::new(net.params().len(), cfg.lr, cfg.clip);
    net.set_energy_lambda(cfg.energy_lambda);
    let mut grad = vec![0.0; net.params().len()];
    let start = Instant::now();
    let mut log = Vec::new();
    let (mut acc_loss, mut acc_n, mut last_gn) = (0.0, 0usize, 0.0);
    for step in 1..=cfg.steps {
        opt.lr = lr_at(cfg, step);
        net.set_energy_lambda(if step >= cfg.energy_start { cfg.energy_lambda } else { 0.0 });
        let batch: Vec<Trial> = (0..cfg.batch).map(|_| task.sample(&mut rng)).collect();
        let loss = net.batch_loss_and_grad(&batch, &mut grad, cfg.threads);
        last_gn = opt.step(net.params_mut(), &grad);
        if let Some(cap) = cfg.radius_cap {
            if step % cfg.project_every.max(1) == 0 {
                net.project_radius(cap);
            }
        }
        acc_loss += loss;
        acc_n += 1;
        if step % cfg.log_every == 0 || step == cfg.steps {
            let line = LogLine {
                step,
                train_loss: acc_loss / acc_n as f64,
                grad_norm: last_gn,
                radius: net.radius_for_log(),
                eval: evaluate(net, &eval_set),
                seconds: start.elapsed().as_secs_f64(),
            };
            on_log(&line);
            log.push(line);
            acc_loss = 0.0;
            acc_n = 0;
        }
    }
    let _ = last_gn;
    net.set_energy_lambda(0.0);
    let final_eval = evaluate(net, &eval_set);
    TrainReport { log, final_eval, seconds: start.elapsed().as_secs_f64() }
}

/// For each system: mean (over its regions) across-trial standard deviation of the
/// excitatory firing rate at the end of the delay. Because the stimulus is identical
/// apart from the cue angle, this is a proxy for how much cue information a region
/// is still carrying when the go signal arrives.
pub fn delay_cue_variability_by_system<M: SequenceModel>(net: &M, trials: &[Trial]) -> Vec<(String, usize, f64)> {
    let region_systems = net.region_systems();
    let r = region_systems.len();
    let mut sum = vec![0.0; r];
    let mut sumsq = vec![0.0; r];
    for tr in trials {
        let act = net.e_activity_at(tr, tr.go);
        for k in 0..r {
            sum[k] += act[k];
            sumsq[k] += act[k] * act[k];
        }
    }
    let m = trials.len().max(1) as f64;
    let sd: Vec<f64> = (0..r).map(|k| ((sumsq[k] / m) - (sum[k] / m).powi(2)).max(0.0).sqrt()).collect();
    let mut systems: Vec<String> = Vec::new();
    for s in region_systems {
        if !systems.contains(s) {
            systems.push(s.clone());
        }
    }
    let mut out: Vec<(String, usize, f64)> = systems
        .into_iter()
        .map(|s| {
            let idx: Vec<usize> = (0..r).filter(|&k| region_systems[k] == s).collect();
            let mean = idx.iter().map(|&k| sd[k]).sum::<f64>() / idx.len() as f64;
            (s, idx.len(), mean)
        })
        .collect();
    out.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    out
}

/// Where the cue-dependent delay-end activity lives on the connectome's harmonic basis.
#[derive(Clone, Debug)]
pub struct HarmonicProfile {
    /// Fractions of cue-dependent E-rate variance at the end of the delay carried by the
    /// lowest, middle and highest third of connectome harmonics (sum to 1).
    pub low: f64,
    pub mid: f64,
    pub high: f64,
    /// Rayleigh quotient (Dirichlet energy / power) of the trial-centred activity.
    pub dirichlet_quotient: f64,
}

/// Project trial-centred E rates at the go time onto the eigenvectors of the connectome's
/// normalised Laplacian (computed by `spectral_hypergraph`; see `topology::harmonics`).
pub fn harmonic_profile<M: SequenceModel>(net: &M, trials: &[Trial], h: &crate::topology::Harmonics) -> HarmonicProfile {
    let r = net.region_systems().len();
    let rows: Vec<Vec<f64>> = trials.iter().map(|tr| net.e_activity_at(tr, tr.go)).collect();
    let m = rows.len().max(1) as f64;
    let mean: Vec<f64> = (0..r).map(|i| rows.iter().map(|v| v[i]).sum::<f64>() / m).collect();
    let mut power = vec![0.0; r];
    let (mut num, mut den) = (0.0, 0.0);
    for v in &rows {
        let c: Vec<f64> = (0..r).map(|i| v[i] - mean[i]).collect();
        for k in 0..r {
            let coef: f64 = (0..r).map(|i| h.vectors[(i, k)] * c[i]).sum();
            power[k] += coef * coef / m;
            num += h.values[k] * coef * coef / m;
        }
        den += c.iter().map(|x| x * x).sum::<f64>() / m;
    }
    let total: f64 = power.iter().sum::<f64>().max(1e-300);
    let third = (r / 3).max(1);
    let low: f64 = power[..third].iter().sum::<f64>() / total;
    let high: f64 = power[r - third..].iter().sum::<f64>() / total;
    HarmonicProfile { low, mid: 1.0 - low - high, high, dirichlet_quotient: num / den.max(1e-300) }
}
