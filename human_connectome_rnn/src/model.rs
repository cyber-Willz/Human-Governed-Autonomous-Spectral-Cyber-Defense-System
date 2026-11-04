//! Common interface for the sequence models trained on the memory-guided task, so
//! that `train::train`, `train::evaluate` and the analysis helpers work for both the
//! leaky-rate `ConnectomeRnn` and the gated `ConnectomeLstm`.

use crate::network::ConnectomeRnn;
use crate::task::Trial;
use std::path::Path;

pub trait SequenceModel: Sync {
    fn params(&self) -> &[f64];
    fn params_mut(&mut self) -> &mut [f64];

    /// Mean loss over `trials`; writes the mean gradient into `grad`.
    fn batch_loss_and_grad(&self, trials: &[Trial], grad: &mut [f64], threads: usize) -> f64;

    /// Network output, `trial.t x n_out` row-major.
    fn outputs(&self, trial: &Trial) -> Vec<f64>;

    /// Firing-rate-like activity of the R excitatory units at time index `step`
    /// (index 0 = initial state, index s = state after s steps).
    fn e_activity_at(&self, trial: &Trial, step: usize) -> Vec<f64>;

    fn region_systems(&self) -> &[String];

    /// Dale's law and wiring-mask verification of the realised weights.
    fn check_constraints(&self) -> Result<(), String>;

    /// Persist the model into `dir`; returns a short description of what was written.
    fn save_to_dir(&self, dir: &Path) -> Result<String, String>;

    // ---- optional RNN-only hooks (defaults are no-ops) ----

    fn set_energy_lambda(&mut self, _lambda: f64) {}
    /// Radius of the delayed linearised dynamics for the training log (NaN if not defined).
    fn radius_for_log(&self) -> f64 {
        f64::NAN
    }
    fn project_radius(&mut self, _cap: f64) {}
}

/// Masked squared-error loss of outputs `y` against the trial targets (shared by all models).
pub fn masked_loss(y: &[f64], trial: &Trial) -> f64 {
    let nout = trial.n_out;
    let denom = trial.mask.iter().sum::<f64>() * nout as f64;
    let norm = if denom > 0.0 { 1.0 / denom } else { 1.0 };
    let mut loss = 0.0;
    for t in 0..trial.t {
        for k in 0..nout {
            let e = y[t * nout + k] - trial.targets[t * nout + k];
            loss += trial.mask[t] * e * e * norm;
        }
    }
    loss
}

impl SequenceModel for ConnectomeRnn {
    fn params(&self) -> &[f64] {
        &self.params
    }
    fn params_mut(&mut self) -> &mut [f64] {
        &mut self.params
    }
    fn batch_loss_and_grad(&self, trials: &[Trial], grad: &mut [f64], threads: usize) -> f64 {
        ConnectomeRnn::batch_loss_and_grad(self, trials, grad, threads)
    }
    fn outputs(&self, trial: &Trial) -> Vec<f64> {
        self.forward(trial).y
    }
    fn e_activity_at(&self, trial: &Trial, step: usize) -> Vec<f64> {
        let (n, r) = (self.n_units, self.n_regions);
        let ro = self.forward(trial);
        ro.r[step * n..step * n + r].to_vec()
    }
    fn region_systems(&self) -> &[String] {
        &self.region_systems
    }
    fn check_constraints(&self) -> Result<(), String> {
        ConnectomeRnn::check_constraints(self)
    }
    fn save_to_dir(&self, dir: &Path) -> Result<String, String> {
        self.save_params(&dir.join("params.bin"))?;
        self.export_weights_csv(&dir.join("effective_weights.csv"))?;
        Ok("params.bin and effective_weights.csv".into())
    }
    fn set_energy_lambda(&mut self, lambda: f64) {
        self.energy_lambda = lambda;
    }
    fn radius_for_log(&self) -> f64 {
        ConnectomeRnn::spectral_radius(self)
    }
    fn project_radius(&mut self, cap: f64) {
        self.enforce_radius_cap(cap);
    }
}
