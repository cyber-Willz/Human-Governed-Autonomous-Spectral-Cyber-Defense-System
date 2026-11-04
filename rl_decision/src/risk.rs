//! Risk model: calibrated `P(malicious | state)` and per-action expected loss.
//!
//! An online logistic regression, regularised toward hand-set prior weights
//! (so it behaves sensibly with zero labels and cannot be dragged far from
//! the prior by a handful of noisy verdicts). Labels come from human
//! decisions and honeytoken-backed containment.

use serde::{Deserialize, Serialize};

use crate::action::{loss, Action, ALL_ACTIONS, N_ACTIONS};
use crate::state::{idx, StateFeatures, STATE_DIM};

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z.clamp(-30.0, 30.0)).exp())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskModel {
    w: [f64; STATE_DIM],
    b: f64,
    w0: [f64; STATE_DIM],
    b0: f64,
    n_labels: u64,
    /// Exponential moving average of the Brier score on labels seen
    /// *before* each update (a prequential, honest calibration signal).
    brier_ema: f64,
    lr: f64,
    l2: f64,
}

impl Default for RiskModel {
    fn default() -> Self {
        let mut w = [0.0; STATE_DIM];
        w[idx::SEVERITY] = 0.8;
        w[idx::CONFIDENCE] = 2.5;
        w[idx::LANES] = 1.2;
        w[idx::HONEYTOKEN] = 5.0;
        w[idx::HYPERGRAPH] = 0.8;
        w[idx::RAMANUJAN] = 0.4;
        w[idx::CONNECTOME] = 0.8;
        w[idx::NSM] = 0.6;
        w[idx::SPECTRAL] = 0.6;
        w[idx::REPEAT_CONFIRMED] = 1.2;
        w[idx::REPEAT_DISMISSED] = -1.5;
        w[idx::PROTECTED] = 0.0;
        w[idx::KNOWN_INCIDENT] = 1.0;
        Self { w, b: -3.0, w0: w, b0: -3.0, n_labels: 0, brier_ema: 0.25, lr: 0.08, l2: 0.01 }
    }
}

impl RiskModel {
    pub fn p_malicious(&self, s: &StateFeatures) -> f64 {
        let mut z = self.b;
        for i in 0..STATE_DIM {
            z += self.w[i] * s.x[i];
        }
        sigmoid(z)
    }

    /// One SGD step on a real labelled outcome (counts toward confidence).
    pub fn update(&mut self, s: &StateFeatures, malicious: bool) {
        self.update_inner(s, malicious, true);
    }

    /// SGD step on a *synthetic* (sandbox) label. Does not count toward
    /// `n_labels`, so simulated data never makes the model look better
    /// evidenced than it is.
    pub fn update_synthetic(&mut self, s: &StateFeatures, malicious: bool) {
        self.update_inner(s, malicious, false);
    }

    /// Make the current weights the regularisation anchor (used after
    /// sandbox pre-training, so real feedback pulls toward the pre-trained
    /// model rather than the raw hand-set prior).
    pub fn rebase_prior(&mut self) {
        self.w0 = self.w;
        self.b0 = self.b;
    }

    fn update_inner(&mut self, s: &StateFeatures, malicious: bool, count: bool) {
        let y = if malicious { 1.0 } else { 0.0 };
        let p = self.p_malicious(s);
        let err = p - y;
        self.brier_ema = 0.98 * self.brier_ema + 0.02 * err * err;
        for i in 0..STATE_DIM {
            let g = err * s.x[i] + self.l2 * (self.w[i] - self.w0[i]);
            self.w[i] -= self.lr * g;
        }
        self.b -= self.lr * (err + self.l2 * (self.b - self.b0));
        if count {
            self.n_labels += 1;
        }
    }

    pub fn n_labels(&self) -> u64 {
        self.n_labels
    }

    pub fn brier(&self) -> f64 {
        self.brier_ema
    }

    /// Epistemic-uncertainty proxy in (0, 1]: 1 with no labels, falling as
    /// evidence accumulates. Used by the shield to cap how severe an action
    /// an under-evidenced model may pick.
    pub fn uncertainty(&self) -> f64 {
        1.0 / (1.0 + self.n_labels as f64 / 40.0).sqrt()
    }

    /// Expected loss of every action given `p = P(malicious)`.
    pub fn expected_losses(&self, s: &StateFeatures, p: f64) -> [f64; N_ACTIONS] {
        let mut out = [0.0; N_ACTIONS];
        for a in ALL_ACTIONS {
            out[a.index()] = p * loss(a, true, s.severity, s.protected) + (1.0 - p) * loss(a, false, s.severity, s.protected);
        }
        out
    }

    /// Loss-minimising action ignoring any safety shield (diagnostic).
    pub fn bayes_action(&self, s: &StateFeatures) -> Action {
        let l = self.expected_losses(s, self.p_malicious(s));
        let mut best = 0;
        for i in 1..N_ACTIONS {
            if l[i] < l[best] {
                best = i;
            }
        }
        Action::from_index(best).unwrap_or(Action::Escalate)
    }
}
