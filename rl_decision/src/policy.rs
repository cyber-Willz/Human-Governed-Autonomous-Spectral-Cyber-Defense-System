//! Policy network: a small MLP (`STATE_DIM -> HIDDEN (tanh) -> N_ACTIONS`)
//! with a softmax head, trained by policy gradient and Adam. Hand-written
//! forward/backward in plain `f64` -- no ML framework, so it adds no
//! dependency conflicts to this workspace and is fully deterministic.
//!
//! # Learning rule
//! The cost model ([`crate::action::reward`]) is known, so once a *label*
//! (malicious / benign) is revealed, the reward of **every** action in that
//! state is computable, not just the one that was executed. Learning
//! therefore uses full information -- no importance weights, no variance
//! from sampling an action, and it is correct off-policy (the logged action
//! is irrelevant). It is also why this layer needs no exploration: an
//! exploring policy in a security product would mean deliberately taking
//! risky actions on live hosts, and it is not necessary here.
//!
//! The textbook softmax policy gradient is `∂J/∂z_j = π_j (r_j − r̄)`. The
//! `π_j` factor makes it *plateau*: an action whose probability has
//! collapsed early gets almost no gradient even when it is the best one
//! (observed here: the heaviest ban tier was never found). So the update
//! uses the **natural** policy-gradient direction for a softmax head,
//! dropping that factor:
//!
//! ```text
//! ∂L/∂z_j = −scale · (r_j − r̄)  +  β · (ln π_j + H(π))
//! ```
//!
//! whose fixed point is the entropy-regularised optimal policy
//! `π(a|s) ∝ exp(scale · E[r(a)|s] / β)` -- near-greedy on clear cases,
//! hedged on ambiguous ones. (`β` is the temperature knob.)

use serde::{Deserialize, Serialize};

use crate::action::{N_ACTIONS};
use crate::rng::Rng;
use crate::state::{StateFeatures, STATE_DIM};

pub const HIDDEN: usize = 24;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    t: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyNet {
    w1: Vec<f64>, // HIDDEN x STATE_DIM
    b1: Vec<f64>, // HIDDEN
    w2: Vec<f64>, // N_ACTIONS x HIDDEN
    b2: Vec<f64>, // N_ACTIONS
    adam: Adam,
    /// Scale applied to reward differences (rewards live in ~[-2, 0], and
    /// raw differences would make softmax gradients tiny).
    pub reward_scale: f64,
    pub entropy_beta: f64,
}

pub fn softmax(z: &[f64]) -> [f64; N_ACTIONS] {
    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut out = [0.0; N_ACTIONS];
    let mut sum = 0.0;
    for i in 0..N_ACTIONS {
        out[i] = (z[i] - m).exp();
        sum += out[i];
    }
    for v in out.iter_mut() {
        *v /= sum;
    }
    out
}

impl PolicyNet {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let lim1 = (6.0 / (STATE_DIM + HIDDEN) as f64).sqrt();
        let lim2 = (6.0 / (HIDDEN + N_ACTIONS) as f64).sqrt();
        let w1 = (0..HIDDEN * STATE_DIM).map(|_| rng.range(-lim1, lim1)).collect();
        let w2 = (0..N_ACTIONS * HIDDEN).map(|_| rng.range(-lim2, lim2)).collect();
        let n = HIDDEN * STATE_DIM + HIDDEN + N_ACTIONS * HIDDEN + N_ACTIONS;
        Self {
            w1,
            b1: vec![0.0; HIDDEN],
            w2,
            b2: vec![0.0; N_ACTIONS],
            adam: Adam { m: vec![0.0; n], v: vec![0.0; n], t: 0 },
            reward_scale: 5.0,
            entropy_beta: 0.15,
        }
    }

    fn forward(&self, x: &[f64; STATE_DIM]) -> ([f64; HIDDEN], [f64; N_ACTIONS]) {
        let mut h = [0.0; HIDDEN];
        for j in 0..HIDDEN {
            let mut z = self.b1[j];
            let row = &self.w1[j * STATE_DIM..(j + 1) * STATE_DIM];
            for i in 0..STATE_DIM {
                z += row[i] * x[i];
            }
            h[j] = z.tanh();
        }
        let mut logits = [0.0; N_ACTIONS];
        for a in 0..N_ACTIONS {
            let mut z = self.b2[a];
            let row = &self.w2[a * HIDDEN..(a + 1) * HIDDEN];
            for j in 0..HIDDEN {
                z += row[j] * h[j];
            }
            logits[a] = z;
        }
        (h, logits)
    }

    pub fn probs(&self, s: &StateFeatures) -> [f64; N_ACTIONS] {
        softmax(&self.forward(&s.x).1)
    }

    /// Argmax restricted to `allowed` (falls back to index 0 if none).
    pub fn argmax_allowed(&self, s: &StateFeatures, allowed: &[bool; N_ACTIONS]) -> usize {
        let p = self.probs(s);
        let mut best: Option<usize> = None;
        for a in 0..N_ACTIONS {
            if allowed[a] && best.map_or(true, |b| p[a] > p[b]) {
                best = Some(a);
            }
        }
        best.unwrap_or(0)
    }

    /// One Adam step on a mini-batch of `(state, per-action rewards)`.
    /// Returns the mean policy objective `Σ π r` (higher is better).
    pub fn train_batch(&mut self, batch: &[(StateFeatures, [f64; N_ACTIONS])], lr: f64) -> f64 {
        if batch.is_empty() {
            return 0.0;
        }
        let n1 = self.w1.len();
        let n2 = self.b1.len();
        let n3 = self.w2.len();
        let n4 = self.b2.len();
        let mut g = vec![0.0; n1 + n2 + n3 + n4];
        let mut mean_obj = 0.0;

        for (s, rew) in batch {
            let (h, logits) = self.forward(&s.x);
            let pi = softmax(&logits);
            let rbar: f64 = (0..N_ACTIONS).map(|a| pi[a] * rew[a]).sum();
            mean_obj += rbar;
            let ent: f64 = -(0..N_ACTIONS).map(|a| pi[a] * (pi[a].max(1e-12)).ln()).sum::<f64>();

            // dLoss/dz  (Loss = -J - beta*H)
            let mut dz = [0.0; N_ACTIONS];
            for a in 0..N_ACTIONS {
                dz[a] = -self.reward_scale * (rew[a] - rbar) + self.entropy_beta * (pi[a].max(1e-12).ln() + ent);
            }
            // layer 2
            let mut dh = [0.0; HIDDEN];
            for a in 0..N_ACTIONS {
                for j in 0..HIDDEN {
                    g[n1 + n2 + a * HIDDEN + j] += dz[a] * h[j];
                    dh[j] += dz[a] * self.w2[a * HIDDEN + j];
                }
                g[n1 + n2 + n3 + a] += dz[a];
            }
            // layer 1 (tanh')
            for j in 0..HIDDEN {
                let dpre = dh[j] * (1.0 - h[j] * h[j]);
                for i in 0..STATE_DIM {
                    g[j * STATE_DIM + i] += dpre * s.x[i];
                }
                g[n1 + j] += dpre;
            }
        }

        let inv = 1.0 / batch.len() as f64;
        // gradient-norm clip for stability
        let norm = g.iter().map(|v| (v * inv) * (v * inv)).sum::<f64>().sqrt();
        let clip = if norm > 5.0 { 5.0 / norm } else { 1.0 };

        self.adam.t += 1;
        let (b1c, b2c, eps): (f64, f64, f64) = (0.9, 0.999, 1e-8);
        let t = self.adam.t as f64;
        let c1 = 1.0 - b1c.powf(t);
        let c2 = 1.0 - b2c.powf(t);
        for k in 0..g.len() {
            let gk = g[k] * inv * clip;
            self.adam.m[k] = b1c * self.adam.m[k] + (1.0 - b1c) * gk;
            self.adam.v[k] = b2c * self.adam.v[k] + (1.0 - b2c) * gk * gk;
            let step = lr * (self.adam.m[k] / c1) / ((self.adam.v[k] / c2).sqrt() + eps);
            if k < n1 {
                self.w1[k] -= step;
            } else if k < n1 + n2 {
                self.b1[k - n1] -= step;
            } else if k < n1 + n2 + n3 {
                self.w2[k - n1 - n2] -= step;
            } else {
                self.b2[k - n1 - n2 - n3] -= step;
            }
        }
        mean_obj * inv
    }
}
