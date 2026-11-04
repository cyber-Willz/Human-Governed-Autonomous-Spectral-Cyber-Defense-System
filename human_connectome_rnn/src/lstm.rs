//! Connectome-constrained LSTM.
//!
//! Same biological scaffold as `ConnectomeRnn` (it is built *from* one, so wiring,
//! delays, E/I split, calibration and null-wiring modes are identical): R regions give
//! N = 2R units, units 0..R excitatory, R..2R inhibitory. What changes is the unit
//! dynamics. Every unit carries a cell state `c` and an output `h`, and four gates read
//! the same masked, delayed, sign-constrained synapses:
//!
//!   z^k_t[u] = b^k[u] + [u in input] W_in^k x_t + sum_e w^k_e * h_{t - delay_e}[pre_e]
//!   i, f, o = sigmoid(z^i, z^f, z^o)        g = tanh(z^g)
//!   c_{t+1} = f * c_t + i * g
//!   h_{t+1} = o * tanh(c_{t+1})
//!   y_t     = W_out h_{t+1}[out_units] + b_out
//!
//! Constraints kept by construction, per gate k: a synapse exists only where the
//! connectome has a tract (the trainable set is the allowed set), long-range
//! projections come from E units only, and Dale's law fixes the sign of every synapse by
//! its presynaptic unit, `w^k_e = sign(pre) * softplus(theta^k_e)`. Each gate has its own
//! magnitudes on the shared wiring, so the recurrent parameter count is 4x the RNN's.
//!
//! Gradients are exact BPTT through the delayed recurrence (checked against finite
//! differences in the test-suite).

use crate::connectome::Connectome;
use crate::model::SequenceModel;
use crate::network::{ConnectomeRnn, RnnConfig};
use crate::rng::Rng;
use crate::task::Trial;
use std::fs;
use std::path::Path;

/// Gate order inside every `[.. * 4]` block.
const GI: usize = 0;
const GF: usize = 1;
const GO: usize = 2;
const GG: usize = 3;

#[derive(Clone, Debug)]
pub struct LstmConfig {
    /// Initial recurrent magnitude of the i/f/o gates relative to the candidate path
    /// (the candidate path starts at the RNN's calibrated strength).
    pub gate_scale: f64,
    /// Initial forget-gate bias (standard LSTM trick: start by remembering).
    pub forget_bias: f64,
}

impl Default for LstmConfig {
    fn default() -> Self {
        LstmConfig { gate_scale: 0.5, forget_bias: 2.0 }
    }
}

#[derive(Clone, Debug)]
pub struct LstmLayout {
    /// `ne * 4`, edge-major: theta[ei * 4 + k]
    pub theta: usize,
    /// `n_in_units * n_in * 4`: w_in[(kk * n_in + c) * 4 + k]
    pub w_in: usize,
    /// `n * 4`: b[u * 4 + k]
    pub b: usize,
    pub w_out: usize,
    pub b_out: usize,
    pub total: usize,
}

pub struct LstmRollout {
    pub t: usize,
    /// (t+1) x N cell states, row 0 = zeros
    pub c: Vec<f64>,
    /// (t+1) x N outputs, row 0 = zeros
    pub h: Vec<f64>,
    /// t x N x 4 gate activations (i, f, o, g)
    pub gates: Vec<f64>,
    /// t x n_out
    pub y: Vec<f64>,
}

pub struct ConnectomeLstm {
    /// Wiring skeleton (edges, masks, delays, roles, config). Its own `params` are only
    /// used to seed the LSTM's initial magnitudes and are not trained.
    pub base: ConnectomeRnn,
    pub lcfg: LstmConfig,
    pub layout: LstmLayout,
    pub params: Vec<f64>,
}

fn softplus(x: f64) -> f64 {
    if x > 30.0 { x } else { x.exp().ln_1p() }
}
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn softplus_inv(m: f64) -> f64 {
    if m > 30.0 { m } else { m + (-(-m).exp_m1()).ln() }
}

impl ConnectomeLstm {
    pub fn new(conn: &Connectome, cfg: RnnConfig, lcfg: LstmConfig) -> Result<Self, String> {
        if lcfg.gate_scale <= 0.0 {
            return Err("gate_scale must be positive".into());
        }
        let base = ConnectomeRnn::new(conn, cfg)?;
        let n = base.n_units;
        let ne = base.edges.len();
        let nin_units = base.in_units.len();
        let (nin, nout) = (base.cfg.n_in, base.cfg.n_out);
        let nq = base.out_units.len();
        let bl = &base.layout;

        let theta = 0;
        let w_in = theta + ne * 4;
        let b = w_in + nin_units * nin * 4;
        let w_out = b + n * 4;
        let b_out = w_out + nout * nq;
        let layout = LstmLayout { theta, w_in, b, w_out, b_out, total: b_out + nout };
        let mut params = vec![0.0; layout.total];

        let mut rng = Rng::new(base.cfg.seed ^ 0x1573_7A11);
        // recurrent magnitudes: candidate path = the RNN's calibrated strength, gates start weaker
        for ei in 0..ne {
            let th = base.params[bl.theta + ei];
            let weak = softplus_inv((lcfg.gate_scale * softplus(th)).max(1e-9));
            for k in 0..4 {
                params[layout.theta + ei * 4 + k] = if k == GG { th } else { weak };
            }
        }
        // input weights: candidate = the RNN's, gates = small random
        let s_in = 1.0 / (nin as f64).sqrt();
        for j in 0..nin_units * nin {
            for k in 0..4 {
                params[layout.w_in + j * 4 + k] =
                    if k == GG { base.params[bl.w_in + j] } else { lcfg.gate_scale * s_in * rng.normal() };
            }
        }
        for u in 0..n {
            params[layout.b + u * 4 + GF] = lcfg.forget_bias;
        }
        params[layout.w_out..layout.total].copy_from_slice(&base.params[bl.w_out..bl.total]);
        Ok(ConnectomeLstm { base, lcfg, layout, params })
    }

    pub fn num_params(&self) -> usize {
        self.params.len()
    }

    fn sign(&self, unit: usize) -> f64 {
        if self.base.is_excitatory(unit) { 1.0 } else { -1.0 }
    }

    /// Effective synaptic weights, edge-major `[ei * 4 + k]`: sign(pre) * softplus(theta).
    pub fn effective_weights(&self) -> Vec<f64> {
        let ne = self.base.edges.len();
        let mut w = vec![0.0; ne * 4];
        for (ei, e) in self.base.edges.iter().enumerate() {
            let s = self.sign(e.pre);
            for k in 0..4 {
                w[ei * 4 + k] = s * softplus(self.params[self.layout.theta + ei * 4 + k]);
            }
        }
        w
    }

    pub fn forward(&self, trial: &Trial) -> LstmRollout {
        self.forward_with(&self.effective_weights(), trial)
    }

    pub fn forward_with(&self, w: &[f64], trial: &Trial) -> LstmRollout {
        let n = self.base.n_units;
        let (nin, nout) = (self.base.cfg.n_in, self.base.cfg.n_out);
        let nq = self.base.out_units.len();
        let l = &self.layout;
        let win = &self.params[l.w_in..l.b];
        let bias = &self.params[l.b..l.w_out];
        let wout = &self.params[l.w_out..l.b_out];
        let bout = &self.params[l.b_out..l.total];
        let t_len = trial.t;
        let mut c = vec![0.0; (t_len + 1) * n];
        let mut h = vec![0.0; (t_len + 1) * n];
        let mut gates = vec![0.0; t_len * n * 4];
        let mut y = vec![0.0; t_len * nout];
        let mut z = vec![0.0; n * 4];
        for t in 0..t_len {
            z.copy_from_slice(bias);
            for (kk, &u) in self.base.in_units.iter().enumerate() {
                for ch in 0..nin {
                    let x = trial.inputs[t * nin + ch];
                    let o = (kk * nin + ch) * 4;
                    for k in 0..4 {
                        z[u * 4 + k] += win[o + k] * x;
                    }
                }
            }
            for (ei, e) in self.base.edges.iter().enumerate() {
                if t >= e.delay {
                    let hv = h[(t - e.delay) * n + e.pre];
                    for k in 0..4 {
                        z[e.post * 4 + k] += w[ei * 4 + k] * hv;
                    }
                }
            }
            for u in 0..n {
                let gi = sigmoid(z[u * 4 + GI]);
                let gf = sigmoid(z[u * 4 + GF]);
                let go = sigmoid(z[u * 4 + GO]);
                let gg = z[u * 4 + GG].tanh();
                let o = (t * n + u) * 4;
                gates[o + GI] = gi;
                gates[o + GF] = gf;
                gates[o + GO] = go;
                gates[o + GG] = gg;
                let cn = gf * c[t * n + u] + gi * gg;
                c[(t + 1) * n + u] = cn;
                h[(t + 1) * n + u] = go * cn.tanh();
            }
            for k in 0..nout {
                let mut s = bout[k];
                for (q, &u) in self.base.out_units.iter().enumerate() {
                    s += wout[k * nq + q] * h[(t + 1) * n + u];
                }
                y[t * nout + k] = s;
            }
        }
        LstmRollout { t: t_len, c, h, gates, y }
    }

    /// Loss for one trial; accumulates dLoss/dparams into `grad`.
    pub fn loss_and_grad_with(&self, w: &[f64], trial: &Trial, grad: &mut [f64]) -> f64 {
        let ro = self.forward_with(w, trial);
        let n = self.base.n_units;
        let (nin, nout) = (self.base.cfg.n_in, self.base.cfg.n_out);
        let nq = self.base.out_units.len();
        let l = &self.layout;
        let wout = &self.params[l.w_out..l.b_out];
        let t_len = trial.t;

        let denom = trial.mask.iter().sum::<f64>() * nout as f64;
        let norm = if denom > 0.0 { 1.0 / denom } else { 1.0 };
        let mut loss = 0.0;
        let mut gh = vec![0.0; (t_len + 1) * n];
        for t in 0..t_len {
            for k in 0..nout {
                let e = ro.y[t * nout + k] - trial.targets[t * nout + k];
                loss += trial.mask[t] * e * e * norm;
                let dy = 2.0 * trial.mask[t] * e * norm;
                if dy == 0.0 {
                    continue;
                }
                grad[l.b_out + k] += dy;
                for (q, &u) in self.base.out_units.iter().enumerate() {
                    grad[l.w_out + k * nq + q] += dy * ro.h[(t + 1) * n + u];
                    gh[(t + 1) * n + u] += dy * wout[k * nq + q];
                }
            }
        }

        // dw/dtheta = sign(pre) * sigmoid(theta)
        let ne = self.base.edges.len();
        let mut dw = vec![0.0; ne * 4];
        for (ei, e) in self.base.edges.iter().enumerate() {
            let s = self.sign(e.pre);
            for k in 0..4 {
                dw[ei * 4 + k] = s * sigmoid(self.params[l.theta + ei * 4 + k]);
            }
        }

        let mut gc = vec![0.0; n]; // gradient reaching c[s] from step s+1
        let mut dz = vec![0.0; n * 4];
        for s in (1..=t_len).rev() {
            let t = s - 1; // the step that produced c[s], h[s]
            for u in 0..n {
                let o = (t * n + u) * 4;
                let (gi, gf, go, gg) = (ro.gates[o + GI], ro.gates[o + GF], ro.gates[o + GO], ro.gates[o + GG]);
                let th = ro.c[s * n + u].tanh();
                let dh = gh[s * n + u];
                let dc = gc[u] + dh * go * (1.0 - th * th);
                let d_o = dh * th;
                let d_i = dc * gg;
                let d_g = dc * gi;
                let d_f = dc * ro.c[t * n + u];
                gc[u] = dc * gf;
                dz[u * 4 + GI] = d_i * gi * (1.0 - gi);
                dz[u * 4 + GF] = d_f * gf * (1.0 - gf);
                dz[u * 4 + GO] = d_o * go * (1.0 - go);
                dz[u * 4 + GG] = d_g * (1.0 - gg * gg);
                for k in 0..4 {
                    grad[l.b + u * 4 + k] += dz[u * 4 + k];
                }
            }
            for (kk, &u) in self.base.in_units.iter().enumerate() {
                for ch in 0..nin {
                    let x = trial.inputs[t * nin + ch];
                    let o = l.w_in + (kk * nin + ch) * 4;
                    for k in 0..4 {
                        grad[o + k] += dz[u * 4 + k] * x;
                    }
                }
            }
            for (ei, e) in self.base.edges.iter().enumerate() {
                if t >= e.delay {
                    let sp = t - e.delay;
                    let hv = ro.h[sp * n + e.pre];
                    let mut back = 0.0;
                    for k in 0..4 {
                        let g = dz[e.post * 4 + k];
                        grad[l.theta + ei * 4 + k] += g * hv * dw[ei * 4 + k];
                        back += w[ei * 4 + k] * g;
                    }
                    gh[sp * n + e.pre] += back;
                }
            }
        }
        loss
    }

    pub fn batch_loss_and_grad(&self, trials: &[Trial], grad: &mut [f64], threads: usize) -> f64 {
        grad.iter_mut().for_each(|g| *g = 0.0);
        if trials.is_empty() {
            return 0.0;
        }
        let w = self.effective_weights();
        let np = self.params.len();
        let nthreads = threads.max(1).min(trials.len());
        let chunk = trials.len().div_ceil(nthreads);
        let results: Vec<(f64, Vec<f64>)> = std::thread::scope(|s| {
            let handles: Vec<_> = trials
                .chunks(chunk)
                .map(|ch| {
                    let w = &w;
                    s.spawn(move || {
                        let mut g = vec![0.0; np];
                        let mut l = 0.0;
                        for tr in ch {
                            l += self.loss_and_grad_with(w, tr, &mut g);
                        }
                        (l, g)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("worker thread panicked")).collect()
        });
        let inv = 1.0 / trials.len() as f64;
        let mut loss = 0.0;
        for (l, g) in results {
            loss += l;
            for i in 0..np {
                grad[i] += g[i];
            }
        }
        grad.iter_mut().for_each(|g| *g *= inv);
        loss * inv
    }

    pub fn save_params(&self, path: &Path) -> Result<(), String> {
        let mut buf = Vec::with_capacity(16 + 8 * self.params.len());
        buf.extend_from_slice(b"HCLSTM01");
        buf.extend_from_slice(&(self.params.len() as u64).to_le_bytes());
        for p in &self.params {
            buf.extend_from_slice(&p.to_le_bytes());
        }
        fs::write(path, buf).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn load_params(&mut self, path: &Path) -> Result<(), String> {
        let buf = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if buf.len() < 16 || &buf[0..8] != b"HCLSTM01" {
            return Err("not an HCLSTM parameter file".into());
        }
        let n = u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize;
        if n != self.params.len() || buf.len() != 16 + 8 * n {
            return Err(format!("parameter count mismatch: file has {n}, network has {}", self.params.len()));
        }
        for i in 0..n {
            self.params[i] = f64::from_le_bytes(buf[16 + 8 * i..24 + 8 * i].try_into().unwrap());
        }
        Ok(())
    }

    /// Verify, for every gate, Dale's law and the connectome mask on the realised weights.
    pub fn check_constraints(&self) -> Result<(), String> {
        let r = self.base.n_regions;
        let w = self.effective_weights();
        for (ei, e) in self.base.edges.iter().enumerate() {
            for k in 0..4 {
                let v = w[ei * 4 + k];
                if v == 0.0 || !v.is_finite() {
                    return Err(format!("gate {k}: edge {ei} has degenerate weight {v}"));
                }
                if (e.pre < r && v < 0.0) || (e.pre >= r && v > 0.0) {
                    return Err(format!("gate {k}: Dale's law violated at post {} pre {}", e.post, e.pre));
                }
            }
            let (rp, rq) = (e.post % r, e.pre % r);
            if rp != rq {
                if e.pre >= r {
                    return Err(format!("inhibitory long-range projection {} -> {}", e.pre, e.post));
                }
                if !self.base.allowed[rp * r + rq] {
                    return Err(format!("weight outside the wiring mask: region {rq} -> {rp}"));
                }
            }
        }
        Ok(())
    }
}

impl SequenceModel for ConnectomeLstm {
    fn params(&self) -> &[f64] {
        &self.params
    }
    fn params_mut(&mut self) -> &mut [f64] {
        &mut self.params
    }
    fn batch_loss_and_grad(&self, trials: &[Trial], grad: &mut [f64], threads: usize) -> f64 {
        ConnectomeLstm::batch_loss_and_grad(self, trials, grad, threads)
    }
    fn outputs(&self, trial: &Trial) -> Vec<f64> {
        self.forward(trial).y
    }
    fn e_activity_at(&self, trial: &Trial, step: usize) -> Vec<f64> {
        let (n, r) = (self.base.n_units, self.base.n_regions);
        let ro = self.forward(trial);
        ro.h[step * n..step * n + r].to_vec()
    }
    fn region_systems(&self) -> &[String] {
        &self.base.region_systems
    }
    fn check_constraints(&self) -> Result<(), String> {
        ConnectomeLstm::check_constraints(self)
    }
    fn save_to_dir(&self, dir: &Path) -> Result<String, String> {
        self.save_params(&dir.join("params_lstm.bin"))?;
        Ok("params_lstm.bin".into())
    }
}
