//! Connectome-constrained recurrent neural network.
//!
//! Every brain region is modelled as an excitatory (E) and an inhibitory (I)
//! population, so a connectome with R regions gives N = 2R units
//! (units 0..R are E, units R..2R are I; region of unit u is u % R).
//!
//! Constraints imposed by construction (there is no mask to forget to apply:
//! the set of trainable synapses *is* the set of allowed connections):
//!
//! * Inter-regional synapses exist only where the connectome has a tract, and
//!   originate from E units (long-range projections are excitatory); they target
//!   both E and I units of the receiving region (feed-forward inhibition).
//! * Each region has a full local E/I microcircuit (EE, EI, IE, II).
//! * Dale's law: the sign of every synapse is fixed by its presynaptic unit;
//!   learning only changes magnitudes, w = sign * softplus(theta).
//! * Inter-regional conduction delays come from fibre length / velocity.
//! * Input enters only through E units of designated input regions; the readout
//!   reads only E units of designated output regions.
//!
//! Dynamics (leaky rate network, Euler step dt, time constant tau, alpha = dt/tau):
//!   a_t     = b + W_in u_t + sum_e w_e * r_{t - delay_e}[pre_e]   (scattered to post_e)
//!   x_{t+1} = (1 - alpha) x_t + alpha * a_t,       r = tanh(x)
//!   y_t     = W_out r_{t+1}[out_units] + b_out
//! Gradients are exact BPTT through the delayed recurrence.

use crate::connectome::Connectome;
use crate::spectral;
use crate::topology;
use crate::rng::Rng;
use crate::task::Trial;
use std::fs;
use std::io::Write;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskMode {
    /// Wiring taken from the connectome.
    Connectome,
    /// Null: same degree sequence and weight multiset, rewired at random.
    DegreePreserving,
    /// Null: random graph with the same number of edges.
    DensityMatched,
    /// Control: all-to-all between regions, same total strength.
    Dense,
}

impl MaskMode {
    pub const ALL: [MaskMode; 4] =
        [MaskMode::Connectome, MaskMode::DegreePreserving, MaskMode::DensityMatched, MaskMode::Dense];

    pub fn name(&self) -> &'static str {
        match self {
            MaskMode::Connectome => "connectome",
            MaskMode::DegreePreserving => "degree-preserving",
            MaskMode::DensityMatched => "density-matched",
            MaskMode::Dense => "dense",
        }
    }

    pub fn parse(s: &str) -> Option<MaskMode> {
        match s.to_lowercase().as_str() {
            "connectome" | "real" => Some(MaskMode::Connectome),
            "degree-preserving" | "shuffled" | "rewired" => Some(MaskMode::DegreePreserving),
            "density-matched" | "random" => Some(MaskMode::DensityMatched),
            "dense" => Some(MaskMode::Dense),
            _ => None,
        }
    }
}

/// How the initial recurrent strength is set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Calibration {
    /// v0.1: power iteration on the instantaneous matrix W, scaled to `gain`.
    Power,
    /// Arnoldi (krylov_ds) on the delayed, leaky linearised map, scaled so its
    /// spectral radius equals `target_radius`.
    Arnoldi,
}

#[derive(Clone, Debug)]
pub struct RnnConfig {
    pub mode: MaskMode,
    pub dt_ms: f64,
    pub tau_ms: f64,
    /// Axonal conduction velocity in mm/ms (= m/s).
    pub velocity_mm_per_ms: f64,
    pub use_delays: bool,
    pub calibration: Calibration,
    /// Power calibration: target spectral radius of the initial matrix W.
    pub gain: f64,
    /// Arnoldi calibration: target radius of the delayed linearised map.
    pub target_radius: f64,
    pub krylov_dim: usize,
    pub local_ee: f64,
    pub local_ei: f64,
    pub local_ie: f64,
    pub local_ii: f64,
    /// Scale of inter-regional weights relative to local ones (mean node strength = 1).
    pub inter_gain: f64,
    /// E->I inter-regional strength relative to E->E.
    pub ei_ratio: f64,
    pub n_in: usize,
    pub n_out: usize,
    pub input_regions: Vec<usize>,
    pub output_regions: Vec<usize>,
    pub seed: u64,
}

impl Default for RnnConfig {
    fn default() -> Self {
        RnnConfig {
            mode: MaskMode::Connectome,
            dt_ms: 5.0,
            tau_ms: 20.0,
            velocity_mm_per_ms: 5.0,
            use_delays: true,
            calibration: Calibration::Power,
            gain: 0.9,
            target_radius: 0.90,
            krylov_dim: 80,
            local_ee: 1.0,
            local_ei: 1.0,
            local_ie: 1.5,
            local_ii: 0.5,
            inter_gain: 1.0,
            ei_ratio: 0.5,
            n_in: 3,
            n_out: 2,
            input_regions: Vec::new(),
            output_regions: Vec::new(),
            seed: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeKind {
    LocalEE,
    LocalEI,
    LocalIE,
    LocalII,
    InterEE,
    InterEI,
}

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub post: usize,
    pub pre: usize,
    pub delay: usize,
    pub kind: EdgeKind,
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub theta: usize,
    pub w_in: usize,
    pub b: usize,
    pub w_out: usize,
    pub b_out: usize,
    pub total: usize,
}

pub struct Rollout {
    pub t: usize,
    /// (t+1) x N membrane states, row 0 = initial state (zeros)
    pub x: Vec<f64>,
    /// (t+1) x N firing rates
    pub r: Vec<f64>,
    /// t x n_out outputs
    pub y: Vec<f64>,
}

pub struct ConnectomeRnn {
    pub cfg: RnnConfig,
    pub n_regions: usize,
    pub n_units: usize,
    pub edges: Vec<Edge>,
    pub in_units: Vec<usize>,
    pub out_units: Vec<usize>,
    pub layout: Layout,
    pub params: Vec<f64>,
    pub alpha: f64,
    /// allowed[i * R + j] is true if region i may receive from region j.
    pub allowed: Vec<bool>,
    pub region_names: Vec<String>,
    pub region_systems: Vec<String>,
    /// Normalised Laplacian (R x R, row-major) of the wiring graph actually used.
    pub laplacian: Vec<f64>,
    /// Weight of the Dirichlet-energy (Rayleigh quotient) regulariser on E rates.
    pub energy_lambda: f64,
}

/// Floor in the Rayleigh quotient denominator so near-silent states are not penalised noisily.
pub const ENERGY_EPS: f64 = 1e-2;

fn softplus(x: f64) -> f64 {
    if x > 30.0 { x } else { x.exp().ln_1p() }
}
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
/// Inverse of softplus for m > 0.
fn softplus_inv(m: f64) -> f64 {
    if m > 30.0 { m } else { m + (-(-m).exp_m1()).ln() }
}

pub fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Pick default input (visual) and output (somatomotor) regions from system labels.
pub fn default_roles(conn: &Connectome) -> Result<(Vec<usize>, Vec<usize>), String> {
    let mut inp = Vec::new();
    let mut out = Vec::new();
    for (i, r) in conn.regions.iter().enumerate() {
        let s = r.system.to_lowercase();
        if s.contains("vis") {
            inp.push(i);
        }
        if s.contains("somat") || s.contains("sommot") || s.contains("motor") {
            out.push(i);
        }
    }
    if inp.is_empty() || out.is_empty() {
        return Err(
            "could not infer input/output regions from system labels; pass --input-regions and --output-regions"
                .into(),
        );
    }
    Ok((inp, out))
}

fn spectral_radius(edges: &[Edge], w: &[f64], n: usize, rng: &mut Rng) -> f64 {
    let mut v: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.iter_mut().for_each(|x| *x /= nrm);
    let (mut acc, mut cnt) = (0.0, 0);
    let mut u = vec![0.0; n];
    for it in 0..400 {
        u.iter_mut().for_each(|x| *x = 0.0);
        for (e, &we) in edges.iter().zip(w) {
            u[e.post] += we * v[e.pre];
        }
        let nrm = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nrm < 1e-300 {
            return 0.0;
        }
        if it >= 150 {
            acc += nrm.ln();
            cnt += 1;
        }
        for i in 0..n {
            v[i] = u[i] / nrm;
        }
    }
    (acc / cnt as f64).exp()
}

impl ConnectomeRnn {
    pub fn new(conn: &Connectome, cfg: RnnConfig) -> Result<Self, String> {
        let r = conn.n();
        if cfg.input_regions.is_empty() || cfg.output_regions.is_empty() {
            return Err("input_regions and output_regions must not be empty".into());
        }
        if cfg.input_regions.iter().chain(&cfg.output_regions).any(|&i| i >= r) {
            return Err("input/output region index out of range".into());
        }
        if cfg.n_in == 0 || cfg.n_out == 0 {
            return Err("n_in and n_out must be positive".into());
        }
        if cfg.dt_ms <= 0.0 || cfg.tau_ms <= 0.0 || cfg.velocity_mm_per_ms <= 0.0 {
            return Err("dt, tau and velocity must be positive".into());
        }
        let mut rng = Rng::new(cfg.seed ^ 0x5EED_C0DE);
        let base = conn.scaled_to_mean_strength(1.0);
        let g = match cfg.mode {
            MaskMode::Connectome => base.clone(),
            MaskMode::DegreePreserving => base.degree_preserving_rewire(&mut rng, 20),
            MaskMode::DensityMatched => base.density_matched_random(&mut rng),
            MaskMode::Dense => base.dense_uniform(),
        };

        let n = 2 * r;
        let delay_of = |len_mm: f64| -> usize {
            if cfg.use_delays {
                ((len_mm / (cfg.velocity_mm_per_ms * cfg.dt_ms)).round() as usize).max(1)
            } else {
                0
            }
        };

        let mut edges = Vec::new();
        let mut mags = Vec::new();
        let mut allowed = vec![false; r * r];
        for i in 0..r {
            allowed[i * r + i] = true;
            let (e, inh) = (i, r + i);
            for (post, pre, kind, m) in [
                (e, e, EdgeKind::LocalEE, cfg.local_ee),
                (inh, e, EdgeKind::LocalEI, cfg.local_ei),
                (e, inh, EdgeKind::LocalIE, cfg.local_ie),
                (inh, inh, EdgeKind::LocalII, cfg.local_ii),
            ] {
                edges.push(Edge { post, pre, delay: 0, kind });
                mags.push(m.max(1e-6));
            }
        }
        for i in 0..r {
            for j in 0..r {
                if i == j {
                    continue;
                }
                let w = g.weight(i, j);
                if w <= 0.0 {
                    continue;
                }
                allowed[i * r + j] = true;
                let d = delay_of(conn.length(i, j));
                edges.push(Edge { post: i, pre: j, delay: d, kind: EdgeKind::InterEE });
                mags.push((cfg.inter_gain * w).max(1e-6));
                edges.push(Edge { post: r + i, pre: j, delay: d, kind: EdgeKind::InterEI });
                mags.push((cfg.inter_gain * cfg.ei_ratio * w).max(1e-6));
            }
        }
        let sign: Vec<f64> = (0..n).map(|u| if u < r { 1.0 } else { -1.0 }).collect();

        let alpha = cfg.dt_ms / cfg.tau_ms;
        let signed: Vec<f64> = edges.iter().zip(&mags).map(|(e, &m)| sign[e.pre] * m).collect();
        match cfg.calibration {
            Calibration::Power => {
                let rho = spectral_radius(&edges, &signed, n, &mut rng);
                if rho > 1e-9 {
                    let f = cfg.gain / rho;
                    mags.iter_mut().for_each(|m| *m = (*m * f).max(1e-9));
                }
            }
            Calibration::Arnoldi => {
                let c = spectral::calibrate_scale(&edges, &signed, n, alpha, cfg.target_radius, cfg.krylov_dim, cfg.seed)?;
                mags.iter_mut().for_each(|m| *m = (*m * c).max(1e-9));
            }
        }
        let laplacian = topology::sym_laplacian(&g);

        let in_units: Vec<usize> = cfg.input_regions.clone();
        let out_units: Vec<usize> = cfg.output_regions.clone();
        let (ne, nq) = (edges.len(), out_units.len());
        let layout = {
            let theta = 0;
            let w_in = theta + ne;
            let b = w_in + in_units.len() * cfg.n_in;
            let w_out = b + n;
            let b_out = w_out + cfg.n_out * nq;
            Layout { theta, w_in, b, w_out, b_out, total: b_out + cfg.n_out }
        };
        let mut params = vec![0.0; layout.total];
        for (k, &m) in mags.iter().enumerate() {
            params[layout.theta + k] = softplus_inv(m);
        }
        let s_in = 1.0 / (cfg.n_in as f64).sqrt();
        for k in 0..in_units.len() * cfg.n_in {
            params[layout.w_in + k] = s_in * rng.normal();
        }
        let s_out = 1.0 / (nq as f64).sqrt();
        for k in 0..cfg.n_out * nq {
            params[layout.w_out + k] = s_out * rng.normal();
        }

        Ok(ConnectomeRnn {
            alpha: cfg.dt_ms / cfg.tau_ms,
            n_regions: r,
            n_units: n,
            edges,
            in_units,
            out_units,
            layout,
            params,
            allowed,
            region_names: conn.regions.iter().map(|x| x.name.clone()).collect(),
            region_systems: conn.regions.iter().map(|x| x.system.clone()).collect(),
            laplacian,
            energy_lambda: 0.0,
            cfg,
        })
    }

    pub fn num_params(&self) -> usize {
        self.params.len()
    }
    pub fn region_of(&self, unit: usize) -> usize {
        unit % self.n_regions
    }
    pub fn is_excitatory(&self, unit: usize) -> bool {
        unit < self.n_regions
    }
    fn sign(&self, unit: usize) -> f64 {
        if self.is_excitatory(unit) { 1.0 } else { -1.0 }
    }
    pub fn max_delay(&self) -> usize {
        self.edges.iter().map(|e| e.delay).max().unwrap_or(0)
    }

    /// Effective synaptic weights per edge: sign(pre) * softplus(theta).
    pub fn effective_weights(&self) -> Vec<f64> {
        self.edges
            .iter()
            .enumerate()
            .map(|(k, e)| self.sign(e.pre) * softplus(self.params[self.layout.theta + k]))
            .collect()
    }

    /// Dense N x N recurrent matrix (row = post, column = pre).
    pub fn dense_weights(&self) -> Vec<f64> {
        let n = self.n_units;
        let w = self.effective_weights();
        let mut m = vec![0.0; n * n];
        for (e, &we) in self.edges.iter().zip(&w) {
            m[e.post * n + e.pre] += we;
        }
        m
    }

    /// Verify Dale's law and the connectome mask on the realised weights.
    pub fn check_constraints(&self) -> Result<(), String> {
        let (n, r) = (self.n_units, self.n_regions);
        let m = self.dense_weights();
        for post in 0..n {
            for pre in 0..n {
                let v = m[post * n + pre];
                if v == 0.0 {
                    continue;
                }
                if (pre < r && v < 0.0) || (pre >= r && v > 0.0) {
                    return Err(format!("Dale's law violated at post {post} pre {pre}"));
                }
                let (rp, rq) = (post % r, pre % r);
                if rp != rq {
                    if pre >= r {
                        return Err(format!("inhibitory long-range projection {pre} -> {post}"));
                    }
                    if !self.allowed[rp * r + rq] {
                        return Err(format!("weight outside the wiring mask: region {rq} -> {rp}"));
                    }
                }
            }
        }
        Ok(())
    }

    /// Radius of the instantaneous recurrent matrix W (power iteration).
    pub fn weight_radius(&self) -> f64 {
        let mut rng = Rng::new(7);
        spectral_radius(&self.edges, &self.effective_weights(), self.n_units, &mut rng)
    }

    /// Radius of the delayed, leaky linearised dynamics (Arnoldi via krylov_ds).
    pub fn spectral_radius(&self) -> f64 {
        spectral::delayed_spectral_radius(&self.edges, &self.effective_weights(), self.n_units, self.alpha, self.cfg.krylov_dim, 7)
    }

    /// Multiply every recurrent synaptic magnitude by `c` (signs unchanged).
    pub fn scale_recurrent(&mut self, c: f64) {
        for k in 0..self.edges.len() {
            let m = softplus(self.params[self.layout.theta + k]);
            self.params[self.layout.theta + k] = softplus_inv((c * m).max(1e-9));
        }
    }

    /// Stability projection: if the delayed radius exceeds `cap`, rescale recurrent
    /// weights so it equals `cap`. Returns the radius before projection.
    pub fn enforce_radius_cap(&mut self, cap: f64) -> f64 {
        let w = self.effective_weights();
        let r = spectral::delayed_spectral_radius(&self.edges, &w, self.n_units, self.alpha, self.cfg.krylov_dim, 7);
        if r.is_finite() && r > cap {
            if let Ok(c) = spectral::calibrate_scale(&self.edges, &w, self.n_units, self.alpha, cap, self.cfg.krylov_dim, 7) {
                self.scale_recurrent(c);
            }
        }
        r
    }

    /// Rayleigh quotient r^T L r / (r^T r + eps) of the E rates at one time index.
    fn rayleigh(&self, ro: &Rollout, s: usize) -> (f64, Vec<f64>, f64) {
        let (n, r) = (self.n_units, self.n_regions);
        let re = &ro.r[s * n..s * n + r];
        let mut lr = vec![0.0; r];
        for i in 0..r {
            let row = &self.laplacian[i * r..(i + 1) * r];
            lr[i] = row.iter().zip(re).map(|(a, b)| a * b).sum();
        }
        let num: f64 = re.iter().zip(&lr).map(|(a, b)| a * b).sum();
        let den: f64 = re.iter().map(|x| x * x).sum::<f64>() + ENERGY_EPS;
        (num / den, lr, den)
    }

    /// Mean-over-time Rayleigh energy of the E rates (weight 1; multiply by lambda for the penalty).
    pub fn mean_energy(&self, ro: &Rollout) -> f64 {
        if ro.t == 0 {
            return 0.0;
        }
        (1..=ro.t).map(|s| self.rayleigh(ro, s).0).sum::<f64>() / ro.t as f64
    }

    /// Task loss plus the energy penalty: the objective that `loss_and_grad_with` differentiates.
    pub fn objective(&self, ro: &Rollout, trial: &Trial) -> f64 {
        self.trial_loss(ro, trial) + self.energy_lambda * self.mean_energy(ro)
    }

    pub fn forward(&self, trial: &Trial) -> Rollout {
        self.forward_with(&self.effective_weights(), trial)
    }

    pub fn forward_with(&self, w: &[f64], trial: &Trial) -> Rollout {
        let n = self.n_units;
        let (nin, nout) = (self.cfg.n_in, self.cfg.n_out);
        let nq = self.out_units.len();
        let l = &self.layout;
        let win = &self.params[l.w_in..l.b];
        let b = &self.params[l.b..l.w_out];
        let wout = &self.params[l.w_out..l.b_out];
        let bout = &self.params[l.b_out..l.total];
        let t_len = trial.t;
        let mut x = vec![0.0; (t_len + 1) * n];
        let mut r = vec![0.0; (t_len + 1) * n];
        let mut y = vec![0.0; t_len * nout];
        let mut a = vec![0.0; n];
        let al = self.alpha;
        for t in 0..t_len {
            a.copy_from_slice(b);
            for (kk, &u) in self.in_units.iter().enumerate() {
                let mut s = 0.0;
                for c in 0..nin {
                    s += win[kk * nin + c] * trial.inputs[t * nin + c];
                }
                a[u] += s;
            }
            for (ei, e) in self.edges.iter().enumerate() {
                if t >= e.delay {
                    a[e.post] += w[ei] * r[(t - e.delay) * n + e.pre];
                }
            }
            for i in 0..n {
                let xn = (1.0 - al) * x[t * n + i] + al * a[i];
                x[(t + 1) * n + i] = xn;
                r[(t + 1) * n + i] = xn.tanh();
            }
            for k in 0..nout {
                let mut s = bout[k];
                for (q, &u) in self.out_units.iter().enumerate() {
                    s += wout[k * nq + q] * r[(t + 1) * n + u];
                }
                y[t * nout + k] = s;
            }
        }
        Rollout { t: t_len, x, r, y }
    }

    pub fn trial_loss(&self, ro: &Rollout, trial: &Trial) -> f64 {
        let nout = self.cfg.n_out;
        let denom = trial.mask.iter().sum::<f64>() * nout as f64;
        let norm = if denom > 0.0 { 1.0 / denom } else { 1.0 };
        let mut loss = 0.0;
        for t in 0..trial.t {
            for k in 0..nout {
                let e = ro.y[t * nout + k] - trial.targets[t * nout + k];
                loss += trial.mask[t] * e * e * norm;
            }
        }
        loss
    }

    /// Loss for one trial; accumulates dLoss/dparams into `grad`.
    pub fn loss_and_grad_with(&self, w: &[f64], trial: &Trial, grad: &mut [f64]) -> f64 {
        let ro = self.forward_with(w, trial);
        let n = self.n_units;
        let (nin, nout) = (self.cfg.n_in, self.cfg.n_out);
        let nq = self.out_units.len();
        let l = &self.layout;
        let wout = &self.params[l.w_out..l.b_out];
        let t_len = trial.t;
        let al = self.alpha;

        let denom = trial.mask.iter().sum::<f64>() * nout as f64;
        let norm = if denom > 0.0 { 1.0 / denom } else { 1.0 };
        let mut loss = 0.0;
        let mut gr = vec![0.0; (t_len + 1) * n];
        for t in 0..t_len {
            for k in 0..nout {
                let e = ro.y[t * nout + k] - trial.targets[t * nout + k];
                loss += trial.mask[t] * e * e * norm;
                let dy = 2.0 * trial.mask[t] * e * norm;
                if dy == 0.0 {
                    continue;
                }
                grad[l.b_out + k] += dy;
                for (q, &u) in self.out_units.iter().enumerate() {
                    grad[l.w_out + k * nq + q] += dy * ro.r[(t + 1) * n + u];
                    gr[(t + 1) * n + u] += dy * wout[k * nq + q];
                }
            }
        }

        if self.energy_lambda > 0.0 && t_len > 0 {
            let lam = self.energy_lambda / t_len as f64;
            for s in 1..=t_len {
                let (q, lr, den) = self.rayleigh(&ro, s);
                loss += lam * q;
                let re = &ro.r[s * n..s * n + self.n_regions];
                for i in 0..self.n_regions {
                    gr[s * n + i] += lam * 2.0 * (lr[i] - q * re[i]) / den;
                }
            }
        }

        // dw/dtheta = sign * sigmoid(theta)
        let dw: Vec<f64> = self
            .edges
            .iter()
            .enumerate()
            .map(|(k, e)| self.sign(e.pre) * sigmoid(self.params[l.theta + k]))
            .collect();

        let mut gx_next = vec![0.0; n];
        let mut gx = vec![0.0; n];
        let mut ga = vec![0.0; n];
        for s in (1..=t_len).rev() {
            for i in 0..n {
                let rv = ro.r[s * n + i];
                gx[i] = gr[s * n + i] * (1.0 - rv * rv) + (1.0 - al) * gx_next[i];
                ga[i] = al * gx[i];
                grad[l.b + i] += ga[i];
            }
            let t = s - 1; // the step that produced x_s
            for (kk, &u) in self.in_units.iter().enumerate() {
                for c in 0..nin {
                    grad[l.w_in + kk * nin + c] += ga[u] * trial.inputs[t * nin + c];
                }
            }
            for (ei, e) in self.edges.iter().enumerate() {
                if t >= e.delay {
                    let sp = t - e.delay;
                    let g = ga[e.post];
                    grad[l.theta + ei] += g * ro.r[sp * n + e.pre] * dw[ei];
                    gr[sp * n + e.pre] += w[ei] * g;
                }
            }
            std::mem::swap(&mut gx, &mut gx_next);
        }
        loss
    }

    /// Mean loss and gradient over a batch (multi-threaded with std threads).
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

    // ------------------------------------------------------------------
    // Persistence / export
    // ------------------------------------------------------------------

    pub fn save_params(&self, path: &Path) -> Result<(), String> {
        let mut buf = Vec::with_capacity(16 + 8 * self.params.len());
        buf.extend_from_slice(b"HCRNN001");
        buf.extend_from_slice(&(self.params.len() as u64).to_le_bytes());
        for p in &self.params {
            buf.extend_from_slice(&p.to_le_bytes());
        }
        fs::write(path, buf).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Load parameters saved from a network built with the same connectome and config.
    pub fn load_params(&mut self, path: &Path) -> Result<(), String> {
        let buf = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if buf.len() < 16 || &buf[0..8] != b"HCRNN001" {
            return Err("not an HCRNN parameter file".into());
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

    /// Write the effective N x N recurrent matrix (rows = post, cols = pre) as CSV,
    /// with unit labels in the first row/column.
    pub fn export_weights_csv(&self, path: &Path) -> Result<(), String> {
        let n = self.n_units;
        let m = self.dense_weights();
        let label = |u: usize| format!("{}_{}", self.region_names[self.region_of(u)], if self.is_excitatory(u) { "E" } else { "I" });
        let mut f = fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut line = String::from("post\\pre");
        for u in 0..n {
            line.push(',');
            line.push_str(&label(u));
        }
        writeln!(f, "{line}").map_err(|e| e.to_string())?;
        for i in 0..n {
            let mut line = label(i);
            for j in 0..n {
                line.push_str(&format!(",{:.6}", m[i * n + j]));
            }
            writeln!(f, "{line}").map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}
