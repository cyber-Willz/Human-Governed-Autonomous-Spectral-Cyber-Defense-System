//! Krylov-based stability analysis of the delayed recurrent dynamics.
//!
//! Linearised around r = 0 (tanh'(0) = 1) the network is the delay system
//!
//!   x_{t+1} = (1 - alpha) x_t + alpha * sum_e w_e * x_{t - d_e}[pre_e]  (scattered to post_e)
//!
//! Stacking the last D+1 states (D = longest delay) turns it into an ordinary
//! linear map on R^{N (D+1)} whose spectral radius decides local stability:
//! rho < 1 decays, rho > 1 grows until tanh saturates (the regime that holds
//! persistent memory). The map is applied matrix-free and its dominant
//! eigenvalue is found with `krylov_ds::Arnoldi`.
//!
//! This is more faithful than the radius of the instantaneous matrix W alone,
//! because it includes the leak and every conduction delay.

use crate::network::Edge;
use krylov_ds::{eig, Arnoldi, LinearOperator};

pub struct DelayedRecurrence<'a> {
    pub edges: &'a [Edge],
    pub w: &'a [f64],
    pub n: usize,
    pub alpha: f64,
    pub d_max: usize,
}

impl<'a> DelayedRecurrence<'a> {
    pub fn new(edges: &'a [Edge], w: &'a [f64], n: usize, alpha: f64) -> Self {
        let d_max = edges.iter().map(|e| e.delay).max().unwrap_or(0);
        DelayedRecurrence { edges, w, n, alpha, d_max }
    }
}

impl LinearOperator<f64> for DelayedRecurrence<'_> {
    fn dim(&self) -> usize {
        self.n * (self.d_max + 1)
    }

    fn apply(&self, s: &[f64], out: &mut [f64]) {
        let n = self.n;
        // block k of s holds x_{t-k}
        for i in 0..n {
            out[i] = (1.0 - self.alpha) * s[i];
        }
        for (e, &we) in self.edges.iter().zip(self.w) {
            out[e.post] += self.alpha * we * s[e.delay * n + e.pre];
        }
        // shift history: new block k+1 = old block k
        if self.d_max > 0 {
            out[n..].copy_from_slice(&s[..n * self.d_max]);
        }
    }
}

fn start_vector(dim: usize, seed: u64) -> Vec<f64> {
    let mut rng = crate::rng::Rng::new(seed ^ 0xA2A1_D01D);
    let mut v: Vec<f64> = (0..dim).map(|_| rng.normal()).collect();
    let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.iter_mut().for_each(|x| *x /= nrm);
    v
}

/// Spectral radius of the delayed linearised map (Arnoldi, `krylov_dim` steps).
pub fn delayed_spectral_radius(
    edges: &[Edge],
    w: &[f64],
    n: usize,
    alpha: f64,
    krylov_dim: usize,
    seed: u64,
) -> f64 {
    let op = DelayedRecurrence::new(edges, w, n, alpha);
    let dim = op.dim();
    let m = krylov_dim.min(dim).max(1);
    let v0 = start_vector(dim, seed);
    match Arnoldi::new(m, 1e-12).run(&op, &v0) {
        Ok(res) => eig::arnoldi_ritz_values(&res).iter().map(|z| (z.re * z.re + z.im * z.im).sqrt()).fold(0.0, f64::max),
        Err(_) => f64::NAN,
    }
}

/// Find the scale c such that the delayed radius of `c * w` equals `target`.
/// The radius at c = 0 is the leak factor 1 - alpha, so `target` must exceed it.
pub fn calibrate_scale(
    edges: &[Edge],
    w: &[f64],
    n: usize,
    alpha: f64,
    target: f64,
    krylov_dim: usize,
    seed: u64,
) -> Result<f64, String> {
    if target <= 1.0 - alpha + 1e-6 {
        return Err(format!("target radius {target} must exceed the leak factor {:.4}", 1.0 - alpha));
    }
    let f = |c: f64| {
        let ws: Vec<f64> = w.iter().map(|x| x * c).collect();
        delayed_spectral_radius(edges, &ws, n, alpha, krylov_dim, seed)
    };
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    let mut grow = 0;
    while f(hi) < target {
        lo = hi;
        hi *= 2.0;
        grow += 1;
        if grow > 40 {
            return Err("could not bracket the target radius (network too weakly coupled?)".into());
        }
    }
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        let r = f(mid);
        if (r - target).abs() < 1e-5 {
            return Ok(mid);
        }
        if r < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}
