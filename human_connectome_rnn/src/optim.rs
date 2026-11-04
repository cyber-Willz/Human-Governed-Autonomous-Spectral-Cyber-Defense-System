//! Adam optimiser with global-norm gradient clipping.

pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub clip: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    t: u64,
}

impl Adam {
    pub fn new(n: usize, lr: f64, clip: f64) -> Self {
        Adam { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, clip, m: vec![0.0; n], v: vec![0.0; n], t: 0 }
    }

    /// Apply one update; returns the pre-clipping gradient norm.
    pub fn step(&mut self, params: &mut [f64], grad: &[f64]) -> f64 {
        let norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        let scale = if self.clip > 0.0 && norm > self.clip { self.clip / norm } else { 1.0 };
        self.t += 1;
        let b1t = 1.0 - self.beta1.powi(self.t as i32);
        let b2t = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..params.len() {
            let g = grad[i] * scale;
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let mh = self.m[i] / b1t;
            let vh = self.v[i] / b2t;
            params[i] -= self.lr * mh / (vh.sqrt() + self.eps);
        }
        norm
    }
}
