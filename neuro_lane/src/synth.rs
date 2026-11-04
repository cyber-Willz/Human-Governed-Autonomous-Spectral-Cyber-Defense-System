//! Synthetic per-host behaviour sequences in feature space.
//!
//! These define the detection problem the model is trained on. They are
//! *assumptions about what behaviours look like*, not measured traffic.
//! The key design choice: [`Behavior::LowSlowScan`] looks innocuous in any
//! single bin (low rate, few ports) and is only distinguishable from the
//! benign "health-check" hard negatives by *persistence over many bins*.

use crate::features::{FeatVec, N_FEAT};
use hcrnn::rng::Rng;
use hcrnn::Trial;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Behavior {
    Benign,
    /// Benign background plus a short (1-4 bin) run of quiet, empty-payload probes.
    HealthCheckBurst,
    /// A short, loud, many-port scan.
    BurstScan,
    /// Persistent, quiet, empty-payload probing on every bin.
    LowSlowScan,
    /// Persistent periodic small check-ins to one destination.
    Beacon,
    /// Persistent high-volume transfer to one destination.
    Exfil,
}

pub const ALL: [Behavior; 6] = [
    Behavior::Benign,
    Behavior::HealthCheckBurst,
    Behavior::BurstScan,
    Behavior::LowSlowScan,
    Behavior::Beacon,
    Behavior::Exfil,
];

impl Behavior {
    pub fn hostile(self) -> bool {
        matches!(self, Behavior::BurstScan | Behavior::LowSlowScan | Behavior::Beacon | Behavior::Exfil)
    }
}

/// Uniform in `[lo, hi]` widened by `0.05 * shift` each side, clamped to [0,1].
fn u(rng: &mut Rng, lo: f64, hi: f64, shift: f64) -> f64 {
    let w = 0.05 * shift;
    rng.range((lo - w).max(0.0), (hi + w).min(1.0).max(lo)).clamp(0.0, 1.0)
}

fn benign_bin(rng: &mut Rng, s: f64) -> FeatVec {
    if rng.uniform() < 0.4 {
        return [0.0; N_FEAT];
    }
    [u(rng, 0.15, 0.6, s), u(rng, 0.05, 0.35, s), u(rng, 0.05, 0.3, s), u(rng, 0.25, 0.8, s), u(rng, 0.0, 0.35, s), u(rng, 0.0, 0.3, s)]
}

fn probe_bin(rng: &mut Rng, s: f64) -> FeatVec {
    [u(rng, 0.05, 0.2, s), u(rng, 0.05, 0.15, s), u(rng, 0.08, 0.25, s), u(rng, 0.0, 0.1, s), u(rng, 0.8, 1.0, s), u(rng, 0.1, 0.5, s)]
}

/// One sequence of `t` bins plus the per-step (target, mask) described in
/// [`to_trial`]. `shift` > 0 widens every parameter range (used for the
/// shifted evaluation).
pub fn sequence(rng: &mut Rng, b: Behavior, t: usize, shift: f64) -> (Vec<FeatVec>, usize) {
    let mut v: Vec<FeatVec> = (0..t).map(|_| benign_bin(rng, shift)).collect();
    let mut onset = 0usize;
    match b {
        Behavior::Benign => {}
        Behavior::HealthCheckBurst => {
            let len = 1 + rng.below(4);
            let start = rng.below(t.saturating_sub(len).max(1));
            for k in start..(start + len).min(t) {
                v[k] = probe_bin(rng, shift);
            }
        }
        Behavior::BurstScan => {
            let len = 3 + rng.below(4);
            let start = rng.below(t.saturating_sub(len + 2).max(1));
            onset = start;
            for k in start..(start + len).min(t) {
                v[k] = [u(rng, 0.5, 0.9, shift), u(rng, 0.05, 0.3, shift), u(rng, 0.7, 1.0, shift), u(rng, 0.0, 0.1, shift), u(rng, 0.9, 1.0, shift), u(rng, 0.0, 0.4, shift)];
            }
        }
        Behavior::LowSlowScan => {
            for k in 0..t {
                v[k] = probe_bin(rng, shift);
            }
        }
        Behavior::Beacon => {
            for k in 0..t {
                v[k] = [u(rng, 0.05, 0.15, shift), u(rng, 0.03, 0.08, shift), u(rng, 0.03, 0.1, shift), u(rng, 0.05, 0.2, shift), u(rng, 0.0, 0.3, shift), u(rng, 0.8, 1.0, shift)];
            }
        }
        Behavior::Exfil => {
            for k in 0..t {
                v[k] = [u(rng, 0.5, 0.9, shift), u(rng, 0.03, 0.08, shift), u(rng, 0.03, 0.1, shift), u(rng, 0.8, 1.0, shift), u(rng, 0.0, 0.1, shift), u(rng, 0.3, 0.7, shift)];
            }
        }
    }
    (v, onset)
}

/// Builds a training `Trial`. Target is 1 (hostile) / 0 (benign) with the
/// first `warm` steps masked out, and for the burst scan also everything
/// before 3 bins into the burst (evidence does not exist yet).
pub fn to_trial(seq: &[FeatVec], label: f64, onset: usize, warm: usize) -> Trial {
    let t = seq.len();
    let mut inputs = Vec::with_capacity(t * N_FEAT);
    for f in seq {
        inputs.extend_from_slice(f);
    }
    let first = warm.max(if label > 0.5 { onset + 3 } else { 0 });
    let mask: Vec<f64> = (0..t).map(|k| if k >= first { 1.0 } else { 0.0 }).collect();
    Trial { t, n_in: N_FEAT, n_out: 1, inputs, targets: vec![label; t], mask, angle: 0.0, cue_end: 0, go: t }
}

pub fn sample_trial(rng: &mut Rng, t: usize, shift: f64, warm: usize) -> (Trial, Behavior) {
    let b = ALL[rng.below(ALL.len())];
    let (seq, onset) = sequence(rng, b, t, shift);
    (to_trial(&seq, if b.hostile() { 1.0 } else { 0.0 }, onset, warm), b)
}
