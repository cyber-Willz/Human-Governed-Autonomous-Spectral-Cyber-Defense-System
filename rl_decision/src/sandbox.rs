//! Sandbox simulator: a digital twin of "what happens if I do X to this host".
//!
//! Two uses:
//!  1. **Pre-training / anchoring** -- [`ScenarioGen`] produces labelled
//!     synthetic situations (attacks, and the benign look-alikes that make
//!     detection hard) so the policy and risk model start sensible and a
//!     fixed anchor set exists to detect harmful updates.
//!  2. **Look-ahead at decision time** -- [`Sandbox::q_values`] runs Monte
//!     Carlo rollouts of each candidate action, including dynamics the flat
//!     cost table lacks (the attacker returning when a short ban expires,
//!     repeat offenders returning more often, source rotation against
//!     weakly-evidenced bans).
//!
//! **Structurally isolated from enforcement**: this module imports nothing
//! from `rustwall` or the control plane and holds no handle to any real
//! resource. A simulated ban cannot touch a real firewall.
//!
//! Everything here is a *model*. Its parameters are hand-set, not fitted to
//! real traffic; see the crate docs for what that does and does not mean.

use serde::{Deserialize, Serialize};

use crate::action::{reward, Action, ALL_ACTIONS, N_ACTIONS};
use crate::rng::Rng;
use crate::state::{idx, StateFeatures, STATE_DIM};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxParams {
    /// Probability the attacker returns after a ban of each tier expires
    /// (index = `Action::index`; `Escalate` has no expiry).
    pub return_prob: [f64; N_ACTIONS],
    /// Multiplier on return probability per unit of `repeat_confirmed`.
    pub repeat_gain: f64,
    /// Fraction of the base miss loss charged when the attacker returns.
    pub return_cost: f64,
    /// Probability a ban is evaded by source rotation / spoofing when the
    /// evidence is a single lane and not a honeytoken.
    pub evasion_single_lane: f64,
    /// Fraction of base miss loss charged on evasion.
    pub evasion_cost: f64,
}

impl Default for SandboxParams {
    fn default() -> Self {
        Self {
            return_prob: [0.0, 0.55, 0.25, 0.08],
            repeat_gain: 0.8,
            return_cost: 0.4,
            evasion_single_lane: 0.30,
            evasion_cost: 0.5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rollout {
    pub malicious: bool,
    pub reward: f64,
}

#[derive(Debug, Clone, Default)]
pub struct Sandbox {
    pub params: SandboxParams,
}

impl Sandbox {
    pub fn new(params: SandboxParams) -> Self {
        Self { params }
    }

    /// One simulated outcome of `action` given the ground truth `malicious`.
    pub fn step(&self, s: &StateFeatures, action: Action, malicious: bool, rng: &mut Rng) -> Rollout {
        let mut r = reward(action, malicious, s.severity, s.protected);
        if malicious && action.is_ban() {
            let base_miss = -reward(Action::Escalate, true, s.severity, s.protected);
            // attacker returns once the ban lapses
            let repeat = s.x[idx::REPEAT_CONFIRMED];
            let p_ret = (self.params.return_prob[action.index()] * (1.0 + self.params.repeat_gain * repeat)).min(0.95);
            if rng.chance(p_ret) {
                r -= self.params.return_cost * base_miss;
            }
            // weakly-evidenced bans are easier to sidestep
            if !s.honeytoken && s.n_lanes < 2 && rng.chance(self.params.evasion_single_lane) {
                r -= self.params.evasion_cost * base_miss;
            }
        }
        Rollout { malicious, reward: r }
    }

    /// Monte-Carlo estimate of each action's expected reward, with common
    /// random numbers (the same malicious/benign draws are reused across
    /// actions so differences between actions are not buried in noise).
    pub fn q_values(&self, s: &StateFeatures, p_malicious: f64, n: usize, rng: &mut Rng) -> [f64; N_ACTIONS] {
        let n = n.max(1);
        let mut acc = [0.0; N_ACTIONS];
        for _ in 0..n {
            let malicious = rng.chance(p_malicious);
            // one seed per trial, replayed per action for common randomness
            let seed = rng.next_u64();
            for a in ALL_ACTIONS {
                let mut sub = Rng::new(seed);
                acc[a.index()] += self.step(s, a, malicious, &mut sub).reward;
            }
        }
        for v in acc.iter_mut() {
            *v /= n as f64;
        }
        acc
    }
}

/// Generator of labelled synthetic situations. Includes the benign
/// look-alikes (flash crowds, authorised scanners, periodic backups,
/// noisy infrastructure) that make the problem non-trivial.
#[derive(Debug, Clone)]
pub struct ScenarioGen {
    rng: Rng,
    /// Prior probability a generated case is malicious.
    pub p_malicious: f64,
}

impl ScenarioGen {
    pub fn new(seed: u64) -> Self {
        Self { rng: Rng::new(seed), p_malicious: 0.45 }
    }

    /// Draws `(state, is_malicious)`.
    pub fn sample(&mut self) -> (StateFeatures, bool) {
        let malicious = self.rng.chance(self.p_malicious);
        let mut x = [0.0f64; STATE_DIM];
        let (sev, lanes): (u8, u8);
        let r = &mut self.rng;
        if malicious {
            match r.categorical(&[1.0, 3.0, 2.5, 2.0, 1.5]) {
                0 => {
                    // honeytoken touch
                    x[idx::HONEYTOKEN] = 1.0;
                    x[idx::CONFIDENCE] = 1.0;
                    sev = 3;
                    lanes = if r.chance(0.5) { 1 } else { 2 };
                }
                1 => {
                    // DDoS-style fan-in, corroborated
                    x[idx::HYPERGRAPH] = r.range(0.45, 0.95);
                    x[idx::RAMANUJAN] = r.range(0.2, 0.9);
                    x[idx::NSM] = 1.0;
                    x[idx::SPECTRAL] = r.range(0.2, 0.8);
                    x[idx::CONFIDENCE] = r.range(0.55, 0.95);
                    sev = if r.chance(0.6) { 3 } else { 2 };
                    lanes = 2 + (r.next_u64() % 2) as u8;
                }
                2 => {
                    // port scan
                    x[idx::HYPERGRAPH] = r.range(0.3, 0.8);
                    x[idx::NSM] = 1.0;
                    x[idx::CONFIDENCE] = r.range(0.4, 0.8);
                    sev = 1 + (r.next_u64() % 2) as u8;
                    lanes = 2;
                }
                3 => {
                    // low-and-slow probing
                    x[idx::CONNECTOME] = r.range(0.45, 0.85);
                    x[idx::NSM] = 1.0;
                    x[idx::CONFIDENCE] = r.range(0.4, 0.7);
                    sev = 1;
                    lanes = 2;
                }
                _ => {
                    // beaconing / C2
                    x[idx::CONNECTOME] = r.range(0.55, 0.9);
                    x[idx::SPECTRAL] = r.range(0.5, 0.9);
                    x[idx::NSM] = 1.0;
                    x[idx::CONFIDENCE] = r.range(0.5, 0.85);
                    sev = 2;
                    lanes = 3;
                }
            }
            if r.chance(0.25) {
                x[idx::REPEAT_CONFIRMED] = r.range(0.33, 0.8);
            }
            if r.chance(0.2) {
                x[idx::KNOWN_INCIDENT] = r.range(0.3, 0.9);
            }
        } else {
            match r.categorical(&[3.0, 2.0, 2.0, 1.5]) {
                0 => {
                    // flash crowd: looks like fan-in, nothing else agrees
                    x[idx::HYPERGRAPH] = r.range(0.35, 0.9);
                    x[idx::RAMANUJAN] = r.range(0.0, 0.4);
                    x[idx::SPECTRAL] = r.range(0.0, 0.35);
                    x[idx::CONFIDENCE] = r.range(0.3, 0.7);
                    sev = 1 + (r.next_u64() % 2) as u8;
                    lanes = 2;
                }
                1 => {
                    // authorised vulnerability scanner
                    x[idx::HYPERGRAPH] = r.range(0.3, 0.7);
                    x[idx::NSM] = 1.0;
                    x[idx::REPEAT_DISMISSED] = r.range(0.4, 0.8);
                    x[idx::CONFIDENCE] = r.range(0.4, 0.75);
                    sev = 1 + (r.next_u64() % 2) as u8;
                    lanes = 2;
                }
                2 => {
                    // periodic backup / monitoring job
                    x[idx::CONNECTOME] = r.range(0.5, 0.85);
                    x[idx::SPECTRAL] = r.range(0.1, 0.5);
                    x[idx::CONFIDENCE] = r.range(0.3, 0.6);
                    sev = 1;
                    lanes = 2;
                }
                _ => {
                    // protected infrastructure being noisy
                    x[idx::PROTECTED] = 1.0;
                    x[idx::HYPERGRAPH] = r.range(0.2, 0.7);
                    x[idx::CONNECTOME] = r.range(0.2, 0.6);
                    x[idx::CONFIDENCE] = r.range(0.4, 0.85);
                    sev = 1 + (r.next_u64() % 2) as u8;
                    lanes = 2;
                }
            }
            if r.chance(0.15) {
                x[idx::REPEAT_DISMISSED] = x[idx::REPEAT_DISMISSED].max(r.range(0.3, 0.7));
            }
        }
        // severity and lane features are derived consistently
        x[idx::SEVERITY] = f64::from(sev) / 3.0;
        x[idx::LANES] = (f64::from(lanes) / 6.0).min(1.0);
        (StateFeatures::from_parts(x, sev, lanes), malicious)
    }
}
