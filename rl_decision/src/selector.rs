//! Action selection: the shield (hard constraints) and the scorer that
//! blends the three inputs from the architecture diagram -- sandbox
//! look-ahead, risk model, policy network.
//!
//! The shield is applied to the *action set* before any learned component
//! gets a vote. A learned model can therefore be wrong about which allowed
//! action is best, but it cannot select a disallowed one, however its
//! weights drift. Shield rules are plain code and are unit-tested.

use serde::{Deserialize, Serialize};

use crate::action::{Action, ALL_ACTIONS, N_ACTIONS};
use crate::state::StateFeatures;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShieldConfig {
    /// Any ban needs at least this `P(malicious)` (honeytoken exempt).
    pub min_ban_p: f64,
    /// `BanExtended` needs at least this `P(malicious)` ...
    pub extended_min_p: f64,
    /// ... and a risk-model uncertainty below this (an under-evidenced
    /// model may not pick the heaviest action).
    pub extended_max_uncertainty: f64,
}

impl Default for ShieldConfig {
    fn default() -> Self {
        Self { min_ban_p: 0.30, extended_min_p: 0.80, extended_max_uncertainty: 0.85 }
    }
}

#[derive(Debug, Clone)]
pub struct ShieldOutcome {
    pub allowed: [bool; N_ACTIONS],
    pub reasons: Vec<&'static str>,
}

pub fn shield(s: &StateFeatures, p_malicious: f64, uncertainty: f64, cfg: &ShieldConfig) -> ShieldOutcome {
    let mut allowed = [true; N_ACTIONS];
    let mut reasons = Vec::new();

    if s.protected {
        // Same stance as the control plane: protected infrastructure is
        // never banned autonomously.
        for a in ALL_ACTIONS {
            if a.is_ban() {
                allowed[a.index()] = false;
            }
        }
        reasons.push("protected host: bans disallowed, human review only");
    } else if s.honeytoken {
        // A touched honeytoken is ground truth by construction: never
        // downgrade it to "just review", never use the shortest ban.
        allowed[Action::Escalate.index()] = false;
        allowed[Action::BanShort.index()] = false;
        reasons.push("honeytoken: containment floor is BanStandard");
    } else {
        if p_malicious < cfg.min_ban_p {
            for a in ALL_ACTIONS {
                if a.is_ban() {
                    allowed[a.index()] = false;
                }
            }
            reasons.push("risk below ban threshold");
        }
        let extended_ok = s.n_lanes >= 2 && p_malicious >= cfg.extended_min_p && uncertainty <= cfg.extended_max_uncertainty;
        if !extended_ok {
            allowed[Action::BanExtended.index()] = false;
            reasons.push("BanExtended needs >=2 lanes, high risk and low model uncertainty");
        }
    }

    if !allowed.iter().any(|a| *a) {
        // unreachable by construction, but never return an empty set
        allowed[Action::Escalate.index()] = true;
    }
    ShieldOutcome { allowed, reasons }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectorConfig {
    /// Weight of the sandbox Monte-Carlo value vs. the risk model's
    /// analytic expected loss (0 = risk only, 1 = sandbox only).
    pub w_sandbox: f64,
    /// Weight on `ln(policy probability)` (the learned prior / tie-break).
    pub w_policy: f64,
}

impl Default for SelectorConfig {
    fn default() -> Self {
        Self { w_sandbox: 0.4, w_policy: 0.05 }
    }
}

/// Picks the best *allowed* action by
/// `(1-w)·(−E[loss]/10) + w·Q_sandbox + w_policy·ln π`.
pub fn select(
    allowed: &[bool; N_ACTIONS],
    expected_loss: &[f64; N_ACTIONS],
    q_sandbox: &[f64; N_ACTIONS],
    policy: &[f64; N_ACTIONS],
    cfg: &SelectorConfig,
) -> (Action, [f64; N_ACTIONS]) {
    let mut scores = [f64::NEG_INFINITY; N_ACTIONS];
    let mut best = Action::Escalate;
    let mut best_score = f64::NEG_INFINITY;
    for a in ALL_ACTIONS {
        let i = a.index();
        if !allowed[i] {
            continue;
        }
        let s = (1.0 - cfg.w_sandbox) * (-expected_loss[i] / 10.0)
            + cfg.w_sandbox * q_sandbox[i]
            + cfg.w_policy * (policy[i] + 0.05).ln(); // floor keeps the learned term bounded (>= ln 0.05)
        scores[i] = s;
        if s > best_score {
            best_score = s;
            best = a;
        }
    }
    (best, scores)
}
