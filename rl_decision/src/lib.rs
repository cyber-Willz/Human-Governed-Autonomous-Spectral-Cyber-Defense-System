//! # rl_decision — the reinforcement-learning decision layer
//!
//! Sits between `correlation_engine` (verdicts) and `ips_control_plane`
//! (enforcement), implementing the "RL decision layer" of the architecture:
//!
//! | Diagram box | Module |
//! |---|---|
//! | Sandbox Sim | [`sandbox`] |
//! | Risk Model | [`risk`] |
//! | Policy Net | [`policy`] |
//! | Action Selection | [`selector`] (shield + scorer) |
//! | Feedback & Rewards, RL Continuous Learning | [`layer`] (+ [`action::reward`]) |
//!
//! See [`layer`] for the authority ladder (Shadow → Advisory → Active) and
//! the list of things this layer can never do. Short version: it can
//! escalate to a human and pick among shielded ban tiers; it cannot bypass
//! the review gate, raise confidence, ban protected hosts, or unban.
//!
//! **Honest scope:** the cost model and sandbox dynamics are hand-set; all
//! pre-training data is synthetic. Treat the learned behaviour as
//! "calibrate-on-your-labels", not as validated on real traffic.

pub mod action;
pub mod layer;
pub mod policy;
pub mod risk;
pub mod rng;
pub mod sandbox;
pub mod selector;
pub mod state;

pub use action::{legacy_action, loss, reward, Action, ALL_ACTIONS, N_ACTIONS};
pub use layer::{Decision, Directive, LayerStats, Mode, Recommendation, RlConfig, RlDecisionLayer, TrainReport};
pub use state::{HostHistory, StateFeatures, STATE_DIM};
