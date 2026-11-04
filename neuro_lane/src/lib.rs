//! # `neuro_lane` -- connectome-LSTM temporal detector lane
//!
//! Fills the gap `SUMMARY.md` calls out: the fan-in/fan-out detectors use
//! hard count/time thresholds, so an attacker who paces under the
//! threshold is invisible to them. This lane keeps a per-host *sequence*
//! of small feature vectors ([`features`]) and scores the whole sequence
//! with a [`hcrnn::ConnectomeLstm`] ([`detector`]), whose
//! gated cells can hold "this host has probed quietly and persistently for
//! N bins" across time.
//!
//! The runtime ([`lane::NeuroLane`]) submits scores to
//! `correlation_engine::ConnectomeSender`, lowers its threshold for hosts
//! on the raise-only [`llm_bridge::ScrutinyBoard`], produces
//! [`llm_bridge::CognitiveReport`]s for the uplink, and learns from analyst
//! verdicts through a guarded fine-tuner (reverts if it degrades a fixed
//! anchor set).
//!
//! ## Honesty notes
//! * The default wiring is `Connectome::synthetic_human` (a stand-in), not
//!   measured tractography; pass a real one (`from_tvb_dir`) to use it.
//! * Training data is synthetic ([`synth`]): the detector learns the
//!   behaviours *defined there*. See the evaluation tests and README for
//!   what is and isn't shown. No real-traffic evaluation exists here.

pub mod detector;
pub mod features;
pub mod lane;
pub mod synth;

pub use detector::{ConnectomeDetector, DetectorConfig, FineTuneOutcome};
pub use features::{FeatVec, HostBinner, N_FEAT};
pub use lane::{BinReport, HostScore, NeuroLane, NeuroLaneConfig};
