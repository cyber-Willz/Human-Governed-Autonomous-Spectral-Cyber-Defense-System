//! State features: a fixed-size numeric view of a `CorrelationVerdict` plus
//! what the decision layer remembers about the host.
//!
//! Only numeric lane outputs go in. Attacker-controlled text (DNS names,
//! alert messages) is never an input, so it cannot steer the policy.

use correlation_engine::{CorrelationVerdict, LaneDetail, LaneSource, Severity};
use serde::{Deserialize, Serialize};

pub const STATE_DIM: usize = 13;

pub mod idx {
    pub const SEVERITY: usize = 0;
    pub const CONFIDENCE: usize = 1;
    pub const LANES: usize = 2;
    pub const HONEYTOKEN: usize = 3;
    pub const HYPERGRAPH: usize = 4;
    pub const RAMANUJAN: usize = 5;
    pub const CONNECTOME: usize = 6;
    pub const NSM: usize = 7;
    pub const SPECTRAL: usize = 8;
    pub const REPEAT_CONFIRMED: usize = 9;
    pub const REPEAT_DISMISSED: usize = 10;
    pub const PROTECTED: usize = 11;
    pub const KNOWN_INCIDENT: usize = 12;
}

/// What the layer has learned about this host from earlier outcomes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostHistory {
    pub confirmed: u32,
    pub dismissed: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StateFeatures {
    pub x: [f64; STATE_DIM],
    /// 0 = Low .. 3 = Critical (needed by the cost model, kept exact).
    pub severity: u8,
    pub protected: bool,
    pub honeytoken: bool,
    pub n_lanes: u8,
}

fn squash(v: f64) -> f64 {
    if v.is_finite() && v > 0.0 { v / (1.0 + v) } else { 0.0 }
}

fn count_squash(n: u32) -> f64 {
    f64::from(n) / (f64::from(n) + 2.0)
}

fn sev_index(s: Severity) -> u8 {
    match s {
        Severity::Low => 0,
        Severity::Medium => 1,
        Severity::High => 2,
        Severity::Critical => 3,
    }
}

impl StateFeatures {
    pub fn from_verdict(v: &CorrelationVerdict, hist: HostHistory, protected: bool) -> Self {
        let mut x = [0.0; STATE_DIM];
        let severity = sev_index(v.severity);
        let honeytoken = v.sources.contains(&LaneSource::Honeytoken);
        // distinct lanes only
        let mut lanes: Vec<LaneSource> = Vec::new();
        for s in &v.sources {
            if !lanes.contains(s) {
                lanes.push(*s);
            }
        }
        let n_lanes = lanes.len().min(255) as u8;

        x[idx::SEVERITY] = f64::from(severity) / 3.0;
        x[idx::CONFIDENCE] = f64::from(v.confidence).clamp(0.0, 1.0);
        x[idx::LANES] = (f64::from(n_lanes) / 6.0).min(1.0);
        x[idx::HONEYTOKEN] = f64::from(u8::from(honeytoken));
        x[idx::NSM] = f64::from(u8::from(lanes.contains(&LaneSource::Nsm)));

        for e in &v.evidence {
            match &e.detail {
                LaneDetail::Hypergraph { fiedler_shift, ramanujan_deviation, .. } => {
                    x[idx::HYPERGRAPH] = x[idx::HYPERGRAPH].max(squash(*fiedler_shift));
                    if let Some(d) = ramanujan_deviation {
                        x[idx::RAMANUJAN] = x[idx::RAMANUJAN].max(squash(*d));
                    }
                }
                LaneDetail::Connectome { score, .. } => {
                    x[idx::CONNECTOME] = x[idx::CONNECTOME].max(score.clamp(0.0, 1.0));
                }
                LaneDetail::Spectral { anomaly_score, .. } => {
                    x[idx::SPECTRAL] = x[idx::SPECTRAL].max(anomaly_score.clamp(0.0, 1.0));
                }
                _ => {}
            }
        }
        x[idx::REPEAT_CONFIRMED] = count_squash(hist.confirmed);
        x[idx::REPEAT_DISMISSED] = count_squash(hist.dismissed);
        x[idx::PROTECTED] = f64::from(u8::from(protected));
        // A close confirmed-incident neighbour is evidence; distance is in
        // the 4-D fused signal space, so 1/(1+d) is a similarity in (0, 1].
        if let Some(n) = &v.nearest_known_incident {
            x[idx::KNOWN_INCIDENT] = 1.0 / (1.0 + n.distance.max(0.0));
        }
        Self { x, severity, protected, honeytoken, n_lanes }
    }

    /// Direct constructor used by the sandbox and tests.
    pub fn from_parts(x: [f64; STATE_DIM], severity: u8, n_lanes: u8) -> Self {
        let mut x = x;
        for v in x.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
            *v = v.clamp(0.0, 1.0);
        }
        Self {
            honeytoken: x[idx::HONEYTOKEN] > 0.5,
            protected: x[idx::PROTECTED] > 0.5,
            x,
            severity: severity.min(3),
            n_lanes,
        }
    }
}
