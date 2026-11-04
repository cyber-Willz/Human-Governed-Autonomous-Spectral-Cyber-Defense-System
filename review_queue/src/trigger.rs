//! Decides whether a [`FlowPrediction`] can be auto-resolved or must be
//! queued for a human. This is the only place that policy thresholds live,
//! so tuning review sensitivity means editing one struct, not hunting
//! through call sites.

use crate::types::{FlowPrediction, TriggerReason};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TriggerConfig {
    /// The label treated as "no attack". Everything else is an attack class
    /// for the purposes of `min_attack_confidence`.
    pub benign_label: String,
    /// Non-benign predictions below this confidence are queued, even if
    /// nothing else about them looks unusual.
    pub min_attack_confidence: f64,
    /// Predictions (of any label, including benign) whose top-1/runner-up
    /// margin is below this are queued: the model was close to picking a
    /// different class, so its output shouldn't be trusted unreviewed.
    pub min_margin: f64,
    /// Benign predictions are still queued if their confidence dips below
    /// this floor -- a low-confidence "benign" call is exactly the kind of
    /// case that produces the false-negative you cannot afford to miss.
    pub min_benign_confidence: f64,
}

impl Default for TriggerConfig {
    fn default() -> Self {
        Self {
            benign_label: "Benign".to_string(),
            min_attack_confidence: 0.90,
            min_margin: 0.15,
            min_benign_confidence: 0.80,
        }
    }
}

pub struct ReviewTrigger;

impl ReviewTrigger {
    /// Returns `Some(reason)` if `pred` must be queued for human review
    /// before any containment action can be taken on it; `None` if it is
    /// safe to auto-resolve.
    pub fn evaluate(pred: &FlowPrediction, cfg: &TriggerConfig) -> Option<TriggerReason> {
        if pred.is_out_of_distribution {
            return Some(TriggerReason::NovelPattern);
        }

        let is_benign = pred.predicted_label == cfg.benign_label;

        if !is_benign && pred.confidence < cfg.min_attack_confidence {
            return Some(TriggerReason::LowConfidenceAttack);
        }

        if is_benign && pred.confidence < cfg.min_benign_confidence {
            return Some(TriggerReason::LowConfidenceAttack);
        }

        if let Some(margin) = pred.margin() {
            if margin < cfg.min_margin {
                return Some(TriggerReason::NarrowMargin);
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_pred() -> FlowPrediction {
        FlowPrediction {
            flow_id: "f1".into(),
            predicted_label: "Benign".into(),
            confidence: 0.99,
            runner_up_label: None,
            runner_up_confidence: None,
            is_out_of_distribution: false,
            observed_at: 0,
        }
    }

    #[test]
    fn confident_benign_auto_resolves() {
        let cfg = TriggerConfig::default();
        assert_eq!(ReviewTrigger::evaluate(&base_pred(), &cfg), None);
    }

    #[test]
    fn confident_attack_auto_resolves() {
        let cfg = TriggerConfig::default();
        let mut p = base_pred();
        p.predicted_label = "DenialOfService".into();
        p.confidence = 1.0;
        assert_eq!(ReviewTrigger::evaluate(&p, &cfg), None);
    }

    #[test]
    fn the_0_856_infiltration_case_is_flagged() {
        // Reproduces the categorical_isolation.rs misclassification:
        // true=Benign, predicted=Infiltration, confidence=0.856.
        let cfg = TriggerConfig::default();
        let mut p = base_pred();
        p.predicted_label = "Infiltration".into();
        p.confidence = 0.856;
        assert_eq!(
            ReviewTrigger::evaluate(&p, &cfg),
            Some(TriggerReason::LowConfidenceAttack)
        );
    }

    #[test]
    fn narrow_margin_flags_even_high_confidence() {
        let cfg = TriggerConfig::default();
        let mut p = base_pred();
        p.predicted_label = "DenialOfService".into();
        p.confidence = 0.95;
        p.runner_up_label = Some("BruteForce".into());
        p.runner_up_confidence = Some(0.88); // margin 0.07 < 0.15
        assert_eq!(
            ReviewTrigger::evaluate(&p, &cfg),
            Some(TriggerReason::NarrowMargin)
        );
    }

    #[test]
    fn out_of_distribution_always_flags() {
        let cfg = TriggerConfig::default();
        let mut p = base_pred();
        p.confidence = 1.0;
        p.is_out_of_distribution = true;
        assert_eq!(
            ReviewTrigger::evaluate(&p, &cfg),
            Some(TriggerReason::NovelPattern)
        );
    }

    #[test]
    fn low_confidence_benign_is_flagged() {
        let cfg = TriggerConfig::default();
        let mut p = base_pred();
        p.confidence = 0.55;
        assert_eq!(
            ReviewTrigger::evaluate(&p, &cfg),
            Some(TriggerReason::LowConfidenceAttack)
        );
    }
}
