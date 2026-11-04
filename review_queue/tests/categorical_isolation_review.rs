//! Reproduces the `categorical_isolation.rs` per-flow prediction set from
//! the NDR classifier's test harness end-to-end through the review queue,
//! and asserts the one misclassified case (`true=Benign,
//! predicted=Infiltration, confidence=0.856`) is queued for human review
//! and blocked from containment until a human resolves it -- while every
//! confidently-classified case auto-resolves without a human touching it.

use review_queue::prelude::*;

/// (predicted_label, confidence) for each of the 29 flows in the original
/// test harness output, in order. Ground truth isn't threaded through here
/// -- the queue only ever sees what a real deployment would see.
const PREDICTIONS: &[(&str, f64)] = &[
    ("Benign", 0.997),
    ("Infiltration", 0.856), // the misclassified one
    ("Infiltration", 0.993),
    ("Infiltration", 0.990),
    ("Infiltration", 0.977),
    ("Infiltration", 0.999),
    ("Infiltration", 1.000),
    ("Infiltration", 0.999),
    ("Infiltration", 1.000),
    ("Infiltration", 0.999),
    ("BruteForce", 1.000),
    ("BruteForce", 0.999),
    ("BruteForce", 1.000),
    ("BruteForce", 1.000),
    ("BruteForce", 0.999),
    ("BruteForce", 1.000),
    ("CommandAndControl", 1.000),
    ("CommandAndControl", 1.000),
    ("CommandAndControl", 1.000),
    ("CommandAndControl", 1.000),
    ("CommandAndControl", 1.000),
    ("DenialOfService", 1.000),
    ("DenialOfService", 1.000),
    ("DenialOfService", 1.000),
    ("DenialOfService", 0.999),
    ("DenialOfService", 1.000),
    ("DenialOfService", 1.000),
    ("DenialOfService", 1.000),
    ("DenialOfService", 1.000),
];

fn ingest_all(queue: &ReviewQueue) -> Vec<(String, IngestOutcome)> {
    PREDICTIONS
        .iter()
        .enumerate()
        .map(|(i, (label, confidence))| {
            let flow_id = format!("flow-{i:02}");
            let outcome = queue
                .ingest(FlowPrediction {
                    flow_id: flow_id.clone(),
                    predicted_label: label.to_string(),
                    confidence: *confidence,
                    runner_up_label: None,
                    runner_up_confidence: None,
                    is_out_of_distribution: false,
                    observed_at: now(),
                })
                .expect("ingest should succeed for a fresh flow_id");
            (flow_id, outcome)
        })
        .collect()
}

#[test]
fn only_the_low_confidence_infiltration_call_is_queued() {
    let queue = ReviewQueue::new(TriggerConfig::default(), SlaPolicy::FailSafe).unwrap();
    let results = ingest_all(&queue);

    let queued: Vec<&str> = results
        .iter()
        .filter(|(_, o)| matches!(o, IngestOutcome::QueuedForReview { .. }))
        .map(|(id, _)| id.as_str())
        .collect();

    assert_eq!(
        queued,
        vec!["flow-01"],
        "exactly the 0.856-confidence Infiltration call should be queued; \
         everything else clears the trigger thresholds cleanly"
    );

    let stats = queue.stats();
    assert_eq!(stats.total, 29);
    assert_eq!(stats.pending, 1);
    assert_eq!(stats.auto_resolved, 28);
}

#[test]
fn queued_flow_is_not_auto_contained() {
    let queue = ReviewQueue::new(TriggerConfig::default(), SlaPolicy::FailSafe).unwrap();
    ingest_all(&queue);

    // The flagged flow must show up as pending, and containment must
    // refuse to act on it -- this is the core guarantee: the classifier's
    // 0.856-confidence call alone is never enough to trigger containment.
    let item = queue.get("flow-01").unwrap();
    assert!(matches!(item.state, ReviewState::Pending { .. }));
    assert_eq!(
        queue.containment_decision("flow-01"),
        ContainmentDecision::NotApproved
    );

    struct Tripwire(std::sync::Arc<std::sync::atomic::AtomicBool>);
    impl ContainmentAction for Tripwire {
        fn execute(&self, _flow_id: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }
    let fired = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let result = queue.execute_containment_if_approved("flow-01", Tripwire(fired.clone()));
    assert!(result.is_err(), "containment must refuse an unapproved flow");
    assert!(!fired.load(std::sync::atomic::Ordering::SeqCst), "the action must never have run");
}

#[test]
fn human_dismissing_the_flow_closes_it_without_containment() {
    let queue = ReviewQueue::new(TriggerConfig::default(), SlaPolicy::FailSafe).unwrap();
    ingest_all(&queue);

    queue.claim("flow-01", "analyst_priya").unwrap();
    assert!(matches!(
        queue.get("flow-01").unwrap().state,
        ReviewState::UnderReview { .. }
    ));

    let decision = queue
        .record_decision(
            "flow-01",
            "analyst_priya",
            Verdict::Dismiss,
            "pcap review: legitimate scheduled backup job, not infiltration",
        )
        .unwrap();
    assert_eq!(decision.verdict, Verdict::Dismiss);

    assert!(matches!(
        queue.get("flow-01").unwrap().state,
        ReviewState::Resolved { .. }
    ));
    assert_eq!(
        queue.containment_decision("flow-01"),
        ContainmentDecision::NotApproved
    );

    // Full audit trail is queryable back through the graph, not just the
    // queue's own cache.
    let decisions = queue.decisions_for_flow("flow-01");
    assert_eq!(decisions.len(), 1);
    assert_eq!(decisions[0].reviewer, Reviewer::Human("analyst_priya".to_string()));
}

#[test]
fn human_confirming_the_flow_authorizes_containment() {
    let queue = ReviewQueue::new(TriggerConfig::default(), SlaPolicy::FailSafe).unwrap();
    ingest_all(&queue);

    queue
        .record_decision(
            "flow-01",
            "analyst_priya",
            Verdict::ExecuteContainment,
            "pcap review confirms lateral movement signature; containing host",
        )
        .unwrap();

    assert!(matches!(
        queue.containment_decision("flow-01"),
        ContainmentDecision::Approved { .. }
    ));

    let fired = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    struct Tripwire(std::sync::Arc<std::sync::atomic::AtomicBool>);
    impl ContainmentAction for Tripwire {
        fn execute(&self, _flow_id: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }
    queue
        .execute_containment_if_approved("flow-01", Tripwire(fired.clone()))
        .expect("approved containment should execute");
    assert!(fired.load(std::sync::atomic::Ordering::SeqCst));

    // Cannot double-record a decision on an already-resolved flow.
    let second = queue.record_decision("flow-01", "analyst_jamal", Verdict::Dismiss, "double check");
    assert!(second.is_err());
}

#[test]
fn sla_fail_safe_never_auto_contains() {
    let queue = ReviewQueue::new(TriggerConfig::default(), SlaPolicy::FailSafe).unwrap();
    ingest_all(&queue);

    // sla_seconds=0 means "anything already enqueued counts as expired".
    let expired = queue.sweep_expired(0).unwrap();
    assert_eq!(expired.len(), 1);
    assert_eq!(expired[0].prediction.flow_id, "flow-01");
    assert_eq!(expired[0].decision.verdict, Verdict::Dismiss);
    assert_eq!(expired[0].decision.reviewer, Reviewer::SystemSlaTimeout);

    assert!(matches!(
        queue.get("flow-01").unwrap().state,
        ReviewState::Resolved { .. }
    ));
    assert_eq!(
        queue.containment_decision("flow-01"),
        ContainmentDecision::NotApproved,
        "fail-safe SLA fallback must dismiss, never auto-contain"
    );
}

/// Safety invariant, not just a single example: sweep a wide spread of
/// labels/confidences through `sweep_expired` under `FailSafe`, and assert
/// none of them ever produces `Verdict::ExecuteContainment` *via the SLA
/// path specifically*.
///
/// Scoped deliberately to flows that were actually `Pending` going into the
/// sweep (i.e. the trigger queued them) -- a confidently-classified attack
/// with a wide margin can still auto-contain immediately at `ingest()`,
/// which is the separate, intentional fast-track for unambiguous
/// predictions (see `trigger::ReviewTrigger`) and is not the case this
/// property is about. What must never happen is a *timed-out, unreviewed*
/// flow getting `ExecuteContainment` -- that's what fail-safe means.
#[test]
fn sla_fail_safe_never_auto_contains_property() {
    let labels = ["Benign", "Infiltration", "BruteForce", "CommandAndControl", "DenialOfService"];
    let confidences = [0.01, 0.5, 0.856, 0.90, 0.999, 1.0];

    for &label in &labels {
        for &confidence in &confidences {
            let queue = ReviewQueue::new(TriggerConfig::default(), SlaPolicy::FailSafe).unwrap();
            let flow_id = format!("{label}-{confidence}");
            let outcome = queue
                .ingest(FlowPrediction {
                    flow_id: flow_id.clone(),
                    predicted_label: label.to_string(),
                    confidence,
                    runner_up_label: None,
                    runner_up_confidence: None,
                    is_out_of_distribution: false,
                    observed_at: now(),
                })
                .unwrap();

            if !matches!(outcome, IngestOutcome::QueuedForReview { .. }) {
                // Cleared the trigger and auto-resolved at ingest; that
                // path is intentionally allowed to auto-contain and is out
                // of scope for this property.
                continue;
            }

            queue.sweep_expired(0).unwrap();

            assert!(
                !matches!(
                    queue.containment_decision(&flow_id),
                    ContainmentDecision::Approved { .. }
                ),
                "label={label} confidence={confidence}: a flow that was queued for review and \
                 then timed out must never end up containment-approved under fail-safe"
            );
            assert_eq!(
                queue.decisions_for_flow(&flow_id)[0].reviewer,
                Reviewer::SystemSlaTimeout
            );
        }
    }
}

/// Direct assertion on `sweep_expired_with_sink`'s alerting contract: a
/// non-benign, unreviewed dismissal must reach the alert sink and be
/// classified as urgent.
#[test]
fn fail_safe_dismissal_of_attack_call_triggers_urgent_alert() {
    use review_queue::alert::{SlaBreach, SlaBreachAlertSink};
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct CapturingSink {
        breaches: Arc<Mutex<Vec<SlaBreach>>>,
    }
    impl SlaBreachAlertSink for CapturingSink {
        fn notify(&self, breach: &SlaBreach) {
            self.breaches.lock().unwrap().push(breach.clone());
        }
    }

    let queue = ReviewQueue::new(TriggerConfig::default(), SlaPolicy::FailSafe).unwrap();
    ingest_all(&queue); // flow-01 is the 0.856 Infiltration call, queued

    let sink = CapturingSink::default();
    let captured = sink.breaches.clone();
    queue.sweep_expired_with_sink(0, &sink).unwrap();

    let breaches = captured.lock().unwrap();
    assert_eq!(breaches.len(), 1);
    assert_eq!(breaches[0].prediction.flow_id, "flow-01");
    assert!(
        breaches[0].is_unreviewed_attack_dismissal(),
        "the 0.856 Infiltration call dismissed under fail-safe must be flagged urgent"
    );
}

#[test]
fn sla_fail_secure_auto_contains_above_threshold() {
    // 0.856 -> 856 millis; set the auto-contain floor just below it so this
    // specific case *would* auto-contain under fail-secure.
    let queue = ReviewQueue::new(
        TriggerConfig::default(),
        SlaPolicy::FailSecure {
            auto_contain_above_confidence: 850,
        },
    )
    .unwrap();
    ingest_all(&queue);

    let expired = queue.sweep_expired(0).unwrap();
    assert_eq!(expired.len(), 1);
    assert_eq!(expired[0].prediction.flow_id, "flow-01");
    assert_eq!(expired[0].decision.verdict, Verdict::ExecuteContainment);

    assert!(matches!(
        queue.containment_decision("flow-01"),
        ContainmentDecision::Approved { .. }
    ));
    let decisions = queue.decisions_for_flow("flow-01");
    assert_eq!(decisions[0].reviewer, Reviewer::SystemSlaTimeout);
}

#[test]
fn duplicate_flow_id_is_rejected() {
    let queue = ReviewQueue::new(TriggerConfig::default(), SlaPolicy::FailSafe).unwrap();
    let pred = FlowPrediction {
        flow_id: "dup".into(),
        predicted_label: "Benign".into(),
        confidence: 0.99,
        runner_up_label: None,
        runner_up_confidence: None,
        is_out_of_distribution: false,
        observed_at: now(),
    };
    queue.ingest(pred.clone()).unwrap();
    assert!(queue.ingest(pred).is_err());
}

#[test]
fn persistence_round_trip_preserves_state_and_audit_graph() {
    let dir = std::env::temp_dir().join(format!(
        "review_queue_test_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("state.json");

    {
        let queue = ReviewQueue::new(TriggerConfig::default(), SlaPolicy::FailSafe).unwrap();
        ingest_all(&queue);
        queue
            .record_decision(
                "flow-01",
                "analyst_priya",
                Verdict::ExecuteContainment,
                "confirmed via pcap",
            )
            .unwrap();
        review_queue::store::save(&queue, &path).unwrap();
    }

    let reloaded =
        review_queue::store::load_or_new(&path, TriggerConfig::default(), SlaPolicy::FailSafe).unwrap();

    assert_eq!(reloaded.stats().total, 29);
    assert!(matches!(
        reloaded.containment_decision("flow-01"),
        ContainmentDecision::Approved { .. }
    ));
    let decisions = reloaded.decisions_for_flow("flow-01");
    assert_eq!(decisions.len(), 1);
    assert_eq!(decisions[0].rationale, "confirmed via pcap");

    std::fs::remove_dir_all(&dir).ok();
}
