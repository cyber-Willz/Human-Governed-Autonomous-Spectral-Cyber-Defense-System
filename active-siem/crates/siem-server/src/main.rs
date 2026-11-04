//! ActiveSIEM orchestrator - demo wiring of the full pipeline:
//!
//!   collector -> rule engine (noisy/known attacks) -> ML scorer (quiet infiltration)
//!            -> alert stream -> response policy (guarded active response)
//!
//! This binary simulates an SSH brute-force (caught by siem-rules) and a
//! slow-beacon infiltration flow (caught by siem-ml, invisible to signatures)
//! to demonstrate why both detection layers are necessary.

use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use siem_collector::synthetic_flow;
use siem_core::EventKind;
use siem_ml::{score, train, AutoencoderConfig, FlowFeatures};
use siem_response::{LogOnly, ResponsePolicy};
use siem_rules::RuleEngine;
use std::collections::HashSet;

type TrainBackend = Autodiff<NdArray<f32>>;
type InferBackend = NdArray<f32>;

fn main() {
    println!("=== ActiveSIEM demo ===\n");

    // --- 1. Rule engine: catches the noisy precursor (brute force) ---
    let rules = RuleEngine::load_yaml(siem_rules::builtin_rules_yaml()).unwrap();
    let mut engine = RuleEngine::new(rules);
    let mut next_id = 1u64;

    println!("-- Simulating SSH brute force from 10.0.0.5 --");
    for i in 0..5 {
        let mut ev = siem_collector::parse_sshd_line(
            "web01",
            "agent-01",
            "Jul 13 10:00:01 web01 sshd[123]: Failed password for root from 10.0.0.5 port 51000 ssh2",
        )
        .unwrap();
        ev.id = next_id;
        ev.timestamp_ms = i * 1000;
        next_id += 1;
        for alert in engine.process(&ev) {
            println!(
                "  ALERT [{}] {} (severity {:?}, mitre {:?})",
                alert.rule_id, alert.title, alert.severity, alert.mitre_technique
            );
        }
    }

    // --- 2. ML scorer: catches the quiet infiltration flow rules would miss ---
    println!("\n-- Training autoencoder on benign traffic shape --");
    let device = Default::default();
    let benign: Vec<FlowFeatures> = (0..200)
        .map(|i| {
            let ev = synthetic_flow(
                "web01",
                "agent-01",
                "10.0.0.20",
                "203.0.113.10",
                40000 + (i % 1000),
                443,
                (2_000 + (i % 500) * 10) as u64, // a couple seconds, variable
                (1500 + (i % 300) * 20) as u64,  // normal browsing-sized upload
                (40_000 + (i % 2000) * 5) as u64, // much larger download than upload
                (20 + (i % 10)) as u64,
            );
            let EventKind::Flow { .. } = &ev.kind else {
                unreachable!()
            };
            FlowFeatures::from_event(&ev.kind).unwrap()
        })
        .collect();

    let config = AutoencoderConfig::default();
    let model = train::<TrainBackend>(&device, &config, &benign, 300, 1e-2);
    let infer_model = model.valid(); // drop autodiff graph for inference

    // Calibrate a threshold from benign reconstruction error (e.g. 99th percentile-ish: max + margin)
    let device_infer: <InferBackend as burn::tensor::backend::Backend>::Device = Default::default();
    let benign_scores: Vec<f32> = benign
        .iter()
        .map(|f| score::<InferBackend>(&infer_model, &device_infer, *f))
        .collect();
    let max_benign = benign_scores.iter().cloned().fold(0f32, f32::max);
    let threshold = max_benign * 1.5 + 0.001;
    println!("  calibrated anomaly threshold: {threshold:.5}");

    println!("\n-- Scoring a suspected infiltration beacon (long, tiny, periodic, upload-heavy) --");
    let beacon = synthetic_flow(
        "web01",
        "agent-01",
        "10.0.0.20",
        "198.51.100.77", // unfamiliar low-reputation-looking destination
        51234,
        8443,
        9 * 60_000, // 9 minutes - unusually long for a "normal" web hit
        1_800,      // small, steady upload (exfil-shaped)
        900,        // barely any response - not a normal request/response pattern
        14,
    );
    let f = FlowFeatures::from_event(&beacon.kind).unwrap();
    let s = score::<InferBackend>(&infer_model, &device_infer, f);
    println!("  reconstruction error: {s:.5} (threshold {threshold:.5})");
    let is_infiltration = s > threshold;
    println!("  flagged as anomalous/infiltration-like: {is_infiltration}");

    // --- 3. Guarded active response ---
    println!("\n-- Active response policy --");
    let mut allowlist = HashSet::new();
    allowlist.insert("10.0.0.20".to_string()); // never auto-block our own host
    let mut policy = ResponsePolicy::new(siem_core::Severity::Medium, allowlist);
    let mut action = LogOnly;

    if is_infiltration {
        let alert = siem_core::Alert {
            id: 999,
            timestamp_ms: siem_core::Event::now_ms(),
            rule_id: "ml-infiltration-anomaly".to_string(),
            title: "Anomalous long-duration low-volume flow (possible infiltration beacon)".to_string(),
            severity: siem_core::Severity::High,
            mitre_technique: Some("T1071".to_string()), // Application Layer Protocol (C2)
            source_events: vec![beacon.id],
            context: Default::default(),
        };
        match policy.handle(&mut action, &alert, "198.51.100.77") {
            Ok(()) => println!("  response executed."),
            Err(reason) => println!("  response withheld: {reason}"),
        }
    }
}
