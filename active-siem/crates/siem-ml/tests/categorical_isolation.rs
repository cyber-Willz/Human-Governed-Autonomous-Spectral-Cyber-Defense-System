//! Validates `siem_ml::classifier` - the categorical system requested after
//! analyzing `spec_engine` (from the uploaded `spectral_homology_tar.gz`):
//! instead of the autoencoder's single anomaly/not-anomaly bit, this checks
//! whether the classifier can isolate *which* attack family a flow belongs
//! to, alongside Benign, using the same category taxonomy sketched from
//! `spec_engine::classify_mitre_from_label`'s MITRE vocabulary.
//!
//! # Data provenance (read this before trusting the numbers)
//! - **Benign** (6 rows) and **Infiltration** (25-row subset of the 50 rows)
//!   are the real CIC-IDS2018-style rows provided earlier in this
//!   conversation - unchanged from `infiltration_dataset.rs`.
//! - **BruteForce**, **CommandAndControl** (the "Bot" label), and
//!   **DenialOfService** are seeded from `spec_engine`'s own synthetic
//!   CIC-IDS2018 dataset (the "1-20: Base Seeds" block in its `lib.rs`) -
//!   real *labels and flow shapes* from that project, but only 1-3 seed
//!   rows exist per category there. To get a train/test split of usable
//!   size, each seed is jittered (+-15% on duration/packet counts/byte
//!   totals, deterministically seeded) into a small family of variations.
//!   This is clearly weaker evidence than the real multi-row Infiltration/
//!   Benign data - it tests "can the model separate these *shapes*", not
//!   "does it generalize across independently-captured examples of each
//!   attack". Treat the per-category accuracy below as a shape-separability
//!   smoke test, not a benchmark.
//! - **Reconnaissance** is in the `Category` taxonomy (mirroring
//!   `spec_engine`'s vocab) but has no seed row in either dataset provided
//!   so far, real or synthetic - it's intentionally excluded from this
//!   test's training/evaluation rather than backed by invented data.

use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use siem_core::EventKind;
use siem_ml::classifier::{predict, train, Category, ClassifierConfig, LabeledFlow};
use siem_ml::FlowFeatures;

type TrainBackend = Autodiff<NdArray<f32>>;
type InferBackend = NdArray<f32>;

struct RawRow {
    dst_port: u16,
    protocol: u8,
    duration_us: i64,
    tot_fwd_pkts: u32,
    tot_bwd_pkts: u32,
    totlen_fwd: f64,
    totlen_bwd: f64,
}

macro_rules! row {
    ($port:expr, $proto:expr, $dur:expr, $fwd_pkts:expr, $bwd_pkts:expr, $fwd_bytes:expr, $bwd_bytes:expr) => {
        RawRow {
            dst_port: $port,
            protocol: $proto,
            duration_us: $dur,
            tot_fwd_pkts: $fwd_pkts,
            tot_bwd_pkts: $bwd_pkts,
            totlen_fwd: $fwd_bytes,
            totlen_bwd: $bwd_bytes,
        }
    };
}

fn to_features(row: &RawRow) -> FlowFeatures {
    let ev = EventKind::Flow {
        src_ip: "0.0.0.0".to_string(),
        dst_ip: "0.0.0.0".to_string(),
        src_port: 0,
        dst_port: row.dst_port,
        proto: row.protocol,
        duration_ms: (row.duration_us / 1000).max(0) as u64,
        bytes_src_to_dst: row.totlen_fwd.max(0.0) as u64,
        bytes_dst_to_src: row.totlen_bwd.max(0.0) as u64,
        packets: (row.tot_fwd_pkts + row.tot_bwd_pkts) as u64,
        flags: String::new(),
    };
    FlowFeatures::from_event(&ev).expect("Flow event always yields features")
}

/// Real Benign rows (deduplicated), same data as `infiltration_dataset.rs`.
fn benign_rows() -> Vec<RawRow> {
    vec![
        row!(3389, 6, 4_046_191, 14, 7, 439.4, 5.19),
        row!(53, 17, 303, 1, 1, 356_435.6, 6600.7),
        row!(443, 6, 120_000_000, 45, 38, 18_432.0, 0.69),
        row!(53, 17, 400, 1, 1, 300_000.0, 5000.0),
        row!(443, 6, 60_000_000, 30, 25, 9_600.0, 0.92),
        row!(80, 6, 3_000_000, 8, 6, 6_400.0, 4.67),
    ]
}

/// A 25-row subset of the real Infiltration dataset used in
/// `infiltration_dataset.rs` (HTTP/HTTPS beaconing + RDP probing rows).
fn infiltration_rows() -> Vec<RawRow> {
    vec![
        row!(80, 6, 4_512_891, 12, 10, 3812.0, 845.0),
        row!(80, 6, 5_120_114, 15, 11, 4910.0, 912.0),
        row!(80, 6, 3_891_002, 10, 8, 2950.0, 610.0),
        row!(443, 6, 12_405_110, 42, 38, 28450.0, 4510.0),
        row!(443, 6, 8_912_440, 30, 28, 18920.0, 3120.0),
        row!(443, 6, 15_100_221, 55, 52, 41020.0, 6210.0),
        row!(443, 6, 7_420_110, 25, 22, 14500.0, 2450.0),
        row!(443, 6, 11_210_500, 38, 35, 24900.0, 3980.0),
        row!(80, 6, 2_981_440, 8, 6, 1820.0, 450.0),
        row!(80, 6, 3_415_110, 9, 7, 2210.0, 510.0),
        row!(443, 6, 19_250_115, 72, 68, 58420.0, 8910.0),
        row!(443, 6, 14_802_330, 52, 48, 39110.0, 5980.0),
        row!(3389, 6, 1_402_669, 8, 7, 1945.6, 10.69),
        row!(3389, 6, 1_391_420, 9, 8, 2015.4, 11.20),
        row!(3389, 6, 1_412_901, 7, 7, 1848.3, 9.72),
        row!(3389, 6, 1_405_330, 8, 7, 1942.0, 10.21),
        row!(3389, 6, 1_398_210, 8, 7, 1929.1, 10.45),
        row!(3389, 6, 1_409_550, 9, 8, 2042.8, 12.98),
        row!(3389, 6, 1_410_230, 8, 7, 1952.1, 11.15),
        row!(3389, 6, 1_389_910, 7, 7, 1842.2, 9.44),
        row!(445, 6, 850_114, 6, 5, 820.0, 140.0),
        row!(445, 6, 912_440, 7, 6, 980.0, 180.0),
        row!(445, 6, 790_110, 5, 4, 690.0, 110.0),
        row!(135, 6, 112_440, 4, 3, 320.0, 180.0),
        row!(139, 6, 420_110, 5, 4, 580.0, 110.0),
    ]
}

/// Seed rows transcribed from `spec_engine`'s synthetic CIC-IDS2018 dataset
/// ("1-20: Base Seeds"), one per label in that block for these categories.
fn bruteforce_seeds() -> Vec<RawRow> {
    vec![
        row!(8080, 6, 5_000_000, 6_000, 6_000, 3_200_000.0, 2_400.0), // "Brute Force -Web"
        row!(22, 6, 300_000_000, 180_000, 180_000, 5_760_000.0, 1_200.0), // "Brute Force -XSS"
        row!(21, 6, 200_000_000, 90_000, 90_000, 720_000.0, 900.0),  // "FTP-BruteForce"
    ]
}

fn bot_seeds() -> Vec<RawRow> {
    vec![
        row!(443, 6, 600_000_000, 120, 120, 3_200.0, 0.4), // "Bot"
    ]
}

fn dos_seeds() -> Vec<RawRow> {
    vec![
        row!(80, 6, 60_000_000, 9_000, 0, 72_000_000.0, 150_000.0), // "DoS attacks-Hulk"
        row!(80, 6, 120_000_000, 1_800, 400, 6_400_000.0, 18_333.0), // "DoS attacks-GoldenEye"
        row!(80, 17, 10_000_000, 500_000, 0, 400_000_000.0, 50_000_000.0), // "DDoS attacks-LOIC-HTTP"
        row!(80, 17, 5_000_000, 800_000, 0, 819_200_000.0, 160_000_000.0), // "DDoS attacks-LOIC-UDP"
        row!(80, 6, 200_000_000_000, 4, 1, 0.5, 0.00002), // "DoS attacks-SlowHTTPTest"
        row!(80, 6, 300_000_000_000, 3, 1, 0.3, 0.00001), // "DoS attacks-Slowloris"
    ]
}

/// Deterministically jitters a seed row +-`frac` on duration/packets/bytes,
/// to turn 1-6 real seed shapes per category into a train/test-sized set
/// without inventing a different shape. See module docs for why this is a
/// weaker evidentiary base than the real multi-row datasets.
fn jitter(seed: &RawRow, rng: &mut StdRng, frac: f64) -> RawRow {
    let mut f = |base: f64| -> f64 {
        let mult = 1.0 + rng.gen_range(-frac..frac);
        (base * mult).max(0.0)
    };
    RawRow {
        dst_port: seed.dst_port,
        protocol: seed.protocol,
        duration_us: f(seed.duration_us as f64) as i64,
        tot_fwd_pkts: f(seed.tot_fwd_pkts as f64) as u32,
        tot_bwd_pkts: f(seed.tot_bwd_pkts as f64) as u32,
        totlen_fwd: f(seed.totlen_fwd),
        totlen_bwd: f(seed.totlen_bwd),
    }
}

fn expand(seeds: &[RawRow], count: usize, seed: u64) -> Vec<RawRow> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|i| jitter(&seeds[i % seeds.len()], &mut rng, 0.15))
        .collect()
}

/// Splits a category's rows into (train, test) - roughly 70/30, in-order
/// (no shuffling needed: jitter already randomizes within each category).
fn split(rows: Vec<RawRow>) -> (Vec<RawRow>, Vec<RawRow>) {
    let split_at = (rows.len() * 7 / 10).max(1).min(rows.len().saturating_sub(1).max(1));
    let mut rows = rows;
    let test = rows.split_off(split_at.max(1));
    (rows, test)
}

#[test]
fn categorical_classifier_isolates_attack_vectors_from_benign_and_each_other() {
    let device = Default::default();

    let (benign_train, benign_test) = split(benign_rows());
    let (infil_train, infil_test) = split(infiltration_rows());
    let (brute_train, brute_test) = split(expand(&bruteforce_seeds(), 20, 1));
    let (bot_train, bot_test) = split(expand(&bot_seeds(), 16, 2));
    let (dos_train, dos_test) = split(expand(&dos_seeds(), 24, 3));

    let mut train_set = Vec::new();
    for r in &benign_train {
        train_set.push(LabeledFlow { features: to_features(r), category: Category::Benign });
    }
    for r in &infil_train {
        train_set.push(LabeledFlow { features: to_features(r), category: Category::Infiltration });
    }
    for r in &brute_train {
        train_set.push(LabeledFlow { features: to_features(r), category: Category::BruteForce });
    }
    for r in &bot_train {
        train_set.push(LabeledFlow { features: to_features(r), category: Category::CommandAndControl });
    }
    for r in &dos_train {
        train_set.push(LabeledFlow { features: to_features(r), category: Category::DenialOfService });
    }

    let config = ClassifierConfig::default();
    let model = train::<TrainBackend>(&device, &config, &train_set, 800, 5e-3);
    let infer_model = model.valid();
    let infer_device: <InferBackend as burn::tensor::backend::Backend>::Device = Default::default();

    let test_groups: Vec<(&str, Category, &Vec<RawRow>)> = vec![
        ("Benign", Category::Benign, &benign_test),
        ("Infiltration", Category::Infiltration, &infil_test),
        ("BruteForce", Category::BruteForce, &brute_test),
        ("CommandAndControl (Bot)", Category::CommandAndControl, &bot_test),
        ("DenialOfService", Category::DenialOfService, &dos_test),
    ];

    // categories (rows) x categories (predicted) confusion matrix
    let mut confusion = [[0usize; siem_ml::classifier::NUM_CLASSES]; 5];
    let cat_order = [
        Category::Benign,
        Category::Infiltration,
        Category::BruteForce,
        Category::CommandAndControl,
        Category::DenialOfService,
    ];

    println!("\n--- per-flow predictions ---");
    for (row_idx, (label, true_cat, rows)) in test_groups.iter().enumerate() {
        for r in rows.iter() {
            let pred = predict::<InferBackend>(&infer_model, &infer_device, to_features(r));
            println!(
                "  true={label:<24} predicted={:<18} confidence={:.3}{}",
                pred.category.name(),
                pred.confidence,
                if pred.category == *true_cat { "" } else { "  <-- MISCLASSIFIED" }
            );
            if let Some(col) = cat_order.iter().position(|c| *c == pred.category) {
                confusion[row_idx][col] += 1;
            }
        }
    }

    println!("\n--- confusion matrix (rows=true, cols=predicted) ---");
    println!(
        "{:<20}{:<12}{:<14}{:<12}{:<20}{:<16}",
        "", "Benign", "Infiltrat.", "BruteForce", "CommandAndControl", "DoS"
    );
    for (i, (label, _, _)) in test_groups.iter().enumerate() {
        println!(
            "{:<20}{:<12}{:<14}{:<12}{:<20}{:<16}",
            label, confusion[i][0], confusion[i][1], confusion[i][2], confusion[i][3], confusion[i][4]
        );
    }

    let total: usize = test_groups.iter().map(|(_, _, rows)| rows.len()).sum();
    let correct: usize = test_groups
        .iter()
        .enumerate()
        .map(|(i, (_, true_cat, _))| {
            let col = cat_order.iter().position(|c| c == true_cat).unwrap();
            confusion[i][col]
        })
        .sum();
    let overall_accuracy = correct as f32 / total.max(1) as f32;
    println!("\noverall test accuracy: {correct}/{total} ({:.1}%)", overall_accuracy * 100.0);

    // Real evidence (Benign, Infiltration) should separate well - they're
    // multi-row real data, not jittered single seeds.
    assert!(
        !benign_test.is_empty() && !infil_test.is_empty(),
        "sanity check: real-data test splits must be non-empty"
    );

    // The bar here is deliberately modest given the jittered categories'
    // thin seed base (see module docs) - this checks the categorical
    // approach is directionally working, not that it's production-grade.
    assert!(
        overall_accuracy > 0.6,
        "expected >60% overall categorical accuracy across all 5 categories; got {:.1}%",
        overall_accuracy * 100.0
    );
}
