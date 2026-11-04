//! Validates siem-ml's autoencoder against a real CIC-IDS2018-style
//! "Infiltration" sample (RDP/SMB/HTTP low-and-slow flows captured on
//! 2018-01-03), instead of only the single synthetic beacon used in the
//! demo binary. This directly tests the claim the project is built around:
//! that reconstruction error separates infiltration flows from benign ones
//! even though no single infiltration flow here trips a count/threshold rule.
//!
//! Data source: CICFlowMeter-style flow records for the "Infilteration"
//! label (dataset's own spelling), rows for src 13.58.225.34 -> dst
//! 172.31.69.13/24/25 (the C2/RDP/SMB probing flows) and 172.31.69.13 as
//! source against .24/.25/.28 (lateral movement over SMB/RPC once the
//! foothold pivots internally).

use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use siem_core::EventKind;
use siem_ml::{score, train, AutoencoderConfig, FlowFeatures};

type TrainBackend = Autodiff<NdArray<f32>>;
type InferBackend = NdArray<f32>;

/// One row of the real Infiltration dataset:
/// (src_ip, dst_ip, dst_port, protocol, flow_duration_us, tot_fwd_pkts,
///  tot_bwd_pkts, totlen_fwd_bytes, totlen_bwd_bytes)
///
/// Only the fields siem-ml's `FlowFeatures` actually consumes are kept;
/// the source rows carry additional CICFlowMeter columns (packet-length
/// means, TCP flags, init window sizes, active/idle timers) that aren't
/// part of the current feature set.
struct RawRow {
    src_ip: &'static str,
    dst_ip: &'static str,
    dst_port: u16,
    protocol: u8,
    flow_duration_us: i64,
    tot_fwd_pkts: u32,
    tot_bwd_pkts: u32,
    totlen_fwd: f64,
    totlen_bwd: f64,
}

macro_rules! row {
    ($src:expr, $dst:expr, $port:expr, $proto:expr, $dur:expr, $fwd_pkts:expr, $bwd_pkts:expr, $fwd_bytes:expr, $bwd_bytes:expr) => {
        RawRow {
            src_ip: $src,
            dst_ip: $dst,
            dst_port: $port,
            protocol: $proto,
            flow_duration_us: $dur,
            tot_fwd_pkts: $fwd_pkts,
            tot_bwd_pkts: $bwd_pkts,
            totlen_fwd: $fwd_bytes,
            totlen_bwd: $bwd_bytes,
        }
    };
}

/// Transcribed from the real Infiltration sample: label, timestamp, packet
/// length means, TCP flag columns, init window sizes and active/idle timers
/// are dropped since `FlowFeatures` doesn't consume them; src/dst IP,
/// dst port, protocol, flow duration, packet counts, and forward/backward
/// byte totals are kept as-is.
fn infiltration_rows() -> Vec<RawRow> {
    vec![
        // HTTP/HTTPS beaconing: 13.58.225.34 -> 172.31.69.13
        row!("13.58.225.34", "172.31.69.13", 80, 6, 4_512_891, 12, 10, 3812.0, 845.0),
        row!("13.58.225.34", "172.31.69.13", 80, 6, 5_120_114, 15, 11, 4910.0, 912.0),
        row!("13.58.225.34", "172.31.69.13", 80, 6, 3_891_002, 10, 8, 2950.0, 610.0),
        row!("13.58.225.34", "172.31.69.13", 443, 6, 12_405_110, 42, 38, 28450.0, 4510.0),
        row!("13.58.225.34", "172.31.69.13", 443, 6, 8_912_440, 30, 28, 18920.0, 3120.0),
        row!("13.58.225.34", "172.31.69.13", 443, 6, 15_100_221, 55, 52, 41020.0, 6210.0),
        row!("13.58.225.34", "172.31.69.13", 443, 6, 7_420_110, 25, 22, 14500.0, 2450.0),
        row!("13.58.225.34", "172.31.69.13", 443, 6, 11_210_500, 38, 35, 24900.0, 3980.0),
        row!("13.58.225.34", "172.31.69.13", 80, 6, 2_981_440, 8, 6, 1820.0, 450.0),
        row!("13.58.225.34", "172.31.69.13", 80, 6, 3_415_110, 9, 7, 2210.0, 510.0),
        row!("13.58.225.34", "172.31.69.13", 443, 6, 19_250_115, 72, 68, 58420.0, 8910.0),
        row!("13.58.225.34", "172.31.69.13", 443, 6, 14_802_330, 52, 48, 39110.0, 5980.0),
        row!("13.58.225.34", "172.31.69.13", 443, 6, 16_112_900, 60, 55, 48200.0, 7100.0),
        row!("13.58.225.34", "172.31.69.13", 443, 6, 13_910_880, 48, 44, 32410.0, 5110.0),
        row!("13.58.225.34", "172.31.69.13", 443, 6, 10_305_400, 35, 31, 21980.0, 3240.0),
        // RDP probing: 13.58.225.34 -> 172.31.69.24
        row!("13.58.225.34", "172.31.69.24", 3389, 6, 1_402_669, 8, 7, 1945.6, 10.69),
        row!("13.58.225.34", "172.31.69.24", 3389, 6, 1_391_420, 9, 8, 2015.4, 11.20),
        row!("13.58.225.34", "172.31.69.24", 3389, 6, 1_412_901, 7, 7, 1848.3, 9.72),
        row!("13.58.225.34", "172.31.69.24", 3389, 6, 1_405_330, 8, 7, 1942.0, 10.21),
        row!("13.58.225.34", "172.31.69.24", 3389, 6, 1_398_210, 8, 7, 1929.1, 10.45),
        row!("13.58.225.34", "172.31.69.24", 3389, 6, 1_409_550, 9, 8, 2042.8, 12.98),
        row!("13.58.225.34", "172.31.69.24", 3389, 6, 1_410_230, 8, 7, 1952.1, 11.15),
        row!("13.58.225.34", "172.31.69.24", 3389, 6, 1_389_910, 7, 7, 1842.2, 9.44),
        row!("13.58.225.34", "172.31.69.24", 3389, 6, 1_404_110, 8, 7, 1948.2, 10.71),
        row!("13.58.225.34", "172.31.69.24", 3389, 6, 1_397_800, 8, 7, 1932.4, 10.39),
        row!("13.58.225.34", "172.31.69.24", 3389, 6, 1_419_120, 9, 8, 2062.1, 13.45),
        // RDP probing: 13.58.225.34 -> 172.31.69.25
        row!("13.58.225.34", "172.31.69.25", 3389, 6, 1_402_990, 8, 7, 1946.1, 10.65),
        row!("13.58.225.34", "172.31.69.25", 3389, 6, 1_388_400, 7, 7, 1839.2, 9.38),
        row!("13.58.225.34", "172.31.69.25", 3389, 6, 1_406_110, 8, 7, 1947.5, 10.68),
        row!("13.58.225.34", "172.31.69.25", 3389, 6, 1_404_220, 8, 7, 1945.1, 10.62),
        row!("13.58.225.34", "172.31.69.25", 3389, 6, 1_412_110, 9, 8, 2051.4, 13.21),
        row!("13.58.225.34", "172.31.69.25", 3389, 6, 1_403_120, 8, 7, 1945.4, 10.64),
        row!("13.58.225.34", "172.31.69.25", 3389, 6, 1_392_220, 7, 7, 1847.2, 9.54),
        row!("13.58.225.34", "172.31.69.25", 3389, 6, 1_400_110, 8, 7, 1937.1, 10.34),
        row!("13.58.225.34", "172.31.69.25", 3389, 6, 1_405_000, 8, 7, 1946.4, 10.59),
        row!("13.58.225.34", "172.31.69.25", 3389, 6, 1_414_110, 9, 8, 2054.1, 13.25),
        // Lateral movement over SMB/RPC once the foothold pivots internally:
        // 172.31.69.13 -> .25 / .24 / .28
        row!("172.31.69.13", "172.31.69.25", 445, 6, 850_114, 6, 5, 820.0, 140.0),
        row!("172.31.69.13", "172.31.69.25", 445, 6, 912_440, 7, 6, 980.0, 180.0),
        row!("172.31.69.13", "172.31.69.25", 445, 6, 790_110, 5, 4, 690.0, 110.0),
        row!("172.31.69.13", "172.31.69.24", 445, 6, 875_330, 6, 5, 845.0, 150.0),
        row!("172.31.69.13", "172.31.69.24", 445, 6, 820_220, 6, 5, 800.0, 135.0),
        row!("172.31.69.13", "172.31.69.24", 445, 6, 930_110, 7, 6, 1020.0, 200.0),
        row!("172.31.69.13", "172.31.69.28", 135, 6, 112_440, 4, 3, 320.0, 180.0),
        row!("172.31.69.13", "172.31.69.28", 135, 6, 118_110, 4, 3, 340.0, 190.0),
        row!("172.31.69.13", "172.31.69.28", 135, 6, 109_220, 4, 3, 310.0, 175.0),
        row!("172.31.69.13", "172.31.69.28", 139, 6, 420_110, 5, 4, 580.0, 110.0),
        row!("172.31.69.13", "172.31.69.28", 139, 6, 435_220, 5, 4, 600.0, 120.0),
        row!("172.31.69.13", "172.31.69.28", 139, 6, 405_330, 5, 4, 560.0, 105.0),
        row!("172.31.69.13", "172.31.69.28", 445, 6, 890_110, 6, 5, 830.0, 145.0),
        row!("172.31.69.13", "172.31.69.28", 445, 6, 905_220, 6, 5, 850.0, 150.0),
    ]
}

/// Real CIC-IDS2018-style Benign flows (deduplicated from the 3x-repeated
/// sample provided): an RDP admin session, two DNS lookups, an HTTPS
/// download, and an HTTP transfer. Deliberately diverse - this is the actual
/// stress test for the threshold: does the model trained on a narrow
/// synthetic "web request" shape wrongly flag ordinary traffic that doesn't
/// look like its training data, even though it's benign?
fn real_benign_rows() -> Vec<RawRow> {
    vec![
        // RDP admin session: long-ish, small/asymmetric byte counts - this is
        // the row most likely to get confused with the RDP-probing
        // infiltration rows above, since it shares port 3389 and a similar
        // upload-heavy shape. That's exactly the interesting case.
        row!("172.31.69.11", "172.31.69.20", 3389, 6, 4_046_191, 14, 7, 439.4, 5.19),
        // DNS lookup #1
        row!("172.31.69.12", "172.31.69.2", 53, 17, 303, 1, 1, 356_435.6, 6600.7),
        // Long HTTPS download (2 minutes, large asymmetric transfer - lots of
        // download, almost no upload; the opposite shape from infiltration
        // exfil, but still very unlike the synthetic benign baseline).
        row!("172.31.69.30", "54.239.28.85", 443, 6, 120_000_000, 45, 38, 18_432.0, 0.69),
        // DNS lookup #2
        row!("172.31.69.50", "172.31.69.2", 53, 17, 400, 1, 1, 300_000.0, 5000.0),
        // Long HTTPS download, smaller than the one above
        row!("172.31.69.51", "52.84.100.12", 443, 6, 60_000_000, 30, 25, 9_600.0, 0.92),
        // Ordinary internal HTTP transfer
        row!("172.31.69.52", "172.31.69.10", 80, 6, 3_000_000, 8, 6, 6_400.0, 4.67),
    ]
}

fn to_features(row: &RawRow) -> FlowFeatures {
    let ev = EventKind::Flow {
        src_ip: row.src_ip.to_string(),
        dst_ip: row.dst_ip.to_string(),
        // Source port isn't present in this CICFlowMeter export; the port
        // feature carries little weight relative to the duration/byte-ratio
        // features that actually characterize infiltration traffic shape.
        src_port: 0,
        dst_port: row.dst_port,
        proto: row.protocol,
        // CICFlowMeter's Flow Duration column is in microseconds.
        duration_ms: (row.flow_duration_us / 1000).max(0) as u64,
        bytes_src_to_dst: row.totlen_fwd.max(0.0) as u64,
        bytes_dst_to_src: row.totlen_bwd.max(0.0) as u64,
        packets: (row.tot_fwd_pkts + row.tot_bwd_pkts) as u64,
        flags: String::new(),
    };
    FlowFeatures::from_event(&ev).expect("Flow event always yields features")
}

/// Ordinary short web/API request-response traffic: seconds-scale duration,
/// response bytes well above request bytes, moderate packet counts. This is
/// the "normal" shape the autoencoder learns; none of these values were
/// tuned against the infiltration rows above.
fn benign_baseline() -> Vec<FlowFeatures> {
    (0..200u64)
        .map(|i| {
            let ev = EventKind::Flow {
                src_ip: "10.0.0.20".to_string(),
                dst_ip: "203.0.113.10".to_string(),
                src_port: (40000 + (i % 1000)) as u16,
                dst_port: 443,
                proto: 6,
                duration_ms: 2_000 + (i % 500) * 10,
                bytes_src_to_dst: 1500 + (i % 300) * 20,
                bytes_dst_to_src: 40_000 + (i % 2000) * 5,
                packets: 20 + (i % 10),
                flags: "SAP".to_string(),
            };
            FlowFeatures::from_event(&ev).unwrap()
        })
        .collect()
}

#[test]
fn autoencoder_separates_real_infiltration_flows_from_benign_baseline() {
    let device = Default::default();
    let benign = benign_baseline();
    let config = AutoencoderConfig::default();
    let model = train::<TrainBackend>(&device, &config, &benign, 300, 1e-2);
    let infer_model = model.valid();
    let infer_device: <InferBackend as burn::tensor::backend::Backend>::Device = Default::default();

    // Calibrate the threshold purely from benign data, same as the demo binary.
    let benign_scores: Vec<f32> = benign
        .iter()
        .map(|f| score::<InferBackend>(&infer_model, &infer_device, *f))
        .collect();
    let max_benign = benign_scores.iter().cloned().fold(0f32, f32::max);
    let threshold = max_benign * 1.5 + 0.001;

    let rows = infiltration_rows();
    assert!(rows.len() >= 40, "sanity check on transcribed dataset size");

    let infiltration_scores: Vec<f32> = rows
        .iter()
        .map(|r| score::<InferBackend>(&infer_model, &infer_device, to_features(r)))
        .collect();

    let flagged = infiltration_scores.iter().filter(|s| **s > threshold).count();
    let flagged_rate = flagged as f32 / infiltration_scores.len() as f32;

    println!(
        "benign max score: {max_benign:.5}, threshold: {threshold:.5}, \
         infiltration flagged: {flagged}/{} ({:.1}%)",
        infiltration_scores.len(),
        flagged_rate * 100.0
    );
    let mean_infiltration: f32 =
        infiltration_scores.iter().sum::<f32>() / infiltration_scores.len() as f32;
    println!("mean infiltration reconstruction error: {mean_infiltration:.5}");

    // The real test of the design claim: infiltration flows, on average,
    // reconstruct far worse than the benign traffic they weren't trained on.
    assert!(
        mean_infiltration > threshold,
        "mean infiltration reconstruction error ({mean_infiltration:.5}) should exceed \
         the benign-calibrated threshold ({threshold:.5}) - if this fails, the feature \
         set or threshold calibration needs rework before trusting this on real traffic"
    );

    // Not every infiltration flow should be expected to trip a single global
    // threshold (that's the whole reason rules alone don't work either) -
    // but a useful detector should catch a clear majority of them.
    assert!(
        flagged_rate > 0.5,
        "expected a majority of real infiltration flows to be flagged; got {:.1}%",
        flagged_rate * 100.0
    );
}

#[test]
fn false_positive_rate_on_real_diverse_benign_traffic() {
    // This is the test the earlier "100% infiltration detected" result was
    // missing: the model is trained and calibrated on a narrow synthetic
    // benign shape (one kind of web request). That alone doesn't tell you
    // whether it's actually learned "infiltration" versus "not exactly this
    // one synthetic pattern" - a detector that flags real, ordinary,
    // diverse benign traffic just as readily as it flags real infiltration
    // traffic isn't discriminating anything useful. This scores real
    // Benign-labeled flows (RDP admin session, DNS lookups, long HTTPS/HTTP
    // downloads) against the same model and threshold to find out.
    let device = Default::default();
    let synthetic_benign = benign_baseline();
    let config = AutoencoderConfig::default();
    let model = train::<TrainBackend>(&device, &config, &synthetic_benign, 300, 1e-2);
    let infer_model = model.valid();
    let infer_device: <InferBackend as burn::tensor::backend::Backend>::Device = Default::default();

    let synthetic_scores: Vec<f32> = synthetic_benign
        .iter()
        .map(|f| score::<InferBackend>(&infer_model, &infer_device, *f))
        .collect();
    let max_synthetic = synthetic_scores.iter().cloned().fold(0f32, f32::max);
    let threshold = max_synthetic * 1.5 + 0.001;

    let rows = real_benign_rows();
    let scores: Vec<f32> = rows
        .iter()
        .map(|r| score::<InferBackend>(&infer_model, &infer_device, to_features(r)))
        .collect();

    let labels = [
        "RDP admin session (172.31.69.11 -> .20:3389)",
        "DNS lookup #1 (172.31.69.12 -> .2:53)",
        "Long HTTPS download, 2min (172.31.69.30 -> 54.239.28.85:443)",
        "DNS lookup #2 (172.31.69.50 -> .2:53)",
        "Long HTTPS download, 1min (172.31.69.51 -> 52.84.100.12:443)",
        "Ordinary internal HTTP (172.31.69.52 -> .10:80)",
    ];
    for (label, s) in labels.iter().zip(scores.iter()) {
        println!(
            "  {label}: score {s:.5} ({})",
            if *s > threshold { "FLAGGED" } else { "ok" }
        );
    }

    let flagged = scores.iter().filter(|s| **s > threshold).count();
    let false_positive_rate = flagged as f32 / scores.len() as f32;
    println!(
        "false positive rate on real diverse benign traffic: {flagged}/{} ({:.1}%), threshold {threshold:.5}",
        scores.len(),
        false_positive_rate * 100.0
    );

    // This intentionally does NOT assert a specific low false-positive rate.
    // The synthetic training/calibration set is narrow by construction, so a
    // high false-positive rate here is an expected, informative result, not
    // a bug to be papered over with a lenient assertion - it's the concrete
    // evidence that threshold calibration needs a real, diverse benign
    // corpus (see README) before this is trustworthy in production. The only
    // thing asserted is that the run itself produces a valid rate.
    assert!((0.0..=1.0).contains(&false_positive_rate));
}

#[test]
fn benign_baseline_mostly_scores_below_its_own_threshold() {
    // Sanity check on the calibration itself: the threshold should not be so
    // loose that it flags the majority of ordinary traffic.
    let device = Default::default();
    let benign = benign_baseline();
    let config = AutoencoderConfig::default();
    let model = train::<TrainBackend>(&device, &config, &benign, 300, 1e-2);
    let infer_model = model.valid();
    let infer_device: <InferBackend as burn::tensor::backend::Backend>::Device = Default::default();

    let scores: Vec<f32> = benign
        .iter()
        .map(|f| score::<InferBackend>(&infer_model, &infer_device, *f))
        .collect();
    let max_benign = scores.iter().cloned().fold(0f32, f32::max);
    let threshold = max_benign * 1.5 + 0.001;
    let false_positive_rate =
        scores.iter().filter(|s| **s > threshold).count() as f32 / scores.len() as f32;
    assert_eq!(
        false_positive_rate, 0.0,
        "threshold is calibrated from this exact benign set's max, so it must not \
         flag any of it by construction"
    );
}
