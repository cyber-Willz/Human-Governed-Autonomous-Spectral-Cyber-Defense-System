//! How does the score grow with the number of *consecutive* quiet probe
//! bins appended to a benign background? (200 random backgrounds per k.)
use hcrnn::rng::Rng;
use neuro_lane::detector::{ConnectomeDetector, DetectorConfig};
use neuro_lane::synth::{self, Behavior};

fn main() {
    let mut d = ConnectomeDetector::new(DetectorConfig::default()).unwrap();
    d.load(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/models/default.params"))).unwrap();
    let t = d.cfg.seq_len;
    println!("k_probe_bins  mean_score  share>=0.5");
    for k in 1..=16usize {
        let mut rng = Rng::new(1000 + k as u64);
        let (mut sum, mut hit) = (0.0, 0);
        for _ in 0..200 {
            let mut seq = synth::sequence(&mut rng, Behavior::Benign, t, 0.0).0;
            let probe = synth::sequence(&mut rng, Behavior::LowSlowScan, t, 0.0).0;
            for i in 0..k {
                seq[t - 1 - i] = probe[i];
            }
            let s = d.score(&seq);
            sum += s;
            hit += (s >= 0.5) as usize;
        }
        println!("{k:>12}  {:>10.3}  {:>9.2}", sum / 200.0, hit as f64 / 200.0);
    }
}
