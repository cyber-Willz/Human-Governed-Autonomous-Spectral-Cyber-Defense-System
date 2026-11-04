//! Trains the default detector on synthetic behaviours, reports held-out
//! metrics (including a distribution-shifted set and a threshold-rule
//! baseline) and writes `models/default.params`.
//!
//! ```text
//! cargo run --release -p neuro_lane --bin neuro_train -- [steps] [out_path]
//! ```
use neuro_lane::detector::{ConnectomeDetector, DetectorConfig};
use neuro_lane::synth::{Behavior, ALL};

fn main() {
    let mut a = std::env::args().skip(1);
    let steps: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(3000);
    let out = a.next().unwrap_or_else(|| "neuro_lane/models/default.params".into());
    let mut d = ConnectomeDetector::new(DetectorConfig::default()).expect("detector");
    println!("params: {}", d.net.params.len());
    let s = d.train(steps, 16, 6e-3, 1);
    println!("trained {} steps in {:.1}s, smoothed train loss {:.4}", s.steps, s.seconds, s.final_train_loss);
    for (name, shift) in [("in-distribution", 0.0), ("shifted (ranges widened)", 2.0)] {
        let set = d.eval_set(300, 0xE7A1 + shift as u64, shift);
        println!("\n[{name}] n={}  accuracy@0.5 = {:.3}", set.len(), d.accuracy(&set, 0.5));
        for b in ALL {
            let sub: Vec<_> = set.iter().filter(|(_, _, x)| *x == b).collect();
            let hit = sub.iter().filter(|(s, y, _)| (d.score(s) >= 0.5) == *y).count();
            // threshold-rule baseline: "some bin has port diversity >= 0.7" (what a per-window count rule sees)
            let rule = sub.iter().filter(|(s, _, _)| s.iter().any(|f| f[2] >= 0.7)).count();
            println!(
                "  {:<17?} correct {:>3}/{:<3}   (baseline rule flags {:>3}/{:<3})",
                b,
                hit,
                sub.len(),
                rule,
                sub.len()
            );
        }
    }
    let _ = Behavior::Benign;
    d.save(std::path::Path::new(&out)).expect("save");
    println!("\nsaved {out}");
}
