//! Pre-trains on the sandbox and prints held-out behaviour: per-class
//! action mix, policy probabilities for canonical cases, and regret against
//! the legacy rule and the oracle. Run:
//! `cargo run --release -p rl_decision --example sandbox_report`
use std::net::{IpAddr, Ipv4Addr};

use rl_decision::sandbox::ScenarioGen;
use rl_decision::state::idx;
use rl_decision::{loss, legacy_action, Mode, RlConfig, RlDecisionLayer, StateFeatures, ALL_ACTIONS, STATE_DIM};

fn main() {
    let layer = RlDecisionLayer::new(RlConfig { mode: Mode::Shadow, ..RlConfig::default() });
    let before = layer.anchor_loss();
    layer.pretrain_in_sandbox(3000, 17);
    println!("anchor loss (lower is better): {before:.3} -> {:.3}", layer.anchor_loss());

    let mut x = [0.0; STATE_DIM];
    x[idx::HONEYTOKEN] = 1.0;
    x[idx::CONFIDENCE] = 1.0;
    x[idx::SEVERITY] = 1.0;
    x[idx::LANES] = 1.0 / 6.0;
    let ht = StateFeatures::from_parts(x, 3, 1);
    let d = layer.decide_state(IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1)), ht);
    println!("honeytoken: p={:.3} policy={:?}", d.p_malicious, d.policy_probs.map(|p| (p * 1000.0).round() / 1000.0));
    println!("            E[loss]={:?} Q_sandbox={:?} -> {:?}", d.expected_loss.map(|v| (v * 100.0).round() / 100.0), d.q_sandbox.map(|v| (v * 1000.0).round() / 1000.0), d.action);

    let mut gen = ScenarioGen::new(31337);
    let (mut rl, mut leg, mut orc) = (0.0, 0.0, 0.0);
    let mut mix = [[0u32; 4]; 2];
    let n = 4000;
    for i in 0..n {
        let (s, y) = gen.sample();
        let h = IpAddr::V4(Ipv4Addr::new(10, (i >> 16) as u8, (i >> 8) as u8, i as u8));
        let d = layer.decide_state(h, s);
        mix[usize::from(y)][d.action.index()] += 1;
        rl += loss(d.action, y, s.severity, s.protected);
        leg += loss(legacy_action(s.severity), y, s.severity, s.protected);
        orc += ALL_ACTIONS.iter().map(|a| loss(*a, y, s.severity, s.protected)).fold(f64::INFINITY, f64::min);
    }
    let f = n as f64;
    println!("mean loss over {n} fresh cases: RL {:.3}  legacy {:.3}  oracle {:.3}", rl / f, leg / f, orc / f);
    println!("action mix [Escalate, Short, Standard, Extended]  benign: {:?}  malicious: {:?}", mix[0], mix[1]);
}
