//! Scenario 6: the RL decision layer.
//!
//! Shows, in order:
//!  1. sandbox pre-training (held-out anchor loss before/after),
//!  2. a labelled stream of attacks and benign look-alikes flowing through
//!     the real correlation engine and control plane while the layer runs in
//!     **Shadow** (it decides and is scored, but changes nothing),
//!  3. its measured regret against the legacy severity-tier rule and the
//!     promotion recommendation,
//!  4. operator promotion to **Active**, and what it does to a fresh burst
//!     (tiered bans; a protected host still goes only to a human).
//!
//! Labels here come from the demo itself knowing which hosts it made
//! malicious -- standing in for the human analyst / honeytoken labels a real
//! deployment would supply.

use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;
use std::time::Duration;

use correlation_engine::{FlowKey, HyperedgeKind, LaneSenders, Protocol};
use ips_control_plane::IpsControlPlane;
use review_queue::prelude::Verdict;
use rl_decision::{Mode, Recommendation, RlConfig, RlDecisionLayer};
use rustwall::quarantine::Quarantine;

fn host(n: usize) -> IpAddr {
    IpAddr::V4(Ipv4Addr::new(10, 6, (n / 250) as u8, (n % 250) as u8 + 1))
}

/// Builds the layer, pre-trains it in the sandbox, attaches it in Shadow.
pub fn attach(plane: &Arc<IpsControlPlane>) -> Arc<RlDecisionLayer> {
    let layer = Arc::new(RlDecisionLayer::new(RlConfig { mode: Mode::Shadow, train_every: 12, ..RlConfig::default() }));
    let before = layer.anchor_loss();
    let rep = layer.pretrain_in_sandbox(3000, 17);
    println!(
        "rl_decision: sandbox pre-training on 3000 synthetic cases, held-out anchor loss {before:.2} -> {:.2} (mode=Shadow)",
        rep.anchor_after
    );
    plane.set_decision_layer(layer.clone());
    layer
}

fn flow(h: IpAddr) -> FlowKey {
    FlowKey { src_ip: h, dst_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 50)), src_port: 40000, dst_port: 443, protocol: Protocol::Tcp }
}

/// Submits the lane events for one case. Attacks: strong, corroborated
/// signal (or a honeytoken). Benign look-alikes: a flash crowd that two
/// lanes find mildly odd but nothing else supports.
fn submit_case(senders: &LaneSenders, h: IpAddr, attack: bool, kind: usize, tag: usize) {
    if attack {
        match kind % 3 {
            0 => senders.honeytoken.submit(h, format!("tok-rl-{tag}"), None),
            1 => {
                senders.nsm.submit(h, "synflood", "SYN flood", nsm::alert::Severity::High);
                senders.hypergraph.submit(&[h], format!("rl-win-{tag}"), HyperedgeKind::DestinationFanIn, 6.0, None, Some(3.0), 0.7, 40);
            }
            _ => {
                senders.nsm.submit(h, "portscan", "port scan", nsm::alert::Severity::Medium);
                senders.spectral.submit(h, flow(h), 0.75, 0.0);
            }
        }
    } else {
        senders.spectral.submit(h, flow(h), 0.5, 0.0);
        senders.hypergraph.submit(&[h], format!("rl-win-{tag}"), HyperedgeKind::DestinationFanIn, 1.0, None, None, 0.2, 12);
    }
}

/// Plays the human analyst: for every pending item, record the verdict that
/// matches the ground truth the demo knows.
fn analyst_pass(plane: &IpsControlPlane, truth: &std::collections::HashMap<IpAddr, bool>) -> usize {
    let mut n = 0;
    for item in plane.queue().list_pending() {
        let id = item.prediction.flow_id.clone();
        // flow ids look like "corr-<ip>-<seq>"; match the whole ip token
        // ("10.6.0.1" is a prefix of "10.6.0.12", so a bare `contains` mislabels)
        let h = truth.iter().find(|(h, _)| id.starts_with(&format!("corr-{h}-"))).map(|(h, a)| (*h, *a));
        if let Some((_, attack)) = h {
            let _ = plane.queue().claim(&id, "analyst-rl");
            let v = if attack { Verdict::ExecuteContainment } else { Verdict::Dismiss };
            if plane.record_human_decision(&id, "analyst-rl", v, "demo ground truth").is_ok() {
                n += 1;
            }
        }
    }
    n
}

pub async fn run(plane: &Arc<IpsControlPlane>, senders: &LaneSenders, quarantine: &Arc<Quarantine>, layer: &Arc<RlDecisionLayer>) {
    println!("\n=== scenario 6: RL decision layer ===");
    let mut truth = std::collections::HashMap::new();

    // Phase A: Shadow. 90 cases, ~40% attacks.
    for i in 0..90usize {
        let h = host(i);
        let attack = i % 5 < 2;
        truth.insert(h, attack);
        submit_case(senders, h, attack, i / 5, i);
        if i % 10 == 9 {
            tokio::time::sleep(Duration::from_millis(60)).await;
            analyst_pass(plane, &truth);
        }
    }
    tokio::time::sleep(Duration::from_millis(150)).await;
    analyst_pass(plane, &truth);

    let st = layer.stats();
    println!(
        "Shadow phase: decisions={} labelled outcomes={} (matched to decisions; {} unmatched)",
        st.decisions, st.labels, st.unmatched_feedback
    );
    println!(
        "  mean regret over last {} labels: RL layer {:.2} vs legacy severity tiers {:.2}  (lower is better; 0 = oracle)",
        st.window_labels, st.mean_regret_rl, st.mean_regret_legacy
    );
    println!(
        "  training steps={} guard reverts={} risk-model Brier={:.3} uncertainty={:.2}",
        st.trainings, st.guard_reverts, st.brier, st.uncertainty
    );
    let rec = layer.recommendation();
    println!("  promotion recommendation: {rec:?}  (an operator decides; the layer never promotes itself)");

    // Phase B: operator promotes step by step if (and only if) recommended.
    if let Recommendation::Promote(m) = rec {
        layer.set_mode(m);
        println!("  operator: promoted to {m:?}");
    }
    layer.set_mode(Mode::Active);
    println!("  operator: Active for the demo (recommendation above shows what the evidence supports)");

    let gateway = IpAddr::V4(Ipv4Addr::new(10, 6, 200, 1));
    plane.protect_hosts([gateway]);
    let before_bans = quarantine.bans_total();
    let (atk, ben) = (host(300), host(301));
    truth.insert(atk, true);
    truth.insert(ben, false);
    submit_case(senders, atk, true, 0, 300); // honeytoken
    submit_case(senders, ben, false, 0, 301); // flash-crowd look-alike
    senders.honeytoken.submit(gateway, "tok-gateway", None); // touch on a protected host
    tokio::time::sleep(Duration::from_millis(250)).await;
    println!(
        "Active phase: honeytoken attacker banned={}, flash-crowd look-alike banned={}, protected gateway banned={}",
        quarantine.is_active(atk),
        quarantine.is_active(ben),
        quarantine.is_active(gateway)
    );
    println!("  new bans this phase: {} (gateway went to a human: pending={})", quarantine.bans_total() - before_bans, plane.queue().stats().pending);
    let st = layer.stats();
    println!(
        "  action mix so far: Escalate={} BanShort={} BanStandard={} BanExtended={}; shield overrides={}; auto-demotions={}",
        st.action_counts[0], st.action_counts[1], st.action_counts[2], st.action_counts[3], st.shield_overrides, st.auto_demotions
    );
    layer.set_mode(Mode::Shadow);
}
