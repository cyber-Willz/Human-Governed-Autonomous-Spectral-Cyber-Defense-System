use std::net::{IpAddr, Ipv4Addr};
use std::time::{Duration, Instant};

use correlation_engine::{
    CorrelationEvent, CorrelationReason, CorrelationVerdict, FeedbackKind, FeedbackObserver, FeedbackSource,
    LaneDetail, LaneSource, Severity,
};
use rl_decision::sandbox::ScenarioGen;
use rl_decision::selector::{shield, ShieldConfig};
use rl_decision::state::idx;
use rl_decision::{loss, Action, Mode, Recommendation, RlConfig, RlDecisionLayer, StateFeatures, STATE_DIM};

fn ip(n: u8) -> IpAddr {
    IpAddr::V4(Ipv4Addr::new(10, 1, 0, n))
}

fn host_n(i: usize) -> IpAddr {
    IpAddr::V4(Ipv4Addr::new(10, 2, (i / 250) as u8, (i % 250) as u8 + 1))
}

fn honeytoken_verdict(h: IpAddr) -> CorrelationVerdict {
    CorrelationVerdict {
        host: h,
        severity: Severity::Critical,
        confidence: 1.0,
        sources: vec![LaneSource::Honeytoken],
        evidence: vec![CorrelationEvent {
            host: h,
            source: LaneSource::Honeytoken,
            severity: Severity::Critical,
            confidence: 1.0,
            observed_at: Instant::now().into(),
            detail: LaneDetail::Honeytoken { token_id: "t".into(), campaign_window_id: None },
        }],
        reason: CorrelationReason::SingleLaneCritical,
        decided_at: Instant::now().into(),
        nearest_known_incident: None,
    }
}

fn pretrained() -> RlDecisionLayer {
    let layer = RlDecisionLayer::new(RlConfig { mode: Mode::Active, ..RlConfig::default() });
    layer.pretrain_in_sandbox(3000, 42);
    layer
}

#[test]
fn cost_model_orders_actions_sensibly() {
    // certain attack -> longest ban is cheapest; certain benign -> escalate is cheapest
    let best = |mal: bool| {
        *[Action::Escalate, Action::BanShort, Action::BanStandard, Action::BanExtended]
            .iter()
            .min_by(|a, b| loss(**a, mal, 2, false).partial_cmp(&loss(**b, mal, 2, false)).unwrap())
            .unwrap()
    };
    assert_eq!(best(true), Action::BanExtended);
    assert_eq!(best(false), Action::Escalate);
    // protected hosts make wrongful bans costlier
    assert!(loss(Action::BanStandard, false, 2, true) > loss(Action::BanStandard, false, 2, false));
}

#[test]
fn shield_rules() {
    let cfg = ShieldConfig::default();
    let mut x = [0.0; STATE_DIM];
    x[idx::PROTECTED] = 1.0;
    let protected = StateFeatures::from_parts(x, 3, 3);
    let sh = shield(&protected, 0.99, 0.0, &cfg);
    assert_eq!(sh.allowed, [true, false, false, false], "protected: human review only");

    let mut x = [0.0; STATE_DIM];
    x[idx::HONEYTOKEN] = 1.0;
    let ht = StateFeatures::from_parts(x, 3, 1);
    let sh = shield(&ht, 0.1, 0.9, &cfg);
    assert!(!sh.allowed[Action::Escalate.index()] && !sh.allowed[Action::BanShort.index()]);
    assert!(sh.allowed[Action::BanStandard.index()]);

    let plain = StateFeatures::from_parts([0.3; STATE_DIM], 1, 2);
    let low = shield(&plain, 0.1, 0.0, &cfg);
    assert_eq!(low.allowed, [true, false, false, false], "low risk: no bans");
    let one_lane = StateFeatures::from_parts([0.3; STATE_DIM], 2, 1);
    assert!(!shield(&one_lane, 0.95, 0.0, &cfg).allowed[Action::BanExtended.index()], "extended needs 2 lanes");
    assert!(!shield(&plain, 0.95, 0.99, &cfg).allowed[Action::BanExtended.index()], "extended needs low uncertainty");
    assert!(shield(&plain, 0.95, 0.1, &cfg).allowed[Action::BanExtended.index()]);
}

#[test]
fn sandbox_pretraining_reduces_anchor_loss() {
    let layer = RlDecisionLayer::new(RlConfig::default());
    let before = layer.anchor_loss();
    let rep = layer.pretrain_in_sandbox(3000, 7);
    let after = layer.anchor_loss();
    println!("anchor loss before={before:.3} after={after:.3} ({rep:?})");
    assert!(after < before, "pretraining should help on held-out sandbox cases");
}

#[test]
fn pretrained_layer_beats_legacy_regret_on_a_fresh_stream() {
    let layer = pretrained();
    let mut gen = ScenarioGen::new(99);
    for i in 0..600 {
        let (s, y) = gen.sample();
        let h = host_n(i);
        layer.decide_state(h, s);
        layer.record_label(h, y);
    }
    let st = layer.stats();
    println!("{st:?}");
    assert!(st.window_labels > 0);
    assert!(
        st.mean_regret_rl < st.mean_regret_legacy,
        "rl regret {:.3} should beat legacy {:.3}",
        st.mean_regret_rl,
        st.mean_regret_legacy
    );
}

#[test]
fn honeytoken_is_always_contained_unless_protected() {
    let layer = pretrained();
    for i in 0..50 {
        let d = layer.decide(&honeytoken_verdict(host_n(i)), false);
        assert!(d.action.is_ban() && d.action != Action::BanShort, "got {:?}", d.action);
    }
    let d = layer.decide(&honeytoken_verdict(ip(1)), true);
    assert_eq!(d.action, Action::Escalate, "protected host is escalated, never banned");
    assert!(d.directive().force_review);
    assert!(d.directive().ban_duration.is_none());
}

#[test]
fn modes_limit_what_a_decision_can_do() {
    let layer = pretrained();
    let v = honeytoken_verdict(ip(2));
    layer.set_mode(Mode::Shadow);
    let d = layer.decide(&v, false);
    assert!(!d.directive().force_review && d.directive().ban_duration.is_none(), "shadow changes nothing");
    layer.set_mode(Mode::Advisory);
    let d = layer.decide(&v, false);
    assert!(d.directive().ban_duration.is_none(), "advisory cannot change durations");
    layer.set_mode(Mode::Active);
    let d = layer.decide(&v, false);
    assert!(d.directive().ban_duration.unwrap() >= Duration::from_secs(2 * 3600));
}

#[test]
fn feedback_matches_pending_decision_and_unmatched_is_counted() {
    let layer = RlDecisionLayer::new(RlConfig::default());
    layer.decide(&honeytoken_verdict(ip(3)), false);
    layer.on_feedback(ip(3), FeedbackKind::Confirmed, FeedbackSource::GroundTruthLane, &[LaneSource::Honeytoken]);
    layer.on_feedback(ip(4), FeedbackKind::Dismissed, FeedbackSource::Human, &[LaneSource::Nsm]);
    let st = layer.stats();
    assert_eq!(st.labels, 1);
    assert_eq!(st.unmatched_feedback, 1);
    // confirmed history now feeds the state of the next verdict for that host
    let d = layer.decide(&honeytoken_verdict(ip(3)), false);
    assert!(d.p_malicious > 0.9);
}

#[test]
fn guard_reverts_updates_that_worsen_the_anchor_set() {
    // Poison: every honeytoken/critical case labelled benign. Unprotected
    // learning on this would push toward Escalate, hurting anchor loss.
    let layer = RlDecisionLayer::new(RlConfig { train_every: 8, guard_tolerance: 0.0, ..RlConfig::default() });
    layer.pretrain_in_sandbox(3000, 5);
    let base = layer.anchor_loss();
    let mut gen = ScenarioGen::new(1234);
    let mut n = 0;
    while n < 160 {
        let (s, y) = gen.sample();
        if y {
            let h = host_n(n);
            layer.decide_state(h, s);
            layer.record_label(h, false); // poisoned label
            n += 1;
        }
    }
    let st = layer.stats();
    let after = layer.anchor_loss();
    println!("trainings={} reverts={} base={base:.3} after={after:.3}", st.trainings, st.guard_reverts);
    assert!(st.guard_reverts > 0, "guard should have rejected some poisoned updates");
    // zero tolerance means no *accepted* update may have worsened the anchors
    assert!(after <= base + 1e-9, "accepted updates must not worsen anchor loss: {base} -> {after}");
}

#[test]
fn auto_demotion_when_rl_is_worse_than_legacy() {
    // Strong-looking, multi-lane, Medium-severity cases: the layer bans,
    // the legacy rule (Medium -> no ban) escalates. If every one of them
    // turns out benign, the layer is measurably worse and must fail safe.
    let layer = RlDecisionLayer::new(RlConfig { mode: Mode::Active, train_every: 100_000, ..RlConfig::default() });
    let mut x = [0.0; STATE_DIM];
    x[idx::CONFIDENCE] = 0.95;
    x[idx::HYPERGRAPH] = 0.9;
    x[idx::NSM] = 1.0;
    x[idx::SPECTRAL] = 0.8;
    x[idx::SEVERITY] = 1.0 / 3.0;
    x[idx::LANES] = 0.5;
    let state = StateFeatures::from_parts(x, 1, 3);
    let mut i = 0;
    let mut banned_any = false;
    while layer.mode() == Mode::Active && i < 400 {
        let h = host_n(i);
        let d = layer.decide_state(h, state);
        banned_any |= d.action.is_ban();
        layer.record_label(h, false);
        i += 1;
    }
    assert!(banned_any, "test premise: the layer must actually have banned");
    assert_eq!(layer.mode(), Mode::Shadow, "should fail safe to shadow");
    assert!(layer.stats().auto_demotions >= 1);
}

#[test]
fn recommendation_promotes_only_with_evidence() {
    let layer = pretrained();
    layer.set_mode(Mode::Shadow);
    assert_eq!(layer.recommendation(), Recommendation::Hold, "no evidence yet");
    let mut gen = ScenarioGen::new(2024);
    for i in 0..120 {
        let (s, y) = gen.sample();
        let h = host_n(i);
        layer.decide_state(h, s);
        layer.record_label(h, y);
    }
    assert_eq!(layer.recommendation(), Recommendation::Promote(Mode::Advisory));
}

#[test]
fn save_and_load_round_trip() {
    let a = pretrained();
    let dir = std::env::temp_dir().join(format!("rl_decision_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.json");
    a.save(&path).unwrap();
    let b = RlDecisionLayer::new(RlConfig::default());
    assert!((a.anchor_loss() - b.anchor_loss()).abs() > 1e-9);
    b.load(&path).unwrap();
    assert!((a.anchor_loss() - b.anchor_loss()).abs() < 1e-12);
    std::fs::remove_dir_all(&dir).ok();
}
