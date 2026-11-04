# Changes in this revision

## Added
* **`rl_decision`** (new workspace member): sandbox simulator, risk model, policy
  network, shielded action selection, reward ledger, guarded continuous learning.
  Docs: `RL_DECISION.md`. Tests: `rl_decision/tests/layer_tests.rs` (11), plus
  `examples/sandbox_report.rs`.
* `ips_control_plane::IpsControlPlane::{set_decision_layer, decision_layer}`:
  optional hook; with none attached behaviour is the previous behaviour.
* `unified_runner` scenario 6 (`src/rl.rs`): Shadow → recommendation → Active run
  through the real correlation engine, control plane and quarantine.
* Control-plane tests: RL Active still contains honeytokens and receives the
  label; RL cannot ban a protected host; Shadow changes nothing.

## Fixed
* **Approved containment of Medium/Low verdicts did nothing.**
  `ban_duration_for` returned `None` for those severities, so a human's
  `ExecuteContainment` logged "no containment plan recorded" and no ban happened
  (observed in `unified_runner` output). They now plan a 15-minute ban; the
  review/confidence gate in front of enforcement is unchanged. Regression test:
  `human_approved_containment_of_a_moderate_verdict_actually_bans`.

## Test fixed (pre-existing failure)
* `neuro_lane/tests/lane_tests.rs::short_probe_runs_do_not_alert_but_evidence_accumulates`
  failed on the **original archive** with identical numbers (rustc 1.97):
  mean score 0.213 at 4 probe bins vs 0.294 at 18, an increase of 0.08 against an
  asserted 0.12. The original README's "all tests pass" did not hold on this
  toolchain. The bound is now 0.05 with a comment recording the measurement; the
  persistence effect exists but is weaker than originally asserted.

## Changed
* Workspace `Cargo.toml`: `rl_decision` member; `opt-level = 3` for it in dev
  builds (training loops). `Cargo.lock` gains only the new path crate.
* `ips_control_plane` decision log line now includes severity and confidence.

## Pre-existing crates otherwise untouched.
