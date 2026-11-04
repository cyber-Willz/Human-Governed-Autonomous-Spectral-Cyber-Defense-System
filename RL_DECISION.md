# RL decision layer (`rl_decision`)

The unified architecture, end to end:

```
Network telemetry
   │
   ▼
NSM / NDR layer ............ nsm (scan · DNS · beacon · signature · SYN flood), XDP fast path, rust-clam, honeypot_engine
   │
   ▼
Hypergraph analytics ....... hg_builder + spectral_hypergraph + nbsc + spectral_jacobi_ds + neural_hmm
   │
   ▼
Connectome-LSTM lane ....... neuro_lane (+ llm_bridge advisory channel)
   │
   ▼
Correlation engine ......... correlation_engine  (≥2 lanes, or a single Critical lane)
   │
   ▼
┌──────────────────────────────── rl_decision ────────────────────────────────┐
│  state features ─┬─► Risk model  (calibrated P(malicious), expected loss)    │
│  (numeric only)  ├─► Sandbox sim (Monte-Carlo look-ahead per action)         │
│                  └─► Policy net  (MLP, natural policy gradient)              │
│                              │                                               │
│        shield (hard rules) ──┴─► Action selection ─► Decision                │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
IPS control plane ......... ips_control_plane  → review_queue gate → rustwall::Quarantine
   │                                                        (Escalate | Ban 15 min | 2 h | 24 h)
   ▼
Quarantine / containment
   │
   ▼
Feedback & rewards ........ human verdicts + honeytoken-backed containment → labels → reward ledger
   │
   ▼
RL continuous learning .... replay buffer → guarded updates (anchor-set check, bit-exact rollback)
```

Everything above the dashed box already existed in the archive; `rl_decision`
and its wiring are what was added, and the five boxes of the original
sketch map onto it as follows.

| Diagram box | Where | What it really is |
|---|---|---|
| Sandbox Sim | `sandbox.rs` | Stochastic model of "what happens if I take action X on this host" (attacker returns when a short ban lapses, repeat offenders return more, weakly-evidenced bans are evaded) plus a generator of labelled attacks *and benign look-alikes*. Holds no handle to any real resource. |
| Risk Model | `risk.rs` | Online logistic regression for `P(malicious)`, regularised toward hand-set prior weights; per-action expected loss; an uncertainty proxy that shrinks as real labels arrive. |
| Policy Net | `policy.rs` | 13→24→4 tanh MLP, hand-written backprop + Adam, no ML framework. |
| Action Selection | `selector.rs` | A **shield** that removes disallowed actions first, then scores the rest by `(1-w)·(−E[loss]/10) + w·Q_sandbox + w_policy·ln(π+0.05)`. |
| Feedback & Rewards / RL Continuous Learning | `layer.rs`, `action.rs` | One cost model prices actions everywhere; labels arrive via the existing `FeedbackObserver` hook; training is guarded. |

## What the layer can and cannot do

Actions: `Escalate` (route to a human, no autonomous ban), `BanShort` (15 min),
`BanStandard` (2 h), `BanExtended` (24 h).

**Authority ladder** (default `Shadow`; promotion is an operator call):

| Mode | Effect |
|---|---|
| `Shadow` | Decides, logs, is scored against later labels. Changes nothing. |
| `Advisory` | May only route a verdict to a human (`Escalate`). |
| `Active` | Also chooses the ban tier. |

In **no** mode can it: bypass `review_queue`; raise a verdict's confidence;
ban a protected or non-routable host; unban anything; or act on attacker-controlled
text (inputs are numeric lane outputs only). The shield additionally forces:
protected host → human review only; honeytoken → at least `BanStandard`; any ban
needs `P(malicious) ≥ 0.30`; `BanExtended` needs ≥ 2 lanes, `P ≥ 0.80` and low model
uncertainty. If measured regret against the legacy rule worsens by 25 % (after ≥ 50
labels) the layer **demotes itself to Shadow**.

## Why there is no exploration

The reward of *every* action becomes known once a label arrives (the cost
model is explicit), so learning is full-information: no sampling of risky
actions on live hosts, no importance weights, correct off-policy.

The update is the entropy-regularised **natural** policy gradient for a
softmax head, `∂L/∂z_j = −scale·(r_j − r̄) + β·(ln π_j + H)`. The textbook
gradient carries an extra `π_j` factor and plateaus: in testing it collapsed
onto `BanStandard` and never found `BanExtended` for honeytoken touches.

## Guarded learning

Live updates are trained from a replay buffer, then evaluated on a fixed
held-out anchor set of sandbox cases. If mean anchor loss worsens by more than
`guard_tolerance`, parameters are restored exactly (same idea as
`neuro_lane::apply_feedback`). Tested with deliberately poisoned labels.

## Results (all on synthetic data — read the caveats)

`cargo run --release -p rl_decision --example sandbox_report`, 4000 fresh
sandbox cases: mean loss **RL 1.27 vs legacy severity tiers 3.69 vs oracle 0.81**.
The oracle knows the label; the gap to it is the irreducible ambiguity between
attacks and their benign look-alikes. `unified_runner` scenario 6 shows the same
loop through the real correlation engine and control plane (regret 0.17 vs 0.33
over 92 labelled outcomes).

## Caveats

* The cost table and sandbox dynamics are **hand-set**, not fitted to real
  incidents. The layer optimises *that* cost model; a different cost model
  gives different "optimal" actions.
* Pre-training data is synthetic and its benign/attack classes are far more
  separable than real traffic (the demo's risk-model Brier of 0.002 is a sign of
  that, not of real accuracy). Calibrate on your own labels in `Shadow` first.
* Labels exist only for verdicts that reach a human or a honeytoken. Anything
  nobody reviews is never labelled, so there is selection bias toward what
  analysts look at.
* Regret/promotion evidence is only as good as those labels and the cost model.
* `Advisory`/`Active` have not been exercised against a live network.
* Training runs inline in the feedback callback (every `train_every` labels,
  well under a millisecond-scale for this model size); move it to a worker if labels ever arrive at volume.

## Using it

```rust
let layer = Arc::new(RlDecisionLayer::new(RlConfig::default())); // Shadow
layer.pretrain_in_sandbox(3000, 17);        // optional; or layer.load(path)
plane.set_decision_layer(layer.clone());    // also subscribes it to labels
// ... later, after reading layer.stats() / layer.recommendation():
layer.set_mode(Mode::Advisory);
layer.save(Path::new("rl_model.json"))?;    // parameters only; authority is never restored from disk
```
