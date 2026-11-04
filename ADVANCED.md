# Advanced build: connectome-LSTM lane + burn-llm bridge

What was added on top of the original workspace, how to run it, and —
just as important — what is and is **not** demonstrated.

## Architecture

```
                         ┌────────────────────────────────────────────┐
 packets ─► ingestion ─► │ hg_builder lanes (count/time thresholds)    │──┐
        │                └────────────────────────────────────────────┘  │
        │                ┌────────────────────────────────────────────┐  │  correlation_engine
        └──────────────► │ neuro_lane: per-host bins → connectome-LSTM │──┼─► (≥2 lanes, or Critical)
                         │ score ∈[0,1] (capped High / conf ≤0.85)     │  │        │
                         └──────▲───────────────┬─────────────────────┘  │        ▼
                  raise-only    │               │ CognitiveReport        │   review_queue ─► rustwall
                  ScrutinyBoard │               ▼ (uplink)               │        │
                         ┌──────┴───────────────────────┐                │        │ human / honeytoken
                         │ llm_bridge (closed grammar)  │◄── SituationReport      │ verdicts
                         │  sanitise → backend → parse  │    (downlink) ──┘        │
                         └──────┬───────────────────────┘                          ▼
                                │ stdio                          FeedbackObserver → labels →
                         ┌──────▼───────────┐                     guarded fine-tune (anchor-checked)
                         │ llm_sidecar      │  (burn-llm, burn 0.21, own workspace/process)
                         └──────────────────┘
```

| Crate | Role |
|---|---|
| `human_connectome_rnn` (from `lstm_rnn_human_connectome_rnn_v0_4`) | Connectome-masked LSTM/RNN; now repointed at the workspace's `krylov_ds` / `spectral_hypergraph` / `nbsc`. |
| `neuro_lane` (new) | Feature binning, synthetic behaviours, detector + trainer, runtime lane, guarded learning from verdicts. |
| `llm_bridge` (new) | The bi-directional language channel (protocol, sanitiser, strict parser, `ScrutinyBoard`, sidecar client). |
| `llm_sidecar/` (new workspace; contains `burn-llm` from `redefining_llm`) | Real LLM inference in a separate process. |
| `correlation_engine`, `ips_control_plane`, `unified_runner` (modified) | `Connectome` lane, `FeedbackObserver` hook, scenario 5. |

### Why the LLM is a sidecar process
`burn-llm` needs Burn 0.21; several existing members (`neural_hmm`,
`siem-ml`, `siem-server`, `nbsc`'s default feature) need Burn 0.13.2, and the
two cannot share one lockfile (`bincode` 2.0.0‑rc.3 vs 2.0.1). Rather than
rewrite those crates, the LLM runs as its own binary (`llm_sidecar/`), spoken
to over a small stdio protocol. Side benefit: a model crash/hang cannot take
down the sensors.

## The bi-directional channel

* **Downlink (stack → connectome side):** a verdict/alert becomes a
  `SituationReport`; the model may answer `HINT: scan|flood|beacon|exfil|benign|unknown`.
* **Uplink (connectome → stack):** a score, trend and the input features that
  drove it (occlusion attribution) become a `CognitiveReport`; the model may
  answer `ADVISE: elevate|none` and `NOTE: …`.
* **Learning path:** human decisions (and honeytoken-backed auto-resolutions)
  flow back as labels; `NeuroLane::apply_feedback` fine-tunes only if a fixed
  anchor set does not degrade (otherwise parameters are restored exactly).

### Safety model (the model's output is untrusted)
Prompts contain attacker-influenced text (DNS names, alert strings). So:
1. untrusted fields are reduced to an inert allowlist (no `:` `=` quotes or
   newlines) → cannot forge `HINT:`/`ADVISE:` lines;
2. output is parsed by a closed grammar; anything else is ignored;
3. the **only** effect of parsed output is `ScrutinyBoard::elevate`:
   raise-only, TTL- and capacity-bounded. It lowers the connectome lane's
   alert threshold for a host; it cannot suppress, ban, or unban;
4. deterministic rules decide everything the model could influence: it can
   only fill an `Unknown` hint, or add an elevation when the numeric score is
   already ≥ 0.2; it can never veto a rule-based elevation;
5. backend error/garbage → rules alone. With `NullBackend` the whole system
   works; the LLM adds analyst-facing notes and a second opinion, not safety.

A spoofed-source attacker could still cause *extra scrutiny* (more
sensitivity, bounded by board capacity/TTL) for hosts they name — that is the
residual risk, and it only increases review load.

The connectome lane itself is capped at `Severity::High` / confidence 0.85,
below the review queue's 0.90 auto-resolve floor: **on its own it can never
cause an automatic ban.** It corroborates other lanes or queues for a human.

## Running

```sh
# everything except the real-LLM test
cargo test -p llm_bridge -p neuro_lane -p correlation_engine -p ips_control_plane
cargo run  -p unified_runner            # scenarios 1-4 as before, then scenario 5

# real burn-llm inference through the sidecar (builds the separate workspace)
scripts/test_sidecar.sh
LLM_MODEL_DIR=/path/to/checkpoint LLM_SIDECAR_BIN=$PWD/llm_sidecar/target/release/llm_sidecar \
  cargo run -p unified_runner

# retrain the detector (≈10 s) and print held-out metrics
cargo run --release -p neuro_lane --bin neuro_train
cargo run --release -p neuro_lane --example persistence_curve
```
Rust ≥ 1.92 is required (burn-llm); see `LIVE_RUN.md` for the toolchain steps.

## What is and is not shown (read this)

**Shown, with tests:**
* The detector separates *low-and-slow* probing (2 packets/bin, no bin near
  any count threshold) from benign traffic and short health-check bursts:
  100 % on the in-distribution held-out set, ≈ 99 % on a set with widened
  ranges; the per-window port-count rule catches 0 of the low-and-slow cases.
* Evidence accumulates with persistence: a 4-bin probe run does not alert;
  a longer one scores higher (`persistence_curve`: mean score ≈ 0.21 up to
  8 bins, rising slowly after).
* Injection text cannot create directives; the model cannot suppress or
  originate an elevation without numeric evidence; the guard reverts
  harmful feedback bit-for-bit.
* The real `burn-llm` model runs end to end through the sidecar.

**Not shown / caveats:**
* **All training and evaluation data is synthetic**, defined in
  `neuro_lane/src/synth.rs`. The perfect in-distribution score reflects an
  easy, self-defined task, not real-world accuracy. No real-traffic,
  false-positive-rate-at-scale, or adversarial-evasion evaluation exists.
  Expect to retrain/validate on your own captures before trusting scores.
* The connectome is `Connectome::synthetic_human` (a stand-in), not measured
  tractography. Whether connectome wiring helps versus an unconstrained LSTM
  was **not** tested here.
* The bundled LLM fixtures have random weights, so model *content* is
  meaningless in tests/demo (20/20 outputs unparseable → rules decided). A
  real instruction-tuned checkpoint is needed for the model to add value,
  and its value (vs. rules alone) is unmeasured.
* The lane's score is not part of `siem_event_index`'s 4-D signal vector, and
  belief from this lane does not feed the spectral math (same limitation the
  original SUMMARY lists for other lanes).
* Timing-regularity is a strong feature: perfectly periodic traffic of any
  kind looks beacon-like (found while writing the tests).
* Minimum history is 10 bins; the first 6 outputs are untrained, by design.

## Other changes to existing code
* `correlation_engine`: `LaneSource::Connectome`, `LaneDetail::Connectome`,
  `ConnectomeSender` (severity ≤ High, confidence ≤ 0.85), `connectome_dropped`
  metric, `FeedbackObserver` trait.
* `ips_control_plane`: classifies the new lane; `add_feedback_observer`;
  human decisions and honeytoken-backed containment emit labelled feedback.
* `unified_runner`: scenario 5 (`src/advanced.rs`).
* Workspace: new members, `exclude = ["llm_sidecar"]`, opt-level overrides for
  the numeric crates in dev builds. `Cargo.lock` only gains the new path crates.
