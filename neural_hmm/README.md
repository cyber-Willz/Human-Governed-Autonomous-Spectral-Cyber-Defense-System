# neural_hmm

Production rewrite of the prototype Neural HMM filter. Builds and passes tests under
**Rust 1.75 (apt)**, CPU-only (`burn` + `NdArray` backend), no `rustup`/network access needed
beyond crates.io.

## What changed vs. the prototype

| Prototype | Production |
|---|---|
| `transition_matrix: Vec<Vec<f32>>`, unchecked | `TransitionMatrix`: validated square + row-stochastic + non-negative on construction, JSON load/save, labeled states |
| Plain `softmax` | `log_softmax` in the network, `.exp()` at the boundary — avoids overflow/underflow that hits plain softmax on extreme logits |
| `filter_step` returns `Vec<f32>` unconditionally, silently falls back to prior on collapse | Returns `HmmResult<Belief>`; `Belief` is a validated wrapper (finite, non-negative, sums to 1); collapse is a typed `HmmError::BeliefCollapse`, not a silent no-op |
| No dimension checking between emission engine and transition matrix | `NeuralHmm::new` rejects mismatched state counts at construction |
| 2-layer MLP, no regularization | 3-layer MLP + dropout |
| No checkpointing | `NeuralEmissionEngine::save`/`load` via Burn's `CompactRecorder` |
| No tests | 10 unit tests covering validation, the Bayes update math standalone, and dimension-mismatch rejection |
| `println!`-only | `#[instrument]` on `filter_step` (via `tracing`) so it plugs into an existing `active-siem`/`spec-engine` tracing setup |

## Layout

```
src/
  error.rs       HmmError (thiserror) — one variant per failure mode, not a generic anyhow blob
  transition.rs  TransitionMatrix: validated, serde, JSON persistence, predict step
  emission.rs    NeuralEmissionEngine: Burn MLP, log-softmax, checkpoint save/load
  hmm.rs         Belief + NeuralHmm: ties predict/update into one filter_step
  lib.rs         re-exports
  main.rs        end-to-end example (same 3-state Normal/Suspicious/Attack scenario as the prototype)
```

## MSRV pin trail (Rust 1.75, apt, no rustup)

`burn 0.13` pulls in several transitive crates whose *newer* published versions require a
newer rustc than 1.75, even though `burn` itself supports 1.75. `cargo build` fails fast with
`error: package X requires rustc 1.8Y`, naming the exact package — the fix each time is
`cargo update -p <pkg> --precise <last-1.75-compatible-version>`. This `Cargo.lock` already has
them applied; if you regenerate the lockfile from scratch, you'll hit them again in this order:

```
cargo update -p indexmap@2.14.0 --precise 2.2.6   # hashbrown 0.17 -> needs edition2024
cargo update -p rmp-serde --precise 1.1.2         # rmp 0.8.15 -> needs edition2024
cargo update -p rmp --precise 0.8.12
cargo update -p uuid --precise 1.10.0             # 1.24 needs rustc 1.85
cargo update -p rayon --precise 1.10.0            # pulls rayon-core 1.13, needs rustc 1.80
cargo update -p rayon-core --precise 1.12.1
cargo update -p bincode --precise 2.0.0-rc.3      # 2.0.1 needs rustc 1.85 (same issue as ai_firewall)
cargo update -p half --precise 2.4.1              # 2.7.1 needs rustc 1.81
```

`default-features = false, features = ["ndarray", "std"]` on the `burn` dependency keeps
`burn-wgpu`/`burn-candle`/GPU deps out of the actual **build** (only `burn-ndarray` compiles),
even though they still briefly appear during dependency *resolution* for the lockfile.

Consider folding this list into `autopin.py`/`autopin2.py` as a known-good pin set for any new
`burn`-based crate on this toolchain, rather than rediscovering it BFS-style each time.

## Usage

```bash
cargo build            # builds src/main.rs + lib
cargo test             # 10 tests, all pure-CPU, no GPU/network needed
cargo run              # runs the Normal/Suspicious/Attack example, one filter_step
```

```rust
let transition_matrix = TransitionMatrix::with_labels(rows, labels)?;
let emission_engine = NeuralEmissionEngine::<NdArray<f32>>::new(&device, &config);
let hmm = NeuralHmm::new(emission_engine, transition_matrix)?;

let mut belief = Belief::uniform(hmm.num_states())?;
belief = hmm.filter_step(&belief, features_tensor)?;
let (state, prob) = belief.argmax();
```

## Known gaps / next steps if you want to go further

- `filter_step` takes a single-row tensor; a true production streaming pipeline (matching your
  `gnn_spec_ingest::StreamingWindow` pattern) would want a `filter_batch` that scores many
  independent sequences per tick and returns `Vec<Belief>`.
- The transition matrix here is static/hand-specified. If you want it learned (Baum-Welch /
  EM), that's a separate offline training loop — happy to build that next if useful.
- No persistence for the *belief state itself* across process restarts — only the network
  weights are checkpointed. Worth adding if this runs as a long-lived daemon.
