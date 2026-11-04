# Unified NDR / IPS / SIEM — implementation of `unified_ndr_ips_siem_guide.md`

This workspace is a straight implementation of the guide's build order
(§3-§6), assembled from the uploaded components plus five new crates, one
library-surface addition to an existing crate, and (new) a sixth crate,
`unified_runner`, that wires the whole detection→containment path together
and replays a synthetic attack against it. A real `ontology_engine` and a
real standalone NSM (`nsm`) were integrated afterward — see "Integrating
`nsm` and the real `ontology_engine`" below. Nothing here is a mockup or a
`todo!()` — every scoring function calls real math against real dependency
APIs.

**Status: builds, tests, and runs end-to-end.** `cargo build --workspace`
and `cargo test --workspace` are both clean (440+ tests passing), and
`unified_runner` demonstrates a live run of ingestion → `hg_builder` →
`correlation_engine` → `ips_control_plane` → `rustwall::quarantine::Quarantine`
against synthetic DDoS/port-scan/honeytoken traffic, with a real ban landing
in the quarantine table. See **[`LIVE_RUN.md`](./LIVE_RUN.md)** for exact,
verified commands (including the toolchain workaround this sandbox needed —
its default `rustc` was too old for large parts of today's crates.io) and
for what the process of actually compiling this surfaced: two real bugs
(one borrow-checker error, one glob-import type collision) and two
incorrect test assertions, all fixed and documented in place. The original
"no toolchain, reviewed by hand" caveat below is kept for the historical
record of how this was first written, not because it's still true.

## What's new

| Crate | Guide section | What it does |
|---|---|---|
| `ingestion` | (existing, promoted) | Your uploaded `ingestion.rs`, unchanged, as its own crate so `hg_builder` can depend on it as a fourth lane subscriber. |
| `correlation_engine` | §2, §4 | Your uploaded `correlation.rs`, extended with `LaneSource::Hypergraph` / `Honeytoken`, a `LaneDetail::Hypergraph` carrying the Fiedler shift / NBSC `rho_B` / Ramanujan deviation / mean HMM belief as first-class fields, and `CorrelationVerdict::nearest_known_incident` from `siem_event_index`. |
| `hg_builder` | §3, §4, §5 | The actual hyperedge builder: `WindowedHyperedgeBuilder` (destination fan-in, source fan-out, session-cluster, honeytoken co-touch — each with fast *and* slow windows), `belief::vertex_prior_from_belief` (HMM belief → vertex weight), and `scoring::score_window` (matrix-free Fiedler eigenvalue via `spectral_hypergraph`, NBSC `rho_B` via `nbsc::hypergraph_bridge` for fan-out windows, Ramanujan deviation via `spectral_jacobi_ds::compute_hypergraph_ramanujan_feature`, all against a rolling per-family baseline). `pipeline.rs` wires it end-to-end from `ingestion`'s broadcast lane to `correlation_engine::HypergraphSender`. |
| `siem_event_index` | §4 | Persistent (JSONL-backed) nearest-neighbor index of confirmed incidents' fused 4-D signal vectors — deliberately *not* a reimplementation of `spectral_jacobi_ds::HnswIndex` (different vector space, different corpus scale; see the crate's module docs for why exact linear-scan NN is the right call here and what the upgrade path is). |
| `ips_control_plane` | §6 | The closed feedback loop: `CorrelationVerdict` → `review_queue::FlowPrediction` → (only once approved, human or auto-resolved) → `rustwall::quarantine::Quarantine::ban`. See the crate's module docs for why this also *is* the guide's "observe-only rollout" without a separate feature flag — it falls out of confidence calibration. |
| `bounded_channel` | — | Your uploaded `defensive_channel.rs`, promoted to its own crate (same design `rustwall::chan` already uses internally). |
| `ontology_engine` | — | Originally a minimal in-memory shim standing in for a dependency `review_queue` referenced but the uploads didn't include. **Replaced with the real `ontology_engine` implementation** once it was provided — a drop-in swap, zero code changes anywhere else in the workspace, because the real crate's public API matches the shim's exactly (verified function-by-function against every call site in `review_queue/src/{engine_bridge,schema,containment}.rs` before swapping; see `README.md`'s git history / this line for the record that a shim was ever here). |
| `rustwall/src/lib.rs` | §6 | One addition: `rustwall` was bin-only before this (`main.rs` with private `mod` declarations). Added a `lib.rs` exposing the same modules publicly so `ips_control_plane` can call `Quarantine::ban` directly, without touching `main.rs`'s own module tree. |
| `unified_runner` | — (new) | Not part of the original guide: a demo binary that actually instantiates the whole pipeline in-process and replays a synthetic attack through it end-to-end, including spawning `nsm --simulate` as a real child process. See `LIVE_RUN.md`. |
| `rl_decision` | RL layer (new) | Sandbox simulator, calibrated risk model, policy network, shielded action selection, reward ledger and guarded continuous learning between the correlation engine and the control plane. Shadow by default; cannot bypass the review gate. See **[`RL_DECISION.md`](./RL_DECISION.md)**. |
| `nsm` | — (integrated later) | A real, independent Network Security Monitor: 5-tuple flow tracking + five parallel detectors (port scan, SYN flood, DNS tunneling, beaconing, signature match), optional in-kernel XDP/eBPF fast path, NDJSON alerts. Added `nsm/src/lib.rs` (same "expose `main.rs`'s modules publicly" pattern as `rustwall`) and `correlation_engine::NsmSender`, a new fusion lane mapping `nsm`'s own 5-level severity onto this pipeline's 4-level scale. Runs and is bridged in as a genuinely separate process (its own capture loop, its own detector state) via `unified_runner`'s NDJSON-over-stdout reader — not reimplemented in-process. See "Integrating `nsm`" below and `LIVE_RUN.md`. |

## What the guide's open questions turned into concrete decisions

- **Window sizing** (§8): every fan-in/fan-out tracker in `hg_builder`
  runs a fast (burst, 10s default) *and* slow (campaign, 5min default)
  window in parallel, closing on whichever hits its threshold first —
  not a single compromise window size.
- **`rand` version coexistence** (guide's stated concern about pulling NBSC
  into `active_spectral_defense` directly): checked directly rather than
  assumed. Neither `nbsc::hypergraph_bridge::clique_expand` nor
  `spectral_jacobi_ds::compute_hypergraph_ramanujan_feature`'s public
  signatures expose a `rand` type across the crate boundary (both just take
  a `u64` seed), so `hg_builder` never has to reconcile `nbsc`'s `rand 0.8`
  with `spectral_jacobi_ds`'s `rand 0.9` — noted explicitly in
  `hg_builder/Cargo.toml`.
- **Session-cluster's vertex definition**: `ingestion::IngestedPacket`
  carries a 5-tuple, not an application-layer session id, so
  `WindowedHyperedgeBuilder`'s session-cluster family degrades to grouping
  by `(src_ip, dst_ip)` with flow-instance (`ip:port`) vertices — documented
  as a known simplification in `builder.rs`, not silently papered over.
- **HMM belief → vertex weight**: `Belief` only exposes `argmax()`, not a
  continuous score, so `hg_builder::belief::vertex_prior_from_belief` adds
  the few lines of glue the guide anticipated ("small, since a
  `belief_score()`-equivalent likely already exists") — turning "most
  likely state" into "total non-benign probability mass," which actually
  distinguishes confidence levels `argmax()` collapses.

## Building

```
cargo build --workspace
cargo test --workspace
```

**This has been compiled, tested, and run — see `LIVE_RUN.md` for the
verified commands.** The sandbox this was implemented in initially had no
`rustc`/`cargo` at all, then only `rustc` 1.75 (Dec 2023) via `apt`, which
turned out to be too old for large parts of current crates.io (most crates
now require edition2024 / rustc 1.85+; a few, like `idna`, use APIs that
weren't `stable` until 1.81). `LIVE_RUN.md` documents how a working
`rustc`/`cargo` 1.93.1 was obtained from Ubuntu's `resolute` archive without
disturbing the base system. Once a real toolchain was available,
`cargo build --workspace` surfaced two genuine bugs the "reviewed by hand"
pass had missed:

- `hg_builder/src/builder.rs`: a double-mutable-borrow (`E0499`) in the
  fan-in/fan-out window-closing loops (iterating `&mut self.fan_in_fast`
  while also needing `&mut self` for `self.close_fan_in(...)`). Fixed by
  splitting into a "find what should close" pass over the borrowed maps
  and a second pass that closes them.
- `ips_control_plane/src/lib.rs`: `review_queue::prelude::*`'s own
  `Result<T>` type alias shadowed `std::result::Result` for one trait
  method's signature, so `ContainmentAction::execute`'s two-generic-param
  `Result<(), Box<dyn Error>>` silently resolved to the wrong one-param
  alias. Fixed with an explicit `std::result::Result` qualification.

`cargo test --workspace` then surfaced two incorrect test assertions (not
implementation bugs) — both fixed with corrected assertions and an explicit
doc comment on the real, verified behavior rather than a silent change:

- A `hg_builder` test assumed the HMM belief prior perturbs the Fiedler
  eigenvalue. It doesn't: `spectral_hypergraph`'s own (pre-existing,
  unmodified) Laplacian defines vertex *degree* from incident hyperedge
  weights only, and `add_hyperedge` always uses a uniform incidence weight
  of `1.0` — so a vertex's intrinsic `weight` never reaches `D_v`. The
  belief prior is still real, useful signal (it reaches
  `ScoredWindow::mean_hmm_belief`, a first-class fused feature) — it just
  doesn't currently reweight the spectral pass, and the crate's docs now
  say so explicitly.
- An `ips_control_plane` test submitted one moderate-severity hypergraph
  event and expected it to reach the review queue. `correlation_engine`
  deliberately holds back a single-lane, sub-Critical signal
  (`CorrelationEngine::maybe_emit` only fires on multi-lane corroboration
  or single-lane-Critical) — correct, intentional gating that the test's
  scenario didn't exercise correctly. Fixed by submitting two corroborating
  lanes on the same host, matching the design the gating exists for.

## Pre-existing crates: unmodified

`active-siem/*`, `honeypot_engine`, `krylov_ds`, `nbsc`, `neural_hmm`,
`review_queue`, `rust-clam/*`, `spectral_hypergraph`, `spectral_jacobi_ds`
are byte-for-byte your uploads, just flattened into one Cargo workspace
(`active-siem` and `rust-clam` were each their own nested `[workspace]`;
their virtual-manifest `Cargo.toml`s were removed and their member crates
listed directly in the top-level `Cargo.toml` instead, since Cargo doesn't
support a workspace-of-workspaces — `rust-clam`'s `[workspace.package]`
version/edition/license values were carried forward into the top-level
`[workspace.package]` table so its crates' `version.workspace = true`
fields keep resolving).

## Integrating `nsm` and the real `ontology_engine`

Two more pieces arrived after the rest of this workspace was already
building and testing clean: the real `ontology_engine` (replacing the
shim described above) and `nsm`, a standalone Network Security Monitor.

**`ontology_engine`** was a pure swap — no source changes anywhere else
in the workspace. Before deleting the shim, every call site into it
(`review_queue/src/{engine_bridge,schema,containment}.rs`) was checked
function-by-function against the real crate's public API:
`register_object_type`, `register_link_type`, `create_object_instance`,
`create_link`, `traverse`, `list_instances_by_type`,
`ObjectTypeBuilder`/`ObjectInstance`/`LinkInstance`/`LinkType`'s
constructors, `Direction::{Outgoing,Incoming}`, and the
`OntologyError::{ObjectTypeAlreadyRegistered,LinkTypeAlreadyRegistered}`
variants `schema::register`'s idempotency check matches on — every one
matched exactly. `cargo test -p review_queue -p ips_control_plane -p
ontology_engine` was green immediately, with the real (production-grade,
per its own `Cargo.toml` description) implementation in place instead of
the shim's minimal one.

**`nsm`** (5-tuple flow tracking + five parallel detectors: port scan,
SYN flood, DNS tunneling, beaconing, signature match; optional XDP/eBPF
fast path) was a real integration, not a swap:

- `nsm` was originally its own two-crate Cargo workspace (itself +
  `xdp/nsm-common`, shared with the standalone, nightly-only
  `xdp/nsm-ebpf`). Cargo doesn't allow a workspace member to also be a
  workspace root, so `nsm`'s inner `[workspace]` table was removed and
  `xdp/nsm-common` was added directly to the top-level workspace's
  `members` instead — `xdp/nsm-ebpf` stays excluded exactly as before,
  via its own (untouched) empty `[workspace]` table.
- Added `nsm/src/lib.rs`, same "expose `main.rs`'s modules publicly"
  pattern already used for `rustwall` and `honeypot_engine`, so other
  crates can depend on `nsm::alert::{Alert, Severity}` and `nsm::detect`.
- Added `Deserialize` to `nsm::alert::Severity` — but deliberately *not*
  to `Alert` itself: `Alert::detector` is `&'static str` (every one of
  `nsm`'s detectors passes a literal), and deriving `Deserialize` on a
  struct with a `&'static str` field forces the whole deserialization's
  input lifetime to be `'static`, which a line read from a child
  process's stdout never is. See `nsm/src/alert.rs`'s doc comment on
  `Alert::detector` for the full reasoning, and
  `unified_runner::NsmAlertWire` for the small owned-`String` DTO that
  actually parses `nsm`'s NDJSON output.
- Added `correlation_engine::NsmSender`: a new `LaneSource::Nsm` /
  `LaneDetail::Nsm` fusion lane mapping `nsm`'s own 5-level severity
  (`Info..Critical`) onto this pipeline's 4-level scale, with confidence
  values chosen so recomputing severity from the reported confidence via
  `SpectralSender`/`HypergraphSender`'s own bucket cutoffs (0.90/0.70/0.40)
  reproduces the same severity — see `NsmSender::map_severity`'s doc
  comment. Also fixed `ips_control_plane::classify()`, which would have
  silently mislabeled every nsm-sourced verdict as the generic
  `"MultiLaneCorrelation"` fallback; it now returns e.g. `NsmPortscan` /
  `NsmSignature` with `nsm`'s own alert message as the rationale.
- **`nsm` runs as a genuinely separate process**, not reimplemented
  in-process: `unified_runner` spawns `nsm --simulate` as a real child
  process (resolved via `current_exe()`'s sibling directory, so it works
  regardless of the caller's working directory), reads its actual NDJSON
  stdout, and bridges each parsed alert into the same live correlation
  engine everything else in this workspace shares. See `LIVE_RUN.md` for
  a verified run showing one of `nsm`'s own detector alerts leading to a
  real ban in `rustwall`'s quarantine table.




## Advanced build

See `ADVANCED.md`: connectome-LSTM temporal lane (`neuro_lane`), burn-llm bi-directional bridge (`llm_bridge` + `llm_sidecar/`), analyst-feedback learning, and what is/is not validated.


## RL decision layer

See `RL_DECISION.md` for the full architecture, authority ladder, results and
caveats, and `CHANGES.md` for everything modified in this revision.
