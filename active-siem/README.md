# ActiveSIEM

A minimal, working Rust SIEM skeleton: rule/threshold correlation (Wazuh/OSSEC-style)
fused with a burn.rs autoencoder for the class of attacks signature engines
structurally can't see - **low-and-slow infiltration**. Guarded active response
closes the loop instead of just alerting.

Builds and passes its full test suite on stock **Rust 1.75** (see "MSRV pins" below).

```
cargo test --workspace
cargo run -p siem-server        # runs the end-to-end demo
```

## What Wazuh, OSSEC, and Security Onion actually do

| | OSSEC | Wazuh | Security Onion |
|---|---|---|---|
| Core model | Agent tails logs -> decoder extracts fields -> rule tree matches, with `frequency`/`timeframe` threshold rules | OSSEC's architecture, forked and extended: same decoder/rule engine, plus FIM, vulnerability detection, and a REST API/indexer (OpenSearch) on top | Not agent-based - a sensor stack (Zeek + Suricata + Wazuh optionally) feeding Elastic/OpenSearch, oriented around Sigma rules and full-packet capture (Stenographer) for retrospective hunting |
| Detects well | Auth failures, known bad log patterns, file integrity changes, rootkit signatures | Everything OSSEC does, plus CVE-matched vulnerable software, compliance (PCI/HIPAA) mappings | Network-level reconnaissance, known-exploit signatures (Suricata/ET rules), protocol anomalies Zeek flags structurally |
| Structurally misses | Anything that doesn't cross a static threshold or match a signature - by design, these are rule engines | Same limitation - Wazuh is OSSEC's architecture, not a different detection paradigm | Passive/detection-only. No active response. Also signature/threshold-first - Sigma rules are declarative pattern matching, same category as OSSEC's rule tree |
| Active response | Yes - `active-response` scripts (e.g. `firewall-drop.sh`) fire on matching alerts | Yes - same mechanism, plus orchestration via the manager | No - explicitly out of scope; it's a monitoring/hunting platform |

The common gap across all three: they are **signature and threshold engines**.
That's exactly why "Infiltration" is the hardest class in datasets like
CIC-IDS2018 - a slow foothold making small, evenly-paced connections is
designed to stay under every static threshold and match no known signature.
Catching it needs a *learned model of what normal traffic looks like*, which
none of the three provide out of the box (Security Onion can bolt on ML via
its stack, but nothing ships by default).

## What this project does about it

Two detection layers, doing different jobs, feeding one alert stream:

1. **`siem-rules`** - a Sigma/OSSEC-lite correlation engine (YAML rules,
   sliding-window thresholds). Catches the *noisy* precursors to infiltration:
   brute force, vertical port scans. This is what Wazuh/OSSEC/Security Onion
   already do well, reimplemented small enough to read in one sitting.

2. **`siem-ml`** - a burn.rs autoencoder trained only on benign flow features
   (duration, byte ratios, packet counts). Reconstruction error above a
   calibrated threshold flags the flow as anomalous. This is the layer that
   catches what the other three miss: a connection that matches no signature
   and never crosses a count threshold, but doesn't *look* like the rest of
   the traffic on the network.

3. **`siem-response`** - unlike Security Onion (no active response at all),
   this closes the loop, but with an explicit fail-to-escalate policy
   (severity floor, allowlist, rate limit, auto-expiring blocks) rather than
   Wazuh/OSSEC's fire-and-forget script model - a bad automated block is a
   self-inflicted outage, so the default posture is "escalate to a human"
   unless every guard passes.

```
crates/
  siem-core       event/alert data model
  siem-collector  log tailing + decoders (agent side)
  siem-rules      YAML threshold/correlation rules
  siem-ml         burn.rs autoencoder for flow anomaly scoring
  siem-response   guarded active response
  siem-server     wires it all together (see main.rs for the demo)
```

## Demo output

```
-- Simulating SSH brute force from 10.0.0.5 --
  ALERT [ssh-bruteforce] Repeated SSH authentication failures from one source (severity High, mitre Some("T1110"))

-- Training autoencoder on benign traffic shape --
  calibrated anomaly threshold: 0.00171

-- Scoring a suspected infiltration beacon (long, tiny, periodic, upload-heavy) --
  reconstruction error: 10.17549 (threshold 0.00171)
  flagged as anomalous/infiltration-like: true

-- Active response policy --
[response:dry-run] would block 198.51.100.77 for alert 'Anomalous long-duration low-volume flow (possible infiltration beacon)' (severity High)
  response executed.
```

The brute force is 5 identical log lines - a plain threshold rule gets it,
same as OSSEC would. The beacon is a single, ordinary-looking TCP flow; no
rule fires on it, but its reconstruction error is ~6000x the calibrated
benign threshold, so the ML layer catches what the rule layer structurally can't.

## MSRV pins (Rust 1.75)

burn 0.13's dependency tree assumes a newer toolchain by default. Every pin
below is `cargo update -p <pkg>@<from> --precise <to>`, kept at the newest
version that still satisfies both the workspace's semver requirements and
rustc 1.75:

- `indexmap` -> 2.2.6 (2.14 requires edition2024)
- `rmp-serde` -> 1.1.2, `rmp` -> 0.8.14 (newer needs edition2024)
- `uuid` -> 1.10.0 (1.23 needs rustc 1.85)
- `rayon` -> 1.10.0, `rayon-core` -> 1.12.1 (newer needs rustc 1.80)
- `burn`/`burn-core` -> 0.13.0 (0.13.1+ pulls a `bincode` range needing rustc 1.85)
- `half` -> 2.4.1 (2.5+ needs rustc 1.81)
- `bincode` -> 2.0.0-rc.3 (burn-core's own required range; final 2.0.0 needs rustc 1.85)

If you have a newer toolchain available, `cargo update` without `--precise`
pins will likely just work and pull current versions instead.

## Validated against real data - including a critical negative result

`crates/siem-ml/tests/infiltration_dataset.rs` runs three tests against real
CIC-IDS2018-style data, not just the synthetic beacon in the demo binary:

1. **50 real Infiltration flows** (RDP/SMB/HTTP low-and-slow traffic:
   `13.58.225.34` probing `172.31.69.13/24/25`, then `172.31.69.13` pivoting
   laterally over SMB/RPC to `.24/.25/.28`) - **100% flagged**, mean
   reconstruction error ~85x the benign-calibrated threshold.

2. **6 real, diverse Benign flows** (an RDP admin session, two DNS lookups,
   a 2-minute HTTPS download, a 1-minute HTTPS download, an ordinary internal
   HTTP transfer) - **also 100% flagged.**

That second result is the important one, and it overturns the first. A model
trained on 200 near-duplicate copies of one synthetic traffic shape
(seconds-scale, download-heavy, ~20-30 packets) isn't discriminating
"infiltration" from "benign" - it's discriminating "matches this one narrow
pattern" from "everything else." Real infiltration traffic and real ordinary
traffic (DNS, RDP, long downloads) both fall outside that narrow envelope, so
both get flagged identically. A detector with a 100% false-positive rate
"detects" 100% of attacks trivially and is useless in production - it would
page someone on every DNS query.

Run both yourself:

```
cargo test -p siem-ml --test infiltration_dataset -- --nocapture
```

**Conclusion:** the architecture (rules for noisy attacks, ML for quiet ones,
guarded response) is sound, but the current model has not demonstrated it
discriminates infiltration from benign traffic - only that it discriminates
"synthetic training shape" from everything else. Before this is trustworthy
against real traffic, the training/calibration set needs to be a large,
genuinely diverse real benign corpus (not synthetic, not narrow), and the
false-positive rate needs to be re-measured against that.

## Categorical attack-vector isolation (`siem-ml::classifier`)

Added after analyzing `spec_engine` (a companion crate in this workspace's
ecosystem): it classifies CIC-IDS2018 labels into MITRE ATT&CK tactics via
keyword matching (`classify_mitre_from_label`), which pointed at the right
fix for this project's autoencoder-only design - a single reconstruction-
error threshold can only ever answer "does this look like the one benign
pattern I trained on", which is both why it flagged 100% of real Infiltration
flows *and* 100% of real ordinary DNS/RDP/HTTPS traffic (see above). A
categorical system needs to ask a different question: "which of several
known shapes does this match, if any" - answered here with a small supervised
softmax classifier (`siem_ml::classifier::Classifier`) over six categories
(`Benign`, `Reconnaissance`, `BruteForce`, `Infiltration`,
`CommandAndControl`, `DenialOfService`), condensed from `spec_engine`'s MITRE
vocabulary down to shapes actually distinguishable at this crate's 8-feature
resolution.

**This is a different, complementary tool to the autoencoder, not a
replacement.** Supervised classification needs labeled examples of every
category it claims to distinguish - including attack examples - which is
exactly why it was *not* used to fix the autoencoder itself: retraining the
autoencoder on attack data would flip what "anomalous" means (it would learn
to reconstruct attacks well and flag benign traffic instead - this was
explicitly requested and declined earlier in this project's history for that
reason). Use both: the classifier for "which known category, if any" with a
confidence score callers can threshold on, and the autoencoder as an
open-set fallback for traffic that doesn't match any known category.

`crates/siem-ml/tests/categorical_isolation.rs` trains and evaluates it:

```
cargo test -p siem-ml --test categorical_isolation -- --nocapture
```

**Result: 28/29 (96.6%) test accuracy**, with a full confusion matrix printed
per run. The one error is informative, not concerning: a Benign flow
misclassified as Infiltration - consistent with the false-positive finding
above, since Benign and Infiltration are the two categories backed by real,
independently-varied data rather than jittered synthetic seeds.

**Data provenance matters here - read before trusting the number:**
- `Benign` (6 rows) and `Infiltration` (25-row subset of 50) are the real
  CIC-IDS2018-style data used throughout this project.
- `BruteForce`, `CommandAndControl`, and `DenialOfService` are seeded from
  `spec_engine`'s own synthetic dataset (1-3 real seed rows per category
  there), each jittered (+-15% on duration/packets/bytes) into a larger set
  purely to get a usable train/test split. This tests whether the classifier
  can separate these *shapes*, not whether it generalizes across
  independently-captured examples of each attack - a materially weaker claim
  than the Benign/Infiltration result.
- `Reconnaissance` is in the taxonomy (mirroring `spec_engine`'s vocab) but
  has no seed row in any dataset provided so far and is deliberately excluded
  from training/evaluation rather than backed by invented data.

## Honest gaps (not yet done)

- No real packet-capture source wired in - `siem-collector::synthetic_flow`
  stands in for what would be a `net_sys`/Zeek `conn.log` feed.
- **The autoencoder's training/calibration data is the core unresolved
  problem**, not a minor detail: it's a narrow synthetic set, and the false-
  positive test above shows it doesn't generalize to real diverse benign
  traffic. This needs a large real benign corpus and a proper percentile-
  based threshold (99th/99.9th) before it's meaningful, and even then the
  6-sample false-positive test here is too small to certify a rate - it's a
  smoke test, not a benchmark.
- The 50-row Infiltration sample is one campaign against one internal subnet;
  it doesn't establish how the model performs against different infiltration
  TTPs (DNS tunneling, ICMP exfil, encrypted C2 over unusual ports).
- The categorical classifier's `BruteForce`/`CommandAndControl`/
  `DenialOfService` categories rest on jittered variations of 1-6 real seed
  rows each (see the categorical section above) - real generalization
  testing needs independently-captured examples per category, the same gap
  the false-positive test exposed for the autoencoder. `Reconnaissance` has
  no backing data at all yet and isn't trained or evaluated.
- No confidence-threshold/fallback wiring yet between the classifier and the
  autoencoder (low-confidence classifier predictions should fall back to the
  open-set anomaly score, per the design note above) - both exist as
  separate tools; the routing logic between them isn't implemented.
- `siem-response::LogOnly` is a dry-run stub - a real executor (nftables/
  iptables rule, EDR API call) needs review before being wired in, per the
  fail-to-escalate policy this crate implements.
- No persistence/storage layer, no multi-agent transport (gRPC/mTLS), no UI -
  this is the detection+response core, not a deployable product.
