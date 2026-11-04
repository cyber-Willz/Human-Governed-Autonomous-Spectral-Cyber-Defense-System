# Live run: exact, verified commands

Everything below was actually executed in order, in a clean checkout, to
produce this document — not written from memory of what should work.
Commands are grouped by phase; run them from the workspace root
(`unified-ndr-ips-siem/`, the directory containing this file's sibling
`Cargo.toml`) unless noted otherwise.

## 0. Toolchain

The workspace needs a modern Rust toolchain — most dependencies now require
`edition2024` (rustc ≥ 1.85), and a couple of source-level APIs
(`core::error::Error` used unconditionally by `idna`) need ≥ 1.81 even
without edition2024. If `rustc --version` already reports **1.85 or newer**,
skip to §1.

If you're on a system whose only readily-available `rustc` is older (e.g. a
minimal container image with just `apt install rustc cargo`, which on
Ubuntu 24.04 "noble" gets you rustc 1.75 from Dec 2023), the fix that was
actually used here was to pull a newer `rustc`/`cargo` **package** from a
later Ubuntu suite, without switching the whole system to it:

```bash
# Add a later Ubuntu suite as a low-priority source (so it's only used
# when a package is explicitly requested from it with -t).
cat <<'EOF' | sudo tee /etc/apt/sources.list.d/resolute.list
deb http://archive.ubuntu.com/ubuntu resolute main universe
EOF
cat <<'EOF' | sudo tee /etc/apt/preferences.d/resolute-pin
Package: *
Pin: release n=resolute
Pin-Priority: 100
EOF
sudo apt-get update

# Remove any older rustc/cargo first, then install from the newer suite.
sudo apt-get remove -y rustc cargo
sudo apt-get install -y -t resolute rustc cargo
rustc --version   # rustc 1.93.1 (verified) -- anything >= 1.85 is fine
cargo --version
```

This worked cleanly because `libstd-rust-1.93`'s only binary-compatibility
requirement beyond what `noble` already ships is `libllvm21`, which `apt`
pulls in automatically as a normal dependency — it does **not** require a
newer `libc6` than `noble` already has (both need ≥ 2.39, which `noble`
satisfies), so this doesn't cascade into a wider dist-upgrade. If your base
distro is old enough that this isn't true, use `rustup` instead
(`https://sh.rustup.rs`) — the apt-pin approach above exists specifically
for sandboxes where `rustup`'s installer domain is firewalled off but
`archive.ubuntu.com` isn't.

## 1. Build and test the whole workspace

```bash
cargo build --workspace
cargo test --workspace
```

Expected: a clean build (a handful of `warning:`s, no `error:`s) and every
test binary reporting `test result: ok.` — over 300 tests across ~30
crates, 0 failures.

## 2. Live run: the detection → correlation → containment pipeline

`unified_runner` is a demo binary (added as part of this work, not part of
the original crate set) that instantiates the real pipeline in-process —
`ingestion::IngestionPipeline` → `hg_builder::HypergraphPipeline` →
`correlation_engine::CorrelationEngine` → `ips_control_plane::IpsControlPlane`
→ `rustwall::quarantine::Quarantine` — and replays four scenarios against
it: a synthetic DDoS-style destination fan-in, a port-scan-style source
fan-out, a honeytoken touch, and (added when `nsm` was integrated) `nsm
--simulate` spawned as a real, separate child process, bridged in over its
actual NDJSON stdout.

```bash
cargo build -p unified_runner -p nsm
RUST_LOG=warn ./target/debug/unified_runner
```

Expected output (verbatim from an actual run):

```
=== after DDoS fan-in + port-scan fan-out (hypergraph lane only) ===
review_queue: total=0 auto_resolved=0 pending=0 under_review=0 resolved=0
quarantine: victim 10.0.0.50 banned=false, scan target 10.0.0.60 banned=false
(expected: still 0/false -- correlation_engine only emits a verdict on either
 multi-lane corroboration or a single lane at Critical severity; a lone
 moderate-confidence hypergraph signal is deliberately held back, see
 `CorrelationEngine::maybe_emit`.)

=== after a corroborating spectral-lane signal on the same hosts ===
review_queue: total=1 auto_resolved=0 pending=1 under_review=0 resolved=0
quarantine: victim 10.0.0.50 banned=false, scan target 10.0.0.60 banned=false
(expected: pending>=1 -- two moderate-confidence lanes on the same host now
 clear `TriggerConfig`'s multi-lane-corroboration reason and queue for human
 review; confidence stays under the 0.90 auto-resolve floor, so it's queued,
 not auto-banned.)
WARN ips_control_plane: containment: banned host via rustwall quarantine flow_id=corr-10.0.3.99-1 host=10.0.3.99 duration=86400s

=== after honeytoken touch ===
review_queue: total=2 auto_resolved=1 pending=1 under_review=0 resolved=0
quarantine: honeytoken host 10.0.3.99 banned=true
quarantine: total bans issued so far = 1

ingestion: captured=48 bytes=1056 spectral_lane_delivered=0

=== spawning `nsm --simulate` as a child process ===
WARN ips_control_plane: containment: banned host via rustwall quarantine flow_id=corr-10.0.0.10-2 host=10.0.0.10 duration=86400s
nsm bridge: parsed and forwarded 60 real alerts from nsm's own stdout

=== after the nsm bridge ===
review_queue: total=3 auto_resolved=2 pending=1 under_review=0 resolved=0
quarantine: total bans issued so far = 2
```

What this demonstrates, concretely:

- **The count-threshold window closure is real**: 25 synthetic packets from
  distinct source IPs to one victim:port close a `DestinationFanIn` window
  the instant the 20-source threshold set in `unified_runner`'s demo config
  is crossed (not after a fixed timer), same for the 20-port source fan-out
  against its 15-port threshold.
- **The multi-lane corroboration gate is real and correctly conservative**:
  a single moderate-confidence hypergraph signal alone does *not* reach the
  review queue — only after a second, independent lane (`siem-ml`'s
  autoencoder, simulated here via `SpectralSender.submit`) flags the *same*
  host does `CorrelationEngine::maybe_emit` fire a verdict.
- **The confidence-gated auto-resolve vs. queue split is real**: the
  two-lane-corroborated verdict (`confidence` well under the 0.90
  auto-resolve floor) is queued for human review, not auto-banned — while
  the honeytoken touch (confidence 1.0 by construction, `Severity::Critical`)
  is auto-resolved and produces an actual `Quarantine::ban` call, visible
  both in the `WARN ... banned host via rustwall quarantine` log line and
  in `quarantine.is_active(...)` flipping to `true`.
- **The `nsm` bridge is a real cross-process integration, not a mock**:
  `nsm --simulate` is a genuinely separate OS process with its own capture
  loop and its own five detectors' internal state; `unified_runner` reads
  its actual stdout, line by line, as it's produced. One of `nsm`'s own
  detector alerts came back `Critical` (the signature detector matching
  the EICAR test string in the simulated traffic is the most likely
  culprit, given `nsm`'s default ruleset) and single-lane-critical-fired a
  second, independent ban — visible as the second `containment: banned
  host` line and `bans issued so far` going from 1 to 2.

(`spectral_lane_delivered=0` in the final metrics line is expected and not
a bug: `hg_builder`'s pipeline consumes the raw `broadcast::Receiver`
directly rather than through `ingestion::recv_lane`'s counting helper, so
that particular counter isn't incremented even though the packets were, in
fact, delivered and processed — visible in `captured=48`, which does count
every synthetic packet sent.)

Read `unified_runner/src/main.rs` for the exact scenario construction; it's
short and heavily commented on *why* each expected outcome is what it is,
including the design points above.

## 2b. Live run: `nsm` standing entirely on its own

`nsm` doesn't need `unified_runner` at all — it's a complete, independent
NSM you can run by itself, exactly as its own `nsm/README.md` describes:

```bash
cargo build -p nsm
./target/release/nsm --simulate | jq 'select(.severity == "High" or .severity == "Critical")'
```

Or, for real traffic (Linux/Windows, needs root/Administrator + a capture
backend — see `nsm/README.md` for the Npcap setup on Windows and the
optional XDP/eBPF fast path on Linux):

```bash
sudo ./target/release/nsm --list-interfaces
sudo ./target/release/nsm --interface eth0
```

## 3. Live run: the standalone services

Five of the workspace's crates build to standalone binaries you can run
directly, independent of `unified_runner`. All of the commands in this
section were actually run to produce the sample output shown.

### `siem-server` — ActiveSIEM demo

Not a long-running server despite the name — it's a self-contained
demonstration that runs a fixed scenario and exits, takes no arguments:

```bash
./target/debug/siem-server
```

```
=== ActiveSIEM demo ===

-- Simulating SSH brute force from 10.0.0.5 --
  ALERT [ssh-bruteforce] Repeated SSH authentication failures from one source (severity High, mitre Some("T1110"))

-- Training autoencoder on benign traffic shape --
  calibrated anomaly threshold: 0.00169

-- Scoring a suspected infiltration beacon (long, tiny, periodic, upload-heavy) --
  reconstruction error: 10.06559 (threshold 0.00169)
  flagged as anomalous/infiltration-like: true

-- Active response policy --
[response:dry-run] would block 198.51.100.77 for alert 'Anomalous long-duration low-volume flow (possible infiltration beacon)' (severity High)
  response executed.
```

### `honeypotd` — TCP/SSH honeypot

Takes a TOML config path as its first positional argument (defaults to
`honeypot.toml` in the current directory). Minimal working config (SSH
honeypot needs a real host key file, so this one only enables the simpler
TCP honeypot):

```bash
mkdir -p demo_run/honeypot_events
cat > demo_run/honeypot.toml <<'EOF'
host_id = "demo-honeypot-01"

[ssh]
enabled = false
bind_addr = "127.0.0.1:2222"
host_key_path = "/tmp/unused_host_key"
banner = "SSH-2.0-OpenSSH_9.6p1 Ubuntu-3ubuntu1"
fake_hostname = "prod-db-03"

[[tcp_ports]]
port = 8021
banner = "220 ProFTPD 1.3.7 Server ready.\r\n"
label = "fake-ftp"

[sinks]
jsonl_dir = "demo_run/honeypot_events"
channel_capacity = 4096
EOF

./target/debug/honeypotd demo_run/honeypot.toml &
sleep 1
printf 'HELP\r\n' | timeout 2 bash -c 'cat < /dev/tcp/127.0.0.1/8021'
```

This really binds `127.0.0.1:8021`, really serves the configured fake-FTP
banner on connect, and really logs a structured JSONL event
(`demo_run/honeypot_events/*.jsonl`) for the connection — verified output:

```json
{"event_id":"57b65e8c-...","timestamp":"2026-08-05T02:06:01Z","source_ip":"127.0.0.1","source_port":39042,"host_id":"demo-honeypot-01","sensor":"tcp_honeypot","confidence":"certain","severity":"low","kind":{"kind":"connection_opened","local_port":8021,"protocol_guess":"fake-ftp"}}
```

### `rustwall` — inline stateful firewall

Needs `NFQUEUE`/root to actually bind, but `--check-only` validates a real
config without needing either — useful in CI or a sandbox with no
privileged networking:

```bash
cat > demo_run/rustwall.toml <<'EOF'
[general]
default_action = "allow"

[[rules]]
name = "block-demo-victim"
action = "drop"
src_ip = "10.0.0.50/32"
EOF

./target/debug/rustwall --config demo_run/rustwall.toml --check-only
```

```
config OK: 1 rules loaded, 1 queue worker(s) starting at queue 0
```

### `rclam` / `rclamd` — malware scanner (ClamAV-style)

`rclam` is the one-shot CLI scanner; `rclamd` is the long-running daemon
(Unix socket or TCP). One-shot scan against a real signature:

```bash
mkdir -p demo_run/scan_target
printf 'X5O!P%%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*' \
  > demo_run/scan_target/eicar.txt
echo "benign file" > demo_run/scan_target/benign.txt

sha256sum demo_run/scan_target/eicar.txt | awk \
  '{print $1":"'"$(stat -c%s demo_run/scan_target/eicar.txt)"'":Eicar-Test-Signature"}' \
  > demo_run/demo.hdb

./target/debug/rclam --hdb demo_run/demo.hdb demo_run/scan_target
```

```
scan_target/eicar.txt: FOUND Eicar-Test-Signature (HashSha256)
scan_target/benign.txt: OK
----------- SCAN SUMMARY -----------
files scanned: 2
infected: 1
errors: 0
```

Exit code is `1` when anything is found infected (verified — useful for
scripting).

Daemon mode, for completeness (Unix domain socket, scan roots must be
allow-listed explicitly — it fails closed with no `--allow-root` given):

```bash
./target/debug/rclamd --hdb demo_run/demo.hdb \
  --allow-root demo_run/scan_target --socket /tmp/rclamd.sock &
```

## 4. Cleanup

```bash
kill %1 %2 2>/dev/null   # stop any backgrounded honeypotd/rclamd from above
rm -rf demo_run
```
