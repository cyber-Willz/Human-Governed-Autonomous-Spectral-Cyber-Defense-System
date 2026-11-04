# honeypot_engine

Production honeypot suite for `active_spectral_defense`: medium-interaction
SSH honeypot (real protocol via `russh`), low-interaction multi-port TCP
honeypot (banner + protocol guessing), and a honeytoken watcher (decoy
files with inotify-based access detection). All three fan into one
`HoneypotEvent` schema over a bounded `tokio::mpsc` channel, matching the
typed fan-in pattern already used in `active-siem`'s correlation engine.

## Bug found and fixed after initial delivery

A real bug surfaced while testing against an RDP-focused flow dataset
(CSE-CIC-IDS2018 "Infiltration" style records, port 3389): `main.rs`'s
shutdown `select!` only waited for **one** task-completion event before
returning, tearing down the whole daemon. Since the honeytoken watcher
and a disabled SSH honeypot both exit immediately by design (nothing
configured / disabled), the daemon would shut itself down within
milliseconds of starting whenever those subsystems finished first --
even though the TCP honeypot and sink were still supposed to be running.
Fixed by looping on `tasks.join_next()` until every subsystem has
finished, rather than returning after the first one. Verified: server
now stays alive indefinitely with only the TCP listener and sink active.

**Verification**: stood up the TCP honeypot on the real RDP port (3389)
and fired 46 real TCP connections at it with payloads proportioned to a
sample RDP infiltration flow dataset's byte volumes. Result: 46/46
connections captured (92 events: `ConnectionOpened` + `RawPayload` per
connection), and the port-based protocol guesser correctly tagged all 46
raw-payload events as `"rdp"`. Note this dataset is CICFlowMeter-derived
flow-summary data (duration, packet/byte counts, IAT stats), not raw
bytes -- the honeypot doesn't parse that format, so this test used it only
to proportion realistic connection volume/timing against a real listener,
not as a literal replay.

## Verification status (be aware of what was and wasn't checked)

This was built and tested in a sandboxed container, not against your real
infrastructure:

- **`cargo check` and `cargo build` (debug) both pass clean, zero warnings**,
  against rustc/cargo 1.75.0 -- matching your MSRV pin exactly.
- **Live smoke test performed**: binary run against `honeypot.example.toml`
  (adapted with local paths), confirmed working:
  - SSH honeypot binds and generates/persists an ed25519 host key (0600 perms).
  - TCP honeypot on a decoy port sends its configured banner and captures
    raw payload + protocol guess on connect.
  - Decoy file is provisioned at startup with 0600 permissions.
  - Honeytoken watcher fires `HoneytokenTriggered` events on **write/modify**
    access to the decoy file, correctly landing in the JSONL sink.
- **`cargo build --release` (LTO) was not completed** -- it exceeded the
  sandbox's execution time limit. LTO release builds are slow but routine;
  this isn't expected to surface new errors (LTO failures are rare and
  usually linker/target-specific), but you should run it yourself before
  deploying and treat it as unverified until you do.
- **Honeytoken *read* detection (as opposed to write) did not fire** in
  the sandbox: a `cat` of the decoy file produced no inotify `Access`
  event, while writing to it did. This is very likely the container's
  overlay filesystem suppressing `IN_ACCESS` notifications (common with
  overlayfs and certain mount/atime options) rather than a bug in the
  watcher logic -- the watcher registered correctly and fired correctly
  for `Modify` events on the same path. **Verify `Access` events fire on
  your actual target filesystem before relying on read-only credential
  theft detection** -- if your production hosts also run on overlayfs
  with `noatime`, you may need to rely on `Modify`/`Close(Write)` events
  only, or add an eBPF-based file-access probe (you already have the
  XDP/eBPF infrastructure in `nsm-xdp` this could reuse) as a fallback
  for pure-read detection.
- **Not tested**: actual SSH client login flow end-to-end (auth
  accept/reject path was verified by reading the code against the real
  `russh` 0.44 `Handler` trait source, not by connecting a real SSH
  client), PostgreSQL sink (no Postgres instance in the sandbox --
  code compiles and the schema/queries are written against sqlx 0.7's
  API, but never executed against a live database), and the Unix socket
  forwarder (same -- compiles, not exercised).

## Architecture

```
src/
  event.rs         Unified HoneypotEvent schema (all sensors emit this)
  config.rs         TOML config schema + loader
  ssh_honeypot.rs    russh-based SSH server, real protocol negotiation
  tcp_honeypot.rs    Low-interaction multi-port TCP listeners
  honeytoken.rs      Decoy file provisioning + inotify watcher
  sink.rs            JSONL (rotation-aware) + Postgres (batched) + Unix socket sinks
  main.rs            Wiring: bounded mpsc bus, task supervision, shutdown
```

## Integration with active_spectral_defense

- **JSONL sink** (`sinks.jsonl_dir`) is the primary integration point --
  same `events_YYYYMMDD.jsonl` rotation convention as your existing
  SSH auth/command logs, so your existing log shipper's harvester config
  should pick these up with no changes.
- **Postgres sink** creates its own `honeypot_events` table (idempotent
  `CREATE TABLE IF NOT EXISTS`) -- point it at the same instance as your
  review queue if you want everything in one place, or a separate DB.
- **`neural_hmm` wiring**: honeytoken and SSH-auth events carry
  `Confidence::Certain` deliberately -- there's no legitimate baseline for
  these signals, so they're well suited as high-confidence emissions that
  should sharply move per-host belief state, rather than needing the
  probabilistic treatment your spectral autoencoder's scores need.

## Before deploying

1. Run `cargo build --release` yourself and confirm it completes (LTO
   build was not verified here due to sandbox time limits).
2. Verify honeytoken `Access` (read) events fire on your actual target
   filesystem -- see caveat above.
3. Change all example credentials/paths in `honeypot.example.toml` --
   the shipped decoy content is a placeholder, not a template to deploy
   verbatim (an attacker fingerprinting known Anthropic/example honeypot
   content is a real detection risk).
4. Run as an unprivileged user with `CAP_NET_BIND_SERVICE` if binding to
   sub-1024 ports, rather than as root -- same posture as your NSM's
   systemd unit with least-privilege capabilities.
5. Isolate the honeypot host/network segment from real production
   segments -- this is a low-interaction/medium-interaction design (no
   real shell, no real filesystem), but defense in depth still applies.
