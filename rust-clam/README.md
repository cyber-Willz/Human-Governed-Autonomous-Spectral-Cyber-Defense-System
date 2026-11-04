# rust-clam

A Rust malware scanner: a hash + wildcard signature engine, zip/gzip
archive recursion with hard-enforced bomb protection, and PE
packer/structural heuristics. Ships as a one-shot CLI (`rclam`) and an
async scanning daemon (`rclamd`) speaking a `clamd`-compatible protocol
subset.

This is a Cargo **workspace** (multiple crates, no root binary) — `cargo
run` at the top level has nothing to run by itself. Use one of:

```powershell
# from the workspace root (C:\...\clam\ in your layout)
cargo run -p rclam-cli --bin rclam -- --ndb signatures\sample.ndb --hdb signatures\sample.hdb <path-to-scan>
cargo run -p rclam-daemon --bin rclamd -- --ndb signatures\sample.ndb

# or, cd into the crate itself, then plain `cargo run` works:
cd crates\rclam-cli
cargo run -- --ndb ..\..\signatures\sample.ndb <path-to-scan>
```

## Building

```
cargo build --release --workspace
```

Binaries land in `target/release/rclam` (and `rclamd`).

Requires a Rust toolchain new enough for edition 2021 (1.75+ is known to
work; CI additionally tests on the latest stable release).

## Running the test suite

```
cargo test --workspace
```

`cargo test --workspace` runs the full suite across `sig-engine`,
`archive-guard`, `pe-analyze`, `scanner-core`, and `rclam-daemon` (60 tests
total), covering (among other things): the archive-recursion depth guard,
zip/gzip/tar-bomb total-size/ratio guards, empty-file handling,
non-regular-file (FIFO/device) rejection, oversized-file skipping,
checked-arithmetic PE header parsing, the scan-path allowlist (including
symlink-escape attempts), per-peer rate/concurrency limiting, and eight
integration tests that drive real `TcpStream` connections through the
daemon's TCP transport end-to-end (`PING`/`VERSION`/`SCAN`/`STATS`,
allowlist denial, oversized-command handling) -- this is the exact code
path Windows uses exclusively, previously compiled on Windows CI but never
functionally exercised there; these tests now run, and pass, on every OS
in the CI matrix.

`sig-update/` (the signature-update tool) is a separate, detached
workspace with its own 17 tests -- see the Signature updates section
below.

```
cargo clippy --workspace --all-targets
cargo fmt --all -- --check
```

## Signature file formats

- `--ndb <file>`: wildcard byte-pattern signatures, `name:hexpattern` per
  line. `??` = single wildcard byte, `*` or `*{min-max}` = variable gap.
- `--hdb <file>`: exact-hash signatures, `hexhash:size:name` per line
  (`size` may be `*` for any size; hash may be MD5/SHA1/SHA256 by length).

Both flags may be repeated to load multiple database files.

## Archive recursion

Both the CLI and the daemon recurse into archive contents automatically:

- **Zip** (`PK\x03\x04` magic): every entry is extracted and scanned,
  including entries that are themselves zips, down to `--max-depth`.
- **Gzip** (`1F 8B` magic), including concatenated gzip members: the
  decompressed stream is fed back through the scanner as one logical
  child, e.g. `payload.bin.gz -> (gunzip)`.
- **Tar** (POSIX `ustar` magic at header offset 257): every regular-file
  entry is extracted and scanned, including a tar member that is itself an
  archive. Combined with gzip recursion above, this covers `.tar.gz`
  transparently — the gzip layer decompresses, the result is detected as
  tar, and each member is scanned individually. Legacy V7 tar (no magic
  field at all) is not detected, since there's no magic to reliably anchor
  detection on without risking false positives on arbitrary binary data.

Every byte that comes out of an archive is charged against a shared
`GuardBudget` (entry count, total uncompressed size, per-entry
compression ratio, and recursion depth), and extraction is aborted the
instant any limit is crossed — the limits are structural, not
configuration that a specific decoder could bypass. A file that isn't a
recognized archive format is simply signature/PE-scanned as-is; there's no
"unsupported archive" error path a bomb could hide behind.

Other container formats (7z, rar, cab) are not yet supported; a file in
one of those formats is scanned as opaque bytes rather than recursed into.
This is a coverage gap worth being aware of, not a silent failure —
nothing crashes or hangs on them, they just aren't unpacked.

## Resource limits

`GuardLimits` (re-exported from `scanner-core` as well) controls:

| Field                     | Default   | Purpose |
|---------------------------|-----------|---------|
| `max_depth`                | 16        | nested-archive recursion depth |
| `max_entries`               | 10,000    | entries extracted per archive |
| `max_total_uncompressed`   | 4 GiB     | cumulative decompressed bytes, whole recursive scan |
| `max_ratio`                 | 1000      | per-entry (uncompressed / compressed) ratio |
| `read_chunk`                | 1 MiB     | single read-call size cap |
| `max_file_size`             | 200 MiB   | top-level file size scanned directly (larger files are **skipped**, not erred) |

Both `rclam` and `rclamd` expose `--max-file-size` and `--max-depth` on the
command line; the rest are currently fixed at their defaults (extend
`Args` in either binary if you need to tune them further).

Files that aren't plain regular files (FIFOs, sockets, device nodes) are
never opened for scanning at all — the check happens via `stat`, before
any `open()` call, specifically so a FIFO with no writer on the other end
can't block a scan indefinitely.

## CLI exit codes

`rclam` follows `clamscan`'s convention, since scripts and CI pipelines
depend on being able to tell these apart:

- `0` — scan completed, nothing found
- `1` — scan completed, at least one infected/detected file
- `2` — an operational error (bad signature file, unreadable path, i/o error)

## `rclamd` daemon

```
rclamd --ndb signatures/sample.ndb --hdb signatures/sample.hdb
```

- **Transport**: a Unix domain socket on Linux/macOS (`--socket <path>`,
  default `/tmp/rclamd.sock`), a TCP loopback socket on Windows
  (`--port <n>`, default `3310`). Force TCP on Unix too with `--tcp`. Both
  transports run through one identical protocol handler.
- **Protocol**: `PING`, `VERSION`, `SCAN <path>` / `CONTSCAN <path>`,
  `SHUTDOWN` — one command per connection, newline-terminated.
- **Socket permissions**: the Unix socket is chmod'd to `--socket-mode`
  (default `0600`, owner-only) right after binding. `clamd`'s historical
  default of a group-writable socket has been a recurring
  privilege-escalation footgun in deployments that got the surrounding
  group membership wrong; the safer default is chosen here and left to the
  operator to loosen deliberately.
- **Concurrency limiting**: `--max-connections` (default 64) bounds how
  many scans run at once, daemon-wide. Once saturated, new connections are
  told `ERROR server busy, try again later` and closed immediately rather
  than queued — this bounds worst-case latency and fd/memory use under
  load instead of degrading silently.
- **Per-peer rate and concurrency limiting**: independent of the daemon-wide
  cap above, `--peer-rate-per-sec` (default 5), `--peer-burst` (default 20),
  and `--peer-max-concurrent` (default 8) bound a *single* client sharing
  the daemon with others. Without this, one noisy or malicious client could
  consume most of `--max-connections` by itself and starve every other
  tenant. "Peer" is the connecting user's uid over a Unix socket
  (`SO_PEERCRED`, via `peer_cred()`) or the source IP over TCP, so two
  local users sharing one daemon are limited independently of each other.
  Rejected connections get `ERROR rate limit exceeded` or `ERROR too many
  concurrent connections from this peer` and are closed, same
  fail-fast/no-queueing philosophy as the global cap.
- **Scan path allowlist (fails closed)**: socket permissions control *who*
  can connect, not *what* a connected client may then ask to be scanned.
  `--allow-root <dir>` (repeatable) is the layer that answers that second
  question: a `SCAN`/`CONTSCAN` request is only honored if the requested
  path canonicalizes (symlinks resolved, so a symlink inside an allowed
  root pointing outside it doesn't grant access) to somewhere underneath
  one of the configured roots. **With no `--allow-root` configured, every
  scan request is refused** — a shared daemon with no explicit roots has no
  safe default other than "scan nothing." Denials are deliberately generic
  (`ERROR access denied`, no distinction between "outside the allowlist"
  and "doesn't exist") so an unprivileged client can't use the daemon as a
  filesystem-existence oracle against paths it has no business probing.
- **Metrics and health**: `--metrics-addr`/`--metrics-port` (default
  `127.0.0.1:9310`, set `--metrics-addr ""` to disable) serve `GET
  /healthz` (plain `200 ok`) and `GET /metrics` (Prometheus text format:
  connection/scan/rejection counters, uptime, free connection-queue slots).
  A `STATS` protocol command is also available for the same counters
  without needing a separate HTTP client. This is what an orchestrator
  needs to page on "the scan queue is growing," not just "the process
  crashed" (which logs alone already covered).
- **DoS-resistant command parsing**: a client that never sends `\n` cannot
  make the daemon buffer unbounded data per connection — the command line
  read is capped at 8 KiB, past which the connection is closed with
  `ERROR command line too long`.
- **Graceful shutdown**: `SIGTERM` or Ctrl+C stops the accept loop and (on
  Unix) removes the socket file, instead of the process being killed
  mid-request by an unhandled signal.
- **Resilience by design**: the workspace profile deliberately keeps
  `panic = "unwind"`. If a single scan request somehow panics, tokio
  catches it at the task boundary — only that one connection is lost, the
  daemon keeps serving everything else. (`panic = "abort"` would trade
  that away for a smaller binary, the wrong trade for a long-running
  service.)
- **Logging**: structured logs via `tracing`, to stderr, controlled with
  `RUST_LOG` (e.g. `RUST_LOG=debug rclamd ...`). Scan *results* on the
  CLI's stdout and the daemon's protocol responses are unaffected by log
  verbosity — they're the tool's actual output, not diagnostics.
- **Startup failures don't panic**: a bad `--ndb`/`--hdb` file logs a clear
  error and exits with a nonzero status, rather than an `unwrap`/`expect`
  panic backtrace.

### Deployment

- `deploy/rclamd.service` — a hardened systemd unit (dedicated user,
  `ProtectSystem=strict`, `NoNewPrivileges`, restricted address families,
  etc). Adjust `ReadOnlyPaths` to whatever directory clients will ask it
  to scan.
- `Dockerfile` — multi-stage build, runs as an unprivileged `rclam` user,
  exposes the TCP transport on `3310`.

```
docker build -t rclam .
docker run --rm -p 3310:3310 -v /path/to/scan:/scan:ro rclam
```

## Signature updates

`sig-update/` is a separate `freshclam`-equivalent CLI (`rclam-sigupdate`),
detached from the main workspace (own `Cargo.toml`, own dependency
resolution) since its dependencies (`ed25519-dalek`, `ureq`) have a higher
MSRV than the daemon's own pinned dependency set:

```
rclam-sigupdate --manifest-url https://sigs.example.com/manifest.json \
    --sig-dir /etc/rclam/signatures \
    --public-key <hex ed25519 pubkey>   # optional but recommended
```

It fetches a small JSON manifest (version + per-file name/sha256/url),
verifies every downloaded file's SHA-256 against it, stages everything in a
temp directory, and only then atomically swaps that directory into place
as `--sig-dir` — nothing in the live signature directory is touched until
every file has verified. The previous generation is kept as
`<sig-dir>.previous` for a one-step rollback (`sig_update::rollback`).
With `--public-key`, it additionally requires and verifies a detached
Ed25519 signature over the manifest itself (fetched from
`<manifest-url>.sig`), which is what catches a compromised or spoofed
*origin*/CDN rather than just an on-path tampering attempt (TLS on the
manifest fetch already covers the latter). This was exercised end-to-end
against a local test HTTP server, including a deliberately tampered file
and manifest (both correctly rejected, live directory left untouched) —
see `sig-update/`'s tests for the parts that don't need a network. Run it
from cron or a systemd timer, the same way `freshclam` traditionally is;
it does not run as its own daemon.

## Fuzzing

`fuzz/` is a `cargo-fuzz` scaffold covering the six parsers that consume
fully attacker-controlled bytes: PE headers, `.ndb`/`.hdb` signature
definitions, and the zip/gzip/tar extractors. Every target type-checks
against the current APIs (`cargo check`, stable toolchain); none has
actually been fuzz-run as part of this change, since that needs a nightly
toolchain and sanitizer instrumentation this environment couldn't install.
See `fuzz/README.md` for exact commands and the reasoning behind each
target — treat it as ready-to-run scaffolding, not as evidence fuzzing has
found (or ruled out) anything yet.

## Dependency auditing

`cargo audit` passes clean (0 advisories against 106 crate dependencies) as
of this change. It previously never completed a run in practice; running
it surfaced one real advisory (`RUSTSEC-2026-0204`, an invalid-pointer
dereference in `crossbeam-epoch`'s `fmt::Pointer` impl, pulled in
transitively via `rayon`), fixed here by bumping `crossbeam-epoch` to
0.9.20 in `Cargo.lock`. Re-run `cargo audit` yourself before relying on
that claim staying true — it's a statement about the dependency tree today,
not a permanent property of the project.

## CI


`.github/workflows/ci.yml` runs, on every push/PR: build + test on
Linux/macOS/Windows, `cargo fmt --check`, `cargo clippy -D warnings`,
`cargo audit` against the dependency tree, and a `fuzz-build` job that
compiles (but does not run) every `cargo-fuzz` target so API drift between
the fuzz harness and the scanned crates is caught immediately.

## Design notes / threat-model callouts

A few decisions in this codebase are deliberate responses to known
historical AV-scanner CVE classes, and are worth knowing about if you're
extending it:

- **No bytecode VM for signatures** (`sig-engine`): matching is pure
  data-driven segment verification. There's no interpreted signature
  language, which removes an entire historical CVE class (malformed-
  bytecode memory corruption in the scanner itself, as ClamAV has hit
  repeatedly).
- **Structural, not configurable, archive limits** (`archive-guard`):
  every extraction call is threaded through a `GuardBudget` decremented as
  bytes come out; there's no code path that produces archive contents
  without going through the budget check.
- **Bounds-checked PE parsing** (`pe-analyze`): every field access goes
  through `get_*` helpers and `checked_add`/`checked_mul` rather than raw
  pointer casts or plain arithmetic over attacker-controlled offsets — a
  malformed or truncated PE can only produce an `Err`, never UB or an
  offset that wraps past a bounds check on a 32-bit target.
- **Stat-before-open** (`scanner-core`): file-type checks happen via
  `stat`, before `File::open` is ever called, because opening a FIFO for
  reading blocks until a writer connects — checking the type *after*
  opening would be too late to prevent the hang.

## Known limitations, as of this change

Honest accounting of what's still open, so nothing above is mistaken for
more coverage than it has:

- **7z/rar/cab are still not unpacked** (tar was added; zip and gzip
  already worked). Files in those formats scan as opaque bytes rather than
  recursing into them — not a crash risk, a detection gap.
- **The fuzz harness is a compiled scaffold, not an executed one.** See
  the Fuzzing section above — nightly + `cargo-fuzz` instrumentation
  couldn't be installed in the environment this change was made in.
- **The signature-update tool's core flow (fetch, per-file SHA-256 check,
  Ed25519 manifest-signature check, atomic staging swap, tamper rejection,
  rollback) was exercised end-to-end against a local test HTTP server**,
  which is real coverage of the logic, but not the same as running it
  against a production distribution point over real TLS, or under
  concurrent/interrupted-update conditions.
- **Windows now has real functional test coverage** of the TCP transport
  it uses exclusively (eight integration tests drive actual `TcpStream`
  connections through the full protocol handler, running on
  `windows-latest` in CI), which is a meaningful step up from "compiles
  there" — but this was still developed and manually exercised on Linux
  only; nothing here is a substitute for someone actually running `rclamd
  --port ... ` on a Windows box before depending on it in production there.
- **The per-peer rate limiter's idle-peer eviction (10 minutes) and the
  metrics endpoint's minimal hand-rolled HTTP parsing** are both
  intentionally simple; neither has been load-tested under sustained
  high-cardinality peer churn or a hostile HTTP client sending partial/
  malformed requests at the metrics port specifically (as opposed to the
  scan protocol, which does have that scrutiny via the oversized-command
  test).
