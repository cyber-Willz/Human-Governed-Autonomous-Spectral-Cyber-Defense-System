# rustwall

An inline, stateful, rule-based firewall for Linux. `rustwall` is the
**decision engine**: it does not itself capture packets off the wire — it
binds to an NFQUEUE and makes accept/drop decisions for packets the kernel's
netfilter hooks hand it. You still need one `nft`/`iptables` rule to route
traffic into the queue. This split mirrors how commercial NGFWs separate the
data path (ASIC/NPU on FortiGate, dataplane on PAN-OS) from the policy engine
— we just use the kernel's existing, battle-tested netfilter data path
instead of writing our own from scratch.

## What this actually is / is not

This is a real, working L3/L4 stateful firewall: 5-tuple + CIDR + port-range
rule matching, first-match-wins evaluation, a genuine connection-tracking
table with TCP-state-aware fast-path for established flows, per-rule token
bucket rate limiting (SYN flood / scan throttling), tiered conntrack
timeouts, structured JSON audit logging, and bounded memory under table-flood
attack (fail-closed on a full conntrack table).

It is **not** a competitor to PAN-OS/FortiOS/FTD feature-for-feature. It has
no App-ID-style L7 application classification, no IPS signature engine, no
TLS decryption, no hardware offload, and no management-plane clustering. Those
are each their own multi-year subsystems. If you want L7 visibility, the
natural extension point is to have `nfqueue.rs` hand payload bytes for
`Established`-but-early-in-flow packets to a classifier (this is exactly where
your `spec_engine`/`nsm` work could plug in as the App-ID-equivalent stage,
with `rustwall` as the enforcement point instead of the passive tap).

## Build

```bash
sudo apt-get install -y libnetfilter-queue-dev pkg-config build-essential
cargo build --release
```

Binary lands at `target/release/rustwall`.

## Wire it into netfilter

`rustwall` supports two deployment models, and picking the right one matters
-- an early test session on this project caught exactly this mistake, so
it's worth being explicit: **the netfilter chain you queue determines
whether return traffic works at all.**

### Gateway / router mode (recommended default)

If `rustwall` sits on a box that routes traffic for other hosts (a
router, bridge, or VM gateway), queue the `forward` chain. Transit traffic
crosses `forward` exactly once per direction, so conntrack sees both legs of
every flow naturally -- this is the model most NGFW appliances actually run.

```bash
sudo nft add table inet rustwall
sudo nft add chain inet rustwall forward '{ type filter hook forward priority 0; }'
sudo nft add rule inet rustwall forward queue num 0-1 fanout
```

The `0-1 fanout` range matches `queue_workers = 2` in the example config --
adjust both together. `fanout` tells the kernel to distribute packets across
the queue range by flow hash so all worker threads actually get traffic
instead of everything landing on queue 0.

### Host firewall mode

If `rustwall` is protecting the box it runs on (not transit traffic), you
need **both** `INPUT` and `OUTPUT` queued -- queuing only `INPUT` means
locally-initiated outbound connections (DNS lookups, NTP, outbound HTTPS)
never get their SYN recorded in conntrack, so the *reply* traffic on `INPUT`
looks like unsolicited inbound traffic and gets dropped under
`default_policy`. This exact failure mode showed up as silently dropped NTP
replies during initial testing of this build -- it's not a hypothetical.

```bash
sudo iptables -I INPUT -j NFQUEUE --queue-num 0 --queue-balance 0:1 --queue-bypass
sudo iptables -I OUTPUT -j NFQUEUE --queue-num 0 --queue-balance 0:1 --queue-bypass
```

`--queue-balance 0:1` is iptables' equivalent of nft's `fanout` for a
2-worker config; omit it (or use just `--queue-num 0`) for `queue_workers = 1`.

`--queue-bypass` matters operationally: if `rustwall` crashes, the kernel
falls back to ACCEPT instead of black-holing all traffic. Decide deliberately
whether fail-open (`--queue-bypass`) or fail-closed (omit it) is correct for
your threat model — a firewall that fails open under load is a known real
weakness in some field deployments, so treat this as a conscious tradeoff,
not a default.

## Run

```bash
sudo RUST_LOG=rustwall=info ./target/release/rustwall --config /etc/rustwall/rustwall.toml
```

Validate a config without binding to the queue (safe to run in CI):

```bash
./target/release/rustwall --config rustwall.example.toml --check-only
```

## systemd unit

```ini
[Unit]
Description=rustwall inline firewall
After=network-pre.target
Before=network.target

[Service]
ExecStart=/usr/local/bin/rustwall --config /etc/rustwall/rustwall.toml
ExecReload=/bin/kill -HUP $MAINPID
Restart=on-failure
RestartSec=1
AmbientCapabilities=CAP_NET_ADMIN CAP_NET_RAW
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true

[Install]
WantedBy=multi-user.target
```

`CAP_NET_ADMIN` is required for NFQUEUE; `CAP_NET_RAW` is required for the
`reject` action's raw socket (used to send TCP RST / ICMP port-unreachable
replies). If you don't use `reject` anywhere in your ruleset you can drop
`CAP_NET_RAW` and `reject` rules will just behave as silent drops (with a
warning logged once at worker startup) -- there's no reason to grant more
than a given deployment actually needs.

`ExecReload` wires `systemctl reload rustwall` to SIGHUP, which re-reads the
config file and hot-swaps the rule engine **without** dropping the conntrack
table or restarting NFQUEUE workers -- existing connections keep flowing
uninterrupted while new connections immediately see the updated rules. Use
this instead of `systemctl restart` for routine rule changes; save the
restart for binary upgrades.

## Config format

See `rustwall.example.toml`. Top-level settings:

| field              | meaning                                                                 |
|--------------------|--------------------------------------------------------------------------|
| `queue_num`        | first NFQUEUE number to bind (default 0)                                |
| `queue_workers`    | number of worker threads across `queue_num..queue_num+N-1` (default 1)  |
| `default_policy`   | `accept` \| `drop` \| `reject`, applied when no rule matches            |
| `log_max_per_sec`  | cap on policy-decision log lines/sec process-wide; 0 = unlimited (default 200) |
| `metrics_listen`   | optional `host:port` to serve Prometheus-format metrics + admin control (default: disabled) |
| `metrics_auth_token` | optional bearer token required on the metrics/control endpoint (default: unauthenticated) |
| `quarantine_max_entries` | cap on distinct IPs tracked by dynamic quarantine at once (default 100,000) |
| `sync_to_os_firewall` | also push quarantine bans into the host's own firewall (default false) -- see OS Firewall Sync below |
| `aliases`          | named, reusable CIDR sets -- see Aliases below                          |
| `trusted`          | IPs that bypass rule evaluation *and* quarantine entirely               |

### Aliases (pfSense/OPNsense pattern)

```toml
[[aliases]]
name = "office"
cidrs = ["10.0.0.0/24", "192.168.1.5/32"]
```

Reference an alias from any rule's `src`/`dst` with `"alias:<name>"` instead
of a literal CIDR:

```toml
[[rules]]
name = "allow-office-ssh"
src = "alias:office"
dst_port = "22"
action = "accept"
```

The point is the same as pfSense/OPNsense Aliases: when the office's IP
range changes, or you're rotating a blocklist, you edit the alias once
instead of hunting down every rule that referenced the literal CIDR. An
alias referenced by name that isn't defined anywhere fails config load
loudly (`Config::load` returns an error naming the rule and the missing
alias) rather than silently matching nothing at runtime.

### Rules

Rules are evaluated top-to-bottom, first match wins, falling through to
`default_policy`. Each rule:

| field              | meaning                                                        |
|--------------------|-----------------------------------------------------------------|
| `protocol`         | `tcp` \| `udp` \| `icmp` \| `any`                               |
| `src` / `dst`      | CIDR (e.g. `10.0.0.0/24`), `*`/`0.0.0.0/0`, or `alias:<name>`   |
| `src_port` / `dst_port` | single port, `lo-hi` range, or `*`                       |
| `action`           | `accept` \| `drop` \| `reject`                                  |
| `rate_limit_pps`   | optional per-source-IP new-connection cap for this rule          |
| `auto_block_secs`  | optional: quarantine the source IP for this many seconds on match -- see Quarantine below |
| `auto_block_threshold` | matches required from the same source, within `auto_block_window_secs`, before the ban fires (default 1 = fire on first match) |
| `auto_block_window_secs` | rolling window, in seconds, that `auto_block_threshold` counts within (default 60) |
| `log`              | emit a structured log line on match                              |

`drop` vs `reject` is a real operator choice, not just phrasing: `drop`
silently discards the packet (port looks filtered/stealthed to a scanner);
`reject` sends back a TCP RST or ICMP port-unreachable so the client fails
fast instead of hanging on a connect() timeout. `reject` requires
`CAP_NET_RAW` (see systemd unit above); without it, `reject` rules behave
like `drop` and a warning is logged once per worker at startup.

### Quarantine / auto-block (Untangle-style behavioral blocking)

Set `auto_block_secs` on any non-`accept` rule and a source that matches it
gets added to a dynamic ban list for that duration -- every subsequent
packet from that IP is dropped immediately, before conntrack lookup or rule
evaluation, until the ban expires (repeated matches extend rather than reset
the remaining time). This is the same "quick block, checked first" pattern
as pfSense's floating quick rules, and functionally what Untangle's
behavioral quarantine and pfBlocker-NG's dynamic tables do on top of
pfSense/OPNsense -- implemented natively here instead of as a bolt-on
package. `trusted` IPs are exempt from quarantine as well as from rule
evaluation.

Quarantine state is intentionally **separate from the rule engine**: it's
runtime state earned by observed behavior, not configuration, so it
survives a SIGHUP rule reload the same way the conntrack table does. It does
*not* survive a full process restart (this falls under the same "no
clustering/HA" limitation documented below). It's also bounded --
`quarantine_max_entries` caps how many distinct IPs it will track at once,
the same way `conntrack.max_entries` bounds the connection table. Extending
an existing ban never fails on capacity; only admitting a never-before-seen
IP can be refused once the table is full, and when that happens the packet
that triggered it was already dropped/rejected by the rule itself -- the
only consequence is that *future* packets from that source go through
normal (slower) rule evaluation instead of the fast pre-rule quarantine
check, not that anything gets let through it shouldn't.

```toml
[[rules]]
name = "reject-known-scanners"
src = "alias:known_scanners"
action = "reject"
auto_block_secs = 3600
log = true
```

**Read this before using `auto_block_secs` anywhere:** quarantine bans by
source IP taken from the packet header, which is not authenticated. A bare
TCP SYN or a UDP packet can carry a forged source address, and the sender
doesn't need to complete a handshake or receive a reply for this firewall to
observe "traffic from X hit a reject rule" and quarantine X. Anyone who
knows a rule like the one above exists can spoof packets claiming to be from
an IP they want knocked offline -- a customer, a partner, your own
monitoring host -- and get this firewall to blacklist it for them, on
command, for the ban duration. This is the same weakness fail2ban-style
systems have always had, not something unique to this implementation, but
it's a real one and it's easy to miss. Prefer `auto_block_secs` on rules
that only match *after* a completed TCP handshake, where spoofing is
impractical, rather than on rules that inspect a bare SYN -- or accept the
risk knowingly, as `rustwall.example.toml` does for illustration.

**`auto_block_threshold` closes the trivial single-packet case of that
weakness, without pretending to close all of it.** By default (threshold 1)
a single match fires the ban, exactly as above. Set
`auto_block_threshold = 3` (with `auto_block_window_secs`, default 60) and
a source needs 3 matches within that window before the ban actually fires
-- the packet is still dropped/rejected every time regardless, it just
doesn't trigger quarantine until the threshold is reached. This is the same
pattern fail2ban's `maxretry` and Suricata's `threshold` rules use, and it's
worth being explicit about what it does and doesn't buy you: one spoofed
packet can no longer trigger a ban on its own once threshold is above 1, but
an attacker willing to send N spoofed packets instead of one is unaffected.
It raises the bar; it does not remove it. A cleverer-sounding single-packet
heuristic ("only trust packets with ACK set") was considered and rejected
for this codebase specifically because it doesn't actually help -- an
attacker spoofing a single packet controls every flag on it, so gating on
any one packet's flags is not a real check.

```toml
[[rules]]
name = "reject-known-scanners"
src = "alias:known_scanners"
action = "reject"
auto_block_secs = 3600
auto_block_threshold = 3
auto_block_window_secs = 120
log = true
```

A quarantined IP can now be removed on demand via
`POST /quarantine/unban/<ip>` on the metrics/control endpoint (see Metrics
below) -- useful for a false positive, or a spoofed trigger, without waiting
out the TTL or restarting the process.

## OS Firewall Sync (nftables / Windows Firewall)

### Architecture: never block the packet path on a subprocess call

An earlier version of this feature called the OS firewall backend's
`nft`/`netsh` subprocess **synchronously**, directly from `Quarantine::ban()`
and `sweep_expired()` -- which run on the NFQUEUE packet-processing thread
and the maintenance thread, respectively. That subprocess call has no
bounded latency; a hang (netlink lock contention, a stuck process table,
anything) would stall packet processing indefinitely, on a firewall whose
whole point is to keep processing packets. That's an availability bug, not
a style nitpick.

The fix: `ban()`/`sweep_expired()` now enqueue a `SyncJob` onto a bounded
MPSC channel (`chan.rs`) via `try_send` -- non-blocking, returns immediately
either way. A dedicated worker thread (`sync_worker.rs`) owns the channel's
receiver and the OS firewall backend, and is the *only* thing that ever
blocks on a subprocess call. If that worker thread hangs, it stalls itself,
not packet processing. If the queue fills up (default capacity 4096) faster
than the worker can drain it, new sync jobs are dropped -- counted via the
`rustwall_quarantine_sync_dropped_total` metric -- rather than applying
backpressure to `ban()`. This is a deliberate tradeoff: rustwall's own
in-process quarantine check is authoritative and completely unaffected by
sync-queue backpressure; OS sync is additive, not load-bearing for
correctness, so it's the one allowed to degrade under pressure.

The channel itself (`chan.rs`) is a general-purpose bounded MPSC primitive
with its own test suite, including a 200-run x 8-thread contention stress
test targeting the lost-wakeup race class specifically (`Condvar::wait`
atomically releases the lock and sleeps, so a notify from a concurrent
pop/disconnect can't land in the gap between a capacity check and going to
sleep). All 9 of its tests pass.

**This was verified end-to-end against a live kernel, not just unit-tested
in isolation.** A real TCP SYN to port 23, through actual NFQUEUE, produced:

```
"rule":"reject-telnet","verdict":"Reject"
"source auto-quarantined","ban_secs":3          <- ban() returned immediately here
"quarantine active, packet blocked before rule evaluation"   <- next packet, pre-rule fast path
```

and, checked independently against the live nftables ruleset immediately
after:

```
$ nft list set inet rustwall_dynamic banned_v4
elements = { 127.0.0.1 timeout 3s expires 1s8ms }
```

-- confirming the ban travelled all the way from a real packet, through rule
evaluation, through the channel, through the dedicated worker thread, into
actual kernel firewall state, while the packet-processing thread kept
handling unrelated traffic without any observable stall.

### Why sync at all: defense in depth and performance

Two reasons this is worth having rather than relying on rustwall's own drop
alone:

1. **Defense in depth.** If rustwall crashes or is killed, its own
   quarantine table dies with the process -- but an nftables set entry (or a
   Windows Firewall rule) keeps blocking the source at the kernel/OS level
   regardless of whether rustwall itself is even running.
2. **Performance.** A banned source's packets get dropped by the kernel
   before ever reaching NFQUEUE/userspace, instead of paying the netlink
   round-trip cost on traffic you already know you want to discard.

### Linux (nftables) -- real, tested, works today

On Linux, enabling this creates a dedicated `rustwall_dynamic` table with
`banned_v4`/`banned_v6` sets (the `timeout` flag set, so nftables expires
entries itself, kernel-side, without rustwall needing to drive that), and
drop rules hooked at both `input` and `forward` priority -10 -- covering
both deployment models this README documents (host firewall and
gateway/router) without extra config, since an unused hook just never sees
matching traffic.

This is a real, live-tested integration, not a sketch: `src/os_firewall.rs`
has an integration test (`cargo test --release -- --ignored`, requires `nft`
+ `CAP_NET_ADMIN`) that creates an actual ban, verifies it via `nft list set`
against the live ruleset, confirms the drop rule is correctly wired to the
set, and confirms nftables' own timeout expires the entry without any
unban call from rustwall.

Requires the `nft` binary installed and `CAP_NET_ADMIN` (see systemd unit
above -- already required for NFQUEUE itself). If nftables initialization
fails for any reason, rustwall logs a warning and falls back to its own
in-process quarantine only; it does not refuse to start over this.

### Windows Firewall -- code exists, not wired into a buildable binary yet

`src/os_firewall.rs` also contains a `WindowsFirewallSync` backend
(`#[cfg(target_os = "windows")]`) that adds/removes Windows Firewall block
rules via `netsh advfirewall firewall`. Being direct about what this is and
isn't: **rustwall's packet-inspection engine (NFQUEUE) is Linux-only.**
Windows has no NFQUEUE equivalent, and `main.rs` unconditionally depends on
the `nfq` crate, so today's single binary does not build for Windows at all
-- adding this backend didn't change that.

What it does provide: the `Quarantine`/`OsFirewallSync` bookkeeping is
plain, platform-independent Rust with no Linux-specific dependency. If you
want a detected threat on a Linux rustwall instance to also block at a
Windows host's firewall, the concrete next step is splitting the
quarantine/OS-sync logic out into a second, NFQUEUE-free `[[bin]]` target
that runs on the Windows side as a small sync agent -- receiving ban/unban
events from the Linux instance over the network (with its own auth/transport,
which is a real design task, not a one-line addition) and applying them
locally via this same `WindowsFirewallSync`. That agent doesn't exist yet;
this backend is the piece of it that does, written and structured so that
work is an extension rather than a rewrite. Unlike nftables, `netsh` rules
don't expire themselves -- whatever drives this backend is responsible for
calling `sync_unban` when a ban's TTL elapses, which is why `Quarantine`'s
sweep calls it on every expired entry rather than assuming OS-side cleanup.

Rules with a single exact `dst_port` (not a range or `*`) are indexed by
`(protocol, port)` for O(1) average-case lookup instead of a linear scan --
this matters once a ruleset grows past a few dozen entries. Range/wildcard
rules are still scanned linearly but merged into evaluation in their
original position, so first-match-wins ordering is identical to what a naive
scan would produce regardless of how many rules are indexed.

Once a flow is `accept`ed, its reply traffic (and subsequent packets on that
flow) bypass rule evaluation entirely via the connection-tracking fast path —
this is what makes it a *stateful* firewall rather than a stateless packet
filter, and it's the same reason ASIC/NPU-backed NGFWs only run full policy
evaluation on the first packet of a flow.

## Metrics & admin control endpoint

Set `metrics_listen = "127.0.0.1:9090"` (or any bind address) to expose:

- `GET /metrics` -- Prometheus-format counters: packets accepted/dropped/
  rejected, fragment drops, conntrack fast-path hits, table-full events,
  parse failures, suppressed log lines, quarantine blocks/bans/capacity
  rejections/sync failures/sync drops, manual unbans, and auth failures.
- `POST /quarantine/unban/<ip>` -- removes a quarantine ban immediately
  (200 `unbanned`), or 404 if that IP wasn't banned, or 400 if `<ip>`
  doesn't parse.

```bash
curl http://127.0.0.1:9090/metrics
curl -X POST http://127.0.0.1:9090/quarantine/unban/203.0.113.5
```

Set `metrics_auth_token = "..."` and both routes require
`Authorization: Bearer <token>`; missing or incorrect tokens get 401,
checked via constant-time comparison. This is off by default (a deliberate
opt-in, not a silent default that would break existing scrape configs on
upgrade) but strongly recommended once you bind beyond a fully trusted
loopback -- and worth turning on even on loopback, since without it any
local process or user can issue unban requests. Verified end-to-end against
a live running instance: no token → 401, wrong token → 401, correct token →
200; unban of a never-banned IP → 404; malformed IP → 400; wrong HTTP
method → 404; the manual-unban counter incrementing exactly once across
that whole sequence, confirming it only counts calls that actually reach
the handler.

```bash
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:9090/metrics
curl -X POST -H "Authorization: Bearer $TOKEN" http://127.0.0.1:9090/quarantine/unban/203.0.113.5
```

## Testing

```bash
cargo test --release          # 56 unit/integration tests
cargo test --release -- --ignored   # + live nftables integration test (needs nft + CAP_NET_ADMIN)
```

Covers rule-matching semantics (first-match-wins, exact-port index
correctness, trusted bypass, rate limiting, reject vs. drop verdicts, alias
resolution, auto-block propagation and threshold gating), quarantine
TTL/extension/sweep/capacity/manual-unban behavior, the OS-sync channel
wiring (jobs enqueue correctly, a full sync queue never blocks or fails the
underlying ban), the bounded channel itself (`chan.rs`, 9 tests including a
200x8-thread contention stress test for the lost-wakeup race class), the
metrics/control HTTP request parser and constant-time token comparison,
IPv4 and IPv6 reject reply construction (round-tripped through the same
parser used on real ingress traffic, not just checked against the writer
that built them), IPv6 extension header walking, and packet-parser
robustness against empty/garbage/truncated/fragmented input -- the parser
sits directly in the attack surface, so it needs to fail closed rather than
panic or misclassify on malformed packets, and that's asserted directly
rather than just hoped for.

## What changed since the first pass

An earlier version of this project was explicitly *not* production-ready.
Since then:

- **Multi-threaded NFQUEUE** — `queue_workers` spawns N worker threads across
  a queue range, paired with kernel-side `fanout`/`--queue-balance` so load
  actually distributes instead of pinning one core.
- **Real `reject`** — sends an actual TCP RST or ICMP port-unreachable via a
  raw socket, instead of silently behaving like `drop`.
- **Hot reload via SIGHUP** — rule changes apply without dropping conntrack
  state or restarting workers (`systemctl reload`, or `kill -HUP`).
- **Indexed rule matching** — exact-dst-port rules are hash-bucketed instead
  of linearly scanned, while preserving first-match-wins ordering exactly.
- **Log-rate limiting** — a flood no longer becomes a self-inflicted logging
  DoS; excess events are counted and summarized instead of logged individually.
- **Metrics endpoint** — Prometheus-format counters for capacity planning and
  alerting, where previously the only signal was grepping JSON logs.
- **Test suite** — rule-matching semantics and packet-parser robustness
  against malformed input are now asserted, not just informally believed.
- **Aliases** (pfSense/OPNsense pattern) — named, reusable CIDR sets
  referenced from rules via `alias:<name>` instead of duplicating literal
  CIDRs everywhere.
- **Dynamic quarantine / auto-block** (Untangle-style behavioral blocking) —
  a rule can set `auto_block_secs` to cut off a source wholesale for a
  duration, checked before rule evaluation on every packet. Bounded by
  `quarantine_max_entries` the same way conntrack is bounded, after an
  initial version of this feature shipped without a cap -- see the gaps
  list below for the spoofing caveat that comes with this feature
  regardless of the cap.
- **OS firewall sync** (`sync_to_os_firewall`) — quarantine bans can also be
  pushed into the host's own firewall: a real, live-tested nftables
  integration on Linux (kernel-enforced, native TTL expiry, defense in depth
  if rustwall itself crashes), plus a Windows Firewall (`netsh`) backend
  that exists in the code but isn't wired into a buildable Windows binary
  yet -- see OS Firewall Sync below for exactly what that would take.
- **Sync worker + bounded channel** (`chan.rs`, `sync_worker.rs`) — the
  first version of OS firewall sync called `nft`/`netsh` synchronously from
  the packet-processing and maintenance threads, with no timeout; a hung
  subprocess call could stall packet processing indefinitely. Fixed by
  routing every ban/unban through a bounded MPSC channel to a dedicated
  worker thread: `Quarantine::ban()`/`sweep_expired()` now `try_send` and
  return immediately, never blocking on a subprocess. The channel itself
  has its own test suite (9 tests, including a 200-run x 8-thread
  contention stress test for the lost-wakeup race class), and the full
  pipeline -- real packet through NFQUEUE, rule match, quarantine, channel,
  worker thread, actual `nft` state -- was verified end-to-end against a
  live kernel, not just unit-tested in isolation. See OS Firewall Sync
  below for the actual log output from that run.

## Resolved since the last audit

These were previously listed as open gaps. Each is closed with working,
tested code -- not just documentation -- and the evidence is summarized
below so "resolved" doesn't have to be taken on faith.

- **Fragmentation is now detected and handled distinctly, not silently
  misclassified.** `packet::parse` returns a three-way `ParseOutcome`
  (`Parsed` / `UnclassifiableFragment` / `Malformed`) instead of a plain
  `Option`. This matters beyond bookkeeping: before this fix, a non-first
  fragment parsed "successfully" as `L4Proto::Other(0)` with port 0 --
  which, since a rule's default port range is the full `0-65535` wildcard,
  meant a fragment could silently slip through any `protocol = "any"` rule
  that didn't explicitly restrict ports. That's a real policy-bypass path
  via fragmentation, not just a cosmetic classification gap, and it's now
  closed: `UnclassifiableFragment` is always fail-closed dropped before
  reaching rule evaluation at all, counted separately via
  `rustwall_fragment_drops_total` so a wiring problem (fragments arriving
  because kernel-side defrag isn't correctly hooked ahead of the queue
  redirect -- see Wire it into netfilter above) is visible and
  distinguishable from actual malformed/hostile traffic. Covered by tests
  proving the first fragment (which does carry a real L4 header per RFC
  791/8200) still classifies normally, and non-first IPv4 and IPv6
  fragments are correctly flagged rather than misread as protocol "any".
- **`reject` now supports IPv6.** `Rejecter` opens both an IPv4
  (`IP_HDRINCL`) and an IPv6 (`IPV6_HDRINCL`) raw socket. TCP gets a RST;
  everything else gets a port-unreachable reply -- ICMP type 3 code 3 for
  IPv4, ICMPv6 type 1 code 4 for IPv6. The IPv6 ICMP checksum is computed
  via etherparse's own `Icmpv6Header::with_checksum`, which correctly folds
  in the IPv6 pseudo-header per RFC 4443 section 2.3 -- unlike ICMPv4,
  which has no pseudo-header at all, so this genuinely needed different
  code, not a copy-paste of the v4 path with wider addresses. Verified by
  tests that build each reply and re-parse it independently through the
  same `packet::parse` this firewall uses on real ingress traffic,
  confirming correct addresses, ports, flags, and (for ICMPv6) a checksum
  that round-trips through independent parsing.
- **IPv6 extension header walking was already handled** -- this was listed
  as a gap before actually checking. `etherparse`'s `IpHeader::Version6`
  carries a full `Ipv6Extensions` (hop-by-hop options, destination options,
  routing, fragment, auth) and walks the chain to find the real transport
  header underneath. A test now proves this directly: a packet with a
  Hop-by-Hop Options header sitting between the IPv6 header and a TCP
  header still classifies as `L4Proto::Tcp` with the correct ports, not as
  unclassified. Worth naming as a lesson, not just a fix: this had been
  carried on the gaps list based on an assumption rather than a check of
  what the dependency actually does.
- **Quarantine's single-packet trigger has a real, honest mitigation.**
  `auto_block_threshold` (default 1, preserving the original behavior)
  requires N matches from the same source within `auto_block_window_secs`
  before a ban actually fires, instead of a single match. This does not
  eliminate spoofing -- an attacker can still send N spoofed packets
  instead of one -- but it closes the trivial single-packet case, the same
  way fail2ban's `maxretry` and Suricata's `threshold` rules work. Also
  worth being honest about a wrong turn: the first idea considered here was
  "only trust packets with the TCP ACK flag set," which turned out to be
  fake protection on inspection -- an attacker spoofing one packet controls
  every flag on it, so that check would have added complexity without
  adding safety. It was abandoned before shipping rather than merged
  anyway.
- **Metrics endpoint now supports bearer-token auth.** Set
  `metrics_auth_token` and both `GET /metrics` and the new admin route
  below require a matching `Authorization: Bearer <token>` header via a
  constant-time comparison; missing/incorrect tokens get 401. Off by
  default (a deliberate opt-in, not a silent default that would break
  existing scrape configs on upgrade) -- see Metrics below for the actual
  401/401/200 sequence this was verified against on a live running
  instance.
- **Manual quarantine unban now exists.** `POST /quarantine/unban/<ip>`
  removes a ban immediately rather than waiting for TTL expiry or a
  process restart, gated by the same bearer token as the metrics route.
  Live-verified (auth-required 401, unknown-IP 404, malformed-IP 400,
  wrong-method 404, and the manual-unban counter incrementing exactly once
  across those calls) against a running instance without needing to
  exercise the NFQUEUE/iptables path at all, since the control endpoint is
  a separate TCP listener.

## Known gaps / honest limitations (still real)

- **No clustering / HA.** State is process-local; a restart drops the
  conntrack table (existing connections will need to re-handshake, which for
  TCP means a brief stall, not necessarily a hard failure, but plan for it).
  SIGHUP reload avoids this for rule changes, but a binary upgrade or crash
  still means every active connection re-handshakes. The nftables OS-sync
  backend is a partial exception for already-issued bans specifically (its
  state lives in the kernel, independent of the rustwall process), but
  conntrack itself has no such independence.
- **No signature-based IPS.** This stops what your rules describe; it does
  not detect exploit payloads inside allowed flows.
- **Quarantine bans an unauthenticated field (source IP).** `auto_block_secs`
  is a real feature but a real weapon: anyone who can spoof a packet
  matching a quarantine-triggering rule can get an arbitrary IP banned by
  this firewall on command. `auto_block_threshold` (see above) closes the
  single-packet case but does not eliminate spoofing outright -- a
  sustained spoofed stream can still reach threshold. Bounded by
  `quarantine_max_entries` so it can't exhaust memory, but the spoofing
  risk itself isn't something a size cap or a match-count threshold fully
  fixes -- see the caveat in `rustwall.example.toml` next to
  `auto_block_secs`. This applies equally, and arguably more seriously,
  when `sync_to_os_firewall` is on: a spoofed trigger now also modifies the
  host's actual firewall, not just rustwall's in-process state.
- **No Windows build.** `main.rs` depends unconditionally on the `nfq`
  crate (Linux netfilter), so rustwall does not compile for Windows today.
  The Windows Firewall sync backend in `os_firewall.rs` is real code, but
  it only becomes reachable if someone splits the quarantine/OS-sync logic
  into a separate NFQUEUE-free binary -- see OS Firewall Sync above.
- **No auth/transport for cross-host ban sync.** `sync_to_os_firewall`
  only talks to the firewall on the same host rustwall is running on.
  Pushing a ban from a Linux rustwall instance to a *different* machine's
  firewall (Windows or otherwise) isn't implemented -- it would need its
  own network protocol and authentication, which is a real design task
  and shouldn't be understated as a follow-on feature.
- **Aliases are static, not feed-based.** pfSense/OPNsense (via pfBlocker-NG)
  can populate an alias from an external URL on a schedule; rustwall's
  aliases are defined directly in the TOML and only change via SIGHUP
  reload of the config file. Feeding an alias from a URL is a natural
  extension but isn't implemented here.
