# nsm — a concise, cross-platform Network Security Monitor in Rust

A single-binary NSM: live packet capture → 5-tuple flow tracking →
five parallel detectors → NDJSON alerts on stdout. Runs on **Linux and
Windows** from the same source tree — `pnet`'s datalink layer abstracts
raw `AF_PACKET` sockets on Linux and Npcap/WinPcap on Windows behind
one API, so `packet.rs`, `flow.rs`, and every detector are 100%
platform-agnostic. Only interface naming and privilege elevation
differ, both handled in `capture.rs` via `cfg(unix)` / `cfg(windows)`.

## Build — Linux

```bash
cargo build --release
sudo ./target/release/nsm --list-interfaces
sudo ./target/release/nsm --interface eth0
```

No extra setup needed: raw-socket capture works out of the box with
root or `CAP_NET_RAW` (see below).

## Build — Windows

Three one-time setup steps (this is `pnet`'s requirement, not this
project's):

1. **Install Npcap** — https://npcap.com/#download — during install,
   check **"Install Npcap in WinPcap API-compatible Mode."**
2. **Get the Npcap SDK** — https://npcap.com/#download (separate "SDK"
   download) — unzip it somewhere, e.g. `C:\npcap-sdk`.
3. **Point the linker at it** before building, in PowerShell:
   ```powershell
   $env:LIB = "C:\npcap-sdk\Lib\x64;$env:LIB"
   cargo build --release
   ```
   You must use the **MSVC** Rust toolchain (`rustup default
   stable-x86_64-pc-windows-msvc`), not the GNU one — Npcap's
   `Packet.lib` only links against MSVC.

Then, from an **elevated (Administrator)** terminal:

```powershell
.\target\release\nsm.exe --list-interfaces
.\target\release\nsm.exe --interface 2
```

Windows device names are opaque (`\Device\NPF_{9B2E1B4E-...}`), so
`--list-interfaces` prints a numeric index and description — pass
either the index (`--interface 2`) or a substring of the description
(`--interface "Ethernet"`) instead of the raw name.

## Build — XDP fast path (Linux only, optional)

By default `nsm` captures via `pnet`/`AF_PACKET`, same as always. On
Linux you can opt into an XDP-based capture + enforcement path instead:
packets are parsed in-kernel at the earliest RX hook, mirrored to
userspace over a lock-free ring buffer instead of a per-packet
`AF_PACKET` copy, and confirmed attackers can be `XDP_DROP`ped by the
kernel before they reach the network stack at all.

This is two separate builds:

1. **The eBPF program** (`xdp/nsm-ebpf`) — a freestanding, `no_std`
   binary that gets loaded into the kernel. Needs its own toolchain:

   ```bash
   rustup toolchain install nightly --component rust-src
   cargo install bpf-linker   # needs LLVM: apt install llvm-dev libclang-dev clang
   ./scripts/build-ebpf.sh
   ```

   This produces `xdp/nsm-ebpf/target/bpfel-unknown-none/release/nsm-ebpf`.

2. **`nsm` itself, with the `xdp` feature**:

   ```bash
   cargo build --release --features xdp
   sudo ./target/release/nsm --interface eth0 --xdp
   # native XDP unsupported by this NIC's driver? fall back to SKB mode:
   sudo ./target/release/nsm --interface eth0 --xdp --xdp-mode skb
   # let confirmed port-scan/SYN-flood sources get dropped in-kernel:
   sudo ./target/release/nsm --interface eth0 --xdp --xdp-auto-block
   ```

   Without `--features xdp` (the default), `nsm` builds exactly as
   before — the `xdp` module, its `aya` dependency, and the `--xdp*`
   flags don't exist in the binary at all.

Requires Linux 5.8+ (ring buffer maps), root or
`CAP_BPF`+`CAP_NET_ADMIN`+`CAP_NET_RAW`, and — for `--xdp-mode native`
(the default) — a NIC driver with native XDP support; `--xdp-mode skb`
works on any NIC as a slower fallback.

## Run without any of the above (either OS)

```bash
# No root/Administrator, no NIC, no Npcap needed --
# exercises every detector with synthetic traffic:
./target/release/nsm --simulate
```

Alerts stream as newline-delimited JSON on stdout (pipe into `jq`,
Filebeat, Vector, or a SIEM); operational logs go to stderr so the two
never mix. Ctrl+C shuts down cleanly on both platforms.

```bash
./target/release/nsm --simulate | jq 'select(.severity == "High" or .severity == "Critical")'
```

## Architecture

```
capture.rs   pnet datalink capture (raw sockets on Linux, Npcap on Windows) -> blocking thread -> mpsc channel
             cross-platform interface resolution (name / index / description substring)
             OS-specific privilege-error hints (root+CAP_NET_RAW vs elevated Administrator+Npcap)
src/xdp/     (Linux, --features xdp) userspace loader: attaches xdp/nsm-ebpf,
mod.rs       streams ring-buffer events -> the SAME mpsc channel as capture.rs,
             exposes block_ipv4()/unblock_ipv4() for kernel-level enforcement
xdp/nsm-ebpf in-kernel XDP program (separate no_std build, see below): parses
  src/       headers, XDP_DROPs blocklisted sources, mirrors PacketEvents to a ring buffer
xdp/nsm-common  #[repr(C)] PacketEvent struct shared by both sides of that boundary
packet.rs    Ethernet/IPv4/IPv6/TCP/UDP/ICMP parsing -> PacketMeta (fully platform-agnostic)
flow.rs      DashMap-backed bidirectional 5-tuple flow table (conn.log-style), idle reaping
detect/
  portscan.rs    distinct dest-port / dest-host count per source in a sliding window
  synflood.rs    SYN rate per destination, weighted by number of distinct sources
  dns_tunnel.rs  DNS label length + Shannon entropy + query-rate heuristics
  beacon.rs      coefficient-of-variation on connection inter-arrival times (C2 check-ins)
  signature.rs   Suricata-style byte-pattern content matching
alert.rs     severity-tagged Alert struct, JSON emission
sim.rs       synthetic traffic generator for --simulate (works identically on both OSes)
```

Each detector maintains its own bounded, self-expiring state
(`DashMap` + sliding windows), so the engine is a single
`analyze(&PacketMeta) -> Vec<Alert>` call per packet with no shared
locks across detectors, and no platform-specific code outside
`capture.rs` (pnet) / `src/xdp/` (XDP).

### eBPF/XDP fast path

`src/xdp/mod.rs` is a second *producer* for the exact same
`mpsc::Sender<PacketMeta>` that `capture::spawn_capture_thread` already
feeds — every detector, `flow.rs`, and `run_event_loop` are completely
unaware of which backend captured a given packet. `--xdp` just changes
who calls `tx.send(meta)`:

- **Capture**: `xdp/nsm-ebpf` parses Ethernet/IPv4/IPv6/TCP/UDP headers
  directly out of the packet buffer at the XDP hook (in the NIC driver
  in native mode) and pushes a fixed-size `PacketEvent` into a `RingBuf`
  map. Userspace reads it asynchronously (`tokio::io::unix::AsyncFd`
  over the ring buffer's fd) and converts each event back into the
  same `PacketMeta` the pnet path produces.
- **Enforcement**: a `BLOCKLIST_V4` eBPF hash map (source IPv4 ->
  expiry timestamp) is checked before any parsing happens. With
  `--xdp-auto-block`, `run_event_loop` pushes the source IP of any
  `Critical` port-scan/SYN-flood alert into that map; the *next*
  packet from that source is `XDP_DROP`ped in the driver, before
  `sk_buff` allocation or the network stack — actual bypass, not just
  faster logging.

Trade-off: the XDP fast path doesn't capture payload bytes (see
`nsm-common::PacketEvent`'s docs), so `signature.rs`'s content matching
never fires on XDP-sourced packets. Everything else (port scan, SYN
flood, DNS tunneling, beaconing) works identically either way, since
they only ever looked at `PacketMeta`'s header fields to begin with.

## Detectors, briefly

- **Port scan** — flags a source touching ≥20 distinct ports on one
  host (vertical) or ≥15 distinct hosts (horizontal) within 10s.
- **SYN flood** — flags ≥100 SYNs at one destination within 5s;
  escalates to Critical once more than 20 distinct sources are involved.
- **DNS tunneling** — combines label length, Shannon entropy of the
  query name, and per-source query rate; five-plus high-entropy/long
  labels in 30s raises a High alert.
- **Beaconing** — tracks inter-arrival time between new connections to
  the same (src, dst, port); a coefficient of variation ≤0.15 over 6+
  samples looks like scheduled malware check-in traffic rather than
  human-driven use.
- **Signatures** — a small default ruleset (cleartext basic-auth,
  UNION SELECT, `${jndi:`, EICAR string, etc.) as a byte-substring
  content-match engine; trivial to extend with your own `Rule` entries.

## Extending

Add a new detector by implementing `observe(&PacketMeta) -> Option<Alert>`
(or `Vec<Alert>`) with your own state struct, then wire it into
`DetectionEngine::new()` / `analyze()` in `detect/mod.rs`. Every
detector owns its state independently, so new ones can't affect
existing detection paths, and none of them need to know which OS
they're running on.

## Notes / limitations

- IPv4/IPv6 + TCP/UDP/ICMP only; no VLAN tag or tunnel (GRE/VXLAN)
  unwrapping yet.
- The DNS parser is intentionally minimal (label-walk only, no full
  RR parsing) — sufficient for the tunneling heuristic, not a general
  DNS decoder.
- Thresholds are tuned for illustration; in production, calibrate them
  against your own baseline traffic to control false-positive rate.
- This is a monitoring/detection tool only — it does not block or
  drop traffic.
- The Windows build steps above (Npcap + SDK + MSVC + `$env:LIB`) come
  from `pnet_datalink` itself; this project's code doesn't add any
  extra Windows-specific dependencies.
- **XDP fast path (`--xdp`) is still a monitoring/enforcement tool,
  not a full IPS**: `--xdp-auto-block` only drops by exact source
  IPv4 (no CIDR ranges, no IPv6 yet — `BLOCKLIST_V4` is v4-only).
  Spoofed source addresses can't be meaningfully blocked this way;
  pair auto-block with your own judgment about which detectors you
  trust to drive it.
- **Clock caveat**: `PacketEvent::ts_ns` is `bpf_ktime_get_ns()`
  (`CLOCK_BOOTTIME`-relative), not wall-clock. `src/xdp/mod.rs`
  currently stamps converted `PacketMeta`s with capture-time wall
  clock rather than converting it, since none of the detectors need
  absolute time, only deltas between consecutive packets. If you add
  a detector that cares about true packet arrival time under load,
  revisit this.
- `xdp/nsm-ebpf` and `src/xdp/mod.rs` were written against `aya 0.13`
  /`network-types 0.0.8` from training knowledge, in a sandbox with no
  Rust toolchain available to compile-check them — `aya`'s API has
  shifted across versions before. Run `cargo check` (userspace) and
  `scripts/build-ebpf.sh` (kernel side) first and expect to fix minor
  method-name drift, especially in `network_types::tcp::TcpHdr`'s
  bitfield accessors and `aya::maps::RingBuf`'s async read loop.
- **No in-kernel `debug!()` logging.** An earlier version used
  `aya-log-ebpf`'s `debug!()` macro for one log line on the blocklist
  drop path. That macro backs itself with its own ring buffer map
  (`AYA_LOGS`), and userspace's `EbpfLogger::init()` call discarded
  its returned handle without keeping it alive — which closed that
  map's fd immediately, so the *next* syscall (`BPF_PROG_LOAD`) failed
  with `fd N is not pointing to valid bpf_map`, confirmed via
  `strace -f -e trace=bpf`. Fixing that properly means getting
  `EbpfLogger`'s async read-loop lifetime right; instead, the logger
  integration was removed entirely, since drop visibility already
  exists via `STATS.dropped` and an `ACTION_DROPPED` event on the
  (unrelated, unaffected) `EVENTS` map. If you reintroduce
  `aya-log-ebpf`, keep the `EbpfLogger` alive for the process
  lifetime (spawn a task that polls and flushes it) rather than
  discarding it.

