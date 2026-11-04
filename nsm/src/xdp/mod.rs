//! Userspace half of the XDP bypass path (Linux only).
//!
//! This module is a second *producer* for the same
//! `mpsc::Sender<PacketMeta>` that `capture::spawn_capture_thread`
//! (pnet/AF_PACKET) already feeds -- `flow.rs`, `detect::DetectionEngine`,
//! and everything downstream of `main::run_event_loop` are completely
//! unaware of which capture backend is in use. Enabling `--xdp` simply
//! swaps which thing calls `tx.send(meta)`.
//!
//! What it adds beyond capture:
//! - Loads and attaches the compiled `nsm-ebpf` XDP program.
//! - Reads `PacketEvent`s off its `RingBuf` asynchronously and
//!   converts them back into `PacketMeta`.
//! - Exposes `block_ipv4` / `unblock_ipv4`, used by `main.rs` to push
//!   confirmed-bad source IPs into the kernel's `BLOCKLIST_V4` map
//!   when `--xdp-auto-block` is set and a detector escalates to
//!   `Critical`.
//! - Periodically logs `Stats` (passed/dropped/truncated/parse_errors)
//!   from the program's `PerCpuArray`.

use crate::packet::{L4Proto, PacketMeta};
use anyhow::{anyhow, Context, Result};
use aya::maps::{HashMap as AyaHashMap, MapData, PerCpuArray as AyaPerCpuArray, RingBuf};
use aya::programs::{Xdp, XdpFlags};
use aya::Ebpf;
use nsm_common::{PacketEvent, PROTO_ICMP, PROTO_TCP, PROTO_UDP};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use tokio::io::unix::AsyncFd;
use tokio::sync::mpsc::Sender;

/// Matches the `xdp_action` values `nsm-ebpf` can be attached with.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum XdpMode {
    /// Driver-native XDP -- fastest, requires NIC driver support.
    Native,
    /// Generic/SKB mode -- works on any NIC, runs later in the stack
    /// (after `sk_buff` allocation), so less of a "bypass" but a safe
    /// fallback when native mode isn't supported.
    Skb,
    /// Offloaded entirely onto NIC hardware. Rare; only a handful of
    /// SmartNICs support it.
    Hw,
}

impl From<XdpMode> for XdpFlags {
    fn from(m: XdpMode) -> Self {
        match m {
            XdpMode::Native => XdpFlags::DRV_MODE,
            XdpMode::Skb => XdpFlags::SKB_MODE,
            XdpMode::Hw => XdpFlags::HW_MODE,
        }
    }
}

/// Owns the loaded eBPF object for as long as the program should stay
/// attached; dropping it detaches XDP from the interface and tears
/// down its maps.
pub struct XdpCapture {
    // Keeps the underlying `Ebpf` (and therefore the attached `Xdp`
    // link + all maps) alive for the process lifetime.
    _ebpf: Arc<Mutex<Ebpf>>,
    blocklist: Arc<Mutex<AyaHashMap<MapData, u32, u64>>>,
    iface: String,
}

/// Loads `nsm-ebpf` from `obj_path`, attaches it to `iface` in `mode`,
/// and spawns the tasks that stream events into `tx` and periodically
/// log stats. Returns once the program is attached and the reader
/// task is running.
pub async fn start(obj_path: &Path, iface: &str, mode: XdpMode, tx: Sender<PacketMeta>) -> Result<XdpCapture> {
    let bytes = std::fs::read(obj_path)
        .with_context(|| format!("reading eBPF object at {}", obj_path.display()))?;
    let mut ebpf = Ebpf::load(&bytes).context("loading nsm-ebpf object")?;

    let program: &mut Xdp = ebpf
        .program_mut("nsm_xdp")
        .ok_or_else(|| anyhow!("nsm-ebpf object has no 'nsm_xdp' program"))?
        .try_into()
        .context("nsm_xdp is not an XDP program")?;
    program.load().context("loading nsm_xdp into the kernel")?;
    program
        .attach(iface, mode.into())
        .with_context(|| format!("attaching nsm_xdp to interface '{iface}' (try --xdp-mode skb if native mode is unsupported by this NIC's driver)"))?;

    let ringbuf_map = ebpf.take_map("EVENTS").ok_or_else(|| anyhow!("nsm-ebpf object has no 'EVENTS' map"))?;
    let ring = RingBuf::try_from(ringbuf_map).context("EVENTS is not a ring buffer map")?;

    // take_map (not map()+clone()) is required here: `map()` returns a
    // borrowed `&Map`, and `&Map` is trivially `Clone` regardless of
    // whether `Map` itself is -- so `.clone()` on it silently produces
    // another borrow, not an owned value. That yields
    // `HashMap<&MapData, ..>`/`PerCpuArray<&MapData, ..>` instead of
    // the `MapData`-owning versions our structs/functions expect, and
    // fails to type-check. take_map() removes the entry from `ebpf`'s
    // internal map table and hands back an owned `Map`, which is what
    // we actually want since these outlive this function.
    let blocklist_map = ebpf
        .take_map("BLOCKLIST_V4")
        .ok_or_else(|| anyhow!("nsm-ebpf object has no 'BLOCKLIST_V4' map"))?;
    let blocklist: AyaHashMap<_, u32, u64> =
        AyaHashMap::try_from(blocklist_map).context("BLOCKLIST_V4 is not a HashMap")?;

    let stats_map = ebpf
        .take_map("STATS")
        .ok_or_else(|| anyhow!("nsm-ebpf object has no 'STATS' map"))?;
    let stats: AyaPerCpuArray<_, StatsRaw> =
        AyaPerCpuArray::try_from(stats_map).context("STATS is not a PerCpuArray")?;

    tokio::spawn(read_events(ring, tx));
    tokio::spawn(log_stats(stats));

    let blocklist = Arc::new(Mutex::new(blocklist));
    tokio::spawn(sweep_expired_blocklist(blocklist.clone()));

    tracing::info!("XDP program attached to {iface} ({mode:?} mode)");

    Ok(XdpCapture {
        _ebpf: Arc::new(Mutex::new(ebpf)),
        blocklist,
        iface: iface.to_string(),
    })
}

impl XdpCapture {
    /// Inserts (or refreshes) `ip` in the kernel blocklist for `ttl`.
    /// Called from `main.rs` when `--xdp-auto-block` is set and a
    /// `Critical` alert names a source IPv4 address. IPv6 sources
    /// aren't supported by the current `BLOCKLIST_V4` map (see
    /// README limitations).
    pub fn block_ipv4(&self, ip: Ipv4Addr, ttl: Duration) -> Result<()> {
        let expiry_ns = now_ns() + ttl.as_nanos() as u64;
        let key = u32::from_be_bytes(ip.octets());
        self.blocklist
            .lock()
            .unwrap()
            .insert(key, expiry_ns, 0)
            .with_context(|| format!("inserting {ip} into BLOCKLIST_V4"))?;
        tracing::warn!("xdp: blocking {ip} on {} for {:?}", self.iface, ttl);
        Ok(())
    }

    /// Removes `ip` from the blocklist immediately, ahead of its TTL.
    /// Not called anywhere yet -- there's no CLI/admin surface for
    /// manual overrides today -- but kept as public API since
    /// `block_ipv4` without a matching release valve is a footgun the
    /// moment someone wants a "never mind, that was a false positive"
    /// path. Natural next step: a `--xdp-unblock <ip>` one-shot flag,
    /// or a small control socket.
    #[allow(dead_code)]
    pub fn unblock_ipv4(&self, ip: Ipv4Addr) -> Result<()> {
        let key = u32::from_be_bytes(ip.octets());
        // Map may not contain the key (already expired/never inserted);
        // that's not an error condition worth surfacing.
        let _ = self.blocklist.lock().unwrap().remove(&key);
        Ok(())
    }
}

/// `BLOCKLIST_V4` entries are only *checked* for expiry on lookup
/// (see `nsm-ebpf`'s `try_nsm_xdp`) -- an expired entry just stops
/// matching, it doesn't disappear. Without this, a long-running
/// `--xdp-auto-block` deployment would slowly fill the map's fixed
/// `BLOCKLIST_MAX_ENTRIES` capacity with dead entries until inserts
/// start failing. Runs independently of `unblock_ipv4` (that's for
/// deliberate/manual overrides; this is routine garbage collection).
async fn sweep_expired_blocklist(blocklist: Arc<Mutex<AyaHashMap<MapData, u32, u64>>>) {
    let mut ticker = tokio::time::interval(Duration::from_secs(60));
    loop {
        ticker.tick().await;
        let now = now_ns();

        let expired: Vec<u32> = {
            let map = blocklist.lock().unwrap();
            map.iter()
                .filter_map(|entry| entry.ok())
                .filter(|(_, expiry)| *expiry <= now)
                .map(|(key, _)| key)
                .collect()
        };
        if expired.is_empty() {
            continue;
        }

        let mut map = blocklist.lock().unwrap();
        let n = expired.len();
        for key in expired {
            let _ = map.remove(&key);
        }
        tracing::debug!("xdp: swept {n} expired blocklist entries");
    }
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
    // Note: this is wall-clock epoch time, not CLOCK_BOOTTIME like
    // bpf_ktime_get_ns(). We only ever compare *durations* we compute
    // ourselves against a single agreed-upon clock -- see the caveat
    // in the README ("Clock caveat") for why this is fine for TTL
    // purposes but shouldn't be read as an absolute boot-time value.
}

/// Converts an eBPF-side `PacketEvent` back into the same `PacketMeta`
/// the pnet capture path produces, so downstream code never needs to
/// know which backend captured a given packet.
fn event_to_meta(ev: &PacketEvent) -> Option<PacketMeta> {
    let (src_ip, dst_ip) = match ev.ip_ver {
        4 => (
            IpAddr::V4(Ipv4Addr::new(ev.src_addr[0], ev.src_addr[1], ev.src_addr[2], ev.src_addr[3])),
            IpAddr::V4(Ipv4Addr::new(ev.dst_addr[0], ev.dst_addr[1], ev.dst_addr[2], ev.dst_addr[3])),
        ),
        6 => (
            IpAddr::V6(Ipv6Addr::from(ev.src_addr)),
            IpAddr::V6(Ipv6Addr::from(ev.dst_addr)),
        ),
        _ => return None,
    };

    let proto = match ev.proto {
        p if p == PROTO_TCP => L4Proto::Tcp,
        p if p == PROTO_UDP => L4Proto::Udp,
        p if p == PROTO_ICMP => L4Proto::Icmp,
        other => L4Proto::Other(other),
    };

    Some(PacketMeta {
        // ev.ts_ns is CLOCK_BOOTTIME-relative (bpf_ktime_get_ns), not
        // wall-clock -- we don't have a cheap way to convert it
        // in-kernel without an extra helper call, and none of the
        // detectors need absolute time, only deltas. Stamp with
        // capture-time wall clock instead; see README "Clock caveat".
        ts: SystemTime::now(),
        src_ip,
        dst_ip,
        src_port: ev.src_port,
        dst_port: ev.dst_port,
        proto,
        tcp_flags: ev.tcp_flags,
        length: ev.length as usize,
        // The XDP fast path intentionally doesn't capture payload
        // bytes (see nsm-common::PacketEvent docs), so signature.rs's
        // content matching won't fire on XDP-sourced packets. Run
        // without --xdp, or extend PacketEvent with a fixed-size
        // payload window, if you need both.
        payload_head: Vec::new(),
    })
}

async fn read_events(ring: RingBuf<MapData>, tx: Sender<PacketMeta>) {
    let mut poll = match AsyncFd::new(ring) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!("xdp: failed to register ring buffer fd: {e}");
            return;
        }
    };

    loop {
        let mut guard = match poll.readable_mut().await {
            Ok(g) => g,
            Err(e) => {
                tracing::error!("xdp: ring buffer poll error: {e}");
                return;
            }
        };

        let ring = guard.get_inner_mut();
        while let Some(item) = ring.next() {
            if item.len() != std::mem::size_of::<PacketEvent>() {
                tracing::warn!(
                    "xdp: ring buffer event size mismatch (got {}, expected {}) -- nsm-ebpf and nsm-common are out of sync, rebuild both",
                    item.len(),
                    std::mem::size_of::<PacketEvent>()
                );
                continue;
            }
            let ev: PacketEvent = *bytemuck::from_bytes(&item);
            // Drop the ring buffer item (and the borrow of `ring` it
            // holds) before the `.await` below -- holding a borrow of
            // the mmap'd ring buffer across an await point isn't
            // needed here (we've already copied everything we need
            // into the owned `ev`/`meta`) and risks making this
            // task's future non-Send, which tokio::spawn requires.
            drop(item);
            if let Some(meta) = event_to_meta(&ev) {
                // NOT blocking_send(): this closure runs inside the
                // tokio runtime (spawned via tokio::spawn), and
                // blocking_send() is for synchronous/non-async
                // callers only -- calling it here blocks a runtime
                // worker thread and panics ("Cannot block the current
                // thread from within a runtime").
                if tx.send(meta).await.is_err() {
                    tracing::warn!("xdp: analysis channel closed, stopping event reader");
                    return;
                }
            }
        }
        guard.clear_ready();
    }
}

// Must match nsm-ebpf's `Stats` struct field-for-field (repr(C), same
// order) -- there's no shared crate for it since it's an
// implementation detail of nsm-ebpf, not part of the public
// kernel/userspace wire contract in nsm-common.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct StatsRaw {
    passed: u64,
    dropped: u64,
    truncated: u64,
    parse_errors: u64,
}

// aya::maps::PerCpuArray<_, V> requires `V: aya::Pod` -- aya's own
// unsafe marker trait (an `unsafe trait Pod: Copy + 'static {}`
// asserting "safe to reinterpret as raw bytes / zero-initialize"),
// separate from `bytemuck::Pod` used in nsm-common. `StatsRaw` is
// `#[repr(C)]`, made entirely of `u64` fields (no padding, no
// pointers, no niches), and `Copy`, so this is sound.
unsafe impl aya::Pod for StatsRaw {}

async fn log_stats(stats: AyaPerCpuArray<MapData, StatsRaw>) {
    let mut ticker = tokio::time::interval(Duration::from_secs(30));
    loop {
        ticker.tick().await;
        let Ok(per_cpu) = stats.get(&0, 0) else { continue };
        let mut total = StatsRaw::default();
        for v in per_cpu.iter() {
            total.passed += v.passed;
            total.dropped += v.dropped;
            total.truncated += v.truncated;
            total.parse_errors += v.parse_errors;
        }
        tracing::info!(
            "xdp stats: passed={} dropped={} truncated={} parse_errors={}",
            total.passed,
            total.dropped,
            total.truncated,
            total.parse_errors
        );
    }
}
