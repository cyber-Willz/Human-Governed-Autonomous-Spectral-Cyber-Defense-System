//! `nsm`'s XDP fast path.
//!
//! Runs at the earliest hook available on the RX path -- inside the
//! NIC driver in native mode, or immediately after the driver in
//! generic/SKB mode -- and does two things per packet, entirely in
//! kernel context:
//!
//! 1. **Bypass capture**: parses Ethernet/IPv4/IPv6/TCP/UDP headers
//!    directly out of the packet buffer and pushes a fixed-size
//!    [`PacketEvent`] into a `RingBuf`. This replaces `pnet`'s
//!    `AF_PACKET` socket capture on the fast path -- no per-packet
//!    syscall, no ring-to-userspace frame copy, no libpcap BPF
//!    interpreter.
//! 2. **Bypass enforcement**: checks the source address against
//!    `BLOCKLIST_V4`, an eBPF map userspace populates when a detector
//!    (port scan / SYN flood) escalates to `Critical`. A hit is
//!    `XDP_DROP`ped here -- before `sk_buff` allocation, before the
//!    kernel network stack, before routing. That's the actual
//!    "bypass": confirmed-bad traffic never gets far enough to cost
//!    CPU anywhere else.
//!
//! Everything else -- flow tracking, DNS tunneling, beaconing,
//! signature matching -- still happens in userspace exactly as
//! before, fed by [`PacketEvent`]s converted back into the existing
//! `PacketMeta` (see `src/xdp/mod.rs` in the main crate). Only the
//! capture + first-line-of-defense layer moved into the kernel.
//!
//! Build with `scripts/build-ebpf.sh` (needs a nightly toolchain with
//! `rust-src`, and `bpf-linker`); see the README.
//!
//! Deliberately does NOT use `aya-log-ebpf`/`debug!()`: that macro
//! creates its own backing ring buffer map (`AYA_LOGS`), and an
//! earlier version of this program hit a real bug where userspace
//! (`src/xdp/mod.rs`) discarded the `EbpfLogger` handle returned by
//! `EbpfLogger::init()` without keeping it alive -- which closed that
//! map's fd immediately, and `BPF_PROG_LOAD` then failed with
//! "fd N is not pointing to valid bpf_map" (confirmed via `strace -e
//! trace=bpf`: the map was created successfully, then closed before
//! the program referencing it got loaded). Rather than get the
//! logger's async read-loop lifetime right, drop visibility goes
//! through `STATS.dropped` and an `ACTION_DROPPED` event on `EVENTS`
//! instead -- both already required for the ordinary capture path, so
//! this removes a dependency instead of adding a fix.

#![no_std]
#![no_main]

use aya_ebpf::{
    bindings::xdp_action,
    macros::{map, xdp},
    maps::{HashMap, PerCpuArray, RingBuf},
    programs::XdpContext,
};
use core::mem;
use network_types::{
    eth::{EthHdr, EtherType},
    ip::{IpProto, Ipv4Hdr, Ipv6Hdr},
    tcp::TcpHdr,
    udp::UdpHdr,
};
use nsm_common::{PacketEvent, ACTION_DROPPED, ACTION_PASSED, BLOCKLIST_MAX_ENTRIES, RINGBUF_BYTE_SIZE};

/// Source IPv4 -> `bpf_ktime_get_ns()` expiry timestamp. Userspace
/// inserts/removes entries via `aya::maps::HashMap` from Rust; see
/// `XdpCapture::block_ipv4` / `unblock_ipv4`.
#[map]
static BLOCKLIST_V4: HashMap<u32, u64> = HashMap::with_max_entries(BLOCKLIST_MAX_ENTRIES, 0);

/// Per-packet telemetry consumed by the userspace async reader.
#[map]
static EVENTS: RingBuf = RingBuf::with_byte_size(RINGBUF_BYTE_SIZE, 0);

#[repr(C)]
#[derive(Clone, Copy)]
struct Stats {
    passed: u64,
    dropped: u64,
    /// Ring buffer was full; the event was dropped, but the packet
    /// itself was still allowed through (we never fail closed on
    /// backpressure -- see `try_nsm_xdp`'s `EVENTS.reserve` handling).
    truncated: u64,
    /// Header didn't parse (truncated frame, unsupported ethertype,
    /// etc.) -- always `XDP_PASS`ed untouched.
    parse_errors: u64,
}

#[map]
static STATS: PerCpuArray<Stats> = PerCpuArray::with_max_entries(1, 0);

#[xdp]
pub fn nsm_xdp(ctx: XdpContext) -> u32 {
    match try_nsm_xdp(ctx) {
        Ok(action) => action,
        Err(_) => {
            bump(|s| s.parse_errors += 1);
            xdp_action::XDP_PASS
        }
    }
}

#[inline(always)]
fn ptr_at<T>(ctx: &XdpContext, offset: usize) -> Result<*const T, ()> {
    let start = ctx.data();
    let end = ctx.data_end();
    let len = mem::size_of::<T>();
    if start + offset + len > end {
        return Err(());
    }
    Ok((start + offset) as *const T)
}

#[inline(always)]
fn bump(f: impl FnOnce(&mut Stats)) {
    if let Some(s) = STATS.get_ptr_mut(0) {
        // Safety: PerCpuArray gives us an exclusive per-CPU slot; no
        // other program instance touches this pointer concurrently.
        unsafe { f(&mut *s) };
    }
}

/// Pushes `event` to the ring buffer. Returns `false` if the ring was
/// full (caller bumps `truncated` rather than `passed`/`dropped`) --
/// note this only means the *event* was dropped, not the packet
/// itself (see the two call sites: XDP_PASS/XDP_DROP is decided
/// independently of whether we could log it).
#[inline(always)]
fn submit_event(event: PacketEvent) -> bool {
    match EVENTS.reserve::<PacketEvent>(0) {
        Some(mut entry) => {
            entry.write(event);
            entry.submit(0);
            true
        }
        None => false,
    }
}

fn try_nsm_xdp(ctx: XdpContext) -> Result<u32, ()> {
    let eth = ptr_at::<EthHdr>(&ctx, 0)?;
    // Safety: ptr_at() bounds-checked `EthHdr::LEN` bytes above.
    let ether_type = unsafe { (*eth).ether_type };

    let mut event = PacketEvent::zeroed();
    event.ts_ns = unsafe { aya_ebpf::helpers::bpf_ktime_get_ns() };
    event.ifindex = unsafe { (*ctx.ctx).ingress_ifindex };
    event.length = (ctx.data_end() - ctx.data()) as u32;

    let (l4_off, l4_proto, src4): (usize, u8, Option<u32>) = match ether_type {
        EtherType::Ipv4 => {
            let ip = ptr_at::<Ipv4Hdr>(&ctx, EthHdr::LEN)?;
            // Safety: bounds-checked by ptr_at above.
            let ip = unsafe { &*ip };
            let ihl = (ip.ihl() as usize) * 4;
            if ihl < core::mem::size_of::<Ipv4Hdr>() {
                return Err(()); // malformed IHL, bail out (pass untouched)
            }

            event.ip_ver = 4;
            event.src_addr[..4].copy_from_slice(&ip.src_addr);
            event.dst_addr[..4].copy_from_slice(&ip.dst_addr);

            let src = u32::from_be_bytes(ip.src_addr);
            (EthHdr::LEN + ihl, ip.proto as u8, Some(src))
        }
        EtherType::Ipv6 => {
            let ip = ptr_at::<Ipv6Hdr>(&ctx, EthHdr::LEN)?;
            // Safety: bounds-checked by ptr_at above.
            let ip = unsafe { &*ip };

            event.ip_ver = 6;
            event.src_addr.copy_from_slice(&ip.src_addr);
            event.dst_addr.copy_from_slice(&ip.dst_addr);

            (EthHdr::LEN + Ipv6Hdr::LEN, ip.next_hdr as u8, None)
        }
        // Non-IP traffic (ARP, etc.): not our concern, hand straight back.
        _ => return Ok(xdp_action::XDP_PASS),
    };

    // --- Enforcement: kernel-level drop, before anything else runs. ---
    if let Some(src) = src4 {
        // Safety: BLOCKLIST_V4 is a HashMap<u32, u64> keyed by a
        // value we compute ourselves (`src`, from bytes ptr_at()
        // already bounds-checked) -- get() is unsafe in aya-ebpf 0.2.x
        // because the returned reference's lifetime/validity is tied
        // to the verifier-enforced access window, not because the key
        // itself is unchecked.
        if let Some(&expiry) = unsafe { BLOCKLIST_V4.get(&src) } {
            let now = unsafe { aya_ebpf::helpers::bpf_ktime_get_ns() };
            if now < expiry {
                // (No in-kernel debug!() log here -- see nsm-ebpf's
                // module docs / README for why aya-log-ebpf was
                // removed. This drop is still fully visible to
                // userspace: it's counted in STATS.dropped and
                // reported via an ACTION_DROPPED event on EVENTS,
                // below.)
                event.proto = l4_proto;
                event.action = ACTION_DROPPED;
                if submit_event(event) {
                    bump(|s| s.dropped += 1);
                } else {
                    bump(|s| s.truncated += 1);
                }
                return Ok(xdp_action::XDP_DROP);
            }
        }
    }

    event.proto = l4_proto;
    match l4_proto {
        p if p == IpProto::Tcp as u8 => {
            if let Ok(tcp) = ptr_at::<TcpHdr>(&ctx, l4_off) {
                // Safety: bounds-checked by ptr_at above.
                let tcp = unsafe { &*tcp };
                // Unlike UdpHdr below, network-types' TcpHdr stores
                // source/dest as a native u16 (still network byte
                // order), not [u8; 2] -- so from_be here, not
                // from_be_bytes. The two structs aren't consistent
                // with each other in this crate; don't assume one
                // implies the other.
                event.src_port = u16::from_be(tcp.source);
                event.dst_port = u16::from_be(tcp.dest);
                event.tcp_flags = tcp_flags_byte(tcp);
            }
        }
        p if p == IpProto::Udp as u8 => {
            if let Ok(udp) = ptr_at::<UdpHdr>(&ctx, l4_off) {
                // Safety: bounds-checked by ptr_at above.
                let udp = unsafe { &*udp };
                event.src_port = u16::from_be_bytes(udp.source);
                event.dst_port = u16::from_be_bytes(udp.dest);
            }
        }
        _ => {} // ICMP / other: address + proto only, matches userspace parser.
    }

    event.action = ACTION_PASSED;
    if submit_event(event) {
        bump(|s| s.passed += 1);
    } else {
        bump(|s| s.truncated += 1); // ring full -- packet still passes, just unlogged
    }

    Ok(xdp_action::XDP_PASS)
}

/// `network-types`' `TcpHdr` exposes flags as individual bitfield
/// accessors; repack them into the single-byte layout
/// `nsm::packet::PacketMeta::tcp_flags` already uses (bit0=FIN,
/// bit1=SYN, bit2=RST, bit3=PSH, bit4=ACK, bit5=URG -- same order
/// `pnet`'s `TcpFlags` uses) so `event_to_meta()` on the userspace
/// side needs no special-casing per capture backend.
#[inline(always)]
fn tcp_flags_byte(tcp: &TcpHdr) -> u8 {
    (tcp.fin() as u8)
        | ((tcp.syn() as u8) << 1)
        | ((tcp.rst() as u8) << 2)
        | ((tcp.psh() as u8) << 3)
        | ((tcp.ack() as u8) << 4)
        | ((tcp.urg() as u8) << 5)
}

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe { core::hint::unreachable_unchecked() }
}

#[link_section = "license"]
#[no_mangle]
static LICENSE: [u8; 13] = *b"Dual MIT/GPL\0";
