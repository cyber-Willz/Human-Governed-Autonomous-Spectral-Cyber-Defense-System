//! Types shared between the `nsm-ebpf` kernel program and `nsm`'s
//! userspace XDP loader (`src/xdp/`).
//!
//! Everything here has to be `no_std`, `#[repr(C)]`, and free of
//! pointers/padding: it's the wire format that crosses the
//! kernel/userspace boundary through a BPF ring buffer, so the layout
//! must be identical (and alignment-stable) on both sides.

#![no_std]

use bytemuck::{Pod, Zeroable};

/// Max simultaneous auto-blocked source IPv4 addresses. Sized well
/// above what any of the userspace detectors would plausibly flag at
/// once; raise if you run `--xdp-auto-block` against a very noisy
/// network.
pub const BLOCKLIST_MAX_ENTRIES: u32 = 65_536;

/// Ring buffer capacity in bytes. Must be a power of two. At
/// `size_of::<PacketEvent>() == 64` bytes this holds ~16k in-flight
/// events, which is generous headroom for the userspace consumer to
/// fall behind a burst without the kernel silently dropping events
/// (tracked in `Stats::truncated`, see `nsm-ebpf`).
pub const RINGBUF_BYTE_SIZE: u32 = 1 << 20; // 1 MiB

/// L4 protocol numbers, mirrors `crate::packet::L4Proto` on the
/// userspace side (kept as a plain u8 here since cross-boundary enums
/// with unknown discriminants aren't safely `Pod`).
pub const PROTO_TCP: u8 = 6;
pub const PROTO_UDP: u8 = 17;
pub const PROTO_ICMP: u8 = 1;

/// Why this event was pushed to the ring buffer.
pub const ACTION_PASSED: u8 = 0;
/// The packet matched `BLOCKLIST_V4` and was `XDP_DROP`ped -- this
/// event exists purely so userspace can log/count the drop; the
/// packet itself never continued past the NIC driver.
pub const ACTION_DROPPED: u8 = 1;

/// One packet's worth of metadata, pushed from the XDP program to
/// userspace via a `RingBuf`. Fixed size, no heap data -- the
/// equivalent of `nsm::packet::PacketMeta` but POD, and without the
/// `payload_head` capture (signature matching on the XDP fast path is
/// intentionally out of scope; see README's "Notes / limitations").
///
/// `src_addr`/`dst_addr` hold the address in network byte order:
/// IPv4 in the first 4 bytes (rest zeroed), IPv6 across all 16.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PacketEvent {
    pub ts_ns: u64,
    pub src_addr: [u8; 16],
    pub dst_addr: [u8; 16],
    pub ip_ver: u8,
    pub proto: u8,
    pub tcp_flags: u8,
    pub action: u8,
    pub src_port: u16,
    pub dst_port: u16,
    pub length: u32,
    pub ifindex: u32,
    pub _reserved: [u8; 8],
}

// Compile-time layout sanity check -- if this ever fails, `Pod`'s
// derive would already refuse to compile, but the explicit assert
// gives a much clearer error message than a derive macro failure.
const _: () = assert!(core::mem::size_of::<PacketEvent>() == 64);

impl PacketEvent {
    pub const fn zeroed() -> Self {
        Self {
            ts_ns: 0,
            src_addr: [0; 16],
            dst_addr: [0; 16],
            ip_ver: 0,
            proto: 0,
            tcp_flags: 0,
            action: 0,
            src_port: 0,
            dst_port: 0,
            length: 0,
            ifindex: 0,
            _reserved: [0; 8],
        }
    }
}
