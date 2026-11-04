//! Memory-bounded state helpers shared by every detector.
//!
//! Every detector keeps per-source / per-pair state in a concurrent map. An
//! attacker who invents source addresses (trivial for spoofed SYN/UDP) can
//! otherwise grow those maps without limit (measured: ~1.4 KB per forged
//! source in the original code). `make_room` enforces a hard cap with an
//! eviction order that protects the entries an attacker would most like to
//! destroy -- the ones that have *accumulated evidence* -- so forging a flood
//! of fresh sources cannot be used to make the sensor forget a slow scanner.

use dashmap::DashMap;
use std::hash::Hash;
use std::net::IpAddr;

/// Default per-detector entry cap. Sized so the four detectors together stay
/// within a few tens of MB under a forged-source flood.
pub const DEFAULT_CAP: usize = 50_000;

/// Ensure `map` has room for one more entry (`len < cap`).
///
/// Eviction order, stopping as soon as there is room:
///   1. entries `stale` (their window has fully expired),
///   2. entries `weak` (a single observation: cheap for an attacker to forge,
///      worthless to keep),
///   3. last resort: drop ~10 % of the table arbitrarily (amortised O(1)).
pub fn make_room<K, V>(map: &DashMap<K, V>, cap: usize, stale: impl Fn(&V) -> bool, weak: impl Fn(&V) -> bool)
where
    K: Eq + Hash,
{
    if map.len() < cap {
        return;
    }
    map.retain(|_, v| !stale(v));
    if map.len() < cap {
        return;
    }
    map.retain(|_, v| !weak(v));
    if map.len() < cap {
        return;
    }
    let mut to_drop = cap / 10 + 1;
    map.retain(|_, _| {
        if to_drop > 0 {
            to_drop -= 1;
            false
        } else {
            true
        }
    });
}

/// Cheap, stable 64-bit hash (FNV-1a) used to keep "distinct X" sets as small
/// `u64`s instead of owning the full value.
pub fn fnv64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Canonical *actor* key for a source address.
///
/// Round-2 red-team findings: an attacker who controls an IPv6 /64 owns 2^64
/// addresses and can use a fresh one for every packet, so any per-address
/// counter (scan thresholds, beacon history, DNS cardinality) never
/// accumulates; and the same IPv4 host has two spellings (`1.2.3.4` and
/// `::ffff:1.2.3.4`) that dual-stack sockets report interchangeably.
///
/// * IPv4 is returned unchanged.
/// * IPv4-mapped IPv6 collapses to the plain IPv4 address.
/// * Any other IPv6 address collapses to its /64 (low 64 bits zeroed): the
///   smallest unit an attacker can rotate inside for free, and the unit
///   operators actually allocate to one subscriber/site.
///
/// `correlation_engine` and `rustwall::quarantine::ban_key` apply the same
/// rule so a detector key, a correlation record and a ban all agree about
/// who the actor is (parity is asserted by a test in `ips_control_plane`).
pub fn canon_src(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V4(_) => ip,
        IpAddr::V6(a) => {
            if let Some(v4) = a.to_ipv4_mapped() {
                return IpAddr::V4(v4);
            }
            let mut o = a.octets();
            for b in &mut o[8..] {
                *b = 0;
            }
            IpAddr::V6(o.into())
        }
    }
}

/// Stable hash of a canonical source, for "distinct sources" sets.
pub fn src_hash(ip: IpAddr) -> u64 {
    match canon_src(ip) {
        IpAddr::V4(a) => fnv64(&a.octets()),
        IpAddr::V6(a) => fnv64(&a.octets()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, Ipv6Addr};

    #[test]
    fn v4_is_unchanged_and_mapped_v6_collapses_to_v4() {
        let v4 = IpAddr::V4(Ipv4Addr::new(198, 51, 100, 66));
        assert_eq!(canon_src(v4), v4);
        let mapped = IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc633, 0x6442));
        assert_eq!(canon_src(mapped), v4);
    }

    #[test]
    fn addresses_in_one_slash64_share_a_key_and_other_prefixes_do_not() {
        let a: IpAddr = "2001:db8:bad:1::1".parse().unwrap();
        let b: IpAddr = "2001:db8:bad:1:aaaa:bbbb:cccc:dddd".parse().unwrap();
        let c: IpAddr = "2001:db8:bad:2::1".parse().unwrap();
        assert_eq!(canon_src(a), canon_src(b));
        assert_ne!(canon_src(a), canon_src(c));
        assert_eq!(src_hash(a), src_hash(b));
    }

    #[test]
    fn canon_is_idempotent() {
        let a: IpAddr = "2001:db8:bad:1:aaaa:bbbb:cccc:dddd".parse().unwrap();
        assert_eq!(canon_src(canon_src(a)), canon_src(a));
    }
}
