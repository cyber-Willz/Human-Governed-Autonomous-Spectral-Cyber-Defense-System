use etherparse::{IpHeader, PacketHeaders, TransportHeader};
use std::net::IpAddr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L4Proto {
    Tcp,
    Udp,
    Icmp,
    Other(u8),
}

#[derive(Debug, Clone, Copy)]
pub struct TcpFlags {
    pub syn: bool,
    pub ack: bool,
    pub fin: bool,
    pub rst: bool,
}

#[derive(Debug, Clone)]
pub struct ParsedPacket {
    pub src_ip: IpAddr,
    pub dst_ip: IpAddr,
    pub proto: L4Proto,
    pub src_port: u16,
    pub dst_port: u16,
    pub tcp_flags: Option<TcpFlags>,
    pub payload_len: usize,
}

/// The result of attempting to classify a raw packet. Splitting this into
/// three outcomes instead of a plain `Option` matters operationally: a
/// non-first IP fragment and a genuinely malformed/garbage packet both end
/// up fail-closed dropped either way, but they mean very different things to
/// whoever is looking at the logs. A steady stream of `Fragment` outcomes
/// means "kernel-side defragmentation isn't wired up correctly ahead of the
/// queue redirect" (a config problem, see README) -- a steady stream of
/// `Malformed` means "something is sending genuinely broken or hostile
/// packets at this interface." Collapsing both into one generic
/// "parse_failures" counter, as an earlier version of this code did, makes
/// that distinction invisible.
#[derive(Debug)]
pub enum ParseOutcome {
    Parsed(ParsedPacket),
    /// A non-first IP fragment (RFC 791 for IPv4, RFC 8200 for IPv6): only
    /// the first fragment of a datagram carries the L4 header, so a fragment
    /// with a non-zero offset cannot be classified by port/protocol at all.
    /// This is fundamentally a fail-closed case, not a bug to fix in this
    /// parser -- the correct fix is kernel-side reassembly before the
    /// packet ever reaches NFQUEUE (see the "Fragmentation" section of the
    /// README for the exact `nft`/`ct` wiring).
    UnclassifiableFragment,
    /// Not a fragmentation issue -- truncated below a valid header size,
    /// not IP at all, or otherwise failed to parse.
    Malformed,
}

/// NFQUEUE hands us raw IP packets (no L2 header) since it hooks into netfilter,
/// which operates above the link layer. We parse starting at the IP header.
pub fn parse(raw: &[u8]) -> ParseOutcome {
    // Fragmentation is checked from the raw header fields BEFORE handing off
    // to etherparse's transport-header parser, because a non-first fragment
    // is expected to fail transport parsing (there's no TCP/UDP header in
    // it) -- that expected failure would otherwise be indistinguishable from
    // a genuinely malformed packet if we only looked at whether `transport`
    // came back `None`.
    if let Some(outcome) = check_fragmentation(raw) {
        return outcome;
    }

    let Ok(headers) = PacketHeaders::from_ip_slice(raw) else {
        return ParseOutcome::Malformed;
    };
    let Some(ip_header) = headers.ip else {
        return ParseOutcome::Malformed;
    };

    let (src_ip, dst_ip) = match ip_header {
        IpHeader::Version4(h, _) => (
            IpAddr::V4(h.source.into()),
            IpAddr::V4(h.destination.into()),
        ),
        IpHeader::Version6(h, _) => (
            IpAddr::V6(h.source.into()),
            IpAddr::V6(h.destination.into()),
        ),
    };

    let payload_len = headers.payload.len();

    let (proto, src_port, dst_port, tcp_flags) = match headers.transport {
        Some(TransportHeader::Tcp(t)) => (
            L4Proto::Tcp,
            t.source_port,
            t.destination_port,
            Some(TcpFlags {
                syn: t.syn,
                ack: t.ack,
                fin: t.fin,
                rst: t.rst,
            }),
        ),
        Some(TransportHeader::Udp(u)) => (L4Proto::Udp, u.source_port, u.destination_port, None),
        Some(TransportHeader::Icmpv4(_)) | Some(TransportHeader::Icmpv6(_)) => {
            (L4Proto::Icmp, 0, 0, None)
        }
        None => (L4Proto::Other(0), 0, 0, None),
    };

    ParseOutcome::Parsed(ParsedPacket {
        src_ip,
        dst_ip,
        proto,
        src_port,
        dst_port,
        tcp_flags,
        payload_len,
    })
}

/// Returns `Some(UnclassifiableFragment)` if this is a non-first IP
/// fragment, `None` if it's not (either unfragmented, or the first fragment
/// -- which DOES carry a real L4 header per RFC 791/8200, so it's left to
/// the normal parse path). Returns `None` (not Malformed) on any parse
/// error here -- this function's only job is fragment detection; genuine
/// malformedness is the main parser's job to report, so we don't want two
/// different code paths independently deciding "this is broken."
fn check_fragmentation(raw: &[u8]) -> Option<ParseOutcome> {
    let ip_header = IpHeader::from_slice(raw).ok()?.0;
    match ip_header {
        IpHeader::Version4(h, _) => {
            if h.fragments_offset != 0 {
                Some(ParseOutcome::UnclassifiableFragment)
            } else {
                None
            }
        }
        IpHeader::Version6(_, ext) => {
            if let Some(frag) = ext.fragment {
                if frag.fragment_offset != 0 {
                    return Some(ParseOutcome::UnclassifiableFragment);
                }
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expect_parsed(outcome: ParseOutcome) -> ParsedPacket {
        match outcome {
            ParseOutcome::Parsed(p) => p,
            other => panic!("expected Parsed, got {:?}", other),
        }
    }

    #[test]
    fn empty_input_does_not_panic_and_fails_closed() {
        assert!(matches!(parse(&[]), ParseOutcome::Malformed));
    }

    #[test]
    fn garbage_bytes_fail_closed_rather_than_misclassify() {
        // Random bytes that aren't a valid IP header at all.
        let garbage = [0xffu8; 40];
        assert!(matches!(parse(&garbage), ParseOutcome::Malformed));
    }

    #[test]
    fn truncated_ipv4_header_fails_closed() {
        // A packet claiming IPv4 (version nibble 4) but far too short to
        // actually contain a full IPv4 header -- this is the kind of input
        // a scanner or malformed-packet attack would send hoping the parser
        // either panics or waves it through.
        let truncated = [0x45u8, 0x00, 0x00];
        assert!(matches!(parse(&truncated), ParseOutcome::Malformed));
    }

    #[test]
    fn well_formed_tcp_syn_parses_correctly() {
        // Build a minimal valid IPv4+TCP SYN packet with etherparse's own
        // builder, so this test doesn't depend on a hand-crafted byte dump
        // staying in sync with header layout details.
        let builder = etherparse::PacketBuilder::ipv4([10, 0, 0, 1], [10, 0, 0, 2], 64)
            .tcp(51820, 443, 0, 64)
            .syn();
        let mut raw = Vec::new();
        builder.write(&mut raw, &[1, 2, 3, 4]).unwrap();

        let parsed = expect_parsed(parse(&raw));
        assert_eq!(parsed.proto, L4Proto::Tcp);
        assert_eq!(parsed.src_port, 51820);
        assert_eq!(parsed.dst_port, 443);
        assert_eq!(parsed.payload_len, 4);
        let flags = parsed.tcp_flags.expect("tcp packet should have flags");
        assert!(flags.syn);
        assert!(!flags.ack);
        assert!(!flags.rst);
        assert!(!flags.fin);
    }

    #[test]
    fn well_formed_udp_parses_correctly() {
        let builder = etherparse::PacketBuilder::ipv4([10, 0, 0, 1], [8, 8, 8, 8], 64).udp(54321, 53);
        let mut raw = Vec::new();
        builder.write(&mut raw, &[9, 9]).unwrap();

        let parsed = expect_parsed(parse(&raw));
        assert_eq!(parsed.proto, L4Proto::Udp);
        assert_eq!(parsed.dst_port, 53);
        assert!(parsed.tcp_flags.is_none());
    }

    #[test]
    fn ipv4_first_fragment_still_classifies_normally() {
        // The first fragment (offset 0) of a larger datagram DOES carry the
        // real L4 header per RFC 791 -- only later fragments don't. This
        // must NOT be flagged as UnclassifiableFragment.
        let mut ip = etherparse::Ipv4Header::new(
            8 + 4,
            64,
            etherparse::ip_number::UDP,
            [10, 0, 0, 1],
            [10, 0, 0, 2],
        );
        ip.more_fragments = true; // more fragments will follow
        ip.fragments_offset = 0; // but this IS the first one

        let udp = etherparse::UdpHeader::without_ipv4_checksum(54321, 53, 4).unwrap();

        let mut raw = Vec::new();
        ip.write(&mut raw).unwrap();
        udp.write(&mut raw).unwrap();
        raw.extend_from_slice(&[1, 2, 3, 4]);

        let parsed = expect_parsed(parse(&raw));
        assert_eq!(parsed.proto, L4Proto::Udp);
        assert_eq!(parsed.dst_port, 53);
    }

    #[test]
    fn ipv4_non_first_fragment_is_reported_as_unclassifiable_not_malformed() {
        // fragments_offset != 0 -- a continuation fragment with no L4 header
        // at all, just raw payload bytes. This must be distinguished from
        // "malformed" so operators can tell fragmentation-wiring problems
        // apart from actually hostile/broken traffic.
        let mut ip = etherparse::Ipv4Header::new(
            20,
            64,
            etherparse::ip_number::UDP,
            [10, 0, 0, 1],
            [10, 0, 0, 2],
        );
        ip.more_fragments = false; // last fragment
        ip.fragments_offset = 100; // but NOT the first one

        let mut raw = Vec::new();
        ip.write(&mut raw).unwrap();
        raw.extend_from_slice(&[0u8; 20]); // arbitrary continuation payload

        assert!(matches!(parse(&raw), ParseOutcome::UnclassifiableFragment));
    }

    #[test]
    fn ipv6_non_first_fragment_is_reported_as_unclassifiable_not_malformed() {
        let payload = [0xAAu8; 16];
        let fragment_header = etherparse::Ipv6FragmentHeader::new(
            etherparse::ip_number::UDP,
            50, // non-zero offset -- a continuation fragment
            false,
            0xdead_beef,
        );

        let ip = etherparse::Ipv6Header {
            traffic_class: 0,
            flow_label: 0,
            payload_length: (fragment_header.header_len() + payload.len()) as u16,
            next_header: etherparse::ip_number::IPV6_FRAG,
            hop_limit: 64,
            source: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            destination: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
        };

        let mut raw = Vec::new();
        ip.write(&mut raw).unwrap();
        fragment_header.write(&mut raw).unwrap();
        raw.extend_from_slice(&payload);

        assert!(matches!(parse(&raw), ParseOutcome::UnclassifiableFragment));
    }

    #[test]
    fn ipv6_hop_by_hop_extension_header_is_walked_to_find_real_tcp_header() {
        // Confirms a claim this project used to list as an open gap ("no
        // IPv6-specific extension header walking beyond what etherparse
        // gives you out of the box") is actually already handled: a Hop-by-
        // Hop Options header sitting between the IPv6 header and the real
        // TCP header must not cause misclassification -- the packet should
        // still resolve to L4Proto::Tcp with the correct ports, not fall
        // through to Other(0)/unclassified the way a naive "only look at
        // the IPv6 next_header field" implementation would.
        let tcp = etherparse::TcpHeader::new(51820, 443, 0, 64);
        let mut tcp_bytes = Vec::new();
        tcp.write(&mut tcp_bytes).unwrap();

        // A minimal valid Hop-by-Hop Options header: next_header = TCP,
        // followed by 6 bytes of options padding (extension headers are
        // sized in 8-octet units; 2 bytes of fixed header + 6 bytes payload
        // = 8 bytes total, hdr_ext_len = 0 meaning "8 bytes total").
        let hop_by_hop =
            etherparse::Ipv6RawExtensionHeader::new_raw(etherparse::ip_number::TCP, &[0u8; 6])
                .unwrap();

        let mut hop_by_hop_bytes = Vec::new();
        hop_by_hop.write(&mut hop_by_hop_bytes).unwrap();

        let ip = etherparse::Ipv6Header {
            traffic_class: 0,
            flow_label: 0,
            payload_length: (hop_by_hop_bytes.len() + tcp_bytes.len()) as u16,
            next_header: etherparse::ip_number::IPV6_HOP_BY_HOP,
            hop_limit: 64,
            source: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            destination: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
        };

        let mut raw = Vec::new();
        ip.write(&mut raw).unwrap();
        raw.extend_from_slice(&hop_by_hop_bytes);
        raw.extend_from_slice(&tcp_bytes);

        let parsed = expect_parsed(parse(&raw));
        assert_eq!(parsed.proto, L4Proto::Tcp);
        assert_eq!(parsed.src_port, 51820);
        assert_eq!(parsed.dst_port, 443);
    }
}
