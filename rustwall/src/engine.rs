use crate::config::{Action, Config, Direction, NetworkMatch, Protocol, Rule};
use crate::packet::{L4Proto, ParsedPacket};
use crate::rate_limit::RateLimiter;
use crate::threshold::ThresholdTracker;
use ipnet::IpNet;
use std::collections::HashMap;
use std::net::IpAddr;

pub struct Engine {
    rules: Vec<Rule>,
    /// Resolved alias table: name -> CIDR list, built once at construction so
    /// per-packet matching never has to do a name lookup through the raw
    /// config -- see config::Alias / config::NetworkMatch. This is the
    /// pfSense/OPNsense "Alias" pattern: rules reference a name, the name
    /// resolves to a CIDR set, and updating the set means editing one place
    /// instead of every rule that used the literal CIDR.
    aliases: HashMap<String, Vec<IpNet>>,
    /// Index: (protocol-as-u8, exact dst_port) -> sorted rule indices, for
    /// rules with a single concrete destination port. Real rulesets skew
    /// heavily toward exact-port rules ("allow tcp/443", "block tcp/23"), so
    /// hashing on the common case turns evaluation from O(rules) into
    /// O(matching rules) instead of scanning every entry on every new flow --
    /// this is the same reason ASIC-backed NGFWs precompile policy into
    /// lookup tables rather than walking a rule list per packet.
    exact_port_index: HashMap<(u8, u16), Vec<usize>>,
    /// Rules that can't be exactly indexed (port ranges, wildcard "*", or
    /// protocol "any") -- always scanned, but this list is typically small
    /// relative to the exact-match set in a well-written ruleset.
    wildcard_rules: Vec<usize>,
    default_policy: Action,
    trusted: Vec<IpAddr>,
    rate_limiter: RateLimiter,
    /// Tracks matches-per-source toward each rule's `auto_block_threshold`
    /// -- see threshold.rs for why single-match triggering is a real
    /// spoofing weakness this closes the trivial case of.
    threshold_tracker: ThresholdTracker,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Accept,
    Drop,
    /// Distinct from Drop: caller should send a TCP RST or ICMP
    /// port-unreachable back to the sender instead of silently discarding.
    Reject,
}

pub struct Decision {
    pub verdict: Verdict,
    pub rule_name: String,
    pub log: bool,
    /// If set, the caller should place the packet's source IP into dynamic
    /// quarantine for this many seconds. Propagated from the matched rule's
    /// `auto_block_secs`, or None for default-policy / non-matching packets.
    pub auto_block_secs: Option<u64>,
}

fn proto_num(proto: Protocol) -> Option<u8> {
    match proto {
        Protocol::Tcp => Some(6),
        Protocol::Udp => Some(17),
        Protocol::Icmp => Some(1),
        Protocol::Any => None,
    }
}

fn pkt_proto_num(proto: L4Proto) -> u8 {
    match proto {
        L4Proto::Tcp => 6,
        L4Proto::Udp => 17,
        L4Proto::Icmp => 1,
        L4Proto::Other(n) => n,
    }
}

fn proto_matches(rule_proto: Protocol, pkt_proto: L4Proto) -> bool {
    match (rule_proto, pkt_proto) {
        (Protocol::Any, _) => true,
        (Protocol::Tcp, L4Proto::Tcp) => true,
        (Protocol::Udp, L4Proto::Udp) => true,
        (Protocol::Icmp, L4Proto::Icmp) => true,
        _ => false,
    }
}

fn direction_matches(_rule_dir: Direction, _pkt: &ParsedPacket) -> bool {
    // Direction is resolved by the caller (which queue/hook the packet
    // arrived on), not derivable from packet bytes alone.
    true
}

impl Engine {
    pub fn new(cfg: &Config) -> Self {
        let rules = cfg.rules.clone();
        let aliases: HashMap<String, Vec<IpNet>> = cfg
            .aliases
            .iter()
            .map(|a| (a.name.clone(), a.cidrs.clone()))
            .collect();

        let mut exact_port_index: HashMap<(u8, u16), Vec<usize>> = HashMap::new();
        let mut wildcard_rules = Vec::new();

        for (idx, rule) in rules.iter().enumerate() {
            let is_exact_port = rule.dst_port.0 == rule.dst_port.1;
            match (proto_num(rule.protocol), is_exact_port) {
                (Some(proto), true) => {
                    exact_port_index
                        .entry((proto, rule.dst_port.0))
                        .or_default()
                        .push(idx);
                }
                _ => wildcard_rules.push(idx),
            }
        }

        Self {
            rules,
            aliases,
            exact_port_index,
            wildcard_rules,
            default_policy: cfg.default_policy,
            trusted: cfg.trusted.clone(),
            rate_limiter: RateLimiter::new(),
            threshold_tracker: ThresholdTracker::new(),
        }
    }

    /// Resolves a rule's src/dst (literal CIDR or alias reference) against
    /// an IP. Unknown alias names can't reach here -- Config::validate()
    /// rejects them at load time -- so an alias that resolves to nothing is
    /// treated as "matches nothing" rather than panicking.
    fn network_matches(&self, m: &NetworkMatch, ip: IpAddr) -> bool {
        match m {
            NetworkMatch::Cidr(net) => net.contains(&ip),
            NetworkMatch::Alias(name) => self
                .aliases
                .get(name)
                .map(|cidrs| cidrs.iter().any(|net| net.contains(&ip)))
                .unwrap_or(false),
        }
    }

    fn rule_matches(&self, rule: &Rule, pkt: &ParsedPacket) -> bool {
        proto_matches(rule.protocol, pkt.proto)
            && self.network_matches(&rule.src, pkt.src_ip)
            && self.network_matches(&rule.dst, pkt.dst_ip)
            && rule.src_port.contains(pkt.src_port)
            && rule.dst_port.contains(pkt.dst_port)
            && direction_matches(rule.direction, pkt)
    }

    /// Decides whether this match should actually trigger `auto_block_secs`
    /// right now, given the rule's configured threshold/window. Returns
    /// `None` if the rule has no auto_block_secs configured, or if it does
    /// but the source hasn't reached the required match count within the
    /// window yet -- in which case the packet is still dropped/rejected as
    /// normal, it just doesn't (yet) trigger a quarantine ban.
    fn gate_auto_block(&self, rule_idx: usize, rule: &Rule, src_ip: IpAddr) -> Option<u64> {
        let secs = rule.auto_block_secs?;
        let reached = self.threshold_tracker.record_and_check(
            rule_idx,
            src_ip,
            rule.auto_block_threshold,
            std::time::Duration::from_secs(rule.auto_block_window_secs),
        );
        if reached {
            Some(secs)
        } else {
            None
        }
    }

    /// Evaluate a newly-seen flow's first packet. Merges the exact-port index
    /// bucket with the wildcard scan list, preserving original rule order so
    /// first-match-wins semantics are identical to a naive linear scan --
    /// indexing is purely a performance optimization, never a behavior change.
    ///
    /// Note: this does NOT check quarantine -- that's a separate, faster
    /// check the caller (nfqueue.rs) performs before even reaching here, on
    /// every packet regardless of whether it's a new flow, since quarantine
    /// is meant to be an immediate kill switch independent of rule state.
    pub fn evaluate(&self, pkt: &ParsedPacket) -> Decision {
        if self.trusted.contains(&pkt.src_ip) {
            return Decision {
                verdict: Verdict::Accept,
                rule_name: "trusted-bypass".to_string(),
                log: false,
                auto_block_secs: None,
            };
        }

        let proto = pkt_proto_num(pkt.proto);
        let mut candidates: Vec<usize> = self
            .exact_port_index
            .get(&(proto, pkt.dst_port))
            .cloned()
            .unwrap_or_default();
        candidates.extend_from_slice(&self.wildcard_rules);
        candidates.sort_unstable();

        for idx in candidates {
            let rule = &self.rules[idx];
            if !self.rule_matches(rule, pkt) {
                continue;
            }

            if let Some(rate) = rule.rate_limit_pps {
                if !self.rate_limiter.allow(pkt.src_ip, idx as u32, rate) {
                    return Decision {
                        verdict: Verdict::Drop,
                        rule_name: format!("{}:rate-limit-exceeded", rule.name),
                        log: true,
                        auto_block_secs: self.gate_auto_block(idx, rule, pkt.src_ip),
                    };
                }
            }

            let verdict = match rule.action {
                Action::Accept => Verdict::Accept,
                Action::Drop => Verdict::Drop,
                Action::Reject => Verdict::Reject,
            };
            return Decision {
                verdict,
                rule_name: rule.name.clone(),
                log: rule.log,
                // Only propagate auto-block on a non-accept verdict --
                // quarantining a source because it successfully matched an
                // *allow* rule would be backwards. Config-level intent is
                // "block and remember", not "allow and remember". Also
                // gated through the threshold tracker: a single match
                // doesn't fire a ban unless auto_block_threshold is 1
                // (the default, preserving original behavior) -- see
                // gate_auto_block / threshold.rs.
                auto_block_secs: if verdict == Verdict::Accept {
                    None
                } else {
                    self.gate_auto_block(idx, rule, pkt.src_ip)
                },
            };
        }

        Decision {
            verdict: match self.default_policy {
                Action::Accept => Verdict::Accept,
                Action::Reject => Verdict::Reject,
                Action::Drop => Verdict::Drop,
            },
            rule_name: "default-policy".to_string(),
            log: true,
            auto_block_secs: None,
        }
    }

    pub fn periodic_maintenance(&self) {
        self.rate_limiter
            .sweep(std::time::Duration::from_secs(300));
        self.threshold_tracker
            .sweep(std::time::Duration::from_secs(300));
    }

    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    pub fn is_trusted(&self, ip: IpAddr) -> bool {
        self.trusted.contains(&ip)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        Action as CfgAction, Alias, ConntrackConfig, PortRange, Protocol as CfgProto,
    };
    use crate::packet::TcpFlags;
    use std::net::Ipv4Addr;

    fn tcp_pkt(src: &str, dst: &str, sport: u16, dport: u16, syn: bool) -> ParsedPacket {
        ParsedPacket {
            src_ip: src.parse().unwrap(),
            dst_ip: dst.parse().unwrap(),
            proto: L4Proto::Tcp,
            src_port: sport,
            dst_port: dport,
            tcp_flags: Some(TcpFlags {
                syn,
                ack: false,
                fin: false,
                rst: false,
            }),
            payload_len: 0,
        }
    }

    fn rule(name: &str, proto: CfgProto, dst_port: (u16, u16), action: CfgAction) -> Rule {
        Rule {
            name: name.to_string(),
            protocol: proto,
            direction: Direction::Any,
            src: NetworkMatch::Cidr("0.0.0.0/0".parse().unwrap()),
            dst: NetworkMatch::Cidr("0.0.0.0/0".parse().unwrap()),
            src_port: PortRange(0, 65535),
            dst_port: PortRange(dst_port.0, dst_port.1),
            action,
            rate_limit_pps: None,
            auto_block_secs: None,
            auto_block_threshold: 1,
            auto_block_window_secs: 60,
            log: false,
        }
    }

    fn cfg_with_rules(rules: Vec<Rule>, default: CfgAction) -> Config {
        Config {
            queue_num: 0,
            queue_workers: 1,
            default_policy: default,
            conntrack: ConntrackConfig::default(),
            rules,
            aliases: vec![],
            trusted: vec![],
            log_max_per_sec: 200,
            metrics_listen: None,
            metrics_auth_token: None,
            quarantine_max_entries: 100_000,
            sync_to_os_firewall: false,
        }
    }

    #[test]
    fn default_deny_when_no_rule_matches() {
        let cfg = cfg_with_rules(vec![], CfgAction::Drop);
        let engine = Engine::new(&cfg);
        let pkt = tcp_pkt("10.0.0.1", "10.0.0.2", 12345, 80, true);
        let d = engine.evaluate(&pkt);
        assert_eq!(d.verdict, Verdict::Drop);
        assert_eq!(d.rule_name, "default-policy");
    }

    #[test]
    fn exact_port_rule_matches_via_index() {
        let cfg = cfg_with_rules(
            vec![rule("allow-https", CfgProto::Tcp, (443, 443), CfgAction::Accept)],
            CfgAction::Drop,
        );
        let engine = Engine::new(&cfg);
        let pkt = tcp_pkt("10.0.0.1", "10.0.0.2", 12345, 443, true);
        let d = engine.evaluate(&pkt);
        assert_eq!(d.verdict, Verdict::Accept);
        assert_eq!(d.rule_name, "allow-https");
    }

    #[test]
    fn first_match_wins_regardless_of_index_bucket() {
        let cfg = cfg_with_rules(
            vec![
                rule("block-all-tcp", CfgProto::Tcp, (0, 65535), CfgAction::Drop),
                rule("allow-https", CfgProto::Tcp, (443, 443), CfgAction::Accept),
            ],
            CfgAction::Drop,
        );
        let engine = Engine::new(&cfg);
        let pkt = tcp_pkt("10.0.0.1", "10.0.0.2", 12345, 443, true);
        let d = engine.evaluate(&pkt);
        assert_eq!(d.verdict, Verdict::Drop);
        assert_eq!(d.rule_name, "block-all-tcp");
    }

    #[test]
    fn trusted_source_bypasses_all_rules() {
        let mut cfg = cfg_with_rules(
            vec![rule("block-all", CfgProto::Tcp, (0, 65535), CfgAction::Drop)],
            CfgAction::Drop,
        );
        cfg.trusted = vec![IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1))];
        let engine = Engine::new(&cfg);
        let pkt = tcp_pkt("10.0.0.1", "10.0.0.2", 12345, 22, true);
        let d = engine.evaluate(&pkt);
        assert_eq!(d.verdict, Verdict::Accept);
        assert_eq!(d.rule_name, "trusted-bypass");
    }

    #[test]
    fn reject_action_produces_reject_verdict_not_drop() {
        let cfg = cfg_with_rules(
            vec![rule("reject-telnet", CfgProto::Tcp, (23, 23), CfgAction::Reject)],
            CfgAction::Drop,
        );
        let engine = Engine::new(&cfg);
        let pkt = tcp_pkt("10.0.0.1", "10.0.0.2", 12345, 23, true);
        let d = engine.evaluate(&pkt);
        assert_eq!(d.verdict, Verdict::Reject);
    }

    #[test]
    fn rate_limit_drops_excess_packets_from_same_source() {
        let cfg = cfg_with_rules(
            vec![Rule {
                rate_limit_pps: Some(1),
                ..rule("allow-throttled", CfgProto::Tcp, (80, 80), CfgAction::Accept)
            }],
            CfgAction::Drop,
        );
        let engine = Engine::new(&cfg);
        let pkt = tcp_pkt("10.0.0.1", "10.0.0.2", 12345, 80, true);
        let first = engine.evaluate(&pkt);
        assert_eq!(first.verdict, Verdict::Accept);
        let second = engine.evaluate(&pkt);
        assert_eq!(second.verdict, Verdict::Drop);
        assert!(second.rule_name.contains("rate-limit-exceeded"));
    }

    #[test]
    fn alias_reference_resolves_and_matches() {
        let mut cfg = cfg_with_rules(
            vec![Rule {
                src: NetworkMatch::Alias("office".to_string()),
                ..rule("allow-office-ssh", CfgProto::Tcp, (22, 22), CfgAction::Accept)
            }],
            CfgAction::Drop,
        );
        cfg.aliases = vec![Alias {
            name: "office".to_string(),
            cidrs: vec!["10.0.0.0/24".parse().unwrap()],
        }];
        let engine = Engine::new(&cfg);

        let inside = tcp_pkt("10.0.0.42", "10.0.0.2", 12345, 22, true);
        assert_eq!(engine.evaluate(&inside).verdict, Verdict::Accept);

        let outside = tcp_pkt("203.0.113.5", "10.0.0.2", 12345, 22, true);
        assert_eq!(engine.evaluate(&outside).verdict, Verdict::Drop);
    }

    #[test]
    fn auto_block_secs_propagates_on_block_not_on_accept() {
        let cfg = cfg_with_rules(
            vec![
                Rule {
                    auto_block_secs: Some(300),
                    ..rule("reject-telnet", CfgProto::Tcp, (23, 23), CfgAction::Reject)
                },
                Rule {
                    auto_block_secs: Some(300),
                    ..rule("allow-https", CfgProto::Tcp, (443, 443), CfgAction::Accept)
                },
            ],
            CfgAction::Drop,
        );
        let engine = Engine::new(&cfg);

        let blocked = tcp_pkt("10.0.0.1", "10.0.0.2", 12345, 23, true);
        let d = engine.evaluate(&blocked);
        assert_eq!(d.auto_block_secs, Some(300));

        // Even though this rule also has auto_block_secs set, an Accept
        // verdict must never trigger a quarantine -- that would be backwards.
        let allowed = tcp_pkt("10.0.0.1", "10.0.0.2", 12345, 443, true);
        let d = engine.evaluate(&allowed);
        assert_eq!(d.auto_block_secs, None);
    }

    #[test]
    fn auto_block_threshold_of_one_fires_on_first_match_default_behavior() {
        // Default threshold (1) must behave exactly like the original
        // single-shot design, so existing configs aren't silently weakened
        // or strengthened by upgrading.
        let cfg = cfg_with_rules(
            vec![Rule {
                auto_block_secs: Some(300),
                ..rule("reject-telnet", CfgProto::Tcp, (23, 23), CfgAction::Reject)
            }],
            CfgAction::Drop,
        );
        let engine = Engine::new(&cfg);
        let pkt = tcp_pkt("10.0.0.1", "10.0.0.2", 12345, 23, true);
        assert_eq!(engine.evaluate(&pkt).auto_block_secs, Some(300));
    }

    #[test]
    fn a_single_matching_packet_cannot_trigger_ban_when_threshold_is_raised() {
        // The actual spoofing-mitigation scenario: raising the threshold
        // means one packet -- spoofed or not -- is not enough on its own.
        let cfg = cfg_with_rules(
            vec![Rule {
                auto_block_secs: Some(300),
                auto_block_threshold: 3,
                ..rule("reject-telnet", CfgProto::Tcp, (23, 23), CfgAction::Reject)
            }],
            CfgAction::Drop,
        );
        let engine = Engine::new(&cfg);
        let pkt = tcp_pkt("10.0.0.1", "10.0.0.2", 12345, 23, true);

        // Still rejected each time (the rule's action still applies) --
        // just not quarantined until the threshold is reached.
        let d1 = engine.evaluate(&pkt);
        assert_eq!(d1.verdict, Verdict::Reject);
        assert_eq!(d1.auto_block_secs, None);

        let d2 = engine.evaluate(&pkt);
        assert_eq!(d2.auto_block_secs, None);

        let d3 = engine.evaluate(&pkt);
        assert_eq!(d3.auto_block_secs, Some(300));
    }

    #[test]
    fn threshold_tracking_is_per_source_not_global() {
        let cfg = cfg_with_rules(
            vec![Rule {
                auto_block_secs: Some(300),
                auto_block_threshold: 2,
                ..rule("reject-telnet", CfgProto::Tcp, (23, 23), CfgAction::Reject)
            }],
            CfgAction::Drop,
        );
        let engine = Engine::new(&cfg);

        let attacker1 = tcp_pkt("10.0.0.1", "10.0.0.2", 12345, 23, true);
        let attacker2 = tcp_pkt("10.0.0.99", "10.0.0.2", 12345, 23, true);

        // One match each from two different sources -- neither should
        // reach a threshold of 2 individually, even though 2 matches have
        // happened in total.
        assert_eq!(engine.evaluate(&attacker1).auto_block_secs, None);
        assert_eq!(engine.evaluate(&attacker2).auto_block_secs, None);
        // Second match from attacker1 specifically now reaches threshold.
        assert_eq!(engine.evaluate(&attacker1).auto_block_secs, Some(300));
    }
}
