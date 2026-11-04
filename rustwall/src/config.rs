use ipnet::IpNet;
use serde::Deserialize;
use std::fs;
use std::net::IpAddr;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Protocol {
    Tcp,
    Udp,
    Icmp,
    Any,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Action {
    Accept,
    Drop,
    Reject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    Inbound,
    Outbound,
    Any,
}

/// An inclusive port range. A single port is represented as (p, p).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortRange(pub u16, pub u16);

impl PortRange {
    pub fn contains(&self, port: u16) -> bool {
        port >= self.0 && port <= self.1
    }
}

impl<'de> Deserialize<'de> for PortRange {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s == "*" || s.is_empty() {
            return Ok(PortRange(0, 65535));
        }
        if let Some((lo, hi)) = s.split_once('-') {
            let lo: u16 = lo.trim().parse().map_err(serde::de::Error::custom)?;
            let hi: u16 = hi.trim().parse().map_err(serde::de::Error::custom)?;
            if lo > hi {
                return Err(serde::de::Error::custom("port range lo > hi"));
            }
            Ok(PortRange(lo, hi))
        } else {
            let p: u16 = s.trim().parse().map_err(serde::de::Error::custom)?;
            Ok(PortRange(p, p))
        }
    }
}

/// A rule's src/dst can be a literal CIDR, or a reference to a named Alias
/// (pfSense/OPNsense pattern: "alias:office_ips" instead of hardcoding
/// "10.0.0.0/24" in every rule that needs it). Resolved against the
/// config's `aliases` table at Engine construction time -- see engine.rs.
#[derive(Debug, Clone)]
pub enum NetworkMatch {
    Cidr(IpNet),
    Alias(String),
}

impl<'de> Deserialize<'de> for NetworkMatch {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if let Some(name) = s.strip_prefix("alias:") {
            Ok(NetworkMatch::Alias(name.to_string()))
        } else {
            let net: IpNet = s
                .parse()
                .map_err(|e| serde::de::Error::custom(format!("invalid CIDR '{}': {}", s, e)))?;
            Ok(NetworkMatch::Cidr(net))
        }
    }
}

/// A named, reusable set of CIDRs -- pfSense/OPNsense call this an "Alias".
/// Rules reference it by name (`src = "alias:office_ips"`) instead of
/// duplicating the same CIDR list across every rule that needs it, so
/// updating the set (e.g. rotating a blocklist) means editing one place.
#[derive(Debug, Clone, Deserialize)]
pub struct Alias {
    pub name: String,
    pub cidrs: Vec<IpNet>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Rule {
    pub name: String,
    #[serde(default = "default_proto")]
    pub protocol: Protocol,
    #[serde(default = "default_direction")]
    pub direction: Direction,
    #[serde(default = "default_net")]
    pub src: NetworkMatch,
    #[serde(default = "default_net")]
    pub dst: NetworkMatch,
    #[serde(default = "default_ports")]
    pub src_port: PortRange,
    #[serde(default = "default_ports")]
    pub dst_port: PortRange,
    pub action: Action,
    /// Optional per-rule new-connection rate limit (packets/sec) from a single source IP.
    /// Used as a SYN-flood / scan throttle, independent of global conntrack rate limiting.
    pub rate_limit_pps: Option<u32>,
    /// If set and this rule matches, the source IP is placed in the dynamic
    /// quarantine list for this many seconds -- every subsequent packet from
    /// that source is dropped immediately, before any rule evaluation, until
    /// the ban expires. This is the Untangle-style "behavioral auto-block"
    /// pattern (also what pfBlocker-NG bolts onto pfSense/OPNsense): a source
    /// that trips something like "reject-telnet" or a scan-detection rate
    /// limit gets cut off wholesale instead of just having that one
    /// connection attempt refused.
    #[serde(default)]
    pub auto_block_secs: Option<u64>,
    /// How many times this rule must match the SAME source within
    /// `auto_block_window_secs` before the ban in `auto_block_secs` actually
    /// fires. Defaults to 1, meaning "ban on the very first match" (the
    /// original behavior). Raise this if you're worried about a single
    /// spoofed packet triggering a ban -- source IP is not an authenticated
    /// field, so a lone packet claiming to be from X is not evidence X did
    /// anything; requiring repeated matches within a window at least closes
    /// the trivial single-packet case. It does not eliminate spoofing --
    /// see the caveat in rustwall.example.toml.
    #[serde(default = "default_auto_block_threshold")]
    pub auto_block_threshold: u32,
    /// The rolling window, in seconds, that `auto_block_threshold` counts
    /// matches within. Irrelevant if threshold is 1.
    #[serde(default = "default_auto_block_window_secs")]
    pub auto_block_window_secs: u64,
    #[serde(default)]
    pub log: bool,
}

fn default_auto_block_threshold() -> u32 {
    1
}
fn default_auto_block_window_secs() -> u64 {
    60
}

fn default_proto() -> Protocol {
    Protocol::Any
}
fn default_direction() -> Direction {
    Direction::Any
}
fn default_net() -> NetworkMatch {
    NetworkMatch::Cidr("0.0.0.0/0".parse().unwrap())
}
fn default_ports() -> PortRange {
    PortRange(0, 65535)
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConntrackConfig {
    #[serde(default = "default_tcp_established_timeout")]
    pub tcp_established_timeout_secs: u64,
    #[serde(default = "default_tcp_transitory_timeout")]
    pub tcp_transitory_timeout_secs: u64,
    #[serde(default = "default_udp_timeout")]
    pub udp_timeout_secs: u64,
    #[serde(default = "default_max_entries")]
    pub max_entries: usize,
}

fn default_tcp_established_timeout() -> u64 {
    3600
}
fn default_tcp_transitory_timeout() -> u64 {
    60
}
fn default_udp_timeout() -> u64 {
    30
}
fn default_max_entries() -> usize {
    250_000
}

impl Default for ConntrackConfig {
    fn default() -> Self {
        Self {
            tcp_established_timeout_secs: default_tcp_established_timeout(),
            tcp_transitory_timeout_secs: default_tcp_transitory_timeout(),
            udp_timeout_secs: default_udp_timeout(),
            max_entries: default_max_entries(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default = "default_queue_num")]
    pub queue_num: u16,
    /// Number of NFQUEUE worker threads, each bound to queue_num..queue_num+N.
    /// Requires a matching `nft`/`iptables` rule using `queue num X-Y fanout`
    /// (see README) -- binding N threads without a fanout rule just leaves
    /// threads 1..N idle since the kernel round-robins by flow hash only
    /// when fanout is requested.
    #[serde(default = "default_queue_workers")]
    pub queue_workers: u16,
    #[serde(default = "default_policy")]
    pub default_policy: Action,
    #[serde(default)]
    pub conntrack: ConntrackConfig,
    #[serde(default)]
    pub rules: Vec<Rule>,
    /// Named, reusable CIDR sets referenced by rules via "alias:<name>" --
    /// see NetworkMatch and Alias above.
    #[serde(default)]
    pub aliases: Vec<Alias>,
    /// IPs that bypass rate limiting / are always allowed regardless of rules (management plane).
    #[serde(default)]
    pub trusted: Vec<IpAddr>,
    /// Max "policy decision" log lines per second, process-wide, before
    /// excess events are counted and suppressed rather than logged
    /// individually. 0 disables the cap.
    #[serde(default = "default_log_max_per_sec")]
    pub log_max_per_sec: u64,
    /// If set, serves Prometheus-format metrics on this address (e.g.
    /// "127.0.0.1:9090"). Omit to disable the metrics endpoint entirely.
    #[serde(default)]
    pub metrics_listen: Option<String>,
    /// If set, requires `Authorization: Bearer <token>` on every request to
    /// the metrics/control endpoint -- both the `GET /metrics` scrape and
    /// the `POST /quarantine/unban/<ip>` admin action (see below). Strongly
    /// recommended if `metrics_listen` binds to anything beyond loopback;
    /// even on loopback, any local process/user can otherwise issue unban
    /// requests. Omit to leave the endpoint unauthenticated (the previous,
    /// and still default, behavior) -- a deliberate opt-in, not a silent
    /// default, since requiring auth by default would break existing
    /// scrape configs on upgrade.
    #[serde(default)]
    pub metrics_auth_token: Option<String>,
    /// Maximum number of distinct IPs the dynamic quarantine list (see
    /// `auto_block_secs` on Rule) will track at once. Bounds memory the same
    /// way `conntrack.max_entries` does -- without this, a source that can
    /// trigger auto-block on demand could grow the table without limit.
    /// Extending an existing ban never counts against this cap; only a
    /// never-before-seen IP being newly admitted does.
    #[serde(default = "default_quarantine_max_entries")]
    pub quarantine_max_entries: usize,
    /// If true, every quarantine ban (see `auto_block_secs`) is also pushed
    /// into the host's own firewall (nftables on Linux; Windows Firewall via
    /// netsh, where applicable -- see os_firewall.rs) in addition to
    /// rustwall's own in-process enforcement. Off by default: this modifies
    /// system-wide firewall state outside rustwall's own process, which is
    /// a bigger blast radius than anything else this config controls and
    /// should be an explicit choice, not a surprise default.
    #[serde(default)]
    pub sync_to_os_firewall: bool,
}

fn default_quarantine_max_entries() -> usize {
    100_000
}

fn default_queue_num() -> u16 {
    0
}
fn default_queue_workers() -> u16 {
    1
}
fn default_policy() -> Action {
    Action::Drop
}
fn default_log_max_per_sec() -> u64 {
    200
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Config> {
        let text = fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading config {}: {}", path.display(), e))?;
        let cfg: Config = toml::from_str(&text)
            .map_err(|e| anyhow::anyhow!("parsing config {}: {}", path.display(), e))?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> anyhow::Result<()> {
        for r in &self.rules {
            if r.src_port.0 > r.src_port.1 || r.dst_port.0 > r.dst_port.1 {
                anyhow::bail!("rule {}: invalid port range", r.name);
            }
        }

        let mut seen = std::collections::HashSet::new();
        for a in &self.aliases {
            if !seen.insert(a.name.as_str()) {
                anyhow::bail!("duplicate alias name '{}'", a.name);
            }
        }
        let alias_names: std::collections::HashSet<&str> =
            self.aliases.iter().map(|a| a.name.as_str()).collect();
        for r in &self.rules {
            for (field, m) in [("src", &r.src), ("dst", &r.dst)] {
                if let NetworkMatch::Alias(name) = m {
                    if !alias_names.contains(name.as_str()) {
                        anyhow::bail!(
                            "rule {}: {} references undefined alias '{}'",
                            r.name,
                            field,
                            name
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
