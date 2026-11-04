use std::io::{self, Write};
use std::net::IpAddr;
use std::process::{Command, Stdio};
use std::time::Duration;
use tracing::debug;
#[cfg(target_os = "windows")]
use tracing::warn;

/// Pushes a quarantine ban into the host's own firewall, in addition to
/// rustwall's internal NFQUEUE-based enforcement. Two reasons this is worth
/// having rather than relying on rustwall's own drop alone:
///
/// 1. **Defense in depth.** If the rustwall process crashes or is killed,
///    its own quarantine table dies with it -- but an nftables set entry (or
///    a Windows Firewall rule) keeps blocking the source at the kernel/OS
///    level regardless of whether the userspace process is even running.
/// 2. **Performance.** A banned source's packets get dropped by the kernel
///    firewall before ever reaching NFQUEUE/userspace, instead of paying the
///    netlink round-trip cost for traffic you already know you want to
///    discard.
///
/// This is genuinely optional and off by default (`sync_to_os_firewall` in
/// config) -- it modifies system-wide firewall state outside rustwall's own
/// process, which is a bigger blast radius than anything else in this
/// codebase and should be an explicit opt-in, not a surprise.
pub trait OsFirewallSync: Send + Sync {
    fn name(&self) -> &'static str;
    fn sync_ban(&self, ip: IpAddr, duration: Duration) -> io::Result<()>;
    fn sync_unban(&self, ip: IpAddr) -> io::Result<()>;
}

// ============================================================================
// Linux: nftables backend
// ============================================================================

/// Manages a dedicated `rustwall_dynamic` nftables table containing two sets
/// (`banned_v4` / `banned_v6`) with the `timeout` flag, plus drop rules
/// hooked at both `input` and `forward` -- covering both deployment models
/// documented in the README (host firewall and gateway/router) without
/// requiring extra config, since an unused hook just never sees matching
/// traffic.
///
/// The `timeout` flag on the sets is the important bit: nftables expires
/// elements itself, kernel-side, on its own schedule. `sync_unban` is
/// therefore a best-effort no-op-on-missing-element call, not something
/// rustwall's own maintenance sweep needs to drive correctness for on
/// Linux -- unlike the Windows backend, where there's no equivalent native
/// expiry and rustwall's sweep is what actually removes the block.
pub struct NftablesSync;

const NFT_TABLE: &str = "rustwall_dynamic";

impl NftablesSync {
    /// Creates (idempotently) the table, both address-family sets, and the
    /// input/forward drop rules. Safe to call every process start --
    /// `add table`/`add set`/`add chain` are no-ops if the object already
    /// exists; the rule itself is guarded by a comment tag and only added if
    /// not already present, so restarting rustwall repeatedly doesn't pile
    /// up duplicate drop rules.
    pub fn new() -> io::Result<Self> {
        let sync = Self;
        sync.ensure_infra()?;
        Ok(sync)
    }

    fn ensure_infra(&self) -> io::Result<()> {
        // Base table/sets/chains: idempotent "add" verbs, safe to re-run.
        let base_script = format!(
            "add table inet {table}\n\
             add set inet {table} banned_v4 {{ type ipv4_addr; flags timeout; }}\n\
             add set inet {table} banned_v6 {{ type ipv6_addr; flags timeout; }}\n\
             add chain inet {table} block_input {{ type filter hook input priority -10; }}\n\
             add chain inet {table} block_forward {{ type filter hook forward priority -10; }}\n",
            table = NFT_TABLE,
        );
        run_nft_script(&base_script)?;

        // Rule presence check -- "add rule" has no idempotency guarantee
        // (running it twice adds the rule twice), so we tag rules with a
        // comment and only insert if a rule with that comment isn't already
        // listed for this chain.
        for (chain, set) in [("block_input", "banned_v4"), ("block_forward", "banned_v4")] {
            self.ensure_rule(chain, "ip", set)?;
        }
        for (chain, set) in [("block_input", "banned_v6"), ("block_forward", "banned_v6")] {
            self.ensure_rule(chain, "ip6", set)?;
        }
        Ok(())
    }

    fn ensure_rule(&self, chain: &str, family_field: &str, set: &str) -> io::Result<()> {
        let comment = format!("rustwall-dynamic-{}-{}", chain, set);
        let listing = run_nft_capture(&["list", "chain", "inet", NFT_TABLE, chain])?;
        if listing.contains(&comment) {
            return Ok(()); // already present from a prior run
        }
        let script = format!(
            "add rule inet {table} {chain} {fam} saddr @{set} counter drop comment \"{comment}\"\n",
            table = NFT_TABLE,
            chain = chain,
            fam = family_field,
            set = set,
            comment = comment,
        );
        run_nft_script(&script)
    }

    fn set_name_for(ip: IpAddr) -> &'static str {
        match ip {
            IpAddr::V4(_) => "banned_v4",
            IpAddr::V6(_) => "banned_v6",
        }
    }
}

impl OsFirewallSync for NftablesSync {
    fn name(&self) -> &'static str {
        "nftables"
    }

    fn sync_ban(&self, ip: IpAddr, duration: Duration) -> io::Result<()> {
        let set = Self::set_name_for(ip);
        let script = format!(
            "add element inet {table} {set} {{ {ip} timeout {secs}s }}\n",
            table = NFT_TABLE,
            set = set,
            ip = ip,
            secs = duration.as_secs().max(1),
        );
        run_nft_script(&script)
    }

    fn sync_unban(&self, ip: IpAddr) -> io::Result<()> {
        // Best-effort: nftables' own `timeout` flag already expires this
        // element kernel-side, so this call is a courtesy for callers that
        // want a proactive removal (e.g. a future manual-unban admin
        // command) rather than something rustwall's sweep depends on for
        // correctness. Deleting an element that's already expired/absent is
        // not treated as an error worth surfacing.
        let set = Self::set_name_for(ip);
        let script = format!(
            "delete element inet {table} {set} {{ {ip} }}\n",
            table = NFT_TABLE,
            set = set,
            ip = ip,
        );
        match run_nft_script(&script) {
            Ok(()) => Ok(()),
            Err(e) => {
                debug!(error = %e, %ip, "nft delete element failed (likely already expired), ignoring");
                Ok(())
            }
        }
    }
}

fn run_nft_script(script: &str) -> io::Result<()> {
    let mut child = Command::new("nft")
        .arg("-f")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .as_mut()
        .expect("stdin piped")
        .write_all(script.as_bytes())?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "nft script failed: {}\nscript was:\n{}",
                String::from_utf8_lossy(&output.stderr),
                script
            ),
        ));
    }
    Ok(())
}

fn run_nft_capture(args: &[&str]) -> io::Result<String> {
    let output = Command::new("nft").args(args).output()?;
    // A missing table/chain (e.g. very first run) is expected and should
    // read as "no existing rule", not an error.
    if !output.status.success() {
        return Ok(String::new());
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

#[cfg(test)]
mod linux_tests {
    use super::*;
    use std::net::Ipv4Addr;

    /// This is a REAL integration test against the actual `nft` binary --
    /// not a mock -- because "does this actually change kernel firewall
    /// state" is exactly the kind of claim that shouldn't be taken on
    /// faith. Requires `nft` installed and CAP_NET_ADMIN (root in CI);
    /// `#[ignore]` so a normal `cargo test` in an unprivileged environment
    /// doesn't fail on missing capability, but it genuinely runs and
    /// verifies against the live nftables ruleset when invoked with
    /// `cargo test -- --ignored`.
    #[test]
    #[ignore = "requires nft binary and CAP_NET_ADMIN; run with --ignored"]
    fn ban_actually_creates_verifiable_nftables_state() {
        // Clean slate in case a previous run left state behind.
        let _ = Command::new("nft")
            .args(["delete", "table", "inet", NFT_TABLE])
            .status();

        let sync = NftablesSync::new().expect("nft infra should initialize");
        let ip = IpAddr::V4(Ipv4Addr::new(198, 51, 100, 77));

        sync.sync_ban(ip, Duration::from_secs(2))
            .expect("sync_ban should succeed");

        let listing = run_nft_capture(&["list", "set", "inet", NFT_TABLE, "banned_v4"])
            .expect("nft list set should run");
        assert!(
            listing.contains("198.51.100.77"),
            "banned IP should appear in the live nft set, got:\n{}",
            listing
        );

        // Verify the drop rule itself is present and wired to the set.
        let chain = run_nft_capture(&["list", "chain", "inet", NFT_TABLE, "block_input"])
            .expect("nft list chain should run");
        assert!(chain.contains("@banned_v4"), "drop rule should reference banned_v4 set");
        assert!(chain.contains("drop"), "chain should contain a drop rule");

        // nftables' own `timeout` flag should expire the element without
        // rustwall calling sync_unban at all -- verify that actually happens
        // rather than just trusting the flag exists in the script.
        std::thread::sleep(Duration::from_secs(3));
        let listing_after = run_nft_capture(&["list", "set", "inet", NFT_TABLE, "banned_v4"])
            .expect("nft list set should run");
        assert!(
            !listing_after.contains("198.51.100.77"),
            "banned IP should have auto-expired from the live nft set by now, got:\n{}",
            listing_after
        );

        let _ = Command::new("nft")
            .args(["delete", "table", "inet", NFT_TABLE])
            .status();
    }
}

// ============================================================================
// Windows: netsh advfirewall backend
// ============================================================================

/// Adds/removes Windows Firewall rules via `netsh advfirewall firewall`.
///
/// Important scope note: rustwall's packet-inspection engine (NFQUEUE) is
/// Linux-only -- Windows has no NFQUEUE equivalent, and there is currently
/// no Windows build of the inline decision loop (see README). This backend
/// exists so the *quarantine/ban bookkeeping* -- which is plain Rust with no
/// Linux-specific dependency -- can drive a real Windows Firewall change if
/// rustwall's Quarantine module is embedded in a Windows-side component
/// (e.g. a small companion sync agent receiving ban events from a Linux
/// rustwall instance over the network). Today's single binary does not
/// build for Windows as-is, because `main.rs` unconditionally depends on
/// the `nfq` crate; wiring this backend into an actual Windows build means
/// splitting the quarantine/OS-sync logic into a second, NFQUEUE-free
/// `[[bin]]` target. That split is documented in the README as the concrete
/// next step, not pretended to already exist.
///
/// Unlike nftables' native `timeout` flag, `netsh` rules don't expire
/// themselves -- whatever calls `sync_ban` is responsible for eventually
/// calling `sync_unban` when the ban's TTL elapses (rustwall's own
/// maintenance sweep does this for the nftables backend too, but only the
/// Windows backend actually depends on it for correctness).
#[cfg(target_os = "windows")]
pub struct WindowsFirewallSync;

#[cfg(target_os = "windows")]
impl WindowsFirewallSync {
    pub fn new() -> io::Result<Self> {
        Ok(Self)
    }

    fn rule_name(ip: IpAddr) -> String {
        format!("rustwall-block-{}", ip)
    }
}

#[cfg(target_os = "windows")]
impl OsFirewallSync for WindowsFirewallSync {
    fn name(&self) -> &'static str {
        "windows-firewall"
    }

    fn sync_ban(&self, ip: IpAddr, _duration: Duration) -> io::Result<()> {
        let name = Self::rule_name(ip);
        let status = Command::new("netsh")
            .args([
                "advfirewall",
                "firewall",
                "add",
                "rule",
                &format!("name={}", name),
                "dir=in",
                "action=block",
                &format!("remoteip={}", ip),
            ])
            .status()?;
        if !status.success() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("netsh add rule failed for {}", ip),
            ));
        }
        Ok(())
    }

    fn sync_unban(&self, ip: IpAddr) -> io::Result<()> {
        let name = Self::rule_name(ip);
        let status = Command::new("netsh")
            .args([
                "advfirewall",
                "firewall",
                "delete",
                "rule",
                &format!("name={}", name),
            ])
            .status()?;
        if !status.success() {
            // Deleting a rule that's already gone (e.g. removed manually)
            // isn't worth treating as a hard error.
            warn!(%ip, "netsh delete rule reported failure (rule may already be absent)");
        }
        Ok(())
    }
}
