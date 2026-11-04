//! siem-response: the part Wazuh calls "active response" and OSSEC pioneered
//! (both can shell out to a script - e.g. `firewall-drop.sh` - on a matching
//! alert). Security Onion, by contrast, is deliberately detection-only/passive.
//! Since the goal here is to *stop* infiltration, not just observe it, this
//! crate is closer to the Wazuh/OSSEC model, but with a safety policy borrowed
//! from `ai_firewall`: automated actions default to reversible, rate-limited,
//! and allowlist-checked, and anything that isn't clearly safe escalates to a
//! human instead of acting (fail-to-escalate, not fail-open/fail-closed).

use siem_core::{Alert, Severity};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

pub trait ResponseAction: Send + Sync {
    /// Returns true if the action was taken.
    fn execute(&mut self, alert: &Alert, target_ip: &str) -> bool;
    fn name(&self) -> &str;
}

/// Dry-run action: logs what *would* happen. Safe default until a real
/// executor (nftables/iptables/EDR API) is wired in and reviewed.
pub struct LogOnly;
impl ResponseAction for LogOnly {
    fn execute(&mut self, alert: &Alert, target_ip: &str) -> bool {
        println!(
            "[response:dry-run] would block {target_ip} for alert '{}' (severity {:?})",
            alert.title, alert.severity
        );
        true
    }
    fn name(&self) -> &str {
        "log-only"
    }
}

/// Guards that must all pass before any real action fires. This is the
/// FailToEscalate-style policy: default to *not* acting automatically.
pub struct ResponsePolicy {
    pub min_severity: Severity,
    pub allowlist: HashSet<String>, // IPs/hosts that must never be auto-blocked
    pub max_actions_per_window: u32,
    pub window: Duration,
    recent_actions: Vec<Instant>,
    active_blocks: HashMap<String, Instant>,
    pub block_duration: Duration,
}

impl ResponsePolicy {
    pub fn new(min_severity: Severity, allowlist: HashSet<String>) -> Self {
        Self {
            min_severity,
            allowlist,
            max_actions_per_window: 10,
            window: Duration::from_secs(60),
            recent_actions: Vec::new(),
            active_blocks: HashMap::new(),
            block_duration: Duration::from_secs(15 * 60),
        }
    }

    fn rate_limited(&mut self) -> bool {
        let now = Instant::now();
        self.recent_actions
            .retain(|t| now.duration_since(*t) < self.window);
        self.recent_actions.len() as u32 >= self.max_actions_per_window
    }

    /// Decide + (if allowed) execute. Returns whether an action was taken,
    /// and if not, why - so it can be surfaced for human review rather than
    /// silently dropped.
    pub fn handle(
        &mut self,
        action: &mut dyn ResponseAction,
        alert: &Alert,
        target_ip: &str,
    ) -> Result<(), &'static str> {
        if alert.severity < self.min_severity {
            return Err("below minimum severity for automated response");
        }
        if self.allowlist.contains(target_ip) {
            return Err("target is allowlisted; escalate to human review");
        }
        if self.active_blocks.contains_key(target_ip) {
            return Err("already actioned; skipping duplicate");
        }
        if self.rate_limited() {
            return Err("rate limit reached; escalate to human review");
        }
        if action.execute(alert, target_ip) {
            self.recent_actions.push(Instant::now());
            self.active_blocks.insert(target_ip.to_string(), Instant::now());
            Ok(())
        } else {
            Err("action executor reported failure")
        }
    }

    /// Call periodically to lift expired blocks (avoid permanent self-inflicted
    /// outages from a false positive).
    pub fn expire_blocks(&mut self) {
        let now = Instant::now();
        self.active_blocks
            .retain(|_, t| now.duration_since(*t) < self.block_duration);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use siem_core::Alert;
    use std::collections::HashMap as Map;

    fn alert(sev: Severity) -> Alert {
        Alert {
            id: 1,
            timestamp_ms: 0,
            rule_id: "test".into(),
            title: "test alert".into(),
            severity: sev,
            mitre_technique: None,
            source_events: vec![],
            context: Map::new(),
        }
    }

    #[test]
    fn allowlisted_target_never_actioned() {
        let mut allow = HashSet::new();
        allow.insert("10.0.0.1".to_string());
        let mut policy = ResponsePolicy::new(Severity::Low, allow);
        let mut action = LogOnly;
        let res = policy.handle(&mut action, &alert(Severity::Critical), "10.0.0.1");
        assert!(res.is_err());
    }

    #[test]
    fn below_severity_is_not_actioned() {
        let mut policy = ResponsePolicy::new(Severity::High, HashSet::new());
        let mut action = LogOnly;
        let res = policy.handle(&mut action, &alert(Severity::Low), "10.0.0.2");
        assert!(res.is_err());
    }

    #[test]
    fn duplicate_target_skipped() {
        let mut policy = ResponsePolicy::new(Severity::Low, HashSet::new());
        let mut action = LogOnly;
        assert!(policy
            .handle(&mut action, &alert(Severity::High), "10.0.0.3")
            .is_ok());
        assert!(policy
            .handle(&mut action, &alert(Severity::High), "10.0.0.3")
            .is_err());
    }
}
