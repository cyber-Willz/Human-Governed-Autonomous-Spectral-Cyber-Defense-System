//! siem-rules: correlation engine.
//!
//! What we borrowed from each project:
//! - OSSEC/Wazuh "frequency rules" (`if_matched_sid` + `frequency`/`timeframe`) are the
//!   backbone of brute-force and scan detection. We reimplement that as
//!   `RuleKind::Threshold` with a sliding time window per group-by key.
//! - Security Onion leans on Sigma rules for portable, declarative detections.
//!   `RuleKind::Match` is a Sigma-lite subset (field equals/contains/regex) so rules
//!   ship as YAML, not Rust code.
//! - Infiltration (MITRE TA0001/TA0008-adjacent behavior: slow recon, lateral movement
//!   after an initial low-noise foothold) is specifically hard for pure signature
//!   matching because no single log line looks malicious. The two built-in rules below
//!   (`port-scan-vertical`, `ssh-bruteforce`) catch the *noisy* precursors; the quiet
//!   cases are handed to siem-ml's autoencoder instead of trying to write a signature
//!   for "conversation that is a bit too regular and a bit too small".

use regex::Regex;
use serde::{Deserialize, Serialize};
use siem_core::{Alert, Event, EventKind, Severity};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum FieldMatch {
    Equals(String),
    Contains(String),
    Regex(String),
}

impl FieldMatch {
    fn is_match(&self, value: &str) -> bool {
        match self {
            FieldMatch::Equals(v) => value == v,
            FieldMatch::Contains(v) => value.contains(v.as_str()),
            FieldMatch::Regex(pat) => Regex::new(pat).map(|r| r.is_match(value)).unwrap_or(false),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum RuleKind {
    /// Single-event pattern match against `fields[field_name]`.
    Match {
        field: String,
        pattern: FieldMatch,
    },
    /// OSSEC-style frequency rule: fire when >= `count` matching events with the same
    /// `group_by` value occur within `window_secs`.
    Threshold {
        field: String,
        pattern: FieldMatch,
        group_by: String,
        count: u32,
        window_secs: u64,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RuleDef {
    pub id: String,
    pub title: String,
    pub severity: SeverityDef,
    pub mitre_technique: Option<String>,
    pub kind: RuleKind,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub enum SeverityDef {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

impl From<SeverityDef> for Severity {
    fn from(s: SeverityDef) -> Self {
        match s {
            SeverityDef::Info => Severity::Info,
            SeverityDef::Low => Severity::Low,
            SeverityDef::Medium => Severity::Medium,
            SeverityDef::High => Severity::High,
            SeverityDef::Critical => Severity::Critical,
        }
    }
}

/// Per-threshold-rule sliding window state, keyed by (rule_id, group value).
struct WindowState {
    hits: VecDeque<u64>, // timestamps (ms) within the window
}

pub struct RuleEngine {
    rules: Vec<RuleDef>,
    windows: HashMap<(String, String), WindowState>,
    next_alert_id: u64,
}

impl RuleEngine {
    pub fn new(rules: Vec<RuleDef>) -> Self {
        Self {
            rules,
            windows: HashMap::new(),
            next_alert_id: 1,
        }
    }

    pub fn load_yaml(yaml: &str) -> Result<Vec<RuleDef>, serde_yaml::Error> {
        serde_yaml::from_str(yaml)
    }

    /// Extract the flat field bag an Event exposes to rules: `fields` plus a
    /// few well-known derived keys for Flow/Log/FileIntegrity variants.
    fn resolve(event: &Event, field: &str) -> Option<String> {
        if let Some(v) = event.fields.get(field) {
            return Some(v.clone());
        }
        match (&event.kind, field) {
            (EventKind::Log { source, .. }, "source") => Some(source.clone()),
            (EventKind::Log { message, .. }, "message") => Some(message.clone()),
            (EventKind::Flow { src_ip, .. }, "src_ip") => Some(src_ip.clone()),
            (EventKind::Flow { dst_ip, .. }, "dst_ip") => Some(dst_ip.clone()),
            (EventKind::Flow { dst_port, .. }, "dst_port") => Some(dst_port.to_string()),
            (EventKind::FileIntegrity { path, .. }, "path") => Some(path.clone()),
            (EventKind::FileIntegrity { change, .. }, "change") => Some(change.clone()),
            _ => None,
        }
    }

    pub fn process(&mut self, event: &Event) -> Vec<Alert> {
        let mut alerts = Vec::new();
        // Clone rule list indices to avoid borrow conflicts with self.windows.
        for i in 0..self.rules.len() {
            let rule = self.rules[i].clone();
            match &rule.kind {
                RuleKind::Match { field, pattern } => {
                    if let Some(v) = Self::resolve(event, field) {
                        if pattern.is_match(&v) {
                            alerts.push(self.build_alert(&rule, event, HashMap::new()));
                        }
                    }
                }
                RuleKind::Threshold {
                    field,
                    pattern,
                    group_by,
                    count,
                    window_secs,
                } => {
                    let Some(v) = Self::resolve(event, field) else {
                        continue;
                    };
                    if !pattern.is_match(&v) {
                        continue;
                    }
                    let Some(group_val) = Self::resolve(event, group_by) else {
                        continue;
                    };
                    let key = (rule.id.clone(), group_val.clone());
                    let state = self.windows.entry(key).or_insert(WindowState {
                        hits: VecDeque::new(),
                    });
                    state.hits.push_back(event.timestamp_ms);
                    let cutoff = event
                        .timestamp_ms
                        .saturating_sub(Duration::from_secs(*window_secs).as_millis() as u64);
                    while let Some(front) = state.hits.front() {
                        if *front < cutoff {
                            state.hits.pop_front();
                        } else {
                            break;
                        }
                    }
                    let hit_count = state.hits.len() as u32;
                    if hit_count >= *count {
                        self.windows
                            .get_mut(&(rule.id.clone(), group_val.clone()))
                            .unwrap()
                            .hits
                            .clear(); // avoid re-firing every subsequent event
                        let mut ctx = HashMap::new();
                        ctx.insert(group_by.clone(), group_val);
                        ctx.insert("hit_count".to_string(), hit_count.to_string());
                        alerts.push(self.build_alert(&rule, event, ctx));
                    }
                }
            }
        }
        alerts
    }

    fn build_alert(
        &mut self,
        rule: &RuleDef,
        event: &Event,
        context: HashMap<String, String>,
    ) -> Alert {
        let id = self.next_alert_id;
        self.next_alert_id += 1;
        Alert {
            id,
            timestamp_ms: event.timestamp_ms,
            rule_id: rule.id.clone(),
            title: rule.title.clone(),
            severity: rule.severity.into(),
            mitre_technique: rule.mitre_technique.clone(),
            source_events: vec![event.id],
            context,
        }
    }
}

/// Built-in rules covering the noisy precursors to infiltration:
/// SSH brute force (credential access) and vertical port scanning (discovery).
pub fn builtin_rules_yaml() -> &'static str {
    r#"
- id: ssh-bruteforce
  title: "Repeated SSH authentication failures from one source"
  severity: High
  mitre_technique: "T1110"
  kind: !Threshold
    field: message
    pattern: !Contains "Failed password"
    group_by: src_ip
    count: 5
    window_secs: 60

- id: port-scan-vertical
  title: "Vertical port scan (many destination ports, single source)"
  severity: Medium
  mitre_technique: "T1046"
  kind: !Threshold
    field: dst_port
    pattern: !Regex '^\d+$'
    group_by: src_ip
    count: 20
    window_secs: 30
"#
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as Map;

    fn log_event(id: u64, ts: u64, src_ip: &str, msg: &str) -> Event {
        let mut fields = Map::new();
        fields.insert("src_ip".to_string(), src_ip.to_string());
        Event {
            id,
            timestamp_ms: ts,
            host: "host1".into(),
            agent_id: "agent1".into(),
            kind: EventKind::Log {
                source: "sshd".into(),
                message: msg.into(),
            },
            fields,
        }
    }

    #[test]
    fn bruteforce_fires_after_threshold() {
        let rules = RuleEngine::load_yaml(builtin_rules_yaml()).unwrap();
        let mut engine = RuleEngine::new(rules);
        let mut fired = 0;
        for i in 0..5 {
            let ev = log_event(i, i * 1000, "10.0.0.5", "Failed password for root");
            fired += engine.process(&ev).len();
        }
        assert_eq!(fired, 1, "should fire exactly once at the 5th failure");
    }

    #[test]
    fn bruteforce_does_not_fire_below_threshold() {
        let rules = RuleEngine::load_yaml(builtin_rules_yaml()).unwrap();
        let mut engine = RuleEngine::new(rules);
        let mut fired = 0;
        for i in 0..4 {
            let ev = log_event(i, i * 1000, "10.0.0.5", "Failed password for root");
            fired += engine.process(&ev).len();
        }
        assert_eq!(fired, 0);
    }

    #[test]
    fn window_expires_old_hits() {
        let rules = RuleEngine::load_yaml(builtin_rules_yaml()).unwrap();
        let mut engine = RuleEngine::new(rules);
        // 4 hits far in the past, then 4 recent ones: should NOT combine to 5+ recent.
        for i in 0..4 {
            let ev = log_event(i, i * 1000, "10.0.0.9", "Failed password for root");
            engine.process(&ev);
        }
        let mut fired = 0;
        for i in 0..4 {
            let ev = log_event(100 + i, 120_000 + i * 1000, "10.0.0.9", "Failed password for root");
            fired += engine.process(&ev).len();
        }
        assert_eq!(fired, 0, "old hits outside the 60s window must not count");
    }
}
