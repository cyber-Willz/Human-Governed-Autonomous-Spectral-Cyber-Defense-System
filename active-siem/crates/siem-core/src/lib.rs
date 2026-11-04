//! siem-core: shared data model for ActiveSIEM.
//!
//! Design notes (from analyzing Wazuh / OSSEC / Security Onion):
//! - Wazuh/OSSEC normalize everything into a "decoded" event (agent id, rule id,
//!   full log, structured fields) before rules ever see it. We do the same with
//!   `Event`, but keep the raw payload alongside structured fields so ML models
//!   downstream can re-derive features without re-parsing text.
//! - Security Onion's strength is that Zeek/Suricata produce *connection-level*
//!   records, not just line-by-line log matches. We model that via `EventKind::Flow`
//!   with numeric fields, which is what the ML crate consumes for infiltration
//!   detection (infiltration attacks look normal at the log level and only show up
//!   as subtle flow anomalies: long idle connections, small periodic beacons, odd
//!   byte ratios).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

pub type EventId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    Info = 0,
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Mirrors the two data sources every real SIEM fuses:
/// host/log telemetry (Wazuh/OSSEC agents) and network flow telemetry
/// (Security Onion's Zeek/Suricata sensors).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventKind {
    /// A decoded log line: auth logs, syslog, Windows EVTX, file integrity, etc.
    Log {
        source: String, // e.g. "sshd", "auditd", "windows-security"
        message: String,
    },
    /// A network flow / connection summary, analogous to a Zeek conn.log entry.
    /// Numeric fields here are exactly the feature set siem-ml uses.
    Flow {
        src_ip: String,
        dst_ip: String,
        src_port: u16,
        dst_port: u16,
        proto: u8, // 6=TCP, 17=UDP
        duration_ms: u64,
        bytes_src_to_dst: u64,
        bytes_dst_to_src: u64,
        packets: u64,
        flags: String, // TCP flags seen, e.g. "SAP", "S"
    },
    /// File integrity monitoring event (à la OSSEC/Wazuh syscheck).
    FileIntegrity {
        path: String,
        change: String, // "added" | "modified" | "deleted" | "perm_changed"
        hash_before: Option<String>,
        hash_after: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: EventId,
    pub timestamp_ms: u64,
    pub host: String,
    pub agent_id: String,
    pub kind: EventKind,
    /// Free-form structured fields extracted by decoders (Wazuh calls these
    /// "dynamic fields"). Kept as a bag so rules can reference arbitrary keys
    /// without changing the schema.
    pub fields: HashMap<String, String>,
}

impl Event {
    pub fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

/// What a rule or model produces. Alerts feed the correlation layer and,
/// above a configured severity, the active-response layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: EventId,
    pub timestamp_ms: u64,
    pub rule_id: String,
    pub title: String,
    pub severity: Severity,
    pub mitre_technique: Option<String>,
    pub source_events: Vec<EventId>,
    pub context: HashMap<String, String>,
}
