//! Alert model and sink. Alerts are emitted as newline-delimited JSON
//! so they can be piped straight into `jq`, Filebeat, or a SIEM.

use serde::{Deserialize, Serialize};
use std::net::IpAddr;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize)]
pub struct Alert {
    pub ts_unix: u64,
    pub severity: Severity,
    /// Every call site passes a `'static` string literal (`"portscan"`,
    /// `"synflood"`, ...), so this is `&'static str` rather than
    /// `String` -- no per-alert allocation for a value that's always one
    /// of five fixed identifiers. Deliberate consequence: `Alert` derives
    /// `Serialize` but not `Deserialize` (a `Deserialize` impl for a
    /// `&'static str` field would force the whole deserialization's input
    /// lifetime to be `'static`, which a transient line of stdin/stdout
    /// text never is). A consumer parsing `Alert::emit`'s NDJSON back out
    /// -- e.g. `correlation_engine::NsmSender`'s child-process bridge --
    /// deserializes into its own small struct with an owned `String`
    /// instead; see that bridge for the pattern.
    pub detector: &'static str,
    pub message: String,
    pub src_ip: Option<IpAddr>,
    pub dst_ip: Option<IpAddr>,
    pub extra: serde_json::Value,
}

impl Alert {
    pub fn new(
        severity: Severity,
        detector: &'static str,
        message: impl Into<String>,
        src_ip: Option<IpAddr>,
        dst_ip: Option<IpAddr>,
        extra: serde_json::Value,
    ) -> Self {
        Self {
            ts_unix: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
            severity,
            detector,
            message: message.into(),
            src_ip,
            dst_ip,
            extra,
        }
    }

    pub fn emit(&self) {
        match serde_json::to_string(self) {
            Ok(line) => println!("{line}"),
            Err(e) => tracing::error!("failed to serialize alert: {e}"),
        }
    }
}
