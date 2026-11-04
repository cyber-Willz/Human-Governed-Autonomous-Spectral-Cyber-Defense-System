//! Unified honeypot event schema.
//!
//! All honeypot subsystems (SSH, multi-port TCP, honeytokens) emit
//! `HoneypotEvent` onto a single bounded `mpsc` channel. This mirrors the
//! fan-in pattern used by `active-siem`'s correlation engine so that
//! ingestion on the SIEM side is a drop-in: the log shipper harvests the
//! JSONL sink, or a downstream process subscribes directly over the Unix
//! socket forwarder.
//!
//! Design note: every event carries `confidence: Confidence::Certain`
//! unless explicitly noted. Honeypot hits have (by construction) no
//! legitimate traffic baseline to separate from -- there is no reason for
//! any real client to ever touch these listeners or tokens -- so unlike
//! the spectral anomaly detector's autoencoder scores, these events do not
//! need probabilistic calibration. They exist to feed the neural_hmm
//! belief filter as near-ground-truth emissions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::net::IpAddr;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    /// No legitimate path to this signal exists (honeytoken touch, auth
    /// attempt against a service with zero real users).
    Certain,
    /// Heuristic / protocol-guess based, e.g. unrecognized binary probe on
    /// an unmonitored port.
    Probable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EventKind {
    /// A TCP connection was accepted on a decoy listener.
    ConnectionOpened { local_port: u16, protocol_guess: Option<String> },
    /// SSH authentication was attempted against the honeypot server.
    SshAuthAttempt {
        username: String,
        method: String,
        credential: Option<String>,
        client_version: Option<String>,
    },
    /// A shell command was issued inside a fake interactive session.
    SshCommandExecuted { command: String, session_id: Uuid },
    /// Raw bytes were sent to a low-interaction port listener.
    RawPayload {
        local_port: u16,
        byte_len: usize,
        printable_sample: String,
        protocol_guess: Option<String>,
    },
    /// A honeytoken (decoy file, config secret, or DB row) was accessed.
    HoneytokenTriggered {
        token_id: String,
        token_kind: HoneytokenKind,
        access_detail: String,
    },
    /// The connection closed / session ended.
    SessionClosed { session_id: Uuid, duration_ms: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HoneytokenKind {
    DecoyFile,
    DecoyDbRow,
    DecoyApiKey,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoneypotEvent {
    pub event_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub source_ip: Option<IpAddr>,
    pub source_port: Option<u16>,
    pub host_id: String,
    pub sensor: String,
    pub confidence: Confidence,
    pub severity: Severity,
    pub kind: EventKind,
}

impl HoneypotEvent {
    pub fn new(
        host_id: &str,
        sensor: &str,
        source_ip: Option<IpAddr>,
        source_port: Option<u16>,
        confidence: Confidence,
        severity: Severity,
        kind: EventKind,
    ) -> Self {
        Self {
            event_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            source_ip,
            source_port,
            host_id: host_id.to_string(),
            sensor: sensor.to_string(),
            confidence,
            severity,
            kind,
        }
    }
}
