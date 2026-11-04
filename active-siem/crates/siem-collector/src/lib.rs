//! siem-collector: the "agent" side. Wazuh/OSSEC agents tail log files and
//! ship decoded events to a manager; Security Onion sensors do the same for
//! Zeek/Suricata output. This crate keeps both paths: `FileTailer` for
//! host logs, `parse_sshd_line` as a minimal decoder example, and
//! `synthetic_flow` for exercising siem-ml without a live packet capture
//! pipeline (a real deployment would plug in `net_sys`/Zeek conn.log here).

use siem_core::{Event, EventKind};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Seek, SeekFrom};
use std::path::PathBuf;

/// Follows a growing log file from wherever it last left off, like `tail -F`.
/// Real deployments should also handle rotation (inode change) - noted as a
/// TODO rather than implemented here to keep this example crate small.
pub struct FileTailer {
    path: PathBuf,
    reader: Option<BufReader<File>>,
    offset: u64,
}

impl FileTailer {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            reader: None,
            offset: 0,
        }
    }

    fn ensure_open(&mut self) -> io::Result<()> {
        if self.reader.is_none() {
            let mut f = File::open(&self.path)?;
            f.seek(SeekFrom::Start(self.offset))?;
            self.reader = Some(BufReader::new(f));
        }
        Ok(())
    }

    /// Returns any newly appended, complete lines since the last call.
    pub fn poll(&mut self) -> io::Result<Vec<String>> {
        self.ensure_open()?;
        let reader = self.reader.as_mut().unwrap();
        let mut lines = Vec::new();
        loop {
            let mut buf = String::new();
            let n = reader.read_line(&mut buf)?;
            if n == 0 {
                break; // caught up to EOF; try again next poll
            }
            if buf.ends_with('\n') {
                self.offset += n as u64;
                lines.push(buf.trim_end().to_string());
            } else {
                // partial line at EOF - rewind so we re-read it once it's complete
                break;
            }
        }
        Ok(lines)
    }
}

/// Minimal sshd auth-log decoder, enough to feed the ssh-bruteforce rule.
/// Example line: "Jul 13 10:00:01 host sshd[123]: Failed password for root from 10.0.0.5 port 51000 ssh2"
pub fn parse_sshd_line(host: &str, agent_id: &str, line: &str) -> Option<Event> {
    if !line.contains("sshd") {
        return None;
    }
    let mut fields = HashMap::new();
    if let Some(idx) = line.find(" from ") {
        let rest = &line[idx + 6..];
        let ip = rest.split_whitespace().next().unwrap_or("").to_string();
        if !ip.is_empty() {
            fields.insert("src_ip".to_string(), ip);
        }
    }
    Some(Event {
        id: 0, // caller should assign a unique id from a counter/sequencer
        timestamp_ms: Event::now_ms(),
        host: host.to_string(),
        agent_id: agent_id.to_string(),
        kind: EventKind::Log {
            source: "sshd".to_string(),
            message: line.to_string(),
        },
        fields,
    })
}

/// Builds a synthetic Flow event, useful for tests/demos and for feeding
/// siem-ml before a real packet-capture source (e.g. net_sys) is wired in.
#[allow(clippy::too_many_arguments)]
pub fn synthetic_flow(
    host: &str,
    agent_id: &str,
    src_ip: &str,
    dst_ip: &str,
    src_port: u16,
    dst_port: u16,
    duration_ms: u64,
    bytes_out: u64,
    bytes_in: u64,
    packets: u64,
) -> Event {
    Event {
        id: 0,
        timestamp_ms: Event::now_ms(),
        host: host.to_string(),
        agent_id: agent_id.to_string(),
        kind: EventKind::Flow {
            src_ip: src_ip.to_string(),
            dst_ip: dst_ip.to_string(),
            src_port,
            dst_port,
            proto: 6,
            duration_ms,
            bytes_src_to_dst: bytes_out,
            bytes_dst_to_src: bytes_in,
            packets,
            flags: "SAP".to_string(),
        },
        fields: HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_failed_password_line() {
        let line = "Jul 13 10:00:01 host sshd[123]: Failed password for root from 10.0.0.5 port 51000 ssh2";
        let ev = parse_sshd_line("host1", "agent1", line).unwrap();
        assert_eq!(ev.fields.get("src_ip").unwrap(), "10.0.0.5");
    }

    #[test]
    fn ignores_non_sshd_lines() {
        assert!(parse_sshd_line("host1", "agent1", "some other log line").is_none());
    }
}
