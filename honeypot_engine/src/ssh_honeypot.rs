//! Medium-interaction SSH honeypot built on `russh`.
//!
//! Real SSH protocol negotiation (via `russh`) rather than a banner-only
//! emulator, so client fingerprinting via KEX/cipher-suite quirks is much
//! harder than a hand-rolled banner responder. Every auth attempt is
//! logged regardless of outcome. Optionally, after N attempts, login is
//! allowed to succeed into a simulated shell so post-auth attacker
//! behavior (recon commands, malware staging attempts) can be observed --
//! this is the same rationale Cowrie/Kippo-style honeypots use.
//!
//! NOTE ON API STABILITY: russh's server `Handler` trait has changed
//! signature shape across 0.4x releases (async_trait usage, return types,
//! and the exact auth method names have all shifted between versions).
//! This module targets russh/russh-keys 0.44 as pinned in Cargo.toml.
//! Before deploying, run `cargo doc --open -p russh` (or check
//! docs.rs/russh/0.44) and diff the `server::Handler` trait against what's
//! implemented below -- this is the single most likely place for a
//! version-drift compile error, and I could not verify compilation in
//! this environment (no rustc/cargo available in the sandbox).

use crate::event::{Confidence, EventKind, HoneypotEvent, Severity};
use anyhow::{Context, Result};
use async_trait::async_trait;
use russh::server::{Auth, Config as RusshConfig, Handler, Msg, Server as RusshServer, Session};
use russh::{Channel, ChannelId};
use russh_keys::key::KeyPair;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::mpsc::Sender;
use tokio::time::Duration;
use tracing::{info, warn};
use uuid::Uuid;

use crate::config::SshConfig;

/// Fake filesystem surface used to answer `ls` / `cat` / `pwd` inside the
/// simulated shell. Kept intentionally small and generic -- do not encode
/// anything about real internal `active_spectral_defense` paths, hostnames
/// or credentials here, since attacker output could leak into logs shared
/// outside the security team.
fn fake_ls_output() -> &'static str {
    "backup.tar.gz\nconfig.yml\nid_rsa\nnotes.txt\n"
}

pub struct SshHoneypot {
    cfg: SshConfig,
    host_id: String,
    tx: Sender<HoneypotEvent>,
    attempt_counts: Arc<std::sync::Mutex<HashMap<std::net::IpAddr, u32>>>,
}

impl SshHoneypot {
    pub fn new(cfg: SshConfig, host_id: String, tx: Sender<HoneypotEvent>) -> Self {
        Self {
            cfg,
            host_id,
            tx,
            attempt_counts: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    pub async fn run(self) -> Result<()> {
        if !self.cfg.enabled {
            info!("ssh honeypot disabled in config, skipping");
            return Ok(());
        }

        let key_pair = load_or_generate_host_key(&self.cfg.host_key_path)?;

        let mut russh_cfg = RusshConfig::default();
        russh_cfg.server_id = russh::SshId::Standard(self.cfg.banner.clone());
        russh_cfg.keys.push(key_pair);
        russh_cfg.auth_rejection_time = Duration::from_millis(750); // resist timing-based valid-user enumeration
        let russh_cfg = Arc::new(russh_cfg);

        let addr: SocketAddr = self
            .cfg
            .bind_addr
            .parse()
            .with_context(|| format!("parsing ssh bind_addr '{}'", self.cfg.bind_addr))?;

        info!(addr = %addr, "ssh honeypot listening");

        let mut server = ServerHandle {
            cfg: self.cfg,
            host_id: self.host_id,
            tx: self.tx,
            attempt_counts: self.attempt_counts,
        };

        server
            .run_on_address(russh_cfg, addr)
            .await
            .context("ssh honeypot server exited")
    }
}

fn load_or_generate_host_key(path: &std::path::Path) -> Result<KeyPair> {
    if path.exists() {
        russh_keys::load_secret_key(path, None)
            .with_context(|| format!("loading existing ssh host key at {}", path.display()))
    } else {
        let key = KeyPair::generate_ed25519()
            .context("generating ed25519 host key (keypair generation returned None)")?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let mut file = std::fs::File::create(path)
            .with_context(|| format!("creating host key file at {}", path.display()))?;
        russh_keys::encode_pkcs8_pem(&key, &mut file)
            .with_context(|| format!("writing generated host key to {}", path.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).ok();
        }
        warn!(path = %path.display(), "generated a new SSH host key (none found)");
        Ok(key)
    }
}

#[derive(Clone)]
struct ServerHandle {
    cfg: SshConfig,
    host_id: String,
    tx: Sender<HoneypotEvent>,
    attempt_counts: Arc<std::sync::Mutex<HashMap<std::net::IpAddr, u32>>>,
}

impl RusshServer for ServerHandle {
    type Handler = SessionHandler;

    fn new_client(&mut self, peer_addr: Option<SocketAddr>) -> Self::Handler {
        SessionHandler {
            cfg: self.cfg.clone(),
            host_id: self.host_id.clone(),
            tx: self.tx.clone(),
            attempt_counts: self.attempt_counts.clone(),
            peer_addr,
            session_id: Uuid::new_v4(),
            client_version: None,
            started_at: std::time::Instant::now(),
            fake_cwd: "/home/user".to_string(),
        }
    }
}

pub struct SessionHandler {
    cfg: SshConfig,
    host_id: String,
    tx: Sender<HoneypotEvent>,
    attempt_counts: Arc<std::sync::Mutex<HashMap<std::net::IpAddr, u32>>>,
    peer_addr: Option<SocketAddr>,
    session_id: Uuid,
    client_version: Option<String>,
    started_at: std::time::Instant,
    fake_cwd: String,
}

impl SessionHandler {
    fn emit(&self, kind: EventKind, confidence: Confidence, severity: Severity) {
        let event = HoneypotEvent::new(
            &self.host_id,
            "ssh_honeypot",
            self.peer_addr.map(|a| a.ip()),
            self.peer_addr.map(|a| a.port()),
            confidence,
            severity,
            kind,
        );
        // Non-blocking best-effort send: a full channel (adversarial burst)
        // must never stall the SSH session loop itself.
        if let Err(e) = self.tx.try_send(event) {
            warn!(error = %e, "ssh honeypot: event bus full or closed, dropping event");
        }
    }

    fn attempts_for(&self, ip: std::net::IpAddr) -> u32 {
        let mut map = self.attempt_counts.lock().unwrap();
        let counter = map.entry(ip).or_insert(0);
        *counter += 1;
        *counter
    }
}

#[async_trait]
impl Handler for SessionHandler {
    type Error = anyhow::Error;

    async fn auth_password(&mut self, user: &str, password: &str) -> Result<Auth, Self::Error> {
        self.emit(
            EventKind::SshAuthAttempt {
                username: user.to_string(),
                method: "password".to_string(),
                credential: Some(password.to_string()),
                client_version: self.client_version.clone(),
            },
            Confidence::Certain,
            Severity::Medium,
        );

        let attempts = self
            .peer_addr
            .map(|a| self.attempts_for(a.ip()))
            .unwrap_or(u32::MAX);

        let allow = self
            .cfg
            .allow_login_after_attempts
            .map(|n| attempts >= n)
            .unwrap_or(false);

        if allow {
            Ok(Auth::Accept)
        } else {
            Ok(Auth::Reject {
                proceed_with_methods: None,
            })
        }
    }

    async fn auth_publickey(
        &mut self,
        user: &str,
        _public_key: &russh_keys::key::PublicKey,
    ) -> Result<Auth, Self::Error> {
        self.emit(
            EventKind::SshAuthAttempt {
                username: user.to_string(),
                method: "publickey".to_string(),
                credential: None,
                client_version: self.client_version.clone(),
            },
            Confidence::Certain,
            Severity::Medium,
        );
        // Never accept pubkey auth -- accepting would require possessing
        // the matching private key server-side, which adds real risk for
        // no additional signal over password/keyboard-interactive capture.
        Ok(Auth::Reject {
            proceed_with_methods: None,
        })
    }

    async fn channel_open_session(
        &mut self,
        _channel: Channel<Msg>,
        _session: &mut Session,
    ) -> Result<bool, Self::Error> {
        Ok(true)
    }

    async fn shell_request(
        &mut self,
        channel: ChannelId,
        session: &mut Session,
    ) -> Result<(), Self::Error> {
        let banner = format!(
            "Welcome to {}\r\nLast login: Mon Jul 20 09:14:02 2026 from 10.0.0.7\r\n$ ",
            self.cfg.fake_hostname
        );
        session.data(channel, banner.into_bytes().into());
        Ok(())
    }

    async fn exec_request(
        &mut self,
        channel: ChannelId,
        data: &[u8],
        session: &mut Session,
    ) -> Result<(), Self::Error> {
        let command = String::from_utf8_lossy(data).to_string();
        self.handle_command(&command, channel, session);
        session.channel_success(channel);
        session.close(channel);
        Ok(())
    }

    async fn data(
        &mut self,
        channel: ChannelId,
        data: &[u8],
        session: &mut Session,
    ) -> Result<(), Self::Error> {
        // Interactive shell input arrives byte-by-byte or line-buffered
        // depending on client; treat newline-terminated input as a command.
        let input = String::from_utf8_lossy(data).to_string();
        for line in input.split(['\n', '\r']).filter(|l| !l.is_empty()) {
            self.handle_command(line, channel, session);
            let prompt = format!("\r\n{}:{}$ ", self.cfg.fake_hostname, self.fake_cwd);
            session.data(channel, prompt.into_bytes().into());
        }
        Ok(())
    }
}

impl SessionHandler {
    fn handle_command(&mut self, command: &str, channel: ChannelId, session: &mut Session) {
        let trimmed = command.trim();
        self.emit(
            EventKind::SshCommandExecuted {
                command: trimmed.to_string(),
                session_id: self.session_id,
            },
            Confidence::Certain,
            Severity::High,
        );

        let response = match trimmed.split_whitespace().next().unwrap_or("") {
            "ls" | "dir" => fake_ls_output().to_string(),
            "pwd" => format!("{}\r\n", self.fake_cwd),
            "whoami" => "user\r\n".to_string(),
            "uname" | "uname -a" => {
                "Linux honeypot 6.8.0-generic #1 SMP x86_64 GNU/Linux\r\n".to_string()
            }
            "id" => "uid=1000(user) gid=1000(user) groups=1000(user)\r\n".to_string(),
            "cat" => "cat: missing operand\r\n".to_string(),
            "exit" | "logout" => {
                session.close(channel);
                return;
            }
            "" => String::new(),
            other => format!("bash: {}: command not found\r\n", other),
        };

        if !response.is_empty() {
            session.data(channel, response.into_bytes().into());
        }
    }
}

impl Drop for SessionHandler {
    fn drop(&mut self) {
        let event = HoneypotEvent::new(
            &self.host_id,
            "ssh_honeypot",
            self.peer_addr.map(|a| a.ip()),
            self.peer_addr.map(|a| a.port()),
            Confidence::Certain,
            Severity::Info,
            EventKind::SessionClosed {
                session_id: self.session_id,
                duration_ms: self.started_at.elapsed().as_millis() as u64,
            },
        );
        let _ = self.tx.try_send(event);
    }
}
