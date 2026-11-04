use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub host_id: String,
    pub ssh: SshConfig,
    #[serde(default)]
    pub tcp_ports: Vec<TcpPortConfig>,
    #[serde(default)]
    pub honeytokens: HoneytokenConfig,
    pub sinks: SinkConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SshConfig {
    pub enabled: bool,
    pub bind_addr: String,
    pub host_key_path: PathBuf,
    /// Banner presented before auth, e.g. "SSH-2.0-OpenSSH_9.6p1 Ubuntu-3ubuntu1"
    pub banner: String,
    /// Simulated hostname shown in the fake shell prompt.
    pub fake_hostname: String,
    /// Max lines of fake filesystem to simulate for `ls`/`cat` interactions.
    #[serde(default = "default_max_session_seconds")]
    pub max_session_seconds: u64,
    /// If true, ALL credentials succeed after N attempts (classic honeypot
    /// pattern: let attacker in after a few tries to observe post-auth
    /// behavior). If false, auth always fails (pure credential harvesting).
    #[serde(default)]
    pub allow_login_after_attempts: Option<u32>,
}

fn default_max_session_seconds() -> u64 {
    600
}

#[derive(Debug, Clone, Deserialize)]
pub struct TcpPortConfig {
    pub port: u16,
    /// Optional banner sent immediately on connect (e.g. fake FTP/SMTP banner).
    pub banner: Option<String>,
    pub label: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct HoneytokenConfig {
    #[serde(default)]
    pub watch_paths: Vec<PathBuf>,
    #[serde(default)]
    pub decoy_files: Vec<DecoyFileConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecoyFileConfig {
    pub path: PathBuf,
    pub token_id: String,
    /// Content written to the decoy file at startup if it doesn't exist.
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SinkConfig {
    pub jsonl_dir: PathBuf,
    #[serde(default)]
    pub postgres: Option<PostgresSinkConfig>,
    #[serde(default)]
    pub unix_socket_path: Option<PathBuf>,
    /// Channel capacity for the internal event bus. Bounded to apply
    /// backpressure under adversarial load rather than unbounded growth.
    #[serde(default = "default_channel_capacity")]
    pub channel_capacity: usize,
}

fn default_channel_capacity() -> usize {
    4096
}

#[derive(Debug, Clone, Deserialize)]
pub struct PostgresSinkConfig {
    pub connection_string: String,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_flush_ms")]
    pub flush_interval_ms: u64,
}

fn default_batch_size() -> usize {
    50
}
fn default_flush_ms() -> u64 {
    2000
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading config file at {}", path.display()))?;
        let cfg: Config = toml::from_str(&raw)
            .with_context(|| format!("parsing TOML config at {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.host_id.trim().is_empty() {
            anyhow::bail!("host_id must not be empty");
        }
        for tp in &self.tcp_ports {
            if tp.port == 0 {
                anyhow::bail!("tcp_ports entry '{}' has invalid port 0", tp.label);
            }
        }
        Ok(())
    }
}
