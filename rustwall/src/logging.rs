use tracing_subscriber::EnvFilter;

/// JSON-structured logs by default so this firewall can feed a SIEM/log
/// pipeline (the equivalent of PAN-OS traffic logs shipping to Panorama, or
/// FortiGate logs shipping to FortiAnalyzer) instead of being human-eyeballs
/// only. Set RUST_LOG=rustwall=debug for verbose output during setup.
pub fn init() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(filter)
        .with_target(false)
        .init();
}
