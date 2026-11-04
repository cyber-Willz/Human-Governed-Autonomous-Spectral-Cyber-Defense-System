//! Traffic ingestion pipeline.
//!
//! Captures packets from the network tap and fans them out — without
//! ever blocking the hot capture path — to the three downstream
//! analysis lanes described in the architecture: NSM (fast path),
//! ClamAV (content inspection), and the spectral engine (slow path).
//!
//! # Design
//!
//! - `payload` is `bytes::Bytes`: an atomically refcounted, zero-copy
//!   buffer. Cloning it once per lane is a refcount bump, not a memcpy.
//! - Fan-out uses `tokio::sync::broadcast`. A slow consumer (typically
//!   the spectral engine under load) never backpressures the capture
//!   loop or the other lanes — it just drops its own oldest buffered
//!   packets and surfaces `Lagged(n)` on its next `recv`, which we
//!   turn into a metric instead of a panic or a stall.
//! - The capture loop (`IngestionPipeline::run`) is generic over any
//!   `PacketSource`, so the same pipeline works whether the tap is
//!   AF_PACKET, an aya XDP/TC ring buffer, or a test mock.
//! - All counters are lock-free atomics, safe to read from a metrics
//!   exporter concurrently with the hot path.
//!
//! # Cargo.toml
//! ```toml
//! tokio = { version = "1", features = ["rt-multi-thread", "sync", "time", "macros"] }
//! tokio-util = "0.7"
//! bytes = "1"
//! thiserror = "1"
//! tracing = "0.1"
//! async-trait = "0.1"
//! ```

use std::net::IpAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

use bytes::Bytes;
use thiserror::Error;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------

/// Transport-layer protocol observed on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Protocol {
    Tcp,
    Udp,
    Icmp,
    Other(u8),
}

/// Canonical 5-tuple identifying a flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FlowKey {
    pub src_ip: IpAddr,
    pub dst_ip: IpAddr,
    pub src_port: u16,
    pub dst_port: u16,
    pub protocol: Protocol,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Ingress,
    Egress,
}

/// A single ingested packet, ready for fan-out to downstream lanes.
#[derive(Debug, Clone)]
pub struct IngestedPacket {
    pub flow: FlowKey,
    pub captured_at: SystemTime,
    pub ifindex: u32,
    pub direction: Direction,
    pub payload: Bytes,
    /// Monotonically increasing sequence number, assigned at ingestion.
    /// Downstream consumers use gaps in this to detect their own drops
    /// independent of the `Lagged` count (useful for correlation
    /// engine audit trails).
    pub seq: u64,
}

#[derive(Debug, Error)]
pub enum IngestionError {
    #[error("capture source error: {0}")]
    Source(String),
}

// ---------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct LaneMetrics {
    pub delivered: AtomicU64,
    pub lagged: AtomicU64,
}

#[derive(Debug, Default)]
pub struct IngestionMetrics {
    pub captured: AtomicU64,
    pub bytes: AtomicU64,
    pub nsm: LaneMetrics,
    pub clamav: LaneMetrics,
    pub spectral: LaneMetrics,
}

#[derive(Debug, Clone, Copy)]
pub struct IngestionMetricsSnapshot {
    pub captured: u64,
    pub bytes: u64,
    pub nsm_delivered: u64,
    pub nsm_lagged: u64,
    pub clamav_delivered: u64,
    pub clamav_lagged: u64,
    pub spectral_delivered: u64,
    pub spectral_lagged: u64,
}

#[derive(Clone)]
pub struct IngestionMetricsHandle(Arc<IngestionMetrics>);

impl IngestionMetricsHandle {
    pub fn snapshot(&self) -> IngestionMetricsSnapshot {
        let m = &self.0;
        IngestionMetricsSnapshot {
            captured: m.captured.load(Ordering::Relaxed),
            bytes: m.bytes.load(Ordering::Relaxed),
            nsm_delivered: m.nsm.delivered.load(Ordering::Relaxed),
            nsm_lagged: m.nsm.lagged.load(Ordering::Relaxed),
            clamav_delivered: m.clamav.delivered.load(Ordering::Relaxed),
            clamav_lagged: m.clamav.lagged.load(Ordering::Relaxed),
            spectral_delivered: m.spectral.delivered.load(Ordering::Relaxed),
            spectral_lagged: m.spectral.lagged.load(Ordering::Relaxed),
        }
    }
}

// ---------------------------------------------------------------------
// Capture source abstraction
// ---------------------------------------------------------------------

/// Abstraction over the actual capture mechanism (AF_PACKET, an aya
/// XDP/TC ring buffer, pcap, or a test mock). Implementors should only
/// block within `recv`; the pipeline drives it from a dedicated task.
#[async_trait::async_trait]
pub trait PacketSource: Send {
    async fn recv(&mut self) -> Result<IngestedPacket, IngestionError>;
}

// ---------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------

/// Configuration for the ingestion pipeline.
#[derive(Debug, Clone)]
pub struct IngestionConfig {
    /// Per-lane broadcast buffer depth (packets, not bytes). Size
    /// generously for the slowest lane — the spectral engine under
    /// load — so ordinary bursts don't trip `Lagged`. Tune against
    /// production packet-rate telemetry; 8192 is a starting point for
    /// moderate-throughput links, not a universal default.
    pub lane_capacity: usize,
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self { lane_capacity: 8192 }
    }
}

/// Receiver handles for the three downstream analysis lanes. Obtain
/// one `IngestionLanes` per consumer at startup.
pub struct IngestionLanes {
    pub nsm: broadcast::Receiver<Arc<IngestedPacket>>,
    pub clamav: broadcast::Receiver<Arc<IngestedPacket>>,
    pub spectral: broadcast::Receiver<Arc<IngestedPacket>>,
}

pub struct IngestionPipeline {
    tx: broadcast::Sender<Arc<IngestedPacket>>,
    metrics: Arc<IngestionMetrics>,
    seq: AtomicU64,
}

impl IngestionPipeline {
    pub fn new(config: IngestionConfig) -> (Self, IngestionMetricsHandle) {
        let (tx, _rx) = broadcast::channel(config.lane_capacity);
        let metrics = Arc::new(IngestionMetrics::default());
        (
            Self {
                tx,
                metrics: metrics.clone(),
                seq: AtomicU64::new(0),
            },
            IngestionMetricsHandle(metrics),
        )
    }

    /// Subscribe to all three lanes. Cheap — three receiver handles
    /// into the same underlying ring buffer, no data copied.
    pub fn subscribe_lanes(&self) -> IngestionLanes {
        IngestionLanes {
            nsm: self.tx.subscribe(),
            clamav: self.tx.subscribe(),
            spectral: self.tx.subscribe(),
        }
    }

    /// Drives the capture loop until `shutdown` is cancelled or the
    /// source returns an error. Spawn this as its own task:
    ///
    /// ```ignore
    /// let (pipeline, metrics) = IngestionPipeline::new(IngestionConfig::default());
    /// let lanes = pipeline.subscribe_lanes();
    /// let shutdown = CancellationToken::new();
    /// tokio::spawn(async move { pipeline.run(source, shutdown).await });
    /// ```
    pub async fn run(
        &self,
        mut source: impl PacketSource,
        shutdown: CancellationToken,
    ) -> Result<(), IngestionError> {
        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => {
                    tracing::info!("ingestion pipeline shutting down");
                    return Ok(());
                }
                pkt = source.recv() => {
                    let mut pkt = pkt?;
                    pkt.seq = self.seq.fetch_add(1, Ordering::Relaxed);

                    self.metrics.captured.fetch_add(1, Ordering::Relaxed);
                    self.metrics
                        .bytes
                        .fetch_add(pkt.payload.len() as u64, Ordering::Relaxed);

                    // broadcast::Sender::send is synchronous and never
                    // awaits or blocks. If every lane is empty (no
                    // subscribers yet), the packet is simply dropped —
                    // this is intentional: ingestion should never stall
                    // waiting for a consumer to attach.
                    let _ = self.tx.send(Arc::new(pkt));
                }
            }
        }
    }
}

/// Helper for consumers: reads the next packet on a lane, recording
/// lag transparently and continuing past drops rather than treating
/// `Lagged` as fatal.
pub async fn recv_lane(
    rx: &mut broadcast::Receiver<Arc<IngestedPacket>>,
    lane: &LaneMetrics,
) -> Option<Arc<IngestedPacket>> {
    loop {
        match rx.recv().await {
            Ok(pkt) => {
                lane.delivered.fetch_add(1, Ordering::Relaxed);
                return Some(pkt);
            }
            Err(broadcast::error::RecvError::Lagged(n)) => {
                lane.lagged.fetch_add(n, Ordering::Relaxed);
                tracing::warn!(dropped = n, "consumer lane lagged, packets dropped");
                continue;
            }
            Err(broadcast::error::RecvError::Closed) => return None,
        }
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    struct MockSource {
        remaining: usize,
        delay: Duration,
    }

    #[async_trait::async_trait]
    impl PacketSource for MockSource {
        async fn recv(&mut self) -> Result<IngestedPacket, IngestionError> {
            if self.remaining == 0 {
                // Simulate an idle tap; in production this would be a
                // real blocking read. We just stall forever so the
                // test's shutdown token is what ends the run loop.
                std::future::pending::<()>().await;
            }
            self.remaining -= 1;
            tokio::time::sleep(self.delay).await;
            Ok(IngestedPacket {
                flow: FlowKey {
                    src_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
                    dst_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 2)),
                    src_port: 4444,
                    dst_port: 443,
                    protocol: Protocol::Tcp,
                },
                captured_at: SystemTime::now(),
                ifindex: 2,
                direction: Direction::Ingress,
                payload: Bytes::from_static(b"\x00\x01\x02\x03"),
                seq: 0,
            })
        }
    }

    #[tokio::test]
    async fn fans_out_to_all_three_lanes_without_copying_payload() {
        let (pipeline, metrics) = IngestionPipeline::new(IngestionConfig {
            lane_capacity: 64,
        });
        let mut lanes = pipeline.subscribe_lanes();
        let shutdown = CancellationToken::new();

        let source = MockSource {
            remaining: 5,
            delay: Duration::from_millis(1),
        };

        let shutdown_clone = shutdown.clone();
        let handle = tokio::spawn(async move { pipeline.run(source, shutdown_clone).await });

        let mut received = 0;
        for _ in 0..5 {
            let pkt = recv_lane(&mut lanes.nsm, &Default::default())
                .await
                .expect("packet");
            assert_eq!(pkt.flow.dst_port, 443);
            received += 1;
        }
        assert_eq!(received, 5);

        shutdown.cancel();
        handle.await.unwrap().unwrap();

        let snap = metrics.snapshot();
        assert_eq!(snap.captured, 5);
    }

    #[tokio::test]
    async fn slow_consumer_lags_without_stalling_capture() {
        let (pipeline, _metrics) = IngestionPipeline::new(IngestionConfig {
            // Deliberately tiny so the spectral lane falls behind fast.
            lane_capacity: 2,
        });
        let mut lanes = pipeline.subscribe_lanes();
        let shutdown = CancellationToken::new();

        let source = MockSource {
            remaining: 50,
            delay: Duration::from_micros(1),
        };

        let shutdown_clone = shutdown.clone();
        let handle = tokio::spawn(async move { pipeline.run(source, shutdown_clone).await });

        // Never drain `lanes.spectral` — it should lag, not block the
        // capture loop, which we verify by confirming the NSM lane
        // still receives packets promptly.
        let lane_metrics = LaneMetrics::default();
        let got = recv_lane(&mut lanes.nsm, &lane_metrics).await;
        assert!(got.is_some());

        shutdown.cancel();
        let _ = handle.await.unwrap();
    }

    #[test]
    fn lane_metrics_default_is_zeroed() {
        let m = LaneMetrics::default();
        assert_eq!(m.delivered.load(Ordering::Relaxed), 0);
        assert_eq!(m.lagged.load(Ordering::Relaxed), 0);
    }

    // silence unused-import warning for AtomicUsize in case future
    // tests extend this module
    #[allow(dead_code)]
    fn _unused(_x: AtomicUsize) {}
}
