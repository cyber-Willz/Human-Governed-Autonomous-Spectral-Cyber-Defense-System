//! End-to-end wiring: `ingestion`'s flow broadcast (and `honeypot_engine`'s
//! honeytoken-touch events) -> [`crate::builder::WindowedHyperedgeBuilder`]
//! -> [`crate::scoring::score_window`] ->
//! `correlation_engine::HypergraphSender`. This is the piece the guide's §3
//! build order describes as "a fourth `broadcast::Receiver` subscriber" on
//! `ingestion`, plumbed all the way through to the SIEM.

use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use correlation_engine::HypergraphSender;
use ingestion::IngestedPacket;
use tokio::sync::{broadcast, mpsc};

use crate::belief::HostBeliefSource;
use crate::builder::{WindowConfig, WindowedHyperedgeBuilder};
use crate::scoring::{score_window, ScoringBaselines};

/// A honeytoken touch, as fed from `honeypot_engine`'s own event sink
/// (not part of `ingestion`'s flow stream -- see
/// [`WindowedHyperedgeBuilder::ingest_honeytoken_touch`]'s docs). This
/// drives the *co-touch* hyperedge family (multiple hosts touching the same
/// token within a campaign window) which is scored like any other window;
/// it's independent of `honeypot_engine` also feeding a single touch
/// straight to `correlation_engine::HoneytokenSender` as an instant
/// `Critical` signal on its own (guide §2) -- both paths run off the same
/// underlying event, for two different purposes.
#[derive(Debug, Clone)]
pub struct HoneytokenTouch {
    pub campaign_id: String,
    pub host: IpAddr,
}

pub struct HypergraphPipeline {
    builder: WindowedHyperedgeBuilder,
    baselines: ScoringBaselines,
    belief_source: Arc<dyn HostBeliefSource>,
    sender: HypergraphSender,
}

impl HypergraphPipeline {
    pub fn new(
        config: WindowConfig,
        belief_source: Arc<dyn HostBeliefSource>,
        sender: HypergraphSender,
    ) -> Self {
        Self {
            builder: WindowedHyperedgeBuilder::new(config),
            baselines: ScoringBaselines::default(),
            belief_source,
            sender,
        }
    }

    /// Drives the pipeline until `flows` and `honeytoken_touches` both
    /// close. `tick_every` should match `correlation_engine`'s own sweep
    /// cadence (both are "close anything that's timed out" passes; keeping
    /// them on the same rhythm means a window's closure latency is bounded
    /// by one cadence, not two compounding ones).
    pub async fn run(
        mut self,
        mut flows: broadcast::Receiver<Arc<IngestedPacket>>,
        mut honeytoken_touches: mpsc::Receiver<HoneytokenTouch>,
        tick_every: Duration,
    ) {
        let mut ticker = tokio::time::interval(tick_every);
        loop {
            tokio::select! {
                biased;
                _ = ticker.tick() => {
                    let closed = self.builder.tick();
                    self.score_and_submit_all(closed);
                }
                touch = honeytoken_touches.recv() => {
                    match touch {
                        Some(t) => {
                            if let Some(closed) = self.builder.ingest_honeytoken_touch(&t.campaign_id, t.host) {
                                self.score_and_submit_all(vec![closed]);
                            }
                        }
                        None => {
                            tracing::info!("hg_builder pipeline: honeytoken touch sender dropped");
                        }
                    }
                }
                pkt = flows.recv() => {
                    match pkt {
                        Ok(pkt) => {
                            let closed = self.builder.ingest(&pkt);
                            self.score_and_submit_all(closed);
                        }
                        Err(broadcast::error::RecvError::Lagged(n)) => {
                            tracing::warn!(dropped = n, "hg_builder's ingestion lane lagged, packets dropped");
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            tracing::info!("hg_builder pipeline: ingestion lane closed, shutting down");
                            return;
                        }
                    }
                }
            }
        }
    }

    fn score_and_submit_all(&mut self, closed: Vec<crate::builder::ClosedWindow>) {
        for window in closed {
            let window_id = window.window_id.clone();
            let correlation_hosts = window.correlation_hosts.clone();
            let kind = window.kind;
            let Some(scored) = score_window(window, self.belief_source.as_ref(), &mut self.baselines) else {
                continue;
            };
            self.sender.submit(
                &correlation_hosts,
                window_id,
                kind,
                scored.fiedler_shift,
                scored.rho_b,
                Some(scored.ramanujan_deviation),
                scored.mean_hmm_belief,
                scored.closed.hyperedge_members.len(),
            );
        }
    }
}
