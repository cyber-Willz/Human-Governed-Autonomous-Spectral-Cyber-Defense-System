//! Runtime: bins -> scores -> `correlation_engine`, scrutiny, feedback.

use std::collections::{HashMap, VecDeque};
use std::net::IpAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use correlation_engine::{ConnectomeSender, FeedbackKind, FeedbackObserver, FeedbackSource, LaneSource};
use ingestion::IngestedPacket;
use llm_bridge::{CognitiveReport, ScrutinyBoard};
use parking_lot::{Mutex, RwLock};

use crate::detector::{ConnectomeDetector, FineTuneOutcome};
use crate::features::{FeatVec, HostBinner};

#[derive(Debug, Clone)]
pub struct NeuroLaneConfig {
    pub alert_threshold: f64,
    /// Threshold for hosts on the scrutiny board. Always <= `alert_threshold`.
    pub elevated_threshold: f64,
    /// Bins of history required before a host is scored.
    pub min_bins: usize,
    pub max_hosts: usize,
    /// Scores at/above this produce a `CognitiveReport` for the uplink.
    pub report_floor: f64,
    pub max_labelled: usize,
    pub min_labelled_to_tune: usize,
    pub anchor_tolerance: f64,
}

impl Default for NeuroLaneConfig {
    fn default() -> Self {
        Self {
            alert_threshold: 0.5,
            elevated_threshold: 0.35,
            min_bins: 10,
            max_hosts: 50_000,
            report_floor: 0.2,
            max_labelled: 256,
            min_labelled_to_tune: 4,
            anchor_tolerance: 0.05,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HostScore {
    pub host: IpAddr,
    pub score: f64,
    pub trend: f64,
    pub bins: usize,
    pub elevated: bool,
    pub emitted: bool,
}

#[derive(Debug, Default)]
pub struct BinReport {
    pub scores: Vec<HostScore>,
    pub reports: Vec<CognitiveReport>,
}

#[derive(Debug, Default)]
pub struct LaneStats {
    pub packets: AtomicU64,
    pub bins_closed: AtomicU64,
    pub hosts_scored: AtomicU64,
    pub emitted: AtomicU64,
    pub labels_accepted: AtomicU64,
    pub labels_ignored: AtomicU64,
}

pub struct NeuroLane {
    detector: RwLock<ConnectomeDetector>,
    binner: Mutex<HostBinner>,
    sender: ConnectomeSender,
    board: ScrutinyBoard,
    cfg: NeuroLaneConfig,
    last: Mutex<HashMap<IpAddr, f64>>,
    labelled: Mutex<VecDeque<(Vec<FeatVec>, f64)>>,
    pub stats: LaneStats,
}

impl NeuroLane {
    pub fn new(detector: ConnectomeDetector, sender: ConnectomeSender, board: ScrutinyBoard, mut cfg: NeuroLaneConfig) -> Self {
        // Scrutiny may only make the lane *more* sensitive.
        cfg.elevated_threshold = cfg.elevated_threshold.min(cfg.alert_threshold);
        let seq_len = detector.cfg.seq_len;
        Self {
            detector: RwLock::new(detector),
            binner: Mutex::new(HostBinner::new(seq_len, cfg.max_hosts)),
            sender,
            board,
            cfg,
            last: Mutex::new(HashMap::new()),
            labelled: Mutex::new(VecDeque::new()),
            stats: LaneStats::default(),
        }
    }

    pub fn observe_packet(&self, p: &IngestedPacket) {
        self.stats.packets.fetch_add(1, Ordering::Relaxed);
        self.binner.lock().observe(p.flow.src_ip, p.flow.dst_ip, p.flow.dst_port, p.payload.len(), p.captured_at);
    }

    /// Test/replay entry point without an `IngestedPacket`.
    pub fn observe_raw(&self, src: IpAddr, dst: IpAddr, dst_port: u16, payload_len: usize, at: SystemTime) {
        self.stats.packets.fetch_add(1, Ordering::Relaxed);
        self.binner.lock().observe(src, dst, dst_port, payload_len, at);
    }

    pub fn sequence(&self, host: IpAddr) -> Option<Vec<FeatVec>> {
        self.binner.lock().sequence(host)
    }

    /// Closes the current time bin, scores every active host with enough
    /// history, emits lane events, and returns uplink reports.
    pub fn close_bin(&self) -> BinReport {
        self.stats.bins_closed.fetch_add(1, Ordering::Relaxed);
        let (active, seqs): (Vec<IpAddr>, Vec<(IpAddr, Vec<FeatVec>)>) = {
            let mut b = self.binner.lock();
            let active = b.close_bin();
            let seqs = active.iter().filter_map(|h| b.sequence(*h).map(|s| (*h, s))).collect();
            (active, seqs)
        };
        let _ = active;
        let det = self.detector.read();
        let mut out = BinReport::default();
        let mut last = self.last.lock();
        for (host, seq) in seqs {
            if seq.len() < self.cfg.min_bins {
                continue;
            }
            let score = det.score(&seq);
            self.stats.hosts_scored.fetch_add(1, Ordering::Relaxed);
            let prev = last.insert(host, score).unwrap_or(score);
            let trend = score - prev;
            let elevated = self.board.is_elevated(host);
            let thr = if elevated { self.cfg.elevated_threshold } else { self.cfg.alert_threshold };
            let emitted = score >= thr;
            if emitted {
                self.sender.submit(host, score, trend, seq.len(), elevated);
                self.stats.emitted.fetch_add(1, Ordering::Relaxed);
            }
            if score >= self.cfg.report_floor {
                out.reports.push(CognitiveReport { host, score, trend, bins: seq.len(), drivers: det.drivers(&seq) });
            }
            out.scores.push(HostScore { host, score, trend, bins: seq.len(), elevated, emitted });
        }
        if last.len() > self.cfg.max_hosts {
            last.clear();
        }
        out
    }

    pub fn pending_labels(&self) -> usize {
        self.labelled.lock().len()
    }

    /// Fine-tunes on the accumulated analyst labels (guarded). Returns
    /// `None` if there are not yet enough labels.
    pub fn apply_feedback(&self, steps: usize, lr: f64, seed: u64) -> Option<FineTuneOutcome> {
        let batch: Vec<(Vec<FeatVec>, f64)> = {
            let l = self.labelled.lock();
            if l.len() < self.cfg.min_labelled_to_tune {
                return None;
            }
            l.iter().cloned().collect()
        };
        let out = self.detector.write().fine_tune_guarded(&batch, steps, lr, self.cfg.anchor_tolerance, seed);
        if out.applied {
            self.labelled.lock().clear();
            self.last.lock().clear();
        }
        Some(out)
    }

    pub fn score_sequence(&self, seq: &[FeatVec]) -> f64 {
        self.detector.read().score(seq)
    }

    pub fn with_detector<R>(&self, f: impl FnOnce(&ConnectomeDetector) -> R) -> R {
        f(&self.detector.read())
    }
}

impl FeedbackObserver for NeuroLane {
    /// Only human decisions and honeytoken-backed auto-resolutions become
    /// labels; the system's own auto-decisions never train it (that would
    /// be a self-reinforcing loop).
    fn on_feedback(&self, host: IpAddr, kind: FeedbackKind, source: FeedbackSource, _lanes: &[LaneSource]) {
        let _ = source; // both permitted sources are accepted; see doc above
        let Some(seq) = self.sequence(host) else {
            self.stats.labels_ignored.fetch_add(1, Ordering::Relaxed);
            return;
        };
        if seq.len() < self.cfg.min_bins {
            self.stats.labels_ignored.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let y = if kind == FeedbackKind::Confirmed { 1.0 } else { 0.0 };
        let mut l = self.labelled.lock();
        if l.len() >= self.cfg.max_labelled {
            l.pop_front();
        }
        l.push_back((seq, y));
        self.stats.labels_accepted.fetch_add(1, Ordering::Relaxed);
    }
}
