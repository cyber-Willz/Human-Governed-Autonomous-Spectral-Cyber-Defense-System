use crate::conntrack::{ConnTrack, LookupResult};
use crate::engine::{Engine, Verdict};
use crate::log_gate::{GateResult, LogGate};
use crate::metrics::Metrics;
use crate::packet;
use crate::quarantine::Quarantine;
use crate::reject::Rejecter;
use arc_swap::ArcSwap;
use nfq::{Queue, Verdict as NfqVerdict};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Runs the inline packet loop against a bound NFQUEUE number. Packets reach
/// this queue because an nftables/iptables rule on the host redirects them
/// here (see the README for the exact `nft`/`iptables` incantation) --
/// rustwall is the *decision engine*, not a replacement for the kernel's
/// packet delivery path. This mirrors how real NGFW dataplanes separate
/// "get the packet in front of the inspection engine" (hardware/kernel job)
/// from "decide what happens to it" (the part we own here).
///
/// `engine` is an ArcSwap so a SIGHUP-triggered config reload can hot-swap
/// the ruleset without restarting this loop or touching `conntrack` --
/// existing flows keep flowing through the fast path uninterrupted while new
/// flows immediately see the updated rules.
pub fn run(
    worker_id: u16,
    queue_num: u16,
    engine: Arc<ArcSwap<Engine>>,
    conntrack: Arc<ConnTrack>,
    quarantine: Arc<Quarantine>,
    metrics: Arc<Metrics>,
    log_gate: Arc<LogGate>,
    running: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let mut queue = Queue::open()?;
    queue.bind(queue_num)?;
    // Nonblocking + poll loop instead of a blocking recv(): a blocking recv
    // has no way to observe `running` flipping to false until the *next*
    // packet arrives, which means SIGTERM/SIGINT on an idle queue (or one
    // that's stopped receiving traffic, e.g. because the redirect rule was
    // removed) would leave this thread parked indefinitely and `main`'s
    // `.join()` on it would hang right along with it. A short poll interval
    // keeps shutdown responsive without meaningfully increasing latency for
    // real traffic -- packets that ARE waiting get processed immediately;
    // only a truly idle queue pays the poll interval, and only while idle.
    queue.set_nonblocking(true);
    const POLL_INTERVAL: Duration = Duration::from_millis(100);

    let rejecter = Rejecter::new().ok();
    if rejecter.is_none() {
        warn!(
            worker_id,
            "failed to open raw socket for active reject; reject rules will behave as silent drops \
             (this usually means CAP_NET_RAW is missing -- check the systemd unit / run as root)"
        );
    }
    info!(worker_id, queue_num, "nfqueue worker bound, entering packet loop");

    while running.load(Ordering::Relaxed) {
        let mut msg = match queue.recv() {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // Nothing waiting right now -- this is the expected,
                // frequent case on an idle queue, not an error worth
                // logging. Sleep briefly and loop back to re-check
                // `running` so shutdown stays responsive.
                std::thread::sleep(POLL_INTERVAL);
                continue;
            }
            Err(e) => {
                warn!(worker_id, error = %e, "nfqueue recv error");
                continue;
            }
        };

        let raw = msg.get_payload();
        let parsed = match packet::parse(raw) {
            packet::ParseOutcome::Parsed(p) => p,
            packet::ParseOutcome::UnclassifiableFragment => {
                // Fail closed, same action as Malformed, but counted and
                // logged separately: a steady stream of these means
                // kernel-side defragmentation isn't correctly wired up
                // ahead of the queue redirect (see README), which is a
                // config problem to fix, not evidence of an attack the way
                // a stream of Malformed packets might be.
                metrics.fragment_drops.fetch_add(1, Ordering::Relaxed);
                if matches!(log_gate.check(), GateResult::Emit) {
                    warn!(
                        worker_id,
                        "dropped a non-first IP fragment (no L4 header available to classify); \
                         if this recurs, kernel-side defragmentation is likely not wired up ahead \
                         of the queue redirect -- see README"
                    );
                } else {
                    metrics.logs_suppressed.fetch_add(1, Ordering::Relaxed);
                }
                msg.set_verdict(NfqVerdict::Drop);
                queue.verdict(msg)?;
                continue;
            }
            packet::ParseOutcome::Malformed => {
                // Unparseable / non-IP traffic on this queue: fail closed.
                // A firewall that can't classify a packet should not wave
                // it through -- that's exactly the kind of parser-confusion
                // gap App-ID/Snort spend enormous effort closing.
                metrics.parse_failures.fetch_add(1, Ordering::Relaxed);
                msg.set_verdict(NfqVerdict::Drop);
                queue.verdict(msg)?;
                continue;
            }
        };

        let verdict = if !engine.load().is_trusted(parsed.src_ip) && quarantine.is_active(parsed.src_ip)
        {
            // Quarantine is checked before conntrack and before rule
            // evaluation -- this is the "quick block, evaluated first"
            // pattern from pfSense's floating rules, and functionally what
            // Untangle's behavioral quarantine does: a source that's been
            // auto-banned gets cut off immediately, on every packet,
            // regardless of whether this specific flow would otherwise be
            // allowed by the current ruleset.
            metrics.quarantine_blocks.fetch_add(1, Ordering::Relaxed);
            if matches!(log_gate.check(), GateResult::Emit) {
                info!(
                    worker_id,
                    src = %parsed.src_ip,
                    dst = %parsed.dst_ip,
                    "quarantine active, packet blocked before rule evaluation"
                );
            } else {
                metrics.logs_suppressed.fetch_add(1, Ordering::Relaxed);
            }
            Verdict::Drop
        } else {
            match conntrack.lookup_or_admit(&parsed) {
                LookupResult::Established => {
                    metrics
                        .conntrack_established_hits
                        .fetch_add(1, Ordering::Relaxed);
                    Verdict::Accept
                }
                LookupResult::TableFull => {
                    metrics.conntrack_table_full.fetch_add(1, Ordering::Relaxed);
                    warn!(worker_id, "conntrack table full, dropping new connection attempt");
                    Verdict::Drop
                }
                LookupResult::New => {
                    let current_engine = engine.load();
                    let decision = current_engine.evaluate(&parsed);

                    if decision.log {
                        match log_gate.check() {
                            GateResult::Emit => {
                                info!(
                                    worker_id,
                                    rule = %decision.rule_name,
                                    src = %parsed.src_ip,
                                    dst = %parsed.dst_ip,
                                    src_port = parsed.src_port,
                                    dst_port = parsed.dst_port,
                                    proto = ?parsed.proto,
                                    verdict = ?decision.verdict,
                                    "policy decision"
                                );
                            }
                            GateResult::Suppress => {
                                metrics.logs_suppressed.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }

                    match decision.verdict {
                        Verdict::Accept => conntrack.record_new(&parsed),
                        Verdict::Reject => {
                            if let Some(r) = &rejecter {
                                r.reject(&parsed, raw);
                            }
                        }
                        Verdict::Drop => {}
                    }

                    if let Some(secs) = decision.auto_block_secs {
                        if quarantine.ban(parsed.src_ip, Duration::from_secs(secs)) {
                            metrics.quarantine_bans.fetch_add(1, Ordering::Relaxed);
                            warn!(
                                worker_id,
                                src = %parsed.src_ip,
                                rule = %decision.rule_name,
                                ban_secs = secs,
                                "source auto-quarantined"
                            );
                        } else {
                            // Quarantine table is at capacity and this is a
                            // new IP -- the packet that triggered this was
                            // already dropped/rejected by the rule itself,
                            // this only means future packets from this
                            // source won't get the fast pre-rule block.
                            warn!(
                                worker_id,
                                src = %parsed.src_ip,
                                rule = %decision.rule_name,
                                "quarantine table at capacity, ban not recorded (packet was still dropped/rejected by rule)"
                            );
                        }
                    }

                    decision.verdict
                }
            }
        };

        match verdict {
            Verdict::Accept => metrics.accepted.fetch_add(1, Ordering::Relaxed),
            Verdict::Drop => metrics.dropped.fetch_add(1, Ordering::Relaxed),
            Verdict::Reject => metrics.rejected.fetch_add(1, Ordering::Relaxed),
        };

        // NFQUEUE itself only understands accept/drop -- the "reject" reply
        // (RST/ICMP) was already sent out-of-band above via the raw socket,
        // but we still tell netfilter to drop the original packet so it
        // never actually reaches the destination.
        let nfq_verdict = match verdict {
            Verdict::Accept => NfqVerdict::Accept,
            Verdict::Drop | Verdict::Reject => NfqVerdict::Drop,
        };
        msg.set_verdict(nfq_verdict);
        if let Err(e) = queue.verdict(msg) {
            debug!(worker_id, error = %e, "failed to post verdict (packet may already be gone)");
        }
    }

    info!(worker_id, "nfqueue worker stopped");
    Ok(())
}
