//! CLI front-end for the review queue.
//!
//! State persists to a JSON file (default `review_queue_state.json` in the
//! current directory, override with `--state <path>`) so `ingest` and
//! `review` can run as separate invocations, e.g. a pipeline calling
//! `ingest` per-flow and an analyst running `review` interactively.
//!
//! Usage:
//!   review_queue ingest --flow-id F --label L --confidence C
//!                        [--runner-up-label L2] [--runner-up-confidence C2]
//!                        [--ood]
//!   review_queue list [--all]
//!   review_queue claim --flow-id F --reviewer NAME
//!   review_queue review --flow-id F --reviewer NAME --verdict contain|dismiss|escalate --rationale "..."
//!   review_queue sweep --sla-seconds N [--policy fail-safe|fail-secure] [--auto-contain-above C]
//!   review_queue check --flow-id F
//!   review_queue stats
//!
//! Every subcommand exits non-zero on error and prints a one-line message
//! to stderr, so this is safe to drive from a shell pipeline or cron job.

use review_queue::prelude::{
    now, ContainmentDecision, FlowPrediction, IngestOutcome, ReviewQueue, SlaPolicy,
    TriggerConfig, Verdict,
};
use std::collections::HashMap;
use std::process::ExitCode;

/// This CLI's own `Result` alias, distinct from `review_queue::error::Result`,
/// so the two don't collide -- CLI-layer errors are always flattened to a
/// display string here.
type Result<T> = std::result::Result<T, String>;

const DEFAULT_STATE_PATH: &str = "review_queue_state.json";

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<()> {
    let args = Args::new(args);
    let command = args.positional(0).ok_or_else(usage)?;

    let state_path = args
        .flag("--state")
        .unwrap_or_else(|| DEFAULT_STATE_PATH.to_string());

    match command.as_str() {
        "ingest" => cmd_ingest(&args, &state_path),
        "list" => cmd_list(&args, &state_path),
        "claim" => cmd_claim(&args, &state_path),
        "review" => cmd_review(&args, &state_path),
        "sweep" => cmd_sweep(&args, &state_path),
        "check" => cmd_check(&args, &state_path),
        "stats" => cmd_stats(&state_path),
        other => Err(format!("unknown command '{other}'\n\n{}", usage())),
    }
}

fn usage() -> String {
    "usage: review_queue <ingest|list|claim|review|sweep|check|stats> [flags...] \
     (run with a subcommand for its specific flags)"
        .to_string()
}

fn load(state_path: &str) -> Result<ReviewQueue> {
    store_load(state_path).map_err(|e| format!("failed to load state: {e}"))
}

fn store_load(state_path: &str) -> review_queue::error::Result<ReviewQueue> {
    review_queue::store::load_or_new(state_path, TriggerConfig::default(), SlaPolicy::FailSafe)
}

fn save(queue: &ReviewQueue, state_path: &str) -> Result<()> {
    review_queue::store::save(queue, state_path).map_err(|e| format!("failed to save state: {e}"))
}

fn cmd_ingest(args: &Args, state_path: &str) -> Result<()> {
    let flow_id = args.require("--flow-id")?;
    let label = args.require("--label")?;
    let confidence: f64 = args
        .require("--confidence")?
        .parse()
        .map_err(|_| "--confidence must be a number in [0, 1]".to_string())?;
    let runner_up_label = args.flag("--runner-up-label");
    let runner_up_confidence = args
        .flag("--runner-up-confidence")
        .map(|s| s.parse::<f64>())
        .transpose()
        .map_err(|_| "--runner-up-confidence must be a number".to_string())?;
    let is_out_of_distribution = args.has_flag("--ood");

    let queue = load(state_path)?;
    let outcome = queue
        .ingest(FlowPrediction {
            flow_id: flow_id.clone(),
            predicted_label: label,
            confidence,
            runner_up_label,
            runner_up_confidence,
            is_out_of_distribution,
            observed_at: now(),
        })
        .map_err(|e| e.to_string())?;

    match &outcome {
        IngestOutcome::AutoResolved { verdict, decision_id } => {
            println!("flow '{flow_id}' auto-resolved: {verdict} (decision {decision_id})");
        }
        IngestOutcome::QueuedForReview { reason } => {
            println!("flow '{flow_id}' queued for human review: {reason}");
        }
    }

    save(&queue, state_path)
}

fn cmd_list(args: &Args, state_path: &str) -> Result<()> {
    let queue = load(state_path)?;
    let items = if args.has_flag("--all") {
        queue.list_all()
    } else {
        queue.list_pending()
    };

    if items.is_empty() {
        println!("(no items)");
        return Ok(());
    }

    let mut items = items;
    items.sort_by_key(|i| i.enqueued_at);
    for item in items {
        println!(
            "{:<24} label={:<20} confidence={:<6.3} state={:<14} enqueued_at={}",
            item.prediction.flow_id,
            item.prediction.predicted_label,
            item.prediction.confidence,
            item.state.to_string(),
            item.enqueued_at,
        );
    }
    Ok(())
}

fn cmd_claim(args: &Args, state_path: &str) -> Result<()> {
    let flow_id = args.require("--flow-id")?;
    let reviewer = args.require("--reviewer")?;

    let queue = load(state_path)?;
    queue.claim(&flow_id, &reviewer).map_err(|e| e.to_string())?;
    println!("flow '{flow_id}' claimed by '{reviewer}'");
    save(&queue, state_path)
}

fn cmd_review(args: &Args, state_path: &str) -> Result<()> {
    let flow_id = args.require("--flow-id")?;
    let reviewer = args.require("--reviewer")?;
    let verdict_raw = args.require("--verdict")?;
    let rationale = args.require("--rationale")?;

    let verdict = Verdict::parse(&verdict_raw)
        .ok_or_else(|| format!("--verdict must be one of: contain, dismiss, escalate (got '{verdict_raw}')"))?;

    let queue = load(state_path)?;
    let decision = queue
        .record_decision(&flow_id, &reviewer, verdict, &rationale)
        .map_err(|e| e.to_string())?;

    println!(
        "flow '{flow_id}' resolved by {} : {} (decision {})",
        decision.reviewer, decision.verdict, decision.decision_id
    );
    save(&queue, state_path)
}

fn cmd_sweep(args: &Args, state_path: &str) -> Result<()> {
    let sla_seconds: i64 = args
        .require("--sla-seconds")?
        .parse()
        .map_err(|_| "--sla-seconds must be an integer".to_string())?;

    // Note: the queue's SLA *policy* (fail-safe vs fail-secure) is fixed at
    // creation time and persisted, not re-specified per sweep -- a
    // production deployment should not let an ad hoc CLI flag silently
    // change how unreviewed attacks are handled. To change policy,
    // recreate the state file (or extend `store` with an explicit
    // `set-policy` command that itself goes through a reviewed change).
    let queue = load(state_path)?;
    // LoggingAlertSink (the default inside sweep_expired) already emits a
    // structured `error!`/`warn!` per resolution; this CLI additionally
    // prints a human-readable summary and, critically, exits non-zero if
    // any non-benign flow was dismissed unreviewed -- so a cron/systemd
    // timer running this command surfaces the failure instead of
    // succeeding silently while an unreviewed attack call sits closed out.
    let resolutions = queue.sweep_expired(sla_seconds).map_err(|e| e.to_string())?;
    save(&queue, state_path)?;

    if resolutions.is_empty() {
        println!("no items past the {sla_seconds}s SLA");
        return Ok(());
    }

    let mut urgent = 0usize;
    println!("{} item(s) resolved via SLA fallback:", resolutions.len());
    for r in &resolutions {
        let breach = review_queue::alert::SlaBreach {
            prediction: r.prediction.clone(),
            decision: r.decision.clone(),
        };
        let flag = if breach.is_unreviewed_attack_dismissal() {
            urgent += 1;
            " *** UNREVIEWED ATTACK CALL DISMISSED -- NEEDS URGENT FOLLOW-UP ***"
        } else {
            ""
        };
        println!(
            "  {:<20} label={:<20} confidence={:<6.3} verdict={:<20}{}",
            r.prediction.flow_id, r.prediction.predicted_label, r.prediction.confidence, r.decision.verdict, flag
        );
    }

    if urgent > 0 {
        Err(format!(
            "{urgent} flow(s) were non-benign predictions dismissed unreviewed under the SLA \
             fallback -- treat this as an alert, not a routine sweep result"
        ))
    } else {
        Ok(())
    }
}

fn cmd_check(args: &Args, state_path: &str) -> Result<()> {
    let flow_id = args.require("--flow-id")?;
    let queue = load(state_path)?;
    let decision = queue.containment_decision(&flow_id);
    println!("flow '{flow_id}': {decision}");
    match decision {
        ContainmentDecision::Approved { .. } => Ok(()),
        _ => Err("containment not authorized".to_string()),
    }
}

fn cmd_stats(state_path: &str) -> Result<()> {
    let queue = load(state_path)?;
    let s = queue.stats();
    println!(
        "total={} auto_resolved={} pending={} under_review={} resolved={}",
        s.total, s.auto_resolved, s.pending, s.under_review, s.resolved
    );
    Ok(())
}

/// Minimal `--flag value` / `--boolean-flag` parser. No external dependency
/// is pulled in for this deliberately: the CLI surface here is small and
/// stable, and a hand-rolled parser keeps the crate's dependency footprint
/// (and therefore its supply-chain audit surface, which matters for a
/// security tool) to exactly what `ontology_engine` already required.
struct Args {
    positionals: Vec<String>,
    flags: HashMap<String, String>,
    bool_flags: std::collections::HashSet<String>,
}

impl Args {
    fn new(raw: Vec<String>) -> Self {
        let mut positionals = Vec::new();
        let mut flags = HashMap::new();
        let mut bool_flags = std::collections::HashSet::new();

        let mut i = 0;
        while i < raw.len() {
            let arg = &raw[i];
            if let Some(name) = arg.strip_prefix("--") {
                let name = format!("--{name}");
                if let Some(value) = raw.get(i + 1) {
                    if value.starts_with("--") {
                        bool_flags.insert(name);
                        i += 1;
                    } else {
                        flags.insert(name, value.clone());
                        i += 2;
                    }
                } else {
                    bool_flags.insert(name);
                    i += 1;
                }
            } else {
                positionals.push(arg.clone());
                i += 1;
            }
        }

        Self {
            positionals,
            flags,
            bool_flags,
        }
    }

    fn positional(&self, idx: usize) -> Option<String> {
        self.positionals.get(idx).cloned()
    }

    fn flag(&self, name: &str) -> Option<String> {
        self.flags.get(name).cloned()
    }

    fn has_flag(&self, name: &str) -> bool {
        self.bool_flags.contains(name)
    }

    fn require(&self, name: &str) -> Result<String> {
        self.flag(name)
            .ok_or_else(|| format!("missing required flag {name}"))
    }
}
