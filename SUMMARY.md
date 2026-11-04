# How the system works — summary

This is the plain-language version. For exact formulas see
`MATHEMATICS.md`; for verified commands to actually run it, see
`LIVE_RUN.md`; for the crate-by-crate map, see `README.md`.

## The one-sentence version

Traffic gets watched by several independent, differently-shaped
detectors; when enough of them agree on the same host, a decision engine
either bans it automatically or hands it to a human — never both blindly
and never silently.

## The pipeline, stage by stage

```
raw traffic
    │
    ▼
[1] Ingestion          — captures packets, fans them out to every
                          downstream consumer without copying payloads
    │
    ▼
[2] Hyperedge builder   — groups traffic into behavioral clusters
    (hg_builder)          ("these 30 IPs all hit this one host in 8
                           seconds", "this one IP scanned 40 ports"),
                           then scores each cluster's shape with three
                           independent spectral-graph techniques
    │
    ├── also: [siem-ml]  — a small autoencoder flags individual flows
    │                       whose shape doesn't look like benign traffic
    │
    ├── also: [honeypot] — fake services that have no legitimate reason
    │                       to ever be touched; any touch is
    │                       maximum-confidence by construction
    ▼
[3] Correlation engine  — collects all of the above per host, and only
                           raises an alert once either (a) two
                           independent detectors agree on the same host,
                           or (b) one detector alone is maximally
                           confident (the honeypot case)
    │
    ▼
[4] Review queue        — decides, from the alert's confidence, whether
                           to act immediately or ask a human first
    │
    ├── high confidence ──────► auto-ban
    │
    └── moderate confidence ──► sits in a queue for a person to confirm
                                 or dismiss
    │
    ▼
[5] Quarantine          — the actual containment: the host is blocked at
    (rustwall)             the firewall for a set duration, extended
                           (never shortened) if it's flagged again
```

## Why it's built this way

**No single detector is trusted alone (mostly).** Port scans and DDoS
patterns are common enough, and normal traffic weird enough, that any one
signal alone has too many false positives to auto-ban on. So the system
requires *corroboration* — two different, independently-reasoned
detectors flagging the same host — before it treats a behavioral pattern
as an alert at all. A honeypot touch is the one deliberate exception: a
fake service exists solely to be a tripwire, so touching it isn't
ambiguous the way "many connections to one server" can be.

**Confidence, not just "alert or don't."** Every detector outputs a
number in [0, 1], not just yes/no. That number determines what happens
next: very high confidence (honeytoken touches, essentially always)
skips the human and bans immediately; moderate confidence (the usual
case for corroborated-but-not-certain patterns) goes to a review queue
instead. This is the difference between a system that autonomously
firewalls people off the network on a hunch and one that reserves
autonomous action for the cases that are actually unambiguous.

**Three different mathematical lenses on "does this traffic cluster look
normal."** Rather than one detector, the hyperedge builder runs three
genuinely different spectral techniques over the same behavioral
cluster — think of them as three different ways of asking "is this
shaped like normal traffic":
- *Fiedler eigenvalue* (algebraic connectivity) — how loosely-connected
  the cluster is, compared to its own recent history.
- *Non-backtracking spectral radius* — sensitive to directed, looping
  structure (useful for beaconing/C2-style fan-out) in a way a plain
  adjacency-matrix analysis misses.
- *Ramanujan-bound deviation* — how far the cluster's structure departs
  from the "as well-mixed as a graph with this many edges could
  possibly be" theoretical bound.

Each one catches a different shape of anomaly; using all three (plus the
autoencoder's separate per-flow view, plus the honeypot's ground truth)
is what makes multi-lane corroboration meaningful instead of just voting
on copies of the same signal.

**Belief carries over time, but honestly, not everywhere yet.** A
lightweight HMM tracks a rolling "how suspicious does this host look"
belief per host, updated as new evidence comes in. It's fed into the
scoring as a per-host prior — but the system is explicit in its own docs
that this belief doesn't yet reach into the spectral-graph math itself
(a documented, verified limitation, not an oversight); it currently
reaches the correlation engine as its own independent signal instead.

**Never contain past what the evidence supports.** Ban duration scales
with severity (24h for the most confident alerts, 2h for the next tier,
no ban at all below that), and re-flagging an already-banned host only
ever *extends* the ban, never shortens or resets it in the attacker's
favor.

## What each piece actually is, concretely

| Stage | Crate | What it really does |
|---|---|---|
| Ingestion | `ingestion` | One packet capture, broadcast to multiple consumers with zero-copy fan-out |
| Independent NSM | `nsm` | A separate, standalone network monitor — its own capture loop, its own five detectors (port scan, SYN flood, DNS tunneling, beaconing, signature match), optional in-kernel XDP fast path. Runs as a real child process, bridged in over its actual NDJSON output, not reimplemented in-process |
| Hyperedge builder | `hg_builder` | Turns raw flows into "fan-in", "fan-out", "session cluster", and "honeytoken co-touch" behavioral groups, scores each with real spectral graph math |
| Spectral graph math | `spectral_hypergraph`, `nbsc`, `spectral_jacobi_ds` | The actual linear algebra: Laplacians, Lanczos/Arnoldi eigensolvers, non-backtracking matrices |
| Per-host belief | `neural_hmm` | A small hidden Markov model tracking suspicion over time |
| Per-flow anomaly | `siem-ml` (part of `active-siem`) | An autoencoder trained only on benign traffic; high reconstruction error = doesn't look like anything it's seen before |
| Deception | `honeypot_engine` | Fake SSH/TCP services that log any interaction as maximum-confidence evidence |
| Fusion | `correlation_engine` | Combines every lane's evidence per host (now six lanes: XDP, ClamAV, spectral, hypergraph, honeytoken, nsm), decides when there's enough to raise a verdict |
| Human-in-the-loop | `review_queue` | The confidence/margin logic deciding auto-resolve vs. queue-for-a-person, backed by `ontology_engine`'s real knowledge-graph engine for the audit trail |
| Enforcement | `ips_control_plane` + `rustwall` | Turns an approved verdict into an actual firewall-level ban, with duration by severity |
| Malware scanning | `rust-clam` (`rclam`/`rclamd`) | Signature-based file scanning, separate from the network path |
| Incident memory | `siem_event_index` | Remembers confirmed past incidents so a new alert can be compared to "have we seen this before" |

## What it's honest about not doing (yet)

- It's a research/prototype-grade implementation, not a hardened
  production deployment — no independent security review, no
  large-scale performance testing.
- It hasn't been evaluated against a real intrusion-detection dataset or
  a baseline system (see the thesis-viability discussion earlier in this
  conversation for what that gap looks like).
- The fan-in/fan-out detectors use hard count/time thresholds, so an
  attacker who paces slowly enough under the threshold is invisible to
  them by construction — a known, not-yet-addressed evasion surface.
