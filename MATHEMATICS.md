# The mathematics of the unified NDR/IPS/SIEM pipeline

Every formula below is transcribed from the actual running code (file and
function named at each step), not derived from the guide's prose. Where a
symbol's meaning depends on runtime state (an `IpAddr`, a window's members)
that's noted inline. Scalars are `f64` unless stated otherwise.

## 1. Windowing: when a hyperedge closes

`hg_builder::builder::WindowedHyperedgeBuilder` groups packets into
trackers keyed by `(dst_ip, dst_port)` (fan-in) or `src_ip` (fan-out), each
tracker running a **fast** and a **slow** copy in parallel. For a tracker
with distinct-member set $S$, first-seen time $t_0$, window length $w$, and
count threshold $\tau$, a window closes the instant either condition holds:

$$
|S| \ge \tau \quad\text{or}\quad (t_\text{now} - t_0) \ge w
$$

Defaults: $w_\text{fast}=10\text{s}$, $w_\text{slow}=300\text{s}$,
$\tau_\text{fan-in}=50$, $\tau_\text{fan-out}=30$. A closed window with
$|S| < 2$ is dropped as degenerate (a hyperedge needs $\ge 2$ members).

The window's **weight** (later the hyperedge weight, §3) is:

$$
\text{weight} = \begin{cases}
\max(\text{byte\_volume},\ |S|) & \text{fan-in} \\
\text{distinct\_ports} & \text{fan-out}
\end{cases}
$$

## 2. HMM belief → vertex prior

`neural_hmm::transition::TransitionMatrix::predict` runs one Markov
predict step over an $n$-state belief vector $b$ with transition matrix
$P$:

$$
\text{predicted}_j = \sum_{i=1}^{n} b_i \, P_{ij}
$$

`neural_hmm::hmm::Hmm::update_with_emissions` then applies a Bayes update
against per-state emission likelihoods $e$ from the observed features:

$$
\text{posterior}_s = \frac{\text{predicted}_s \cdot e_s}{\sum_{k=1}^n \text{predicted}_k \cdot e_k}
$$

(if the normalizer is $\le$ a small epsilon, this is treated as belief
collapse and rejected rather than silently producing NaNs).

`hg_builder::belief::vertex_prior_from_belief` turns a belief vector into
the single suspicion scalar in $[0,1]$ that `hg_builder` actually
consumes — total probability mass on every state *except* the configured
benign state $s_0$:

$$
\text{prior}(b) = \operatorname{clamp}_{[0,1]}\bigl(1 - b_{s_0}\bigr)
$$

`hg_builder::scoring::score_window` then floors this at a small epsilon
before using it as a hypergraph vertex weight (§3 requires strictly
positive vertex weight):

$$
\text{vertex\_weight} = \max(\text{prior}, 0.01)
$$

and separately tracks the **unfloored** mean across a window's $m$
members as a first-class fused feature:

$$
\text{mean\_hmm\_belief} = \frac{1}{m}\sum_{i=1}^m \text{prior}(b_i)
$$

**Verified limitation:** the vertex weight above is stored on the graph
via `add_weighted_vertex`, but per §3 below, the normalized Laplacian's
vertex degree $D_v$ comes from *incident hyperedge weight*, not from a
vertex's own intrinsic weight — so this quantity does not currently
perturb the Fiedler eigenvalue (§4). Only `mean_hmm_belief` reaches the
correlation engine as a live signal today; see
`hg_builder/src/scoring.rs`'s doc comment on `score_window` for the full
account and `spectral_hypergraph/src/hypergraph.rs`'s `vertex_degree` doc
comment for why the library is built this way.

## 3. The normalized hypergraph Laplacian

`spectral_hypergraph::laplacian` implements the Zhou–Huang–Schölkopf
(NeurIPS 2006) normalized hypergraph Laplacian:

$$
\Delta = I - D_v^{-1/2} H W D_e^{-1} H^\top D_v^{-1/2}
$$

where $H \in \{0,1\}^{|V|\times|E|}$ is the incidence matrix, $W$ is the
diagonal hyperedge-weight matrix (the window weight from §1), $D_v$ is the
diagonal weighted vertex-degree matrix, and $D_e$ is the diagonal
hyperedge-cardinality matrix. $\Delta$ is symmetric PSD with smallest
eigenvalue $0$, achieved by $D_v^{1/2}\mathbf{1}$ — exactly mirroring the
ordinary normalized graph Laplacian it generalizes.

`spectral_hypergraph::laplacian::HypergraphOperator` never forms $\Delta$
as a dense matrix; it applies it matrix-free in $O(\text{nnz}(H))$ per
matvec, which is what `spectral::lanczos_smallest` (Lanczos with full
reorthogonalization) runs against to get the $k$ algebraically-smallest
eigenpairs without ever materializing an $n\times n$ matrix.

### The Fiedler eigenvalue

`hg_builder::scoring::score_window` runs Lanczos with $k=2$. The smallest
eigenvalue is always $0$ (the trivial eigenvector); the **Fiedler
eigenvalue** — algebraic connectivity — is the second-smallest:

$$
\lambda_2(\Delta) = \text{fiedler\_eigenvalue}
$$

Lower $\lambda_2$ means a more loosely-connected (i.e. more
anomaly-suggestive, per the guide's "sharp drop relative to normal")
window structure.

### Rolling baseline and shift

`hg_builder::scoring::ScoringBaselines` keeps one exponential moving
average per `HyperedgeKind` $k$ (destination fan-in, source fan-out,
session-cluster, honeytoken co-touch), smoothing factor $\alpha=0.1$:

$$
\mu_k \leftarrow (1-\alpha)\,\mu_k + \alpha\,\lambda_2
$$

and reports the **shift** used downstream as everything else's anomaly
signal, computed *before* the update above (using the pre-update $\mu_k$):

$$
\text{fiedler\_shift} = \max(\mu_k - \lambda_2,\ 0)
$$

The first window of a given kind has no baseline yet, so it seeds
$\mu_k \leftarrow \lambda_2$ and reports a shift of exactly $0$ — a lone
data point can't be anomalous relative to itself.

## 4. NBSC's $\rho_B$: non-backtracking spectral radius

Computed only for `SourceFanOut` windows. `hg_builder::scoring` first
clique-expands the window's hypergraph to a plain graph via
`nbsc::hypergraph_bridge::clique_expand`, then runs
`nbsc::spectral::estimate_spectral_radius`.

The **Hashimoto (non-backtracking) matrix** $B$ of a graph is indexed by
directed edges; $B_{(u\to v),(x\to y)} = 1$ iff $v=x$ and $y\neq u$ (no
immediate backtrack). `nbsc::spectral` uses the standard Bass reduction:
rather than forming $B$ (size $2|E|\times 2|E|$) directly, it works with
its eigenvalue-equivalent linearization $M$ (size $2n\times 2n$, $n=|V|$):

$$
M = \begin{pmatrix} A & I - D \\ I_n & 0 \end{pmatrix}
$$

where $A$ is the adjacency matrix and $D$ the diagonal degree matrix.
$M$'s eigenvalues are exactly the roots of the Ihara-zeta quadratic
eigenvalue problem — the non-trivial part of $B$'s spectrum (every
Hashimoto eigenvalue except the $2(m-n)$ trivial ones sitting at $\pm 1$).

`HashimotoLinearization` applies $M$ matrix-free (one sparse adjacency
matvec plus two elementwise passes per application — no $A$, $D$, or $M$
is ever formed). `krylov_ds::Arnoldi` (32-dimensional Krylov subspace,
deterministic seeded start vector) estimates the extremal Ritz values
$\{\mu_i\}$ of $M$, and:

$$
\rho_B = \max_i |\mu_i| = \max_i \sqrt{\operatorname{Re}(\mu_i)^2 + \operatorname{Im}(\mu_i)^2}
$$

A large $\rho_B$ relative to the graph's degree structure is the
non-backtracking-walk analogue of "this fan-out graph has directed/cyclic
structure" (beaconing, C2 loop patterns) that an ordinary adjacency
spectrum can miss.

## 5. Ramanujan-bound deviation

`spectral_jacobi_ds::compute_hypergraph_ramanujan_feature` clique-expands
the window's hypergraph (`spectral_hypergraph::laplacian::clique_expansion_adjacency`)
and reuses the plain-graph Ramanujan feature machinery. For a graph with
vertex degrees $\{d_i\}$:

$$
\bar d = \frac{1}{n}\sum_i d_i, \qquad
d_\text{for-bound} = \begin{cases} d & \text{if every } d_i = d \text{ (regular)} \\ \bar d & \text{otherwise (heuristic)} \end{cases}
$$

$$
q = \max(d_\text{for-bound} - 1,\ 0), \qquad
\text{bound} = 2\sqrt{q}
$$

This $2\sqrt{d-1}$ bound is the Alon–Boppana / Ramanujan-graph theorem on
the second-largest eigenvalue magnitude of a $d$-regular graph's
*adjacency* spectrum. `spectral_hypergraph::spectral::dense_eigen` gets
the adjacency spectrum $\{\mu_i\}$ of the clique-expanded window graph;
excluding the trivial eigenvalue at $d_\text{for-bound}$:

$$
\mu_\text{max-nontrivial} = \max_{i:\ |\mu_i - d_\text{for-bound}| > 10^{-6}} |\mu_i|
$$

$$
\text{ramanujan\_deviation} = \max(\mu_\text{max-nontrivial} - \text{bound},\ 0)
$$

Since clique expansion is a lossy reduction, the hypergraph version always
sets `is_heuristic = true` — this is a validation/anomaly *feature*, not a
claim that the window graph is a genuine Ramanujan graph.

## 6. The autoencoder's reconstruction error (spectral lane)

`active-siem/crates/siem-ml`'s autoencoder is a 4-layer MLP
(`enc1 → relu → enc2 → relu → dec1 → relu → dec2`) trained only on benign
flow features $x\in\mathbb{R}^d$. Its anomaly score is per-sample mean
squared reconstruction error:

$$
\hat x = \text{Autoencoder}(x), \qquad
\text{anomaly\_score}(x) = \frac{1}{d}\sum_{j=1}^d (\hat x_j - x_j)^2
$$

A flow is flagged anomalous when $\text{anomaly\_score}(x) > \theta$,
where $\theta$ is calibrated as (per the module docs) the 99th percentile
of reconstruction error on held-out benign traffic.

## 7. Turning scores into severity and confidence

Both `correlation_engine::SpectralSender::submit` and
`HypergraphSender::submit` map a scalar into the same four-level severity
ladder:

$$
\text{severity}(c) = \begin{cases}
\text{Critical} & c \ge 0.90 \\
\text{High} & 0.70 \le c < 0.90 \\
\text{Medium} & 0.40 \le c < 0.70 \\
\text{Low} & c < 0.40
\end{cases}
$$

**Spectral lane:** $c = \operatorname{clamp}_{[0,1]}(\text{anomaly\_score})$
directly (§6's MSE, already roughly bounded).

**Hypergraph lane:** the three §3–§5 features are squashed through one
logistic-style map so they share a scale, taking whichever of
`fiedler_shift` or `ramanujan_deviation` is larger (both are
"larger = more anomalous, unbounded above"):

$$
\text{raw} = \max(\text{fiedler\_shift},\ \text{ramanujan\_deviation}), \qquad
c = \operatorname{clamp}_{[0,1]}\!\left(\frac{\text{raw}}{\text{raw}+1}\right)
$$

**Honeytoken lane:** fixed by construction, no scoring needed —
$c = 1.0$, severity $=$ Critical, always. A honeytoken touch means
something connected to infrastructure that has no legitimate reason to be
touched at all.

**`nsm` lane (added when `nsm` was integrated):** `nsm` is a separate,
independent NSM with its own five detectors (port scan, SYN flood, DNS
tunneling, beaconing, signature match), each already assigning its own
severity on a *five*-level scale
($\text{Info} < \text{Low} < \text{Medium} < \text{High} < \text{Critical}$).
`correlation_engine::NsmSender::map_severity` maps this straight onto the
four-level ladder above, choosing each confidence value to sit just
inside the matching bucket — so recomputing severity from the reported
confidence via this section's own $\text{severity}(c)$ cutoffs reproduces
the *same* severity, not a different one:

$$
(\text{severity}, c) = \begin{cases}
(\text{Critical},\ 0.92) & \text{nsm Critical} \\
(\text{High},\ 0.75) & \text{nsm High} \\
(\text{Medium},\ 0.50) & \text{nsm Medium} \\
(\text{Low},\ 0.25) & \text{nsm Low} \\
(\text{Low},\ 0.10) & \text{nsm Info}
\end{cases}
$$

Unlike the spectral/hypergraph lanes, nothing here is recomputed from a
raw anomaly score — `nsm`'s detectors already did their own scoring
(e.g. port scan: $\ge 20$ distinct ports or $\ge 15$ distinct hosts from
one source within 10s; SYN flood: $\ge 100$ SYNs at one destination
within 5s, escalating to Critical past 20 distinct sources; beaconing:
coefficient of variation $\le 0.15$ over $\ge 6$ inter-arrival samples to
the same destination — see `nsm/README.md`'s "Detectors, briefly" for the
complete list), this mapping only reconciles that decision onto the
shared severity scale so it fuses correctly with everything else in §8.

## 8. Multi-lane fusion

`correlation_engine::HostRecord` accumulates every `CorrelationEvent` for
a host. Two quantities matter for what happens next:

$$
\bar c_\text{host} = \frac{1}{|E|}\sum_{e \in E} c_e
\qquad\text{(mean\_confidence, over all accumulated evidence } E\text{)}
$$

$$
\text{sources}(host) = \bigl|\{\, \text{lane}(e) : e \in E \,\}\bigr|
\qquad\text{(count of \emph{distinct} lanes, not events)}
$$

`CorrelationEngine::maybe_emit` fires a `CorrelationVerdict` — carrying
$\bar c_\text{host}$ as its confidence — iff:

$$
\text{sources}(host) \ge 2
\quad\lor\quad
\max_{e\in E}\text{severity}(e) = \text{Critical}
$$

i.e. either two independent lanes corroborate on the same host
(`MultiLaneCorroboration`), or one lane alone already screams Critical
(`SingleLaneCritical` — always true for a honeytoken touch). A lone
moderate-confidence single-lane signal is deliberately held back.

## 9. The review-queue trigger

`review_queue::trigger::ReviewTrigger::evaluate` decides auto-resolve vs.
queue for a `FlowPrediction` with confidence $c$, predicted label $\ell$,
and (optional) runner-up confidence $c'$. Margin, when a runner-up exists:

$$
\text{margin} = c - c'
$$

Evaluated in order, first match wins (defaults shown):

$$
\text{queue if } \begin{cases}
\text{is\_out\_of\_distribution} & \Rightarrow \texttt{NovelPattern} \\
\ell \ne \text{benign} \ \land\ c < 0.90 & \Rightarrow \texttt{LowConfidenceAttack} \\
\ell = \text{benign} \ \land\ c < 0.80 & \Rightarrow \texttt{LowConfidenceAttack} \\
\text{margin exists} \ \land\ \text{margin} < 0.15 & \Rightarrow \texttt{NarrowMargin} \\
\text{otherwise} & \Rightarrow \text{auto-resolve}
\end{cases}
$$

`ips_control_plane` builds each `FlowPrediction` with
`runner_up_confidence = None` (no second-best class is available from a
`CorrelationVerdict`), so in practice only the first three rules fire; the
honeytoken lane's $c=1.0 \ge 0.90$ auto-resolves immediately, while a
two-lane-corroborated moderate verdict (§8, typically $c \approx 0.5$–$0.7$)
queues under `LowConfidenceAttack`.

## 10. Containment: ban duration and extension

`ips_control_plane::ban_duration_for` maps the verdict's severity straight
to a containment window, with no containment action at all below High:

$$
\text{duration}(\text{severity}) = \begin{cases}
86400\text{s} = 24\text{h} & \text{Critical} \\
7200\text{s} = 2\text{h} & \text{High} \\
\text{none (no ban)} & \text{Medium},\ \text{Low}
\end{cases}
$$

`rustwall::quarantine::Quarantine::ban` computes a new expiry and only
ever *extends* an existing ban, never shortens it:

$$
t_\text{new} = t_\text{now} + \text{duration}, \qquad
t_\text{expiry} \leftarrow \max(t_\text{expiry},\ t_\text{new})
$$

(subject to a capacity guard: a *new* IP is rejected outright once the
quarantine table is at `max_entries`; re-banning an IP already in the
table always succeeds regardless of capacity). A host is considered
actively banned while $t_\text{now} < t_\text{expiry}$.

## End-to-end, symbolically

Collapsing every stage above into one expression for a single host $h$
observed over an evidence set $E$ accumulated from the hypergraph,
spectral, and honeytoken lanes (note: `ips_control_plane::classify`
never labels a verdict "benign" — every verdict is a scored attack
class — so only the attack branch of §9's trigger is live in this
pipeline):

$$
\text{banned}(h) \iff
\Bigl[\text{sources}(h)\ge 2 \ \lor\ \max_{e\in E}\text{severity}(e)=\text{Critical}\Bigr]
\ \land\
\bar c_h \ge 0.90
\ \land\
\text{severity}(\bar c_h) \in \{\text{Critical},\ \text{High}\}
$$

with $\bar c_h$ and each lane's contributing score built from §§1–7, and
the actual containment window given by §10. In practice the only path
that satisfies all three clauses at once is the honeytoken lane
($\bar c_h = 1.0$, Critical, single-lane-critical) — everything reaching
$\bar c_h \ge 0.90$ via multi-lane corroboration alone, with the fusion
weights currently in play, queues under §9 rather than clearing the
auto-resolve floor; see `LIVE_RUN.md` for a concrete run showing exactly
that split.
