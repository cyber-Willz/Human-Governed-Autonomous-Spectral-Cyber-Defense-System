//! `hg_builder`: guide §3-§5 — windowed hyperedge construction from
//! `nsm`/`ingestion`'s flow stream, HMM-belief vertex weighting, and
//! hypergraph-Laplacian + NBSC `rho_B` + Ramanujan-deviation window
//! scoring, feeding [`correlation_engine`]'s `HypergraphSender`.
//!
//! # Module map
//!
//! - [`builder`]: [`builder::WindowedHyperedgeBuilder`] — the rolling
//!   windowed builder itself. Ingests one flow observation at a time from
//!   `ingestion`'s spectral lane (guide §3 recommends a fourth
//!   `broadcast::Receiver` subscriber, no changes to `ingestion` itself);
//!   returns [`builder::ClosedWindow`]s once a fan-in/fan-out/session/
//!   honeytoken-co-touch window closes, either on a time boundary or a
//!   count threshold, whichever fires first — mirroring the honeypot
//!   daemon's existing connection-timeout debounce pattern rather than
//!   inventing a second scheduler (guide §3).
//! - [`belief`]: the HMM-belief-as-vertex-weight integration point (guide
//!   §4). [`belief::vertex_prior_from_belief`] is the actual few lines of
//!   glue the guide's build order anticipated ("small, since
//!   `belief_score()`-equivalent likely already exists" — it doesn't
//!   *exactly* exist on `neural_hmm::hmm::Belief`, only `argmax()`, so this
//!   turns that into the continuous `[0,1]` suspicion score a hyperedge
//!   vertex weight needs). [`belief::HostBeliefSource`] is the trait
//!   `scoring` depends on, so a live per-host `Belief` cache (fed by
//!   `Hmm::filter_step` against the same feature pipeline that already
//!   drives the autoencoder) can be plugged in without this crate needing
//!   to know about `burn` tensors or feature extraction at all.
//! - [`scoring`]: [`scoring::score_window`] — builds the per-window
//!   [`spectral_hypergraph::hypergraph::SpectralHypergraph`], runs the
//!   matrix-free Fiedler-eigenvalue pass (guide §4: hot path, not the
//!   dense `O(n^3)` one), the NBSC `rho_B` pass for fan-out windows, and
//!   the Ramanujan-deviation feature (guide §5, reusing
//!   `spectral_jacobi_ds::compute_hypergraph_ramanujan_feature` as-is
//!   rather than re-deriving it), against a rolling per-hyperedge-kind
//!   baseline so "shift"/"deviation" mean *relative to this window
//!   family's own recent normal*, not an absolute constant.
//!
//! # Two window sizes, not one (guide §8, "Window sizing")
//!
//! [`builder::WindowConfig`] carries both a `fast_window` (burst — HOIC/
//! LOIC-style near-simultaneous floods) and a `slow_window` (campaign —
//! slow port scans, low-and-slow exfil) per fan-in/fan-out tracker, exactly
//! as the guide's open question recommends rather than picking one and
//! hoping it covers both regimes.

pub mod belief;
pub mod builder;
pub mod pipeline;
pub mod scoring;

pub use belief::{vertex_prior_from_belief, HostBeliefSource, StaticBeliefSource};
pub use builder::{ClosedWindow, WindowConfig, WindowedHyperedgeBuilder};
pub use pipeline::{HoneytokenTouch, HypergraphPipeline};
pub use scoring::{score_window, ScoredWindow, ScoringBaselines};
