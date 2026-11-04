//! Real-world benchmark dataset loading: the Cora citation network
//! (McCallum et al. 2000; the standard Kipf & Welling GCN benchmark).
//!
//! ## Data provenance
//! `data/cora/cora.content` and `data/cora/cora.cites` are the plain-text
//! release used by `tkipf/pygcn` (<https://github.com/tkipf/pygcn>),
//! derived from the original Cora corpus (splits popularized by Sen et al.
//! 2008 and Yang et al. 2016 -- the "Planetoid" line of work). Verified
//! against published statistics at load time by this module's own tests:
//! 2708 papers, 1433-dimensional binary bag-of-words features, 7 classes,
//! class sizes matching the literature (Neural_Networks=818 largest,
//! Rule_Learning=180 smallest).
//!
//! ## Train/val/test split -- a deliberate, documented deviation
//! The standard "Planetoid" split (Yang et al. 2016) ships as Python
//! pickles, not plain text, and is **not** reproduced here (parsing
//! `numpy`/`scipy` pickle protocol from Rust is a separate yak-shave with
//! no payoff for this project's goals). [`stratified_split`] instead builds
//! a split with the same *shape* as the standard semi-supervised protocol
//! -- a small fixed number of labeled examples per class for training, a
//! separate validation set, and a held-out test set -- but it is **not
//! bit-identical** to the published Planetoid split.
//!
//! **Consequence for the thesis writeup:** absolute accuracy numbers
//! produced against this split are *not* directly comparable to published
//! Cora leaderboard numbers (e.g. the oft-cited GCN 81.5% / GAT 83.0%
//! figures use the exact Planetoid split). Only comparisons *within this
//! codebase*, across models trained and evaluated on the identical split
//! produced by the same seed, are valid. State this explicitly in the
//! methodology section rather than presenting numbers next to a
//! leaderboard table.

use crate::graph::Graph;
use crate::spectral::FeatureMatrix;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("malformed content line {line_no}: expected id + features + label, got {n_fields} fields")]
    MalformedContent { line_no: usize, n_fields: usize },
    #[error("inconsistent feature width at line {line_no}: found {found}, expected {expected}")]
    InconsistentFeatureWidth {
        line_no: usize,
        found: usize,
        expected: usize,
    },
    #[error("malformed cites line {line_no}: {raw:?}")]
    MalformedCites { line_no: usize, raw: String },
}

/// A labeled graph dataset: topology (`graph`), per-node features, per-node
/// integer class labels, and train/val/test node masks. Deliberately
/// separate from [`Graph`] itself (which stays pure topology, as used by
/// the synthetic generators in `graph.rs`) so nothing in `spectral.rs`,
/// `gcn.rs`, or `burn_layer.rs` needs to change to consume real data --
/// they already only ever take a `&Graph` plus a `FeatureMatrix`/tensor.
pub struct Dataset {
    pub graph: Graph,
    pub features: FeatureMatrix,
    pub labels: Vec<usize>,
    pub class_names: Vec<String>,
    pub train_mask: Vec<bool>,
    pub val_mask: Vec<bool>,
    pub test_mask: Vec<bool>,
}

impl Dataset {
    pub fn num_classes(&self) -> usize {
        self.class_names.len()
    }

    pub fn train_indices(&self) -> Vec<usize> {
        (0..self.graph.n).filter(|&i| self.train_mask[i]).collect()
    }

    pub fn val_indices(&self) -> Vec<usize> {
        (0..self.graph.n).filter(|&i| self.val_mask[i]).collect()
    }

    pub fn test_indices(&self) -> Vec<usize> {
        (0..self.graph.n).filter(|&i| self.test_mask[i]).collect()
    }

    /// Loads Cora from the bundled `data/cora/{cora.content,cora.cites}`
    /// files (resolved relative to this crate's manifest directory, so it
    /// works regardless of the caller's current working directory), then
    /// applies [`stratified_split`] with `train_per_class=20, val=500,
    /// test=1000` (matching the *shape* of the standard semi-supervised
    /// protocol -- see module docs for why it isn't bit-identical),
    /// seeded by `split_seed`.
    pub fn load_cora_default(split_seed: u64) -> Result<Dataset, DatasetError> {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let content = Path::new(manifest_dir).join("data/cora/cora.content");
        let cites = Path::new(manifest_dir).join("data/cora/cora.cites");
        let mut ds = Self::load_planetoid_style(&content, &cites)?;
        let (train, val, test) = stratified_split(&ds.labels, ds.num_classes(), 20, 500, 1000, split_seed);
        ds.train_mask = train;
        ds.val_mask = val;
        ds.test_mask = test;
        Ok(ds)
    }

    /// Loads Citeseer from the bundled
    /// `data/citeseer/{citeseer.content,citeseer.cites}` files, same
    /// two-file plain-text layout as Cora. Source: the `data/citeseer/`
    /// folder committed directly (not a pickle, not a download link) in
    /// `ialireza13/expanded_gcn` on GitHub, itself derived from the
    /// original LINQS/Sen et al. 2008 release. Verified at load time by
    /// this module's own tests against published statistics: 3327 papers
    /// (3312 with non-zero bag-of-words features, 15 zero-padded — a known
    /// property of this dataset, not a parsing artifact), 3703-dimensional
    /// binary features, 6 classes. Applies the same [`stratified_split`]
    /// shape as Cora (`train_per_class=20, val=500, test=1000`) — see the
    /// module-level docs above for why this is not the literature's exact
    /// Planetoid split.
    pub fn load_citeseer_default(split_seed: u64) -> Result<Dataset, DatasetError> {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let content = Path::new(manifest_dir).join("data/citeseer/citeseer.content");
        let cites = Path::new(manifest_dir).join("data/citeseer/citeseer.cites");
        let mut ds = Self::load_planetoid_style(&content, &cites)?;
        let (train, val, test) = stratified_split(&ds.labels, ds.num_classes(), 20, 500, 1000, split_seed);
        ds.train_mask = train;
        ds.val_mask = val;
        ds.test_mask = test;
        Ok(ds)
    }

    /// Parses a `.content`/`.cites` pair in the standard Planetoid
    /// plain-text layout (Cora as bundled; Citeseer if you obtain it in
    /// the same two-file shape from a compatible source). Returns a
    /// `Dataset` with empty train/val/test masks -- call
    /// [`stratified_split`] separately, or use [`Dataset::load_cora_default`]
    /// for Cora with the default split already applied.
    pub fn load_planetoid_style(content_path: &Path, cites_path: &Path) -> Result<Dataset, DatasetError> {
        let content_raw = std::fs::read_to_string(content_path).map_err(|e| DatasetError::Io {
            path: content_path.display().to_string(),
            source: e,
        })?;
        let cites_raw = std::fs::read_to_string(cites_path).map_err(|e| DatasetError::Io {
            path: cites_path.display().to_string(),
            source: e,
        })?;

        let mut id_to_index: HashMap<String, usize> = HashMap::new();
        let mut feature_rows: Vec<Vec<f64>> = Vec::new();
        let mut class_index: HashMap<String, usize> = HashMap::new();
        let mut class_names: Vec<String> = Vec::new();
        let mut labels: Vec<usize> = Vec::new();
        let mut feature_width: Option<usize> = None;

        for (line_no, line) in content_raw.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 3 {
                return Err(DatasetError::MalformedContent { line_no, n_fields: fields.len() });
            }
            let id = fields[0].to_string();
            let label = fields[fields.len() - 1].to_string();
            let feat_fields = &fields[1..fields.len() - 1];
            let width = feat_fields.len();
            match feature_width {
                None => feature_width = Some(width),
                Some(expected) if expected != width => {
                    return Err(DatasetError::InconsistentFeatureWidth { line_no, found: width, expected });
                }
                _ => {}
            }
            let feats: Vec<f64> = feat_fields.iter().map(|s| s.parse::<f64>().unwrap_or(0.0)).collect();

            let idx = feature_rows.len();
            id_to_index.insert(id, idx);
            feature_rows.push(feats);

            let class_idx = match class_index.get(&label) {
                Some(&ci) => ci,
                None => {
                    let ci = class_names.len();
                    class_names.push(label.clone());
                    class_index.insert(label, ci);
                    ci
                }
            };
            labels.push(class_idx);
        }

        let n = feature_rows.len();
        let f = feature_width.unwrap_or(0);
        let mut flat = Vec::with_capacity(n * f);
        for row in &feature_rows {
            flat.extend_from_slice(row);
        }
        let features = FeatureMatrix::from_rows(n, f, flat);

        let mut graph = Graph::new(n);
        let mut skipped_edges = 0usize;
        for (line_no, line) in cites_raw.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() != 2 {
                return Err(DatasetError::MalformedCites { line_no, raw: line.to_string() });
            }
            let (cited, citing) = (fields[0], fields[1]);
            match (id_to_index.get(cited), id_to_index.get(citing)) {
                (Some(&u), Some(&v)) => graph.add_edge(u, v),
                _ => skipped_edges += 1, // citation references a paper ID absent from .content
            }
        }
        if skipped_edges > 0 {
            eprintln!(
                "note: skipped {skipped_edges} citation edge(s) referencing paper IDs not present in the .content file"
            );
        }

        Ok(Dataset {
            graph,
            features,
            labels,
            class_names,
            train_mask: vec![false; n],
            val_mask: vec![false; n],
            test_mask: vec![false; n],
        })
    }
}

/// Stratified train/val/test split: `train_per_class` nodes per class
/// (clamped to however many exist in that class) go to train; `val_count`
/// and `test_count` nodes are then drawn from the remainder (also clamped
/// if the remainder is smaller than requested). Deterministic given `seed`.
/// See module docs for why this is not bit-identical to the published
/// Planetoid split.
pub fn stratified_split(
    labels: &[usize],
    num_classes: usize,
    train_per_class: usize,
    val_count: usize,
    test_count: usize,
    seed: u64,
) -> (Vec<bool>, Vec<bool>, Vec<bool>) {
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    let n = labels.len();
    let mut rng = StdRng::seed_from_u64(seed);

    let mut by_class: Vec<Vec<usize>> = vec![Vec::new(); num_classes];
    for (i, &l) in labels.iter().enumerate() {
        by_class[l].push(i);
    }
    for v in by_class.iter_mut() {
        v.shuffle(&mut rng);
    }

    let mut train = vec![false; n];
    let mut remaining: Vec<usize> = Vec::new();
    for v in by_class.iter() {
        let take = train_per_class.min(v.len());
        for &idx in &v[..take] {
            train[idx] = true;
        }
        remaining.extend_from_slice(&v[take..]);
    }
    remaining.shuffle(&mut rng);

    let mut val = vec![false; n];
    let mut test = vec![false; n];
    let val_take = val_count.min(remaining.len());
    for &idx in &remaining[..val_take] {
        val[idx] = true;
    }
    let rest_after_val = remaining.len() - val_take;
    let test_take = test_count.min(rest_after_val);
    for &idx in &remaining[val_take..val_take + test_take] {
        test[idx] = true;
    }
    (train, val, test)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn cora_paths() -> (PathBuf, PathBuf) {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        (
            Path::new(manifest_dir).join("data/cora/cora.content"),
            Path::new(manifest_dir).join("data/cora/cora.cites"),
        )
    }

    fn citeseer_paths() -> (PathBuf, PathBuf) {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        (
            Path::new(manifest_dir).join("data/citeseer/citeseer.content"),
            Path::new(manifest_dir).join("data/citeseer/citeseer.cites"),
        )
    }

    #[test]
    fn citeseer_loads_with_published_statistics() {
        let (content, cites) = citeseer_paths();
        let ds = Dataset::load_planetoid_style(&content, &cites).expect("citeseer should parse");
        assert_eq!(ds.graph.n, 3327, "Citeseer has 3327 entries (3312 with real features, 15 zero-padded)");
        assert_eq!(ds.num_classes(), 6, "Citeseer has 6 classes");
        assert_eq!(ds.features.f, 3703, "Citeseer bag-of-words vocabulary is 3703 words");
        assert_eq!(ds.labels.len(), 3327);
        assert_eq!(counts_sum(&ds), 3327);
    }

    #[test]
    fn citeseer_class_distribution_matches_published_counts() {
        let (content, cites) = citeseer_paths();
        let ds = Dataset::load_planetoid_style(&content, &cites).expect("citeseer should parse");
        let mut counts = vec![0usize; ds.num_classes()];
        for &l in &ds.labels {
            counts[l] += 1;
        }
        // Published/observed per-class counts for this release: 264, 590,
        // 668, 701, 596, 508 in some order (label order depends on
        // first-seen order in the file, so check the multiset, not indices).
        let mut sorted = counts.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![264, 508, 590, 596, 668, 701]);
    }

    #[test]
    fn load_citeseer_default_produces_disjoint_masks_with_expected_sizes() {
        let ds = Dataset::load_citeseer_default(0).expect("citeseer should load with default split");
        for i in 0..ds.graph.n {
            let flags = [ds.train_mask[i], ds.val_mask[i], ds.test_mask[i]];
            assert!(flags.iter().filter(|&&b| b).count() <= 1);
        }
        assert_eq!(ds.train_indices().len(), 20 * ds.num_classes());
        assert_eq!(ds.val_indices().len(), 500);
        assert_eq!(ds.test_indices().len(), 1000);
    }

    fn counts_sum(ds: &Dataset) -> usize {
        let mut counts = vec![0usize; ds.num_classes()];
        for &l in &ds.labels {
            counts[l] += 1;
        }
        counts.iter().sum()
    }

    #[test]
    fn cora_loads_with_published_statistics() {
        let (content, cites) = cora_paths();
        let ds = Dataset::load_planetoid_style(&content, &cites).expect("cora should parse");
        assert_eq!(ds.graph.n, 2708, "Cora has 2708 papers");
        assert_eq!(ds.num_classes(), 7, "Cora has 7 classes");
        assert_eq!(ds.features.f, 1433, "Cora bag-of-words vocabulary is 1433 words");
        assert_eq!(ds.labels.len(), 2708);
        // add_edge dedupes symmetrized (cited,citing)/(citing,cited) pairs
        // down to one undirected edge, so m() should be positive and at
        // most the raw .cites line count (5429).
        assert!(ds.graph.m() > 0 && ds.graph.m() <= 5429);
    }

    #[test]
    fn cora_class_distribution_matches_published_counts() {
        let (content, cites) = cora_paths();
        let ds = Dataset::load_planetoid_style(&content, &cites).expect("cora should parse");
        let mut counts = vec![0usize; ds.num_classes()];
        for &l in &ds.labels {
            counts[l] += 1;
        }
        assert_eq!(counts.iter().sum::<usize>(), 2708);
        // Don't hardcode which index is which class (depends on first-seen
        // order in the file); just check the max/min counts match the
        // published per-class sizes (Neural_Networks=818, Rule_Learning=180).
        assert_eq!(*counts.iter().max().unwrap(), 818);
        assert_eq!(*counts.iter().min().unwrap(), 180);
    }

    #[test]
    fn stratified_split_has_no_overlap_and_respects_counts() {
        let labels: Vec<usize> = (0..300).map(|i| i % 3).collect();
        let (train, val, test) = stratified_split(&labels, 3, 10, 30, 60, 42);
        for i in 0..labels.len() {
            let flags = [train[i], val[i], test[i]];
            assert!(flags.iter().filter(|&&b| b).count() <= 1, "node {i} in multiple splits");
        }
        assert_eq!(train.iter().filter(|&&b| b).count(), 30); // 10 per class * 3 classes
        assert_eq!(val.iter().filter(|&&b| b).count(), 30);
        assert_eq!(test.iter().filter(|&&b| b).count(), 60);
    }

    #[test]
    fn load_cora_default_produces_disjoint_masks_with_expected_sizes() {
        let ds = Dataset::load_cora_default(0).expect("cora should load with default split");
        for i in 0..ds.graph.n {
            let flags = [ds.train_mask[i], ds.val_mask[i], ds.test_mask[i]];
            assert!(flags.iter().filter(|&&b| b).count() <= 1);
        }
        assert_eq!(ds.train_indices().len(), 20 * ds.num_classes());
        assert_eq!(ds.val_indices().len(), 500);
        assert_eq!(ds.test_indices().len(), 1000);
    }
}
