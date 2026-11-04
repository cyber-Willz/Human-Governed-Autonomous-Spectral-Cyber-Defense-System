//! Structural connectome data model, loaders, a synthetic human-like generator
//! and null models (degree-preserving rewiring, density-matched random graph).
//!
//! A `Connectome` is an undirected, weighted region-by-region graph with fibre
//! lengths in millimetres. Diffusion tractography cannot resolve direction, so
//! matrices are symmetrised on load.

use crate::rng::Rng;
use std::collections::HashSet;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Region {
    pub name: String,
    /// 'L', 'R' or '?'
    pub hemi: char,
    /// Coarse functional/anatomical system label (e.g. "Visual", "Somatomotor").
    pub system: String,
    pub pos: [f64; 3],
}

#[derive(Clone, Debug)]
pub struct Connectome {
    pub regions: Vec<Region>,
    w: Vec<f64>,
    len: Vec<f64>,
    pub source: String,
}

fn euclid(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn components(w: &[f64], n: usize) -> Vec<usize> {
    let mut label = vec![usize::MAX; n];
    let mut c = 0;
    for s in 0..n {
        if label[s] != usize::MAX {
            continue;
        }
        let mut stack = vec![s];
        label[s] = c;
        while let Some(u) = stack.pop() {
            for v in 0..n {
                if w[u * n + v] > 0.0 && label[v] == usize::MAX {
                    label[v] = c;
                    stack.push(v);
                }
            }
        }
        c += 1;
    }
    label
}

/// Coarse grouping of Desikan-Killiany gyri by anatomical lobe/system.
/// This is a name-based heuristic, not a functional parcellation.
pub fn desikan_system(code: &str) -> &'static str {
    match code.to_uppercase().as_str() {
        "CUN" | "LOCC" | "LING" | "PCAL" => "Visual",
        "PREC" | "PSTC" | "PARC" => "Somatomotor",
        "CMF" | "RMF" | "SF" | "PORB" | "PTRI" | "POPE" | "LOF" | "MOF" | "FP" => "Frontal",
        "IP" | "SP" | "SMAR" => "Parietal",
        "BSTS" | "ENT" | "FUS" | "IT" | "MT" | "PARH" | "ST" | "TP" | "TT" => "Temporal",
        "CAC" | "RAC" | "PC" | "ISTC" | "PCUN" => "Cingulate-medial",
        _ => "Unassigned",
    }
}

/// Coarse system label for a Glasser HCP-MMP1.0 area name such as `L_V1_ROI`,
/// `R_3b_ROI` or plain `V1`. Only unambiguous visual, somatomotor and auditory
/// areas are assigned; everything else is "Unassigned". Name-based heuristic,
/// not a functional parcellation.
pub fn glasser_system(name: &str) -> &'static str {
    let mut a = name.trim();
    if a.len() > 2 && (a.starts_with("L_") || a.starts_with("R_")) {
        a = &a[2..];
    }
    let a = a.strip_suffix("_ROI").unwrap_or(a);
    match a {
        "V1" | "V2" | "V3" | "V4" | "V3A" | "V3B" | "V3CD" | "V4t" | "V6" | "V6A" | "V7" | "V8" | "LO1"
        | "LO2" | "LO3" | "PIT" | "MT" | "MST" | "FST" | "VMV1" | "VMV2" | "VMV3" | "VVC" | "DVT" | "PH" => "Visual",
        "4" | "3a" | "3b" | "1" | "2" | "6d" | "6mp" | "6v" | "6ma" => "Somatomotor",
        "A1" | "LBelt" | "MBelt" | "PBelt" | "RI" | "A4" | "A5" | "TA2" => "Auditory",
        _ => "Unassigned",
    }
}

impl Connectome {
    /// Validate, symmetrise, zero the diagonal and fill missing fibre lengths.
    pub fn new(
        regions: Vec<Region>,
        mut w: Vec<f64>,
        mut len: Vec<f64>,
        source: impl Into<String>,
    ) -> Result<Self, String> {
        let n = regions.len();
        if n < 2 {
            return Err("a connectome needs at least 2 regions".into());
        }
        if w.len() != n * n || len.len() != n * n {
            return Err(format!("weight/length matrices must be {n}x{n}"));
        }
        for v in w.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
            if *v < 0.0 {
                return Err("negative connection weight found".into());
            }
        }
        for v in len.iter_mut() {
            if !v.is_finite() || *v < 0.0 {
                *v = 0.0;
            }
        }
        for i in 0..n {
            w[i * n + i] = 0.0;
            len[i * n + i] = 0.0;
            for j in (i + 1)..n {
                let a = 0.5 * (w[i * n + j] + w[j * n + i]);
                w[i * n + j] = a;
                w[j * n + i] = a;
                let (l1, l2) = (len[i * n + j], len[j * n + i]);
                let l = if l1 > 0.0 && l2 > 0.0 { 0.5 * (l1 + l2) } else { l1.max(l2) };
                len[i * n + j] = l;
                len[j * n + i] = l;
            }
        }
        // Fill lengths that are missing (typically where no tract was found) from
        // geometry, so null models and dense controls still get conduction delays.
        let have_pos = regions.iter().any(|r| r.pos != [0.0; 3]);
        let mut ratios = Vec::new();
        let mut known = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let l = len[i * n + j];
                if l > 0.0 {
                    known.push(l);
                    if have_pos {
                        let d = euclid(&regions[i].pos, &regions[j].pos);
                        if d > 1e-9 {
                            ratios.push(l / d);
                        }
                    }
                }
            }
        }
        let mean_len = if known.is_empty() {
            50.0
        } else {
            known.iter().sum::<f64>() / known.len() as f64
        };
        let ratio = if ratios.is_empty() { None } else { Some(median(&mut ratios)) };
        for i in 0..n {
            for j in (i + 1)..n {
                if len[i * n + j] <= 0.0 {
                    let l = match ratio {
                        Some(r) => (r * euclid(&regions[i].pos, &regions[j].pos)).max(1.0),
                        None => mean_len,
                    };
                    len[i * n + j] = l;
                    len[j * n + i] = l;
                }
            }
        }
        Ok(Connectome { regions, w, len, source: source.into() })
    }

    pub fn n(&self) -> usize {
        self.regions.len()
    }
    pub fn weight(&self, i: usize, j: usize) -> f64 {
        self.w[i * self.n() + j]
    }
    pub fn length(&self, i: usize, j: usize) -> f64 {
        self.len[i * self.n() + j]
    }
    pub fn weights(&self) -> &[f64] {
        &self.w
    }
    pub fn lengths(&self) -> &[f64] {
        &self.len
    }

    pub fn degree(&self, i: usize) -> usize {
        let n = self.n();
        (0..n).filter(|&j| self.w[i * n + j] > 0.0).count()
    }
    pub fn strength(&self, i: usize) -> f64 {
        let n = self.n();
        self.w[i * n..(i + 1) * n].iter().sum()
    }
    pub fn num_edges(&self) -> usize {
        let n = self.n();
        let mut m = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                if self.w[i * n + j] > 0.0 {
                    m += 1;
                }
            }
        }
        m
    }
    pub fn density(&self) -> f64 {
        let n = self.n() as f64;
        self.num_edges() as f64 / (n * (n - 1.0) / 2.0)
    }
    pub fn is_connected(&self) -> bool {
        components(&self.w, self.n()).iter().all(|&c| c == 0)
    }

    pub fn interhemispheric_fraction(&self) -> f64 {
        let n = self.n();
        let (mut cross, mut total) = (0.0, 0.0);
        for i in 0..n {
            for j in (i + 1)..n {
                let w = self.w[i * n + j];
                total += w;
                let (a, b) = (self.regions[i].hemi, self.regions[j].hemi);
                if a != '?' && b != '?' && a != b {
                    cross += w;
                }
            }
        }
        if total > 0.0 { cross / total } else { 0.0 }
    }

    pub fn summary(&self) -> String {
        let n = self.n();
        let degs: Vec<usize> = (0..n).map(|i| self.degree(i)).collect();
        let mean_deg = degs.iter().sum::<usize>() as f64 / n as f64;
        let max_deg = degs.iter().copied().max().unwrap_or(0);
        let mean_strength = (0..n).map(|i| self.strength(i)).sum::<f64>() / n as f64;
        let mut lens = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if self.w[i * n + j] > 0.0 {
                    lens.push(self.len[i * n + j]);
                }
            }
        }
        let mean_len = lens.iter().sum::<f64>() / lens.len().max(1) as f64;
        let max_len = lens.iter().cloned().fold(0.0, f64::max);
        format!(
            "source: {}\nregions: {}   undirected edges: {}   density: {:.3}   connected: {}\n\
             mean degree: {:.1}   max degree: {}   mean strength: {:.3}\n\
             interhemispheric weight fraction: {:.3}   edge length (mm): mean {:.1}, max {:.1}",
            self.source,
            n,
            self.num_edges(),
            self.density(),
            self.is_connected(),
            mean_deg,
            max_deg,
            mean_strength,
            self.interhemispheric_fraction(),
            mean_len,
            max_len
        )
    }

    /// Rescale so that the mean node strength equals `target`.
    pub fn scaled_to_mean_strength(&self, target: f64) -> Connectome {
        let n = self.n();
        let mean = self.w.iter().sum::<f64>() / n as f64;
        let f = if mean > 0.0 { target / mean } else { 1.0 };
        let mut c = self.clone();
        for v in c.w.iter_mut() {
            *v *= f;
        }
        c
    }

    /// Compress the heavy-tailed streamline weights: w -> ln(1 + w / median(w>0)).
    pub fn log_compressed(&self) -> Connectome {
        let mut pos: Vec<f64> = self.w.iter().copied().filter(|&v| v > 0.0).collect();
        if pos.is_empty() {
            return self.clone();
        }
        let m = median(&mut pos);
        let mut c = self.clone();
        for v in c.w.iter_mut() {
            if *v > 0.0 {
                *v = (1.0 + *v / m).ln();
            }
        }
        c.source = format!("{} [log-compressed weights]", c.source);
        c
    }

    /// Null model: Maslov-Sneppen edge swaps. Preserves every region's degree
    /// and the multiset of edge weights; destroys higher-order topology.
    /// Fibre lengths stay attached to region pairs (geometry is preserved).
    pub fn degree_preserving_rewire(&self, rng: &mut Rng, swaps_per_edge: usize) -> Connectome {
        let n = self.n();
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();
        let mut present: HashSet<(usize, usize)> = HashSet::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let w = self.w[i * n + j];
                if w > 0.0 {
                    edges.push((i, j, w));
                    present.insert((i, j));
                }
            }
        }
        let m = edges.len();
        let key = |a: usize, b: usize| if a < b { (a, b) } else { (b, a) };
        if m >= 2 {
            let target = swaps_per_edge * m;
            let (mut done, mut tries) = (0usize, 0usize);
            while done < target && tries < target * 50 {
                tries += 1;
                let (ia, ib) = (rng.below(m), rng.below(m));
                if ia == ib {
                    continue;
                }
                let (a, b, wa) = edges[ia];
                let (c0, d0, wb) = edges[ib];
                let (c, d) = if rng.uniform() < 0.5 { (c0, d0) } else { (d0, c0) };
                if a == c || a == d || b == c || b == d {
                    continue;
                }
                let e1 = key(a, d);
                let e2 = key(c, b);
                if present.contains(&e1) || present.contains(&e2) {
                    continue;
                }
                present.remove(&key(a, b));
                present.remove(&key(c0, d0));
                present.insert(e1);
                present.insert(e2);
                edges[ia] = (e1.0, e1.1, wa);
                edges[ib] = (e2.0, e2.1, wb);
                done += 1;
            }
        }
        let mut w = vec![0.0; n * n];
        for &(i, j, v) in &edges {
            w[i * n + j] = v;
            w[j * n + i] = v;
        }
        Connectome {
            regions: self.regions.clone(),
            w,
            len: self.len.clone(),
            source: format!("{} [degree-preserving rewire]", self.source),
        }
    }

    /// Null model: Erdos-Renyi graph with the same number of edges; the original
    /// weights are randomly reassigned to the new edges.
    pub fn density_matched_random(&self, rng: &mut Rng) -> Connectome {
        let n = self.n();
        let mut weights = Vec::new();
        let mut pairs = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                pairs.push((i, j));
                if self.w[i * n + j] > 0.0 {
                    weights.push(self.w[i * n + j]);
                }
            }
        }
        rng.shuffle(&mut pairs);
        rng.shuffle(&mut weights);
        let mut w = vec![0.0; n * n];
        for (k, &v) in weights.iter().enumerate() {
            let (i, j) = pairs[k];
            w[i * n + j] = v;
            w[j * n + i] = v;
        }
        Connectome {
            regions: self.regions.clone(),
            w,
            len: self.len.clone(),
            source: format!("{} [density-matched random]", self.source),
        }
    }

    /// Control: all-to-all graph carrying the same total strength.
    pub fn dense_uniform(&self) -> Connectome {
        let n = self.n();
        let total: f64 = self.w.iter().sum();
        let each = total / (n * (n - 1)) as f64;
        let mut w = vec![each; n * n];
        for i in 0..n {
            w[i * n + i] = 0.0;
        }
        Connectome {
            regions: self.regions.clone(),
            w,
            len: self.len.clone(),
            source: format!("{} [dense uniform]", self.source),
        }
    }

    // ------------------------------------------------------------------
    // Loaders
    // ------------------------------------------------------------------

    /// Load a generic connectome from CSV/whitespace matrices.
    ///
    /// * `weights`: N x N matrix (e.g. streamline counts or FA-weighted).
    /// * `lengths`: optional N x N matrix of mean fibre lengths in mm.
    /// * `labels`: optional CSV with one row per region: `name[,hemi[,system[,x,y,z]]]`.
    pub fn from_csv(
        weights: &Path,
        lengths: Option<&Path>,
        labels: Option<&Path>,
    ) -> Result<Connectome, String> {
        let (w, n) = parse_matrix(weights)?;
        let len = match lengths {
            Some(p) => {
                let (l, nl) = parse_matrix(p)?;
                if nl != n {
                    return Err(format!("lengths are {nl}x{nl} but weights are {n}x{n}"));
                }
                l
            }
            None => vec![0.0; n * n],
        };
        let regions = match labels {
            Some(p) => parse_labels(p, n)?,
            None => (0..n)
                .map(|i| Region {
                    name: format!("region{i}"),
                    hemi: '?',
                    system: "Unassigned".into(),
                    pos: [0.0; 3],
                })
                .collect(),
        };
        Connectome::new(regions, w, len, format!("csv:{}", weights.display()))
    }

    /// Build a region graph from a parcel-by-feature table (rows in the same order
    /// as `labels`), for data that has no tractography, e.g. parcellated maps.
    ///
    /// Edge weight = Pearson correlation between two parcels' feature profiles
    /// (columns z-scored first); each parcel keeps its `knn` strongest positive
    /// partners and the graph is symmetrised (union). Fibre lengths are the
    /// Euclidean distances between the label positions, optionally rescaled so the
    /// longest equals `max_length_mm`. This is a structural-covariance style graph,
    /// not axonal wiring; the source string says so.
    pub fn from_features(
        features: &Path,
        labels: &Path,
        knn: usize,
        max_length_mm: Option<f64>,
    ) -> Result<Connectome, String> {
        let text = fs::read_to_string(features).map_err(|e| format!("{}: {e}", features.display()))?;
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let r: Result<Vec<f64>, _> = line
                .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
                .filter(|t| !t.is_empty())
                .map(|t| t.parse::<f64>())
                .collect();
            rows.push(r.map_err(|e| format!("{}: {e}", features.display()))?);
        }
        let regions = parse_labels(labels, rows.len())?;
        Connectome::from_feature_rows(regions, rows, knn, max_length_mm, format!("features:{} [correlation kNN, not tractography]", features.display()))
    }

    /// In-memory form of [`Connectome::from_features`].
    pub fn from_feature_rows(
        regions: Vec<Region>,
        mut x: Vec<Vec<f64>>,
        knn: usize,
        max_length_mm: Option<f64>,
        source: String,
    ) -> Result<Connectome, String> {
        let n = regions.len();
        if n < 3 || x.len() != n {
            return Err(format!("need >= 3 regions and one feature row per region (got {} rows, {n} regions)", x.len()));
        }
        let d = x[0].len();
        if d < 3 || x.iter().any(|r| r.len() != d) {
            return Err("feature rows must all have the same length, at least 3 columns".into());
        }
        if knn == 0 || knn >= n {
            return Err(format!("--knn must be in 1..{n}"));
        }
        if regions.iter().all(|r| r.pos == [0.0; 3]) {
            return Err("from_features needs label positions (x,y,z) to derive conduction lengths".into());
        }
        // z-score each feature column so no measure dominates the profile
        for k in 0..d {
            let m = x.iter().map(|r| r[k]).sum::<f64>() / n as f64;
            let sd = (x.iter().map(|r| (r[k] - m).powi(2)).sum::<f64>() / n as f64).sqrt();
            for r in x.iter_mut() {
                r[k] = if sd > 1e-12 { (r[k] - m) / sd } else { 0.0 };
            }
        }
        let corr = |a: &[f64], b: &[f64]| -> f64 {
            let (ma, mb) = (a.iter().sum::<f64>() / d as f64, b.iter().sum::<f64>() / d as f64);
            let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
            for k in 0..d {
                let (u, v) = (a[k] - ma, b[k] - mb);
                sab += u * v;
                saa += u * u;
                sbb += v * v;
            }
            if saa < 1e-12 || sbb < 1e-12 { 0.0 } else { sab / (saa * sbb).sqrt() }
        };
        let mut w = vec![0.0; n * n];
        for i in 0..n {
            let mut c: Vec<(usize, f64)> = (0..n).filter(|&j| j != i).map(|j| (j, corr(&x[i], &x[j]))).collect();
            c.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for &(j, r) in c.iter().take(knn) {
                if r > 0.0 {
                    let v = r.max(w[i * n + j]).max(w[j * n + i]);
                    w[i * n + j] = v;
                    w[j * n + i] = v;
                }
            }
        }
        let mut len = vec![0.0; n * n];
        let mut dmax = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let l = euclid(&regions[i].pos, &regions[j].pos);
                len[i * n + j] = l;
                dmax = dmax.max(l);
            }
        }
        if let Some(target) = max_length_mm {
            if dmax > 1e-9 && target > 0.0 {
                len.iter_mut().for_each(|l| *l *= target / dmax);
            }
        }
        Connectome::new(regions, w, len, source)
    }

    /// Load a TVB-style connectivity directory (`weights.txt`, `tract_lengths.txt`,
    /// `centres.txt`) such as the human 66-region group-average connectome.
    pub fn from_tvb_dir(dir: &Path) -> Result<Connectome, String> {
        let (w, n) = parse_matrix(&dir.join("weights.txt"))?;
        let (len, nl) = parse_matrix(&dir.join("tract_lengths.txt"))?;
        if nl != n {
            return Err("weights.txt and tract_lengths.txt differ in size".into());
        }
        let text = fs::read_to_string(dir.join("centres.txt"))
            .map_err(|e| format!("{}: {e}", dir.join("centres.txt").display()))?;
        let mut regions = Vec::new();
        for line in text.lines() {
            let toks: Vec<&str> = line.split_whitespace().collect();
            if toks.is_empty() {
                continue;
            }
            let name = toks[0].to_string();
            let mut pos = [0.0; 3];
            for k in 0..3 {
                if let Some(v) = toks.get(1 + k).and_then(|t| t.parse::<f64>().ok()) {
                    pos[k] = v;
                }
            }
            let bytes: Vec<char> = name.chars().collect();
            let (hemi, code) = if bytes.len() > 1 && bytes[1].is_uppercase() && (bytes[0] == 'r' || bytes[0] == 'l') {
                (if bytes[0] == 'r' { 'R' } else { 'L' }, name[1..].to_string())
            } else if let Some(stem) = name.strip_suffix("_R") {
                ('R', stem.to_string())
            } else if let Some(stem) = name.strip_suffix("_L") {
                ('L', stem.to_string())
            } else {
                ('?', name.clone())
            };
            let system = desikan_system(&code).to_string();
            regions.push(Region { name, hemi, system, pos });
        }
        if regions.len() != n {
            return Err(format!("centres.txt lists {} regions but weights are {n}x{n}", regions.len()));
        }
        Connectome::new(regions, w, len, format!("tvb:{}", dir.display()))
    }

    // ------------------------------------------------------------------
    // Synthetic human-like connectome
    // ------------------------------------------------------------------

    /// Generate a synthetic bilateral connectome with human-like organisation:
    /// seven Yeo-style systems laid out on a brain-shaped geometry, a distance-
    /// dependent (exponential) connection rule, within-system clustering, strong
    /// homotopic (left-right) links, hub regions and log-normal weights.
    ///
    /// This is a stand-in for real tractography; use `from_tvb_dir`/`from_csv`
    /// for measured data.
    pub fn synthetic_human(per_hemi: usize, target_density: f64, seed: u64) -> Result<Connectome, String> {
        if per_hemi < 7 {
            return Err("synthetic connectome needs at least 7 regions per hemisphere".into());
        }
        if !(0.02..=0.9).contains(&target_density) {
            return Err("density must be within 0.02..0.9".into());
        }
        const NETS: [(&str, f64, [f64; 3], [f64; 3]); 7] = [
            ("Vis", 0.10, [-18.0, -88.0, 4.0], [9.0, 8.0, 12.0]),
            ("SomMot", 0.12, [-40.0, -22.0, 52.0], [8.0, 10.0, 9.0]),
            ("DorsAttn", 0.11, [-28.0, -56.0, 50.0], [10.0, 12.0, 8.0]),
            ("VentAttn", 0.10, [-50.0, -8.0, 14.0], [8.0, 16.0, 10.0]),
            ("Limbic", 0.06, [-28.0, 14.0, -24.0], [10.0, 10.0, 6.0]),
            ("Cont", 0.14, [-42.0, 26.0, 30.0], [8.0, 16.0, 10.0]),
            ("Default", 0.17, [-10.0, 12.0, 28.0], [8.0, 30.0, 18.0]),
        ];
        let mut rng = Rng::new(seed);
        let wsum: f64 = NETS.iter().map(|n| n.1).sum();
        let mut counts: Vec<usize> =
            NETS.iter().map(|n| ((n.1 / wsum) * per_hemi as f64).floor() as usize).collect();
        let mut order: Vec<usize> = (0..NETS.len()).collect();
        order.sort_by(|a, b| NETS[*b].1.partial_cmp(&NETS[*a].1).unwrap());
        let mut k = 0;
        while counts.iter().sum::<usize>() < per_hemi {
            counts[order[k % NETS.len()]] += 1;
            k += 1;
        }
        for ni in 0..NETS.len() {
            if counts[ni] == 0 {
                let m = (0..NETS.len()).max_by_key(|&i| counts[i]).unwrap();
                counts[m] -= 1;
                counts[ni] += 1;
            }
        }

        // Left hemisphere, then mirrored right hemisphere (region i <-> i + per_hemi).
        let mut net_of = Vec::new();
        let mut left_pos = Vec::new();
        for (ni, &c) in counts.iter().enumerate() {
            for _ in 0..c {
                let (_, _, cen, spr) = NETS[ni];
                let mut p = [0.0; 3];
                for d in 0..3 {
                    p[d] = cen[d] + spr[d] * rng.normal();
                }
                p[0] = -(p[0].abs().max(4.0));
                net_of.push(ni);
                left_pos.push(p);
            }
        }
        let n = 2 * per_hemi;
        let mut regions = Vec::with_capacity(n);
        let mut within = vec![0usize; NETS.len()];
        for i in 0..per_hemi {
            let ni = net_of[i];
            regions.push(Region {
                name: format!("{}_{:02}_L", NETS[ni].0, within[ni]),
                hemi: 'L',
                system: NETS[ni].0.to_string(),
                pos: left_pos[i],
            });
            within[ni] += 1;
        }
        let mut within = vec![0usize; NETS.len()];
        for i in 0..per_hemi {
            let ni = net_of[i];
            let p = left_pos[i];
            regions.push(Region {
                name: format!("{}_{:02}_R", NETS[ni].0, within[ni]),
                hemi: 'R',
                system: NETS[ni].0.to_string(),
                pos: [-p[0] + 1.5 * rng.normal(), p[1] + 1.5 * rng.normal(), p[2] + 1.5 * rng.normal()],
            });
            within[ni] += 1;
        }
        let net_full: Vec<usize> = (0..n).map(|i| net_of[i % per_hemi]).collect();

        // Hub propensity.
        let mut hub = vec![0.0; n];
        for i in 0..per_hemi {
            hub[i] = (0.6 * rng.normal()).exp();
        }
        for i in 0..per_hemi {
            hub[i + per_hemi] = hub[i] * (0.15 * rng.normal()).exp();
        }

        let lambda = 38.0; // mm, exponential distance rule
        let dist = |i: usize, j: usize| euclid(&regions[i].pos, &regions[j].pos);
        let mut prop = Vec::new(); // (i, j, propensity) for non-homotopic pairs
        for i in 0..n {
            for j in (i + 1)..n {
                if j == i + per_hemi {
                    continue;
                }
                let same = net_full[i] == net_full[j];
                let cross = (i < per_hemi) != (j < per_hemi);
                let g = (-dist(i, j) / lambda).exp()
                    * if same { 2.5 } else { 1.0 }
                    * if cross { 0.5 } else { 1.0 }
                    * (hub[i] * hub[j]).sqrt();
                prop.push((i, j, g));
            }
        }
        let n_pairs = (n * (n - 1) / 2) as f64;
        let homotopic_p = 0.97;
        let want = (target_density * n_pairs - homotopic_p * per_hemi as f64).max(0.0);
        let (mut lo, mut hi) = (0.0f64, 1e9f64);
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            let e: f64 = prop.iter().map(|&(_, _, g)| (mid * g).min(1.0)).sum();
            if e < want {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let a = 0.5 * (lo + hi);

        let mut w = vec![0.0; n * n];
        for &(i, j, g) in &prop {
            if rng.uniform() < (a * g).min(1.0) {
                let same = net_full[i] == net_full[j];
                let v = (0.9 * rng.normal()).exp()
                    * (-dist(i, j) / (1.5 * lambda)).exp()
                    * (hub[i] * hub[j]).sqrt()
                    * if same { 1.5 } else { 1.0 };
                w[i * n + j] = v;
                w[j * n + i] = v;
            }
        }
        for i in 0..per_hemi {
            let j = i + per_hemi;
            if rng.uniform() < homotopic_p {
                let v = 3.0 * (0.4 * rng.normal()).exp() * (hub[i] * hub[j]).sqrt() * (-dist(i, j) / (3.0 * lambda)).exp();
                w[i * n + j] = v;
                w[j * n + i] = v;
            }
        }
        // Guarantee a single connected component by bridging with the shortest links.
        let mean_w = {
            let nz: Vec<f64> = w.iter().copied().filter(|&v| v > 0.0).collect();
            nz.iter().sum::<f64>() / nz.len().max(1) as f64
        };
        loop {
            let lab = components(&w, n);
            if lab.iter().all(|&c| c == 0) {
                break;
            }
            let mut best = (f64::INFINITY, 0, 0);
            for i in 0..n {
                for j in 0..n {
                    if lab[i] == 0 && lab[j] != 0 {
                        let d = dist(i, j);
                        if d < best.0 {
                            best = (d, i, j);
                        }
                    }
                }
            }
            let (_, i, j) = best;
            w[i * n + j] = 0.05 * mean_w;
            w[j * n + i] = 0.05 * mean_w;
        }

        let mut len = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let cross = (i < per_hemi) != (j < per_hemi);
                    len[i * n + j] = 1.25 * dist(i, j) + 5.0 + if cross { 10.0 } else { 0.0 };
                }
            }
        }
        Connectome::new(
            regions,
            w,
            len,
            format!("synthetic-human(per_hemi={per_hemi}, density={target_density}, seed={seed})"),
        )
    }
}

fn parse_matrix(path: &Path) -> Result<(Vec<f64>, usize), String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let row: Result<Vec<f64>, _> = line
            .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
            .filter(|t| !t.is_empty())
            .map(|t| t.parse::<f64>())
            .collect();
        rows.push(row.map_err(|e| format!("{}: {e}", path.display()))?);
    }
    let n = rows.len();
    if n == 0 {
        return Err(format!("{}: empty matrix", path.display()));
    }
    if rows.iter().any(|r| r.len() != n) {
        return Err(format!("{}: matrix is not square ({n} rows)", path.display()));
    }
    Ok((rows.into_iter().flatten().collect(), n))
}

fn parse_labels(path: &Path, n: usize) -> Result<Vec<Region>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut regions = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if idx == 0 && ["name", "label", "region"].contains(&f[0].to_lowercase().as_str()) {
            continue;
        }
        let hemi = f
            .get(1)
            .and_then(|s| s.chars().next())
            .map(|c| c.to_ascii_uppercase())
            .filter(|c| *c == 'L' || *c == 'R')
            .unwrap_or('?');
        let system = match f.get(2).filter(|s| !s.is_empty() && !s.eq_ignore_ascii_case("auto")) {
            Some(s) => s.to_string(),
            None => glasser_system(f[0]).to_string(),
        };
        let mut pos = [0.0; 3];
        for k in 0..3 {
            if let Some(v) = f.get(3 + k).and_then(|s| s.parse::<f64>().ok()) {
                pos[k] = v;
            }
        }
        regions.push(Region { name: f[0].to_string(), hemi, system, pos });
    }
    if regions.len() != n {
        return Err(format!("{}: {} labels for {n} regions", path.display(), regions.len()));
    }
    Ok(regions)
}
