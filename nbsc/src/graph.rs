//! Simple undirected graph representation plus synthetic generators used
//! throughout the benchmark: a Stochastic Block Model (community-structured,
//! node-classification friendly, and — for reasonable parameters — a
//! non-bipartite near-regular graph, which is exactly the regime where the
//! Ihara-zeta / non-backtracking spectrum is best-conditioned, per §6 of
//! `ihara_zeta.rs`) and a tree (bipartite-like, no cycles, worst case for
//! non-backtracking filters — useful as a negative control).

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

/// An undirected, simple (no self-loops, no multi-edges) graph on `n`
/// vertices, stored as an edge list plus an adjacency list for fast
/// mat-vec / recursion use.
#[derive(Debug, Clone)]
pub struct Graph {
    pub n: usize,
    /// `neighbors[v]` = sorted list of neighbors of `v`.
    pub neighbors: Vec<Vec<usize>>,
    /// Undirected edges, each stored once as `(min, max)`.
    pub edges: Vec<(usize, usize)>,
}

impl Graph {
    pub fn new(n: usize) -> Self {
        Self { n, neighbors: vec![Vec::new(); n], edges: Vec::new() }
    }

    pub fn add_edge(&mut self, u: usize, v: usize) {
        if u == v {
            return; // no self-loops: the Hashimoto/non-backtracking construction assumes a simple graph
        }
        if self.neighbors[u].contains(&v) {
            return; // no multi-edges
        }
        self.neighbors[u].push(v);
        self.neighbors[v].push(u);
        let e = if u < v { (u, v) } else { (v, u) };
        self.edges.push(e);
    }

    pub fn m(&self) -> usize {
        self.edges.len()
    }

    pub fn degree(&self, v: usize) -> usize {
        self.neighbors[v].len()
    }

    pub fn degrees(&self) -> Vec<f64> {
        (0..self.n).map(|v| self.degree(v) as f64).collect()
    }

    pub fn min_degree(&self) -> usize {
        (0..self.n).map(|v| self.degree(v)).min().unwrap_or(0)
    }

    /// Dense adjacency matrix, row-major, `n x n`, `f64`.
    ///
    /// Only used for small graphs (unit tests, and the Burn v1 layer per the
    /// documented TODO in §8 of `ihara_zeta.rs`); large sparse graphs should
    /// stay on `neighbors`/CSR throughout.
    pub fn dense_adjacency(&self) -> Vec<f64> {
        let mut a = vec![0.0f64; self.n * self.n];
        for &(u, v) in &self.edges {
            a[u * self.n + v] = 1.0;
            a[v * self.n + u] = 1.0;
        }
        a
    }

    /// Is the graph connected? (BFS from vertex 0.) Used to sanity-check
    /// generated graphs before running spectral diagnostics on them.
    pub fn is_connected(&self) -> bool {
        if self.n == 0 {
            return true;
        }
        let mut seen = vec![false; self.n];
        let mut stack = vec![0usize];
        seen[0] = true;
        let mut count = 1;
        while let Some(u) = stack.pop() {
            for &v in &self.neighbors[u] {
                if !seen[v] {
                    seen[v] = true;
                    count += 1;
                    stack.push(v);
                }
            }
        }
        count == self.n
    }

    /// True if the graph contains no odd cycle (i.e. is bipartite), found by
    /// 2-coloring via BFS. A bipartite graph is the degenerate case where the
    /// non-backtracking spectrum collapses to +-eigenvalues of a related
    /// symmetric object and the "oriented cycle" structure NBSC is built to
    /// exploit is largely absent — useful as a negative control.
    pub fn is_bipartite(&self) -> bool {
        let mut color: Vec<i8> = vec![-1; self.n];
        for start in 0..self.n {
            if color[start] != -1 {
                continue;
            }
            color[start] = 0;
            let mut stack = vec![start];
            while let Some(u) = stack.pop() {
                for &v in &self.neighbors[u] {
                    if color[v] == -1 {
                        color[v] = 1 - color[u];
                        stack.push(v);
                    } else if color[v] == color[u] {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// Stochastic Block Model: `k` equal-sized communities of `block_size`
/// nodes each, intra-community edge probability `p_in`, inter-community
/// edge probability `p_out` (`p_in > p_out` gives assortative, classifiable
/// communities). Returns the graph plus a ground-truth label per node.
///
/// For `p_in`, `p_out` in the regime used by the benchmark (moderate
/// density, `p_in` not too close to 1) the result is, with overwhelming
/// probability, connected and non-bipartite — triangles form within blocks
/// whenever `p_in > 0` and `block_size >= 3`, so odd cycles are essentially
/// guaranteed. This is the graph family the derivation's §6 empirical
/// near-Ramanujan claim is aimed at, unlike a tree or a bipartite graph.
pub fn stochastic_block_model(
    k: usize,
    block_size: usize,
    p_in: f64,
    p_out: f64,
    seed: u64,
) -> (Graph, Vec<usize>) {
    let n = k * block_size;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut g = Graph::new(n);
    let mut labels = vec![0usize; n];
    for (v, label) in labels.iter_mut().enumerate() {
        *label = v / block_size;
    }
    for u in 0..n {
        for v in (u + 1)..n {
            let p = if labels[u] == labels[v] { p_in } else { p_out };
            if rng.gen::<f64>() < p {
                g.add_edge(u, v);
            }
        }
    }
    (g, labels)
}

/// A random tree on `n` nodes (uniform random recursive attachment). Trees
/// are bipartite and cycle-free by construction: the intended negative
/// control for the non-backtracking / Ihara-zeta construction, since every
/// non-backtracking walk on a tree is automatically non-repeating and the
/// zeta function degenerates.
pub fn random_tree(n: usize, seed: u64) -> Graph {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut g = Graph::new(n);
    let mut order: Vec<usize> = (1..n).collect();
    order.shuffle(&mut rng);
    for v in order {
        let parent = rng.gen_range(0..v);
        g.add_edge(parent, v);
    }
    g
}

/// A random `d`-regular-ish graph via repeated random-perfect-matching-style
/// pairing (configuration model, simplified: retried until simple). Not
/// exactly regular for small `n` due to rejected self-loops/multi-edges, but
/// close, sparse, and non-bipartite whenever `d >= 3` (triangles appear with
/// high probability once density crosses the giant-component threshold),
/// which is the near-Ramanujan-expander regime referenced in §6.
pub fn random_near_regular(n: usize, d: usize, seed: u64) -> Graph {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut g = Graph::new(n);
    let mut stubs: Vec<usize> = (0..n).flat_map(|v| std::iter::repeat(v).take(d)).collect();
    for _ in 0..200 {
        stubs.shuffle(&mut rng);
        let mut ok = true;
        let mut trial = g.clone();
        for pair in stubs.chunks(2) {
            if pair.len() < 2 {
                continue;
            }
            let (u, v) = (pair[0], pair[1]);
            if u == v || trial.neighbors[u].contains(&v) {
                ok = false;
                break;
            }
            trial.add_edge(u, v);
        }
        if ok {
            g = trial;
            break;
        }
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sbm_is_connected_and_non_bipartite_in_expected_regime() {
        let (g, labels) = stochastic_block_model(4, 30, 0.35, 0.02, 42);
        assert_eq!(labels.len(), g.n);
        assert!(g.is_connected(), "SBM should be connected at this density");
        assert!(!g.is_bipartite(), "SBM with p_in > 0 should contain triangles");
    }

    #[test]
    fn tree_is_bipartite_and_acyclic_edge_count() {
        let t = random_tree(50, 7);
        assert_eq!(t.m(), t.n - 1);
        assert!(t.is_bipartite());
        assert!(t.is_connected());
    }

    #[test]
    fn near_regular_graph_degree_close_to_target() {
        let g = random_near_regular(60, 4, 3);
        let avg_deg: f64 = g.degrees().iter().sum::<f64>() / g.n as f64;
        assert!((avg_deg - 4.0).abs() < 1.0, "avg degree {avg_deg} should be close to 4");
    }
}
