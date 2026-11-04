use nalgebra::{DMatrix, DVector, Vector2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// `spectral_hypergraph` is the production hypergraph crate this file bridges
// into (see MODULE: hg_bridge below). `laplacian`/`spectral` stay qualified
// at their call sites since `HypergraphOperator`, `clique_expansion_adjacency`,
// `dense_eigen`, and `lanczos_smallest` all live there, and spelling the
// module out keeps it obvious whether a given routine is this file's own
// vendored implementation or the crate's.
use spectral_hypergraph::{HypergraphBuilder, HypergraphError, SpectralHypergraph, VertexId};

// =========================================================================
// MODULE: jacobi_ds
// =========================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum JacobiError {
    EmptyCoefficients,
    InvalidParameters { alpha: f64, beta: f64 },
    OutOfDomain(f64),
}

impl fmt::Display for JacobiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JacobiError::EmptyCoefficients => {
                write!(f, "coefficient vector must contain at least one element")
            }
            JacobiError::InvalidParameters { alpha, beta } => write!(
                f,
                "invalid Jacobi parameters alpha={alpha}, beta={beta}: both must be > -1"
            ),
            JacobiError::OutOfDomain(x) => {
                write!(f, "x = {x} is outside the Jacobi domain [-1, 1]")
            }
        }
    }
}
impl std::error::Error for JacobiError {}

fn check_params(alpha: f64, beta: f64) -> Result<(), JacobiError> {
    if alpha <= -1.0 || beta <= -1.0 || !alpha.is_finite() || !beta.is_finite() {
        return Err(JacobiError::InvalidParameters { alpha, beta });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub struct JacobiPoly {
    alpha: f64,
    beta: f64,
    coeffs: Vec<f64>,
}

impl JacobiPoly {
    pub fn new(alpha: f64, beta: f64, coeffs: Vec<f64>) -> Result<Self, JacobiError> {
        check_params(alpha, beta)?;
        if coeffs.is_empty() {
            return Err(JacobiError::EmptyCoefficients);
        }
        Ok(Self { alpha, beta, coeffs })
    }

    pub fn eval(&self, x: f64) -> Result<f64, JacobiError> {
        if !(-1.0..=1.0).contains(&x) {
            return Err(JacobiError::OutOfDomain(x));
        }
        let n = self.coeffs.len();
        let mut sum = self.coeffs[0];
        if n == 1 {
            return Ok(sum);
        }
        let p1 = 0.5 * ((self.alpha + self.beta + 2.0) * x + (self.alpha - self.beta));
        sum += self.coeffs[1] * p1;
        if n == 2 {
            return Ok(sum);
        }
        let mut p_prev2 = 1.0_f64;
        let mut p_prev1 = p1;
        for k in 2..n {
            let nf = k as f64;
            let a_plus_b = self.alpha + self.beta;
            let t = 2.0 * nf + a_plus_b;
            let a_coef = 2.0 * nf * (nf + a_plus_b) * (t - 2.0);
            let b_coef =
                (t - 1.0) * (t * (t - 2.0) * x + self.alpha * self.alpha - self.beta * self.beta);
            let c_coef = 2.0 * (nf + self.alpha - 1.0) * (nf + self.beta - 1.0) * t;
            let p_curr = (b_coef * p_prev1 - c_coef * p_prev2) / a_coef;
            sum += self.coeffs[k] * p_curr;
            p_prev2 = p_prev1;
            p_prev1 = p_curr;
        }
        Ok(sum)
    }
}

// =========================================================================
// MODULE: graph
// =========================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    EmptyGraph,
    IndexOutOfBounds { node: usize, num_nodes: usize },
    SelfLoop(usize),
    Disconnected { components: usize },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::EmptyGraph => write!(f, "graph must contain at least one node"),
            GraphError::IndexOutOfBounds { node, num_nodes } => write!(
                f,
                "edge references node {node}, but graph only has {num_nodes} nodes"
            ),
            GraphError::SelfLoop(n) => write!(f, "self-loop on node {n} is not allowed"),
            GraphError::Disconnected { components } => write!(
                f,
                "graph has {components} connected components; spectral embedding via \
                 smallest-algebraic eigenvalues requires a single connected component"
            ),
        }
    }
}
impl std::error::Error for GraphError {}

#[derive(Debug, Clone)]
pub struct CsrGraph {
    num_nodes: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    degrees: Vec<f64>,
    diagonal_shift: Vec<f64>,
}

impl CsrGraph {
    pub fn from_edges(num_nodes: usize, edges: &[(usize, usize)]) -> Result<Self, GraphError> {
        if num_nodes == 0 {
            return Err(GraphError::EmptyGraph);
        }
        let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); num_nodes];
        for &(u, v) in edges {
            if u >= num_nodes {
                return Err(GraphError::IndexOutOfBounds { node: u, num_nodes });
            }
            if v >= num_nodes {
                return Err(GraphError::IndexOutOfBounds { node: v, num_nodes });
            }
            if u == v {
                return Err(GraphError::SelfLoop(u));
            }
            adj[u].insert(v);
            adj[v].insert(u);
        }

        let components = count_components(&adj);
        if components != 1 {
            return Err(GraphError::Disconnected { components });
        }

        let mut row_ptr = Vec::with_capacity(num_nodes + 1);
        let mut col_idx = Vec::new();
        row_ptr.push(0);
        for neighbors in &adj {
            let mut sorted: Vec<usize> = neighbors.iter().copied().collect();
            sorted.sort_unstable();
            col_idx.extend(sorted);
            row_ptr.push(col_idx.len());
        }
        let degrees: Vec<f64> = (0..num_nodes)
            .map(|i| (row_ptr[i + 1] - row_ptr[i]) as f64)
            .collect();
        let diagonal_shift = vec![0.0; num_nodes];

        Ok(Self { num_nodes, row_ptr, col_idx, degrees, diagonal_shift })
    }

    fn neighbors(&self, i: usize) -> &[usize] {
        &self.col_idx[self.row_ptr[i]..self.row_ptr[i + 1]]
    }

    pub fn laplacian_multiply(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.num_nodes {
            let mut row_sum = (self.degrees[i] + self.diagonal_shift[i]) * x[i];
            for &j in self.neighbors(i) {
                row_sum -= x[j];
            }
            y[i] = row_sum;
        }
    }

    pub fn adjacency_multiply(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.num_nodes {
            let mut row_sum = 0.0;
            for &j in self.neighbors(i) {
                row_sum += x[j];
            }
            y[i] = row_sum;
        }
    }

    pub fn is_regular(&self) -> Option<usize> {
        let first = self.degrees[0];
        if self.degrees.iter().all(|&d| (d - first).abs() < 1e-9) {
            Some(first.round() as usize)
        } else {
            None
        }
    }

    pub fn is_bipartite(&self) -> bool {
        let mut color = vec![-1i8; self.num_nodes];
        for start in 0..self.num_nodes {
            if color[start] != -1 {
                continue;
            }
            color[start] = 0;
            let mut queue = VecDeque::new();
            queue.push_back(start);
            while let Some(u) = queue.pop_front() {
                for &v in self.neighbors(u) {
                    if color[v] == -1 {
                        color[v] = 1 - color[u];
                        queue.push_back(v);
                    } else if color[v] == color[u] {
                        return false;
                    }
                }
            }
        }
        true
    }
}

// =========================================================================
// MODULE: eigensolve — vendored Hermitian Lanczos
// =========================================================================

pub trait SymmetricOp {
    fn dim(&self) -> usize;
    fn multiply(&self, x: &[f64], y: &mut [f64]);
}

pub struct LaplacianOp<'a>(pub &'a CsrGraph);
impl<'a> SymmetricOp for LaplacianOp<'a> {
    fn dim(&self) -> usize {
        self.0.num_nodes
    }
    fn multiply(&self, x: &[f64], y: &mut [f64]) {
        self.0.laplacian_multiply(x, y);
    }
}

pub struct AdjacencyOp<'a>(pub &'a CsrGraph);
impl<'a> SymmetricOp for AdjacencyOp<'a> {
    fn dim(&self) -> usize {
        self.0.num_nodes
    }
    fn multiply(&self, x: &[f64], y: &mut [f64]) {
        self.0.adjacency_multiply(x, y);
    }
}

#[derive(Debug, Clone)]
pub struct HermitianLanczos {
    pub eigenvalues: DVector<f64>,
    pub eigenvectors: DMatrix<f64>,
}

impl HermitianLanczos {
    pub fn new(op: &impl SymmetricOp, maximum_iterations: usize, seed: u64) -> Result<Self, String> {
        let n = op.dim();
        if maximum_iterations == 0 || maximum_iterations > n {
            return Err(format!(
                "maximum_iterations must be in 1..={n}, got {maximum_iterations}"
            ));
        }
        let tolerance = 1e-10;
        let mut rng = StdRng::seed_from_u64(seed);

        let mut alphas = vec![0.0_f64; maximum_iterations];
        let mut betas = vec![0.0_f64; maximum_iterations.saturating_sub(1)];
        let mut vs = DMatrix::<f64>::zeros(n, maximum_iterations);

        let x0 = DVector::from_iterator(n, (0..n).map(|_| rng.random::<f64>() - 0.5));
        vs.set_column(0, &(&x0 / x0.norm()));

        let mut built = maximum_iterations;

        for i in 0..maximum_iterations {
            let col_i = vs.column(i).clone_owned();
            let mut tmp_buf = vec![0.0; n];
            op.multiply(col_i.as_slice(), &mut tmp_buf);
            let mut tmp = DVector::from_vec(tmp_buf);
            alphas[i] = tmp.dot(&col_i);

            if i == 0 {
                tmp -= alphas[0] * &col_i;
            } else {
                let col_prev = vs.column(i - 1).clone_owned();
                tmp -= alphas[i] * &col_i + betas[i - 1] * &col_prev;
            }

            for k in 0..i {
                let col_k = vs.column(k).clone_owned();
                let projection = tmp.dot(&col_k);
                tmp -= projection * &col_k;
            }

            if i < maximum_iterations - 1 {
                let beta = tmp.norm();
                if beta < tolerance {
                    alphas.truncate(i + 1);
                    betas.truncate(i);
                    built = i + 1;
                    break;
                }
                betas[i] = beta;
                vs.set_column(i + 1, &(&tmp / beta));
            }
        }

        let m = built;
        let alphas_v = DVector::from_vec(alphas);
        let betas_v = DVector::from_vec(betas);
        let tridiagonal = Self::construct_tridiagonal(&alphas_v, &betas_v);
        let eig = nalgebra::linalg::SymmetricEigen::new(tridiagonal);

        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&a, &b| eig.eigenvalues[a].partial_cmp(&eig.eigenvalues[b]).unwrap());

        let eigenvalues = DVector::from_iterator(m, order.iter().map(|&i| eig.eigenvalues[i]));
        let mut ritz_coeffs = DMatrix::<f64>::zeros(m, m);
        for (col, &orig) in order.iter().enumerate() {
            ritz_coeffs.set_column(col, &eig.eigenvectors.column(orig));
        }
        let eigenvectors = vs.columns(0, m) * ritz_coeffs;

        Ok(Self { eigenvalues, eigenvectors })
    }

    fn construct_tridiagonal(alphas: &DVector<f64>, betas: &DVector<f64>) -> DMatrix<f64> {
        let dim = alphas.len();
        DMatrix::<f64>::from_fn(dim, dim, |i, j| {
            if i == j {
                alphas[i]
            } else if i == j + 1 {
                betas[j]
            } else if j == i + 1 {
                betas[i]
            } else {
                0.0
            }
        })
    }
}

fn count_components(adj: &[HashSet<usize>]) -> usize {
    let n = adj.len();
    let mut visited = vec![false; n];
    let mut components = 0;
    for start in 0..n {
        if visited[start] {
            continue;
        }
        components += 1;
        visited[start] = true;
        let mut queue = VecDeque::new();
        queue.push_back(start);
        while let Some(u) = queue.pop_front() {
            for &v in &adj[u] {
                if !visited[v] {
                    visited[v] = true;
                    queue.push_back(v);
                }
            }
        }
    }
    components
}

// =========================================================================
// MODULE: hg_bridge -- wiring this file into the `spectral_hypergraph` crate
// =========================================================================
//
// Everything above this point (`CsrGraph`, `HermitianLanczos`,
// `SymmetricOp`) only understands plain pairwise-edge graphs. `spectral_hypergraph`
// generalizes that to genuine hyperedges (>= 2 members, not just 2) via its
// own validated `SpectralHypergraph` type, matrix-free normalized-Laplacian
// operator (`laplacian::HypergraphOperator`), and eigensolvers
// (`spectral::lanczos_smallest`, `spectral::fiedler_vector`,
// `spectral::dense_eigen`). This module is the seam between the two:
//
// * `SymmetricOp` is implemented directly for `spectral_hypergraph`'s
//   `HypergraphOperator` (legal under the orphan rule -- `SymmetricOp` is
//   the local trait here, `HypergraphOperator` is the foreign type), so
//   this file's own vendored, already-tested `HermitianLanczos` can run
//   *unmodified* against a real hypergraph's normalized Laplacian.
// * `csr_graph_to_hypergraph` embeds any existing `CsrGraph` as a
//   `SpectralHypergraph` with one 2-member hyperedge per plain edge, so the
//   two representations round-trip and can be cross-checked against each
//   other (see the `hg_bridge` tests at the bottom of this file).
//
// `SymmetricOp::dim`/`multiply` below call `spectral_hypergraph::LinearOperator`
// through fully-qualified syntax (`<Self as LinearOperator>::dim(self)`)
// rather than plain `self.dim()`. That's deliberate, not decorative: this
// impl block is itself defining `SymmetricOp::dim` for this type, so an
// unqualified `self.dim()` here would recurse into the very method being
// written instead of reaching `LinearOperator::dim`.

impl SymmetricOp for spectral_hypergraph::laplacian::HypergraphOperator {
    fn dim(&self) -> usize {
        <Self as spectral_hypergraph::LinearOperator>::dim(self)
    }

    fn multiply(&self, x: &[f64], y: &mut [f64]) {
        let xv = DVector::from_row_slice(x);
        let yv = <Self as spectral_hypergraph::LinearOperator>::apply(self, &xv);
        y.copy_from_slice(yv.as_slice());
    }
}

/// Embeds `graph` as a `SpectralHypergraph` with one weight-1.0, 2-member
/// hyperedge per undirected edge of `graph`. Vertex `i` is labeled `"v{i}"`
/// so `VertexId(i)` and the original `CsrGraph` node index always agree --
/// callers never need a separate id-remapping table.
///
/// This is a faithful embedding, not an approximation: for a hypergraph
/// whose hyperedges are *all* 2-member, `laplacian::clique_expansion_adjacency`
/// reduces to exactly `graph`'s own adjacency (each hyperedge of cardinality
/// 2 contributes `weight / (2 - 1) = weight` to exactly one vertex pair),
/// which is what the `hg_bridge` tests below verify directly rather than
/// asserting by construction.
pub fn csr_graph_to_hypergraph(graph: &CsrGraph) -> Result<SpectralHypergraph, SpectralError> {
    let mut builder = HypergraphBuilder::with_capacity(graph.num_nodes, graph.col_idx.len() / 2, 2);
    let ids: Vec<VertexId> = (0..graph.num_nodes)
        .map(|i| builder.add_vertex(format!("v{i}")))
        .collect::<Result<_, _>>()?;

    for u in 0..graph.num_nodes {
        for &v in graph.neighbors(u) {
            if v > u {
                builder.add_hyperedge(&[ids[u], ids[v]], 1.0)?;
            }
        }
    }
    Ok(builder.build()?)
}

// =========================================================================
// MODULE: spectral embedding
// =========================================================================

#[derive(Debug)]
pub enum SpectralError {
    Graph(GraphError),
    Jacobi(JacobiError),
    Eigensolve(String),
    InsufficientEigenpairs { requested: usize, available: usize },
    /// Wraps `spectral_hypergraph::HypergraphError` verbatim -- surfaced by
    /// every `hg_bridge` routine below rather than lossily collapsed into
    /// `Eigensolve(String)`, so callers can `match` on e.g.
    /// `HypergraphError::IsolatedVertex` the same way they already match on
    /// this crate's own `GraphError`/`JacobiError` variants.
    Hypergraph(HypergraphError),
}

impl fmt::Display for SpectralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpectralError::Graph(e) => write!(f, "graph error: {e}"),
            SpectralError::Jacobi(e) => write!(f, "jacobi error: {e}"),
            SpectralError::Eigensolve(msg) => write!(f, "eigensolver failed to converge: {msg}"),
            SpectralError::InsufficientEigenpairs { requested, available } => write!(
                f,
                "requested {requested} eigenpairs but the solver only returned {available}"
            ),
            SpectralError::Hypergraph(e) => write!(f, "hypergraph error: {e}"),
        }
    }
}
impl std::error::Error for SpectralError {}
impl From<GraphError> for SpectralError {
    fn from(e: GraphError) -> Self {
        SpectralError::Graph(e)
    }
}
impl From<JacobiError> for SpectralError {
    fn from(e: JacobiError) -> Self {
        SpectralError::Jacobi(e)
    }
}
impl From<HypergraphError> for SpectralError {
    fn from(e: HypergraphError) -> Self {
        SpectralError::Hypergraph(e)
    }
}

#[derive(Debug, Clone)]
pub struct SpectralNode {
    pub id: usize,
    pub embedding: Vector2<f64>,
    pub geometric_moments: JacobiPoly,
}

fn lanczos_subspace_size(n: usize, nev: usize) -> usize {
    (nev.saturating_mul(4)).clamp(nev, n)
}

pub fn build_spectral_nodes(
    graph: &CsrGraph,
    nev: usize,
    seed: u64,
) -> Result<Vec<SpectralNode>, SpectralError> {
    let n = graph.num_nodes;
    if nev < 3 {
        return Err(SpectralError::InsufficientEigenpairs { requested: nev, available: 0 });
    }
    if nev > n {
        return Err(SpectralError::InsufficientEigenpairs { requested: nev, available: n });
    }

    let subspace_size = lanczos_subspace_size(n, nev);
    let solved = HermitianLanczos::new(&LaplacianOp(graph), subspace_size, seed)
        .map_err(SpectralError::Eigensolve)?;

    if solved.eigenvectors.ncols() < 3 || solved.eigenvectors.nrows() != n {
        return Err(SpectralError::InsufficientEigenpairs {
            requested: 3,
            available: solved.eigenvectors.ncols(),
        });
    }

    let mut spectral_nodes = Vec::with_capacity(n);
    for i in 0..n {
        let x_coord = solved.eigenvectors[(i, 1)];
        let y_coord = solved.eigenvectors[(i, 2)];
        let continuous_pos = Vector2::new(x_coord, y_coord);

        let x_clamped = x_coord.clamp(-1.0, 1.0);
        let y_clamped = y_coord.clamp(-1.0, 1.0);

        let poly = JacobiPoly::new(0.0, 0.0, vec![1.0, x_clamped, y_clamped])?;

        spectral_nodes.push(SpectralNode { id: i, embedding: continuous_pos, geometric_moments: poly });
    }
    Ok(spectral_nodes)
}

/// Hypergraph analog of [`build_spectral_nodes`]: same 2D embedding (second
/// and third Laplacian eigenvector as x/y) and same Jacobi-moments wiring,
/// but the eigenpairs come from `spectral_hypergraph`'s matrix-free
/// normalized-Laplacian operator via this file's own `HermitianLanczos`
/// (see `hg_bridge`'s `SymmetricOp` impl above) rather than from
/// `LaplacianOp`/`CsrGraph`. `hg`'s vertex ids must be dense `0..n` (true of
/// every `SpectralHypergraph` built via `HypergraphBuilder`, including ones
/// produced by [`csr_graph_to_hypergraph`]) so a vertex's `VertexId(i)`
/// lines up with `SpectralNode.id == i` exactly as it does in the
/// plain-graph path.
pub fn build_spectral_nodes_hg(
    hg: &SpectralHypergraph,
    nev: usize,
    seed: u64,
) -> Result<Vec<SpectralNode>, SpectralError> {
    let n = hg.num_vertices();
    if nev < 3 {
        return Err(SpectralError::InsufficientEigenpairs { requested: nev, available: 0 });
    }
    if nev > n {
        return Err(SpectralError::InsufficientEigenpairs { requested: nev, available: n });
    }

    let op = spectral_hypergraph::laplacian::HypergraphOperator::new(hg)?;
    let subspace_size = lanczos_subspace_size(n, nev);
    let solved = HermitianLanczos::new(&op, subspace_size, seed).map_err(SpectralError::Eigensolve)?;

    if solved.eigenvectors.ncols() < 3 || solved.eigenvectors.nrows() != n {
        return Err(SpectralError::InsufficientEigenpairs {
            requested: 3,
            available: solved.eigenvectors.ncols(),
        });
    }

    let mut spectral_nodes = Vec::with_capacity(n);
    for i in 0..n {
        let x_coord = solved.eigenvectors[(i, 1)];
        let y_coord = solved.eigenvectors[(i, 2)];
        let continuous_pos = Vector2::new(x_coord, y_coord);

        let x_clamped = x_coord.clamp(-1.0, 1.0);
        let y_clamped = y_coord.clamp(-1.0, 1.0);

        let poly = JacobiPoly::new(0.0, 0.0, vec![1.0, x_clamped, y_clamped])?;

        spectral_nodes.push(SpectralNode { id: i, embedding: continuous_pos, geometric_moments: poly });
    }
    Ok(spectral_nodes)
}

// =========================================================================
// MODULE: ramanujan
// =========================================================================
#[derive(Debug, Clone)]
pub struct RamanujanFeature {
    pub degree_regular: Option<usize>,
    pub mean_degree: f64,
    pub bipartite: bool,
    pub bound: f64,
    pub extremal_eigenvalues: Vec<f64>,
    pub max_nontrivial_abs: f64,
    pub deviation: f64,
    pub is_heuristic: bool,
}

pub fn compute_ramanujan_feature(
    graph: &CsrGraph,
    seed: u64,
) -> Result<RamanujanFeature, SpectralError> {
    let n = graph.num_nodes;
    let mean_degree = graph.degrees.iter().sum::<f64>() / n as f64;
    let degree_regular = graph.is_regular();
    let d_for_bound = degree_regular.map(|d| d as f64).unwrap_or(mean_degree);
    let q = (d_for_bound - 1.0).max(0.0);
    let bound = 2.0 * q.sqrt();
    let bipartite = graph.is_bipartite();

    let subspace = lanczos_subspace_size(n, 3).max(4.min(n));
    let solved = HermitianLanczos::new(&AdjacencyOp(graph), subspace, seed)
        .map_err(SpectralError::Eigensolve)?;
    let extremal_eigenvalues: Vec<f64> = solved.eigenvalues.iter().copied().collect();

    let is_trivial = |mu: f64| -> bool {
        (mu - d_for_bound).abs() < 1e-6 || (bipartite && (mu + d_for_bound).abs() < 1e-6)
    };

    let max_nontrivial_abs = extremal_eigenvalues
        .iter()
        .copied()
        .filter(|&mu| !is_trivial(mu))
        .map(f64::abs)
        .fold(0.0_f64, f64::max);

    let deviation = (max_nontrivial_abs - bound).max(0.0);

    Ok(RamanujanFeature {
        degree_regular,
        mean_degree,
        bipartite,
        bound,
        extremal_eigenvalues,
        max_nontrivial_abs,
        deviation,
        is_heuristic: degree_regular.is_none(),
    })
}

/// Hypergraph analog of [`compute_ramanujan_feature`]. The `2*sqrt(d-1)`
/// Ramanujan bound is a theorem about the *adjacency* spectrum of a
/// `d`-regular ordinary graph, and a genuine hyperedge (cardinality > 2)
/// has no canonical adjacency matrix -- so this always goes through
/// `laplacian::clique_expansion_adjacency` (the crate's classical
/// "flatten every hyperedge into a weighted clique" reduction) before
/// reusing the same bound/deviation arithmetic as the plain-graph version.
/// Two consequences worth being explicit about, unlike the plain-graph
/// version: `is_heuristic` is unconditionally `true` (even a
/// vertex-degree-regular hypergraph is not "regular" in the sense the
/// Ramanujan theorem needs, since clique expansion is a lossy reduction),
/// and `bipartite` is unconditionally `false` (there is no standard
/// hypergraph bipartiteness check here to exempt a second trivial
/// eigenvalue at `-d`, so this stays conservative rather than guessing).
pub fn compute_hypergraph_ramanujan_feature(
    hg: &SpectralHypergraph,
    seed: u64,
) -> Result<RamanujanFeature, SpectralError> {
    let _ = seed; // kept for API symmetry with compute_ramanujan_feature; dense_eigen below is deterministic.
    let n = hg.num_vertices();
    let degrees: Vec<f64> = hg
        .vertex_ids()
        .map(|v| hg.vertex_degree(v))
        .collect::<Result<_, _>>()?;
    let mean_degree = degrees.iter().sum::<f64>() / n as f64;
    let first = degrees[0];
    let degree_regular = if degrees.iter().all(|&d| (d - first).abs() < 1e-9) {
        Some(first.round() as usize)
    } else {
        None
    };
    let d_for_bound = degree_regular.map(|d| d as f64).unwrap_or(mean_degree);
    let q = (d_for_bound - 1.0).max(0.0);
    let bound = 2.0 * q.sqrt();

    let adjacency = spectral_hypergraph::laplacian::clique_expansion_adjacency(hg)?;
    let eig = spectral_hypergraph::spectral::dense_eigen(&adjacency);
    let extremal_eigenvalues: Vec<f64> = eig.eigenvalues.iter().copied().collect();

    let is_trivial = |mu: f64| (mu - d_for_bound).abs() < 1e-6;
    let max_nontrivial_abs = extremal_eigenvalues
        .iter()
        .copied()
        .filter(|&mu| !is_trivial(mu))
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let deviation = (max_nontrivial_abs - bound).max(0.0);

    Ok(RamanujanFeature {
        degree_regular,
        mean_degree,
        bipartite: false,
        bound,
        extremal_eigenvalues,
        max_nontrivial_abs,
        deviation,
        is_heuristic: true,
    })
}

// =========================================================================
// MODULE: HNSW
// =========================================================================

#[derive(Debug, Clone)]
pub struct HnswConfig {
    pub max_layers: usize,
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub level_mult: f64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            max_layers: 4,
            m: 8,
            m0: 16,
            ef_construction: 32,
            level_mult: 1.0 / (2.0_f64).ln(),
        }
    }
}

pub struct HnswIndex {
    nodes: Vec<SpectralNode>,
    layers: Vec<HashMap<usize, Vec<usize>>>,
    node_levels: Vec<usize>,
    entry_point: Option<usize>,
    config: HnswConfig,
}

impl HnswIndex {
    pub fn new(config: HnswConfig) -> Self {
        let max_layers = config.max_layers.max(1);
        Self {
            nodes: Vec::new(),
            layers: vec![HashMap::new(); max_layers],
            node_levels: Vec::new(),
            entry_point: None,
            config: HnswConfig { max_layers, ..config },
        }
    }

    pub fn build(nodes: Vec<SpectralNode>, config: HnswConfig, seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_rng(&mut rand::rng()),
        };
        let mut index = Self::new(config);
        for node in nodes {
            index.insert(node, &mut rng);
        }
        index
    }

    fn distance_to(&self, node_id: usize, query: Vector2<f64>) -> f64 {
        (self.nodes[node_id].embedding - query).norm()
    }

    fn random_level(&self, rng: &mut impl Rng) -> usize {
        let r: f64 = rng.random::<f64>().max(1e-12);
        let level = (-r.ln() * self.config.level_mult).floor().max(0.0) as usize;
        level.min(self.config.max_layers - 1)
    }

    fn greedy_closest(&self, layer: usize, entry: usize, query: Vector2<f64>) -> usize {
        let mut current = entry;
        let mut current_dist = self.distance_to(current, query);
        loop {
            let mut improved = false;
            if let Some(neighbors) = self.layers[layer].get(&current) {
                for &cand in neighbors {
                    let d = self.distance_to(cand, query);
                    if d < current_dist {
                        current_dist = d;
                        current = cand;
                        improved = true;
                    }
                }
            }
            if !improved {
                return current;
            }
        }
    }

    fn search_layer(
        &self,
        layer: usize,
        entry: usize,
        query: Vector2<f64>,
        ef: usize,
    ) -> Vec<(usize, f64)> {
        let mut visited: HashSet<usize> = HashSet::new();
        visited.insert(entry);
        let mut result: Vec<(usize, f64)> = vec![(entry, self.distance_to(entry, query))];
        let mut frontier = vec![entry];

        while !frontier.is_empty() {
            let mut next_frontier = Vec::new();
            for &current in &frontier {
                if let Some(neighbors) = self.layers[layer].get(&current) {
                    for &cand in neighbors {
                        if visited.insert(cand) {
                            result.push((cand, self.distance_to(cand, query)));
                            next_frontier.push(cand);
                        }
                    }
                }
            }
            result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            result.truncate(ef.max(1));
            frontier = next_frontier;
        }
        result
    }

    pub fn insert(&mut self, node: SpectralNode, rng: &mut impl Rng) {
        let node_id = self.nodes.len();
        let level = self.random_level(rng);
        let query = node.embedding;
        self.nodes.push(node);
        self.node_levels.push(level);

        let Some(mut ep) = self.entry_point else {
            self.entry_point = Some(node_id);
            for l in 0..=level {
                self.layers[l].insert(node_id, Vec::new());
            }
            return;
        };

        let top_level = self.node_levels[ep];

        if top_level > level {
            for l in (level + 1..=top_level.min(self.config.max_layers - 1)).rev() {
                ep = self.greedy_closest(l, ep, query);
            }
        }

        let positions: Vec<Vector2<f64>> = self.nodes.iter().map(|n| n.embedding).collect();
        for l in (0..=level.min(self.config.max_layers - 1)).rev() {
            let candidates = self.search_layer(l, ep, query, self.config.ef_construction);
            let m_max = if l == 0 { self.config.m0 } else { self.config.m };
            let selected: Vec<usize> = candidates.iter().take(m_max).map(|&(id, _)| id).collect();

            self.layers[l].insert(node_id, selected.clone());
            for &neighbor in &selected {
                let npos = positions[neighbor];
                let entry = self.layers[l].entry(neighbor).or_default();
                entry.push(node_id);
                if entry.len() > m_max {
                    entry.sort_by(|&a, &b| {
                        let da = (positions[a] - npos).norm();
                        let db = (positions[b] - npos).norm();
                        da.partial_cmp(&db).unwrap_or(Ordering::Equal)
                    });
                    entry.truncate(m_max);
                }
            }
            if let Some(&(closest, _)) = candidates.first() {
                ep = closest;
            }
        }

        if level > top_level {
            self.entry_point = Some(node_id);
        }
    }

    pub fn search(&self, query: Vector2<f64>, ef_search: usize) -> Option<usize> {
        let mut ep = self.entry_point?;
        let top_level = self.node_levels[ep];
        for l in (1..=top_level.min(self.config.max_layers - 1)).rev() {
            ep = self.greedy_closest(l, ep, query);
        }
        self.search_layer(0, ep, query, ef_search.max(1))
            .into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .map(|(id, _)| id)
    }
}

// =========================================================================
// MODULE: anomaly — reconstruction-error autoencoder, WITHOUT burn
// =========================================================================
//
// Same role as the original `burn_anomaly` module and the same public
// shape (`extract_anomaly_features` / `FeatureNormalizer` /
// `train_anomaly_detector` / `score_nodes`), but with Burn's autodiff
// swapped for a from-scratch forward pass + manual backprop over
// `nalgebra` matrices. Reason: `burn 0.13`'s dependency tree requires
// Cargo's `edition2024` feature several layers deep (first `zeroize`,
// pin that and `rmp-serde` needs it next, and so on) -- not a single
// pinnable crate, so not practical to chase in a sandbox stuck on
// rustc/cargo 1.75.0 with no path to a newer toolchain. `nalgebra` and
// `rand` are already load-bearing elsewhere in this file, so this reuses
// them rather than pulling in anything new.
//
// Architecture matches the original exactly: 7 -> hidden -> bottleneck ->
// hidden -> 7, ReLU on every layer except the final reconstruction, MSE
// loss, Adam optimizer. Same honest scope caveat applies: on a 6-node demo
// graph there are more free parameters than training rows, so it
// memorizes every node and the reconstruction-error signal is not
// meaningful at this scale -- wired end-to-end for correctness, not
// presented as a real anomaly detector on 6 samples.

pub const ANOMALY_FEATURE_DIM: usize = 7;
const JACOBI_PROBE_POINTS: [f64; 3] = [-0.5, 0.0, 0.5];

#[derive(Debug, Clone)]
pub struct NodeAnomalyFeatures {
    pub node_id: usize,
    pub values: [f64; ANOMALY_FEATURE_DIM],
}

#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyError {
    EmptyDataset,
    ZeroVarianceFeature { column: usize },
    NonFiniteLoss { epoch: usize },
}

impl fmt::Display for AnomalyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnomalyError::EmptyDataset => write!(f, "cannot train on an empty node set"),
            AnomalyError::ZeroVarianceFeature { column } => write!(
                f,
                "feature column {column} has ~zero variance across the dataset; \
                 z-score normalization would divide by ~0"
            ),
            AnomalyError::NonFiniteLoss { epoch } => {
                write!(f, "training loss became non-finite at epoch {epoch}")
            }
        }
    }
}
impl std::error::Error for AnomalyError {}

/// Extracts one feature vector per node: Laplacian (x,y), degree relative
/// to graph mean, the node's Jacobi polynomial evaluated at three fixed
/// probe points, and the graph-level Ramanujan deviation broadcast to
/// every node. Identical to the original burn-based version.
pub fn extract_anomaly_features(
    nodes: &[SpectralNode],
    graph: &CsrGraph,
    ramanujan: &RamanujanFeature,
) -> Result<Vec<NodeAnomalyFeatures>, SpectralError> {
    let mean_degree = ramanujan.mean_degree.max(1e-9);
    let mut out = Vec::with_capacity(nodes.len());
    for node in nodes {
        let degree = graph.degrees[node.id];
        let mut jacobi_vals = [0.0; 3];
        for (slot, &p) in JACOBI_PROBE_POINTS.iter().enumerate() {
            jacobi_vals[slot] = node.geometric_moments.eval(p)?;
        }
        out.push(NodeAnomalyFeatures {
            node_id: node.id,
            values: [
                node.embedding.x,
                node.embedding.y,
                degree / mean_degree,
                jacobi_vals[0],
                jacobi_vals[1],
                jacobi_vals[2],
                ramanujan.deviation,
            ],
        });
    }
    Ok(out)
}

/// Hypergraph analog of [`extract_anomaly_features`]: identical feature
/// layout (Laplacian x/y, degree relative to mean, three Jacobi probe
/// evaluations, graph-level Ramanujan deviation), sourcing `degree` from
/// `hg.vertex_degree` (weighted, hyperedge-aware) instead of `CsrGraph`'s
/// plain unweighted degree count.
pub fn extract_anomaly_features_hg(
    nodes: &[SpectralNode],
    hg: &SpectralHypergraph,
    ramanujan: &RamanujanFeature,
) -> Result<Vec<NodeAnomalyFeatures>, SpectralError> {
    let mean_degree = ramanujan.mean_degree.max(1e-9);
    let mut out = Vec::with_capacity(nodes.len());
    for node in nodes {
        let degree = hg.vertex_degree(VertexId(node.id))?;
        let mut jacobi_vals = [0.0; 3];
        for (slot, &p) in JACOBI_PROBE_POINTS.iter().enumerate() {
            jacobi_vals[slot] = node.geometric_moments.eval(p)?;
        }
        out.push(NodeAnomalyFeatures {
            node_id: node.id,
            values: [
                node.embedding.x,
                node.embedding.y,
                degree / mean_degree,
                jacobi_vals[0],
                jacobi_vals[1],
                jacobi_vals[2],
                ramanujan.deviation,
            ],
        });
    }
    Ok(out)
}

/// Per-column mean/std used to z-score features before they hit the
/// network, stored so the same normalization can be reapplied later to
/// score new nodes against a model trained once on a reference
/// population.
#[derive(Debug, Clone)]
pub struct FeatureNormalizer {
    mean: [f64; ANOMALY_FEATURE_DIM],
    std: [f64; ANOMALY_FEATURE_DIM],
}

impl FeatureNormalizer {
    pub fn fit(rows: &[NodeAnomalyFeatures]) -> Result<Self, AnomalyError> {
        if rows.is_empty() {
            return Err(AnomalyError::EmptyDataset);
        }
        let n = rows.len() as f64;
        let mut mean = [0.0; ANOMALY_FEATURE_DIM];
        for r in rows {
            for c in 0..ANOMALY_FEATURE_DIM {
                mean[c] += r.values[c];
            }
        }
        for m in &mut mean {
            *m /= n;
        }
        let mut std = [0.0; ANOMALY_FEATURE_DIM];
        for r in rows {
            for c in 0..ANOMALY_FEATURE_DIM {
                std[c] += (r.values[c] - mean[c]).powi(2);
            }
        }
        for c in 0..ANOMALY_FEATURE_DIM {
            std[c] = (std[c] / n).sqrt();
            if std[c] < 1e-9 {
                return Err(AnomalyError::ZeroVarianceFeature { column: c });
            }
        }
        Ok(Self { mean, std })
    }

    pub fn transform(&self, row: &NodeAnomalyFeatures) -> [f64; ANOMALY_FEATURE_DIM] {
        let mut out = [0.0; ANOMALY_FEATURE_DIM];
        for c in 0..ANOMALY_FEATURE_DIM {
            out[c] = (row.values[c] - self.mean[c]) / self.std[c];
        }
        out
    }
}

/// One dense layer plus its Adam moment buffers. Weight matrix is
/// (out_dim x in_dim) so forward is a plain `w * x + b` with samples laid
/// out as columns of `x` (feature_dim x n_samples) -- lets a whole batch
/// go through as one matmul instead of a per-sample loop.
struct Linear {
    w: DMatrix<f64>,
    b: DVector<f64>,
    m_w: DMatrix<f64>,
    v_w: DMatrix<f64>,
    m_b: DVector<f64>,
    v_b: DVector<f64>,
}

impl Linear {
    fn new(in_dim: usize, out_dim: usize, rng: &mut impl Rng) -> Self {
        // Xavier/Glorot uniform init, matching the range PyTorch-style
        // frameworks (Burn included) default `Linear` layers to.
        let limit = (6.0 / (in_dim + out_dim) as f64).sqrt();
        let w = DMatrix::from_fn(out_dim, in_dim, |_, _| rng.random::<f64>() * 2.0 * limit - limit);
        Self {
            w,
            b: DVector::zeros(out_dim),
            m_w: DMatrix::zeros(out_dim, in_dim),
            v_w: DMatrix::zeros(out_dim, in_dim),
            m_b: DVector::zeros(out_dim),
            v_b: DVector::zeros(out_dim),
        }
    }

    fn forward(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let mut z = &self.w * x;
        for mut col in z.column_iter_mut() {
            col += &self.b;
        }
        z
    }

    /// Applies one Adam step given the loss gradients w.r.t. this layer's
    /// weight and bias, and returns dL/dx (in_dim x n_samples) so the
    /// caller can keep propagating backward through the previous layer.
    fn adam_step(
        &mut self,
        x: &DMatrix<f64>,
        grad_z: &DMatrix<f64>,
        lr: f64,
        t: i32,
    ) -> DMatrix<f64> {
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;

        let grad_w = grad_z * x.transpose();
        let grad_b = DVector::from_iterator(
            grad_z.nrows(),
            grad_z.row_iter().map(|r| r.iter().sum::<f64>()),
        );
        let grad_x = self.w.transpose() * grad_z;

        self.m_w = beta1 * &self.m_w + (1.0 - beta1) * &grad_w;
        self.v_w = beta2 * &self.v_w + (1.0 - beta2) * grad_w.component_mul(&grad_w);
        let m_hat_w = &self.m_w / (1.0 - beta1.powi(t));
        let v_hat_w = &self.v_w / (1.0 - beta2.powi(t));
        let update_w = m_hat_w.zip_map(&v_hat_w, |m, v| lr * m / (v.sqrt() + eps));
        self.w -= update_w;

        self.m_b = beta1 * &self.m_b + (1.0 - beta1) * &grad_b;
        self.v_b = beta2 * &self.v_b + (1.0 - beta2) * grad_b.component_mul(&grad_b);
        let m_hat_b = &self.m_b / (1.0 - beta1.powi(t));
        let v_hat_b = &self.v_b / (1.0 - beta2.powi(t));
        let update_b = m_hat_b.zip_map(&v_hat_b, |m, v| lr * m / (v.sqrt() + eps));
        self.b -= update_b;

        grad_x
    }
}

fn relu(z: &DMatrix<f64>) -> DMatrix<f64> {
    z.map(|v| v.max(0.0))
}

/// Elementwise ReLU derivative mask applied to an upstream gradient.
fn relu_backward(grad_out: &DMatrix<f64>, z: &DMatrix<f64>) -> DMatrix<f64> {
    grad_out.zip_map(z, |g, zv| if zv > 0.0 { g } else { 0.0 })
}

/// Small symmetric bottleneck autoencoder: 7 -> hidden -> bottleneck ->
/// hidden -> 7. Reconstruction error (not the bottleneck) is the anomaly
/// signal.
pub struct AnomalyAutoencoder {
    enc1: Linear,
    enc2: Linear,
    dec1: Linear,
    dec2: Linear,
}

impl AnomalyAutoencoder {
    pub fn new(hidden_dim: usize, bottleneck_dim: usize, rng: &mut impl Rng) -> Self {
        Self {
            enc1: Linear::new(ANOMALY_FEATURE_DIM, hidden_dim, rng),
            enc2: Linear::new(hidden_dim, bottleneck_dim, rng),
            dec1: Linear::new(bottleneck_dim, hidden_dim, rng),
            dec2: Linear::new(hidden_dim, ANOMALY_FEATURE_DIM, rng),
        }
    }

    pub fn forward(&self, input: &DMatrix<f64>) -> DMatrix<f64> {
        let h = relu(&self.enc1.forward(input));
        let z = relu(&self.enc2.forward(&h));
        let h2 = relu(&self.dec1.forward(&z));
        self.dec2.forward(&h2)
    }
}

#[derive(Debug, Clone)]
pub struct AnomalyTrainConfig {
    pub hidden_dim: usize,
    pub bottleneck_dim: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub seed: u64,
}

impl Default for AnomalyTrainConfig {
    fn default() -> Self {
        Self { hidden_dim: 5, bottleneck_dim: 3, epochs: 300, learning_rate: 1e-2, seed: 42 }
    }
}

fn rows_to_matrix(rows: &[[f64; ANOMALY_FEATURE_DIM]]) -> DMatrix<f64> {
    // feature_dim x n_samples, column-per-sample.
    DMatrix::from_fn(ANOMALY_FEATURE_DIM, rows.len(), |r, c| rows[c][r])
}

/// Trains the autoencoder on `rows` (already extracted via
/// `extract_anomaly_features`) with full-batch gradient descent (Adam) and
/// manually-derived backprop through the four linear layers. CPU-only, no
/// external ML framework -- matches the rest of this pipeline's
/// zero-external-service posture, just via a different route than the
/// original `burn` version.
pub fn train_anomaly_detector(
    rows: &[NodeAnomalyFeatures],
    config: &AnomalyTrainConfig,
) -> Result<(AnomalyAutoencoder, FeatureNormalizer), AnomalyError> {
    if rows.is_empty() {
        return Err(AnomalyError::EmptyDataset);
    }
    let normalizer = FeatureNormalizer::fit(rows)?;
    let normalized: Vec<[f64; ANOMALY_FEATURE_DIM]> =
        rows.iter().map(|r| normalizer.transform(r)).collect();
    let input = rows_to_matrix(&normalized);
    let n_samples = rows.len() as f64;

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut model = AnomalyAutoencoder::new(config.hidden_dim, config.bottleneck_dim, &mut rng);

    for epoch in 0..config.epochs {
        // ---- forward, caching every pre-/post-activation needed for backprop ----
        let z1 = model.enc1.forward(&input);
        let h1 = relu(&z1);
        let z2 = model.enc2.forward(&h1);
        let bottleneck = relu(&z2);
        let z3 = model.dec1.forward(&bottleneck);
        let h2 = relu(&z3);
        let reconstruction = model.dec2.forward(&h2);

        let diff = &reconstruction - &input;
        let loss_value: f64 = diff.iter().map(|d| d * d).sum::<f64>()
            / (ANOMALY_FEATURE_DIM as f64 * n_samples);
        if !loss_value.is_finite() {
            return Err(AnomalyError::NonFiniteLoss { epoch });
        }

        // ---- backward: dL/d(reconstruction) for mean-squared-error ----
        let grad_scale = 2.0 / (ANOMALY_FEATURE_DIM as f64 * n_samples);
        let grad_out = diff.map(|d| d * grad_scale);

        let t = (epoch + 1) as i32;
        let grad_h2 = model.dec2.adam_step(&h2, &grad_out, config.learning_rate, t);
        let grad_z3 = relu_backward(&grad_h2, &z3);
        let grad_bottleneck = model.dec1.adam_step(&bottleneck, &grad_z3, config.learning_rate, t);
        let grad_z2 = relu_backward(&grad_bottleneck, &z2);
        let grad_h1 = model.enc2.adam_step(&h1, &grad_z2, config.learning_rate, t);
        let grad_z1 = relu_backward(&grad_h1, &z1);
        let _grad_input = model.enc1.adam_step(&input, &grad_z1, config.learning_rate, t);
    }

    Ok((model, normalizer))
}

#[derive(Debug, Clone)]
pub struct NodeAnomalyScore {
    pub node_id: usize,
    pub reconstruction_error: f64,
    pub z_score: f64,
}

/// Scores `rows` against a trained model: reconstruction MSE per node,
/// plus a z-score relative to the batch being scored.
pub fn score_nodes(
    model: &AnomalyAutoencoder,
    normalizer: &FeatureNormalizer,
    rows: &[NodeAnomalyFeatures],
) -> Result<Vec<NodeAnomalyScore>, AnomalyError> {
    if rows.is_empty() {
        return Err(AnomalyError::EmptyDataset);
    }
    let normalized: Vec<[f64; ANOMALY_FEATURE_DIM]> =
        rows.iter().map(|r| normalizer.transform(r)).collect();
    let input = rows_to_matrix(&normalized);
    let reconstruction = model.forward(&input);
    let diff = &reconstruction - &input;

    let errors: Vec<f64> = diff
        .column_iter()
        .map(|col| col.iter().map(|v| v * v).sum::<f64>() / ANOMALY_FEATURE_DIM as f64)
        .collect();

    let n = errors.len() as f64;
    let mean = errors.iter().sum::<f64>() / n;
    let variance = errors.iter().map(|&e| (e - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt().max(1e-9);

    Ok(rows
        .iter()
        .zip(errors.iter())
        .map(|(row, &err)| NodeAnomalyScore {
            node_id: row.node_id,
            reconstruction_error: err,
            z_score: (err - mean) / std,
        })
        .collect())
}

// =========================================================================
// SYSTEM EXECUTION ENGINE
// =========================================================================

#[derive(Debug)]
pub enum PipelineError {
    Spectral(SpectralError),
    Anomaly(AnomalyError),
}
impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineError::Spectral(e) => write!(f, "{e}"),
            PipelineError::Anomaly(e) => write!(f, "{e}"),
        }
    }
}
impl From<SpectralError> for PipelineError {
    fn from(e: SpectralError) -> Self {
        PipelineError::Spectral(e)
    }
}
impl From<AnomalyError> for PipelineError {
    fn from(e: AnomalyError) -> Self {
        PipelineError::Anomaly(e)
    }
}

/// Builds a small corpus of distinct graphs so the pooled feature set fed
/// to the autoencoder has real cross-node variance in every column --
/// including feature index 6 (graph-level Ramanujan deviation broadcast to
/// every node of a graph). On a *single* graph that column is identical
/// for all its nodes by construction, which is exactly what triggered
/// `AnomalyError::ZeroVarianceFeature` below when Step 7 first ran against
/// only the Step 1 demo graph -- a real, previously-unexercised bug this
/// execution surfaced (this file could never actually run past Step 6
/// before, since `burn` wouldn't compile here). The `anomaly` module's own
/// doc comment already says this signal "becomes meaningful once nodes
/// number in the hundreds+ (e.g. per-time-window connection graphs)" --
/// this corpus is a minimal stand-in for that multi-snapshot population,
/// not a fix to the zero-variance check itself (which is correct: it's
/// refusing to divide by ~0 rather than silently emitting NaN/Inf).
/// Node ids are offset by `graph_index * 1000` to stay unique across the
/// pooled set; `labels` maps each offset id back to (graph name,
/// original node id) for readable output.
fn build_anomaly_demo_corpus() -> Result<(Vec<NodeAnomalyFeatures>, HashMap<usize, (&'static str, usize)>), PipelineError> {
    let graph_specs: Vec<(&'static str, usize, Vec<(usize, usize)>)> = vec![
        (
            "triangle+4cycle (Step 1 graph)",
            6,
            vec![(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)],
        ),
        (
            // Q3 hypercube: 3-regular, bipartite, and -- unlike K4 -- its
            // Laplacian has 4 distinct eigenvalues {0,2,4,6} rather than
            // just {0,n}. That matters here: single-vector Lanczos can only
            // resolve as many distinct Ritz values as the minimal
            // polynomial degree of the (generic) starting vector, which for
            // K4's 2-eigenvalue spectrum is exactly 2 -- one short of the
            // 3 `build_spectral_nodes` requires. K4 was tried first and hit
            // exactly that: `InsufficientEigenpairs { requested: 3,
            // available: 2 }`, a second real limitation this run surfaced.
            "Q3 hypercube (3-regular, bipartite)",
            8,
            vec![
                (0, 1), (0, 2), (0, 4),
                (1, 3), (1, 5),
                (2, 3), (2, 6),
                (3, 7),
                (4, 5), (4, 6),
                (5, 7),
                (6, 7),
            ],
        ),
        ("C6 (6-cycle, 2-regular, bipartite)", 6, vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]),
        ("P6 (path, irregular)", 6, vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
        ("S5 (star, one hub degree-5)", 6, vec![(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]),
    ];

    let mut pooled = Vec::new();
    let mut labels = HashMap::new();

    for (graph_idx, (label, n, edges)) in graph_specs.into_iter().enumerate() {
        let g = CsrGraph::from_edges(n, &edges).map_err(SpectralError::from)?;
        let nodes = build_spectral_nodes(&g, 3, 42)?;
        let ram = compute_ramanujan_feature(&g, 42)?;
        let rows = extract_anomaly_features(&nodes, &g, &ram)?;
        for mut row in rows {
            let original_id = row.node_id;
            let offset_id = graph_idx * 1000 + original_id;
            labels.insert(offset_id, (label, original_id));
            row.node_id = offset_id;
            pooled.push(row);
        }
    }

    Ok((pooled, labels))
}

/// Builds a `SpectralHypergraph` from a plain vertex count plus a list of
/// member-index sets, one call site both `build_hypergraph_anomaly_demo_corpus`
/// and the STEP 8 walkthrough in `run()` share so the two stay in sync.
fn build_demo_hypergraph(n: usize, hyperedges: &[Vec<usize>]) -> Result<SpectralHypergraph, SpectralError> {
    let mut builder = HypergraphBuilder::with_capacity(n, hyperedges.len(), 3);
    let ids: Vec<VertexId> = (0..n)
        .map(|i| builder.add_vertex(format!("v{i}")))
        .collect::<Result<_, _>>()?;
    for members in hyperedges {
        let vids: Vec<VertexId> = members.iter().map(|&i| ids[i]).collect();
        builder.add_hyperedge(&vids, 1.0)?;
    }
    Ok(builder.build()?)
}

/// Hypergraph counterpart to `build_anomaly_demo_corpus`: same "pool several
/// small graphs so every feature column has real cross-node variance"
/// rationale (see that function's doc comment -- the graph-level Ramanujan
/// deviation broadcast to every node of a *single* hypergraph is exactly as
/// zero-variance as it is for a single plain graph), but every hyperedge
/// here has genuine cardinality > 2 in at least one hyperedge per
/// hypergraph, so this actually exercises `spectral_hypergraph`'s hyperedge
/// machinery rather than just re-deriving the plain-graph case through the
/// `csr_graph_to_hypergraph` embedding.
fn build_hypergraph_anomaly_demo_corpus(
) -> Result<(Vec<NodeAnomalyFeatures>, HashMap<usize, (&'static str, usize)>), PipelineError> {
    let hg_specs: Vec<(&'static str, usize, Vec<Vec<usize>>)> = vec![
        (
            "counterparty rings {0,1,2}/{3,4,5} + weak bridge {2,3}",
            6,
            vec![vec![0, 1, 2], vec![3, 4, 5], vec![2, 3]],
        ),
        (
            "shared-address hyperedge over all 6 accounts + 2 direct transfers",
            6,
            vec![vec![0, 1, 2, 3, 4, 5], vec![0, 1], vec![3, 4]],
        ),
        (
            "overlapping-triple counterparty ring",
            6,
            vec![
                vec![0, 1, 2],
                vec![1, 2, 3],
                vec![2, 3, 4],
                vec![3, 4, 5],
                vec![4, 5, 0],
                vec![5, 0, 1],
            ],
        ),
    ];

    let mut pooled = Vec::new();
    let mut labels = HashMap::new();

    for (hg_idx, (label, n, hyperedges)) in hg_specs.into_iter().enumerate() {
        let hg = build_demo_hypergraph(n, &hyperedges)?;
        let nodes = build_spectral_nodes_hg(&hg, 3, 42)?;
        let ram = compute_hypergraph_ramanujan_feature(&hg, 42)?;
        let rows = extract_anomaly_features_hg(&nodes, &hg, &ram)?;
        for mut row in rows {
            let original_id = row.node_id;
            let offset_id = hg_idx * 1000 + original_id;
            labels.insert(offset_id, (label, original_id));
            row.node_id = offset_id;
            pooled.push(row);
        }
    }

    Ok((pooled, labels))
}

pub fn run() -> Result<(), PipelineError> {
    println!("=== STEP 1: Building & Validating Sparse Graph Topology ===");
    let n = 6;
    let edges = vec![(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)];
    let graph = CsrGraph::from_edges(n, &edges).map_err(SpectralError::from)?;

    println!("\n=== STEP 2/3: Sparse Laplacian Matvec + Lanczos Krylov Solve ===");
    let spectral_nodes = build_spectral_nodes(&graph, 3, 42)?;

    println!("\n=== STEP 3.5: Graph-Level Ramanujan-Deviation Anomaly Signal ===");
    let ramanujan = compute_ramanujan_feature(&graph, 42)?;
    println!(
        "mean degree={:.3} regular={:?} bipartite={} bound=2*sqrt(q)={:.4}",
        ramanujan.mean_degree, ramanujan.degree_regular, ramanujan.bipartite, ramanujan.bound
    );
    println!(
        "extremal adjacency eigenvalues resolved: {:?}",
        ramanujan.extremal_eigenvalues.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>()
    );
    println!(
        "max |nontrivial mu| = {:.4}  deviation = {:.4}  ({})",
        ramanujan.max_nontrivial_abs,
        ramanujan.deviation,
        if ramanujan.is_heuristic {
            "heuristic: graph is not regular, bound uses mean degree"
        } else {
            "exact: graph is regular, bound is the proven Ramanujan theorem"
        }
    );

    println!("\n=== STEP 4: Jacobi Orthogonal-Space Bridging ===");
    for node in &spectral_nodes {
        match node.geometric_moments.eval(0.5) {
            Ok(v) => println!(
                "Node {} -> Coordinates: ({:>6.3}, {:>6.3}) | Poly eval at 0.5: {:>6.3}",
                node.id, node.embedding.x, node.embedding.y, v
            ),
            Err(e) => println!("Node {} -> polynomial eval failed: {e}", node.id),
        }
    }

    // Extracted here, before spectral_nodes is moved into HnswIndex::build
    // below.
    let feature_rows = extract_anomaly_features(&spectral_nodes, &graph, &ramanujan)?;

    println!("\n=== STEP 5: Building Bounded-Degree Incremental HNSW Index ===");
    let hnsw = HnswIndex::build(spectral_nodes, HnswConfig::default(), Some(42));
    println!("HNSW index built (bounded degree, O(n log n)-ish construction).");

    println!("\n=== STEP 6: Approximate Nearest Neighbor Search ===");
    let query_vector = Vector2::new(-0.42, -0.12);
    println!("Query point: {query_vector:?}");
    match hnsw.search(query_vector, 16) {
        Some(id) => println!(" -> Nearest match -> Node ID: {id}"),
        None => println!(" -> Index is empty, no match found"),
    }

    println!("\n=== STEP 7: Autoencoder Reconstruction-Error Anomaly Scoring ===");
    println!(
        "NOTE: {n} nodes vs. a network with more free parameters than that -- this WILL overfit. \
         Scores below are wired end-to-end but not a meaningful anomaly signal at this scale; \
         see the anomaly module doc comment.",
    );
    let train_config = AnomalyTrainConfig::default();
    match train_anomaly_detector(&feature_rows, &train_config) {
        Ok((model, normalizer)) => {
            let scores = score_nodes(&model, &normalizer, &feature_rows)?;
            for score in &scores {
                println!(
                    "Node {} -> reconstruction_error={:.6} z={:.3}{}",
                    score.node_id,
                    score.reconstruction_error,
                    score.z_score,
                    if score.z_score.abs() > 2.0 { "  <-- flagged" } else { "" }
                );
            }
        }
        Err(AnomalyError::ZeroVarianceFeature { column }) => {
            println!(
                "-> normalization correctly refused: feature column {column} (graph-level \
                 Ramanujan deviation, broadcast to every node) is identical across all nodes \
                 of a SINGLE graph, so z-scoring it would divide by ~0. This is the checked \
                 error working as designed, not a crash -- but it does mean Step 7 needs more \
                 than one graph's worth of nodes to be well-posed at all, not just to be a \
                 meaningful signal. Falling back to a small multi-graph demo corpus instead:"
            );
            let (pooled_rows, labels) = build_anomaly_demo_corpus()?;
            let (model, normalizer) = train_anomaly_detector(&pooled_rows, &train_config)?;
            let scores = score_nodes(&model, &normalizer, &pooled_rows)?;
            for score in &scores {
                let (label, original_id) = labels[&score.node_id];
                println!(
                    "[{label}] node {original_id} -> reconstruction_error={:.6} z={:.3}{}",
                    score.reconstruction_error,
                    score.z_score,
                    if score.z_score.abs() > 2.0 { "  <-- flagged" } else { "" }
                );
            }
        }
        Err(e) => return Err(PipelineError::from(e)),
    }

    println!("\n=== STEP 8: Same Pipeline Against a Genuine Hypergraph ===");
    println!(
        "Everything above ran on `CsrGraph` (plain pairwise edges). `spectral_hypergraph` \
         generalizes that to hyperedges of any cardinality >= 2; this step re-runs Steps \
         2-3.5 and 7 against a small hypergraph with one real 3-ary hyperedge, using the \
         `hg_bridge` module's `SymmetricOp` impl for `spectral_hypergraph`'s matrix-free \
         normalized-Laplacian operator (see file header) instead of `LaplacianOp`/`CsrGraph`."
    );
    let hg = build_demo_hypergraph(6, &[vec![0, 1, 2], vec![3, 4, 5], vec![2, 3]])?;
    println!(
        "hypergraph: {} vertices, {} hyperedges (two 3-way counterparty rings {{0,1,2}} \
         and {{3,4,5}} joined by a weak bridge hyperedge {{2,3}})",
        hg.num_vertices(),
        hg.num_hyperedges()
    );

    let hg_spectral_nodes = build_spectral_nodes_hg(&hg, 3, 42)?;
    let hg_ramanujan = compute_hypergraph_ramanujan_feature(&hg, 42)?;
    println!(
        "mean weighted degree={:.3} vertex-degree-regular={:?} bound=2*sqrt(q)={:.4} \
         (is_heuristic={} -- clique expansion of a real hyperedge is always a lossy \
         reduction, so this bound is never the exact Ramanujan theorem the way the \
         plain-graph K4 case above can be)",
        hg_ramanujan.mean_degree, hg_ramanujan.degree_regular, hg_ramanujan.bound, hg_ramanujan.is_heuristic
    );
    for node in &hg_spectral_nodes {
        println!(
            "Vertex {} -> Coordinates: ({:>6.3}, {:>6.3})",
            node.id, node.embedding.x, node.embedding.y
        );
    }

    let hg_feature_rows = extract_anomaly_features_hg(&hg_spectral_nodes, &hg, &hg_ramanujan)?;
    match train_anomaly_detector(&hg_feature_rows, &train_config) {
        Ok((model, normalizer)) => {
            let scores = score_nodes(&model, &normalizer, &hg_feature_rows)?;
            for score in &scores {
                println!(
                    "Vertex {} -> reconstruction_error={:.6} z={:.3}{}",
                    score.node_id,
                    score.reconstruction_error,
                    score.z_score,
                    if score.z_score.abs() > 2.0 { "  <-- flagged" } else { "" }
                );
            }
        }
        Err(AnomalyError::ZeroVarianceFeature { column }) => {
            println!(
                "-> normalization correctly refused: feature column {column} (graph-level \
                 Ramanujan deviation) is identical across every vertex of this SINGLE \
                 hypergraph, same root cause as Step 7 above. Falling back to a small \
                 multi-hypergraph demo corpus with genuine >2-ary hyperedges instead:"
            );
            let (pooled_rows, labels) = build_hypergraph_anomaly_demo_corpus()?;
            let (model, normalizer) = train_anomaly_detector(&pooled_rows, &train_config)?;
            let scores = score_nodes(&model, &normalizer, &pooled_rows)?;
            for score in &scores {
                let (label, original_id) = labels[&score.node_id];
                println!(
                    "[{label}] vertex {original_id} -> reconstruction_error={:.6} z={:.3}{}",
                    score.reconstruction_error,
                    score.z_score,
                    if score.z_score.abs() > 2.0 { "  <-- flagged" } else { "" }
                );
            }
        }
        Err(e) => return Err(PipelineError::from(e)),
    }

    Ok(())
}

#[cfg(feature = "nbsc_bridge")]
pub mod nbsc_bridge;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_out_of_bounds_edge() {
        let err = CsrGraph::from_edges(3, &[(0, 5)]).unwrap_err();
        assert_eq!(err, GraphError::IndexOutOfBounds { node: 5, num_nodes: 3 });
    }

    #[test]
    fn rejects_self_loop() {
        let err = CsrGraph::from_edges(3, &[(1, 1)]).unwrap_err();
        assert_eq!(err, GraphError::SelfLoop(1));
    }

    #[test]
    fn rejects_disconnected_graph() {
        let err = CsrGraph::from_edges(4, &[(0, 1), (2, 3)]).unwrap_err();
        assert_eq!(err, GraphError::Disconnected { components: 2 });
    }

    #[test]
    fn dedupes_repeated_edges() {
        let graph = CsrGraph::from_edges(3, &[(0, 1), (1, 0), (1, 2)]).unwrap();
        assert_eq!(graph.neighbors(0), &[1]);
    }

    #[test]
    fn laplacian_matches_dense_reference_on_triangle() {
        let graph = CsrGraph::from_edges(3, &[(0, 1), (1, 2), (2, 0)]).unwrap();
        let x = [1.0, 0.0, 0.0];
        let mut y = [0.0; 3];
        graph.laplacian_multiply(&x, &mut y);
        assert_eq!(y, [2.0, -1.0, -1.0]);
    }

    #[test]
    fn jacobi_rejects_out_of_domain() {
        let poly = JacobiPoly::new(0.0, 0.0, vec![1.0, 0.5]).unwrap();
        assert!(poly.eval(2.0).is_err());
        assert!(poly.eval(0.5).is_ok());
    }

    #[test]
    fn hnsw_finds_exact_nearest_on_small_set() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut index = HnswIndex::new(HnswConfig::default());
        let coords = [(0.0, 0.0), (1.0, 0.0), (5.0, 5.0), (5.1, 5.1)];
        for (i, &(x, y)) in coords.iter().enumerate() {
            let poly = JacobiPoly::new(0.0, 0.0, vec![1.0]).unwrap();
            index.insert(
                SpectralNode { id: i, embedding: Vector2::new(x, y), geometric_moments: poly },
                &mut rng,
            );
        }
        let found = index.search(Vector2::new(5.05, 5.05), 8).unwrap();
        assert!(found == 2 || found == 3);
    }

    #[test]
    fn lanczos_eigenvalues_match_dense_reference_solve() {
        let edges = vec![(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)];
        let n = 6;
        let graph = CsrGraph::from_edges(n, &edges).unwrap();

        let mut dense = DMatrix::<f64>::zeros(n, n);
        for &(u, v) in &edges {
            dense[(u, u)] += 1.0;
            dense[(v, v)] += 1.0;
            dense[(u, v)] -= 1.0;
            dense[(v, u)] -= 1.0;
        }
        let reference = nalgebra::linalg::SymmetricEigen::new(dense);
        let mut ref_vals: Vec<f64> = reference.eigenvalues.iter().copied().collect();
        ref_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let subspace = lanczos_subspace_size(n, 3);
        let solved = HermitianLanczos::new(&LaplacianOp(&graph), subspace, 42).unwrap();
        let got_vals: Vec<f64> = solved.eigenvalues.iter().take(3).copied().collect();

        assert!(got_vals[0].abs() < 1e-6, "smallest eigenvalue should be ~0, got {}", got_vals[0]);
        for i in 0..3 {
            assert!(
                (got_vals[i] - ref_vals[i]).abs() < 1e-4,
                "eigenvalue {i}: lanczos={} dense_reference={}",
                got_vals[i],
                ref_vals[i]
            );
        }
    }

    #[test]
    fn ramanujan_feature_complete_graph_has_zero_deviation() {
        let edges = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let graph = CsrGraph::from_edges(4, &edges).unwrap();
        let feature = compute_ramanujan_feature(&graph, 42).unwrap();

        assert_eq!(feature.degree_regular, Some(3));
        assert!(!feature.is_heuristic);
        assert!((feature.bound - 2.0 * 2.0_f64.sqrt()).abs() < 1e-9);
        assert!(
            feature.deviation < 1e-6,
            "K4 is a known Ramanujan graph, expected ~0 deviation, got {}",
            feature.deviation
        );
    }

    #[test]
    fn ramanujan_feature_irregular_graph_is_heuristic() {
        let edges = vec![(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)];
        let graph = CsrGraph::from_edges(6, &edges).unwrap();
        let feature = compute_ramanujan_feature(&graph, 42).unwrap();

        assert_eq!(feature.degree_regular, None);
        assert!(feature.is_heuristic);
        assert!(feature.mean_degree > 0.0);
        assert!(feature.bound >= 0.0);
    }

    #[test]
    fn feature_normalizer_rejects_zero_variance_column() {
        // Single graph -> Ramanujan deviation (column 6) is identical
        // across every node -> must be rejected, not silently divided.
        let edges = vec![(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)];
        let graph = CsrGraph::from_edges(6, &edges).unwrap();
        let nodes = build_spectral_nodes(&graph, 3, 42).unwrap();
        let ram = compute_ramanujan_feature(&graph, 42).unwrap();
        let rows = extract_anomaly_features(&nodes, &graph, &ram).unwrap();
        let err = FeatureNormalizer::fit(&rows).unwrap_err();
        assert_eq!(err, AnomalyError::ZeroVarianceFeature { column: 6 });
    }

    #[test]
    fn autoencoder_reduces_reconstruction_loss_over_training() {
        // Pooled multi-graph corpus (real cross-node variance in every
        // column) -> training should meaningfully cut mean squared
        // reconstruction error vs. an untrained (freshly initialized)
        // network scored on the same data.
        let (rows, _labels) = build_anomaly_demo_corpus().unwrap();

        let mut rng = StdRng::seed_from_u64(42);
        let normalizer = FeatureNormalizer::fit(&rows).unwrap();
        let normalized: Vec<[f64; ANOMALY_FEATURE_DIM]> =
            rows.iter().map(|r| normalizer.transform(r)).collect();
        let input = rows_to_matrix(&normalized);
        let untrained = AnomalyAutoencoder::new(5, 3, &mut rng);
        let untrained_mse = {
            let diff = untrained.forward(&input) - &input;
            diff.iter().map(|d| d * d).sum::<f64>() / diff.len() as f64
        };

        let config = AnomalyTrainConfig::default();
        let (model, normalizer2) = train_anomaly_detector(&rows, &config).unwrap();
        let scores = score_nodes(&model, &normalizer2, &rows).unwrap();
        let trained_mse = scores.iter().map(|s| s.reconstruction_error).sum::<f64>() / scores.len() as f64;

        assert!(
            trained_mse < untrained_mse,
            "training should reduce reconstruction error: untrained={untrained_mse:.6} trained={trained_mse:.6}"
        );
        assert_eq!(scores.len(), rows.len());
    }

    #[test]
    fn score_nodes_rejects_empty_input() {
        let rows: Vec<NodeAnomalyFeatures> = vec![];
        assert_eq!(FeatureNormalizer::fit(&rows).unwrap_err(), AnomalyError::EmptyDataset);
    }

    #[test]
    fn hnsw_build_is_deterministic_given_a_seed() {
        let make = |seed: u64| {
            let nodes: Vec<SpectralNode> = (0..20)
                .map(|i| SpectralNode {
                    id: i,
                    embedding: Vector2::new(i as f64 * 0.3, (i as f64 * 0.7).sin()),
                    geometric_moments: JacobiPoly::new(0.0, 0.0, vec![1.0]).unwrap(),
                })
                .collect();
            HnswIndex::build(nodes, HnswConfig::default(), Some(seed))
        };
        let a = make(123);
        let b = make(123);
        assert_eq!(a.search(Vector2::new(1.0, 1.0), 8), b.search(Vector2::new(1.0, 1.0), 8));
    }

    // ---------------------------------------------------------------
    // hg_bridge: does the seam into `spectral_hypergraph` actually hold?
    // ---------------------------------------------------------------

    /// `csr_graph_to_hypergraph` claims a *faithful* embedding for
    /// all-pairwise graphs, not merely a plausible-looking one: every
    /// hyperedge it creates has cardinality exactly 2, and
    /// `clique_expansion_adjacency` gives a weight-1 hyperedge of
    /// cardinality 2 a share of `1.0 / (2 - 1) = 1.0` on exactly the pair
    /// it connects -- i.e. the crate's own clique-expansion reduction of
    /// the embedded hypergraph must equal `graph`'s dense adjacency matrix
    /// exactly, not approximately.
    #[test]
    fn csr_graph_to_hypergraph_round_trips_through_clique_expansion() {
        let edges = vec![(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)];
        let n = 6;
        let graph = CsrGraph::from_edges(n, &edges).unwrap();
        let hg = csr_graph_to_hypergraph(&graph).unwrap();

        assert_eq!(hg.num_vertices(), n);
        assert_eq!(hg.num_hyperedges(), edges.len());

        let clique_adjacency = spectral_hypergraph::laplacian::clique_expansion_adjacency(&hg).unwrap();

        let mut expected = DMatrix::<f64>::zeros(n, n);
        for &(u, v) in &edges {
            expected[(u, v)] = 1.0;
            expected[(v, u)] = 1.0;
        }

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (clique_adjacency[(i, j)] - expected[(i, j)]).abs() < 1e-12,
                    "mismatch at ({i},{j}): got {} expected {}",
                    clique_adjacency[(i, j)],
                    expected[(i, j)]
                );
            }
        }
    }

    /// Cross-checks this file's own vendored `HermitianLanczos` (running
    /// against `spectral_hypergraph`'s `HypergraphOperator` via the
    /// `SymmetricOp` impl in `hg_bridge`) against the crate's own dense
    /// eigensolver on the exact same hypergraph. If the `SymmetricOp`
    /// wiring (the fully-qualified `LinearOperator` calls) were wrong --
    /// e.g. accidentally applying the operator twice, or transposing the
    /// vector layout -- the two eigenvalue sets would disagree.
    #[test]
    fn hermitian_lanczos_matches_dense_normalized_laplacian_on_hypergraph() {
        let mut b = HypergraphBuilder::new();
        let a = b.add_vertex("a").unwrap();
        let v = b.add_vertex("b").unwrap();
        let c = b.add_vertex("c").unwrap();
        let d = b.add_vertex("d").unwrap();
        let e = b.add_vertex("e").unwrap();
        let f = b.add_vertex("f").unwrap();
        b.add_hyperedge(&[a, v, c], 5.0).unwrap();
        b.add_hyperedge(&[d, e, f], 5.0).unwrap();
        b.add_hyperedge(&[c, d], 0.1).unwrap();
        let hg = b.build().unwrap();

        let dense = spectral_hypergraph::laplacian::dense_normalized_laplacian(&hg).unwrap();
        let dense_eig = spectral_hypergraph::spectral::dense_eigen(&dense);
        let mut ref_vals: Vec<f64> = dense_eig.eigenvalues.iter().copied().collect();
        ref_vals.sort_by(|x, y| x.partial_cmp(y).unwrap());

        let op = spectral_hypergraph::laplacian::HypergraphOperator::new(&hg).unwrap();
        let solved = HermitianLanczos::new(&op, 6, 42).unwrap();
        let got_vals: Vec<f64> = solved.eigenvalues.iter().copied().collect();

        // Full-reorthogonalization Lanczos resolves exactly one Ritz value
        // per *distinct* eigenvalue and deflates (stops early) once the
        // Krylov subspace built from a single starting vector becomes
        // invariant -- so on a hypergraph whose Laplacian has repeated
        // eigenvalues (this one does, from the {a,b,c}/{d,e,f} symmetry),
        // `got_vals` legitimately has fewer entries than the dense
        // reference's 6. What must hold is that every resolved value is a
        // real eigenvalue of the dense reference, not that the counts
        // match -- exactly the same caveat this file's own comments raise
        // for K4 in `ramanujan_feature_complete_graph_has_zero_deviation`'s
        // neighborhood.
        assert!(got_vals.len() >= 2, "expected at least the trivial + Fiedler eigenvalues");
        for &gv in &got_vals {
            let closest = ref_vals
                .iter()
                .fold(f64::INFINITY, |acc, &rv| acc.min((rv - gv).abs()));
            assert!(closest < 1e-6, "resolved value {gv} has no match in dense reference {ref_vals:?}");
        }
    }

    /// `build_spectral_nodes_hg`'s 2D embedding should agree in *sign
    /// structure* with the crate's own `fiedler_vector` on a hypergraph
    /// built from two well-separated cliques joined by a weak bridge: both
    /// are reading off the same underlying eigenvector (up to sign/scale),
    /// so nodes {a,b,c} must land on one side and {d,e,f} on the other.
    #[test]
    fn build_spectral_nodes_hg_separates_two_hyperedge_cliques() {
        let mut b = HypergraphBuilder::new();
        let a = b.add_vertex("a").unwrap();
        let v = b.add_vertex("b").unwrap();
        let c = b.add_vertex("c").unwrap();
        let d = b.add_vertex("d").unwrap();
        let e = b.add_vertex("e").unwrap();
        let f = b.add_vertex("f").unwrap();
        b.add_hyperedge(&[a, v, c], 5.0).unwrap();
        b.add_hyperedge(&[d, e, f], 5.0).unwrap();
        b.add_hyperedge(&[c, d], 0.1).unwrap();
        let hg = b.build().unwrap();

        let fiedler = spectral_hypergraph::spectral::fiedler_vector(&hg).unwrap();
        let nodes = build_spectral_nodes_hg(&hg, 3, 42).unwrap();

        let fiedler_sign = |i: usize| fiedler[i].signum();
        let embedding_sign = |i: usize| nodes[i].embedding.x.signum();

        // Both readings must agree *within themselves* on which side of the
        // bridge each clique falls -- whether that side is "positive" in
        // this particular run is an artifact of an arbitrary eigenvector
        // sign flip, not a claim either function makes.
        assert_eq!(fiedler_sign(0), fiedler_sign(1));
        assert_eq!(fiedler_sign(1), fiedler_sign(2));
        assert_eq!(fiedler_sign(3), fiedler_sign(4));
        assert_eq!(fiedler_sign(4), fiedler_sign(5));
        assert_ne!(fiedler_sign(0), fiedler_sign(3));

        assert_eq!(embedding_sign(0), embedding_sign(1));
        assert_eq!(embedding_sign(1), embedding_sign(2));
        assert_eq!(embedding_sign(3), embedding_sign(4));
        assert_eq!(embedding_sign(4), embedding_sign(5));
        assert_ne!(embedding_sign(0), embedding_sign(3));
    }

    /// `compute_hypergraph_ramanujan_feature` must always report
    /// `is_heuristic = true` and `bipartite = false` -- unlike the
    /// plain-graph version, there is no exact case here, since clique
    /// expansion is a lossy reduction even for a vertex-degree-regular
    /// hypergraph. This test pins that contract down explicitly so a
    /// future edit can't silently start claiming exactness.
    #[test]
    fn hypergraph_ramanujan_feature_is_always_heuristic() {
        let mut b = HypergraphBuilder::new();
        let a = b.add_vertex("a").unwrap();
        let v = b.add_vertex("b").unwrap();
        let c = b.add_vertex("c").unwrap();
        let d = b.add_vertex("d").unwrap();
        b.add_hyperedge(&[a, v, c], 1.0).unwrap();
        b.add_hyperedge(&[a, v, d], 1.0).unwrap();
        b.add_hyperedge(&[a, c, d], 1.0).unwrap();
        b.add_hyperedge(&[v, c, d], 1.0).unwrap();
        let hg = b.build().unwrap();

        let feature = compute_hypergraph_ramanujan_feature(&hg, 42).unwrap();
        assert!(feature.is_heuristic);
        assert!(!feature.bipartite);
        assert_eq!(feature.degree_regular, Some(3));
        assert!(feature.bound >= 0.0);
    }

    /// Same zero-variance guard as `feature_normalizer_rejects_zero_variance_column`,
    /// but on the hypergraph feature path: the graph-level Ramanujan
    /// deviation broadcast to every vertex of one hypergraph is identical
    /// across all its vertices, same as the plain-graph case.
    #[test]
    fn hg_feature_normalizer_rejects_zero_variance_column() {
        let hg = build_demo_hypergraph(6, &[vec![0, 1, 2], vec![3, 4, 5], vec![2, 3]]).unwrap();
        let nodes = build_spectral_nodes_hg(&hg, 3, 42).unwrap();
        let ram = compute_hypergraph_ramanujan_feature(&hg, 42).unwrap();
        let rows = extract_anomaly_features_hg(&nodes, &hg, &ram).unwrap();
        let err = FeatureNormalizer::fit(&rows).unwrap_err();
        assert_eq!(err, AnomalyError::ZeroVarianceFeature { column: 6 });
    }

    /// End-to-end: the pooled hypergraph demo corpus (real cross-vertex
    /// variance in every column, including the Ramanujan-deviation one)
    /// should let the same autoencoder/training code Step 7 uses train
    /// successfully and cut reconstruction error, exactly like the
    /// plain-graph corpus does.
    #[test]
    fn hypergraph_autoencoder_reduces_reconstruction_loss_over_training() {
        let (rows, _labels) = build_hypergraph_anomaly_demo_corpus().unwrap();

        let mut rng = StdRng::seed_from_u64(42);
        let normalizer = FeatureNormalizer::fit(&rows).unwrap();
        let normalized: Vec<[f64; ANOMALY_FEATURE_DIM]> =
            rows.iter().map(|r| normalizer.transform(r)).collect();
        let input = rows_to_matrix(&normalized);
        let untrained = AnomalyAutoencoder::new(5, 3, &mut rng);
        let untrained_mse = {
            let diff = untrained.forward(&input) - &input;
            diff.iter().map(|d| d * d).sum::<f64>() / diff.len() as f64
        };

        let config = AnomalyTrainConfig::default();
        let (model, normalizer2) = train_anomaly_detector(&rows, &config).unwrap();
        let scores = score_nodes(&model, &normalizer2, &rows).unwrap();
        let trained_mse = scores.iter().map(|s| s.reconstruction_error).sum::<f64>() / scores.len() as f64;

        assert!(
            trained_mse < untrained_mse,
            "training should reduce reconstruction error: untrained={untrained_mse:.6} trained={trained_mse:.6}"
        );
        assert_eq!(scores.len(), rows.len());
    }
}

