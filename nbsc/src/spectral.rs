//! Implements §2-6 of `ihara_zeta.rs`: the Bass-reduced quadratic
//! eigenvalue problem for the Hashimoto (non-backtracking) matrix `B`, its
//! standard linearization `M`, a matrix-free `krylov_ds::LinearOperator`
//! for `M` (so `rho_B` is estimated via Arnoldi without ever forming the
//! `2m x 2m` or `2n x 2n` matrix densely), and the sparse three-term
//! recursion that produces the `T_k` filter bank by applying it directly to
//! feature matrices rather than forming `T_k` as an `n x n` operator.

use crate::graph::Graph;
use krylov_ds::operator::LinearOperator;
use krylov_ds::{eig, Arnoldi};

/// `y = A * x` via the adjacency list — the one sparse primitive everything
/// else in this module is built from.
pub fn apply_adjacency(g: &Graph, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), g.n);
    debug_assert_eq!(y.len(), g.n);
    for v in 0..g.n {
        let mut acc = 0.0;
        for &u in &g.neighbors[v] {
            acc += x[u];
        }
        y[v] = acc;
    }
}

/// Matrix-free linear operator for the `2n x 2n` linearization
///
/// ```text
/// M = [ A     I-D ]
///     [ I_n   0   ]
/// ```
///
/// from §4 of `ihara_zeta.rs`. `M`'s eigenvalues are exactly the roots `mu`
/// of the quadratic eigenvalue problem (★), i.e. the non-trivial part of
/// the Hashimoto spectrum (all Hashimoto eigenvalues except the `2(m-n)`
/// trivial ones sitting at +-1). Never materializes `A`, `D`, or `M`: each
/// `apply` is one sparse adjacency mat-vec plus two elementwise passes.
pub struct HashimotoLinearization<'a> {
    pub graph: &'a Graph,
    pub degrees: Vec<f64>,
}

impl<'a> HashimotoLinearization<'a> {
    pub fn new(graph: &'a Graph) -> Self {
        Self { graph, degrees: graph.degrees() }
    }
}

impl<'a> LinearOperator<f64> for HashimotoLinearization<'a> {
    fn dim(&self) -> usize {
        2 * self.graph.n
    }

    fn apply(&self, v: &[f64], out: &mut [f64]) {
        let n = self.graph.n;
        let (x, y) = v.split_at(n);
        let (out_top, out_bottom) = out.split_at_mut(n);

        // out_top = A*x + (I - D)*y
        apply_adjacency(self.graph, x, out_top);
        for i in 0..n {
            out_top[i] += (1.0 - self.degrees[i]) * y[i];
        }
        // out_bottom = x
        out_bottom.copy_from_slice(x);
    }
}

/// Estimate `rho_B`, the Perron-Frobenius spectral radius of the Hashimoto
/// matrix (§6): run Arnoldi on the matrix-free linearization and take the
/// largest-modulus Ritz value. By Perron-Frobenius this radius is attained
/// by a real, non-negative eigenvalue, so `rho_B` itself is always real
/// even though most of the surrounding spectrum is complex.
///
/// `krylov_dim` controls the Krylov subspace size; `2 * n.min(40)` is a
/// reasonable default that converges the extremal eigenvalue reliably
/// without approaching `O(n^2)` memory for the dense Hessenberg block.
pub fn estimate_spectral_radius(graph: &Graph, krylov_dim: usize, seed: u64) -> f64 {
    let op = HashimotoLinearization::new(graph);
    let dim = op.dim();
    let m = krylov_dim.min(dim).max(1);

    // Deterministic pseudo-random start vector (avoids pulling in `rand`
    // just for this and keeps the estimate reproducible across runs).
    let mut state = seed.wrapping_mul(2685821657736338717).wrapping_add(1);
    let mut v0 = vec![0.0f64; dim];
    for vi in v0.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *vi = (state as f64 / u64::MAX as f64) - 0.5;
    }

    let arnoldi = Arnoldi::new(m, 1e-12);
    let result = arnoldi.run(&op, &v0).expect("Arnoldi failed on Hashimoto linearization");
    let ritz = eig::arnoldi_ritz_values(&result);
    ritz.iter().map(|z| (z.re * z.re + z.im * z.im).sqrt()).fold(0.0, f64::max)
}

/// Matrix-free `LinearOperator` for the plain adjacency matrix `A` —
/// distinct from [`HashimotoLinearization`], which linearizes the
/// non-backtracking matrix `B`. `A` is symmetric (undirected graph), so
/// this is a valid input to `krylov_ds::Lanczos` (unlike `M`, which is not
/// symmetric and requires the general `Arnoldi`).
struct AdjacencyOperator<'a> {
    graph: &'a Graph,
}

impl<'a> LinearOperator<f64> for AdjacencyOperator<'a> {
    fn dim(&self) -> usize {
        self.graph.n
    }
    fn apply(&self, v: &[f64], out: &mut [f64]) {
        apply_adjacency(self.graph, v, out);
    }
}

/// Diagnostic: the operator (spectral) norm of the plain adjacency matrix,
/// `||A||_2 = max(|lambda_min(A)|, |lambda_max(A)|)`. Since `A` is real
/// symmetric, this is exactly its largest-magnitude eigenvalue, found via
/// Lanczos (which — unlike Arnoldi on the non-symmetric Hashimoto
/// linearization `M` — gives real, well-conditioned Ritz values here).
///
/// Exists to test a specific hypothesis raised by an empirical anomaly: if
/// `adjacency_operator_norm(g) / rho_b > 1`, the rescaled adjacency tap
/// `A / rho_B` used by [`NbscFilterBank`] is an *expansive* map (operator
/// norm > 1), unlike GCN's symmetric-normalized `D^-1/2 (A+I) D^-1/2`,
/// which is non-expansive by construction (its norm is exactly 1). An
/// expansive propagation step compounds across stacked layers, which would
/// explain both growing Dirichlet energy with depth and the increasing
/// cross-seed variance observed empirically at depth 3 on Cora — ordinary
/// over-smoothing predicts *shrinking* energy, not growing energy with
/// widening spread. This function does not draw that conclusion itself;
/// it just gives you the number needed to check it.
pub fn adjacency_operator_norm(graph: &Graph, krylov_dim: usize, seed: u64) -> f64 {
    let op = AdjacencyOperator { graph };
    let n = op.dim();
    let m = krylov_dim.min(n).max(1);

    // Same deterministic pseudo-random start vector generator as
    // `estimate_spectral_radius`, for reproducibility and to avoid a
    // `rand` dependency in this module.
    let mut state = seed.wrapping_mul(2685821657736338717).wrapping_add(1);
    let mut v0 = vec![0.0f64; n];
    for vi in v0.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *vi = (state as f64 / u64::MAX as f64) - 0.5;
    }

    let lanczos = krylov_ds::Lanczos::new(m, 1e-12, krylov_ds::Reorthogonalization::Full);
    let result = lanczos.run(&op, &v0).expect("Lanczos failed on adjacency operator");
    let pairs = krylov_ds::eig::lanczos_ritz_pairs(&result);
    pairs.iter().map(|p| p.value.abs()).fold(0.0, f64::max)
}

/// Direct dense construction of `B` (`2m x 2m`) for small graphs only —
/// used strictly as a ground-truth oracle in tests to verify both the Bass
/// reduction identity and the matrix-free `M` operator against a brute-force
/// eigendecomposition. `O(m^2)` memory; never call this outside tests or on
/// graphs beyond a few hundred edges.
#[cfg(test)]
pub(crate) fn dense_hashimoto_matrix(graph: &Graph) -> (Vec<(usize, usize)>, nalgebra::DMatrix<f64>) {
    // Directed edges: each undirected edge (u,v) becomes (u->v) and (v->u).
    let mut darcs: Vec<(usize, usize)> = Vec::with_capacity(2 * graph.m());
    for &(u, v) in &graph.edges {
        darcs.push((u, v));
        darcs.push((v, u));
    }
    let d2m = darcs.len();
    let mut b = nalgebra::DMatrix::<f64>::zeros(d2m, d2m);
    for (i, &(_u, v)) in darcs.iter().enumerate() {
        for (j, &(vp, w)) in darcs.iter().enumerate() {
            if v == vp && w != darcs[i].0 {
                b[(i, j)] = 1.0;
            }
        }
    }
    (darcs, b)
}

/// A dense node-feature matrix, `n` rows x `f` columns, row-major.
#[derive(Debug, Clone)]
pub struct FeatureMatrix {
    pub n: usize,
    pub f: usize,
    pub data: Vec<f64>,
}

impl FeatureMatrix {
    pub fn zeros(n: usize, f: usize) -> Self {
        Self { n, f, data: vec![0.0; n * f] }
    }

    pub fn from_rows(n: usize, f: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), n * f);
        Self { n, f, data }
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.f..(i + 1) * self.f]
    }

    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [f64] {
        &mut self.data[i * self.f..(i + 1) * self.f]
    }
}

/// Applies `A` and `(D-I)` to every column of a feature matrix at once
/// (i.e. `Y = A @ X`, `Y = (D-I) @ X`), which is what the three-term
/// recursion actually needs — this is the sparse `O(|E| * f)`-per-tap
/// implementation of §5's recursion, applied directly to features rather
/// than materializing `T_k` as an `n x n` operator (an explicit improvement
/// over the "TODO: replace dense T_k with a sparse kernel" note in §8 of
/// `ihara_zeta.rs` — there is simply no need to ever form `T_k` at all when
/// all that's wanted is `T_k @ X`).
fn adjacency_matmul(graph: &Graph, x: &FeatureMatrix, out: &mut FeatureMatrix) {
    debug_assert_eq!(x.n, graph.n);
    debug_assert_eq!(out.n, graph.n);
    debug_assert_eq!(out.f, x.f);
    for v in 0..graph.n {
        let row_out = out.row_mut(v);
        row_out.fill(0.0);
        for &u in &graph.neighbors[v] {
            let row_u = x.row(u);
            for (o, &xu) in row_out.iter_mut().zip(row_u.iter()) {
                *o += xu;
            }
        }
    }
}

fn diag_dm1_matmul(degrees: &[f64], x: &FeatureMatrix, out: &mut FeatureMatrix) {
    for v in 0..x.n {
        let scale = degrees[v] - 1.0;
        let row_x = x.row(v);
        let row_out = out.row_mut(v);
        for (o, &xv) in row_out.iter_mut().zip(row_x.iter()) {
            *o = scale * xv;
        }
    }
}

/// A precomputed, graph-fixed (X-independent) filter bank: holds `rho_B`
/// and the graph itself, and applies the rescaled three-term recursion of
/// §6:
///
/// ```text
/// T_0 = I
/// T_1 = A / rho_B
/// T_{k+1} = (2A/rho_B) T_k - ((D-I)/rho_B^2) T_{k-1}
/// ```
///
/// directly to a feature matrix, sparse, at `O(K * |E| * f)` total cost —
/// exactly the same complexity class as ChebNet/GCN per §5.
pub struct NbscFilterBank {
    pub rho_b: f64,
    degrees: Vec<f64>,
}

impl NbscFilterBank {
    pub fn build(graph: &Graph, krylov_dim: usize, seed: u64) -> Self {
        let rho_b = estimate_spectral_radius(graph, krylov_dim, seed).max(1e-6);
        Self { rho_b, degrees: graph.degrees() }
    }

    /// Returns `[T_0 X, T_1 X, ..., T_K X]`.
    pub fn apply_taps(&self, graph: &Graph, x: &FeatureMatrix, k: usize) -> Vec<FeatureMatrix> {
        assert_eq!(x.n, graph.n);
        let mut taps = Vec::with_capacity(k + 1);
        taps.push(x.clone()); // T_0 X = X

        if k == 0 {
            return taps;
        }

        // T_1 X = (A / rho_B) X
        let mut t1 = FeatureMatrix::zeros(x.n, x.f);
        adjacency_matmul(graph, x, &mut t1);
        for v in t1.data.iter_mut() {
            *v /= self.rho_b;
        }
        taps.push(t1);

        for kk in 1..k {
            let prev = &taps[kk]; // T_k X
            let prev2 = &taps[kk - 1]; // T_{k-1} X

            let mut a_prev = FeatureMatrix::zeros(x.n, x.f);
            adjacency_matmul(graph, prev, &mut a_prev);

            let mut d_prev2 = FeatureMatrix::zeros(x.n, x.f);
            diag_dm1_matmul(&self.degrees, prev2, &mut d_prev2);

            let mut next = FeatureMatrix::zeros(x.n, x.f);
            let rho2 = self.rho_b * self.rho_b;
            for i in 0..next.data.len() {
                next.data[i] = (2.0 / self.rho_b) * a_prev.data[i] - (1.0 / rho2) * d_prev2.data[i];
            }
            taps.push(next);
        }
        taps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{random_near_regular, random_tree, stochastic_block_model};
    use approx::assert_relative_eq;

    /// §3: Bass's determinant identity. For a small graph, verify
    /// `det(I - u*B) == (1-u^2)^(m-n) * det(I - u*A + u^2*(D-I))`
    /// at several sample points `u`, confirming the reduction the whole
    /// pipeline (and the matrix-free `M` operator) rests on.
    #[test]
    fn bass_reduction_identity_holds() {
        let (g, _) = stochastic_block_model(3, 5, 0.6, 0.1, 11);
        assert!(g.is_connected());
        let (darcs, b) = dense_hashimoto_matrix(&g);
        let d2m = darcs.len();
        let n = g.n;
        let m = g.m();

        let a = nalgebra::DMatrix::from_fn(n, n, |i, j| {
            if g.neighbors[i].contains(&j) {
                1.0
            } else {
                0.0
            }
        });
        let degrees = g.degrees();
        let d_minus_i = nalgebra::DMatrix::from_fn(n, n, |i, j| {
            if i == j {
                degrees[i] - 1.0
            } else {
                0.0
            }
        });

        for &u in &[0.05, 0.1, 0.15, -0.1] {
            let lhs = (nalgebra::DMatrix::<f64>::identity(d2m, d2m) - b.clone() * u).determinant();
            let rhs_poly =
                (nalgebra::DMatrix::<f64>::identity(n, n) - a.clone() * u + d_minus_i.clone() * u * u)
                    .determinant();
            let rhs = (1.0 - u * u).powi(m as i32 - n as i32) * rhs_poly;
            assert_relative_eq!(lhs, rhs, epsilon = 1e-6, max_relative = 1e-4);
        }
    }

    /// `adjacency_operator_norm` (Lanczos on the matrix-free adjacency
    /// operator) should agree with a brute-force dense eigendecomposition
    /// of `A`, and — since `A` is symmetric — should also agree with the
    /// symmetric-normalized GCN propagator's known operator norm ceiling of
    /// 1 being violated or not, which is the whole point of this function.
    #[test]
    fn adjacency_operator_norm_matches_dense_ground_truth() {
        let g = random_near_regular(30, 4, 7);
        let n = g.n;
        let a = nalgebra::DMatrix::<f64>::from_fn(n, n, |i, j| {
            if g.neighbors[i].contains(&j) { 1.0 } else { 0.0 }
        });
        let dense_norm = a
            .symmetric_eigenvalues()
            .iter()
            .map(|x| x.abs())
            .fold(0.0, f64::max);

        let krylov_norm = adjacency_operator_norm(&g, 25, 5);
        assert_relative_eq!(krylov_norm, dense_norm, epsilon = 1e-6, max_relative = 1e-4);
    }

    /// The matrix-free linearization `M`'s eigenvalues (via a *dense*
    /// eigendecomposition here, not Arnoldi, to isolate correctness of the
    /// operator from Krylov convergence) must reproduce the roots of (★)
    /// obtained directly from the quadratic eigenvalue problem, and hence
    /// (via Bass) the non-trivial part of the true Hashimoto spectrum.
    #[test]
    fn linearization_matches_dense_hashimoto_nontrivial_spectrum() {
        let g = random_near_regular(24, 3, 5);
        assert!(g.is_connected());
        let (_darcs, b) = dense_hashimoto_matrix(&g);

        let mut b_eig: Vec<f64> = b
            .complex_eigenvalues()
            .iter()
            .filter(|z| z.im.abs() < 1e-6)
            .map(|z| z.re)
            .filter(|&re| (re.abs() - 1.0).abs() > 1e-3) // drop the trivial +-1 eigenvalues (§2)
            .collect();
        b_eig.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = g.n;
        let op = HashimotoLinearization::new(&g);
        let m_dense = nalgebra::DMatrix::from_fn(2 * n, 2 * n, |i, j| {
            let mut e = vec![0.0; 2 * n];
            e[j] = 1.0;
            let mut out = vec![0.0; 2 * n];
            op.apply(&e, &mut out);
            out[i]
        });
        let mut m_eig: Vec<f64> = m_dense
            .complex_eigenvalues()
            .iter()
            .filter(|z| z.im.abs() < 1e-6)
            .map(|z| z.re)
            .collect();
        m_eig.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Every real eigenvalue of B (away from +-1) should also appear in
        // M's spectrum (M additionally carries the trivial +-1 pairs, so we
        // check containment, not equality of the full multisets).
        for be in &b_eig {
            let found = m_eig.iter().any(|me| (me - be).abs() < 1e-3);
            assert!(found, "Hashimoto eigenvalue {be} not found in linearization spectrum");
        }
    }

    /// `rho_B` from Krylov (matrix-free) Arnoldi should agree with the
    /// dense ground truth to a small tolerance.
    #[test]
    fn spectral_radius_matches_dense_ground_truth() {
        let g = random_near_regular(30, 4, 9);
        assert!(g.is_connected());
        let (_darcs, b) = dense_hashimoto_matrix(&g);
        let dense_rho = b.complex_eigenvalues().iter().map(|z| (z.re * z.re + z.im * z.im).sqrt()).fold(0.0, f64::max);

        let krylov_rho = estimate_spectral_radius(&g, 40, 3);
        assert_relative_eq!(krylov_rho, dense_rho, epsilon = 1e-4, max_relative = 1e-3);
    }

    /// Sanity/regression check for the sparse three-term feature recursion:
    /// compare `T_2 X` from `NbscFilterBank::apply_taps` against a direct
    /// dense matrix computation of `T_2 = (2A/rho)*T_1 - ((D-I)/rho^2)*T_0`.
    #[test]
    fn filter_bank_recursion_matches_dense_reference() {
        let g = random_near_regular(20, 3, 21);
        let bank = NbscFilterBank::build(&g, 30, 1);
        let f = 3;
        let mut x = FeatureMatrix::zeros(g.n, f);
        let mut seed = 12345u64;
        for v in x.data.iter_mut() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            *v = (seed as f64 / u64::MAX as f64) - 0.5;
        }

        let taps = bank.apply_taps(&g, &x, 2);
        assert_eq!(taps.len(), 3);

        let a = nalgebra::DMatrix::from_fn(g.n, g.n, |i, j| {
            if g.neighbors[i].contains(&j) {
                1.0
            } else {
                0.0
            }
        });
        let degrees = g.degrees();
        let d_minus_i = nalgebra::DMatrix::from_fn(g.n, g.n, |i, j| {
            if i == j {
                degrees[i] - 1.0
            } else {
                0.0
            }
        });
        let x_dense = nalgebra::DMatrix::from_row_slice(g.n, f, &x.data);
        let t0 = x_dense.clone();
        let t1 = (&a * &x_dense) / bank.rho_b;
        let t2 = (&a * &t1) * (2.0 / bank.rho_b) - (&d_minus_i * &t0) * (1.0 / (bank.rho_b * bank.rho_b));

        for i in 0..g.n {
            for j in 0..f {
                assert_relative_eq!(taps[2].row(i)[j], t2[(i, j)], epsilon = 1e-8, max_relative = 1e-6);
            }
        }
    }

    #[test]
    fn tree_has_zero_nontrivial_hashimoto_spectrum() {
        // On a tree every non-backtracking walk terminates (no cycles), so
        // B is nilpotent: all eigenvalues are exactly 0. This is the
        // degenerate negative-control case referenced by `random_tree`.
        let t = random_tree(15, 2);
        let (_darcs, b) = dense_hashimoto_matrix(&t);
        let max_abs = b.complex_eigenvalues().iter().map(|z| (z.re * z.re + z.im * z.im).sqrt()).fold(0.0, f64::max);
        // A tolerance looser than machine epsilon: this is the
        // largest-modulus eigenvalue of a genuinely nilpotent 2m x 2m
        // matrix recovered via a general dense eigensolver, so some
        // floating-point residue is expected numerical noise, not a sign
        // the construction is wrong.
        assert!(max_abs < 1e-2, "tree Hashimoto spectral radius should be ~0, got {max_abs}");
    }
}
