use num_traits::Float;

use crate::error::KrylovError;
use crate::operator::{axpy, dot, norm, LinearOperator};

/// DGKS reorthogonalization criterion constant (Daniel-Gragg-Kaufman-Stewart).
/// If the norm drops by more than this factor during modified Gram-Schmidt,
/// a second orthogonalization pass is performed. This is the standard,
/// numerically-robust choice used in LAPACK-adjacent Krylov codes.
const DGKS_ETA: f64 = 0.717;

/// Result of running `k` steps of the Arnoldi process on an `n x n` operator `A`.
///
/// Produces an orthonormal basis `Q = [q_0, ..., q_k]` of the Krylov subspace
/// `K_{k+1}(A, v0) = span{v0, A v0, ..., A^k v0}` and an upper Hessenberg
/// matrix `H` such that `A * Q[:, :k] = Q[:, :k+1] * H`, i.e. the classic
/// Arnoldi relation `A Q_k = Q_k H_k + h_{k+1,k} q_{k+1} e_k^T`.
#[derive(Debug, Clone)]
pub struct ArnoldiResult<T> {
    /// Orthonormal Krylov basis vectors. `q.len() == steps + 1` unless a
    /// happy breakdown occurred, in which case `q.len() == steps` and the
    /// Krylov subspace is exactly `A`-invariant.
    pub q: Vec<Vec<T>>,
    /// Upper Hessenberg matrix stored densely, row-major, of shape
    /// `(steps + 1) x steps` (or `steps x steps` on happy breakdown).
    pub h: Vec<Vec<T>>,
    /// Number of Arnoldi steps actually completed.
    pub steps: usize,
    /// True if the iteration terminated early because the residual vector
    /// was (numerically) zero, meaning the Krylov subspace is `A`-invariant
    /// and every Ritz value is an exact eigenvalue of `A`.
    pub breakdown: bool,
}

/// Configuration for the Arnoldi process.
#[derive(Debug, Clone, Copy)]
pub struct Arnoldi<T> {
    /// Maximum Krylov subspace dimension `m` to build (must be `<= n`).
    pub max_dim: usize,
    /// Breakdown / convergence tolerance, relative to the norm of `v0`.
    pub tol: T,
}

impl<T: Float> Arnoldi<T> {
    pub fn new(max_dim: usize, tol: T) -> Self {
        Self { max_dim, tol }
    }

    /// Run the Arnoldi process starting from initial vector `v0`.
    pub fn run(
        &self,
        op: &dyn LinearOperator<T>,
        v0: &[T],
    ) -> Result<ArnoldiResult<T>, KrylovError> {
        let n = op.dim();
        if v0.len() != n {
            return Err(KrylovError::DimensionMismatch { op_dim: n, vec_dim: v0.len() });
        }
        if self.max_dim == 0 {
            return Err(KrylovError::ZeroSubspace);
        }
        if self.max_dim > n {
            return Err(KrylovError::SubspaceTooLarge { requested: self.max_dim, max: n });
        }

        let beta0 = norm(v0);
        if beta0 <= self.tol {
            return Err(KrylovError::ZeroInitialVector);
        }

        let eta = T::from(DGKS_ETA).unwrap();
        let m = self.max_dim;

        let mut q: Vec<Vec<T>> = Vec::with_capacity(m + 1);
        q.push(v0.iter().map(|&x| x / beta0).collect());

        // Dense (m+1) x m Hessenberg storage; trimmed to actual size at the end.
        let mut h: Vec<Vec<T>> = vec![vec![T::zero(); m]; m + 1];

        let mut steps = 0usize;
        let mut breakdown = false;

        for j in 0..m {
            let mut w = vec![T::zero(); n];
            op.apply(&q[j], &mut w);
            let w_norm_pre = norm(&w);

            // First modified Gram-Schmidt pass.
            for i in 0..=j {
                let hij = dot(&q[i], &w);
                axpy(-hij, &q[i], &mut w);
                h[i][j] = h[i][j] + hij;
            }

            let mut w_norm = norm(&w);

            // DGKS selective reorthogonalization: if the projection removed
            // most of the vector's length, floating point error means the
            // remaining component may not be accurately orthogonal to Q_j.
            // A single extra MGS pass restores orthogonality to machine
            // precision in virtually all practical cases.
            if w_norm <= eta * w_norm_pre {
                for i in 0..=j {
                    let corr = dot(&q[i], &w);
                    axpy(-corr, &q[i], &mut w);
                    h[i][j] = h[i][j] + corr;
                }
                w_norm = norm(&w);
            }

            h[j + 1][j] = w_norm;
            steps = j + 1;

            if w_norm <= self.tol * beta0.max(T::one()) {
                breakdown = true;
                break;
            }

            let scale = T::one() / w_norm;
            q.push(w.iter().map(|&x| x * scale).collect());
        }

        let rows = if breakdown { steps } else { steps + 1 };
        let h_trunc: Vec<Vec<T>> = (0..rows).map(|i| h[i][..steps].to_vec()).collect();

        Ok(ArnoldiResult { q, h: h_trunc, steps, breakdown })
    }
}
