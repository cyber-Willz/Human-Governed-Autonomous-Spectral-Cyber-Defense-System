use num_traits::Float;

use crate::error::KrylovError;
use crate::operator::{axpy, dot, norm, LinearOperator};

/// Reorthogonalization strategy for Lanczos.
///
/// The three-term Lanczos recurrence is only exactly orthogonal in infinite
/// precision. In floating point, orthogonality among the `q_i` degrades
/// quickly once Ritz values start converging, which produces spurious
/// ("ghost") duplicate eigenvalues if left uncorrected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reorthogonalization {
    /// No correction: fastest and lowest memory, but only reliable for a
    /// small number of steps or when only extremal Ritz values (not full
    /// spectra) are needed and are checked against a second run.
    None,
    /// Full reorthogonalization against every previously generated basis
    /// vector at every step (cost equivalent to Arnoldi). This is the
    /// robust, recommended default for production use.
    Full,
}

/// Result of running `k` steps of the Lanczos process on a symmetric
/// operator `A`.
///
/// Produces an orthonormal basis `Q = [q_0, ..., q_k]` of the Krylov
/// subspace and a real symmetric tridiagonal matrix `T` (given by the
/// diagonal `alpha` and off-diagonal `beta`) such that
/// `A Q_k = Q_k T_k + beta_k q_{k+1} e_k^T`.
#[derive(Debug, Clone)]
pub struct LanczosResult<T> {
    /// Orthonormal Krylov basis vectors.
    pub q: Vec<Vec<T>>,
    /// Diagonal entries of the tridiagonal matrix, length `steps`.
    pub alpha: Vec<T>,
    /// Off-diagonal entries, length `steps - 1` (or `steps` if no breakdown,
    /// where the last entry is the residual norm `beta_k`, useful for
    /// bounding Ritz-pair error but not part of `T` itself).
    pub beta: Vec<T>,
    /// Number of Lanczos steps actually completed.
    pub steps: usize,
    /// True if the iteration terminated early because the residual vector
    /// was (numerically) zero, meaning the Krylov subspace is `A`-invariant.
    pub breakdown: bool,
}

/// Configuration for the Lanczos process.
#[derive(Debug, Clone, Copy)]
pub struct Lanczos<T> {
    /// Maximum Krylov subspace dimension `m` to build (must be `<= n`).
    pub max_dim: usize,
    /// Breakdown / convergence tolerance, relative to the norm of `v0`.
    pub tol: T,
    pub reorth: Reorthogonalization,
}

impl<T: Float> Lanczos<T> {
    pub fn new(max_dim: usize, tol: T, reorth: Reorthogonalization) -> Self {
        Self { max_dim, tol, reorth }
    }

    /// Run the Lanczos process starting from initial vector `v0`.
    ///
    /// `op` is assumed symmetric (`A = A^T`); this is the caller's
    /// responsibility to guarantee, as it cannot be cheaply verified from a
    /// matrix-vector product alone. Running Lanczos on a non-symmetric
    /// operator will silently produce a meaningless tridiagonal projection.
    pub fn run(
        &self,
        op: &dyn LinearOperator<T>,
        v0: &[T],
    ) -> Result<LanczosResult<T>, KrylovError> {
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

        let m = self.max_dim;
        let mut q: Vec<Vec<T>> = Vec::with_capacity(m + 1);
        q.push(v0.iter().map(|&x| x / beta0).collect());

        let mut alpha: Vec<T> = Vec::with_capacity(m);
        let mut beta: Vec<T> = Vec::with_capacity(m);

        let mut q_prev: Vec<T> = vec![T::zero(); n];
        let mut beta_prev = T::zero();

        let mut steps = 0usize;
        let mut breakdown = false;

        for j in 0..m {
            let mut w = vec![T::zero(); n];
            op.apply(&q[j], &mut w);

            // Standard three-term recurrence subtraction.
            axpy(-beta_prev, &q_prev, &mut w);
            let aj = dot(&q[j], &w);
            axpy(-aj, &q[j], &mut w);
            alpha.push(aj);

            if self.reorth == Reorthogonalization::Full {
                // Re-project against the full basis built so far to counter
                // floating-point loss of orthogonality.
                for qi in q.iter() {
                    let c = dot(qi, &w);
                    axpy(-c, qi, &mut w);
                }
            }

            let bj = norm(&w);
            steps = j + 1;

            if bj <= self.tol * beta0.max(T::one()) {
                breakdown = true;
                break;
            }

            beta.push(bj);
            q_prev = q[j].clone();
            beta_prev = bj;
            q.push(w.iter().map(|&x| x / bj).collect());
        }

        Ok(LanczosResult { q, alpha, beta, steps, breakdown })
    }
}
