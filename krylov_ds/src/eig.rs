//! Eigenvalue (Ritz value) extraction from the small dense projections
//! produced by [`crate::arnoldi::Arnoldi`] and [`crate::lanczos::Lanczos`].
//!
//! The whole point of Krylov methods is that the projected `k x k` problem
//! (Hessenberg for Arnoldi, tridiagonal for Lanczos) is cheap to solve
//! exactly with a dense eigensolver, even though the original `n x n`
//! problem may be far too large to solve directly. We lean on `nalgebra`
//! here rather than hand-rolling a shifted-QR implementation, since a
//! battle-tested dense eigensolver is exactly the kind of building block
//! that should not be reinvented.

use nalgebra::{DMatrix, DVector};
use num_traits::Float;

use crate::arnoldi::ArnoldiResult;
use crate::lanczos::LanczosResult;

/// A Ritz pair: an approximate eigenvalue of the original operator `A`
/// together with the corresponding approximate eigenvector (lifted back
/// into the original `n`-dimensional space via the Krylov basis) and an
/// a-posteriori residual norm bound.
#[derive(Debug, Clone)]
pub struct RitzPair<T> {
    pub value: T,
    pub vector: Vec<T>,
    /// `|| A x - value * x ||` estimate, computed cheaply from the last
    /// Hessenberg/tridiagonal row without forming `A x` explicitly.
    pub residual_norm: T,
}

/// Compute Ritz values/vectors from an Arnoldi Hessenberg projection.
///
/// Note: for a non-symmetric (or non-normal) operator, Ritz values from a
/// truncated Krylov subspace are approximations to eigenvalues of `A`
/// (typically converging first for extremal eigenvalues) and are generally
/// complex even if `A` is real; this routine returns only the real part
/// together with the imaginary part, since `nalgebra`'s general eigensolver
/// yields complex eigenvalues for a real Hessenberg matrix.
pub fn arnoldi_ritz_values(result: &ArnoldiResult<f64>) -> Vec<nalgebra::Complex<f64>> {
    let k = result.steps;
    let h_square = DMatrix::from_fn(k, k, |i, j| result.h[i][j]);
    h_square.complex_eigenvalues().iter().copied().collect()
}

/// Compute Ritz pairs (value + vector lifted to R^n + residual bound) from
/// an Arnoldi run, valid in general only when the Ritz value is real (the
/// common case for symmetric or near-symmetric operators, or for the
/// dominant real eigenvalue of a general operator).
pub fn arnoldi_real_ritz_pairs(result: &ArnoldiResult<f64>) -> Vec<RitzPair<f64>> {
    let k = result.steps;
    let h_square = DMatrix::from_fn(k, k, |i, j| result.h[i][j]);
    let eig = h_square.clone().complex_eigenvalues();

    // h_{k+1,k}, i.e. the last subdiagonal entry, needed for the residual
    // bound || A x - lambda x || = |h_{k+1,k}| * |e_k^T y| for Ritz vector y.
    let h_last = if result.h.len() > k { result.h[k][k - 1] } else { 0.0 };

    let mut pairs = Vec::new();
    for lambda in eig.iter() {
        if lambda.im.abs() > 1e-9 * (1.0 + lambda.re.abs()) {
            continue; // skip complex-conjugate pairs; not representable as a single real vector
        }
        let lambda_re = lambda.re;
        // Real eigenvector of H for a (near-)real eigenvalue via inverse
        // iteration would be more robust, but for the common symmetric/
        // near-symmetric case a direct real eigensolve is simpler and exact.
        if let Some(y) = real_eigenvector_for(&h_square, lambda_re) {
            let mut x = vec![0.0f64; result.q[0].len()];
            for (i, qi) in result.q.iter().take(k).enumerate() {
                let yi = y[i];
                for (xj, &qij) in x.iter_mut().zip(qi.iter()) {
                    *xj += yi * qij;
                }
            }
            let residual_norm = (h_last * y[k - 1]).abs();
            pairs.push(RitzPair { value: lambda_re, vector: x, residual_norm });
        }
    }
    pairs
}

/// Compute Ritz values/vectors from a Lanczos tridiagonal projection.
/// Because `A` is symmetric, all Ritz values are real and eigenvectors are
/// orthonormal; this is the numerically well-conditioned, common case.
pub fn lanczos_ritz_pairs<T: Float + nalgebra::RealField + Copy>(
    result: &LanczosResult<T>,
) -> Vec<RitzPair<T>> {
    let k = result.steps;
    if k == 0 {
        return Vec::new();
    }
    let mut t = DMatrix::<T>::zeros(k, k);
    for i in 0..k {
        t[(i, i)] = result.alpha[i];
    }
    for i in 0..k.saturating_sub(1) {
        let b = result.beta[i];
        t[(i, i + 1)] = b;
        t[(i + 1, i)] = b;
    }

    let eig = t.symmetric_eigen();
    let n = result.q[0].len();

    let beta_last = if result.beta.len() >= k { Some(result.beta[k - 1]) } else { None };

    let mut pairs = Vec::with_capacity(k);
    for col in 0..k {
        let mut x = vec![T::zero(); n];
        for (i, qi) in result.q.iter().take(k).enumerate() {
            let yi = eig.eigenvectors[(i, col)];
            for (xj, &qij) in x.iter_mut().zip(qi.iter()) {
                *xj = *xj + yi * qij;
            }
        }
        let last_component = eig.eigenvectors[(k - 1, col)];
        let residual_norm = match beta_last {
            Some(b) => Float::abs(b * last_component),
            None => T::zero(), // happy breakdown: exact eigenvalue
        };
        pairs.push(RitzPair { value: eig.eigenvalues[col], vector: x, residual_norm });
    }
    // Sort by algebraic value, ascending, as is conventional.
    pairs.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());
    pairs
}

/// Solve `(H - lambda I) y = 0` for a real eigenvalue `lambda` of a
/// (possibly non-symmetric) small dense matrix `H`, via a single step of
/// inverse iteration from a fixed starting vector. This is adequate for
/// well-separated Ritz values in the small `k x k` projection.
fn real_eigenvector_for(h: &DMatrix<f64>, lambda: f64) -> Option<Vec<f64>> {
    let k = h.nrows();
    let shifted = h - DMatrix::identity(k, k) * lambda;
    let mut v = DVector::from_element(k, 1.0 / (k as f64).sqrt());
    for _ in 0..3 {
        let regularized = &shifted + DMatrix::identity(k, k) * 1e-10;
        let decomp = regularized.clone().lu();
        let solved = decomp.solve(&v)?;
        let norm = solved.norm();
        if norm <= 0.0 || !norm.is_finite() {
            return None;
        }
        v = solved / norm;
    }
    Some(v.iter().copied().collect())
}
