//! `krylov_ds`: production-ready Arnoldi and Lanczos Krylov subspace methods.
//!
//! - [`arnoldi::Arnoldi`] — general (non-symmetric) operators, produces an
//!   upper Hessenberg projection with DGKS selective reorthogonalization.
//! - [`lanczos::Lanczos`] — symmetric operators, produces a tridiagonal
//!   projection via the three-term recurrence, with optional full
//!   reorthogonalization to counter floating-point drift.
//! - [`eig`] — extracts Ritz values/vectors (approximate eigenpairs of the
//!   original operator) from either projection using `nalgebra`.
//! - [`operator`] — the [`operator::LinearOperator`] trait plus dense and
//!   CSR-sparse implementations; anything with a matrix-vector product can
//!   be plugged in, including matrix-free operators.
//!
//! # Example
//! ```
//! use krylov_ds::{Arnoldi, DenseMatrix};
//! use krylov_ds::eig::arnoldi_ritz_values;
//!
//! let a = DenseMatrix::from_fn(4, |i, j| if i == j { (i + 1) as f64 } else { 0.1 });
//! let v0 = vec![1.0, 0.0, 0.0, 0.0];
//! let result = Arnoldi::new(4, 1e-12).run(&a, &v0).unwrap();
//! let ritz = arnoldi_ritz_values(&result);
//! assert_eq!(ritz.len(), 4);
//! ```

pub mod arnoldi;
pub mod eig;
pub mod error;
pub mod lanczos;
pub mod operator;

pub use arnoldi::{Arnoldi, ArnoldiResult};
pub use error::KrylovError;
pub use lanczos::{Lanczos, LanczosResult, Reorthogonalization};
pub use operator::{CsrMatrix, DenseMatrix, LinearOperator};
