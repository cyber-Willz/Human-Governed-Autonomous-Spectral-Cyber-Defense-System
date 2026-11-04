use thiserror::Error;

/// Errors that can occur while building or using a Krylov subspace.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum KrylovError {
    #[error("operator dimension is {op_dim}, but the supplied vector has dimension {vec_dim}")]
    DimensionMismatch { op_dim: usize, vec_dim: usize },

    #[error("requested Krylov subspace dimension {requested} exceeds the operator dimension {max}")]
    SubspaceTooLarge { requested: usize, max: usize },

    #[error("requested Krylov subspace dimension must be at least 1")]
    ZeroSubspace,

    #[error("initial vector has (numerically) zero norm and cannot be normalized")]
    ZeroInitialVector,
}
