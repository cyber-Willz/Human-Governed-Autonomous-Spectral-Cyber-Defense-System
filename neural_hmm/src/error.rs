use thiserror::Error;

/// All failure modes for the neural HMM pipeline. Kept exhaustive and specific
/// so callers (e.g. active-siem, spec-engine) can match on them instead of
/// string-matching a generic anyhow error.
#[derive(Error, Debug)]
pub enum HmmError {
    #[error("transition matrix row {row} has {actual} columns, expected {expected} (matrix must be square)")]
    NonSquareTransitionMatrix { row: usize, expected: usize, actual: usize },

    #[error("transition matrix row {row} does not sum to 1.0 (sum={sum}, tolerance={tolerance})")]
    InvalidRowSum { row: usize, sum: f32, tolerance: f32 },

    #[error("negative transition probability at [{row}][{col}]: {value}")]
    NegativeProbability { row: usize, col: usize, value: f32 },

    #[error("empty transition matrix")]
    EmptyTransitionMatrix,

    #[error("dimension mismatch: expected {expected} states, got {actual}")]
    StateDimensionMismatch { expected: usize, actual: usize },

    #[error("prior belief does not sum to 1.0 (sum={0}); call NeuralHmm::normalize_prior first")]
    InvalidPriorSum(f32),

    #[error("emission/prediction collapsed to all-zero probability mass (likely numerical underflow or a malformed feature vector)")]
    BeliefCollapse,

    #[error("non-finite value encountered during filtering at state {state}: {value}")]
    NonFiniteValue { state: usize, value: f32 },

    #[error("config I/O error: {0}")]
    ConfigIo(String),

    #[error("config parse error: {0}")]
    ConfigParse(String),

    #[error("model checkpoint error: {0}")]
    Checkpoint(String),

    #[error("empty observation batch")]
    EmptyBatch,
}

pub type HmmResult<T> = Result<T, HmmError>;
