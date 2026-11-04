use thiserror::Error;

#[derive(Debug, Error)]
pub enum SigError {
    #[error("malformed hex signature line {line}: {reason}")]
    BadHexSig { line: usize, reason: String },

    #[error("malformed hash signature line {line}: {reason}")]
    BadHashSig { line: usize, reason: String },

    #[error("odd number of hex digits in token")]
    OddHexDigits,

    #[error("invalid hex digit")]
    InvalidHexDigit,

    #[error("signature has no literal anchor (all-wildcard pattern is rejected)")]
    NoAnchor,

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, SigError>;
