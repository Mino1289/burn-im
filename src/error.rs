//! Error types for `burn-im`.

use thiserror::Error;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors returned by `burn-im`.
#[derive(Debug, Error)]
pub enum Error {
    /// The requested architecture is not registered.
    #[error("unknown model architecture `{0}`")]
    UnknownModel(String),

    /// The architecture exists but the tag does not.
    #[error("unknown tag `{tag}` for architecture `{arch}` (available tags: {available})")]
    UnknownTag {
        /// Architecture name.
        arch: String,
        /// Requested tag.
        tag: String,
        /// Tags known for this architecture.
        available: String,
    },

    /// The model identifier does not follow `<arch>` or `<arch>.<tag>`.
    #[error("invalid model identifier `{0}` (expected `<arch>` or `<arch>.<tag>`)")]
    InvalidModelFormat(String),

    /// The requested configuration cannot be instantiated.
    #[error("invalid model configuration: {0}")]
    InvalidConfiguration(String),

    /// A Hugging Face Hub operation failed.
    #[error("hugging face hub error: {0}")]
    HubError(String),

    /// `config.json` could not be parsed.
    #[error("failed to parse model config: {0}")]
    ConfigParseError(String),

    /// Weights could not be loaded or do not match the model.
    #[error("failed to load weights: {0}")]
    WeightLoadError(String),

    /// Local filesystem error.
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),
}
