//! Shared neural building blocks reused across architecture families.

pub mod drop;
pub mod head;
pub mod norm;
pub mod pool;
pub mod scale;

pub use drop::{DropPath, drop_path_schedule};
pub use head::{ClassifierHead, NormMlpClassifierHead};
pub use norm::{GlobalResponseNorm, LayerNorm2d};
pub use pool::{PoolMode, SelectAdaptivePool2d};
pub use scale::{LayerScale, LayerScale2d};
