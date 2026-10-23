//! Shared neural building blocks reused across architecture families.

pub mod head;
pub mod pool;

pub use head::ClassifierHead;
pub use pool::{PoolMode, SelectAdaptivePool2d};
