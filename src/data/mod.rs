//! Image preprocessing and ImageNet-1k class labels.

pub mod imagenet_labels;
pub mod transform;

pub use imagenet_labels::{IMAGENET_LABELS, imagenet_label};
pub use transform::{ImageGeometry, ImageProcessor};
