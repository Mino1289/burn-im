//! # burn-im (Burn Image Models)
//!
//! A pure-Rust computer vision library built on top of the [Burn](https://burn.dev) deep
//! learning framework. Inspired by PyTorch Image Models ([`timm`](https://github.com/huggingface/pytorch-image-models)).
//!
//! `burn-im` provides idiomatic implementations of modern vision backbones, native loading of
//! the `.safetensors` checkpoints published by the `timm` organization on the Hugging Face Hub,
//! and multi-scale feature extraction for downstream tasks such as detection and segmentation.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use burn_im::{CreateModelOptions, create_model};
//! use burn_im::prelude::*;
//!
//! # fn main() -> burn_im::Result<()> {
//! let device = Device::default();
//! let model = create_model(
//!     "resnet50.a1_in1k",
//!     &CreateModelOptions::pretrained(),
//!     &device,
//! )?;
//!
//! let input = Tensor::<4>::zeros([1, 3, 224, 224], &device);
//! let logits = model.forward(input);
//! assert_eq!(logits.dims(), [1, 1000]);
//! # Ok(())
//! # }
//! ```
//!
//! ## Interface levels
//!
//! * **Static API** — each family exposes its own configuration and module
//!   ([`models::resnet::ResNet`], [`models::alexnet::AlexNet`]) for autocompletion and zero
//!   dispatch overhead.
//! * **Dynamic API** — [`create_model`] resolves `<arch>.<tag>` strings through the registry and
//!   returns a `Box<dyn VisionModel>`, convenient for servers and benchmarks.
//!
//! ## Weight loading
//!
//! Pretrained weights are fetched synchronously (no async runtime required) from the Hub and
//! cached locally. Tensor names are matched against the Rust module hierarchy with a small set
//! of per-family regex rewrites; linear weights are transposed and normalization parameters
//! renamed by Burn's `PyTorchToBurnAdapter`.
//!
//! Non-RGB inputs (`in_chans != 3`) are supported: the first convolution is adapted from the
//! checkpoint with the same channel sum/mean used by `timm`'s `adapt_input_conv`.
//!
//! ## Backends and GPU
//!
//! Models are device-agnostic and run on whatever backend the tensors use. Burn does not select
//! a GPU by default (`Device::flex()` and `Device::default()` are CPU). Enable the crate's `gpu`
//! feature to link Burn's Vulkan backend, then pick a device explicitly:
//!
//! ```rust,no_run
//! # #[cfg(feature = "gpu")]
//! # fn gpu_example() {
//! use burn::tensor::{Device, DeviceKind};
//!
//! let device = Device::vulkan(DeviceKind::IntegratedGpu(0));
//! # let _ = device;
//! # }
//! ```

#![warn(missing_docs)]

pub mod error;
pub mod hub;
pub mod layers;
pub mod models;
pub mod registry;
pub mod traits;

pub use error::{Error, Result};
pub use hub::config::{Interpolation, PretrainedCfg, TimmHfConfig};
pub use registry::{
    CreateModelOptions, ModelId, create_model, get_pretrained_cfg, is_model, list_models,
    list_pretrained, split_model_name_tag,
};
pub use traits::{FeatureFormat, FeatureInfo, IntermediatesOptions, VisionModel};

/// Commonly used types and traits.
pub mod prelude {
    pub use burn::prelude::*;

    pub use crate::registry::{CreateModelOptions, create_model};
    pub use crate::traits::{FeatureFormat, FeatureInfo, IntermediatesOptions, VisionModel};
}
