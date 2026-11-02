//! Channel-wise normalization layers: `LayerNorm2d`, ConvNeXt-V2 `GlobalResponseNorm`,
//! and root-mean-square normalizations (`RmsNorm`, `RMSNorm2d`).

use burn::module::Param;
use burn::nn::{Initializer, LayerNorm, LayerNormConfig};
use burn::prelude::*;
use burn::tensor::DType;

/// Layer normalization over the channel dimension of a `[B, C, H, W]` tensor.
///
/// Equivalent to `timm.layers.LayerNorm2d`: the channel axis is moved last, normalized by a
/// regular [`burn::nn::LayerNorm`], then moved back. The parameters keep Burn's native names,
/// `gamma` and `beta`, so the `PyTorchToBurnAdapter` maps `weight`/`bias` onto them.
#[derive(Module, Debug)]
pub struct LayerNorm2d {
    norm: LayerNorm,
    #[module(skip)]
    force_fp32: bool,
}

impl LayerNorm2d {
    /// Creates the normalization for `num_channels` channels.
    ///
    /// `timm`'s ConvNeXt uses `eps = 1e-6`; PyTorch's `LayerNorm` default is `1e-5`.
    pub fn new(num_channels: usize, eps: f64, device: &Device) -> Self {
        Self {
            norm: LayerNormConfig::new(num_channels)
                .with_epsilon(eps)
                .init(device),
            force_fp32: false,
        }
    }

    /// Sets whether internal variance and mean accumulation are forced to `f32`.
    pub fn with_force_fp32(mut self, force_fp32: bool) -> Self {
        self.force_fp32 = force_fp32;
        self
    }

    /// Applies channel-wise layer normalization to `[B, C, H, W]`.
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let orig_dtype = input.dtype();
        let x = if self.force_fp32 && orig_dtype != DType::F32 {
            input.cast(DType::F32)
        } else {
            input
        };
        let x = x.permute([0, 2, 3, 1]);
        let normalized = self.norm.forward(x).permute([0, 3, 1, 2]);
        if self.force_fp32 && orig_dtype != DType::F32 {
            normalized.cast(orig_dtype)
        } else {
            normalized
        }
    }
}

/// Root Mean Square normalization across the last dimension (`timm.layers.RmsNorm`).
///
/// Normalizes inputs by their root-mean-square value and scales by a learnable `weight` parameter:
/// `y = x * weight / sqrt(mean(x^2) + eps)`
#[derive(Module, Debug)]
pub struct RmsNorm {
    /// Learnable per-channel scale parameter (`weight` in PyTorch).
    pub weight: Param<Tensor<1>>,
    /// Numerical stability epsilon.
    pub eps: f64,
    #[module(skip)]
    force_fp32: bool,
}

impl RmsNorm {
    /// Creates an RMSNorm layer for `dim` features with affine scale initialized to ones.
    pub fn new(dim: usize, eps: f64, device: &Device) -> Self {
        Self {
            weight: Initializer::Ones.init([dim], device),
            eps,
            force_fp32: false,
        }
    }

    /// Enables or disables forcing `f32` accumulation for lower-precision inputs.
    pub fn with_force_fp32(mut self, force_fp32: bool) -> Self {
        self.force_fp32 = force_fp32;
        self
    }

    /// Normalizes across the trailing dimension of a tensor of rank `D`.
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        let orig_dtype = input.dtype();
        let x = if self.force_fp32 && orig_dtype != DType::F32 {
            input.cast(DType::F32)
        } else {
            input
        };
        let last_dim = D - 1;
        let v = x.clone().powf_scalar(2.0).mean_dim(last_dim);
        let normalized = x / (v.add_scalar(self.eps)).sqrt();
        let mut shape = [1; D];
        shape[last_dim] = self.weight.dims()[0];
        let weight = if self.force_fp32 && orig_dtype != DType::F32 {
            self.weight.val().cast(DType::F32).reshape(shape)
        } else {
            self.weight.val().reshape(shape)
        };
        let out = normalized * weight;
        if self.force_fp32 && orig_dtype != DType::F32 {
            out.cast(orig_dtype)
        } else {
            out
        }
    }
}

/// 2D channel-wise Root Mean Square normalization (`timm.layers.RmsNorm2d`).
///
/// Operates directly on `[B, C, H, W]` spatial feature maps by reducing along the channel axis.
#[derive(Module, Debug)]
pub struct RMSNorm2d {
    /// Learnable per-channel scale parameter (`weight` in PyTorch).
    pub weight: Param<Tensor<1>>,
    /// Numerical stability epsilon.
    pub eps: f64,
    #[module(skip)]
    force_fp32: bool,
}

/// Alias matching `timm`'s naming convention.
pub type RmsNorm2d = RMSNorm2d;

impl RMSNorm2d {
    /// Creates an `RMSNorm2d` layer for `num_channels` channels.
    pub fn new(num_channels: usize, eps: f64, device: &Device) -> Self {
        Self {
            weight: Initializer::Ones.init([num_channels], device),
            eps,
            force_fp32: false,
        }
    }

    /// Enables or disables forcing `f32` accumulation for lower-precision inputs.
    pub fn with_force_fp32(mut self, force_fp32: bool) -> Self {
        self.force_fp32 = force_fp32;
        self
    }

    /// Applies 2D RMS normalization to `[B, C, H, W]`.
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let orig_dtype = input.dtype();
        let x = if self.force_fp32 && orig_dtype != DType::F32 {
            input.cast(DType::F32)
        } else {
            input
        };
        let v = x.clone().powf_scalar(2.0).mean_dim(1);
        let normalized = x / (v.add_scalar(self.eps)).sqrt();
        let channels = self.weight.dims()[0];
        let weight = if self.force_fp32 && orig_dtype != DType::F32 {
            self.weight
                .val()
                .cast(DType::F32)
                .reshape([1, channels, 1, 1])
        } else {
            self.weight.val().reshape([1, channels, 1, 1])
        };
        let out = normalized * weight;
        if self.force_fp32 && orig_dtype != DType::F32 {
            out.cast(orig_dtype)
        } else {
            out
        }
    }
}

/// Global Response Normalization from ConvNeXt-V2 (`timm.layers.GlobalResponseNorm`).
///
/// For each sample, the L2 norm over the spatial dimensions gives one value per channel; those
/// values are divided by their mean over channels and used to rescale the input. A residual
/// connection and a per-channel affine `gamma`/`beta` complete the layer:
///
/// `y = x + beta + gamma * x * (||x||_spatial / (mean_c ||x||_spatial + eps))`
///
/// `gamma` and `beta` are zero-initialized, as in `timm`.
#[derive(Module, Debug)]
pub struct GlobalResponseNorm {
    /// Trainable per-channel scale (`timm`'s `weight`).
    pub weight: Param<Tensor<1>>,
    /// Trainable per-channel bias (`timm`'s `bias`).
    pub bias: Param<Tensor<1>>,
}

impl GlobalResponseNorm {
    /// Numerical floor of the channel mean, matching `timm`'s default.
    const EPS: f64 = 1e-6;

    /// Creates a zero-initialized GRN layer for `num_channels` channels.
    pub fn new(num_channels: usize, device: &Device) -> Self {
        Self {
            weight: Initializer::Zeros.init([num_channels], device),
            bias: Initializer::Zeros.init([num_channels], device),
        }
    }

    /// Applies GRN to `[B, C, H, W]`.
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let channels = input.dims()[1];
        let norm = input.clone().square().sum_dim(2).sum_dim(3).sqrt();
        let response = norm.clone() / norm.mean_dim(1).add_scalar(Self::EPS);
        input.clone()
            + self.bias.val().reshape([1, channels, 1, 1])
            + self.weight.val().reshape([1, channels, 1, 1]) * input * response
    }

    /// Applies GRN to `[B, H, W, C]` (channels-last layout, as used inside ConvNeXt-V2 blocks).
    pub fn forward_channels_last(&self, input: Tensor<4>) -> Tensor<4> {
        let channels = input.dims()[3];
        let norm = input.clone().square().sum_dim(1).sum_dim(2).sqrt();
        let response = norm.clone() / norm.mean_dim(3).add_scalar(Self::EPS);
        input.clone()
            + self.bias.val().reshape([1, 1, 1, channels])
            + self.weight.val().reshape([1, 1, 1, channels]) * input * response
    }
}
