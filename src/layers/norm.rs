//! Channel-wise normalization layers: `LayerNorm2d` and ConvNeXt-V2 `GlobalResponseNorm`.

use burn::module::Param;
use burn::nn::{Initializer, LayerNorm, LayerNormConfig};
use burn::prelude::*;

/// Layer normalization over the channel dimension of a `[B, C, H, W]` tensor.
///
/// Equivalent to `timm.layers.LayerNorm2d`: the channel axis is moved last, normalized by a
/// regular [`burn::nn::LayerNorm`], then moved back. The parameters keep Burn's native names,
/// `gamma` and `beta`, so the `PyTorchToBurnAdapter` maps `weight`/`bias` onto them.
#[derive(Module, Debug)]
pub struct LayerNorm2d {
    norm: LayerNorm,
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
        }
    }

    /// Applies channel-wise layer normalization to `[B, C, H, W]`.
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let x = input.permute([0, 2, 3, 1]);
        self.norm.forward(x).permute([0, 3, 1, 2])
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
    pub gamma: Param<Tensor<1>>,
    /// Trainable per-channel bias (`timm`'s `bias`).
    pub beta: Param<Tensor<1>>,
}

impl GlobalResponseNorm {
    /// Numerical floor of the channel mean, matching `timm`'s default.
    const EPS: f64 = 1e-6;

    /// Creates a zero-initialized GRN layer for `num_channels` channels.
    pub fn new(num_channels: usize, device: &Device) -> Self {
        Self {
            gamma: Initializer::Zeros.init([num_channels], device),
            beta: Initializer::Zeros.init([num_channels], device),
        }
    }

    /// Applies GRN to `[B, C, H, W]`.
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let channels = input.dims()[1];
        let norm = input.clone().square().sum_dim(2).sum_dim(3).sqrt();
        let response = norm.clone() / norm.mean_dim(1).add_scalar(Self::EPS);
        input.clone()
            + self.beta.val().reshape([1, channels, 1, 1])
            + self.gamma.val().reshape([1, channels, 1, 1]) * input * response
    }
}
