//! Per-channel residual scaling (`timm.layers.LayerScale`).

use burn::module::Param;
use burn::nn::Initializer;
use burn::prelude::*;

/// Layer scale for transformer tensors: `gamma` has shape `[C]` and scales `[B, N, C]`.
///
/// `timm`'s `LayerScale` with `layer_scale_type='1d'`.
#[derive(Module, Debug)]
pub struct LayerScale {
    /// Per-channel scale, initialized to `init_values`.
    pub gamma: Param<Tensor<1>>,
}

impl LayerScale {
    /// Creates a layer scale with every channel initialized to `init_values`
    /// (`1e-5` in `timm`, `1e-6` in ConvNeXt).
    pub fn new(num_channels: usize, init_values: f64, device: &Device) -> Self {
        Self {
            gamma: Initializer::Constant { value: init_values }.init([num_channels], device),
        }
    }

    /// Scales `[B, N, C]` per channel.
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let channels = self.gamma.dims()[0];
        input * self.gamma.val().reshape([1, 1, channels])
    }
}

/// Layer scale for convolution tensors: `gamma` has shape `[C, 1, 1]` and scales `[B, C, H, W]`.
///
/// `timm`'s `LayerScale2d`, used by ConvNeXt blocks.
#[derive(Module, Debug)]
pub struct LayerScale2d {
    /// Per-channel scale, initialized to `init_values`.
    pub gamma: Param<Tensor<3>>,
}

impl LayerScale2d {
    /// Creates a 2D layer scale with every channel initialized to `init_values`.
    pub fn new(num_channels: usize, init_values: f64, device: &Device) -> Self {
        Self {
            gamma: Initializer::Constant { value: init_values }.init([num_channels, 1, 1], device),
        }
    }

    /// Scales `[B, C, H, W]` per channel.
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let channels = self.gamma.dims()[0];
        input * self.gamma.val().reshape([1, channels, 1, 1])
    }
}
