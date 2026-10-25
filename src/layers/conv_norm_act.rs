//! Convolution + normalization + activation block, equivalent to `timm.layers.ConvNormAct`.
//!
//! The block is the assembly unit of the convolutional families (`ResNet`, `MobileNet`,
//! `EfficientNet`, `CSPNeXt`): a convolution with explicit padding and dilation, a batch
//! normalization and an optional activation. Keeping it as a single module means the families
//! only choose channel counts, kernel/stride/dilation and whether the activation is present,
//! instead of re-assembling three layers each time.

use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig};
use burn::prelude::*;
use burn::tensor::activation::relu;

/// Activation applied after the normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Activation {
    /// No activation: the block is a plain convolution + normalization.
    None,
    /// Rectified linear unit.
    #[default]
    Relu,
}

impl Activation {
    fn apply(self, x: Tensor<4>) -> Tensor<4> {
        match self {
            Self::None => x,
            Self::Relu => relu(x),
        }
    }
}

/// Builder for [`ConvNormAct`].
#[derive(Debug, Clone)]
pub struct ConvNormActConfig {
    /// Input and output channels.
    pub channels: [usize; 2],
    /// Square kernel size.
    pub kernel_size: usize,
    /// Square stride.
    pub stride: usize,
    /// Explicit symmetric padding. Usually `kernel_size / 2 * dilation` for a "same" output.
    pub padding: usize,
    /// Square dilation of the convolution.
    pub dilation: usize,
    /// Convolution groups (`1` for a dense convolution).
    pub groups: usize,
    /// Activation applied after the normalization.
    pub activation: Activation,
}

impl ConvNormActConfig {
    /// Creates a `3x3`, stride 1, "same"-padded ReLU block.
    pub fn new(in_chans: usize, out_chans: usize) -> Self {
        Self {
            channels: [in_chans, out_chans],
            kernel_size: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            activation: Activation::Relu,
        }
    }

    /// Sets the kernel size.
    pub fn with_kernel_size(mut self, kernel_size: usize) -> Self {
        self.kernel_size = kernel_size;
        self
    }

    /// Sets the stride.
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Sets the explicit padding.
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// Sets the dilation.
    ///
    /// Callers using a dilated `3x3` convolution normally want `padding == dilation` to keep the
    /// spatial dimensions unchanged.
    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    /// Sets the convolution groups.
    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }

    /// Removes the activation (convolution + normalization only).
    pub fn without_activation(mut self) -> Self {
        self.activation = Activation::None;
        self
    }

    /// Sets the activation applied after the normalization.
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Instantiates the block. Parameters are lazily initialized (see [`ConvNormAct`]).
    pub fn init(&self, device: &Device) -> ConvNormAct {
        let conv = Conv2dConfig::new(self.channels, [self.kernel_size, self.kernel_size])
            .with_stride([self.stride, self.stride])
            .with_padding(PaddingConfig2d::Explicit(
                self.padding,
                self.padding,
                self.padding,
                self.padding,
            ))
            .with_dilation([self.dilation, self.dilation])
            .with_groups(self.groups)
            .with_bias(false)
            .init(device);

        ConvNormAct {
            norm: BatchNormConfig::new(self.channels[1]).init(device),
            conv,
            activation: self.activation,
        }
    }
}

/// Convolution followed by batch normalization and an optional activation.
///
/// The parameters are created lazily by Burn, so building a model does not allocate its weights:
/// loading a checkpoint fills the pending parameters directly instead of initializing them with
/// random values first.
#[derive(Module, Debug)]
pub struct ConvNormAct {
    conv: Conv2d,
    norm: BatchNorm,
    #[module(skip)]
    activation: Activation,
}

impl ConvNormAct {
    /// Creates a `3x3`, stride 1, ReLU block with a dense convolution.
    pub fn new(in_chans: usize, out_chans: usize, device: &Device) -> Self {
        ConvNormActConfig::new(in_chans, out_chans).init(device)
    }

    /// Applies convolution, normalization and activation.
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        self.activation
            .apply(self.norm.forward(self.conv.forward(input)))
    }

    /// Convolution sub-module.
    pub fn conv(&self) -> &Conv2d {
        &self.conv
    }

    /// Mutable convolution sub-module.
    pub fn conv_mut(&mut self) -> &mut Conv2d {
        &mut self.conv
    }
}
