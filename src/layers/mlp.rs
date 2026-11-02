//! Multi-Layer Perceptrons: standard `Mlp` and gated `SwiGLU`.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::{gelu, silu};

/// Standard MLP block used in Vision Transformers (`timm.layers.Mlp`).
#[derive(Module, Debug)]
pub struct Mlp {
    /// First projection (`in_features -> hidden_features`).
    pub fc1: Linear,
    /// Second projection (`hidden_features -> out_features`).
    pub fc2: Linear,
}

impl Mlp {
    /// Creates a standard MLP block with GELU activation.
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
    ) -> Self {
        let fc1 = LinearConfig::new(in_features, hidden_features)
            .with_bias(bias)
            .init(device);
        let fc2 = LinearConfig::new(hidden_features, out_features)
            .with_bias(bias)
            .init(device);
        Self { fc1, fc2 }
    }

    /// Forward pass through the MLP.
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let x = self.fc1.forward(x);
        let x = gelu(x);
        self.fc2.forward(x)
    }
}

/// Gated SwiGLU MLP block used in modern architectures like DINOv2 Giant (`timm.layers.SwiGLU`).
#[derive(Module, Debug)]
pub struct SwiGLU {
    /// Gate projection.
    pub fc1_g: Linear,
    /// Value projection.
    pub fc1_x: Linear,
    /// Output projection.
    pub fc2: Linear,
}

impl SwiGLU {
    /// Creates a SwiGLU block.
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
    ) -> Self {
        let fc1_g = LinearConfig::new(in_features, hidden_features)
            .with_bias(bias)
            .init(device);
        let fc1_x = LinearConfig::new(in_features, hidden_features)
            .with_bias(bias)
            .init(device);
        let fc2 = LinearConfig::new(hidden_features, out_features)
            .with_bias(bias)
            .init(device);
        Self { fc1_g, fc1_x, fc2 }
    }

    /// Forward pass through the SwiGLU block: `(silu(fc1_g(x)) * fc1_x(x)) -> fc2`.
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let gate = silu(self.fc1_g.forward(x.clone()));
        let hidden = self.fc1_x.forward(x);
        let x = gate * hidden;
        self.fc2.forward(x)
    }
}
