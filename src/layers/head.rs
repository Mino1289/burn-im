//! Pooling + dropout + optional linear head, equivalent to `timm.layers.ClassifierHead` and
//! `timm.layers.NormMlpClassifierHead`.

use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::gelu;

use crate::layers::norm::LayerNorm2d;
use crate::layers::pool::{PoolMode, SelectAdaptivePool2d};

/// Standard classification head: global pooling, dropout and an optional linear classifier.
#[derive(Module, Debug)]
pub struct ClassifierHead {
    pool: SelectAdaptivePool2d,
    drop: Dropout,
    fc: Option<Linear>,
    #[module(skip)]
    num_features: usize,
    #[module(skip)]
    num_classes: usize,
}

impl ClassifierHead {
    /// Creates a head.
    ///
    /// `num_classes == 0` removes the linear layer: the head then behaves as an identity on the
    /// pooled embedding, matching `timm`'s headless mode.
    pub fn new(
        num_features: usize,
        num_classes: usize,
        pool_mode: PoolMode,
        drop_rate: f64,
        device: &Device,
    ) -> Self {
        let pool = SelectAdaptivePool2d::new(pool_mode, [1, 1])
            .expect("classifier head only supports a 1x1 pooling output");
        let num_features = pool.mode().out_channels(num_features);
        let fc =
            (num_classes > 0).then(|| LinearConfig::new(num_features, num_classes).init(device));
        Self {
            pool,
            drop: DropoutConfig::new(drop_rate).init(),
            fc,
            num_features,
            num_classes: if num_classes == 0 { 0 } else { num_classes },
        }
    }

    /// Embedding dimension produced before the final linear layer.
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Number of classes (0 when the linear layer was removed).
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Applies the head to `[B, C, H, W]` features.
    ///
    /// Returns `[B, num_features]` when `pre_logits` is true (or when the head is headless),
    /// otherwise `[B, num_classes]`.
    pub fn forward(&self, features: Tensor<4>, pre_logits: bool) -> Tensor<2> {
        let pooled: Tensor<2> = self.pool.forward(features).flatten(1, 3);
        let Some(fc) = &self.fc else {
            return pooled;
        };
        if pre_logits {
            return pooled;
        }
        fc.forward(self.drop.forward(pooled))
    }
}

/// Pool -> norm -> MLP -> dropout -> optional linear head, equivalent to
/// `timm.layers.NormMlpClassifierHead` (the ConvNeXt head).
///
/// The optional hidden layer is `timm`'s pre-logits `Linear` + GELU pair, applied after flattening.
/// `num_classes == 0` removes the final linear layer: the head then returns the pre-logits
/// embedding, like [`ClassifierHead`].
#[derive(Module, Debug)]
pub struct NormMlpClassifierHead {
    pool: SelectAdaptivePool2d,
    norm: Option<LayerNorm2d>,
    pre_logits: Option<Linear>,
    drop: Dropout,
    fc: Option<Linear>,
    #[module(skip)]
    num_features: usize,
    #[module(skip)]
    num_classes: usize,
}

impl NormMlpClassifierHead {
    /// Creates a head.
    ///
    /// `hidden_features` inserts the pre-logits MLP and `norm_eps` a [`LayerNorm2d`] after
    /// pooling (`timm` uses `1e-6` for ConvNeXt).
    pub fn new(
        num_features: usize,
        num_classes: usize,
        hidden_features: Option<usize>,
        pool_mode: PoolMode,
        drop_rate: f64,
        norm_eps: Option<f64>,
        device: &Device,
    ) -> Self {
        let pool = SelectAdaptivePool2d::new(pool_mode, [1, 1])
            .expect("classifier head only supports a 1x1 pooling output");
        let pooled_features = pool.mode().out_channels(num_features);
        let pre_logits =
            hidden_features.map(|hidden| LinearConfig::new(pooled_features, hidden).init(device));
        Self {
            pool,
            norm: norm_eps.map(|eps| LayerNorm2d::new(pooled_features, eps, device)),
            pre_logits,
            drop: DropoutConfig::new(drop_rate).init(),
            fc: (num_classes > 0).then(|| {
                LinearConfig::new(hidden_features.unwrap_or(pooled_features), num_classes)
                    .init(device)
            }),
            num_features: hidden_features.unwrap_or(pooled_features),
            num_classes,
        }
    }

    /// Embedding dimension produced before the final linear layer.
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Number of classes (0 when the linear layer was removed).
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Applies the head to `[B, C, H, W]` features.
    ///
    /// Returns `[B, num_features]` when `pre_logits` is true (or when the head is headless),
    /// otherwise `[B, num_classes]`.
    pub fn forward(&self, features: Tensor<4>, pre_logits: bool) -> Tensor<2> {
        let x = self.pool.forward(features);
        let x = match &self.norm {
            Some(norm) => norm.forward(x),
            None => x,
        };
        let x = x.flatten(1, 3);
        let x = match &self.pre_logits {
            Some(fc) => gelu(fc.forward(x)),
            None => x,
        };
        let x = self.drop.forward(x);
        match (&self.fc, pre_logits) {
            (Some(fc), false) => fc.forward(x),
            _ => x,
        }
    }
}
