//! Pooling + dropout + optional linear head, equivalent to `timm.layers.ClassifierHead`.

use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::prelude::*;

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
