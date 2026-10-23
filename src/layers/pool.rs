//! Global pooling strategies equivalent to `timm.layers.SelectAdaptivePool2d`.

use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::prelude::*;

use crate::{Error, Result};

/// Pooling strategy applied before the classifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PoolMode {
    /// Global average pooling.
    #[default]
    Avg,
    /// Global max pooling.
    Max,
    /// Average of global average and global max pooling.
    AvgMax,
    /// Concatenation of global average and global max pooling (doubles the channel count).
    CatAvgMax,
    /// Alias for [`PoolMode::Avg`] kept for `timm` compatibility.
    Fast,
    /// No pooling: spatial dimensions are kept.
    Identity,
}

impl PoolMode {
    /// Parses a `timm` pool type string (`avg`, `max`, `avgmax`, `catavgmax`, `fast`, ``).
    pub fn from_str_opt(value: &str) -> Result<Self> {
        match value {
            "avg" => Ok(Self::Avg),
            "max" => Ok(Self::Max),
            "avgmax" => Ok(Self::AvgMax),
            "catavgmax" => Ok(Self::CatAvgMax),
            "fast" => Ok(Self::Fast),
            "" | "identity" => Ok(Self::Identity),
            other => Err(Error::InvalidConfiguration(format!(
                "unknown pool mode `{other}` (expected avg, max, avgmax, catavgmax, fast or identity)"
            ))),
        }
    }

    /// Number of output channels for `num_chs` input channels.
    pub fn out_channels(self, num_chs: usize) -> usize {
        match self {
            Self::CatAvgMax => num_chs * 2,
            _ => num_chs,
        }
    }
}

/// Adaptive global pooling with a selectable reduction mode.
///
/// Non-average modes only support a `1x1` output; adaptive average pooling accepts any
/// target size (used by AlexNet, which pools to `6x6`).
#[derive(Module, Debug)]
pub struct SelectAdaptivePool2d {
    pool: Option<AdaptiveAvgPool2d>,
    #[module(skip)]
    mode: PoolMode,
    #[module(skip)]
    output_size: [usize; 2],
}

impl SelectAdaptivePool2d {
    /// Creates a pooling layer producing `output_size`.
    pub fn new(mode: PoolMode, output_size: [usize; 2]) -> Result<Self> {
        if mode != PoolMode::Avg
            && mode != PoolMode::Fast
            && mode != PoolMode::Identity
            && output_size != [1, 1]
        {
            return Err(Error::InvalidConfiguration(format!(
                "pool mode {mode:?} only supports a 1x1 output, got {output_size:?}"
            )));
        }
        let pool = matches!(mode, PoolMode::Avg | PoolMode::Fast)
            .then(|| AdaptiveAvgPool2dConfig::new(output_size).init());
        Ok(Self {
            pool,
            mode,
            output_size,
        })
    }

    /// Global average pooling to `1x1`.
    pub fn avg() -> Self {
        Self::new(PoolMode::Avg, [1, 1]).expect("1x1 average pooling is always valid")
    }

    /// Output spatial size.
    pub fn output_size(&self) -> [usize; 2] {
        self.output_size
    }

    /// Pooling mode.
    pub fn mode(&self) -> PoolMode {
        self.mode
    }

    /// Applies pooling to `[B, C, H, W]`.
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let avg = |x: Tensor<4>| {
            self.pool
                .as_ref()
                .expect("average pooling layer is present")
                .forward(x)
        };
        match self.mode {
            PoolMode::Avg | PoolMode::Fast => avg(input),
            PoolMode::Max => input.max_dim(2).max_dim(3),
            PoolMode::AvgMax => avg(input.clone()).add(avg_max(input)).mul_scalar(0.5),
            PoolMode::CatAvgMax => Tensor::cat(vec![avg(input.clone()), avg_max(input)], 1),
            PoolMode::Identity => input,
        }
    }
}

fn avg_max(input: Tensor<4>) -> Tensor<4> {
    input.max_dim(2).max_dim(3)
}
