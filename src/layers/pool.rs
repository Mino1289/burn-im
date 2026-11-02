//! Global pooling strategies: `SelectAdaptivePool2d` and `AttentionPoolPrr` (PRR).

use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;

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
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        match self.mode {
            PoolMode::Avg | PoolMode::Fast => self.pool.as_ref().unwrap().forward(x),
            PoolMode::Max => {
                let dims = x.dims();
                x.max_dim(2).max_dim(3).reshape([dims[0], dims[1], 1, 1])
            }
            PoolMode::AvgMax => {
                let avg = x.clone().mean_dim(2).mean_dim(3);
                let max = x.max_dim(2).max_dim(3);
                (avg + max) * 0.5
            }
            PoolMode::CatAvgMax => {
                let avg = x.clone().mean_dim(2).mean_dim(3);
                let max = x.max_dim(2).max_dim(3);
                Tensor::cat(vec![avg, max], 1)
            }
            PoolMode::Identity => x,
        }
    }
}

/// Patch Representation Refinement (PRR) pooling (`timm.layers.AttentionPoolPrr`).
///
/// Parameter-free multi-head self-attention that refines patch representations
/// before pooling.
#[derive(Module, Debug)]
pub struct AttentionPoolPrr {
    /// Optional pre-normalization.
    pub pre_norm: Option<LayerNorm>,
    /// Optional post-normalization.
    pub post_norm: Option<LayerNorm>,
    #[module(skip)]
    /// Number of attention heads.
    pub num_heads: usize,
    #[module(skip)]
    /// Dimension per head.
    pub head_dim: usize,
    #[module(skip)]
    /// Attention scale factor.
    pub scale: f64,
    #[module(skip)]
    /// Pool type ("token" or "avg").
    pub pool_type: String,
}

impl AttentionPoolPrr {
    /// Creates an `AttentionPoolPrr` layer.
    pub fn new(
        dim: usize,
        num_heads: usize,
        pool_type: &str,
        pre_norm: bool,
        post_norm: bool,
        device: &Device,
    ) -> Self {
        assert_eq!(dim % num_heads, 0, "dim must be divisible by num_heads");
        let head_dim = dim / num_heads;
        let scale = (head_dim as f64).powf(-0.5);

        let pre_norm = pre_norm.then(|| LayerNormConfig::new(dim).with_epsilon(1e-6).init(device));
        let post_norm =
            post_norm.then(|| LayerNormConfig::new(dim).with_epsilon(1e-6).init(device));

        Self {
            pre_norm,
            post_norm,
            num_heads,
            head_dim,
            scale,
            pool_type: pool_type.to_string(),
        }
    }

    /// Applies PRR attention pooling to token sequence `[B, N, C] -> [B, C]`.
    pub fn forward(&self, x: Tensor<3>) -> Tensor<2> {
        let [b, n, c] = x.dims();

        let x = match &self.pre_norm {
            Some(norm) => norm.forward(x),
            None => x,
        };

        // Reshape parameter-free QKV: [B, num_heads, N, head_dim]
        let qkv = x
            .reshape([b, n, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let q = qkv.clone() * self.scale;
        let k_t = qkv.clone().permute([0, 1, 3, 2]);
        let scores = q.matmul(k_t); // [B, num_heads, N, N]
        let attn = softmax(scores, 3);
        let out = attn.matmul(qkv); // [B, num_heads, N, head_dim]

        let out = out.permute([0, 2, 1, 3]).reshape([b, n, c]);
        let out = match &self.post_norm {
            Some(norm) => norm.forward(out),
            None => out,
        };

        if self.pool_type == "avg" {
            out.mean_dim(1).reshape([b, c])
        } else {
            out.slice([0..b, 0..1, 0..c]).reshape([b, c])
        }
    }
}
