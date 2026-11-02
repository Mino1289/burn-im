//! Multi-head self-attention module (`Attention`).

use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::Bool;
use burn::tensor::activation::softmax;

use crate::layers::pos_embed::{RotaryEmbeddingCat, apply_rot_embed_cat};

/// Attention mask configuration for [`Attention`].
#[derive(Debug, Clone)]
pub enum AttentionMask {
    /// Additive float bias mask `[B, num_heads, N, N]` (or broadcastable `[1, 1, N, N]`).
    Float(Tensor<4>),
    /// 4D boolean mask `[B, 1, N, N]` (or `[1, 1, N, N]`) where `true` keeps token and `false` masks out.
    Bool(Tensor<4, Bool>),
    /// Key-only 2D boolean mask `[B, N]` where `true` keeps key token and `false` masks out.
    KeyOnly(Tensor<2, Bool>),
}

/// Multi-head self-attention module (`timm.models.vision_transformer.Attention`).
#[derive(Module, Debug)]
pub struct Attention {
    /// Combined QKV linear projection.
    pub qkv: Linear,
    /// Optional LayerNorm on queries.
    pub q_norm: Option<LayerNorm>,
    /// Optional LayerNorm on keys.
    pub k_norm: Option<LayerNorm>,
    /// Output projection.
    pub proj: Linear,
    #[module(skip)]
    /// Number of attention heads.
    pub num_heads: usize,
    #[module(skip)]
    /// Dimension per head.
    pub head_dim: usize,
    #[module(skip)]
    /// Scaling factor `1 / sqrt(head_dim)`.
    pub scale: f64,
    /// Optional rotary position embedding.
    pub rope: Option<RotaryEmbeddingCat>,
}

impl Attention {
    /// Creates a multi-head self-attention module.
    pub fn new(
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        qk_norm: bool,
        proj_bias: bool,
        device: &Device,
    ) -> Self {
        assert_eq!(dim % num_heads, 0, "dim must be divisible by num_heads");
        let head_dim = dim / num_heads;
        let scale = (head_dim as f64).powf(-0.5);

        let qkv = LinearConfig::new(dim, dim * 3)
            .with_bias(qkv_bias)
            .init(device);

        let q_norm = qk_norm.then(|| {
            LayerNormConfig::new(head_dim)
                .with_epsilon(1e-6)
                .init(device)
        });
        let k_norm = qk_norm.then(|| {
            LayerNormConfig::new(head_dim)
                .with_epsilon(1e-6)
                .init(device)
        });

        let proj = LinearConfig::new(dim, dim)
            .with_bias(proj_bias)
            .init(device);

        Self {
            qkv,
            q_norm,
            k_norm,
            proj,
            num_heads,
            head_dim,
            scale,
            rope: None,
        }
    }

    /// Attaches a rotary position embedding module to this attention layer.
    pub fn with_rope(mut self, rope: RotaryEmbeddingCat) -> Self {
        self.rope = Some(rope);
        self
    }

    /// Convenience forward pass without masks or causal restrictions.
    pub fn forward_plain(&self, x: Tensor<3>) -> Tensor<3> {
        self.forward(x, None, false)
    }

    /// Applies multi-head self-attention to token sequences `[B, N, C]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
        attn_mask: Option<AttentionMask>,
        is_causal: bool,
    ) -> Tensor<3> {
        let [b, n, c] = x.dims();
        let device = x.device();

        // Combined QKV projection: [B, N, 3 * C]
        let qkv = self.qkv.forward(x);

        let q = qkv.clone().slice([0..b, 0..n, 0..c]);
        let k = qkv.clone().slice([0..b, 0..n, c..2 * c]);
        let v = qkv.slice([0..b, 0..n, 2 * c..3 * c]);

        let mut q = q
            .reshape([b, n, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let mut k = k
            .reshape([b, n, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let v = v
            .reshape([b, n, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);

        if let Some(norm) = &self.q_norm {
            q = norm.forward(q);
        }
        if let Some(norm) = &self.k_norm {
            k = norm.forward(k);
        }

        if let Some(rope) = &self.rope {
            let hw = (n as f64).sqrt().round() as usize;
            let emb = rope.get_embed((hw, hw));
            q = apply_rot_embed_cat(q, emb.clone(), rope.rotate_half);
            k = apply_rot_embed_cat(k, emb, rope.rotate_half);
        }

        let q = q * self.scale;
        // q: [B, num_heads, N, head_dim], k: [B, num_heads, N, head_dim] -> k^T: [B, num_heads, head_dim, N]
        let k_t = k.permute([0, 1, 3, 2]);
        let mut scores = q.matmul(k_t); // [B, num_heads, N, N]

        if is_causal {
            // Upper triangular causal mask with -1e9
            let mut mask_data = vec![0.0f32; n * n];
            for i in 0..n {
                for j in (i + 1)..n {
                    mask_data[i * n + j] = -1e9;
                }
            }
            let causal_mask = Tensor::from_data(
                burn::tensor::TensorData::new(mask_data, [1, 1, n, n]),
                &device,
            );
            scores = scores + causal_mask;
        } else if let Some(mask) = attn_mask {
            match mask {
                AttentionMask::Float(bias) => {
                    scores = scores + bias;
                }
                AttentionMask::Bool(mask_bool) => {
                    scores = scores.mask_fill(mask_bool.bool_not(), -1e9);
                }
                AttentionMask::KeyOnly(key_mask) => {
                    let [b_m, n_k] = key_mask.dims();
                    let mask_4d = key_mask.reshape([b_m, 1, 1, n_k]);
                    scores = scores.mask_fill(mask_4d.bool_not(), -1e9);
                }
            }
        }

        let attn_weights = softmax(scores, 3);
        let out = attn_weights.matmul(v); // [B, num_heads, N, head_dim]

        // Permute and reshape back to [B, N, C]
        let out = out.permute([0, 2, 1, 3]).reshape([b, n, c]);
        self.proj.forward(out)
    }
}
