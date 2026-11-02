//! Attention pooling with learned latent queries (`AttentionPoolLatent` / MAP head).

use burn::module::Param;
use burn::nn::{Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;

use crate::layers::mlp::Mlp;

/// Attention pooling with learned latent queries (`timm.layers.AttentionPoolLatent`).
///
/// Used by models without a `cls_token` (such as SigLIP) or for multi-scale attention pooling.
#[derive(Module, Debug)]
pub struct AttentionPoolLatent {
    /// Learnable latent queries `[1, latent_len, latent_dim]`.
    pub latent: Param<Tensor<3>>,
    /// Query projection.
    pub q: Option<Linear>,
    /// Key/value projection.
    pub kv: Linear,
    /// Query normalization.
    pub q_norm: Option<LayerNorm>,
    /// Key normalization.
    pub k_norm: Option<LayerNorm>,
    /// Output projection.
    pub proj: Option<Linear>,
    /// Normalization before MLP.
    pub norm: Option<LayerNorm>,
    /// MLP refinement.
    pub mlp: Option<Mlp>,
    #[module(skip)]
    /// Number of attention heads.
    pub num_heads: usize,
    #[module(skip)]
    /// Dimension per head.
    pub head_dim: usize,
    #[module(skip)]
    /// Scaling factor.
    pub scale: f64,
    #[module(skip)]
    /// Pooling type ("token" or "avg").
    pub pool: String,
    #[module(skip)]
    /// Length of latent sequence.
    pub latent_len: usize,
    #[module(skip)]
    /// Dimension of latent tokens.
    pub latent_dim: usize,
}

impl AttentionPoolLatent {
    /// Creates an `AttentionPoolLatent` module.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_features: usize,
        out_features: Option<usize>,
        embed_dim: Option<usize>,
        num_heads: usize,
        mlp_ratio: f64,
        qkv_bias: bool,
        q_proj: bool,
        qk_norm: bool,
        latent_len: usize,
        pool_type: &str,
        device: &Device,
    ) -> Self {
        let embed_dim = embed_dim.unwrap_or(in_features);
        let out_features = out_features.unwrap_or(in_features);
        assert_eq!(
            embed_dim % num_heads,
            0,
            "embed_dim must be divisible by num_heads"
        );
        let head_dim = embed_dim / num_heads;
        let scale = (head_dim as f64).powf(-0.5);

        let latent_dim = embed_dim;
        let latent = Initializer::Zeros.init([1, latent_len, latent_dim], device);

        let q = q_proj.then(|| {
            LinearConfig::new(latent_dim, embed_dim)
                .with_bias(qkv_bias)
                .init(device)
        });

        let kv = LinearConfig::new(embed_dim, embed_dim * 2)
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

        let (proj, norm, mlp) = if out_features > 0 {
            let proj = LinearConfig::new(embed_dim, out_features).init(device);
            let norm = LayerNormConfig::new(out_features)
                .with_epsilon(1e-6)
                .init(device);
            let hidden_features = (out_features as f64 * mlp_ratio) as usize;
            let mlp = Mlp::new(out_features, hidden_features, out_features, true, device);
            (Some(proj), Some(norm), Some(mlp))
        } else {
            (None, None, None)
        };

        Self {
            latent,
            q,
            kv,
            q_norm,
            k_norm,
            proj,
            norm,
            mlp,
            num_heads,
            head_dim,
            scale,
            pool: pool_type.to_string(),
            latent_len,
            latent_dim,
        }
    }

    /// Applies latent attention pooling to token sequence `[B, N, C] -> [B, out_features]`.
    #[allow(clippy::single_range_in_vec_init)]
    pub fn forward(&self, x: Tensor<3>) -> Tensor<2> {
        let [b, n, _c] = x.dims();

        // Latents: repeat across batch -> [B, latent_len, latent_dim]
        let q_latent = self.latent.val().repeat_dim(0, b);
        let q = match &self.q {
            Some(proj) => proj.forward(q_latent),
            None => q_latent,
        };
        let q = q
            .reshape([b, self.latent_len, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]); // [B, num_heads, latent_len, head_dim]

        let kv = self.kv.forward(x);
        let kv = kv
            .reshape([b, n, 2, self.num_heads, self.head_dim])
            .permute([2, 0, 3, 1, 4]);

        let mut k = kv.clone().slice([0..1]).squeeze::<4>();
        let v = kv.slice([1..2]).squeeze::<4>();

        let q = match &self.q_norm {
            Some(norm) => norm.forward(q),
            None => q,
        };
        if let Some(norm) = &self.k_norm {
            k = norm.forward(k);
        }

        let q = q * self.scale;
        let k_t = k.permute([0, 1, 3, 2]);
        let scores = q.matmul(k_t); // [B, num_heads, latent_len, N]
        let attn = softmax(scores, 3);
        let out = attn.matmul(v); // [B, num_heads, latent_len, head_dim]

        let out =
            out.permute([0, 2, 1, 3])
                .reshape([b, self.latent_len, self.head_dim * self.num_heads]);

        let out = match &self.proj {
            Some(proj) => proj.forward(out),
            None => out,
        };

        let out = if let (Some(mlp), Some(norm)) = (&self.mlp, &self.norm) {
            out.clone() + mlp.forward(norm.forward(out))
        } else {
            out
        };

        let [_b, _len, out_dim] = out.dims();
        if self.pool == "avg" {
            out.mean_dim(1).squeeze::<2>()
        } else {
            out.slice([0..b, 0..1, 0..out_dim]).squeeze::<2>()
        }
    }
}
