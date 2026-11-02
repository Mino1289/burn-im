//! Vision Transformer (`ViT`, `DeiT-III`, `DINOv2`, `SigLIP`) backbones.
//!
//! Reproduces `timm`'s `VisionTransformer` (see `timm/models/vision_transformer.py`):
//! - Patch-based embedding stem with non-overlapping 2D convolutions.
//! - Dynamic prefix token management (`cls_token` and `reg_tokens`).
//! - Absolute 2D position embeddings (with bicubic grid resampling) or rotary position embeddings (RoPE).
//! - Transformer encoder blocks with multi-head self-attention, optional LayerScale, and MLP / SwiGLU.
//! - Modern pooling heads: class token, global average, MAP head ([`AttentionPoolLatent`]), and PRR ([`AttentionPoolPrr`]).
//! - Multi-scale intermediate features (`blocks.0` through `blocks.N-1`) converted automatically
//!   from token sequences to 2D spatial feature maps in `[B, C, H, W]` layout.
//! - Truncated construction via `features_only` and `out_indices`.

use burn::module::{Param, ParamId};
use burn::nn::conv::Conv2d;
use burn::nn::{Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;

use crate::error::{Error, Result};
use crate::hub::config::PretrainedCfg;
use crate::layers::attention::{Attention, AttentionMask};
use crate::layers::attention_pool::AttentionPoolLatent;
use crate::layers::drop::{DropPath, drop_path_schedule};
use crate::layers::mlp::Mlp;
use crate::layers::patch_embed::PatchEmbed;
use crate::layers::pool::AttentionPoolPrr;
use crate::layers::pos_embed::resample_abs_pos_embed;
use crate::models::ModelMeta;
use crate::traits::{FeatureFormat, FeatureInfo, IntermediatesOptions, VisionModel};

/// Pooling mode for [`VisionTransformer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViTPoolMode {
    /// Extracts the leading class token `cls_token`.
    #[default]
    Token,
    /// Averages across spatial patch tokens.
    Avg,
    /// Multi-head latent attention pooling (MAP head / [`AttentionPoolLatent`]).
    Map,
    /// Patch Representation Refinement ([`AttentionPoolPrr`]).
    Prr,
}

/// Configuration for [`VisionTransformer`].
#[derive(Debug, Clone)]
pub struct ViTConfig {
    /// Nominal input image resolution `(height, width)`.
    pub img_size: (usize, usize),
    /// Patch dimensions `(patch_h, patch_w)`.
    pub patch_size: (usize, usize),
    /// Number of input channels (typically 3 for RGB).
    pub in_chans: usize,
    /// Number of target classes. `0` builds a headless model.
    pub num_classes: usize,
    /// Hidden embedding dimension across blocks.
    pub embed_dim: usize,
    /// Number of transformer encoder blocks.
    pub depth: usize,
    /// Number of attention heads in each block.
    pub num_heads: usize,
    /// Expansion ratio for the MLP hidden layer.
    pub mlp_ratio: f64,
    /// Whether QKV linear projections include learnable bias.
    pub qkv_bias: bool,
    /// Whether Query and Key vectors are normalized before dot-product.
    pub qk_norm: bool,
    /// Initial value for LayerScale. `None` disables LayerScale.
    pub init_values: Option<f64>,
    /// Whether to insert a learnable class token.
    pub class_token: bool,
    /// Whether class token is excluded from positional embedding addition.
    pub no_embed_class: bool,
    /// Number of register tokens (e.g. 4 for DINOv2 with registers).
    pub reg_tokens: usize,
    /// Whether to use Rotary Position Embedding (RoPE) instead of absolute pos_embed.
    pub use_rope: bool,
    /// Pooling mode before the classifier.
    pub pool_mode: ViTPoolMode,
    /// Maximum stochastic depth rate across blocks.
    pub drop_path_rate: f64,
    /// Build a truncated feature extractor stopping at the deepest requested block.
    pub features_only: bool,
    /// Block levels retained when [`Self::features_only`] is set.
    pub out_indices: Option<Vec<usize>>,
}

impl ViTConfig {
    /// Creates a ViT configuration with standard defaults.
    pub fn new(embed_dim: usize, depth: usize, num_heads: usize) -> Self {
        Self {
            img_size: (224, 224),
            patch_size: (16, 16),
            in_chans: 3,
            num_classes: 1000,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio: 4.0,
            qkv_bias: true,
            qk_norm: false,
            init_values: None,
            class_token: true,
            no_embed_class: false,
            reg_tokens: 0,
            use_rope: false,
            pool_mode: ViTPoolMode::Token,
            drop_path_rate: 0.0,
            features_only: false,
            out_indices: None,
        }
    }

    /// Sets the input image resolution.
    pub fn with_img_size(mut self, img_size: (usize, usize)) -> Self {
        self.img_size = img_size;
        self
    }

    /// Sets the patch size.
    pub fn with_patch_size(mut self, patch_size: (usize, usize)) -> Self {
        self.patch_size = patch_size;
        self
    }

    /// Sets the number of input channels.
    pub fn with_in_chans(mut self, in_chans: usize) -> Self {
        self.in_chans = in_chans;
        self
    }

    /// Sets the number of target classes. `0` removes the classifier head.
    pub fn with_num_classes(mut self, num_classes: usize) -> Self {
        self.num_classes = num_classes;
        self
    }

    /// Sets the expansion ratio for the MLP hidden dimension.
    pub fn with_mlp_ratio(mut self, mlp_ratio: f64) -> Self {
        self.mlp_ratio = mlp_ratio;
        self
    }

    /// Sets whether class token is excluded from positional embedding addition.
    pub fn with_no_embed_class(mut self, no_embed_class: bool) -> Self {
        self.no_embed_class = no_embed_class;
        self
    }

    /// Sets whether QKV projections have bias.
    pub fn with_qkv_bias(mut self, qkv_bias: bool) -> Self {
        self.qkv_bias = qkv_bias;
        self
    }

    /// Sets whether QK normalization is applied.
    pub fn with_qk_norm(mut self, qk_norm: bool) -> Self {
        self.qk_norm = qk_norm;
        self
    }

    /// Sets the initial LayerScale parameter value.
    pub fn with_init_values(mut self, init_values: Option<f64>) -> Self {
        self.init_values = init_values;
        self
    }

    /// Sets whether a class token is used.
    pub fn with_class_token(mut self, class_token: bool) -> Self {
        self.class_token = class_token;
        self
    }

    /// Sets the number of register tokens.
    pub fn with_reg_tokens(mut self, reg_tokens: usize) -> Self {
        self.reg_tokens = reg_tokens;
        self
    }

    /// Sets whether RoPE is used instead of absolute positional embedding.
    pub fn with_use_rope(mut self, use_rope: bool) -> Self {
        self.use_rope = use_rope;
        self
    }

    /// Sets the pooling mode.
    pub fn with_pool_mode(mut self, pool_mode: ViTPoolMode) -> Self {
        self.pool_mode = pool_mode;
        self
    }

    /// Sets the stochastic depth drop-path rate.
    pub fn with_drop_path_rate(mut self, drop_path_rate: f64) -> Self {
        self.drop_path_rate = drop_path_rate;
        self
    }

    /// Builds a truncated feature extractor retaining only the levels in `out_indices`.
    pub fn with_features_only(mut self, out_indices: impl IntoIterator<Item = usize>) -> Self {
        self.features_only = true;
        self.num_classes = 0;
        self.out_indices = Some(out_indices.into_iter().collect());
        self
    }

    /// Validates configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if self.in_chans == 0 {
            return Err(Error::InvalidConfiguration(
                "in_chans must be greater than 0".to_string(),
            ));
        }
        if self.patch_size.0 == 0 || self.patch_size.1 == 0 {
            return Err(Error::InvalidConfiguration(
                "patch_size must be positive".to_string(),
            ));
        }
        if !self.embed_dim.is_multiple_of(self.num_heads) {
            return Err(Error::InvalidConfiguration(format!(
                "embed_dim ({}) must be divisible by num_heads ({})",
                self.embed_dim, self.num_heads
            )));
        }
        if self.features_only {
            if self.num_classes != 0 {
                return Err(Error::InvalidConfiguration(
                    "`features_only` requires `num_classes == 0`".to_string(),
                ));
            }
            if let Some(indices) = &self.out_indices {
                if indices.is_empty() {
                    return Err(Error::InvalidConfiguration(
                        "`out_indices` must not be empty".to_string(),
                    ));
                }
                for &idx in indices {
                    if idx >= self.depth {
                        return Err(Error::InvalidConfiguration(format!(
                            "out_index {idx} out of bounds (model exposes {} blocks: 0..={})",
                            self.depth,
                            self.depth.saturating_sub(1)
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// Presets
// ----------------------------------------------------------------------------

/// `vit_tiny_patch16_224`: dim 192, depth 12, heads 3.
pub fn vit_tiny_patch16_224(num_classes: usize) -> ViTConfig {
    ViTConfig::new(192, 12, 3).with_num_classes(num_classes)
}

/// `vit_small_patch16_224`: dim 384, depth 12, heads 6.
pub fn vit_small_patch16_224(num_classes: usize) -> ViTConfig {
    ViTConfig::new(384, 12, 6).with_num_classes(num_classes)
}

/// `vit_base_patch16_224`: dim 768, depth 12, heads 12.
pub fn vit_base_patch16_224(num_classes: usize) -> ViTConfig {
    ViTConfig::new(768, 12, 12).with_num_classes(num_classes)
}

/// `vit_large_patch16_224`: dim 1024, depth 24, heads 16.
pub fn vit_large_patch16_224(num_classes: usize) -> ViTConfig {
    ViTConfig::new(1024, 24, 16).with_num_classes(num_classes)
}

/// `deit_base_patch16_224`: dim 768, depth 12, heads 12.
pub fn deit_base_patch16_224(num_classes: usize) -> ViTConfig {
    vit_base_patch16_224(num_classes)
}

/// `vit_small_patch14_dinov2`: patch 14, dim 384, depth 12, heads 6, LayerScale 1e-5.
pub fn vit_small_patch14_dinov2(num_classes: usize) -> ViTConfig {
    ViTConfig::new(384, 12, 6)
        .with_img_size((518, 518))
        .with_patch_size((14, 14))
        .with_init_values(Some(1e-5))
        .with_num_classes(num_classes)
}

/// `vit_base_patch16_siglip_224`: patch 16, dim 768, depth 12, heads 12, MAP attention pool head.
pub fn vit_base_patch16_siglip_224(num_classes: usize) -> ViTConfig {
    ViTConfig::new(768, 12, 12)
        .with_class_token(false)
        .with_pool_mode(ViTPoolMode::Map)
        .with_num_classes(num_classes)
}

// ----------------------------------------------------------------------------
// Architectural Blocks
// ----------------------------------------------------------------------------

/// Transformer encoder block of [`VisionTransformer`].
#[derive(Module, Debug)]
pub struct ViTBlock {
    /// Pre-attention layer normalization.
    pub norm1: LayerNorm,
    /// Multi-head self-attention module.
    pub attn: Attention,
    /// Optional LayerScale vector for attention output.
    pub ls1: Option<Param<Tensor<1>>>,
    /// Stochastic depth for attention branch.
    pub drop_path1: DropPath,
    /// Pre-MLP layer normalization.
    pub norm2: LayerNorm,
    /// MLP feed-forward block.
    pub mlp: Mlp,
    /// Optional LayerScale vector for MLP output.
    pub ls2: Option<Param<Tensor<1>>>,
    /// Stochastic depth for MLP branch.
    pub drop_path2: DropPath,
}

impl ViTBlock {
    /// Creates a transformer encoder block.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        num_heads: usize,
        mlp_ratio: f64,
        qkv_bias: bool,
        qk_norm: bool,
        init_values: Option<f64>,
        drop_path_rate: f64,
        device: &Device,
    ) -> Self {
        let norm1 = LayerNormConfig::new(dim).with_epsilon(1e-6).init(device);
        let attn = Attention::new(dim, num_heads, qkv_bias, qk_norm, true, device);
        let ls1 = init_values.map(|v| Initializer::Constant { value: v }.init([dim], device));
        let drop_path1 = DropPath::new(drop_path_rate);

        let norm2 = LayerNormConfig::new(dim).with_epsilon(1e-6).init(device);
        let hidden_dim = (dim as f64 * mlp_ratio) as usize;
        let mlp = Mlp::new(dim, hidden_dim, dim, true, device);
        let ls2 = init_values.map(|v| Initializer::Constant { value: v }.init([dim], device));
        let drop_path2 = DropPath::new(drop_path_rate);

        Self {
            norm1,
            attn,
            ls1,
            drop_path1,
            norm2,
            mlp,
            ls2,
            drop_path2,
        }
    }

    /// Forward pass through the encoder block: `[B, N, C] -> [B, N, C]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
        attn_mask: Option<AttentionMask>,
        is_causal: bool,
    ) -> Tensor<3> {
        let shortcut = x.clone();
        let mut attn_out = self
            .attn
            .forward(self.norm1.forward(x), attn_mask, is_causal);
        if let Some(ls1) = &self.ls1 {
            let dim = ls1.dims()[0];
            attn_out = attn_out * ls1.val().reshape([1, 1, dim]);
        }
        let x = shortcut + self.drop_path1.forward_3d(attn_out);

        let shortcut = x.clone();
        let mut mlp_out = self.mlp.forward(self.norm2.forward(x));
        if let Some(ls2) = &self.ls2 {
            let dim = ls2.dims()[0];
            mlp_out = mlp_out * ls2.val().reshape([1, 1, dim]);
        }
        shortcut + self.drop_path2.forward_3d(mlp_out)
    }
}

// ----------------------------------------------------------------------------
// Model Structure
// ----------------------------------------------------------------------------

/// Pure-Rust implementation of Vision Transformer (`ViT`).
#[derive(Module, Debug)]
pub struct VisionTransformer {
    /// Patch embedding stem.
    pub patch_embed: PatchEmbed,
    /// Optional learnable class token `[1, 1, embed_dim]`.
    pub cls_token: Option<Param<Tensor<3>>>,
    /// Optional learnable register tokens `[1, num_reg, embed_dim]`.
    pub reg_tokens: Option<Param<Tensor<3>>>,
    /// Optional absolute positional embeddings `[1, num_patches + num_prefix, embed_dim]`.
    pub pos_embed: Option<Param<Tensor<3>>>,
    /// Sequence of transformer encoder blocks.
    pub blocks: Vec<ViTBlock>,
    /// Post-transformer layer normalization.
    pub norm: LayerNorm,
    /// Classification head linear projection.
    pub head: Option<Linear>,
    /// Optional MAP attention pooling head.
    pub attn_pool: Option<AttentionPoolLatent>,
    /// Optional PRR attention pooling head.
    pub prr_pool: Option<AttentionPoolPrr>,
    #[module(skip)]
    meta: ModelMeta,
    #[module(skip)]
    pool_mode: ViTPoolMode,
    #[module(skip)]
    no_embed_class: bool,
    #[module(skip)]
    num_prefix_tokens: usize,
}

impl VisionTransformer {
    /// Constructs a `VisionTransformer` model from `config`.
    pub fn new(config: ViTConfig, device: &Device) -> Self {
        config
            .validate()
            .unwrap_or_else(|err| panic!("invalid ViT configuration: {err}"));

        let patch_embed = PatchEmbed::new(
            config.img_size,
            config.patch_size,
            config.in_chans,
            config.embed_dim,
            false,
            true,
            device,
        );

        let num_prefix_tokens = (config.class_token as usize) + config.reg_tokens;

        let cls_token = config
            .class_token
            .then(|| Initializer::Zeros.init([1, 1, config.embed_dim], device));

        let reg_tokens = (config.reg_tokens > 0)
            .then(|| Initializer::Zeros.init([1, config.reg_tokens, config.embed_dim], device));

        let pos_embed = if !config.use_rope {
            let total_patches = patch_embed.num_patches;
            let embed_len = if config.no_embed_class {
                total_patches
            } else {
                total_patches + num_prefix_tokens
            };
            Some(Initializer::Zeros.init([1, embed_len, config.embed_dim], device))
        } else {
            None
        };

        let dp_schedule = drop_path_schedule(config.drop_path_rate, config.depth);
        let max_block = if config.features_only {
            config
                .out_indices
                .as_ref()
                .and_then(|indices| indices.iter().max().copied())
                .unwrap_or(config.depth - 1)
        } else {
            config.depth - 1
        };

        let blocks = (0..=max_block)
            .map(|i| {
                ViTBlock::new(
                    config.embed_dim,
                    config.num_heads,
                    config.mlp_ratio,
                    config.qkv_bias,
                    config.qk_norm,
                    config.init_values,
                    dp_schedule[i],
                    device,
                )
            })
            .collect();

        let norm = LayerNormConfig::new(config.embed_dim)
            .with_epsilon(1e-6)
            .init(device);

        let (head, attn_pool, prr_pool) = if config.num_classes > 0 {
            match config.pool_mode {
                ViTPoolMode::Map => {
                    let pool = AttentionPoolLatent::new(
                        config.embed_dim,
                        Some(config.num_classes),
                        Some(config.embed_dim),
                        config.num_heads,
                        config.mlp_ratio,
                        true,
                        true,
                        false,
                        1,
                        "token",
                        device,
                    );
                    (None, Some(pool), None)
                }
                ViTPoolMode::Prr => {
                    let pool = AttentionPoolPrr::new(
                        config.embed_dim,
                        config.num_heads,
                        "token",
                        false,
                        false,
                        device,
                    );
                    let head = LinearConfig::new(config.embed_dim, config.num_classes).init(device);
                    (Some(head), None, Some(pool))
                }
                _ => {
                    let head = LinearConfig::new(config.embed_dim, config.num_classes).init(device);
                    (Some(head), None, None)
                }
            }
        } else {
            (None, None, None)
        };

        let feature_info = (0..config.depth)
            .map(|i| {
                FeatureInfo::new(
                    i,
                    config.embed_dim,
                    config.patch_size.0,
                    Box::leak(format!("blocks.{i}").into_boxed_str()),
                )
            })
            .collect();

        let meta = ModelMeta {
            name: String::new(),
            num_classes: config.num_classes,
            num_features: config.embed_dim,
            feature_info,
            pretrained_cfg: None,
        };

        Self {
            patch_embed,
            cls_token,
            reg_tokens,
            pos_embed,
            blocks,
            norm,
            head,
            attn_pool,
            prr_pool,
            meta,
            pool_mode: config.pool_mode,
            no_embed_class: config.no_embed_class,
            num_prefix_tokens,
        }
    }

    /// Attaches metadata to the model instance.
    pub fn set_meta(&mut self, meta: ModelMeta) {
        self.meta = meta;
    }

    /// First convolution of the patch embedding stem.
    pub fn first_conv_mut(&mut self) -> Option<&mut Conv2d> {
        Some(&mut self.patch_embed.proj)
    }

    /// Replaces first convolution weights (used when adapting `in_chans`).
    pub fn set_first_conv_weight(&mut self, weight: Tensor<4>) {
        self.patch_embed.proj.weight = Param::initialized(ParamId::new(), weight);
    }

    /// Key remapping rules for checkpoint loading from PyTorch `timm`.
    pub fn key_remaps() -> Vec<(&'static str, &'static str)> {
        vec![
            ("^blocks\\.(\\d+)\\.ls1\\.gamma$", "blocks.$1.ls1"),
            ("^blocks\\.(\\d+)\\.ls2\\.gamma$", "blocks.$1.ls2"),
            ("^reg_token$", "reg_tokens"),
        ]
    }

    fn apply_pos_embed(&self, x: Tensor<3>, orig_h: usize, orig_w: usize) -> Tensor<3> {
        let [b, _n, _c] = x.dims();
        let mut tokens_to_cat = Vec::new();

        if let Some(cls) = &self.cls_token {
            tokens_to_cat.push(cls.val().repeat_dim(0, b));
        }
        if let Some(reg) = &self.reg_tokens {
            tokens_to_cat.push(reg.val().repeat_dim(0, b));
        }

        let Some(pos_embed) = &self.pos_embed else {
            return if !tokens_to_cat.is_empty() {
                tokens_to_cat.push(x);
                Tensor::cat(tokens_to_cat, 1)
            } else {
                x
            };
        };

        let current_grid = self.patch_embed.dynamic_feat_size((orig_h, orig_w));
        let pos_embed = if current_grid != self.patch_embed.grid_size {
            resample_abs_pos_embed(
                pos_embed.val(),
                current_grid,
                Some(self.patch_embed.grid_size),
                if self.no_embed_class {
                    0
                } else {
                    self.num_prefix_tokens
                },
            )
        } else {
            pos_embed.val()
        };

        if self.no_embed_class {
            let x = x + pos_embed;
            if !tokens_to_cat.is_empty() {
                tokens_to_cat.push(x);
                Tensor::cat(tokens_to_cat, 1)
            } else {
                x
            }
        } else {
            let x = if !tokens_to_cat.is_empty() {
                tokens_to_cat.push(x);
                Tensor::cat(tokens_to_cat, 1)
            } else {
                x
            };
            x + pos_embed
        }
    }

    fn run_intermediates(
        &self,
        input: Tensor<4>,
        options: &IntermediatesOptions,
    ) -> (Tensor<4>, Vec<Tensor<4>>) {
        let [b, _c, in_h, in_w] = input.dims();
        let (grid_h, grid_w) = self.patch_embed.dynamic_feat_size((in_h, in_w));

        let (selected, max_idx) = options
            .resolve(self.feature_info().len())
            .unwrap_or_else(|err| panic!("{err}"));

        let mut x = self.patch_embed.forward(input);
        x = self.apply_pos_embed(x, in_h, in_w);

        let mut intermediates = Vec::new();

        for (idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(x, None, false);

            if selected.contains(&idx) {
                let intermediate = if options.norm {
                    self.norm.forward(x.clone())
                } else {
                    x.clone()
                };

                // Strip prefix tokens (cls_token, reg_tokens) to get spatial tokens
                let spatial = if self.num_prefix_tokens > 0 {
                    let total_n = intermediate.dims()[1];
                    intermediate.slice([0..b, self.num_prefix_tokens..total_n])
                } else {
                    intermediate
                };

                let out = match options.format {
                    FeatureFormat::Nchw => spatial
                        .reshape([b, grid_h, grid_w, self.meta.num_features])
                        .permute([0, 3, 1, 2]),
                    FeatureFormat::Nhwc => {
                        spatial.reshape([b, grid_h, grid_w, self.meta.num_features])
                    }
                };
                intermediates.push(out);
            }

            if options.stop_early && idx == max_idx {
                break;
            }
        }

        let final_normed = self.norm.forward(x);
        let [_b, total_tokens, channels] = final_normed.dims();
        // Return 4D representation [B, 1, N, C] preserving prefix tokens for forward_head
        let final_4d = final_normed.reshape([b, 1, total_tokens, channels]);

        (final_4d, intermediates)
    }
}

impl VisionModel for VisionTransformer {
    fn forward(&self, input: Tensor<4>) -> Tensor<2> {
        let features = self.forward_features(input);
        self.forward_head(features, false)
    }

    fn forward_features(&self, input: Tensor<4>) -> Tensor<4> {
        let [b, _c, in_h, in_w] = input.dims();
        let mut x = self.patch_embed.forward(input);
        x = self.apply_pos_embed(x, in_h, in_w);

        for block in &self.blocks {
            x = block.forward(x, None, false);
        }
        let x = self.norm.forward(x);
        let [_b, total_tokens, channels] = x.dims();
        x.reshape([b, 1, total_tokens, channels])
    }

    fn forward_head(&self, features: Tensor<4>, pre_logits: bool) -> Tensor<2> {
        let [b, _one, total_tokens, channels] = features.dims();
        let x = features.reshape([b, total_tokens, channels]);

        if let Some(attn_pool) = &self.attn_pool {
            return attn_pool.forward(x);
        }

        let pooled = match self.pool_mode {
            ViTPoolMode::Token => x.slice([0..b, 0..1, 0..channels]).reshape([b, channels]),
            ViTPoolMode::Avg => {
                if self.num_prefix_tokens > 0 {
                    x.slice([0..b, self.num_prefix_tokens..total_tokens, 0..channels])
                        .mean_dim(1)
                        .reshape([b, channels])
                } else {
                    x.mean_dim(1).reshape([b, channels])
                }
            }
            ViTPoolMode::Prr => {
                let pool = self.prr_pool.as_ref().unwrap();
                pool.forward(x)
            }
            ViTPoolMode::Map => unreachable!(),
        };

        if pre_logits {
            pooled
        } else if let Some(head) = &self.head {
            head.forward(pooled)
        } else {
            pooled
        }
    }

    fn forward_intermediates(
        &self,
        input: Tensor<4>,
        options: &IntermediatesOptions,
    ) -> Vec<Tensor<4>> {
        self.run_intermediates(input, options).1
    }

    fn forward_features_with_intermediates(
        &self,
        input: Tensor<4>,
        options: &IntermediatesOptions,
    ) -> (Tensor<4>, Vec<Tensor<4>>) {
        self.run_intermediates(input, options)
    }

    fn feature_info(&self) -> &[FeatureInfo] {
        &self.meta.feature_info
    }

    fn pretrained_cfg(&self) -> Option<&PretrainedCfg> {
        self.meta.pretrained_cfg.as_ref()
    }

    fn num_classes(&self) -> usize {
        self.meta.num_classes
    }

    fn num_features(&self) -> usize {
        self.meta.num_features
    }

    fn model_name(&self) -> &str {
        &self.meta.name
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
