//! Image to patch embedding module (`PatchEmbed`).

use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::prelude::*;

/// Splits 2D images into flat patch embeddings (`timm.layers.PatchEmbed`).
#[derive(Module, Debug)]
pub struct PatchEmbed {
    /// 2D convolution projection with kernel size and stride equal to patch size.
    pub proj: Conv2d,
    /// Optional normalization applied to the patch tokens.
    pub norm: Option<LayerNorm>,
    #[module(skip)]
    /// Nominal input image resolution `(height, width)`.
    pub img_size: (usize, usize),
    #[module(skip)]
    /// Size of each spatial patch `(patch_h, patch_w)`.
    pub patch_size: (usize, usize),
    #[module(skip)]
    /// Grid dimensions `(grid_h, grid_w)`.
    pub grid_size: (usize, usize),
    #[module(skip)]
    /// Total number of patches in the grid (`grid_h * grid_w`).
    pub num_patches: usize,
    #[module(skip)]
    /// Whether to dynamically pad images whose dimensions are not divisible by patch size.
    pub dynamic_img_pad: bool,
}

impl PatchEmbed {
    /// Creates a `PatchEmbed` layer.
    pub fn new(
        img_size: (usize, usize),
        patch_size: (usize, usize),
        in_chans: usize,
        embed_dim: usize,
        norm_layer: bool,
        bias: bool,
        device: &Device,
    ) -> Self {
        let grid_size = (img_size.0 / patch_size.0, img_size.1 / patch_size.1);
        let num_patches = grid_size.0 * grid_size.1;

        let proj = Conv2dConfig::new([in_chans, embed_dim], [patch_size.0, patch_size.1])
            .with_stride([patch_size.0, patch_size.1])
            .with_bias(bias)
            .init(device);

        let norm = norm_layer.then(|| {
            LayerNormConfig::new(embed_dim)
                .with_epsilon(1e-5)
                .init(device)
        });

        Self {
            proj,
            norm,
            img_size,
            patch_size,
            grid_size,
            num_patches,
            dynamic_img_pad: false,
        }
    }

    /// Enables dynamic padding for arbitrary input resolutions.
    pub fn with_dynamic_img_pad(mut self, dynamic_img_pad: bool) -> Self {
        self.dynamic_img_pad = dynamic_img_pad;
        self
    }

    /// Computes the patch grid size for a given input resolution.
    pub fn dynamic_feat_size(&self, img_size: (usize, usize)) -> (usize, usize) {
        let h = img_size.0.div_ceil(self.patch_size.0);
        let w = img_size.1.div_ceil(self.patch_size.1);
        (h, w)
    }

    /// Projects images `[B, in_chans, H, W]` to tokens `[B, N, embed_dim]`.
    pub fn forward(&self, x: Tensor<4>) -> Tensor<3> {
        let [b, _c, _h, _w] = x.dims();
        let x = self.proj.forward(x);
        let [_b, embed_dim, h_p, w_p] = x.dims();
        // Permute to [B, H_p, W_p, C] then flatten spatial dims to [B, N, C]
        let x = x.permute([0, 2, 3, 1]).reshape([b, h_p * w_p, embed_dim]);
        match &self.norm {
            Some(norm) => norm.forward(x),
            None => x,
        }
    }
}
