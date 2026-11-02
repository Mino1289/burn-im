//! Positional embeddings and Rotary Position Embeddings (RoPE).
//!
//! Provides:
//! - 2D bicubic resampling of absolute positional embeddings (`resample_abs_pos_embed`)
//!   with timm 1.0.29 constant-token-count resampling bugfix.
//! - Unified RoPE modules:
//!   - [`RotaryEmbeddingCat`]: Axial RoPE with frequency bands.
//!   - [`RotaryEmbeddingDinoV3`]: DINOv3 mixed RoPE with coordinate transforms.
//!   - [`RotaryEmbeddingMRope`]: Multimodal interleaved RoPE.
//!   - [`RotaryEmbeddingMixed`]: Depth-dependent learnable frequencies.
//!   - [`apply_rot_embed_cat`]: Application of rotary embeddings to query/key tensors.

use std::f64::consts::PI;

use burn::module::{Module, Param};
use burn::prelude::*;

/// Meshgrid coordinate ordering for rotary position embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GridIndexing {
    /// Matrix indexing (row-major / height first, `(i, j)`).
    #[default]
    Ij,
    /// Cartesian indexing (column-major / width first, `(x, y)`).
    Xy,
}

// ----------------------------------------------------------------------------
// 2D Bicubic Interpolation & Resampling
// ----------------------------------------------------------------------------

fn cubic_weight(d: f64, a: f64) -> f64 {
    let d = d.abs();
    if d <= 1.0 {
        ((a + 2.0) * d - (a + 3.0)) * d * d + 1.0
    } else if d < 2.0 {
        ((a * d - 5.0 * a) * d + 8.0 * a) * d - 4.0 * a
    } else {
        0.0
    }
}

/// 2D bicubic interpolation matching PyTorch's `F.interpolate(mode='bicubic', align_corners=False)`.
pub fn interpolate_bicubic_2d(
    input: &[f32],
    c: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    let a = -0.75f64;
    let mut out = vec![0.0f32; c * out_h * out_w];

    // Precalculate horizontal weights and indices
    let mut w_indices = Vec::with_capacity(out_w);
    let mut w_weights = Vec::with_capacity(out_w);
    for ow in 0..out_w {
        let u = (ow as f64 + 0.5) * (in_w as f64 / out_w as f64) - 0.5;
        let left = u.floor() as isize;
        let mut idxs = [0usize; 4];
        let mut weights = [0.0f64; 4];
        let mut total_w = 0.0f64;
        for i in 0..4 {
            let ix = left - 1 + i as isize;
            idxs[i] = ix.clamp(0, (in_w - 1) as isize) as usize;
            let d = u - ix as f64;
            let w = cubic_weight(d, a);
            weights[i] = w;
            total_w += w;
        }
        for w in &mut weights {
            *w /= total_w.max(1e-12);
        }
        w_indices.push(idxs);
        w_weights.push(weights);
    }

    // Precalculate vertical weights and indices
    let mut h_indices = Vec::with_capacity(out_h);
    let mut h_weights = Vec::with_capacity(out_h);
    for oh in 0..out_h {
        let v = (oh as f64 + 0.5) * (in_h as f64 / out_h as f64) - 0.5;
        let top = v.floor() as isize;
        let mut idxs = [0usize; 4];
        let mut weights = [0.0f64; 4];
        let mut total_w = 0.0f64;
        for i in 0..4 {
            let iy = top - 1 + i as isize;
            idxs[i] = iy.clamp(0, (in_h - 1) as isize) as usize;
            let d = v - iy as f64;
            let w = cubic_weight(d, a);
            weights[i] = w;
            total_w += w;
        }
        for w in &mut weights {
            *w /= total_w.max(1e-12);
        }
        h_indices.push(idxs);
        h_weights.push(weights);
    }

    for ch in 0..c {
        let in_ch_offset = ch * in_h * in_w;
        let out_ch_offset = ch * out_h * out_w;

        for oh in 0..out_h {
            let h_idx = &h_indices[oh];
            let h_w = &h_weights[oh];
            let out_row_offset = out_ch_offset + oh * out_w;

            for ow in 0..out_w {
                let w_idx = &w_indices[ow];
                let w_w = &w_weights[ow];

                let mut val = 0.0f64;
                for i in 0..4 {
                    let y = h_idx[i];
                    let yw = h_w[i];
                    let in_row_offset = in_ch_offset + y * in_w;
                    for j in 0..4 {
                        let x = w_idx[j];
                        let xw = w_w[j];
                        let pixel = input[in_row_offset + x] as f64;
                        val += pixel * yw * xw;
                    }
                }
                out[out_row_offset + ow] = val as f32;
            }
        }
    }

    out
}

/// Resamples absolute positional embeddings to a new grid size (`timm.layers.resample_abs_pos_embed`).
///
/// Handles prefix tokens (class token, register tokens) and preserves constant token count
/// resampling (timm 1.0.29 fix for `H_old * W_old == H_new * W_new` with `H_old != H_new`).
pub fn resample_abs_pos_embed(
    posemb: Tensor<3>,
    new_size: (usize, usize),
    old_size: Option<(usize, usize)>,
    num_prefix_tokens: usize,
) -> Tensor<3> {
    let [_batch, num_pos_tokens, embed_dim] = posemb.dims();
    let num_new_tokens = new_size.0 * new_size.1 + num_prefix_tokens;

    // Fast path: grid already matches exactly
    if num_new_tokens == num_pos_tokens {
        if let Some(old) = old_size {
            if old == new_size {
                return posemb;
            }
        } else if new_size.0 == new_size.1 {
            return posemb;
        }
    }

    let resolved_old_size = old_size.unwrap_or_else(|| {
        let spatial_tokens = num_pos_tokens.saturating_sub(num_prefix_tokens);
        let hw = (spatial_tokens as f64).sqrt().round() as usize;
        (hw, hw)
    });

    let (prefix, spatial) = if num_prefix_tokens > 0 {
        let prefix = posemb.clone().slice([0..1, 0..num_prefix_tokens]);
        let spatial = posemb.slice([0..1, num_prefix_tokens..num_pos_tokens]);
        (Some(prefix), spatial)
    } else {
        (None, posemb)
    };

    let device = spatial.device();
    // Reshape [1, N_old, C] -> [1, C, H_old, W_old]
    let (in_h, in_w) = resolved_old_size;
    let (out_h, out_w) = new_size;

    let spatial_data: Vec<f32> = spatial.into_data().as_slice::<f32>().unwrap().to_vec();

    // Permute from NLC layout to CHW
    let mut chw_in = vec![0.0f32; embed_dim * in_h * in_w];
    for hw in 0..(in_h * in_w) {
        for c in 0..embed_dim {
            chw_in[c * in_h * in_w + hw] = spatial_data[hw * embed_dim + c];
        }
    }

    let chw_out = interpolate_bicubic_2d(&chw_in, embed_dim, in_h, in_w, out_h, out_w);

    // Permute from CHW back to NLC [1, out_h * out_w, embed_dim]
    let mut nlc_out = vec![0.0f32; out_h * out_w * embed_dim];
    for hw in 0..(out_h * out_w) {
        for c in 0..embed_dim {
            nlc_out[hw * embed_dim + c] = chw_out[c * out_h * out_w + hw];
        }
    }

    let resampled = Tensor::from_data(
        burn::tensor::TensorData::new(nlc_out, [1, out_h * out_w, embed_dim]),
        &device,
    );

    if let Some(prefix) = prefix {
        Tensor::cat(vec![prefix, resampled], 1)
    } else {
        resampled
    }
}

// ----------------------------------------------------------------------------
// RoPE: Frequency Generation & Rotations
// ----------------------------------------------------------------------------

/// Applies half-split rotation: `[-x2, x1]`.
pub fn rope_rotate_half(x: Tensor<4>) -> Tensor<4> {
    let [b, h, n, head_dim] = x.dims();
    let half = head_dim / 2;
    let x1 = x.clone().slice([0..b, 0..h, 0..n, 0..half]);
    let x2 = x.slice([0..b, 0..h, 0..n, half..head_dim]);
    Tensor::cat(vec![x2.neg(), x1], 3)
}

/// Applies interleaved rotation: `[-x1, x0, -x3, x2, ...]`.
pub fn rope_rotate_interleaved(x: Tensor<4>) -> Tensor<4> {
    let [b, h, n, head_dim] = x.dims();
    let pairs = head_dim / 2;
    let flat_batch = b * h * n;
    let x_flat = x.reshape([flat_batch, pairs, 2]);
    let x0 = x_flat.clone().slice([0..flat_batch, 0..pairs, 0..1]);
    let x1 = x_flat.slice([0..flat_batch, 0..pairs, 1..2]);
    let rotated = Tensor::cat(vec![x1.neg(), x0], 2);
    rotated.reshape([b, h, n, head_dim])
}

/// Applies rotary position embedding to `x` (`timm.layers.apply_rot_embed_cat`).
///
/// `x` is `[B, num_heads, N, head_dim]`.
/// `emb` is `[N, 2 * head_dim]` (concatenated `sin` and `cos`).
pub fn apply_rot_embed_cat(x: Tensor<4>, emb: Tensor<2>, rotate_half: bool) -> Tensor<4> {
    let [_batch, _num_heads, n, head_dim] = x.dims();
    let orig_dtype = x.dtype();

    // Split emb into sin and cos [N, head_dim]
    let sin = emb.clone().slice([0..n, 0..head_dim]);
    let cos = emb.slice([0..n, head_dim..head_dim * 2]);

    // Reshape to broadcast: [1, 1, N, head_dim]
    let sin = sin.reshape([1, 1, n, head_dim]);
    let cos = cos.reshape([1, 1, n, head_dim]);

    let sin = if sin.dtype() != orig_dtype {
        sin.cast(orig_dtype)
    } else {
        sin
    };
    let cos = if cos.dtype() != orig_dtype {
        cos.cast(orig_dtype)
    } else {
        cos
    };

    if rotate_half {
        x.clone() * cos + rope_rotate_half(x) * sin
    } else {
        x.clone() * cos + rope_rotate_interleaved(x) * sin
    }
}

// ----------------------------------------------------------------------------
// RoPE Modules
// ----------------------------------------------------------------------------

/// Axial Rotary Position Embedding with concatenated sin and cos (`timm.layers.RotaryEmbeddingCat`).
#[derive(Module, Debug)]
pub struct RotaryEmbeddingCat {
    #[module(skip)]
    /// Embedding dimension per head.
    pub dim: usize,
    #[module(skip)]
    /// Maximum resolution.
    pub max_res: usize,
    #[module(skip)]
    /// Temperature for frequency calculation.
    pub temperature: f64,
    #[module(skip)]
    /// Whether bands are in pixel space or inv-freq space.
    pub in_pixels: bool,
    #[module(skip)]
    /// Whether frequency bands are linearly spaced.
    pub linear_bands: bool,
    #[module(skip)]
    /// Coordinate indexing (`Ij` or `Xy`).
    pub grid_indexing: GridIndexing,
    #[module(skip)]
    /// Emits half layout `[f0..fn, f0..fn]` if true, else interleaved.
    pub rotate_half: bool,
    /// Frequency bands buffer in `f32`.
    pub bands: Tensor<1>,
}

impl RotaryEmbeddingCat {
    /// Creates a `RotaryEmbeddingCat` module.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        max_res: usize,
        temperature: f64,
        in_pixels: bool,
        linear_bands: bool,
        grid_indexing: GridIndexing,
        rotate_half: bool,
        device: &Device,
    ) -> Self {
        let num_bands = dim / 4;
        let mut bands_vec = Vec::with_capacity(num_bands);
        if in_pixels {
            let max_freq = max_res as f64;
            for i in 0..num_bands {
                let val = if linear_bands {
                    1.0 + (max_freq / 2.0 - 1.0) * (i as f64) / (num_bands - 1).max(1) as f64
                } else {
                    let log_max = (max_freq).log2() - 1.0;
                    2.0f64.powf(log_max * (i as f64) / (num_bands - 1).max(1) as f64)
                };
                bands_vec.push((val * PI) as f32);
            }
        } else {
            for i in 0..num_bands {
                let exponent = (2 * i) as f64 / dim as f64;
                let val = 1.0 / temperature.powf(exponent);
                bands_vec.push(val as f32);
            }
        }

        let bands = Tensor::from_data(
            burn::tensor::TensorData::new(bands_vec, [num_bands]),
            device,
        );

        Self {
            dim,
            max_res,
            temperature,
            in_pixels,
            linear_bands,
            grid_indexing,
            rotate_half,
            bands,
        }
    }

    /// Computes concatenated sin/cos embedding `[H*W, 2*dim]`.
    pub fn get_embed(&self, feat_shape: (usize, usize)) -> Tensor<2> {
        let (mut h, mut w) = feat_shape;
        if self.grid_indexing == GridIndexing::Xy {
            std::mem::swap(&mut h, &mut w);
        }

        let device = self.bands.device();
        let num_bands = self.dim / 4;
        let bands_data = self
            .bands
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();

        let num_pos = h * w;
        let mut sin_vals = vec![0.0f32; num_pos * (num_bands * 2)];
        let mut cos_vals = vec![0.0f32; num_pos * (num_bands * 2)];

        for i in 0..h {
            let t_h = if self.in_pixels {
                -1.0 + 2.0 * (i as f64) / (h - 1).max(1) as f64
            } else {
                i as f64
            };
            for j in 0..w {
                let t_w = if self.in_pixels {
                    -1.0 + 2.0 * (j as f64) / (w - 1).max(1) as f64
                } else {
                    j as f64
                };
                let (coord0, coord1) = match self.grid_indexing {
                    GridIndexing::Ij => (t_h, t_w),
                    GridIndexing::Xy => (t_w, t_h),
                };

                let pos_idx = i * w + j;
                for b in 0..num_bands {
                    let band = bands_data[b] as f64;
                    let angle0 = coord0 * band;
                    let angle1 = coord1 * band;

                    sin_vals[pos_idx * (num_bands * 2) + b] = angle0.sin() as f32;
                    cos_vals[pos_idx * (num_bands * 2) + b] = angle0.cos() as f32;
                    sin_vals[pos_idx * (num_bands * 2) + num_bands + b] = angle1.sin() as f32;
                    cos_vals[pos_idx * (num_bands * 2) + num_bands + b] = angle1.cos() as f32;
                }
            }
        }

        // Layout formatting: rotate_half tiles sin, interleaved duplicates
        let mut final_sin = vec![0.0f32; num_pos * self.dim];
        let mut final_cos = vec![0.0f32; num_pos * self.dim];

        if self.rotate_half {
            for p in 0..num_pos {
                for k in 0..(num_bands * 2) {
                    final_sin[p * self.dim + k] = sin_vals[p * (num_bands * 2) + k];
                    final_cos[p * self.dim + k] = cos_vals[p * (num_bands * 2) + k];
                    final_sin[p * self.dim + (num_bands * 2) + k] =
                        sin_vals[p * (num_bands * 2) + k];
                    final_cos[p * self.dim + (num_bands * 2) + k] =
                        cos_vals[p * (num_bands * 2) + k];
                }
            }
        } else {
            for p in 0..num_pos {
                for k in 0..(num_bands * 2) {
                    final_sin[p * self.dim + 2 * k] = sin_vals[p * (num_bands * 2) + k];
                    final_sin[p * self.dim + 2 * k + 1] = sin_vals[p * (num_bands * 2) + k];
                    final_cos[p * self.dim + 2 * k] = cos_vals[p * (num_bands * 2) + k];
                    final_cos[p * self.dim + 2 * k + 1] = cos_vals[p * (num_bands * 2) + k];
                }
            }
        }

        let mut cat_emb = vec![0.0f32; num_pos * (self.dim * 2)];
        for p in 0..num_pos {
            for k in 0..self.dim {
                cat_emb[p * 2 * self.dim + k] = final_sin[p * self.dim + k];
                cat_emb[p * 2 * self.dim + self.dim + k] = final_cos[p * self.dim + k];
            }
        }

        Tensor::from_data(
            burn::tensor::TensorData::new(cat_emb, [num_pos, self.dim * 2]),
            &device,
        )
    }
}

/// DINOv3 Mixed Rotary Position Embedding (`timm.layers.RotaryEmbeddingDinoV3`).
#[derive(Module, Debug)]
pub struct RotaryEmbeddingDinoV3 {
    #[module(skip)]
    /// Embedding dimension per head.
    pub dim: usize,
    #[module(skip)]
    /// Base temperature.
    pub temperature: f64,
    #[module(skip)]
    /// Grid indexing.
    pub grid_indexing: GridIndexing,
    #[module(skip)]
    /// Coordinate normalization mode ("separate", "min", "max").
    pub normalize_coords: String,
    #[module(skip)]
    /// Rotation format.
    pub rotate_half: bool,
    /// Periods buffer in `f32`.
    pub periods: Tensor<1>,
}

impl RotaryEmbeddingDinoV3 {
    /// Creates a `RotaryEmbeddingDinoV3` module.
    pub fn new(
        dim: usize,
        temperature: f64,
        grid_indexing: GridIndexing,
        normalize_coords: String,
        rotate_half: bool,
        device: &Device,
    ) -> Self {
        let num_periods = dim / 4;
        let mut periods_vec = Vec::with_capacity(num_periods);
        for i in 0..num_periods {
            let exponent = (2.0 * i as f64) / (dim as f64 / 2.0);
            periods_vec.push(temperature.powf(exponent) as f32);
        }
        let periods = Tensor::from_data(
            burn::tensor::TensorData::new(periods_vec, [num_periods]),
            device,
        );

        Self {
            dim,
            temperature,
            grid_indexing,
            normalize_coords,
            rotate_half,
            periods,
        }
    }

    /// Computes concatenated sin/cos embedding `[H*W, 2*dim]`.
    pub fn get_embed(&self, feat_shape: (usize, usize)) -> Tensor<2> {
        let (h, w) = feat_shape;
        let num_pos = h * w;
        let num_periods = self.dim / 4;
        let periods_data = self
            .periods
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let device = self.periods.device();

        let (denom_h, denom_w) = match self.normalize_coords.as_str() {
            "min" => {
                let m = (h.min(w)) as f64;
                (m, m)
            }
            "max" => {
                let m = (h.max(w)) as f64;
                (m, m)
            }
            _ => (h as f64, w as f64),
        };

        let mut cat_emb = vec![0.0f32; num_pos * (self.dim * 2)];

        for i in 0..h {
            let norm_h = (i as f64 + 0.5) / denom_h;
            let coord_h = 2.0 * norm_h - 1.0;
            for j in 0..w {
                let norm_w = (j as f64 + 0.5) / denom_w;
                let coord_w = 2.0 * norm_w - 1.0;

                let (ch, cw) = match self.grid_indexing {
                    GridIndexing::Ij => (coord_h, coord_w),
                    GridIndexing::Xy => (coord_w, coord_h),
                };

                let pos_idx = i * w + j;
                for p in 0..num_periods {
                    let period = periods_data[p] as f64;
                    let angle_h = 2.0 * PI * ch / period;
                    let angle_w = 2.0 * PI * cw / period;

                    let (sin_h, cos_h) = (angle_h.sin() as f32, angle_h.cos() as f32);
                    let (sin_w, cos_w) = (angle_w.sin() as f32, angle_w.cos() as f32);

                    if self.rotate_half {
                        cat_emb[pos_idx * 2 * self.dim + p] = sin_h;
                        cat_emb[pos_idx * 2 * self.dim + num_periods + p] = sin_w;
                        cat_emb[pos_idx * 2 * self.dim + 2 * num_periods + p] = sin_h;
                        cat_emb[pos_idx * 2 * self.dim + 3 * num_periods + p] = sin_w;

                        cat_emb[pos_idx * 2 * self.dim + self.dim + p] = cos_h;
                        cat_emb[pos_idx * 2 * self.dim + self.dim + num_periods + p] = cos_w;
                        cat_emb[pos_idx * 2 * self.dim + self.dim + 2 * num_periods + p] = cos_h;
                        cat_emb[pos_idx * 2 * self.dim + self.dim + 3 * num_periods + p] = cos_w;
                    } else {
                        cat_emb[pos_idx * 2 * self.dim + 2 * p] = sin_h;
                        cat_emb[pos_idx * 2 * self.dim + 2 * p + 1] = sin_h;
                        cat_emb[pos_idx * 2 * self.dim + 2 * num_periods + 2 * p] = sin_w;
                        cat_emb[pos_idx * 2 * self.dim + 2 * num_periods + 2 * p + 1] = sin_w;

                        cat_emb[pos_idx * 2 * self.dim + self.dim + 2 * p] = cos_h;
                        cat_emb[pos_idx * 2 * self.dim + self.dim + 2 * p + 1] = cos_h;
                        cat_emb[pos_idx * 2 * self.dim + self.dim + 2 * num_periods + 2 * p] =
                            cos_w;
                        cat_emb[pos_idx * 2 * self.dim + self.dim + 2 * num_periods + 2 * p + 1] =
                            cos_w;
                    }
                }
            }
        }

        Tensor::from_data(
            burn::tensor::TensorData::new(cat_emb, [num_pos, self.dim * 2]),
            &device,
        )
    }
}

/// Multimodal Interleaved RoPE (`timm.layers.RotaryEmbeddingMRope`).
#[derive(Module, Debug)]
pub struct RotaryEmbeddingMRope {
    #[module(skip)]
    /// Embedding dimension per head.
    pub dim: usize,
    #[module(skip)]
    /// Sections for temporal, height, width.
    pub mrope_section: (usize, usize, usize),
    #[module(skip)]
    /// Base temperature.
    pub temperature: f64,
    #[module(skip)]
    /// Grid indexing.
    pub grid_indexing: GridIndexing,
    /// Inverse frequencies buffer in `f32`.
    pub inv_freq: Tensor<1>,
}

impl RotaryEmbeddingMRope {
    /// Creates a `RotaryEmbeddingMRope` module.
    pub fn new(
        dim: usize,
        mrope_section: (usize, usize, usize),
        temperature: f64,
        grid_indexing: GridIndexing,
        device: &Device,
    ) -> Self {
        assert_eq!(dim % 2, 0, "dim must be even");
        let half = dim / 2;
        let mut inv_freq_vec = Vec::with_capacity(half);
        for i in 0..half {
            let val = 1.0 / temperature.powf((2 * i) as f64 / dim as f64);
            inv_freq_vec.push(val as f32);
        }
        let inv_freq =
            Tensor::from_data(burn::tensor::TensorData::new(inv_freq_vec, [half]), device);

        Self {
            dim,
            mrope_section,
            temperature,
            grid_indexing,
            inv_freq,
        }
    }

    /// Computes multimodal interleaved RoPE embedding `[H*W, 2*dim]`.
    pub fn get_embed(&self, feat_shape: (usize, usize)) -> Tensor<2> {
        let (h, w) = feat_shape;
        let num_pos = h * w;
        let half = self.dim / 2;
        let inv_freq_data = self
            .inv_freq
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let device = self.inv_freq.device();

        let (_sec_t, sec_h, sec_w) = self.mrope_section;

        let mut cat_emb = vec![0.0f32; num_pos * (self.dim * 2)];

        for i in 0..h {
            let y = i as f64;
            for j in 0..w {
                let x = j as f64;
                let pos_idx = i * w + j;

                for c in 0..half {
                    let freq = inv_freq_data[c] as f64;
                    let p = if c > 0 && (c - 1) % 3 == 0 && c < sec_h * 3 {
                        y
                    } else if c > 1 && (c - 2) % 3 == 0 && c < sec_w * 3 {
                        x
                    } else {
                        0.0
                    };
                    let angle = p * freq;
                    let (sin_val, cos_val) = (angle.sin() as f32, angle.cos() as f32);

                    cat_emb[pos_idx * 2 * self.dim + c] = sin_val;
                    cat_emb[pos_idx * 2 * self.dim + half + c] = sin_val;
                    cat_emb[pos_idx * 2 * self.dim + self.dim + c] = cos_val;
                    cat_emb[pos_idx * 2 * self.dim + self.dim + half + c] = cos_val;
                }
            }
        }

        Tensor::from_data(
            burn::tensor::TensorData::new(cat_emb, [num_pos, self.dim * 2]),
            &device,
        )
    }
}

/// Rotary position embedding with depth-dependent learnable frequencies (`timm.layers.RotaryEmbeddingMixed`).
#[derive(Module, Debug)]
pub struct RotaryEmbeddingMixed {
    /// Learnable frequency parameters with shape `[2, depth, num_heads, head_dim / 2]`.
    pub freqs: Param<Tensor<4>>,
    #[module(skip)]
    /// Total embedding dimension across all heads.
    pub dim: usize,
    #[module(skip)]
    /// Number of transformer blocks (depth).
    pub depth: usize,
    #[module(skip)]
    /// Number of attention heads.
    pub num_heads: usize,
    #[module(skip)]
    /// Coordinate indexing (`Ij` or `Xy`).
    pub grid_indexing: GridIndexing,
}

impl RotaryEmbeddingMixed {
    /// Creates a `RotaryEmbeddingMixed` layer with learnable frequencies.
    pub fn new(
        dim: usize,
        depth: usize,
        num_heads: usize,
        temperature: f64,
        grid_indexing: GridIndexing,
        device: &Device,
    ) -> Self {
        assert_eq!(dim % num_heads, 0, "dim must be divisible by num_heads");
        let head_dim = dim / num_heads;
        assert_eq!(head_dim % 4, 0, "head_dim must be divisible by 4");
        let num_bands = head_dim / 4;

        let mut mag_vec = Vec::with_capacity(num_bands);
        for i in 0..num_bands {
            let exponent = (4 * i) as f64 / head_dim as f64;
            let val = 1.0 / temperature.powf(exponent);
            mag_vec.push(val as f32);
        }

        // Initialize frequencies [2, depth, num_heads, head_dim / 2]
        let half_dim = head_dim / 2;
        let total_elems = 2 * depth * num_heads * half_dim;
        let mut freqs_vec = vec![0.0f32; total_elems];

        for d in 0..depth {
            for h in 0..num_heads {
                let base_idx_0 = (d * num_heads + h) * half_dim;
                let base_idx_1 = (depth * num_heads + d * num_heads + h) * half_dim;

                for b in 0..num_bands {
                    let m = mag_vec[b];
                    // fx = [mag, 0], fy = [0, mag]
                    freqs_vec[base_idx_0 + b] = m;
                    freqs_vec[base_idx_0 + num_bands + b] = 0.0;
                    freqs_vec[base_idx_1 + b] = 0.0;
                    freqs_vec[base_idx_1 + num_bands + b] = m;
                }
            }
        }

        let freqs = Param::from_data(
            burn::tensor::TensorData::new(freqs_vec, [2, depth, num_heads, half_dim]),
            device,
        );

        Self {
            freqs,
            dim,
            depth,
            num_heads,
            grid_indexing,
        }
    }

    /// Computes rotary embeddings for a specific transformer block `depth_idx` and feature shape `(H, W)`.
    ///
    /// Returns a tensor of shape `[num_heads, H*W, head_dim * 2]`.
    pub fn get_embed_for_block(&self, depth_idx: usize, feat_shape: (usize, usize)) -> Tensor<3> {
        let (mut h, mut w) = feat_shape;
        if self.grid_indexing == GridIndexing::Xy {
            std::mem::swap(&mut h, &mut w);
        }

        let num_pos = h * w;
        let head_dim = self.dim / self.num_heads;
        let half_dim = head_dim / 2;
        let device = self.freqs.device();

        let freqs_data = self
            .freqs
            .val()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();

        let mut out_vec = vec![0.0f32; self.num_heads * num_pos * head_dim * 2];

        for head in 0..self.num_heads {
            let offset_x = (depth_idx * self.num_heads + head) * half_dim;
            let offset_y =
                (self.depth * self.num_heads + depth_idx * self.num_heads + head) * half_dim;

            for i in 0..h {
                let t_x = i as f32;
                for j in 0..w {
                    let t_y = j as f32;
                    let (coord_x, coord_y) = match self.grid_indexing {
                        GridIndexing::Ij => (t_x, t_y),
                        GridIndexing::Xy => (t_y, t_x),
                    };

                    let pos_idx = i * w + j;
                    let base_out = (head * num_pos + pos_idx) * head_dim * 2;

                    for c in 0..half_dim {
                        let fx = freqs_data[offset_x + c];
                        let fy = freqs_data[offset_y + c];
                        let angle = (coord_x * fx + coord_y * fy) as f64;

                        let sin_val = angle.sin() as f32;
                        let cos_val = angle.cos() as f32;

                        // Repeat interleave
                        out_vec[base_out + 2 * c] = sin_val;
                        out_vec[base_out + 2 * c + 1] = sin_val;
                        out_vec[base_out + head_dim + 2 * c] = cos_val;
                        out_vec[base_out + head_dim + 2 * c + 1] = cos_val;
                    }
                }
            }
        }

        Tensor::from_data(
            burn::tensor::TensorData::new(out_vec, [self.num_heads, num_pos, head_dim * 2]),
            &device,
        )
    }
}
