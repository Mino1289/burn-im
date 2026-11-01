//! `ConvNeXt` and `ConvNeXtV2` backbones.
//!
//! Reproduces `timm`'s `ConvNeXt` implementation (see `timm/models/convnext.py`):
//! - Patch-based stem with 4x4 convolution (stride 4) and channel-wise layer normalization.
//! - 4 stages of depthwise 7x7 convolutions, channel-wise layer norm, inverted MLP with GELU,
//!   and residual connections.
//! - LayerScale in ConvNeXt V1 vs. Global Response Normalization (GRN) in ConvNeXt V2.
//! - Multi-scale intermediate features (`stages.0` through `stages.3` at reductions `[4, 8, 16, 32]`).
//! - Dilation support via `output_stride` (OS 16 and 8).
//! - Truncated construction via `features_only` and `out_indices`.

use burn::module::{Param, ParamId};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::prelude::*;
use burn::tensor::activation::gelu;

use crate::error::{Error, Result};
use crate::hub::config::PretrainedCfg;
use crate::layers::drop::{DropPath, drop_path_schedule};
use crate::layers::head::NormMlpClassifierHead;
use crate::layers::norm::{GlobalResponseNorm, LayerNorm2d};
use crate::layers::pool::PoolMode;
use crate::models::ModelMeta;
use crate::traits::{FeatureFormat, FeatureInfo, IntermediatesOptions, VisionModel};

/// Configuration for [`ConvNeXt`].
#[derive(Debug, Clone)]
pub struct ConvNeXtConfig {
    /// Number of blocks in each of the 4 stages.
    pub depths: [usize; 4],
    /// Channel dimensions of each of the 4 stages.
    pub dims: [usize; 4],
    /// Number of target classes. `0` builds a headless model.
    pub num_classes: usize,
    /// Number of input channels.
    pub in_chans: usize,
    /// Classifier dropout probability.
    pub drop_rate: f64,
    /// Maximum stochastic depth rate across the network.
    pub drop_path_rate: f64,
    /// Initial value for LayerScale (V1). Set to `None` for ConvNeXt V2.
    pub ls_init_value: Option<f64>,
    /// Whether to insert Global Response Normalization (GRN) into the MLP (V2).
    pub use_grn: bool,
    /// Spatial reduction factor of the deepest stage: `32` (default), `16`, or `8`.
    pub output_stride: Option<usize>,
    /// Build a truncated feature extractor stopping at the deepest requested stage.
    pub features_only: bool,
    /// Feature levels retained when [`Self::features_only`] is set.
    pub out_indices: Option<Vec<usize>>,
}

impl ConvNeXtConfig {
    /// Creates a configuration with standard defaults (1000 classes, 3 channels, LayerScale 1e-6).
    pub fn new(depths: [usize; 4], dims: [usize; 4]) -> Self {
        Self {
            depths,
            dims,
            num_classes: 1000,
            in_chans: 3,
            drop_rate: 0.0,
            drop_path_rate: 0.0,
            ls_init_value: Some(1e-6),
            use_grn: false,
            output_stride: None,
            features_only: false,
            out_indices: None,
        }
    }

    /// Sets the number of target classes. `0` removes the classifier head.
    pub fn with_num_classes(mut self, num_classes: usize) -> Self {
        self.num_classes = num_classes;
        self
    }

    /// Sets the number of input channels.
    pub fn with_in_chans(mut self, in_chans: usize) -> Self {
        self.in_chans = in_chans;
        self
    }

    /// Sets the classifier dropout rate.
    pub fn with_drop_rate(mut self, drop_rate: f64) -> Self {
        self.drop_rate = drop_rate;
        self
    }

    /// Sets the maximum stochastic depth rate.
    pub fn with_drop_path_rate(mut self, drop_path_rate: f64) -> Self {
        self.drop_path_rate = drop_path_rate;
        self
    }

    /// Sets the LayerScale initial value (`None` disables LayerScale).
    pub fn with_ls_init_value(mut self, ls_init_value: Option<f64>) -> Self {
        self.ls_init_value = ls_init_value;
        self
    }

    /// Enables or disables Global Response Normalization (GRN) in the MLPs.
    pub fn with_grn(mut self, use_grn: bool) -> Self {
        self.use_grn = use_grn;
        self
    }

    /// Sets the output stride (`32`, `16`, or `8`).
    pub fn with_output_stride(mut self, output_stride: usize) -> Self {
        self.output_stride = Some(output_stride);
        self
    }

    /// Builds a truncated feature extractor retaining only the stages in `out_indices`.
    pub fn with_features_only(mut self, out_indices: impl IntoIterator<Item = usize>) -> Self {
        self.features_only = true;
        self.num_classes = 0;
        self.out_indices = Some(out_indices.into_iter().collect());
        self
    }

    /// Validates the configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if self.in_chans == 0 {
            return Err(Error::InvalidConfiguration(
                "in_chans must be greater than 0".to_string(),
            ));
        }
        if !(0.0..1.0).contains(&self.drop_rate) {
            return Err(Error::InvalidConfiguration(format!(
                "drop_rate must be in [0, 1), got {}",
                self.drop_rate
            )));
        }
        if !(0.0..1.0).contains(&self.drop_path_rate) {
            return Err(Error::InvalidConfiguration(format!(
                "drop_path_rate must be in [0, 1), got {}",
                self.drop_path_rate
            )));
        }
        if let Some(stride) = self.output_stride {
            if stride != 32 && stride != 16 && stride != 8 {
                return Err(Error::InvalidConfiguration(format!(
                    "output_stride must be 32, 16, or 8, got {stride}"
                )));
            }
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
                    if idx >= 4 {
                        return Err(Error::InvalidConfiguration(format!(
                            "out_index {idx} out of bounds (ConvNeXt exposes 4 stages: 0..=3)"
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// Presets: ConvNeXt V1
// ----------------------------------------------------------------------------

/// `convnext_atto`: depths [2, 2, 6, 2], dims [40, 80, 160, 320].
pub fn convnext_atto(num_classes: usize) -> ConvNeXtConfig {
    ConvNeXtConfig::new([2, 2, 6, 2], [40, 80, 160, 320]).with_num_classes(num_classes)
}

/// `convnext_femto`: depths [2, 2, 6, 2], dims [48, 96, 192, 384].
pub fn convnext_femto(num_classes: usize) -> ConvNeXtConfig {
    ConvNeXtConfig::new([2, 2, 6, 2], [48, 96, 192, 384]).with_num_classes(num_classes)
}

/// `convnext_pico`: depths [2, 2, 6, 2], dims [64, 128, 256, 512].
pub fn convnext_pico(num_classes: usize) -> ConvNeXtConfig {
    ConvNeXtConfig::new([2, 2, 6, 2], [64, 128, 256, 512]).with_num_classes(num_classes)
}

/// `convnext_nano`: depths [2, 2, 8, 2], dims [80, 160, 320, 640].
pub fn convnext_nano(num_classes: usize) -> ConvNeXtConfig {
    ConvNeXtConfig::new([2, 2, 8, 2], [80, 160, 320, 640]).with_num_classes(num_classes)
}

/// `convnext_tiny`: depths [3, 3, 9, 3], dims [96, 192, 384, 768].
pub fn convnext_tiny(num_classes: usize) -> ConvNeXtConfig {
    ConvNeXtConfig::new([3, 3, 9, 3], [96, 192, 384, 768]).with_num_classes(num_classes)
}

/// `convnext_small`: depths [3, 3, 27, 3], dims [96, 192, 384, 768].
pub fn convnext_small(num_classes: usize) -> ConvNeXtConfig {
    ConvNeXtConfig::new([3, 3, 27, 3], [96, 192, 384, 768]).with_num_classes(num_classes)
}

/// `convnext_base`: depths [3, 3, 27, 3], dims [128, 256, 512, 1024].
pub fn convnext_base(num_classes: usize) -> ConvNeXtConfig {
    ConvNeXtConfig::new([3, 3, 27, 3], [128, 256, 512, 1024]).with_num_classes(num_classes)
}

/// `convnext_large`: depths [3, 3, 27, 3], dims [192, 384, 768, 1536].
pub fn convnext_large(num_classes: usize) -> ConvNeXtConfig {
    ConvNeXtConfig::new([3, 3, 27, 3], [192, 384, 768, 1536]).with_num_classes(num_classes)
}

/// `convnext_xlarge`: depths [3, 3, 27, 3], dims [256, 512, 1024, 2048].
pub fn convnext_xlarge(num_classes: usize) -> ConvNeXtConfig {
    ConvNeXtConfig::new([3, 3, 27, 3], [256, 512, 1024, 2048]).with_num_classes(num_classes)
}

// ----------------------------------------------------------------------------
// Presets: ConvNeXt V2
// ----------------------------------------------------------------------------

/// `convnextv2_atto`: depths [2, 2, 6, 2], dims [40, 80, 160, 320] with GRN and no LayerScale.
pub fn convnextv2_atto(num_classes: usize) -> ConvNeXtConfig {
    convnext_atto(num_classes).with_grn(true).with_ls_init_value(None)
}

/// `convnextv2_femto`: depths [2, 2, 6, 2], dims [48, 96, 192, 384] with GRN and no LayerScale.
pub fn convnextv2_femto(num_classes: usize) -> ConvNeXtConfig {
    convnext_femto(num_classes).with_grn(true).with_ls_init_value(None)
}

/// `convnextv2_pico`: depths [2, 2, 6, 2], dims [64, 128, 256, 512] with GRN and no LayerScale.
pub fn convnextv2_pico(num_classes: usize) -> ConvNeXtConfig {
    convnext_pico(num_classes).with_grn(true).with_ls_init_value(None)
}

/// `convnextv2_nano`: depths [2, 2, 8, 2], dims [80, 160, 320, 640] with GRN and no LayerScale.
pub fn convnextv2_nano(num_classes: usize) -> ConvNeXtConfig {
    convnext_nano(num_classes).with_grn(true).with_ls_init_value(None)
}

/// `convnextv2_tiny`: depths [3, 3, 9, 3], dims [96, 192, 384, 768] with GRN and no LayerScale.
pub fn convnextv2_tiny(num_classes: usize) -> ConvNeXtConfig {
    convnext_tiny(num_classes).with_grn(true).with_ls_init_value(None)
}

/// `convnextv2_small`: depths [3, 3, 27, 3], dims [96, 192, 384, 768] with GRN and no LayerScale.
pub fn convnextv2_small(num_classes: usize) -> ConvNeXtConfig {
    convnext_small(num_classes).with_grn(true).with_ls_init_value(None)
}

/// `convnextv2_base`: depths [3, 3, 27, 3], dims [128, 256, 512, 1024] with GRN and no LayerScale.
pub fn convnextv2_base(num_classes: usize) -> ConvNeXtConfig {
    convnext_base(num_classes).with_grn(true).with_ls_init_value(None)
}

/// `convnextv2_large`: depths [3, 3, 27, 3], dims [192, 384, 768, 1536] with GRN and no LayerScale.
pub fn convnextv2_large(num_classes: usize) -> ConvNeXtConfig {
    convnext_large(num_classes).with_grn(true).with_ls_init_value(None)
}

/// `convnextv2_huge`: depths [3, 3, 27, 3], dims [352, 704, 1408, 2816] with GRN and no LayerScale.
pub fn convnextv2_huge(num_classes: usize) -> ConvNeXtConfig {
    ConvNeXtConfig::new([3, 3, 27, 3], [352, 704, 1408, 2816])
        .with_num_classes(num_classes)
        .with_grn(true)
        .with_ls_init_value(None)
}

// ----------------------------------------------------------------------------
// Architectural Modules
// ----------------------------------------------------------------------------

/// Stem of ConvNeXt: 4x4 convolution (stride 4) followed by `LayerNorm2d`.
#[derive(Module, Debug)]
pub struct ConvNeXtStem {
    /// Convolution mapping input channels to stage 0 dimensions.
    pub conv: Conv2d,
    /// Normalization across channels.
    pub norm: LayerNorm2d,
}

impl ConvNeXtStem {
    /// Creates a stem.
    pub fn new(in_chans: usize, out_chans: usize, device: &Device) -> Self {
        let conv = Conv2dConfig::new([in_chans, out_chans], [4, 4])
            .with_stride([4, 4])
            .with_bias(true)
            .init(device);
        let norm = LayerNorm2d::new(out_chans, 1e-6, device);
        Self { conv, norm }
    }

    /// Applies the stem: `[B, in_chans, H, W] -> [B, out_chans, H/4, W/4]`.
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        self.norm.forward(self.conv.forward(x))
    }
}

/// Downsampling layer between stages: `LayerNorm2d` followed by 2x2 convolution.
#[derive(Module, Debug)]
pub struct ConvNeXtDownsample {
    /// Normalization on the input feature maps before strided convolution.
    pub norm: LayerNorm2d,
    /// 2x2 convolution performing the spatial reduction and channel expansion.
    pub conv: Conv2d,
}

impl ConvNeXtDownsample {
    /// Creates a downsampling layer.
    pub fn new(
        in_chans: usize,
        out_chans: usize,
        stride: usize,
        dilation: (usize, usize),
        device: &Device,
    ) -> Self {
        let norm = LayerNorm2d::new(in_chans, 1e-6, device);
        let conv = if stride > 1 {
            // Standard downsample: 2x2 conv, stride 2, valid padding (0).
            Conv2dConfig::new([in_chans, out_chans], [2, 2])
                .with_stride([stride, stride])
                .with_bias(true)
                .init(device)
        } else if dilation.0 == 1 {
            // Dilated stage with entry dilation 1 (e.g. stage 3 in OS 16, or stage 2 in OS 8):
            // Conv2dSame with kernel 2, stride 1: format is (top, left, bottom, right), so (0, 0, 1, 1).
            Conv2dConfig::new([in_chans, out_chans], [2, 2])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(0, 0, 1, 1))
                .with_bias(true)
                .init(device)
        } else {
            // Dilated stage with entry dilation 2 (stage 3 in OS 8):
            // Conv2d with kernel 2, stride 1, dilation 2, symmetric padding 1.
            Conv2dConfig::new([in_chans, out_chans], [2, 2])
                .with_stride([1, 1])
                .with_dilation([dilation.0, dilation.0])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(true)
                .init(device)
        };
        Self { norm, conv }
    }

    /// Applies downsampling: `[B, in_chans, H, W] -> [B, out_chans, H', W']`.
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        self.conv.forward(self.norm.forward(x))
    }
}

/// Inverted MLP block inside [`ConvNeXtBlock`].
#[derive(Module, Debug)]
pub struct ConvNeXtMlp {
    /// Projection from `dim` to `4 * dim`.
    pub fc1: Linear,
    /// Optional Global Response Normalization (ConvNeXt-V2).
    pub grn: Option<GlobalResponseNorm>,
    /// Projection from `4 * dim` back to `dim`.
    pub fc2: Linear,
}

impl ConvNeXtMlp {
    /// Creates an inverted MLP.
    pub fn new(dim: usize, use_grn: bool, device: &Device) -> Self {
        let hidden_dim = 4 * dim;
        let fc1 = LinearConfig::new(dim, hidden_dim).init(device);
        let grn = use_grn.then(|| GlobalResponseNorm::new(hidden_dim, device));
        let fc2 = LinearConfig::new(hidden_dim, dim).init(device);
        Self { fc1, grn, fc2 }
    }

    /// Applies the MLP to a channels-last tensor `[B, H, W, C]`.
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.fc1.forward(x);
        let x = gelu(x);
        let x = match &self.grn {
            Some(grn) => grn.forward_channels_last(x),
            None => x,
        };
        self.fc2.forward(x)
    }
}

/// ConvNeXt residual block.
#[derive(Module, Debug)]
pub struct ConvNeXtBlock {
    /// 7x7 depthwise convolution.
    pub conv_dw: Conv2d,
    /// Normalization over channels.
    pub norm: LayerNorm,
    /// Inverted MLP.
    pub mlp: ConvNeXtMlp,
    /// Optional LayerScale scaling parameter (V1).
    pub gamma: Option<Param<Tensor<1>>>,
    /// Stochastic depth.
    pub drop_path: DropPath,
}

impl ConvNeXtBlock {
    /// Creates a ConvNeXt block.
    pub fn new(
        dim: usize,
        dilation: usize,
        drop_path_rate: f64,
        ls_init_value: Option<f64>,
        use_grn: bool,
        device: &Device,
    ) -> Self {
        let padding = 3 * dilation;
        let conv_dw = Conv2dConfig::new([dim, dim], [7, 7])
            .with_stride([1, 1])
            .with_dilation([dilation, dilation])
            .with_padding(PaddingConfig2d::Explicit(padding, padding, padding, padding))
            .with_groups(dim)
            .with_bias(true)
            .init(device);
        let norm = LayerNormConfig::new(dim).with_epsilon(1e-6).init(device);
        let mlp = ConvNeXtMlp::new(dim, use_grn, device);
        let gamma = ls_init_value.map(|val| {
            Initializer::Constant { value: val }.init([dim], device)
        });
        let drop_path = DropPath::new(drop_path_rate);

        Self {
            conv_dw,
            norm,
            mlp,
            gamma,
            drop_path,
        }
    }

    /// Forward pass through the residual block: `[B, C, H, W] -> [B, C, H, W]`.
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let shortcut = input.clone();
        let x = self.conv_dw.forward(input);
        // Permute to [B, H, W, C] for channels-last normalization and MLP.
        let x = x.permute([0, 2, 3, 1]);
        let x = self.norm.forward(x);
        let x = self.mlp.forward(x);
        let x = x.permute([0, 3, 1, 2]);

        let x = match &self.gamma {
            Some(gamma) => {
                let c = x.dims()[1];
                x * gamma.val().reshape([1, c, 1, 1])
            }
            None => x,
        };

        self.drop_path.forward(x) + shortcut
    }
}

/// A stage of ConvNeXt consisting of an optional downsample layer followed by a sequence of blocks.
#[derive(Module, Debug)]
pub struct ConvNeXtStage {
    /// Optional downsampling layer at the entrance of the stage.
    pub downsample: Option<ConvNeXtDownsample>,
    /// Stack of residual blocks.
    pub blocks: Vec<ConvNeXtBlock>,
}

impl ConvNeXtStage {
    /// Creates a stage.
    pub fn new(
        in_chans: usize,
        out_chans: usize,
        depth: usize,
        stride: usize,
        dilation: (usize, usize),
        drop_path_rates: &[f64],
        ls_init_value: Option<f64>,
        use_grn: bool,
        device: &Device,
    ) -> Self {
        let downsample = (in_chans != out_chans || stride > 1 || dilation.0 != dilation.1).then(|| {
            ConvNeXtDownsample::new(in_chans, out_chans, stride, dilation, device)
        });

        let blocks = (0..depth)
            .map(|i| {
                ConvNeXtBlock::new(
                    out_chans,
                    dilation.1,
                    drop_path_rates[i],
                    ls_init_value,
                    use_grn,
                    device,
                )
            })
            .collect();

        Self { downsample, blocks }
    }

    /// Forward pass through the stage.
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let mut x = match &self.downsample {
            Some(ds) => ds.forward(x),
            None => x,
        };
        for block in &self.blocks {
            x = block.forward(x);
        }
        x
    }
}

// ----------------------------------------------------------------------------
// Model
// ----------------------------------------------------------------------------

/// Pure-Rust implementation of ConvNeXt (V1 and V2) backbones.
#[derive(Module, Debug)]
pub struct ConvNeXt {
    /// Patch stem mapping input images to stage 0 resolution.
    pub stem: ConvNeXtStem,
    /// 4 feature resolution stages (or fewer when truncated).
    pub stages: Vec<ConvNeXtStage>,
    /// Classifier head.
    pub head: NormMlpClassifierHead,
    #[module(skip)]
    meta: ModelMeta,
}

impl ConvNeXt {
    /// Constructs a `ConvNeXt` model from `config`.
    pub fn new(config: ConvNeXtConfig, device: &Device) -> Self {
        config
            .validate()
            .unwrap_or_else(|err| panic!("invalid ConvNeXt configuration: {err}"));

        let stem = ConvNeXtStem::new(config.in_chans, config.dims[0], device);

        let total_blocks: usize = config.depths.iter().sum();
        let dp_schedule = drop_path_schedule(config.drop_path_rate, total_blocks);

        let output_stride = config.output_stride.unwrap_or(32);
        let max_stage = if config.features_only {
            config
                .out_indices
                .as_ref()
                .and_then(|indices| indices.iter().max().copied())
                .unwrap_or(3)
        } else {
            3
        };

        let mut stages = Vec::with_capacity(max_stage + 1);
        let mut feature_info = Vec::with_capacity(max_stage + 1);

        let mut prev_chs = config.dims[0];
        let mut curr_stride = 4usize;
        let mut dilation = 1usize;
        let mut block_offset = 0usize;

        const MODULE_NAMES: [&str; 4] = ["stages.0", "stages.1", "stages.2", "stages.3"];

        for i in 0..=max_stage {
            let mut stride = if i > 0 { 2 } else { 1 };
            if curr_stride >= output_stride && stride > 1 {
                dilation *= stride;
                stride = 1;
            }
            curr_stride *= stride;
            let first_dilation = if dilation == 1 || dilation == 2 { 1 } else { 2 };
            let out_chs = config.dims[i];
            let depth = config.depths[i];

            let stage_dp = &dp_schedule[block_offset..block_offset + depth];
            block_offset += depth;

            let stage = ConvNeXtStage::new(
                prev_chs,
                out_chs,
                depth,
                stride,
                (first_dilation, dilation),
                stage_dp,
                config.ls_init_value,
                config.use_grn,
                device,
            );
            stages.push(stage);
            prev_chs = out_chs;

            feature_info.push(FeatureInfo::new(i, prev_chs, curr_stride, MODULE_NAMES[i]));
        }

        let num_features = prev_chs;
        let norm_eps = if config.features_only { None } else { Some(1e-6) };
        let head = NormMlpClassifierHead::new(
            num_features,
            config.num_classes,
            None,
            PoolMode::Avg,
            config.drop_rate,
            norm_eps,
            device,
        );

        Self {
            stem,
            stages,
            head,
            meta: ModelMeta {
                name: "convnext".to_string(),
                num_classes: config.num_classes,
                num_features,
                feature_info,
                pretrained_cfg: None,
            },
        }
    }

    /// Replaces model metadata (used by the registry).
    pub(crate) fn set_meta(&mut self, meta: ModelMeta) {
        self.meta = meta;
    }

    /// First convolution of the stem, if present.
    pub fn first_conv_mut(&mut self) -> Option<&mut Conv2d> {
        Some(&mut self.stem.conv)
    }

    /// Replaces the first convolution weight (used when adapting `in_chans`).
    pub fn set_first_conv_weight(&mut self, weight: Tensor<4>) {
        self.stem.conv.weight = Param::initialized(ParamId::new(), weight);
    }

    /// Weight-name rewrites from `timm`'s PyTorch state dict to this module hierarchy.
    pub fn key_remaps() -> Vec<(&'static str, &'static str)> {
        vec![
            ("^stem\\.0\\.([a-z_]+)$", "stem.conv.$1"),
            ("^stem\\.1\\.([a-z_]+)$", "stem.norm.norm.$1"),
            ("^stages\\.(\\d+)\\.downsample\\.0\\.([a-z_]+)$", "stages.$1.downsample.norm.norm.$2"),
            ("^stages\\.(\\d+)\\.downsample\\.1\\.([a-z_]+)$", "stages.$1.downsample.conv.$2"),
            ("^head\\.norm\\.([a-z_]+)$", "head.norm.norm.$1"),
        ]
    }

    fn run_intermediates(
        &self,
        input: Tensor<4>,
        options: &IntermediatesOptions,
    ) -> (Tensor<4>, Vec<Tensor<4>>) {
        let (selected, max_idx) = options
            .resolve(self.feature_info().len())
            .unwrap_or_else(|err| panic!("{err}"));

        let mut intermediates = Vec::new();
        let mut x = self.stem.forward(input);

        for (idx, stage) in self.stages.iter().enumerate() {
            x = stage.forward(x);
            if selected.contains(&idx) {
                let out = match options.format {
                    FeatureFormat::Nchw => x.clone(),
                    FeatureFormat::Nhwc => x.clone().permute([0, 2, 3, 1]),
                };
                intermediates.push(out);
            }
            if options.stop_early && idx == max_idx {
                break;
            }
        }

        (x, intermediates)
    }
}

impl VisionModel for ConvNeXt {
    fn forward(&self, input: Tensor<4>) -> Tensor<2> {
        let features = self.forward_features(input);
        self.forward_head(features, false)
    }

    fn forward_features(&self, input: Tensor<4>) -> Tensor<4> {
        let mut x = self.stem.forward(input);
        for stage in &self.stages {
            x = stage.forward(x);
        }
        x
    }

    fn forward_head(&self, features: Tensor<4>, pre_logits: bool) -> Tensor<2> {
        self.head.forward(features, pre_logits)
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

