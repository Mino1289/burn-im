//! `ResNet`, `ResNet-D` and `ResNeXt` backbones.
//!
//! Reproduces `timm`'s `ResNet` implementation (see `timm/models/resnet.py`): standard and deep
//! stems, `BasicBlock`/`Bottleneck` residual blocks, average-pool downsample (`-d` variants) and
//! grouped convolutions (`ResNeXt`).

use burn::module::{Param, ParamId};
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::pool::{AvgPool2d, AvgPool2dConfig, MaxPool2d, MaxPool2dConfig};
use burn::prelude::*;
use burn::tensor::activation::relu;

use crate::error::{Error, Result};
use crate::layers::conv_norm_act::{Activation, ConvNormAct, ConvNormActConfig};
use crate::layers::head::ClassifierHead;
use crate::layers::pool::PoolMode;
use crate::models::ModelMeta;
use crate::traits::{FeatureFormat, FeatureInfo, IntermediatesOptions, VisionModel};

/// Number of feature levels exposed by the family: the stem activation plus four stages.
const NUM_LEVELS: usize = 5;

/// Index of the deepest stage, used as the default `features_only` level.
const LAST_LEVEL: usize = NUM_LEVELS - 1;

/// Residual block type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    /// Two 3x3 convolutions (`ResNet-18/34`).
    Basic,
    /// 1x1 -> 3x3 -> 1x1 bottleneck (`ResNet-50` and deeper).
    Bottleneck,
}

impl BlockKind {
    /// Channel expansion factor of the block.
    pub const fn expansion(self) -> usize {
        match self {
            Self::Basic => 1,
            Self::Bottleneck => 4,
        }
    }
}

/// Stem type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StemKind {
    /// Single 7x7 stride-2 convolution (`ResNet`, `ResNeXt`).
    Standard,
    /// Three stacked 3x3 convolutions (`ResNet-D`, `-c`, `-e`, `-s`).
    Deep,
}

/// Strides and dilations of one residual stage.
///
/// See [`ResNetConfig::stage_plan`] for how `timm` derives them from the output stride.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StagePlan {
    /// Stride of the stage's first block.
    stride: usize,
    /// Dilation of the stage's 3x3 convolutions, from the second block on.
    dilation: usize,
    /// Dilation in effect when the stage starts, used by its first block.
    entry_dilation: usize,
}

impl Default for StagePlan {
    fn default() -> Self {
        Self {
            stride: 1,
            dilation: 1,
            entry_dilation: 1,
        }
    }
}

/// Configuration of a ResNet variant.
#[derive(Debug, Clone)]
pub struct ResNetConfig {
    /// Number of blocks per stage.
    pub layers: [usize; 4],
    /// Base channel width per stage.
    pub widths: [usize; 4],
    /// Block type.
    pub block: BlockKind,
    /// Stem type.
    pub stem: StemKind,
    /// Use average pooling in the residual downsample path (`ResNet-D`).
    pub avg_down: bool,
    /// Convolution groups of the 3x3 bottleneck convolution (`ResNeXt`).
    pub groups: usize,
    /// Base width multiplier of the bottleneck (`ResNeXt`, wide ResNets).
    pub base_width: usize,
    /// Number of output classes (`0` removes the classifier).
    pub num_classes: usize,
    /// Number of input channels.
    pub in_chans: usize,
    /// Dropout probability before the classifier.
    pub drop_rate: f64,
    /// Build a headless extractor stopping after the last requested level.
    pub features_only: bool,
    /// Feature levels to keep in `features_only` mode.
    pub out_indices: Option<Vec<usize>>,
    /// Spatial reduction of the deepest stage: `32` (default), `16` or `8`.
    ///
    /// Values below `32` replace the stride of the last stages with dilated convolutions, the
    /// torchvision/"DeepLab" convention also followed by `timm`.
    pub output_stride: usize,
}

impl ResNetConfig {
    /// Creates a configuration with `timm`'s defaults for the given block type.
    pub fn new(layers: [usize; 4], widths: [usize; 4], block: BlockKind) -> Self {
        Self {
            layers,
            widths,
            block,
            stem: StemKind::Standard,
            avg_down: false,
            groups: 1,
            base_width: 64,
            num_classes: 1000,
            in_chans: 3,
            drop_rate: 0.0,
            features_only: false,
            out_indices: None,
            output_stride: 32,
        }
    }

    /// Uses a deep stem.
    pub fn with_deep_stem(mut self) -> Self {
        self.stem = StemKind::Deep;
        self
    }

    /// Uses average pooling in downsample paths.
    pub fn with_avg_down(mut self) -> Self {
        self.avg_down = true;
        self
    }

    /// Sets the grouped-convolution parameters.
    pub fn with_groups(mut self, groups: usize, base_width: usize) -> Self {
        self.groups = groups;
        self.base_width = base_width;
        self
    }

    /// Sets the number of classes.
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

    /// Builds a headless feature extractor keeping the levels in `out_indices`.
    ///
    /// Stages deeper than the last requested level are not instantiated at all, so the memory and
    /// loading time match the extractor actually used. The classifier is removed, hence
    /// `num_classes == 0`. Indices follow the family's feature list: `0` is the stem activation and
    /// `1..=4` are the four stages.
    pub fn with_features_only(mut self, out_indices: impl IntoIterator<Item = usize>) -> Self {
        self.features_only = true;
        self.out_indices = Some(out_indices.into_iter().collect());
        self.num_classes = 0;
        self
    }

    /// Sets the spatial reduction of the deepest stage (`32`, `16` or `8`).
    pub fn with_output_stride(mut self, output_stride: usize) -> Self {
        self.output_stride = output_stride;
        self
    }

    /// Checks the configuration before instantiation.
    ///
    /// Returns [`Error::InvalidConfiguration`] for an unsupported `output_stride`, an empty or
    /// out-of-range `out_indices`, `out_indices` without `features_only`, and a classifier kept
    /// alongside `features_only`.
    pub fn validate(&self) -> Result<()> {
        if !matches!(self.output_stride, 8 | 16 | 32) {
            return Err(Error::InvalidConfiguration(format!(
                "unsupported output_stride {} (expected 8, 16 or 32)",
                self.output_stride
            )));
        }
        if let Some(indices) = &self.out_indices {
            if !self.features_only {
                return Err(Error::InvalidConfiguration(
                    "`out_indices` requires `features_only`".to_string(),
                ));
            }
            if indices.is_empty() {
                return Err(Error::InvalidConfiguration(
                    "`out_indices` must not be empty".to_string(),
                ));
            }
            if let Some(&idx) = indices.iter().find(|idx| **idx >= NUM_LEVELS) {
                return Err(Error::InvalidConfiguration(format!(
                    "feature index {idx} out of range (the family exposes {NUM_LEVELS} levels: 0..={LAST_LEVEL})"
                )));
            }
        }
        if self.features_only && self.num_classes != 0 {
            return Err(Error::InvalidConfiguration(
                "`features_only` removes the classifier (`num_classes` must be 0)".to_string(),
            ));
        }
        Ok(())
    }

    /// Feature levels built and exposed by the model, sorted and de-duplicated.
    pub fn effective_out_indices(&self) -> Vec<usize> {
        let mut indices = match (self.features_only, &self.out_indices) {
            (true, Some(indices)) => indices.clone(),
            (true, None) => vec![LAST_LEVEL],
            (false, _) => (0..NUM_LEVELS).collect(),
        };
        indices.sort_unstable();
        indices.dedup();
        indices
    }

    /// Per-stage plan implementing [`Self::output_stride`].
    ///
    /// Reproduces `timm`'s `_make_layer`: once the nominal reduction reaches
    /// [`Self::output_stride`], the stride of the following stages is folded into the dilation of
    /// their 3x3 convolution instead of sampling the feature map more coarsely.
    ///
    /// The plan carries two dilations because `timm` passes `first_dilation=prev_dilation` to each
    /// block, where `prev_dilation` is the dilation of the *previous* block: the first block of a
    /// dilated stage therefore convolves with the dilation of the stage before it, and only the
    /// following blocks use [`StagePlan::dilation`].
    fn stage_plan(&self) -> [StagePlan; 4] {
        let mut plans = [StagePlan::default(); 4];
        // The stem (conv stride 2 + max-pool stride 2) already reduces by 4.
        let mut net_stride = 4;
        let mut dilation = 1;
        let mut previous_dilation = 1;

        for (stage, plan) in plans.iter_mut().enumerate() {
            let mut stride = if stage == 0 { 1 } else { 2 };
            if net_stride >= self.output_stride {
                dilation *= stride;
                stride = 1;
            } else {
                net_stride *= stride;
            }
            *plan = StagePlan {
                stride,
                dilation,
                entry_dilation: previous_dilation,
            };
            previous_dilation = dilation;
        }

        plans
    }

    /// Embedding dimension produced by the last stage.
    pub fn num_features(&self) -> usize {
        self.widths[3] * self.block.expansion()
    }
}

/// One 3x3 (basic) or 1x1-3x3-1x1 (bottleneck) residual block.
#[derive(Module, Debug)]
pub struct ResNetBlock {
    conv1: ConvNormAct,
    conv2: ConvNormAct,
    conv3: Option<ConvNormAct>,
    downsample: Option<Downsample>,
}

impl ResNetBlock {
    /// Applies the residual block.
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let shortcut = input.clone();

        let x = self.conv1.forward(input);
        let x = self.conv2.forward(x);
        let x = match &self.conv3 {
            Some(conv3) => conv3.forward(x),
            None => x,
        };

        let shortcut = match &self.downsample {
            Some(downsample) => downsample.forward(shortcut),
            None => shortcut,
        };
        relu(x.add(shortcut))
    }
}

/// Projection shortcut of a residual block.
#[derive(Module, Debug)]
pub struct Downsample {
    avg: Option<AvgPool2d>,
    body: ConvNormAct,
}

impl Downsample {
    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let x = match &self.avg {
            Some(avg) => avg.forward(input),
            None => input,
        };
        self.body.forward(x)
    }
}

/// Convolution stem, standard (one 7x7 conv) or deep (three 3x3 convs).
#[derive(Module, Debug)]
pub struct ResNetStem {
    conv1: Option<ConvNormAct>,
    conv_a: Option<ConvNormAct>,
    conv_b: Option<ConvNormAct>,
    conv_c: Option<ConvNormAct>,
    #[module(skip)]
    deep: bool,
}

impl ResNetStem {
    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        if !self.deep {
            let conv1 = self.conv1.as_ref().expect("standard stem convolution");
            return conv1.forward(input);
        }

        let x = self
            .conv_a
            .as_ref()
            .expect("deep stem block a")
            .forward(input);
        let x = self.conv_b.as_ref().expect("deep stem block b").forward(x);
        self.conv_c.as_ref().expect("deep stem block c").forward(x)
    }

    fn first_conv(&mut self) -> Option<&mut Conv2d> {
        if self.deep {
            self.conv_a.as_mut().map(ConvNormAct::conv_mut)
        } else {
            self.conv1.as_mut().map(ConvNormAct::conv_mut)
        }
    }
}

/// ResNet / ResNet-D / ResNeXt backbone.
#[derive(Module, Debug)]
pub struct ResNet {
    stem: ResNetStem,
    maxpool: MaxPool2d,
    layer1: Vec<ResNetBlock>,
    layer2: Vec<ResNetBlock>,
    layer3: Vec<ResNetBlock>,
    layer4: Vec<ResNetBlock>,
    head: ClassifierHead,
    #[module(skip)]
    meta: ModelMeta,
}

impl ResNet {
    /// Instantiates a ResNet from a configuration.
    ///
    /// # Panics
    ///
    /// Panics if [`ResNetConfig::validate`] fails; the registry validates first and returns
    /// [`Error::InvalidConfiguration`] instead.
    pub fn new(config: ResNetConfig, device: &Device) -> Self {
        config
            .validate()
            .expect("invalid ResNet configuration; call `validate` first");

        let expansion = config.block.expansion();
        let deep = config.stem == StemKind::Deep;
        let stem_chs = 64;

        let stem = if deep {
            // timm's `-d` variants: stem widths 32, 32, 64.
            ResNetStem {
                conv1: None,
                conv_a: Some(cna(
                    config.in_chans,
                    32,
                    3,
                    2,
                    1,
                    1,
                    1,
                    Activation::Relu,
                    device,
                )),
                conv_b: Some(cna(32, 32, 3, 1, 1, 1, 1, Activation::Relu, device)),
                conv_c: Some(cna(32, stem_chs, 3, 1, 1, 1, 1, Activation::Relu, device)),
                deep,
            }
        } else {
            ResNetStem {
                conv1: Some(cna(
                    config.in_chans,
                    stem_chs,
                    7,
                    2,
                    3,
                    1,
                    1,
                    Activation::Relu,
                    device,
                )),
                conv_a: None,
                conv_b: None,
                conv_c: None,
                deep,
            }
        };

        // Levels kept by `features_only`: stages deeper than the last level are never built.
        let out_indices = config.effective_out_indices();
        let last_level = out_indices.last().copied().unwrap_or(LAST_LEVEL);
        let plan = config.stage_plan();

        let mut inplanes = stem_chs;
        let mut stages: [Vec<ResNetBlock>; 4] = std::array::from_fn(|_| Vec::new());
        for (stage, stage_stage) in stages.iter_mut().enumerate() {
            let level = stage + 1;
            if level > last_level {
                break;
            }
            build_stage(
                &config,
                config.layers[stage],
                config.widths[stage],
                plan[stage],
                &mut inplanes,
                device,
                stage_stage,
            );
        }
        let [layer1, layer2, layer3, layer4] = stages;

        // Deepest level actually built, which is what the head pools.
        let num_features = if last_level == 0 {
            stem_chs
        } else {
            config.widths[last_level - 1] * expansion
        };
        let head = ClassifierHead::new(
            num_features,
            config.num_classes,
            PoolMode::Avg,
            config.drop_rate,
            device,
        );

        // Level 0 is the stem activation (conv stride 2). The stem's 3x3 max-pool then halves the
        // map once more before the first stage, so it also applies when a stage is not built.
        let mut reductions = Vec::with_capacity(NUM_LEVELS);
        let mut reduction = 2;
        reductions.push(reduction);
        reduction *= 2;
        for stage in &plan {
            reduction *= stage.stride;
            reductions.push(reduction);
        }

        let feature_info = (0..=last_level)
            .map(|level| match level {
                0 => FeatureInfo::new(0, stem_chs, reductions[0], "act1"),
                stage => FeatureInfo::new(
                    level,
                    config.widths[stage - 1] * expansion,
                    reductions[level],
                    ["layer1", "layer2", "layer3", "layer4"][stage - 1],
                ),
            })
            .collect();

        Self {
            stem,
            maxpool: MaxPool2dConfig::new([3, 3])
                .with_strides([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(),
            layer1,
            layer2,
            layer3,
            layer4,
            head,
            meta: ModelMeta {
                name: "resnet".to_string(),
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
        self.stem.first_conv()
    }

    /// Replaces the first convolution weight (used when adapting `in_chans`).
    pub fn set_first_conv_weight(&mut self, weight: Tensor<4>) {
        if let Some(conv) = self.stem.first_conv() {
            conv.weight = Param::initialized(ParamId::new(), weight);
        }
    }

    /// Weight-name rewrites from `timm`'s PyTorch state dict to this module hierarchy.
    pub fn key_remaps(config: &ResNetConfig) -> Vec<(&'static str, &'static str)> {
        let mut remaps = vec![("^fc\\.", "head.fc.")];
        match config.stem {
            StemKind::Standard => {
                remaps.push(("^conv1\\.([a-z_]+)$", "stem.conv1.conv.$1"));
                remaps.push(("^bn1\\.([a-z_]+)$", "stem.conv1.norm.$1"));
            }
            StemKind::Deep => {
                remaps.push(("^conv1\\.0\\.([a-z_]+)$", "stem.conv_a.conv.$1"));
                remaps.push(("^conv1\\.1\\.([a-z_]+)$", "stem.conv_a.norm.$1"));
                remaps.push(("^conv1\\.3\\.([a-z_]+)$", "stem.conv_b.conv.$1"));
                remaps.push(("^conv1\\.4\\.([a-z_]+)$", "stem.conv_b.norm.$1"));
                remaps.push(("^conv1\\.6\\.([a-z_]+)$", "stem.conv_c.conv.$1"));
                remaps.push(("^bn1\\.([a-z_]+)$", "stem.conv_c.norm.$1"));
            }
        }
        remaps.push((
            "^(layer[1-4]\\.\\d+)\\.conv([123])\\.([a-z_]+)$",
            "$1.conv$2.conv.$3",
        ));
        remaps.push((
            "^(layer[1-4]\\.\\d+)\\.bn([123])\\.([a-z_]+)$",
            "$1.conv$2.norm.$3",
        ));
        if config.avg_down {
            remaps.push(("\\.downsample\\.1\\.", ".downsample.body.conv."));
            remaps.push(("\\.downsample\\.2\\.", ".downsample.body.norm."));
        } else {
            remaps.push(("\\.downsample\\.0\\.", ".downsample.body.conv."));
            remaps.push(("\\.downsample\\.1\\.", ".downsample.body.norm."));
        }
        remaps
    }

    fn stages(&self) -> [&Vec<ResNetBlock>; 4] {
        [&self.layer1, &self.layer2, &self.layer3, &self.layer4]
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
        if selected.contains(&0) {
            intermediates.push(x.clone());
        }

        if !(options.stop_early && max_idx == 0) {
            x = self.maxpool.forward(x);
            for (stage_idx, stage) in self.stages().iter().enumerate() {
                let feature_idx = stage_idx + 1;
                if options.stop_early && feature_idx > max_idx {
                    break;
                }
                for block in stage.iter() {
                    x = block.forward(x);
                }
                if selected.contains(&feature_idx) {
                    intermediates.push(x.clone());
                }
            }
        }

        let intermediates = intermediates
            .into_iter()
            .map(|tensor| match options.format {
                FeatureFormat::Nchw => tensor,
                FeatureFormat::Nhwc => tensor.permute([0, 2, 3, 1]),
            })
            .collect();
        (x, intermediates)
    }
}

impl VisionModel for ResNet {
    fn forward(&self, input: Tensor<4>) -> Tensor<2> {
        self.forward_head(self.forward_features(input), false)
    }

    fn forward_features(&self, input: Tensor<4>) -> Tensor<4> {
        let mut x = self.stem.forward(input);
        x = self.maxpool.forward(x);
        for stage in self.stages() {
            for block in stage.iter() {
                x = block.forward(x);
            }
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

    fn pretrained_cfg(&self) -> Option<&crate::hub::config::PretrainedCfg> {
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

#[allow(clippy::too_many_arguments)]
fn cna(
    in_chans: usize,
    out_chans: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    activation: Activation,
    device: &Device,
) -> ConvNormAct {
    ConvNormActConfig::new(in_chans, out_chans)
        .with_kernel_size(kernel)
        .with_stride(stride)
        .with_padding(padding)
        .with_dilation(dilation)
        .with_groups(groups)
        .with_activation(activation)
        .init(device)
}

/// Pooling of the average-pool downsample path (`ResNet-D`).
///
/// `timm` replaces the stride of the projection shortcut with an average pool: 2x2/2 for a
/// strided stage, and a "same"-padded 2x2/1 pool when the stride was folded into the dilation, so
/// the shortcut keeps the stage's resolution. That "same" pool is padded with a tensor operation
/// and still divides by the whole kernel area, so the zeros are counted as values.
fn ds_pool(config: &ResNetConfig, plan: StagePlan) -> Option<AvgPool2d> {
    if !config.avg_down || (plan.stride == 1 && plan.dilation == 1) {
        return None;
    }

    let avg = AvgPool2dConfig::new([2, 2]).with_ceil_mode(true);
    Some(if plan.stride == 1 {
        avg.with_strides([1, 1])
            .with_padding(PaddingConfig2d::Same)
            .with_count_include_pad(true)
            .init()
    } else {
        avg.with_strides([2, 2])
            .with_count_include_pad(false)
            .init()
    })
}

#[allow(clippy::too_many_arguments)]
fn build_stage(
    config: &ResNetConfig,
    num_blocks: usize,
    planes: usize,
    plan: StagePlan,
    inplanes: &mut usize,
    device: &Device,
    out: &mut Vec<ResNetBlock>,
) {
    let expansion = config.block.expansion();
    let outplanes = planes * expansion;
    let width = match config.block {
        BlockKind::Basic => {
            debug_assert_eq!(config.groups, 1, "BasicBlock does not support groups");
            planes
        }
        BlockKind::Bottleneck => {
            (planes as f64 * (config.base_width as f64 / 64.0)).floor() as usize * config.groups
        }
    };

    let needs_projection = plan.stride != 1 || *inplanes != outplanes;
    let downsample = needs_projection.then(|| Downsample {
        avg: ds_pool(config, plan),
        body: cna(
            *inplanes,
            outplanes,
            1,
            if config.avg_down { 1 } else { plan.stride },
            0,
            1,
            1,
            Activation::None,
            device,
        ),
    });

    for block_idx in 0..num_blocks {
        let block_stride = if block_idx == 0 { plan.stride } else { 1 };
        // `timm` hands each block the dilation of the block before it (`first_dilation`).
        let first_dilation = if block_idx == 0 {
            plan.entry_dilation
        } else {
            plan.dilation
        };
        let block_downsample = if block_idx == 0 {
            downsample.clone()
        } else {
            None
        };

        let conv1 = match config.block {
            BlockKind::Basic => cna(
                *inplanes,
                width,
                3,
                block_stride,
                first_dilation,
                first_dilation,
                1,
                Activation::Relu,
                device,
            ),
            BlockKind::Bottleneck => cna(*inplanes, width, 1, 1, 0, 1, 1, Activation::Relu, device),
        };

        let conv2 = match config.block {
            BlockKind::Basic => cna(
                width,
                outplanes,
                3,
                1,
                plan.dilation,
                plan.dilation,
                1,
                Activation::None,
                device,
            ),
            BlockKind::Bottleneck => cna(
                width,
                width,
                3,
                block_stride,
                first_dilation,
                first_dilation,
                config.groups,
                Activation::Relu,
                device,
            ),
        };

        let conv3 = match config.block {
            BlockKind::Bottleneck => Some(cna(
                width,
                outplanes,
                1,
                1,
                0,
                1,
                1,
                Activation::None,
                device,
            )),
            BlockKind::Basic => None,
        };

        out.push(ResNetBlock {
            conv1,
            conv2,
            conv3,
            downsample: block_downsample,
        });
        *inplanes = outplanes;
    }
}

/// Convenience: a `ResNet-18` configuration.
pub fn resnet18(num_classes: usize) -> ResNetConfig {
    ResNetConfig::new([2, 2, 2, 2], [64, 128, 256, 512], BlockKind::Basic)
        .with_num_classes(num_classes)
}

/// Convenience: a `ResNet-34` configuration.
pub fn resnet34(num_classes: usize) -> ResNetConfig {
    ResNetConfig::new([3, 4, 6, 3], [64, 128, 256, 512], BlockKind::Basic)
        .with_num_classes(num_classes)
}

/// Convenience: a `ResNet-50` configuration.
pub fn resnet50(num_classes: usize) -> ResNetConfig {
    ResNetConfig::new([3, 4, 6, 3], [64, 128, 256, 512], BlockKind::Bottleneck)
        .with_num_classes(num_classes)
}

/// Convenience: a `ResNet-101` configuration.
pub fn resnet101(num_classes: usize) -> ResNetConfig {
    ResNetConfig::new([3, 4, 23, 3], [64, 128, 256, 512], BlockKind::Bottleneck)
        .with_num_classes(num_classes)
}

/// Convenience: a `ResNet-152` configuration.
pub fn resnet152(num_classes: usize) -> ResNetConfig {
    ResNetConfig::new([3, 8, 36, 3], [64, 128, 256, 512], BlockKind::Bottleneck)
        .with_num_classes(num_classes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(stride, padding, dilation)` of a 3x3 convolution.
    fn window(conv: &ConvNormAct) -> Option<[usize; 3]> {
        let conv = conv.conv();
        if conv.kernel_size != [3, 3] {
            return None;
        }
        let padding = match conv.padding {
            PaddingConfig2d::Explicit(top, _, _, _) => top,
            _ => panic!("explicit padding expected"),
        };
        Some([conv.stride[0], padding, conv.dilation[0]])
    }

    /// `(stride, padding, dilation)` of the 3x3 windows of a block, `conv1` before `conv2`.
    fn windows(block: &ResNetBlock) -> Vec<[usize; 3]> {
        [Some(&block.conv1), Some(&block.conv2), block.conv3.as_ref()]
            .into_iter()
            .flatten()
            .filter_map(window)
            .collect()
    }

    /// Windows expected from `timm`'s block construction for one stage.
    ///
    /// `stride`, `entry` and `dilation` are the three values of `StagePlan`: the first block
    /// convolves with `entry` (the dilation of the previous stage) and only the following ones use
    /// `dilation`.
    fn expected_windows(
        num_blocks: usize,
        stride: usize,
        entry: usize,
        dilation: usize,
        basic: bool,
    ) -> Vec<Vec<[usize; 3]>> {
        (0..num_blocks)
            .map(|block| {
                let (block_stride, first) = if block == 0 {
                    (stride, entry)
                } else {
                    (1, dilation)
                };
                if basic {
                    vec![[block_stride, first, first], [1, dilation, dilation]]
                } else {
                    vec![[block_stride, first, first]]
                }
            })
            .collect()
    }

    #[test]
    fn output_stride_follows_timm_first_dilation() {
        let device = Device::flex();

        // `(stride, entry_dilation, dilation)` per stage, read off `timm`'s `_make_layer`.
        let plans = [
            (32, [(1, 1, 1), (2, 1, 1), (2, 1, 1), (2, 1, 1)]),
            (16, [(1, 1, 1), (2, 1, 1), (2, 1, 1), (1, 1, 2)]),
            (8, [(1, 1, 1), (2, 1, 1), (1, 1, 2), (1, 2, 4)]),
        ];

        for (output_stride, stages) in plans {
            for (basic, config) in [(true, resnet18(0)), (false, resnet50(0))] {
                let layers = config.layers;
                let model = ResNet::new(config.with_output_stride(output_stride), &device);

                for (idx, stage) in model.stages().iter().enumerate() {
                    let (stride, entry, dilation) = stages[idx];
                    let actual: Vec<Vec<[usize; 3]>> = stage.iter().map(windows).collect();
                    assert_eq!(
                        actual,
                        expected_windows(layers[idx], stride, entry, dilation, basic),
                        "stage {idx} at output stride {output_stride}"
                    );
                }
            }
        }
    }
}
