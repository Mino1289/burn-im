//! `ResNet`, `ResNet-D` and `ResNeXt` backbones.
//!
//! Reproduces `timm`'s `ResNet` implementation (see `timm/models/resnet.py`): standard and deep
//! stems, `BasicBlock`/`Bottleneck` residual blocks, average-pool downsample (`-d` variants) and
//! grouped convolutions (`ResNeXt`).

use burn::module::{Param, ParamId};
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AvgPool2d, AvgPool2dConfig, MaxPool2d, MaxPool2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig};
use burn::prelude::*;
use burn::tensor::activation::relu;

use crate::layers::head::ClassifierHead;
use crate::layers::pool::PoolMode;
use crate::models::ModelMeta;
use crate::traits::{FeatureFormat, FeatureInfo, IntermediatesOptions, VisionModel};

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

    /// Embedding dimension produced by the last stage.
    pub fn num_features(&self) -> usize {
        self.widths[3] * self.block.expansion()
    }
}

/// One 3x3 (basic) or 1x1-3x3-1x1 (bottleneck) residual block.
#[derive(Module, Debug)]
pub struct ResNetBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    conv3: Option<Conv2d>,
    bn3: Option<BatchNorm>,
    downsample: Option<Downsample>,
}

impl ResNetBlock {
    /// Applies the residual block.
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let shortcut = input.clone();

        let x = self.conv1.forward(input);
        let x = relu(self.bn1.forward(x));
        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        let x = match (&self.conv3, &self.bn3) {
            (Some(conv3), Some(bn3)) => {
                let x = relu(x);
                bn3.forward(conv3.forward(x))
            }
            _ => x,
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
    conv: Conv2d,
    bn: BatchNorm,
}

impl Downsample {
    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let x = match &self.avg {
            Some(avg) => avg.forward(input),
            None => input,
        };
        self.bn.forward(self.conv.forward(x))
    }
}

/// Convolution stem, standard (one 7x7 conv) or deep (three 3x3 convs).
#[derive(Module, Debug)]
pub struct ResNetStem {
    conv1: Option<Conv2d>,
    conv_a: Option<Conv2d>,
    bn_a: Option<BatchNorm>,
    conv_b: Option<Conv2d>,
    bn_b: Option<BatchNorm>,
    conv_c: Option<Conv2d>,
    bn1: BatchNorm,
    #[module(skip)]
    deep: bool,
}

impl ResNetStem {
    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        if !self.deep {
            let conv1 = self.conv1.as_ref().expect("standard stem convolution");
            return relu(self.bn1.forward(conv1.forward(input)));
        }

        let x = relu(
            self.bn_a.as_ref().expect("deep stem norm a").forward(
                self.conv_a
                    .as_ref()
                    .expect("deep stem conv a")
                    .forward(input),
            ),
        );
        let x = relu(
            self.bn_b
                .as_ref()
                .expect("deep stem norm b")
                .forward(self.conv_b.as_ref().expect("deep stem conv b").forward(x)),
        );
        relu(
            self.bn1
                .forward(self.conv_c.as_ref().expect("deep stem conv c").forward(x)),
        )
    }

    fn first_conv(&mut self) -> Option<&mut Conv2d> {
        if self.deep {
            self.conv_a.as_mut()
        } else {
            self.conv1.as_mut()
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
    pub fn new(config: ResNetConfig, device: &Device) -> Self {
        let expansion = config.block.expansion();
        let deep = config.stem == StemKind::Deep;
        let stem_chs = 64;

        let (conv1, conv_a, bn_a, conv_b, bn_b, conv_c) = if deep {
            // timm's `-d` variants: stem widths 32, 32, 64.
            let conv_a = conv_layer(config.in_chans, 32, 3, 2, 1, 1, 1, device);
            let conv_b = conv_layer(32, 32, 3, 1, 1, 1, 1, device);
            let conv_c = conv_layer(32, stem_chs, 3, 1, 1, 1, 1, device);
            (
                None,
                Some(conv_a),
                Some(BatchNormConfig::new(32).init(device)),
                Some(conv_b),
                Some(BatchNormConfig::new(32).init(device)),
                Some(conv_c),
            )
        } else {
            (
                Some(conv_layer(config.in_chans, stem_chs, 7, 2, 3, 1, 1, device)),
                None,
                None,
                None,
                None,
                None,
            )
        };

        let stem = ResNetStem {
            conv1,
            conv_a,
            bn_a,
            conv_b,
            bn_b,
            conv_c,
            bn1: BatchNormConfig::new(stem_chs).init(device),
            deep,
        };

        let mut inplanes = stem_chs;
        let mut layer1 = Vec::with_capacity(config.layers[0]);
        let mut layer2 = Vec::with_capacity(config.layers[1]);
        let mut layer3 = Vec::with_capacity(config.layers[2]);
        let mut layer4 = Vec::with_capacity(config.layers[3]);

        build_stage(
            &config,
            config.layers[0],
            config.widths[0],
            1,
            &mut inplanes,
            device,
            &mut layer1,
        );
        build_stage(
            &config,
            config.layers[1],
            config.widths[1],
            2,
            &mut inplanes,
            device,
            &mut layer2,
        );
        build_stage(
            &config,
            config.layers[2],
            config.widths[2],
            2,
            &mut inplanes,
            device,
            &mut layer3,
        );
        build_stage(
            &config,
            config.layers[3],
            config.widths[3],
            2,
            &mut inplanes,
            device,
            &mut layer4,
        );

        let num_features = config.num_features();
        let head = ClassifierHead::new(
            num_features,
            config.num_classes,
            PoolMode::Avg,
            config.drop_rate,
            device,
        );

        let feature_info = vec![
            FeatureInfo::new(0, stem_chs, 2, "act1"),
            FeatureInfo::new(1, config.widths[0] * expansion, 4, "layer1"),
            FeatureInfo::new(2, config.widths[1] * expansion, 8, "layer2"),
            FeatureInfo::new(3, config.widths[2] * expansion, 16, "layer3"),
            FeatureInfo::new(4, config.widths[3] * expansion, 32, "layer4"),
        ];

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
                remaps.push(("^conv1\\.", "stem.conv1."));
            }
            StemKind::Deep => {
                remaps.push(("^conv1\\.0\\.", "stem.conv_a."));
                remaps.push(("^conv1\\.1\\.", "stem.bn_a."));
                remaps.push(("^conv1\\.3\\.", "stem.conv_b."));
                remaps.push(("^conv1\\.4\\.", "stem.bn_b."));
                remaps.push(("^conv1\\.6\\.", "stem.conv_c."));
            }
        }
        remaps.push(("^bn1\\.", "stem.bn1."));
        if config.avg_down {
            remaps.push(("\\.downsample\\.1\\.", ".downsample.conv."));
            remaps.push(("\\.downsample\\.2\\.", ".downsample.bn."));
        } else {
            remaps.push(("\\.downsample\\.0\\.", ".downsample.conv."));
            remaps.push(("\\.downsample\\.1\\.", ".downsample.bn."));
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
fn conv_layer(
    in_chans: usize,
    out_chans: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    device: &Device,
) -> Conv2d {
    Conv2dConfig::new([in_chans, out_chans], [kernel, kernel])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(
            padding, padding, padding, padding,
        ))
        .with_dilation([dilation, dilation])
        .with_groups(groups)
        .with_bias(false)
        .init(device)
}

#[allow(clippy::too_many_arguments)]
fn build_stage(
    config: &ResNetConfig,
    num_blocks: usize,
    planes: usize,
    stride: usize,
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

    let needs_projection = stride != 1 || *inplanes != outplanes;
    let downsample = needs_projection.then(|| Downsample {
        avg: (config.avg_down && stride != 1)
            .then(|| AvgPool2dConfig::new([2, 2]).with_strides([2, 2]).init()),
        conv: conv_layer(
            *inplanes,
            outplanes,
            1,
            if config.avg_down { 1 } else { stride },
            0,
            1,
            1,
            device,
        ),
        bn: BatchNormConfig::new(outplanes).init(device),
    });

    for block_idx in 0..num_blocks {
        let block_stride = if block_idx == 0 { stride } else { 1 };
        let block_downsample = if block_idx == 0 {
            downsample.clone()
        } else {
            None
        };

        let (conv1, bn1) = match config.block {
            BlockKind::Basic => (
                conv_layer(*inplanes, width, 3, block_stride, 1, 1, 1, device),
                BatchNormConfig::new(width).init(device),
            ),
            BlockKind::Bottleneck => (
                conv_layer(*inplanes, width, 1, 1, 0, 1, 1, device),
                BatchNormConfig::new(width).init(device),
            ),
        };

        let (conv2, bn2) = match config.block {
            BlockKind::Basic => (
                conv_layer(width, outplanes, 3, 1, 1, 1, 1, device),
                BatchNormConfig::new(outplanes).init(device),
            ),
            BlockKind::Bottleneck => (
                conv_layer(width, width, 3, block_stride, 1, 1, config.groups, device),
                BatchNormConfig::new(width).init(device),
            ),
        };

        let (conv3, bn3) = match config.block {
            BlockKind::Bottleneck => (
                Some(conv_layer(width, outplanes, 1, 1, 0, 1, 1, device)),
                Some(BatchNormConfig::new(outplanes).init(device)),
            ),
            BlockKind::Basic => (None, None),
        };

        out.push(ResNetBlock {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
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
