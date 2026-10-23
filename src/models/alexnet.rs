//! AlexNet backbone (the torchvision / `timm` 0.9 variant, without local response
//! normalization).
//!
//! Weight names follow PyTorch's `features.<idx>` and `classifier.<idx>` layout and are rewritten
//! by [`AlexNet::key_remaps`].

use burn::module::{Param, ParamId};
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::relu;

use crate::layers::pool::{PoolMode, SelectAdaptivePool2d};
use crate::models::ModelMeta;
use crate::traits::{FeatureFormat, FeatureInfo, IntermediatesOptions, VisionModel};

/// Configuration of an AlexNet variant.
#[derive(Debug, Clone)]
pub struct AlexNetConfig {
    /// Number of output classes (`0` removes the final linear layer).
    pub num_classes: usize,
    /// Number of input channels.
    pub in_chans: usize,
    /// Dropout probability before the two fully-connected layers.
    pub drop_rate: f64,
}

impl AlexNetConfig {
    /// Creates the standard 1000-class configuration.
    pub fn new(num_classes: usize) -> Self {
        Self {
            num_classes,
            in_chans: 3,
            drop_rate: 0.5,
        }
    }

    /// Sets the number of input channels.
    pub fn with_in_chans(mut self, in_chans: usize) -> Self {
        self.in_chans = in_chans;
        self
    }

    /// Sets the dropout probability.
    pub fn with_drop_rate(mut self, drop_rate: f64) -> Self {
        self.drop_rate = drop_rate;
        self
    }
}

/// AlexNet.
#[derive(Module, Debug)]
pub struct AlexNet {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv4: Conv2d,
    conv5: Conv2d,
    pool1: MaxPool2d,
    pool2: MaxPool2d,
    pool3: MaxPool2d,
    global_pool: SelectAdaptivePool2d,
    drop: Dropout,
    fc1: Linear,
    fc2: Linear,
    fc3: Option<Linear>,
    #[module(skip)]
    meta: ModelMeta,
}

impl AlexNet {
    /// Instantiates an AlexNet.
    pub fn new(config: AlexNetConfig, device: &Device) -> Self {
        let conv1 = conv_layer(config.in_chans, 64, 11, 4, 2, 1, device);
        let conv2 = conv_layer(64, 192, 5, 1, 2, 1, device);
        let conv3 = conv_layer(192, 384, 3, 1, 1, 1, device);
        let conv4 = conv_layer(384, 256, 3, 1, 1, 1, device);
        let conv5 = conv_layer(256, 256, 3, 1, 1, 1, device);

        let maxpool = || MaxPool2dConfig::new([3, 3]).with_strides([2, 2]).init();

        let num_features = 4096;
        Self {
            conv1,
            conv2,
            conv3,
            conv4,
            pool1: maxpool(),
            pool2: maxpool(),
            pool3: maxpool(),
            conv5,
            global_pool: SelectAdaptivePool2d::new(PoolMode::Avg, [6, 6])
                .expect("6x6 average pooling is supported"),
            drop: DropoutConfig::new(config.drop_rate).init(),
            fc1: LinearConfig::new(256 * 6 * 6, num_features).init(device),
            fc2: LinearConfig::new(num_features, num_features).init(device),
            fc3: (config.num_classes > 0)
                .then(|| LinearConfig::new(num_features, config.num_classes).init(device)),
            meta: ModelMeta {
                name: "alexnet".to_string(),
                num_classes: config.num_classes,
                num_features,
                feature_info: vec![
                    FeatureInfo::new(0, 64, 8, "pool1"),
                    FeatureInfo::new(1, 192, 16, "pool2"),
                    FeatureInfo::new(2, 256, 32, "pool3"),
                ],
                pretrained_cfg: None,
            },
        }
    }

    /// Replaces model metadata (used by the registry).
    pub(crate) fn set_meta(&mut self, meta: ModelMeta) {
        self.meta = meta;
    }

    /// First convolution of the model.
    pub fn first_conv_mut(&mut self) -> &mut Conv2d {
        &mut self.conv1
    }

    /// Replaces the first convolution weight (used when adapting `in_chans`).
    pub fn set_first_conv_weight(&mut self, weight: Tensor<4>) {
        self.conv1.weight = Param::initialized(ParamId::new(), weight);
    }

    /// Weight-name rewrites from `timm`'s PyTorch state dict to this module hierarchy.
    pub fn key_remaps() -> Vec<(&'static str, &'static str)> {
        vec![
            ("^features\\.0\\.", "conv1."),
            ("^features\\.3\\.", "conv2."),
            ("^features\\.6\\.", "conv3."),
            ("^features\\.8\\.", "conv4."),
            ("^features\\.10\\.", "conv5."),
            ("^classifier\\.1\\.", "fc1."),
            ("^classifier\\.4\\.", "fc2."),
            ("^classifier\\.6\\.", "fc3."),
        ]
    }

    fn features_to_mid(&self, input: Tensor<4>) -> Tensor<4> {
        let x = self.pool1.forward(relu(self.conv1.forward(input)));
        self.pool2.forward(relu(self.conv2.forward(x)))
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

        let x = relu(self.conv1.forward(input));
        let x = self.pool1.forward(x);
        if selected.contains(&0) {
            intermediates.push(x.clone());
        }
        if options.stop_early && max_idx == 0 {
            return (x, convert(intermediates, options.format));
        }

        let x = relu(self.conv2.forward(x));
        let x = self.pool2.forward(x);
        if selected.contains(&1) {
            intermediates.push(x.clone());
        }
        if options.stop_early && max_idx == 1 {
            return (x, convert(intermediates, options.format));
        }

        let x = self
            .conv5
            .forward(relu(self.conv4.forward(relu(self.conv3.forward(x)))));
        let x = self.pool3.forward(relu(x));
        if selected.contains(&2) {
            intermediates.push(x.clone());
        }

        (x, convert(intermediates, options.format))
    }
}

fn convert(tensors: Vec<Tensor<4>>, format: FeatureFormat) -> Vec<Tensor<4>> {
    tensors
        .into_iter()
        .map(|tensor| match format {
            FeatureFormat::Nchw => tensor,
            FeatureFormat::Nhwc => tensor.permute([0, 2, 3, 1]),
        })
        .collect()
}

impl VisionModel for AlexNet {
    fn forward(&self, input: Tensor<4>) -> Tensor<2> {
        self.forward_head(self.forward_features(input), false)
    }

    fn forward_features(&self, input: Tensor<4>) -> Tensor<4> {
        let x = self.features_to_mid(input);
        let x = self
            .conv5
            .forward(relu(self.conv4.forward(relu(self.conv3.forward(x)))));
        self.pool3.forward(relu(x))
    }

    fn forward_head(&self, features: Tensor<4>, pre_logits: bool) -> Tensor<2> {
        let pooled: Tensor<2> = self.global_pool.forward(features).flatten(1, 3);

        let x = relu(self.fc1.forward(self.drop.forward(pooled)));
        let x = relu(self.fc2.forward(self.drop.forward(x)));
        match (&self.fc3, pre_logits) {
            (_, true) | (None, false) => x,
            (Some(fc3), false) => fc3.forward(x),
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

fn conv_layer(
    in_chans: usize,
    out_chans: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    groups: usize,
    device: &Device,
) -> Conv2d {
    Conv2dConfig::new([in_chans, out_chans], [kernel, kernel])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(
            padding, padding, padding, padding,
        ))
        .with_groups(groups)
        .init(device)
}
