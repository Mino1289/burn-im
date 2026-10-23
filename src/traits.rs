//! Shared model interface: feature metadata, intermediate extraction options and the
//! [`VisionModel`] contract.

use std::fmt;

use burn::tensor::Tensor;

use crate::hub::config::PretrainedCfg;

/// Describes one feature level produced by a backbone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureInfo {
    /// Index of the stage within the model.
    pub stage_idx: usize,
    /// Number of output channels.
    pub num_chs: usize,
    /// Spatial reduction factor relative to the input (2, 4, 8, 16, 32...).
    pub reduction: usize,
    /// Name of the module producing this feature level.
    pub module_name: &'static str,
}

impl FeatureInfo {
    /// Creates a new feature descriptor.
    pub const fn new(
        stage_idx: usize,
        num_chs: usize,
        reduction: usize,
        module_name: &'static str,
    ) -> Self {
        Self {
            stage_idx,
            num_chs,
            reduction,
            module_name,
        }
    }
}

/// Layout of the tensors returned by [`VisionModel::forward_intermediates`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FeatureFormat {
    /// Spatial layout `[batch, channels, height, width]` (convolutional default).
    #[default]
    Nchw,
    /// Channels-last layout `[batch, height, width, channels]`.
    Nhwc,
}

impl fmt::Display for FeatureFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nchw => f.write_str("NCHW"),
            Self::Nhwc => f.write_str("NHWC"),
        }
    }
}

/// Options controlling [`VisionModel::forward_intermediates`].
#[derive(Debug, Clone)]
pub struct IntermediatesOptions {
    /// Explicit list of feature levels to return. `None` returns every level.
    /// Indices refer to [`FeatureInfo::stage_idx`] positions in the model's feature list.
    pub indices: Option<Vec<usize>>,
    /// Stop the forward pass once the last requested level is computed.
    pub stop_early: bool,
    /// Apply the model's final normalization to compatible outputs (no-op for CNN families).
    pub norm: bool,
    /// Layout of the returned tensors.
    pub format: FeatureFormat,
}

impl Default for IntermediatesOptions {
    fn default() -> Self {
        Self {
            indices: None,
            stop_early: false,
            norm: true,
            format: FeatureFormat::Nchw,
        }
    }
}

impl IntermediatesOptions {
    /// Creates options returning every level, without early stopping.
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests a specific set of feature levels.
    pub fn with_indices(mut self, indices: impl IntoIterator<Item = usize>) -> Self {
        self.indices = Some(indices.into_iter().collect());
        self
    }

    /// Stops the forward pass after the last requested level.
    pub fn with_stop_early(mut self, stop_early: bool) -> Self {
        self.stop_early = stop_early;
        self
    }

    /// Selects the output layout.
    pub fn with_format(mut self, format: FeatureFormat) -> Self {
        self.format = format;
        self
    }

    /// Resolves the selected levels against a model exposing `len` levels.
    ///
    /// Returns the sorted, de-duplicated indices and the highest index to compute.
    pub fn resolve(&self, len: usize) -> crate::Result<(Vec<usize>, usize)> {
        let selected = match &self.indices {
            None => (0..len).collect::<Vec<_>>(),
            Some(indices) => {
                let mut indices = indices.clone();
                indices.sort_unstable();
                indices.dedup();
                for &idx in &indices {
                    if idx >= len {
                        return Err(crate::Error::InvalidConfiguration(format!(
                            "feature index {idx} out of range (model exposes {len} levels)"
                        )));
                    }
                }
                indices
            }
        };
        let max = selected.last().copied().unwrap_or(0);
        Ok((selected, max))
    }
}

/// The unified contract implemented by every vision backbone in this crate.
///
/// All methods are backend-agnostic: tensors carry their device, the backend is selected
/// through [`Tensor`] creation and [`Device`](burn::tensor::Device).
pub trait VisionModel: Send + Sync {
    /// Full forward pass: images `[B, C, H, W]` to logits `[B, num_classes]`.
    fn forward(&self, input: Tensor<4>) -> Tensor<2>;

    /// Backbone features before global pooling, in `[B, C, H, W]` layout.
    fn forward_features(&self, input: Tensor<4>) -> Tensor<4>;

    /// Classification head applied to `[B, C, H, W]` features.
    ///
    /// When `pre_logits` is true, returns the pooled embedding `[B, num_features]` instead of
    /// the classifier output.
    fn forward_head(&self, features: Tensor<4>, pre_logits: bool) -> Tensor<2>;

    /// Multi-scale feature maps, one per selected level.
    fn forward_intermediates(
        &self,
        input: Tensor<4>,
        options: &IntermediatesOptions,
    ) -> Vec<Tensor<4>>;

    /// Runs the backbone once, returning final features and the selected intermediates.
    fn forward_features_with_intermediates(
        &self,
        input: Tensor<4>,
        options: &IntermediatesOptions,
    ) -> (Tensor<4>, Vec<Tensor<4>>);

    /// Feature levels exposed by this model.
    fn feature_info(&self) -> &[FeatureInfo];

    /// Pretrained configuration the model was built from, when known.
    fn pretrained_cfg(&self) -> Option<&PretrainedCfg>;

    /// Number of classes of the active head (0 when the head was removed).
    fn num_classes(&self) -> usize;

    /// Dimension of the embedding produced before the final linear layer.
    fn num_features(&self) -> usize;

    /// Canonical `<arch>.<tag>` identifier this model was created from.
    fn model_name(&self) -> &str;

    /// Downcast hook used by the registry to reach the concrete family type.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}
