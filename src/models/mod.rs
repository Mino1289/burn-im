//! Model families.

pub mod alexnet;
pub mod resnet;

use crate::hub::config::PretrainedCfg;
use crate::traits::FeatureInfo;

/// Metadata carried by every model instance.
#[derive(Debug, Clone)]
pub struct ModelMeta {
    /// Canonical `<arch>.<tag>` identifier.
    pub name: String,
    /// Number of classes of the active head.
    pub num_classes: usize,
    /// Embedding dimension before the final linear layer.
    pub num_features: usize,
    /// Feature levels exposed by the backbone.
    pub feature_info: Vec<FeatureInfo>,
    /// Pretrained recipe the model was instantiated from.
    pub pretrained_cfg: Option<PretrainedCfg>,
}
