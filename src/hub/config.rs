//! Deserialization of the `config.json` files published alongside `timm` checkpoints on the
//! Hugging Face Hub.

use serde::{Deserialize, Serialize};

/// Interpolation algorithm used at inference time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Interpolation {
    /// Nearest neighbour.
    Nearest,
    /// Bilinear.
    Bilinear,
    /// Bicubic (timm's default for most pretrained recipes).
    #[default]
    Bicubic,
    /// Any other value published by timm.
    #[serde(other)]
    Other,
}

/// Pretrained recipe and preprocessing metadata for one `<arch>.<tag>` pair.
///
/// Mirrors the `pretrained_cfg` object of `timm`'s Hugging Face `config.json`. Deserialization is
/// tolerant: any missing field falls back to the ImageNet defaults below.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PretrainedCfg {
    /// Original weight URL recorded by timm.
    pub url: Option<String>,
    /// Weight file name recorded by timm.
    pub file: Option<String>,
    /// Recipe tag, e.g. `a1_in1k`.
    pub tag: Option<String>,
    /// Whether timm needs a custom loading routine.
    pub custom_load: bool,
    /// Expected input shape `[channels, height, width]`.
    pub input_size: [usize; 3],
    /// Input shape used for testing/validation.
    pub test_input_size: Option<[usize; 3]>,
    /// Whether the model only accepts a fixed input size.
    pub fixed_input_size: bool,
    /// Resize interpolation.
    pub interpolation: Interpolation,
    /// Center-crop ratio applied at inference.
    pub crop_pct: f64,
    /// Center-crop ratio used for validation.
    pub test_crop_pct: Option<f64>,
    /// Crop mode (`center`, `squash`, ...).
    pub crop_mode: String,
    /// Per-channel normalization mean.
    pub mean: [f64; 3],
    /// Per-channel normalization standard deviation.
    pub std: [f64; 3],
    /// Number of classes the checkpoint was trained for.
    pub num_classes: usize,
    /// Spatial size produced by the backbone before pooling.
    pub pool_size: Option<[usize; 2]>,
    /// Name of the first convolution, used when adapting input channels.
    pub first_conv: String,
    /// Name of the classifier layer, used for head filtering.
    pub classifier: String,
    /// Source repository of the recipe.
    pub origin_url: Option<String>,
    /// Associated paper identifiers.
    pub paper_ids: Option<String>,
}

impl Default for PretrainedCfg {
    fn default() -> Self {
        Self {
            url: None,
            file: None,
            tag: None,
            custom_load: false,
            input_size: [3, 224, 224],
            test_input_size: None,
            fixed_input_size: false,
            interpolation: Interpolation::Bicubic,
            crop_pct: 0.875,
            test_crop_pct: None,
            crop_mode: "center".to_string(),
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            num_classes: 1000,
            pool_size: Some([7, 7]),
            first_conv: "conv1".to_string(),
            classifier: "fc".to_string(),
            origin_url: None,
            paper_ids: None,
        }
    }
}

impl PretrainedCfg {
    /// Input resolution `(height, width)`.
    pub fn input_hw(&self) -> (usize, usize) {
        (self.input_size[1], self.input_size[2])
    }

    /// Number of input channels expected by the recipe.
    pub fn in_chans(&self) -> usize {
        self.input_size[0]
    }
}

fn default_num_classes() -> usize {
    1000
}

/// Top-level `config.json` published by the `timm` organization on the Hugging Face Hub.
#[derive(Debug, Clone, Deserialize)]
pub struct TimmHfConfig {
    /// Architecture name (`resnet50`, `alexnet`, ...).
    pub architecture: String,
    /// Number of classes of the released checkpoint.
    #[serde(default = "default_num_classes")]
    pub num_classes: usize,
    /// Embedding dimension before the classifier.
    #[serde(default)]
    pub num_features: Option<usize>,
    /// Pretrained recipe metadata.
    #[serde(default)]
    pub pretrained_cfg: PretrainedCfg,
}

impl TimmHfConfig {
    /// Parses a `config.json` payload.
    pub fn from_json(json: &str) -> crate::Result<Self> {
        serde_json::from_str(json).map_err(|err| crate::Error::ConfigParseError(err.to_string()))
    }

    /// Reads and parses a `config.json` file.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> crate::Result<Self> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json)
    }
}
