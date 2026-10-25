//! Static model registry: name parsing, discovery and dynamic instantiation.

use std::path::PathBuf;

use burn::prelude::*;

use crate::error::{Error, Result};
use crate::hub::config::PretrainedCfg;
use crate::hub::head_filter::{HeadAction, adapt_input_conv, head_action, target_num_classes};
use crate::hub::{Hub, loader};
use crate::models::ModelMeta;
use crate::models::alexnet::{AlexNet, AlexNetConfig};
use crate::models::resnet::{BlockKind, ResNet, ResNetConfig, StemKind};
use crate::traits::VisionModel;

/// Options for [`create_model`].
#[derive(Debug, Clone, Default)]
pub struct CreateModelOptions {
    /// Load the official pretrained weights from the Hugging Face Hub.
    pub pretrained: bool,
    /// Override the number of classes. `Some(0)` removes the classifier head.
    pub num_classes: Option<usize>,
    /// Override the number of input channels.
    pub in_chans: Option<usize>,
    /// Override the classifier dropout rate.
    pub drop_rate: Option<f64>,
    /// Override the stochastic-depth rate (currently unused by CNN families).
    pub drop_path_rate: Option<f64>,
    /// Load weights from a local `.safetensors` checkpoint instead of the Hub.
    pub checkpoint_path: Option<PathBuf>,
    /// Build a headless feature extractor that stops after the deepest requested level.
    pub features_only: bool,
    /// Feature levels kept by [`Self::features_only`]; defaults to the deepest level.
    pub out_indices: Option<Vec<usize>>,
    /// Spatial reduction of the deepest stage: `32` (default), `16` or `8`.
    pub output_stride: Option<usize>,
    /// Wrap the checkpoint in the `NaFlexViT` variable-resolution container.
    ///
    /// Accepted but not implemented yet: the registry rejects it with
    /// [`Error::InvalidConfiguration`] until the container lands.
    pub use_naflex: bool,
}

impl CreateModelOptions {
    /// Options requesting pretrained weights.
    pub fn pretrained() -> Self {
        Self {
            pretrained: true,
            ..Self::default()
        }
    }

    /// Sets the number of classes.
    pub fn with_num_classes(mut self, num_classes: usize) -> Self {
        self.num_classes = Some(num_classes);
        self
    }

    /// Sets the number of input channels.
    pub fn with_in_chans(mut self, in_chans: usize) -> Self {
        self.in_chans = Some(in_chans);
        self
    }

    /// Loads weights from a local checkpoint.
    pub fn with_checkpoint(mut self, path: impl Into<PathBuf>) -> Self {
        self.checkpoint_path = Some(path.into());
        self
    }

    /// Builds a headless extractor keeping only the levels in `out_indices`.
    pub fn with_features_only(mut self, out_indices: impl IntoIterator<Item = usize>) -> Self {
        self.features_only = true;
        self.out_indices = Some(out_indices.into_iter().collect());
        self
    }

    /// Sets the spatial reduction of the deepest stage (`32`, `16` or `8`).
    pub fn with_output_stride(mut self, output_stride: usize) -> Self {
        self.output_stride = Some(output_stride);
        self
    }

    /// Loads the checkpoint through the `NaFlexViT` container (not implemented yet).
    pub fn with_naflex(mut self, use_naflex: bool) -> Self {
        self.use_naflex = use_naflex;
        self
    }
}

/// Parsed model identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelId {
    /// Architecture name.
    pub arch: String,
    /// Recipe tag, if provided.
    pub tag: Option<String>,
    /// Full Hub repository id when the `hf-hub:` prefix was used.
    pub repo: Option<String>,
    /// Canonical `<arch>.<tag>` name.
    pub name: String,
}

/// Parses `<arch>`, `<arch>.<tag>` or `hf-hub:org/repo`.
///
/// For `hf-hub:` identifiers the architecture is taken from the repository name, which is how
/// `timm` repositories are named (`timm/resnet50.a1_in1k`).
pub fn split_model_name_tag(name: &str) -> Result<ModelId> {
    let name = name.trim();
    if name.is_empty() {
        return Err(Error::InvalidModelFormat(name.to_string()));
    }

    let (repo, spec) = match name.strip_prefix("hf-hub:") {
        Some(rest) if !rest.is_empty() => {
            let spec = rest.rsplit('/').next().unwrap_or(rest);
            (Some(rest.to_string()), spec)
        }
        Some(_) => return Err(Error::InvalidModelFormat(name.to_string())),
        None => (None, name),
    };

    let (arch, tag) = match spec.rsplit_once('.') {
        Some((arch, tag)) if !arch.is_empty() && !tag.is_empty() => {
            (arch.to_string(), Some(tag.to_string()))
        }
        Some(_) => return Err(Error::InvalidModelFormat(name.to_string())),
        None => (spec.to_string(), None),
    };

    let canonical = match &tag {
        Some(tag) => format!("{arch}.{tag}"),
        None => arch.clone(),
    };

    Ok(ModelId {
        arch,
        tag,
        repo,
        name: canonical,
    })
}

/// Architecture family of a registry entry.
#[derive(Debug, Clone, Copy)]
enum Family {
    ResNet(ResNetSpec),
    AlexNet,
}

#[derive(Debug, Clone, Copy)]
struct ResNetSpec {
    layers: [usize; 4],
    widths: [usize; 4],
    block: BlockKind,
    stem: StemKind,
    avg_down: bool,
    groups: usize,
    base_width: usize,
}

#[derive(Debug, Clone, Copy)]
struct ModelInfo {
    arch: &'static str,
    family: Family,
    default_tag: Option<&'static str>,
    tags: &'static [&'static str],
    num_features: usize,
}

const fn resnet(
    layers: [usize; 4],
    block: BlockKind,
    stem: StemKind,
    avg_down: bool,
    groups: usize,
    base_width: usize,
) -> Family {
    Family::ResNet(ResNetSpec {
        layers,
        widths: [64, 128, 256, 512],
        block,
        stem,
        avg_down,
        groups,
        base_width,
    })
}

static MODELS: &[ModelInfo] = &[
    ModelInfo {
        arch: "resnet18",
        family: resnet(
            [2, 2, 2, 2],
            BlockKind::Basic,
            StemKind::Standard,
            false,
            1,
            64,
        ),
        default_tag: Some("a1_in1k"),
        tags: &[
            "a1_in1k",
            "a2_in1k",
            "a3_in1k",
            "fb_ssl_yfcc100m_ft_in1k",
            "fb_swsl_ig1b_ft_in1k",
            "gluon_in1k",
            "tv_in1k",
        ],
        num_features: 512,
    },
    ModelInfo {
        arch: "resnet34",
        family: resnet(
            [3, 4, 6, 3],
            BlockKind::Basic,
            StemKind::Standard,
            false,
            1,
            64,
        ),
        default_tag: Some("a1_in1k"),
        tags: &[
            "a1_in1k",
            "a2_in1k",
            "a3_in1k",
            "bt_in1k",
            "gluon_in1k",
            "ra4_e3600_r224_in1k",
            "tv_in1k",
        ],
        num_features: 512,
    },
    ModelInfo {
        arch: "resnet50",
        family: resnet(
            [3, 4, 6, 3],
            BlockKind::Bottleneck,
            StemKind::Standard,
            false,
            1,
            64,
        ),
        default_tag: Some("a1_in1k"),
        tags: &[
            "a1_in1k",
            "a1h_in1k",
            "a2_in1k",
            "a3_in1k",
            "am_in1k",
            "b1k_in1k",
            "b2k_in1k",
            "bt_in1k",
            "c1_in1k",
            "c2_in1k",
            "d_in1k",
            "fb_ssl_yfcc100m_ft_in1k",
            "fb_swsl_ig1b_ft_in1k",
            "gluon_in1k",
            "ra_in1k",
            "ram_in1k",
            "tv2_in1k",
            "tv_in1k",
        ],
        num_features: 2048,
    },
    ModelInfo {
        arch: "resnet101",
        family: resnet(
            [3, 4, 23, 3],
            BlockKind::Bottleneck,
            StemKind::Standard,
            false,
            1,
            64,
        ),
        default_tag: Some("a1_in1k"),
        tags: &[
            "a1_in1k",
            "a1h_in1k",
            "a2_in1k",
            "a3_in1k",
            "gluon_in1k",
            "tv2_in1k",
            "tv_in1k",
        ],
        num_features: 2048,
    },
    ModelInfo {
        arch: "resnet152",
        family: resnet(
            [3, 8, 36, 3],
            BlockKind::Bottleneck,
            StemKind::Standard,
            false,
            1,
            64,
        ),
        default_tag: Some("a1_in1k"),
        tags: &[
            "a1_in1k",
            "a1h_in1k",
            "a2_in1k",
            "a3_in1k",
            "gluon_in1k",
            "tv2_in1k",
            "tv_in1k",
        ],
        num_features: 2048,
    },
    ModelInfo {
        arch: "resnet18d",
        family: resnet([2, 2, 2, 2], BlockKind::Basic, StemKind::Deep, true, 1, 64),
        default_tag: Some("ra2_in1k"),
        tags: &["ra2_in1k", "ra4_e3600_r224_in1k"],
        num_features: 512,
    },
    ModelInfo {
        arch: "resnet34d",
        family: resnet([3, 4, 6, 3], BlockKind::Basic, StemKind::Deep, true, 1, 64),
        default_tag: Some("ra2_in1k"),
        tags: &["ra2_in1k"],
        num_features: 512,
    },
    ModelInfo {
        arch: "resnet50d",
        family: resnet(
            [3, 4, 6, 3],
            BlockKind::Bottleneck,
            StemKind::Deep,
            true,
            1,
            64,
        ),
        default_tag: Some("ra2_in1k"),
        tags: &[
            "a1_in1k",
            "a2_in1k",
            "a3_in1k",
            "gluon_in1k",
            "ra2_in1k",
            "ra4_e3600_r224_in1k",
        ],
        num_features: 2048,
    },
    ModelInfo {
        arch: "resnet101d",
        family: resnet(
            [3, 4, 23, 3],
            BlockKind::Bottleneck,
            StemKind::Deep,
            true,
            1,
            64,
        ),
        default_tag: Some("ra2_in1k"),
        tags: &["gluon_in1k", "ra2_in1k"],
        num_features: 2048,
    },
    ModelInfo {
        arch: "resnet152d",
        family: resnet(
            [3, 8, 36, 3],
            BlockKind::Bottleneck,
            StemKind::Deep,
            true,
            1,
            64,
        ),
        default_tag: Some("ra2_in1k"),
        tags: &["gluon_in1k", "ra2_in1k"],
        num_features: 2048,
    },
    ModelInfo {
        arch: "resnext50_32x4d",
        family: resnet(
            [3, 4, 6, 3],
            BlockKind::Bottleneck,
            StemKind::Standard,
            false,
            32,
            4,
        ),
        default_tag: Some("a1h_in1k"),
        tags: &[
            "a1_in1k",
            "a1h_in1k",
            "a2_in1k",
            "a3_in1k",
            "gluon_in1k",
            "ra_in1k",
            "tv2_in1k",
            "tv_in1k",
        ],
        num_features: 2048,
    },
    ModelInfo {
        arch: "resnext101_32x8d",
        family: resnet(
            [3, 4, 23, 3],
            BlockKind::Bottleneck,
            StemKind::Standard,
            false,
            32,
            8,
        ),
        default_tag: Some("tv_in1k"),
        tags: &[
            "fb_ssl_yfcc100m_ft_in1k",
            "fb_swsl_ig1b_ft_in1k",
            "fb_wsl_ig1b_ft_in1k",
            "tv2_in1k",
            "tv_in1k",
        ],
        num_features: 2048,
    },
    ModelInfo {
        arch: "wide_resnet50_2",
        family: resnet(
            [3, 4, 6, 3],
            BlockKind::Bottleneck,
            StemKind::Standard,
            false,
            1,
            128,
        ),
        default_tag: Some("tv_in1k"),
        tags: &["racm_in1k", "tv2_in1k", "tv_in1k"],
        num_features: 2048,
    },
    ModelInfo {
        arch: "wide_resnet101_2",
        family: resnet(
            [3, 4, 23, 3],
            BlockKind::Bottleneck,
            StemKind::Standard,
            false,
            1,
            128,
        ),
        default_tag: Some("tv_in1k"),
        tags: &["tv2_in1k", "tv_in1k"],
        num_features: 2048,
    },
    ModelInfo {
        arch: "alexnet",
        family: Family::AlexNet,
        default_tag: None,
        tags: &[],
        num_features: 4096,
    },
];

fn find(arch: &str) -> Option<&'static ModelInfo> {
    MODELS.iter().find(|info| info.arch == arch)
}

fn glob_match(pattern: &str, value: &str) -> bool {
    // `*` glob with backtracking, supporting several wildcards (`*resnet*`, `resnet5?` no).
    let pattern: Vec<char> = pattern.chars().collect();
    let value: Vec<char> = value.chars().collect();
    let (mut p, mut v) = (0, 0);
    let mut star: Option<(usize, usize)> = None;

    while v < value.len() {
        if p < pattern.len() && pattern[p] == value[v] {
            p += 1;
            v += 1;
        } else if p < pattern.len() && pattern[p] == '*' {
            star = Some((p, v));
            p += 1;
        } else if let Some((star_p, star_v)) = star {
            p = star_p + 1;
            v = star_v + 1;
            star = Some((star_p, star_v + 1));
        } else {
            return false;
        }
    }

    pattern[p..].iter().all(|c| *c == '*')
}

/// Lists registered architectures, optionally filtered by a `*` glob.
pub fn list_models(filter: Option<&str>) -> Vec<&'static str> {
    MODELS
        .iter()
        .filter(|info| filter.is_none_or(|pattern| glob_match(pattern, info.arch)))
        .map(|info| info.arch)
        .collect()
}

/// Lists known `<arch>.<tag>` pairs, optionally filtered by a `*` glob.
pub fn list_pretrained(filter: Option<&str>) -> Vec<String> {
    let mut models = Vec::new();
    for info in MODELS {
        for tag in info.tags {
            let name = format!("{}.{}", info.arch, tag);
            if filter.is_none_or(|pattern| glob_match(pattern, &name)) {
                models.push(name);
            }
        }
    }
    models
}

/// Whether an architecture is registered.
pub fn is_model(name: &str) -> bool {
    let name = name.strip_prefix("hf-hub:").unwrap_or(name);
    let arch = name.rsplit('/').next().unwrap_or(name);
    let arch = arch.split('.').next().unwrap_or(arch);
    find(arch).is_some()
}

/// Retrieves the pretrained recipe of `<arch>.<tag>` without downloading weights.
pub fn get_pretrained_cfg(name: &str) -> Result<PretrainedCfg> {
    let id = split_model_name_tag(name)?;
    let info = find(&id.arch).ok_or_else(|| Error::UnknownModel(id.arch.clone()))?;
    let tag = resolve_tag(&id, info)?;
    let repo = id
        .repo
        .clone()
        .unwrap_or_else(|| Hub::repo_id(&id.arch, &tag));

    let hub = Hub::new()?;
    let mut cfg = hub.config(&repo)?.pretrained_cfg;
    cfg.tag.get_or_insert(tag);
    Ok(cfg)
}

fn resolve_tag(id: &ModelId, info: &ModelInfo) -> Result<String> {
    match (&id.tag, info.default_tag) {
        (Some(tag), _) => Ok(tag.clone()),
        (None, Some(tag)) => Ok(tag.to_string()),
        (None, None) => Err(Error::UnknownTag {
            arch: info.arch.to_string(),
            tag: "<none>".to_string(),
            available: "none (no pretrained weights published on the Hub)".to_string(),
        }),
    }
}

/// Creates a model by name.
///
/// `name` accepts `<arch>`, `<arch>.<tag>` or `hf-hub:org/repo`. Pretrained weights are
/// downloaded from the Hub when [`CreateModelOptions::pretrained`] is set, or read from
/// [`CreateModelOptions::checkpoint_path`] when provided.
pub fn create_model(
    name: &str,
    options: &CreateModelOptions,
    device: &Device,
) -> Result<Box<dyn VisionModel>> {
    let id = split_model_name_tag(name)?;
    let info = find(&id.arch).ok_or_else(|| Error::UnknownModel(id.arch.clone()))?;

    let tag = match (&id.tag, info.default_tag) {
        (Some(tag), _) => Some(tag.clone()),
        (None, Some(tag)) => Some(tag.to_string()),
        (None, None) if options.pretrained => {
            return Err(Error::UnknownTag {
                arch: id.arch.clone(),
                tag: "<none>".to_string(),
                available: "none (no pretrained weights published on the Hub)".to_string(),
            });
        }
        (None, None) => None,
    };

    let repo = id
        .repo
        .clone()
        .or_else(|| tag.as_ref().map(|tag| Hub::repo_id(&id.arch, tag)));

    // Resolve the recipe: from the Hub when pretrained weights are requested, otherwise defaults.
    let (pretrained_cfg, hub_config) = match (&repo, options.pretrained) {
        (Some(repo), true) => {
            let hub = Hub::new()?;
            let config = hub.config(repo)?;
            (Some(config.pretrained_cfg.clone()), Some(config))
        }
        _ => (None, None),
    };

    let pretrained_classes = hub_config
        .as_ref()
        .map(|config| config.num_classes)
        .or_else(|| pretrained_cfg.as_ref().map(|cfg| cfg.num_classes))
        .unwrap_or(1000);
    let head = if options.features_only {
        HeadAction::Remove
    } else {
        head_action(pretrained_classes, options.num_classes)
    };
    let num_classes = if options.features_only {
        0
    } else {
        target_num_classes(pretrained_classes, options.num_classes)
    };
    let in_chans = options
        .in_chans
        .or_else(|| pretrained_cfg.as_ref().map(|cfg| cfg.in_chans()))
        .unwrap_or(3);

    if options.use_naflex {
        return Err(Error::InvalidConfiguration(
            "`use_naflex` requires the NaFlexViT container, which is not implemented yet"
                .to_string(),
        ));
    }
    if options.features_only {
        if options.num_classes.unwrap_or(0) != 0 {
            return Err(Error::InvalidConfiguration(
                "`features_only` removes the classifier (`num_classes` must be 0 or unset)"
                    .to_string(),
            ));
        }
        if !matches!(info.family, Family::ResNet(_)) {
            return Err(Error::InvalidConfiguration(format!(
                "`features_only` is not supported by architecture `{}`",
                id.arch
            )));
        }
    } else if options.out_indices.is_some() {
        return Err(Error::InvalidConfiguration(
            "`out_indices` requires `features_only`".to_string(),
        ));
    }

    // Families that cannot fold their stride into dilation must say so instead of ignoring the
    // option; `output_stride = 32` is their nominal reduction, so it remains accepted.
    if !matches!(info.family, Family::ResNet(_))
        && let Some(output_stride) = options.output_stride.filter(|stride| *stride != 32)
    {
        return Err(Error::InvalidConfiguration(format!(
            "`output_stride` {output_stride} is not supported by architecture `{}`",
            id.arch
        )));
    }

    let mut model: Box<dyn VisionModel> = match info.family {
        Family::ResNet(spec) => {
            let mut config = ResNetConfig::new(spec.layers, spec.widths, spec.block)
                .with_groups(spec.groups, spec.base_width)
                .with_num_classes(num_classes)
                .with_in_chans(in_chans);
            if spec.stem == StemKind::Deep {
                config = config.with_deep_stem();
            }
            if spec.avg_down {
                config = config.with_avg_down();
            }
            if let Some(drop_rate) = options.drop_rate {
                config = config.with_drop_rate(drop_rate);
            }
            if let Some(output_stride) = options.output_stride {
                config = config.with_output_stride(output_stride);
            }
            if options.features_only {
                config = config
                    .with_features_only(options.out_indices.clone().unwrap_or_else(|| vec![4]));
            }
            config.validate()?;

            let mut resnet = ResNet::new(config.clone(), device);
            let feature_info = resnet.feature_info().to_vec();
            let num_features = resnet.num_features();
            resnet.set_meta(ModelMeta {
                name: id.name.clone(),
                num_classes,
                num_features,
                feature_info,
                pretrained_cfg: pretrained_cfg.clone(),
            });
            Box::new(resnet)
        }
        Family::AlexNet => {
            let mut config = AlexNetConfig::new(num_classes).with_in_chans(in_chans);
            if let Some(drop_rate) = options.drop_rate {
                config = config.with_drop_rate(drop_rate);
            }
            let mut alexnet = AlexNet::new(config, device);
            let feature_info = alexnet.feature_info().to_vec();
            alexnet.set_meta(ModelMeta {
                name: id.name.clone(),
                num_classes,
                num_features: info.num_features,
                feature_info,
                pretrained_cfg: pretrained_cfg.clone(),
            });
            Box::new(alexnet)
        }
    };

    if let Some(path) = &options.checkpoint_path {
        load_into(&mut *model, name, path, info, head, in_chans, device)?;
    } else if options.pretrained {
        let repo = repo.ok_or_else(|| Error::UnknownTag {
            arch: id.arch.clone(),
            tag: "<none>".to_string(),
            available: "none".to_string(),
        })?;
        let hub = Hub::new()?;
        let path = hub.weights(&repo)?;
        load_into(&mut *model, name, &path, info, head, in_chans, device)?;
    }

    Ok(model)
}

fn load_into(
    model: &mut dyn VisionModel,
    name: &str,
    path: &std::path::Path,
    info: &ModelInfo,
    head: HeadAction,
    in_chans: usize,
    device: &Device,
) -> Result<()> {
    let remaps = match info.family {
        Family::ResNet(spec) => {
            let mut config = ResNetConfig::new(spec.layers, spec.widths, spec.block)
                .with_groups(spec.groups, spec.base_width);
            if spec.stem == StemKind::Deep {
                config = config.with_deep_stem();
            }
            if spec.avg_down {
                config = config.with_avg_down();
            }
            ResNet::key_remaps(&config)
        }
        Family::AlexNet => AlexNet::key_remaps(),
    };

    // `load_safetensors` is generic over `Module`, so dispatch on the concrete family.
    match info.family {
        Family::ResNet(_) => {
            let model = model
                .as_any_mut()
                .downcast_mut::<ResNet>()
                .ok_or_else(|| Error::WeightLoadError("unexpected model type".to_string()))?;
            let report = loader::load_safetensors(model, path, &remaps)?;
            if in_chans != 3 {
                let weight = loader::read_tensor::<4>(path, first_conv_key(info.family), device)?;
                model.set_first_conv_weight(adapt_input_conv(weight, in_chans));
            }
            check_report(&report, head, in_chans, name)
        }
        Family::AlexNet => {
            let model = model
                .as_any_mut()
                .downcast_mut::<AlexNet>()
                .ok_or_else(|| Error::WeightLoadError("unexpected model type".to_string()))?;
            let report = loader::load_safetensors(model, path, &remaps)?;
            if in_chans != 3 {
                let weight = loader::read_tensor::<4>(path, first_conv_key(info.family), device)?;
                model.set_first_conv_weight(adapt_input_conv(weight, in_chans));
            }
            check_report(&report, head, in_chans, name)
        }
    }
}

/// Checkpoint tensor name of the first convolution, before remapping.
fn first_conv_key(family: Family) -> &'static str {
    match family {
        Family::ResNet(spec) if spec.stem == StemKind::Deep => "conv1.0.weight",
        Family::ResNet(_) => "conv1.weight",
        Family::AlexNet => "features.0.weight",
    }
}

fn check_report(
    report: &loader::LoadReport,
    head: HeadAction,
    in_chans: usize,
    name: &str,
) -> Result<()> {
    let head_error = |message: &str| {
        message.contains("head.fc")
            || message.contains("fc1")
            || message.contains("fc2")
            || message.contains("fc3")
    };
    // With a non-default `in_chans` the first convolution is adapted after loading and its
    // 3-channel checkpoint tensor is expected to mismatch.
    let conv_error = |message: &str| in_chans != 3 && message.contains("conv1");

    let unexpected: Vec<&String> = report
        .errors
        .iter()
        .filter(|message| {
            let allowed = (head != HeadAction::Keep && head_error(message)) || conv_error(message);
            !allowed
        })
        .collect();

    if !unexpected.is_empty() {
        return Err(Error::WeightLoadError(format!(
            "shape mismatches while loading `{name}`: {unexpected:?}"
        )));
    }
    Ok(())
}
