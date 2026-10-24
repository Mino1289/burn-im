//! Conversion of PyTorch `.safetensors` checkpoints into `burn-im` modules.
//!
//! Tensors are matched by name against the Rust field hierarchy. Families provide a small set
//! of regular-expression rewrites for the places where PyTorch's `nn.Sequential` indices do not
//! map onto named Rust fields (for example `downsample.0` -> `downsample.conv`).

use std::path::Path;

use burn::module::Module;
use burn::tensor::{DType, Float, Tensor, TensorData};
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

use crate::error::{Error, Result};

/// Outcome of a weight loading operation.
#[derive(Debug, Clone, Default)]
pub struct LoadReport {
    /// Names of the tensors that were applied to the module.
    pub applied: usize,
    /// Module parameters that were not found in the checkpoint.
    pub missing: Vec<String>,
    /// Checkpoint tensors that matched no module parameter.
    pub unused: Vec<String>,
    /// Tensors rejected because of a shape or dtype mismatch.
    pub errors: Vec<String>,
}

impl LoadReport {
    /// Whether every module parameter was found and no tensor failed to load.
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty() && self.errors.is_empty()
    }
}

/// Loads a `.safetensors` checkpoint into `model`.
///
/// * Linear weights are transposed and normalization parameters renamed by
///   [`PyTorchToBurnAdapter`].
/// * Shape mismatches are recorded in [`LoadReport::errors`] instead of aborting, which is how a
///   classifier head with a different `num_classes` (or a first convolution with a different
///   `in_chans`) keeps its fresh initialization.
/// * `num_batches_tracked` counters published by PyTorch are ignored.
pub fn load_safetensors<M: Module>(
    model: &mut M,
    path: &Path,
    remaps: &[(&str, &str)],
) -> Result<LoadReport> {
    let mut store = SafetensorsStore::from_file(path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .allow_partial(true)
        .validate(false)
        .skip_enum_variants(true);

    for (from, to) in remaps {
        store = store.with_key_remapping(*from, *to);
    }

    let result = model
        .load_from(&mut store)
        .map_err(|err| Error::WeightLoadError(format!("{}: {err}", path.display())))?;

    let report = LoadReport {
        applied: result.applied.len(),
        missing: result.missing.into_iter().map(|(name, _)| name).collect(),
        unused: result.unused,
        errors: result.errors.iter().map(|err| err.to_string()).collect(),
    };

    if report.applied == 0 {
        return Err(Error::WeightLoadError(format!(
            "no checkpoint tensor matched the model in {}",
            path.display()
        )));
    }

    Ok(report)
}

/// Reads one tensor from a `.safetensors` checkpoint.
///
/// Used for transformations that change a parameter's shape (for example adapting the first
/// convolution to a different number of input channels), which `burn-store` cannot express as a
/// key remap because it validates shapes before applying parameter mappers.
pub fn read_tensor<const D: usize>(
    path: &Path,
    name: &str,
    device: &burn::tensor::Device,
) -> Result<Tensor<D>> {
    let bytes = std::fs::read(path)?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes)
        .map_err(|err| Error::WeightLoadError(format!("{}: {err}", path.display())))?;
    let tensor = tensors.tensor(name).map_err(|err| {
        Error::WeightLoadError(format!("{}: tensor `{name}`: {err}", path.display()))
    })?;

    let dtype = match tensor.dtype() {
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::BF16 => DType::BF16,
        other => {
            return Err(Error::WeightLoadError(format!(
                "{}: tensor `{name}` has unsupported dtype {other:?}",
                path.display()
            )));
        }
    };
    let shape: [usize; D] = tensor.shape().try_into().map_err(|_| {
        Error::WeightLoadError(format!(
            "{}: tensor `{name}` has rank {}, expected {D}",
            path.display(),
            tensor.shape().len()
        ))
    })?;
    let data = TensorData::from_bytes_vec(tensor.data().to_vec(), shape, dtype);
    Ok(Tensor::<D, Float>::from_data(data, device).cast(DType::F32))
}
