//! Numerical parity against `timm` reference outputs.
//!
//! Fixtures under `tests/fixtures/golden/` are produced by
//! `scripts/generate_fixtures.py` with the real pretrained weights. These tests therefore need
//! network access (for the checkpoint) and are ignored by default:
//!
//! ```bash
//! cargo test --test golden -- --ignored --nocapture
//! ```

use std::path::Path;

use burn::tensor::{Device, Tensor, TensorData};
use burn_im::traits::{IntermediatesOptions, VisionModel};
use burn_im::{CreateModelOptions, create_model};

/// Maximum absolute difference allowed against the timm/torchvision reference.
///
/// `10e-5` is `1e-4`: the f32 accumulation noise floor of these networks.
const TOLERANCE: f32 = 10e-5;

fn device() -> Device {
    Device::flex()
}

fn load_tensor<const D: usize>(path: &Path, name: &str, device: &Device) -> Tensor<D> {
    let bytes = std::fs::read(path).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let tensor = tensors.tensor(name).unwrap();
    let values: Vec<f32> = tensor
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let shape: [usize; D] = tensor.shape().try_into().unwrap();
    Tensor::from_data(TensorData::new(values, shape), device)
}

fn has_tensor(path: &Path, name: &str) -> bool {
    let bytes = std::fs::read(path).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    tensors.tensor(name).is_ok()
}

fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor.into_data().as_slice::<f32>().unwrap().to_vec()
}

fn assert_close(actual: &[f32], expected: &[f32], what: &str) -> f32 {
    assert_eq!(actual.len(), expected.len(), "{what}: element count");
    let max = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max < TOLERANCE,
        "{what}: max abs diff {max:.3e} exceeds {TOLERANCE:.0e}"
    );
    max
}

fn check(arch: &str, tag: &str) {
    let name = format!("{arch}.{tag}");
    let path = Path::new("tests/fixtures/golden").join(format!("{name}.safetensors"));
    if !path.exists() {
        panic!(
            "missing fixture {}; run scripts/generate_fixtures.py",
            path.display()
        );
    }

    let model = create_model(&name, &CreateModelOptions::pretrained(), &device()).unwrap();
    compare(&*model, &name, &path);
}

fn compare(model: &dyn VisionModel, name: &str, path: &Path) {
    let input: Tensor<4> = load_tensor(path, "input", &device());
    let expected_logits: Tensor<2> = load_tensor(path, "logits", &device());
    let expected_features: Tensor<4> = load_tensor(path, "features", &device());

    let logits = model.forward(input.clone());
    let logits_diff = assert_close(
        &values(logits.clone()),
        &values(expected_logits),
        &format!("{name} logits"),
    );

    let features = model.forward_features(input.clone());
    let features_diff = assert_close(
        &values(features),
        &values(expected_features),
        &format!("{name} features"),
    );

    if has_tensor(path, "pre_logits") {
        let expected: Tensor<2> = load_tensor(path, "pre_logits", &device());
        let actual = model.forward_head(model.forward_features(input.clone()), true);
        assert_close(
            &values(actual),
            &values(expected),
            &format!("{name} pre_logits"),
        );
    }

    let intermediates = model.forward_intermediates(input, &IntermediatesOptions::new());
    assert_eq!(intermediates.len(), model.feature_info().len());
    let mut level_diff = 0.0_f32;
    for (idx, level) in intermediates.into_iter().enumerate() {
        let expected: Tensor<4> = load_tensor(path, &format!("level_{idx}"), &device());
        level_diff = level_diff.max(assert_close(
            &values(level),
            &values(expected),
            &format!("{name} level {idx}"),
        ));
    }

    println!(
        "{name}: logits {logits_diff:.3e}, features {features_diff:.3e}, levels {level_diff:.3e}"
    );
}

#[test]
#[ignore = "requires target/fixtures/alexnet.safetensors from scripts/generate_fixtures.py"]
fn alexnet_matches_torchvision() {
    let weights = Path::new("target/fixtures/alexnet.safetensors");
    if !weights.exists() {
        eprintln!(
            "skipping: {} missing, run scripts/generate_fixtures.py",
            weights.display()
        );
        return;
    }

    let name = "alexnet.tv_in1k";
    let path = Path::new("tests/fixtures/golden").join(format!("{name}.safetensors"));
    let options = CreateModelOptions::default().with_checkpoint(weights);
    let model = create_model("alexnet", &options, &device()).unwrap();
    assert_eq!(model.num_classes(), 1000);
    compare(&*model, name, &path);
}

#[test]
#[ignore = "downloads pretrained weights from the Hugging Face Hub"]
fn resnet18_matches_timm() {
    check("resnet18", "a1_in1k");
}

#[test]
#[ignore = "downloads pretrained weights from the Hugging Face Hub"]
fn resnet18_in_chans_adaptation_matches_timm() {
    let path = Path::new("tests/fixtures/golden/resnet18.a1_in1k.safetensors");
    let options = CreateModelOptions::pretrained().with_in_chans(1);
    let model = create_model("resnet18.a1_in1k", &options, &device()).unwrap();
    assert_eq!(model.num_classes(), 1000);

    let input: Tensor<4> = load_tensor(path, "input_1ch", &device());
    let expected: Tensor<2> = load_tensor(path, "logits_1ch", &device());
    let diff = assert_close(
        &values(model.forward(input)),
        &values(expected),
        "resnet18 1-channel logits",
    );
    println!("resnet18 in_chans=1: logits {diff:.3e}");
}

#[test]
#[ignore = "downloads pretrained weights from the Hugging Face Hub"]
fn resnet50_matches_timm() {
    check("resnet50", "a1_in1k");
}

#[test]
#[ignore = "downloads pretrained weights from the Hugging Face Hub"]
fn resnet50d_matches_timm() {
    check("resnet50d", "ra2_in1k");
}

#[test]
#[ignore = "downloads pretrained weights from the Hugging Face Hub"]
fn resnext50_matches_timm() {
    check("resnext50_32x4d", "a1h_in1k");
}
