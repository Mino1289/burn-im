//! End-to-end weight loading from the Hugging Face Hub.
//!
//! These tests download real checkpoints and are therefore ignored by default:
//!
//! ```bash
//! cargo test --test weights -- --ignored --nocapture
//! ```

use burn::tensor::{Device, Tensor};
use burn_im::hub::Hub;
use burn_im::hub::loader::load_safetensors;
use burn_im::models::resnet::{BlockKind, ResNet, ResNetConfig};
use burn_im::traits::VisionModel;

fn device() -> Device {
    Device::flex()
}

fn load(arch: &str, tag: &str, config: ResNetConfig) -> burn_im::hub::loader::LoadReport {
    let repo = Hub::repo_id(arch, tag);
    let hub = Hub::new().unwrap();
    let path = hub.weights(&repo).unwrap();

    let remaps = ResNet::key_remaps(&config);
    let mut model = ResNet::new(config, &device());
    load_safetensors(&mut model, &path, &remaps).unwrap()
}

#[test]
#[ignore = "downloads weights from the Hugging Face Hub"]
fn loads_resnet50_a1_in1k() {
    let config = ResNetConfig::new([3, 4, 6, 3], [64, 128, 256, 512], BlockKind::Bottleneck);
    let report = load("resnet50", "a1_in1k", config);

    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert!(report.missing.is_empty(), "{:?}", report.missing);
    // 320 tensors in the checkpoint, 53 of which are PyTorch-only `num_batches_tracked` counters.
    assert_eq!(report.applied, 267);
    assert_eq!(report.applied + report.unused.len(), 320);
    assert!(
        report
            .unused
            .iter()
            .all(|name| name.ends_with("num_batches_tracked")),
        "unexpected unused tensors: {:?}",
        report.unused
    );
}

#[test]
#[ignore = "downloads weights from the Hugging Face Hub"]
fn loads_resnet50d_deep_stem_and_avg_down() {
    let config = ResNetConfig::new([3, 4, 6, 3], [64, 128, 256, 512], BlockKind::Bottleneck)
        .with_deep_stem()
        .with_avg_down();
    let report = load("resnet50d", "ra2_in1k", config);

    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert!(report.missing.is_empty(), "{:?}", report.missing);
    assert!(report.applied > 267, "applied: {}", report.applied);
}

#[test]
#[ignore = "downloads weights from the Hugging Face Hub"]
fn loads_resnext50_grouped_convolutions() {
    let config = ResNetConfig::new([3, 4, 6, 3], [64, 128, 256, 512], BlockKind::Bottleneck)
        .with_groups(32, 4);
    let report = load("resnext50_32x4d", "a1h_in1k", config);

    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert!(report.missing.is_empty(), "{:?}", report.missing);
}

#[test]
#[ignore = "downloads weights from the Hugging Face Hub"]
fn loads_resnet18_and_runs_inference() {
    let repo = Hub::repo_id("resnet18", "a1_in1k");
    let hub = Hub::new().unwrap();
    let path = hub.weights(&repo).unwrap();

    let config = ResNetConfig::new([2, 2, 2, 2], [64, 128, 256, 512], BlockKind::Basic);
    let remaps = ResNet::key_remaps(&config);
    let mut model = ResNet::new(config, &device());
    let report = load_safetensors(&mut model, &path, &remaps).unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert!(report.missing.is_empty(), "{:?}", report.missing);

    let logits = model.forward(Tensor::zeros([1, 3, 224, 224], &device()));
    assert_eq!(logits.dims(), [1, 1000]);
    assert!(
        logits
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite())
    );
}

#[test]
#[ignore = "downloads weights from the Hugging Face Hub"]
fn head_reset_keeps_fresh_classifier() {
    let repo = Hub::repo_id("resnet18", "a1_in1k");
    let hub = Hub::new().unwrap();
    let path = hub.weights(&repo).unwrap();

    let config =
        ResNetConfig::new([2, 2, 2, 2], [64, 128, 256, 512], BlockKind::Basic).with_num_classes(10);
    let remaps = ResNet::key_remaps(&config);
    let mut model = ResNet::new(config, &device());
    let report = load_safetensors(&mut model, &path, &remaps).unwrap();

    // The 1000-class checkpoint head is skipped, everything else is loaded.
    assert_eq!(report.errors.len(), 2, "{:?}", report.errors);
    assert!(
        report.errors.iter().all(|err| err.contains("head.fc")),
        "{:?}",
        report.errors
    );
    assert!(report.missing.is_empty());

    let logits = model.forward(Tensor::zeros([1, 3, 64, 64], &device()));
    assert_eq!(logits.dims(), [1, 10]);
}

#[test]
#[ignore = "downloads weights from the Hugging Face Hub"]
fn headless_pretrained_resnet50() {
    let options = burn_im::CreateModelOptions::pretrained().with_num_classes(0);
    let model = burn_im::create_model("resnet50.a1_in1k", &options, &device()).unwrap();
    assert_eq!(model.num_classes(), 0);

    let out = model.forward(Tensor::zeros([1, 3, 64, 64], &device()));
    assert_eq!(out.dims(), [1, 2048]);
}

#[test]
#[ignore = "downloads weights from the Hugging Face Hub"]
fn resized_head_pretrained_resnet50() {
    let options = burn_im::CreateModelOptions::pretrained().with_num_classes(10);
    let model = burn_im::create_model("resnet50.a1_in1k", &options, &device()).unwrap();
    assert_eq!(model.num_classes(), 10);

    let out = model.forward(Tensor::zeros([1, 3, 64, 64], &device()));
    assert_eq!(out.dims(), [1, 10]);
}

#[test]
#[ignore = "downloads weights from the Hugging Face Hub"]
fn one_channel_pretrained_resnet18() {
    let options = burn_im::CreateModelOptions::pretrained().with_in_chans(1);
    let model = burn_im::create_model("resnet18.a1_in1k", &options, &device()).unwrap();

    let out = model.forward(Tensor::zeros([1, 1, 64, 64], &device()));
    assert_eq!(out.dims(), [1, 1000]);
}
