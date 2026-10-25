//! Feature-extractor construction options: `features_only`, `output_stride` and lazy
//! parameter initialization.
//!
//! Weight loading is exercised separately in `registry.rs` and `golden.rs`; the tests here use
//! seeded initialization so two models can be built with identical weights without network access.

use burn::module::{Module, ModuleVisitor, Param};
use burn::prelude::*;
use burn::tensor::Tensor;
use burn_im::models::resnet::{ResNet, resnet18, resnet50};
use burn_im::traits::{IntermediatesOptions, VisionModel};
use burn_im::{CreateModelOptions, create_model};

fn device() -> Device {
    Device::flex()
}

/// Deterministic trigonometric input, same recipe as the golden suite.
fn input(device: &Device) -> Tensor<4> {
    let (h, w) = (64usize, 64usize);
    let mut data = Vec::with_capacity(3 * h * w);
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let xf = x as f32;
                let yf = y as f32;
                data.push(((xf * 0.1 + yf * 0.2 + c as f32).sin() + 0.5).abs());
            }
        }
    }
    Tensor::<4>::from_data(TensorData::new(data, [1, 3, h, w]), device)
}

fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor.into_data().as_slice::<f32>().unwrap().to_vec()
}

fn all_levels(len: usize) -> IntermediatesOptions {
    IntermediatesOptions::new().with_indices(0..len)
}

#[derive(Default)]
struct LazyParams {
    tensors: usize,
    pending: usize,
}

impl ModuleVisitor for LazyParams {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        self.tensors += 1;
        if !param.is_initialized() {
            self.pending += 1;
        }
    }
}

#[test]
fn construction_does_not_allocate_weights() {
    let device = device();
    let model = ResNet::new(resnet18(1000), &device);

    let mut lazy = LazyParams::default();
    model.visit(&mut lazy);
    assert!(
        lazy.pending > 0,
        "building a model must leave its weights lazily initialized"
    );
    assert_eq!(
        lazy.pending,
        lazy.tensors - 40,
        "only the batch-norm running statistics (20 norms x 2) are allocated eagerly"
    );

    // The first forward materializes the pending parameters.
    model.forward(input(&device));
    let mut after = LazyParams::default();
    model.visit(&mut after);
    assert_eq!(after.pending, 0);
}

#[test]
fn features_only_matches_full_model_intermediates() {
    // Copy the weights explicitly instead of relying on seeded initialization: the global RNG is
    // shared with the other tests, and a truncated model materializes fewer parameters.
    let device = device();
    let full = ResNet::new(resnet18(1000), &device);
    // `allow_unused`: the record holds the deeper stages the extractor does not build.
    let record = full.into_record().allow_unused(true);

    let expected = ResNet::new(resnet18(1000), &device)
        .load_record(record.clone())
        .forward_intermediates(input(&device), &all_levels(3));

    let truncated =
        ResNet::new(resnet18(0).with_features_only([0, 1, 2]), &device).load_record(record);
    let actual = truncated.forward_intermediates(input(&device), &IntermediatesOptions::new());

    assert_eq!(truncated.num_classes(), 0);
    assert_eq!(truncated.feature_info().len(), 3);

    assert_eq!(actual.len(), expected.len());
    for (level, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            values(actual.clone()),
            values(expected.clone()),
            "level {level} differs between the truncated and the full model"
        );
    }

    // Headless models return the pooled embedding instead of logits.
    let pooled = truncated.forward(input(&device));
    assert_eq!(pooled.dims(), [1, truncated.num_features()]);
}

#[test]
fn features_only_skips_deeper_stages() {
    let device = device();
    let params = |indices: &[usize]| {
        let config = resnet50(0).with_features_only(indices.iter().copied());
        ResNet::new(config, &device).num_params()
    };

    let full = ResNet::new(resnet50(1000), &device).num_params();
    let last = params(&[4]);
    let two = params(&[0, 1]);
    let four = params(&[0, 1, 2, 3]);

    assert!(last < full, "a headless model must be smaller");
    assert!(two < four, "keeping fewer levels must drop parameters");
    assert!(
        four < full,
        "an extractor without the classifier must be smaller than the full model"
    );
}

#[test]
fn output_stride_changes_the_reduction_of_the_deepest_stages() {
    let device = device();
    let reductions = |output_stride: usize| {
        let config = resnet50(1000).with_output_stride(output_stride);
        let model = ResNet::new(config, &device);
        let reductions: Vec<usize> = model.feature_info().iter().map(|f| f.reduction).collect();
        let dims: Vec<[usize; 4]> = model
            .forward_intermediates(input(&device), &IntermediatesOptions::new())
            .into_iter()
            .map(|feature| feature.dims())
            .collect();
        (reductions, dims)
    };

    assert_eq!(
        reductions(32),
        (
            vec![2, 4, 8, 16, 32],
            vec![
                [1, 64, 32, 32],
                [1, 256, 16, 16],
                [1, 512, 8, 8],
                [1, 1024, 4, 4],
                [1, 2048, 2, 2],
            ]
        )
    );
    assert_eq!(
        reductions(16),
        (
            vec![2, 4, 8, 16, 16],
            vec![
                [1, 64, 32, 32],
                [1, 256, 16, 16],
                [1, 512, 8, 8],
                [1, 1024, 4, 4],
                [1, 2048, 4, 4],
            ]
        )
    );
    assert_eq!(
        reductions(8),
        (
            vec![2, 4, 8, 8, 8],
            vec![
                [1, 64, 32, 32],
                [1, 256, 16, 16],
                [1, 512, 8, 8],
                [1, 1024, 8, 8],
                [1, 2048, 8, 8],
            ]
        )
    );
}

#[test]
fn basic_blocks_report_the_dilated_reductions() {
    let device = device();

    for (output_stride, reductions, dims) in [
        (32, vec![2, 4, 8, 16, 32], [1, 512, 2, 2]),
        (16, vec![2, 4, 8, 16, 16], [1, 512, 4, 4]),
        (8, vec![2, 4, 8, 8, 8], [1, 512, 8, 8]),
    ] {
        let model = ResNet::new(resnet18(1000).with_output_stride(output_stride), &device);
        let actual: Vec<usize> = model.feature_info().iter().map(|f| f.reduction).collect();
        assert_eq!(actual, reductions, "output stride {output_stride}");

        let features = model.forward_intermediates(input(&device), &IntermediatesOptions::new());
        assert_eq!(
            features.last().unwrap().dims(),
            dims,
            "output stride {output_stride}"
        );
    }
}

#[test]
fn output_stride_keeps_the_shared_stages_unchanged() {
    let device = device();
    let model = ResNet::new(resnet18(1000), &device);
    // Same weight-copy scheme as `features_only_matches_full_model_intermediates`.
    let record = model.into_record();
    let default_features = ResNet::new(resnet18(1000), &device)
        .load_record(record.clone())
        .forward_intermediates(input(&device), &all_levels(5));

    let dilated_features = ResNet::new(resnet18(1000).with_output_stride(16), &device)
        .load_record(record)
        .forward_intermediates(input(&device), &all_levels(5));

    for level in 0..4 {
        assert_eq!(
            values(default_features[level].clone()),
            values(dilated_features[level].clone()),
            "level {level} must not depend on the output stride"
        );
    }
}

#[test]
fn invalid_configurations_are_rejected() {
    let config = resnet18(1000).with_output_stride(24);
    assert!(config.validate().is_err());

    let config = resnet18(0).with_features_only([0, 5]);
    assert!(config.validate().is_err());

    let config = resnet18(0).with_features_only([]);
    assert!(config.validate().is_err());

    let mut config = resnet18(1000);
    config.out_indices = Some(vec![1]);
    assert!(
        config.validate().is_err(),
        "out_indices requires features_only"
    );

    let mut config = resnet18(1000).with_features_only([1]);
    config.num_classes = 10;
    assert!(
        config.validate().is_err(),
        "features_only removes the classifier"
    );
}

/// `Box<dyn VisionModel>` is not `Debug`, so `unwrap_err` cannot be used.
fn rejection(id: &str, options: &CreateModelOptions, device: &Device) -> String {
    match create_model(id, options, device) {
        Ok(_) => panic!("`{id}` must be rejected"),
        Err(error) => error.to_string(),
    }
}

#[test]
fn registry_rejects_unsupported_options() {
    let device = device();

    let message = rejection(
        "resnet18",
        &CreateModelOptions::default().with_naflex(true),
        &device,
    );
    assert!(message.contains("use_naflex"), "{message}");

    let message = rejection(
        "alexnet",
        &CreateModelOptions::default().with_features_only([1]),
        &device,
    );
    assert!(message.contains("features_only"), "{message}");

    let options = CreateModelOptions {
        out_indices: Some(vec![1]),
        ..Default::default()
    };
    let message = rejection("resnet18", &options, &device);
    assert!(message.contains("out_indices"), "{message}");

    let message = rejection(
        "resnet18",
        &CreateModelOptions::default()
            .with_features_only([1])
            .with_num_classes(10),
        &device,
    );
    assert!(message.contains("features_only"), "{message}");

    let message = rejection(
        "resnet18",
        &CreateModelOptions::default().with_output_stride(12),
        &device,
    );
    assert!(message.contains("output_stride"), "{message}");

    let message = rejection(
        "alexnet",
        &CreateModelOptions::default().with_output_stride(16),
        &device,
    );
    assert!(message.contains("output_stride"), "{message}");

    // `alexnet` has a fixed reduction: asking for its nominal stride stays valid.
    create_model(
        "alexnet",
        &CreateModelOptions::default().with_output_stride(32),
        &device,
    )
    .unwrap();
}

#[test]
fn registry_builds_a_truncated_extractor() {
    let device = device();
    let model = create_model(
        "resnet18",
        &CreateModelOptions::default().with_features_only([0, 1]),
        &device,
    )
    .unwrap();

    assert_eq!(model.num_classes(), 0);
    assert_eq!(model.feature_info().len(), 2);
    assert_eq!(
        model
            .forward_intermediates(input(&device), &IntermediatesOptions::new())
            .len(),
        2
    );
    assert_eq!(model.forward(input(&device)).dims(), [1, 64]);
}

#[test]
#[ignore = "downloads pretrained weights from the Hugging Face Hub"]
fn pretrained_features_only_matches_full_intermediates() {
    let name = "resnet50.a1_in1k";
    let device = device();
    let full = create_model(name, &CreateModelOptions::pretrained(), &device).unwrap();
    let input = input(&device);
    let expected = full.forward_intermediates(input.clone(), &all_levels(3));

    let extractor = create_model(
        name,
        &CreateModelOptions::pretrained().with_features_only([0, 1, 2]),
        &device,
    )
    .unwrap();
    assert_eq!(extractor.num_classes(), 0);
    let actual = extractor.forward_intermediates(input, &IntermediatesOptions::new());

    assert_eq!(actual.len(), expected.len());
    for (level, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        let (actual, expected) = (values(actual.clone()), values(expected.clone()));
        assert_eq!(actual.len(), expected.len(), "level {level} element count");
        let max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // Same checkpoint and same arithmetic: the shared levels are bit-identical.
        assert_eq!(max_abs, 0.0, "level {level} differs (max abs {max_abs:e})");
    }
}
