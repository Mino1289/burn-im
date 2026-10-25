//! Feature extraction and weight-name contract tests.

use burn::tensor::{Device, Tensor};
use burn_im::models::resnet::{BlockKind, ResNet, ResNetConfig, StemKind, resnet50};
use burn_im::traits::{FeatureFormat, IntermediatesOptions, VisionModel};
use burn_im::{CreateModelOptions, create_model};
use burn_store::ModuleSnapshot;

fn device() -> Device {
    Device::flex()
}

fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor.into_data().as_slice::<f32>().unwrap().to_vec()
}

#[test]
fn resnet_feature_levels_match_timm() {
    let model = resnet50(1000);
    let model = ResNet::new(model, &device());

    let info = model.feature_info();
    let reductions: Vec<usize> = info.iter().map(|f| f.reduction).collect();
    let channels: Vec<usize> = info.iter().map(|f| f.num_chs).collect();
    assert_eq!(reductions, vec![2, 4, 8, 16, 32]);
    assert_eq!(channels, vec![64, 256, 512, 1024, 2048]);

    let input = Tensor::zeros([1, 3, 64, 64], &device());
    let feats = model.forward_intermediates(input, &IntermediatesOptions::new());
    assert_eq!(feats.len(), 5);
    assert_eq!(feats[0].dims(), [1, 64, 32, 32]);
    assert_eq!(feats[1].dims(), [1, 256, 16, 16]);
    assert_eq!(feats[4].dims(), [1, 2048, 2, 2]);
}

#[test]
fn stop_early_preserves_values() {
    let model = ResNet::new(resnet50(1000), &device());
    let input = Tensor::random(
        [1, 3, 64, 64],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device(),
    );

    let options = IntermediatesOptions::new().with_indices([1, 2, 3]);
    let full = model.forward_intermediates(input.clone(), &options);
    let early = model.forward_intermediates(input, &options.clone().with_stop_early(true));

    assert_eq!(full.len(), early.len());
    for (a, b) in full.into_iter().zip(early) {
        assert_eq!(a.dims(), b.dims());
        assert_eq!(values(a), values(b));
    }
}

#[test]
fn supports_sequential_feature_selection() {
    let model = ResNet::new(resnet50(1000), &device());
    let input = Tensor::zeros([1, 3, 64, 64], &device());

    let feats = model.forward_intermediates(
        input.clone(),
        &IntermediatesOptions::new().with_indices([3, 1]),
    );
    assert_eq!(feats.len(), 2);
    assert_eq!(feats[0].dims(), [1, 256, 16, 16]);
    assert_eq!(feats[1].dims(), [1, 1024, 4, 4]);

    let nhwc = model.forward_intermediates(
        input,
        &IntermediatesOptions::new()
            .with_indices([1])
            .with_format(FeatureFormat::Nhwc),
    );
    assert_eq!(nhwc[0].dims(), [1, 16, 16, 256]);
}

#[test]
fn backward_features_match_final_intermediate() {
    let model = ResNet::new(resnet50(1000), &device());
    let input = Tensor::zeros([2, 3, 64, 64], &device());

    let (features, feats) =
        model.forward_features_with_intermediates(input.clone(), &IntermediatesOptions::new());
    assert_eq!(features.dims(), [2, 2048, 2, 2]);
    assert_eq!(values(features.clone()), values(feats[4].clone()));
    assert_eq!(
        values(model.forward_features(input.clone())),
        values(features)
    );
}

#[test]
fn prediction_head_is_pooling_then_linear() {
    let model = ResNet::new(resnet50(1000), &device());
    let input = Tensor::zeros([1, 3, 64, 64], &device());
    let features = model.forward_features(input.clone());

    let pre_logits = model.forward_head(features.clone(), true);
    assert_eq!(pre_logits.dims(), [1, 2048]);

    let logits = model.forward_head(features, false);
    assert_eq!(logits.dims(), [1, 1000]);
    assert_eq!(values(logits), values(model.forward(input)));
}

#[test]
fn alexnet_feature_levels() {
    let model = create_model("alexnet", &CreateModelOptions::default(), &device()).unwrap();
    let reductions: Vec<usize> = model.feature_info().iter().map(|f| f.reduction).collect();
    assert_eq!(reductions, vec![8, 16, 32]);

    let input = Tensor::zeros([1, 3, 224, 224], &device());
    let feats = model.forward_intermediates(input, &IntermediatesOptions::new());
    assert_eq!(feats.len(), 3);
    assert_eq!(feats[0].dims(), [1, 64, 27, 27]);
    assert_eq!(feats[1].dims(), [1, 192, 13, 13]);
    assert_eq!(feats[2].dims(), [1, 256, 6, 6]);
}

#[test]
fn module_paths_match_checkpoint_layout() {
    let model = ResNet::new(resnet50(1000), &device());
    let paths: Vec<String> = model
        .collect(None, None, true)
        .into_iter()
        .map(|tensor| tensor.name)
        .collect();

    for expected in [
        "stem.conv1.conv.weight",
        "stem.conv1.norm.gamma",
        "stem.conv1.norm.running_mean",
        "layer1.0.conv1.conv.weight",
        "layer1.0.downsample.body.conv.weight",
        "layer1.0.downsample.body.norm.running_var",
        "layer4.2.conv3.conv.weight",
        "head.fc.weight",
        "head.fc.bias",
    ] {
        assert!(
            paths.iter().any(|path| path == expected),
            "missing module path `{expected}` in {paths:?}"
        );
    }

    let mut deep = ResNetConfig::new([2, 2, 2, 2], [64, 128, 256, 512], BlockKind::Basic)
        .with_deep_stem()
        .with_avg_down();
    deep.stem = StemKind::Deep;
    let model = ResNet::new(deep, &device());
    let paths: Vec<String> = model
        .collect(None, None, true)
        .into_iter()
        .map(|tensor| tensor.name)
        .collect();
    for expected in [
        "stem.conv_a.conv.weight",
        "stem.conv_a.norm.gamma",
        "stem.conv_b.conv.weight",
        "stem.conv_b.norm.gamma",
        "stem.conv_c.conv.weight",
        "stem.conv_c.norm.gamma",
        "layer2.0.downsample.body.conv.weight",
        "layer2.0.downsample.body.norm.gamma",
    ] {
        assert!(
            paths.iter().any(|path| path == expected),
            "missing module path `{expected}` in {paths:?}"
        );
    }
}
