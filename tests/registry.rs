//! Registry, config parsing and model construction tests.

use burn::tensor::{Device, Tensor};
use burn_im::hub::config::TimmHfConfig;
use burn_im::registry::split_model_name_tag;
use burn_im::{CreateModelOptions, create_model, is_model, list_models, list_pretrained};

fn device() -> Device {
    Device::flex()
}

#[test]
fn parses_model_identifiers() {
    let id = split_model_name_tag("resnet50.a1_in1k").unwrap();
    assert_eq!(id.arch, "resnet50");
    assert_eq!(id.tag.as_deref(), Some("a1_in1k"));
    assert_eq!(id.name, "resnet50.a1_in1k");
    assert!(id.repo.is_none());

    let id = split_model_name_tag("resnet18").unwrap();
    assert_eq!(id.arch, "resnet18");
    assert_eq!(id.tag, None);
    assert_eq!(id.name, "resnet18");

    let id = split_model_name_tag("hf-hub:timm/resnet50d.ra2_in1k").unwrap();
    assert_eq!(id.arch, "resnet50d");
    assert_eq!(id.tag.as_deref(), Some("ra2_in1k"));
    assert_eq!(id.repo.as_deref(), Some("timm/resnet50d.ra2_in1k"));

    assert!(split_model_name_tag("").is_err());
    assert!(split_model_name_tag("resnet50.").is_err());
}

#[test]
fn lists_and_finds_models() {
    let resnets = list_models(Some("*resnet*"));
    assert!(resnets.contains(&"resnet18"));
    assert!(resnets.contains(&"resnet50d"));
    assert!(!resnets.contains(&"alexnet"));
    assert!(list_models(Some("*resnext*")).contains(&"resnext50_32x4d"));
    assert!(list_models(None).contains(&"alexnet"));

    assert!(is_model("resnet50"));
    assert!(is_model("resnet50.a1_in1k"));
    assert!(is_model("hf-hub:timm/resnet50.a1_in1k"));
    assert!(!is_model("convnext_tiny"));

    let pairs = list_pretrained(Some("*resnet50d*"));
    assert!(pairs.contains(&"resnet50d.ra2_in1k".to_string()));
    assert!(!pairs.iter().any(|pair| pair.contains("resnet18")));
}

#[test]
fn parses_published_config() {
    let cfg = TimmHfConfig::from_file("tests/fixtures/resnet50.a1_in1k.json").unwrap();
    assert_eq!(cfg.architecture, "resnet50");
    assert_eq!(cfg.num_classes, 1000);
    assert_eq!(cfg.num_features, Some(2048));
    assert_eq!(cfg.pretrained_cfg.input_size, [3, 224, 224]);
    assert_eq!(cfg.pretrained_cfg.crop_pct, 0.95);
    assert_eq!(cfg.pretrained_cfg.first_conv, "conv1");
    assert_eq!(cfg.pretrained_cfg.classifier, "fc");
    assert_eq!(cfg.pretrained_cfg.pool_size, Some([7, 7]));

    let cfg = TimmHfConfig::from_file("tests/fixtures/resnet50d.ra2_in1k.json").unwrap();
    assert_eq!(cfg.architecture, "resnet50d");
    assert_eq!(cfg.pretrained_cfg.first_conv, "conv1.0");
}

#[test]
fn unknown_model_is_rejected() {
    let err = create_model("not_a_model", &CreateModelOptions::default(), &device())
        .err()
        .expect("unknown model must fail");
    assert!(matches!(err, burn_im::Error::UnknownModel(_)));
}

#[test]
fn creates_resnet18_from_registry() {
    let model = create_model("resnet18", &CreateModelOptions::default(), &device()).unwrap();
    assert_eq!(model.num_classes(), 1000);
    assert_eq!(model.num_features(), 512);
    assert_eq!(model.feature_info().len(), 5);
    assert_eq!(model.feature_info()[4].reduction, 32);
    assert_eq!(model.model_name(), "resnet18");

    let logits = model.forward(Tensor::zeros([1, 3, 64, 64], &device()));
    assert_eq!(logits.dims(), [1, 1000]);
}

#[test]
fn creates_headless_and_resized_heads() {
    let options = CreateModelOptions::default().with_num_classes(0);
    let model = create_model("resnet18", &options, &device()).unwrap();
    assert_eq!(model.num_classes(), 0);
    let out = model.forward(Tensor::zeros([2, 3, 64, 64], &device()));
    assert_eq!(out.dims(), [2, 512]);

    let options = CreateModelOptions::default().with_num_classes(10);
    let model = create_model("resnet18", &options, &device()).unwrap();
    let out = model.forward(Tensor::zeros([2, 3, 64, 64], &device()));
    assert_eq!(out.dims(), [2, 10]);
}

#[test]
fn creates_deep_and_grouped_variants() {
    for arch in ["resnet18d", "resnet50d", "resnet101d", "resnext50_32x4d"] {
        let model = create_model(arch, &CreateModelOptions::default(), &device()).unwrap();
        let logits = model.forward(Tensor::zeros([1, 3, 64, 64], &device()));
        assert_eq!(logits.dims(), [1, 1000], "{arch}");
    }
}

#[test]
fn creates_alexnet() {
    let model = create_model("alexnet", &CreateModelOptions::default(), &device()).unwrap();
    assert_eq!(model.num_features(), 4096);
    let logits = model.forward(Tensor::zeros([1, 3, 224, 224], &device()));
    assert_eq!(logits.dims(), [1, 1000]);

    let options = CreateModelOptions::default().with_num_classes(0);
    let model = create_model("alexnet", &options, &device()).unwrap();
    let out = model.forward(Tensor::zeros([1, 3, 224, 224], &device()));
    assert_eq!(out.dims(), [1, 4096]);
}

#[test]
fn alexnet_has_no_pretrained_hub_weights() {
    let err = create_model("alexnet", &CreateModelOptions::pretrained(), &device())
        .err()
        .expect("alexnet has no Hub weights");
    assert!(matches!(err, burn_im::Error::UnknownTag { .. }));
}
