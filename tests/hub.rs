//! Hugging Face Hub `config.json` pipeline.
//!
//! The fixture is the unmodified `config.json` downloaded from
//! `https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k/raw/main/config.json`.
//!
//! The live round-trip is ignored by default because it needs network access:
//!
//! ```bash
//! cargo test --test hub -- --ignored --nocapture
//! ```

use burn_im::hub::Hub;
use burn_im::hub::config::{Interpolation, TimmHfConfig};

const REPO: &str = "timm/convnext_tiny.fb_in22k_ft_in1k";

fn assert_convnext_tiny(cfg: &TimmHfConfig) {
    assert_eq!(cfg.architecture, "convnext_tiny");
    assert_eq!(cfg.num_classes, 1000);
    assert_eq!(cfg.num_features, Some(768));

    let pretrained = &cfg.pretrained_cfg;
    assert_eq!(pretrained.tag.as_deref(), Some("fb_in22k_ft_in1k"));
    assert_eq!(pretrained.num_classes, 1000);
    assert_eq!(pretrained.in_chans(), 3);
    assert_eq!(pretrained.input_size, [3, 224, 224]);
    assert_eq!(pretrained.input_hw(), (224, 224));
    assert_eq!(pretrained.test_input_size, Some([3, 288, 288]));
    assert!(!pretrained.fixed_input_size);
    assert!(!pretrained.custom_load);
    assert_eq!(pretrained.mean, [0.485, 0.456, 0.406]);
    assert_eq!(pretrained.std, [0.229, 0.224, 0.225]);
    assert_eq!(pretrained.crop_pct, 0.875);
    assert_eq!(pretrained.test_crop_pct, Some(1.0));
    assert_eq!(pretrained.crop_mode, "center");
    assert_eq!(pretrained.interpolation, Interpolation::Bicubic);
    assert_eq!(pretrained.pool_size, Some([7, 7]));
    assert_eq!(pretrained.first_conv, "stem.0");
    assert_eq!(pretrained.classifier, "head.fc");
}

#[test]
fn parses_convnext_tiny_config_fixture() {
    let cfg = TimmHfConfig::from_file("tests/fixtures/convnext_tiny.fb_in22k_ft_in1k.json")
        .expect("committed convnext_tiny config must parse");
    assert_convnext_tiny(&cfg);
}

#[test]
#[ignore = "downloads config.json from the Hugging Face Hub"]
fn downloads_convnext_tiny_config() {
    let cfg = Hub::new().unwrap().config(REPO).unwrap();
    println!("{cfg:#?}");
    assert_convnext_tiny(&cfg);
}
