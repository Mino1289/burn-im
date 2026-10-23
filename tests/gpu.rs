//! GPU execution smoke test (Burn `wgpu`/Vulkan backend).
//!
//! ```bash
//! cargo test --features gpu --test gpu -- --nocapture
//! ```

#![cfg(feature = "gpu")]

use std::time::Instant;

use burn::module::Module;
use burn::tensor::{Device, DeviceKind, Tensor};
use burn_im::models::resnet::{ResNet, resnet18};
use burn_im::traits::VisionModel;
use burn_im::{CreateModelOptions, create_model};

fn gpu() -> Device {
    Device::vulkan(DeviceKind::IntegratedGpu(0))
}

#[test]
fn resnet18_runs_on_gpu() {
    let device = gpu();
    println!("device: {device:?}");

    let model = create_model("resnet18", &CreateModelOptions::default(), &device).unwrap();
    let input = Tensor::zeros([1, 3, 64, 64], &device);
    let out = model.forward(input);
    assert_eq!(out.dims(), [1, 1000]);
    println!("gpu logits: {:?}", out.dims());
}

#[test]
fn gpu_matches_cpu() {
    let model = ResNet::new(resnet18(1000), &Device::flex());
    let input = Tensor::random(
        [1, 3, 64, 64],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &Device::flex(),
    );
    let cpu = model.forward(input.clone());

    let gpu_model = model.to_device(&gpu());
    let gpu = gpu_model.forward(input.to_device(&gpu()));

    let cpu_values: Vec<f32> = cpu.into_data().as_slice::<f32>().unwrap().to_vec();
    let gpu_values: Vec<f32> = gpu.into_data().as_slice::<f32>().unwrap().to_vec();
    let max = cpu_values
        .iter()
        .zip(&gpu_values)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("max |cpu - gpu| = {max:.3e}");
    assert!(max < 1e-4, "cpu/gpu mismatch: {max}");
}

#[test]
fn gpu_throughput() {
    let device = gpu();
    let model = create_model("resnet50", &CreateModelOptions::default(), &device).unwrap();
    let input = Tensor::zeros([1, 3, 224, 224], &device);

    // Warm up shader compilation.
    let _ = model.forward(input.clone());

    let start = Instant::now();
    let iterations = 5;
    for _ in 0..iterations {
        let _ = model.forward(input.clone());
    }
    println!(
        "resnet50@224 GPU: {:.1} ms/iter",
        start.elapsed().as_secs_f64() * 1000.0 / iterations as f64
    );
}

#[test]
fn cpu_throughput() {
    let device = Device::flex();
    let model = create_model("resnet50", &CreateModelOptions::default(), &device).unwrap();
    let input = Tensor::zeros([1, 3, 224, 224], &device);

    let start = Instant::now();
    let iterations = 2;
    for _ in 0..iterations {
        let _ = model.forward(input.clone());
    }
    println!(
        "resnet50@224 CPU (flex): {:.1} ms/iter",
        start.elapsed().as_secs_f64() * 1000.0 / iterations as f64
    );
}

#[test]
#[ignore = "downloads weights from the Hugging Face Hub"]
fn pretrained_weights_load_on_gpu() {
    let model = create_model(
        "resnet18.a1_in1k",
        &CreateModelOptions::pretrained(),
        &gpu(),
    )
    .unwrap();
    let out = model.forward(Tensor::zeros([1, 3, 224, 224], &gpu()));
    assert_eq!(out.dims(), [1, 1000]);
    assert!(
        out.into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite())
    );
}
