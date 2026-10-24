//! Unit tests for the shared building blocks in `src/layers/`.

use burn::module::{Module, Param};
use burn::nn::LayerNormConfig;
use burn::tensor::{Device, Distribution, Tensor, TensorData};
use burn_im::layers::drop::drop_path_schedule;
use burn_im::layers::{
    DropPath, GlobalResponseNorm, LayerNorm2d, LayerScale, LayerScale2d, NormMlpClassifierHead,
    PoolMode,
};

fn device() -> Device {
    Device::flex()
}

fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor.into_data().as_slice::<f32>().unwrap().to_vec()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tolerance,
            "value {index}: {a} != {e} (tolerance {tolerance})"
        );
    }
}

#[test]
fn layer_norm2d_matches_permuted_layer_norm() {
    let device = device();
    let eps = 1e-6;
    let input = Tensor::random([2, 3, 4, 5], Distribution::Uniform(-1.0, 1.0), &device);

    let norm2d = LayerNorm2d::new(3, eps, &device);
    let reference = LayerNormConfig::new(3).with_epsilon(eps).init(&device);

    let output = norm2d.forward(input.clone());
    let expected = reference
        .forward(input.permute([0, 2, 3, 1]))
        .permute([0, 3, 1, 2]);

    assert_eq!(output.dims(), [2, 3, 4, 5]);
    assert_close(&values(output), &values(expected), 1e-5);
}

#[test]
fn global_response_norm_matches_reference() {
    let device = device();
    let input: [[[[f32; 3]; 3]; 2]; 1] = [[
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        [[-1.0, 0.0, 1.0], [2.0, -2.0, 0.0], [0.5, 1.5, -1.5]],
    ]];
    let gamma = [1.5f32, -0.5];
    let beta = [0.25f32, 0.75];
    let eps = 1e-6f64;

    let mut grn = GlobalResponseNorm::new(2, &device);
    grn.gamma = Param::from_data(gamma, &device);
    grn.beta = Param::from_data(beta, &device);

    let output = values(grn.forward(Tensor::from_data(TensorData::from(input), &device)));

    // Reference computed in f64 from the raw values.
    let mut norms = [0f64; 2];
    for (channel, plane) in input[0].iter().enumerate() {
        for row in plane {
            for value in row {
                norms[channel] += (*value as f64).powi(2);
            }
        }
        norms[channel] = norms[channel].sqrt();
    }
    let mean = (norms[0] + norms[1]) / 2.0;

    for (channel, plane) in input[0].iter().enumerate() {
        let response = norms[channel] / (mean + eps);
        for (row, row_values) in plane.iter().enumerate() {
            for (column, value) in row_values.iter().enumerate() {
                let x = *value as f64;
                let expected = x + beta[channel] as f64 + gamma[channel] as f64 * x * response;
                let index = channel * 9 + row * 3 + column;
                assert!(
                    (output[index] as f64 - expected).abs() < 1e-5,
                    "value {index}: {} != {expected}",
                    output[index]
                );
            }
        }
    }
}

#[test]
fn drop_path_is_identity_when_inactive() {
    let device = device();
    let input = Tensor::ones([4, 3, 2, 2], &device);
    let expected = values(input.clone());

    // p == 0.
    assert_eq!(values(DropPath::new(0.0).forward(input.clone())), expected);

    // Frozen: the training flag is cleared by the module traversal.
    let frozen = DropPath::new(0.5).freeze();
    assert_eq!(values(frozen.forward(input.clone())), expected);

    // Inference (not an autodiff device): identity even while armed. The stochastic branch is
    // unit-tested in `src/layers/drop.rs`, since Burn's `autodiff` feature is not enabled for
    // this crate's test build.
    assert_eq!(values(DropPath::new(0.5).forward(input)), expected);
}

#[test]
fn drop_path_schedule_linear_endpoints() {
    let schedule = drop_path_schedule(0.5, 4);
    assert_eq!(schedule.len(), 4);
    assert_eq!(schedule[0], 0.0);
    assert_eq!(schedule[3], 0.5);
    assert_close(
        &schedule.iter().map(|v| *v as f32).collect::<Vec<_>>(),
        &[0.0, 0.5 / 3.0, 1.0 / 3.0, 0.5],
        1e-6,
    );

    // torch.linspace(0, rate, 1) -> [0.0]
    assert_eq!(drop_path_schedule(0.3, 1), vec![0.0]);
    assert_eq!(drop_path_schedule(0.0, 5), vec![0.0; 5]);
    assert!(drop_path_schedule(0.0, 0).is_empty());

    let schedule = drop_path_schedule(0.7, 9);
    assert!(schedule.windows(2).all(|pair| pair[1] > pair[0]));
}

#[test]
fn layer_scale_multiplies_per_channel() {
    let device = device();
    let gamma = [2.0f32, -1.0, 0.5];

    let mut scale = LayerScale::new(3, 0.5, &device);
    assert_eq!(scale.gamma.dims(), [3]);
    assert_close(&values(scale.gamma.val()), &[0.5; 3], 0.0);
    scale.gamma = Param::from_data(gamma, &device);

    let input = Tensor::<3>::ones([2, 4, 3], &device);
    let output = values(scale.forward(input));
    for batch in 0..2 {
        for token in 0..4 {
            for channel in 0..3 {
                assert_eq!(output[(batch * 4 + token) * 3 + channel], gamma[channel]);
            }
        }
    }

    let mut scale = LayerScale2d::new(3, 0.1, &device);
    assert_eq!(scale.gamma.dims(), [3, 1, 1]);
    assert_close(&values(scale.gamma.val()), &[0.1; 3], 1e-7);
    scale.gamma = Param::from_data(TensorData::from([[[2.0f32]], [[-1.0]], [[0.5]]]), &device);

    let input = Tensor::<4>::ones([1, 3, 2, 2], &device);
    let output = values(scale.forward(input));
    for channel in 0..3 {
        for pixel in 0..4 {
            assert_eq!(output[channel * 4 + pixel], gamma[channel]);
        }
    }
}

#[test]
fn norm_mlp_head_shapes() {
    let device = device();
    let input = Tensor::random([2, 8, 3, 3], Distribution::Uniform(-1.0, 1.0), &device);

    // Headless: pre-logits embedding.
    let head = NormMlpClassifierHead::new(8, 0, None, PoolMode::Avg, 0.0, Some(1e-6), &device);
    assert_eq!(head.num_features(), 8);
    assert_eq!(head.num_classes(), 0);
    assert_eq!(head.forward(input.clone(), false).dims(), [2, 8]);

    // Replaced classifier.
    let head = NormMlpClassifierHead::new(8, 10, None, PoolMode::Avg, 0.0, Some(1e-6), &device);
    assert_eq!(head.forward(input.clone(), true).dims(), [2, 8]);
    assert_eq!(head.forward(input.clone(), false).dims(), [2, 10]);

    // Hidden MLP widens the pre-logits embedding.
    let head = NormMlpClassifierHead::new(8, 10, Some(16), PoolMode::Avg, 0.0, Some(1e-6), &device);
    assert_eq!(head.num_features(), 16);
    assert_eq!(head.forward(input.clone(), true).dims(), [2, 16]);
    assert_eq!(head.forward(input.clone(), false).dims(), [2, 10]);

    // No norm layer.
    let head = NormMlpClassifierHead::new(8, 10, None, PoolMode::Avg, 0.0, None, &device);
    assert_eq!(head.forward(input, false).dims(), [2, 10]);
}
