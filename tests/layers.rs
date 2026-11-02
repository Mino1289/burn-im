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
    grn.weight = Param::from_data(gamma, &device);
    grn.bias = Param::from_data(beta, &device);

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

    // Channels-last variant: permute [B, C, H, W] to [B, H, W, C], run forward_channels_last,
    // and verify it matches the permuted NCHW output.
    let input_nhwc = Tensor::from_data(TensorData::from(input), &device).permute([0, 2, 3, 1]);
    let output_nhwc = grn.forward_channels_last(input_nhwc).permute([0, 3, 1, 2]);
    assert_close(&values(output_nhwc), &output, 1e-6);
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

#[test]
fn rms_norm_and_rmsnorm2d_numerical_parity() {
    let device = device();
    let eps = 1e-6;

    let input_data = [[1.0f32, -2.0, 3.0], [4.0, 0.5, -1.0]];
    let input: Tensor<2> = Tensor::from_data(TensorData::from(input_data), &device);

    let norm = burn_im::layers::RmsNorm::new(3, eps, &device);
    let output = values(norm.forward(input));

    // Manual computation: y = x / sqrt(mean(x^2) + eps)
    for (row_idx, row) in input_data.iter().enumerate() {
        let mean_sq = (row[0].powi(2) + row[1].powi(2) + row[2].powi(2)) / 3.0;
        let rms = (mean_sq as f64 + eps).sqrt() as f32;
        for c in 0..3 {
            let expected = row[c] / rms;
            let actual = output[row_idx * 3 + c];
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    // RMSNorm2d
    let input_4d: [[[[f32; 2]; 2]; 3]; 1] = [[
        [[1.0, 2.0], [3.0, 4.0]],
        [[-1.0, 0.0], [1.0, 2.0]],
        [[0.5, -0.5], [1.5, -1.5]],
    ]];
    let input_tensor: Tensor<4> = Tensor::from_data(TensorData::from(input_4d), &device);
    let norm2d = burn_im::layers::RMSNorm2d::new(3, eps, &device);
    let out_4d = norm2d.forward(input_tensor);
    assert_eq!(out_4d.dims(), [1, 3, 2, 2]);

    // Check force_fp32 preserves output shape and executes correctly
    let norm_fp32 = norm.with_force_fp32(true);
    let in_fp32: Tensor<2> = Tensor::from_data(TensorData::from(input_data), &device);
    let out_fp32 = norm_fp32.forward(in_fp32);
    assert_eq!(out_fp32.dims(), [2, 3]);
}

#[test]
fn rope_xy_vs_ij_differ_and_buffers_are_f32() {
    let device = device();

    let rope_ij = burn_im::layers::RotaryEmbeddingCat::new(
        64,
        224,
        10000.0,
        false,
        false,
        burn_im::layers::GridIndexing::Ij,
        true,
        &device,
    );
    let rope_xy = burn_im::layers::RotaryEmbeddingCat::new(
        64,
        224,
        10000.0,
        false,
        false,
        burn_im::layers::GridIndexing::Xy,
        true,
        &device,
    );

    // Frequency buffer must be f32
    assert_eq!(rope_ij.bands.dtype(), burn::tensor::DType::F32);
    assert_eq!(rope_xy.bands.dtype(), burn::tensor::DType::F32);

    let emb_ij = values(rope_ij.get_embed((8, 12)));
    let emb_xy = values(rope_xy.get_embed((8, 12)));

    assert_eq!(emb_ij.len(), emb_xy.len());
    // Failsafe check: 'xy' grid must produce different embeddings than 'ij'
    let max_diff = emb_ij
        .iter()
        .zip(&emb_xy)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-3,
        "'xy' grid indexing must differ from 'ij' (got max diff {max_diff})"
    );

    // DINOv3 RoPE test
    let dino_ij = burn_im::layers::RotaryEmbeddingDinoV3::new(
        64,
        100.0,
        burn_im::layers::GridIndexing::Ij,
        "separate".to_string(),
        true,
        &device,
    );
    let dino_xy = burn_im::layers::RotaryEmbeddingDinoV3::new(
        64,
        100.0,
        burn_im::layers::GridIndexing::Xy,
        "separate".to_string(),
        true,
        &device,
    );
    assert_eq!(dino_ij.periods.dtype(), burn::tensor::DType::F32);
    let dino_emb_ij = values(dino_ij.get_embed((7, 9)));
    let dino_emb_xy = values(dino_xy.get_embed((7, 9)));
    let dino_diff = dino_emb_ij
        .iter()
        .zip(&dino_emb_xy)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(dino_diff > 1e-3, "DINOv3 RoPE 'xy' must differ from 'ij'");

    // MRoPE test
    let mrope = burn_im::layers::RotaryEmbeddingMRope::new(
        64,
        (8, 12, 12),
        10000.0,
        burn_im::layers::GridIndexing::Ij,
        &device,
    );
    assert_eq!(mrope.inv_freq.dtype(), burn::tensor::DType::F32);
    let mrope_emb = mrope.get_embed((8, 8));
    assert_eq!(mrope_emb.dims(), [64, 128]);
}

#[test]
fn pos_embed_constant_tokens_resampling_timm_1029() {
    let device = device();
    // Grid 4x6 vs 6x4: identical total tokens (24 tokens), but different (H, W) shapes!
    let embed_dim = 16;
    let old_pos: Tensor<3> = Tensor::random(
        [1, 25, embed_dim],
        Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let resampled =
        burn_im::layers::resample_abs_pos_embed(old_pos.clone(), (6, 4), Some((4, 6)), 1);
    assert_eq!(resampled.dims(), [1, 25, embed_dim]);

    // Resampling across different shapes must not be a no-op identity
    let diff = values(resampled)
        .iter()
        .zip(values(old_pos))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff > 1e-4,
        "Resampling with different (H, W) must interpolate, not return identity"
    );
}

#[test]
fn transformer_bricks_patch_dropout_swiglu_prr_map() {
    let device = device();

    // PatchDropout is identity during eval
    let patch_drop = burn_im::layers::PatchDropout::new(0.5, 1);
    let tokens = Tensor::random([2, 17, 32], Distribution::Uniform(-1.0, 1.0), &device);
    let out = patch_drop.forward(tokens.clone());
    assert_eq!(out.dims(), [2, 17, 32]);
    assert_eq!(values(out), values(tokens));

    // SwiGLU forward
    let swiglu = burn_im::layers::SwiGLU::new(32, 64, 32, true, &device);
    let x = Tensor::random([2, 10, 32], Distribution::Uniform(-1.0, 1.0), &device);
    let out_swiglu = swiglu.forward(x);
    assert_eq!(out_swiglu.dims(), [2, 10, 32]);

    // AttentionPoolPrr (PRR) forward
    let prr = burn_im::layers::AttentionPoolPrr::new(32, 4, "token", true, true, &device);
    let tokens_prr = Tensor::random([2, 17, 32], Distribution::Uniform(-1.0, 1.0), &device);
    let pooled_prr = prr.forward(tokens_prr);
    assert_eq!(pooled_prr.dims(), [2, 32]);

    // AttentionPoolLatent (MAP head) forward
    let map_head = burn_im::layers::AttentionPoolLatent::new(
        32,
        Some(10),
        Some(32),
        4,
        4.0,
        true,
        true,
        false,
        1,
        "token",
        &device,
    );
    let tokens_map = Tensor::random([2, 17, 32], Distribution::Uniform(-1.0, 1.0), &device);
    let out_map = map_head.forward(tokens_map);
    assert_eq!(out_map.dims(), [2, 10]);
}

#[test]
fn attention_masks_causal_boolean_and_key_only() {
    let device = device();
    let attn = burn_im::layers::Attention::new(32, 4, true, false, true, &device);
    let x = Tensor::random([2, 8, 32], Distribution::Uniform(-1.0, 1.0), &device);

    // Unmasked plain forward
    let out_plain = attn.forward_plain(x.clone());
    assert_eq!(out_plain.dims(), [2, 8, 32]);

    // Causal mask
    let out_causal = attn.forward(x.clone(), None, true);
    assert_eq!(out_causal.dims(), [2, 8, 32]);
    let causal_diff = values(out_plain.clone())
        .iter()
        .zip(values(out_causal))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        causal_diff > 1e-4,
        "Causal attention must differ from bidirectional attention"
    );

    // Float mask
    let float_mask = Tensor::zeros([2, 4, 8, 8], &device);
    let out_float = attn.forward(
        x.clone(),
        Some(burn_im::layers::AttentionMask::Float(float_mask)),
        false,
    );
    assert_close(&values(out_plain.clone()), &values(out_float), 1e-5);

    // Boolean mask
    let bool_mask = Tensor::<4, burn::tensor::Bool>::from_data(
        burn::tensor::TensorData::new(vec![true; 2 * 8 * 8], [2, 1, 8, 8]),
        &device,
    );
    let out_bool = attn.forward(
        x.clone(),
        Some(burn_im::layers::AttentionMask::Bool(bool_mask)),
        false,
    );
    assert_close(&values(out_plain.clone()), &values(out_bool), 1e-5);

    // Key-only mask: keep first 6 tokens, mask out last 2
    let mut key_mask_vec = vec![true; 2 * 8];
    for b in 0..2 {
        key_mask_vec[b * 8 + 6] = false;
        key_mask_vec[b * 8 + 7] = false;
    }
    let key_mask = Tensor::<2, burn::tensor::Bool>::from_data(
        burn::tensor::TensorData::new(key_mask_vec, [2, 8]),
        &device,
    );
    let out_key = attn.forward(
        x,
        Some(burn_im::layers::AttentionMask::KeyOnly(key_mask)),
        false,
    );
    assert_eq!(out_key.dims(), [2, 8, 32]);
    let key_diff = values(out_plain)
        .iter()
        .zip(values(out_key))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        key_diff > 1e-4,
        "Key-only mask must change attention outputs"
    );
}

#[test]
fn rope_mixed_and_dtype_preservation() {
    let device = device();

    // Mixed RoPE
    let mixed = burn_im::layers::RotaryEmbeddingMixed::new(
        64,
        4,
        4,
        10.0,
        burn_im::layers::GridIndexing::Ij,
        &device,
    );
    let embed = mixed.get_embed_for_block(0, (6, 6));
    assert_eq!(embed.dims(), [4, 36, 32]);

    // Dtype preservation in apply_rot_embed_cat
    let q = Tensor::<4>::random([1, 4, 36, 16], Distribution::Uniform(-1.0, 1.0), &device);
    let rope_cat = burn_im::layers::RotaryEmbeddingCat::new(
        16,
        224,
        10000.0,
        false,
        false,
        burn_im::layers::GridIndexing::Ij,
        true,
        &device,
    );
    let emb = rope_cat.get_embed((6, 6));
    let q_rot = burn_im::layers::apply_rot_embed_cat(q.clone(), emb, true);
    assert_eq!(q_rot.dims(), [1, 4, 36, 16]);
    assert_eq!(q_rot.dtype(), q.dtype());
}
