//! Stochastic depth (`DropPath`), `PatchDropout` and the `timm` linear schedule.

use burn::module::{Flag, Module, Param};
use burn::tensor::{Distribution, Tensor};

/// Stochastic depth: drops the whole residual branch of a sample.
///
/// Mirrors [`burn::nn::Dropout`]: the layer is an identity when `drop_prob == 0`, when it has been
/// frozen (`Module::freeze`, which clears `training`), or when the input is not on an autodiff
/// device — i.e. during inference. Otherwise one Bernoulli mask is drawn per sample and the kept
/// samples are scaled by `1 / (1 - drop_prob)`.
#[derive(Module, Debug)]
pub struct DropPath {
    /// Probability of dropping a sample's residual branch. Must be in `[0, 1)`.
    pub drop_prob: f64,
    /// Whether to behave as during training. Cleared by `freeze` traversals.
    pub training: Param<Flag>,
}

impl DropPath {
    /// Creates a drop-path layer.
    ///
    /// # Panics
    ///
    /// Panics if `drop_prob` is not finite or outside `[0, 1)`, like [`burn::nn::Dropout`].
    pub fn new(drop_prob: f64) -> Self {
        if !drop_prob.is_finite() || !(0.0..1.0).contains(&drop_prob) {
            panic!(
                "DropPath probability must be finite and in the range [0, 1), but got {drop_prob}"
            );
        }
        Self {
            drop_prob,
            training: Param::from_bool(true),
        }
    }

    /// Applies drop path to `[B, C, H, W]`.
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        if !self.training.is_enabled() || !input.device().is_autodiff() || self.drop_prob == 0.0 {
            return input;
        }
        stochastic_depth(input, self.drop_prob)
    }

    /// Applies drop path to `[B, N, C]` (used in transformer blocks).
    pub fn forward_3d(&self, input: Tensor<3>) -> Tensor<3> {
        if !self.training.is_enabled() || !input.device().is_autodiff() || self.drop_prob == 0.0 {
            return input;
        }
        let keep = 1.0 - self.drop_prob;
        let batch = input.dims()[0];
        let mask = Tensor::<3>::random(
            [batch, 1, 1],
            Distribution::Bernoulli(keep),
            &input.device(),
        );
        input * mask * (1.0 / keep)
    }
}

/// Randomly drops patch tokens (excluding prefix tokens) during training (`timm.layers.PatchDropout`).
///
/// In evaluation mode, or when `prob == 0.0`, input tokens pass through untouched.
#[derive(Module, Debug)]
pub struct PatchDropout {
    /// Probability of dropping each spatial token.
    pub prob: f64,
    /// Number of leading prefix tokens (e.g. class token, register tokens) to always keep.
    pub num_prefix_tokens: usize,
    /// Whether the layer is in training mode.
    pub training: Param<Flag>,
}

impl PatchDropout {
    /// Creates a `PatchDropout` module.
    pub fn new(prob: f64, num_prefix_tokens: usize) -> Self {
        Self {
            prob,
            num_prefix_tokens,
            training: Param::from_bool(true),
        }
    }

    /// Applies patch dropout to token sequences `[B, N, C]`.
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        if !self.training.is_enabled() || !input.device().is_autodiff() || self.prob == 0.0 {
            return input;
        }
        let [batch, num_tokens, _channels] = input.dims();
        if num_tokens <= self.num_prefix_tokens {
            return input;
        }
        let keep_prob = 1.0 - self.prob;
        let spatial_tokens = num_tokens - self.num_prefix_tokens;
        let mask = Tensor::<3>::random(
            [batch, spatial_tokens, 1],
            Distribution::Bernoulli(keep_prob),
            &input.device(),
        );

        if self.num_prefix_tokens > 0 {
            let prefix = input.clone().slice([0..batch, 0..self.num_prefix_tokens]);
            let spatial = input.slice([0..batch, self.num_prefix_tokens..num_tokens]);
            let spatial = spatial * mask * (1.0 / keep_prob);
            Tensor::cat(vec![prefix, spatial], 1)
        } else {
            input * mask * (1.0 / keep_prob)
        }
    }
}

/// Draws one Bernoulli mask per sample, applies it and scales kept samples by `1 / (1 - p)`.
fn stochastic_depth(input: Tensor<4>, drop_prob: f64) -> Tensor<4> {
    let keep = 1.0 - drop_prob;
    let batch = input.dims()[0];
    let mask = Tensor::<4>::random(
        [batch, 1, 1, 1],
        Distribution::Bernoulli(keep),
        &input.device(),
    );
    input * mask * (1.0 / keep)
}

/// Linear drop-path schedule `linspace(0, rate, depth)`, as used by `timm`.
pub fn drop_path_schedule(rate: f64, depth: usize) -> Vec<f64> {
    match depth {
        0 => Vec::new(),
        1 => vec![0.0],
        _ => (0..depth)
            .map(|i| rate * i as f64 / (depth - 1) as f64)
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::prelude::*;

    #[test]
    fn mask_is_per_sample_scaled_and_centred_on_p() {
        let device = Device::flex();
        let batch = 512;
        let output: Vec<f32> = stochastic_depth(Tensor::ones([batch, 4, 2, 2], &device), 0.5)
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();

        let mut dropped = 0;
        let mut sum = 0.0f64;
        for sample in output.chunks(4 * 2 * 2) {
            let value = sample[0];
            assert!(
                sample.iter().all(|v| *v == value),
                "the mask must be constant per sample"
            );
            if value == 0.0 {
                dropped += 1;
            } else {
                assert!((value - 2.0).abs() < 1e-4);
                sum += value as f64;
            }
        }

        let mean = sum / batch as f64;
        assert!(
            (mean - 1.0).abs() < 0.1,
            "mean over batch should be close to 1.0, got {mean}"
        );
        let drop_rate = dropped as f64 / batch as f64;
        assert!(
            (drop_rate - 0.5).abs() < 0.1,
            "observed drop rate should be close to p=0.5, got {drop_rate}"
        );
    }
}
