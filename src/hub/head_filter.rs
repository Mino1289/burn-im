//! Classifier-head filtering and input-channel adaptation applied when a pretrained checkpoint
//! does not match the requested model configuration.

use burn::prelude::*;

/// What to do with the pretrained classifier head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadAction {
    /// Keep the pretrained classifier (same number of classes).
    Keep,
    /// Re-initialize the classifier for a different number of classes.
    Reset,
    /// Remove the classifier entirely (`num_classes = 0`).
    Remove,
}

/// Decides how the classifier head must be treated.
///
/// * `requested == None` keeps the checkpoint's head.
/// * `requested == Some(0)` removes it.
/// * `requested == Some(n)` with `n == pretrained` keeps it, otherwise resets it.
pub fn head_action(pretrained_classes: usize, requested: Option<usize>) -> HeadAction {
    match requested {
        None => HeadAction::Keep,
        Some(0) => HeadAction::Remove,
        Some(n) if n == pretrained_classes => HeadAction::Keep,
        Some(_) => HeadAction::Reset,
    }
}

/// Number of classes the constructed head must expose.
pub fn target_num_classes(pretrained_classes: usize, requested: Option<usize>) -> usize {
    match requested {
        None => pretrained_classes,
        Some(n) => n,
    }
}

/// Adapts a pretrained first convolution to a different number of input channels.
///
/// Mirrors `timm`'s `adapt_input_conv`:
/// * `1` channel sums the RGB kernels,
/// * any other count repeats the RGB kernel and rescales by `3 / in_chans`.
pub fn adapt_input_conv(weight: Tensor<4>, in_chans: usize) -> Tensor<4> {
    let [_out, src, _kh, _kw] = weight.dims();
    if src == in_chans {
        return weight;
    }
    if in_chans == 1 {
        return weight.sum_dim(1);
    }
    let repeat = src.div_ceil(3);
    Tensor::cat(vec![weight; repeat], 1)
        .narrow(1, 0, in_chans)
        .mul_scalar(3.0 / in_chans as f64)
}
