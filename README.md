# burn-im — Burn Image Models 🔥🦀

[![Crates.io](https://img.shields.io/crates/v/burn-im.svg)](https://crates.io/crates/burn-im)
[![Documentation](https://docs.rs/burn-im/badge.svg)](https://docs.rs/burn-im)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

**`burn-im`** is a pure-Rust computer vision library built on top of the [Burn](https://burn.dev/) deep learning framework, inspired by [`timm` (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models).

It provides idiomatic implementations of modern vision backbones, native loading of the `.safetensors` checkpoints published by the `timm` organization on the Hugging Face Hub, and multi-scale feature extraction for downstream tasks (detection, segmentation).

- No Python at runtime, no `pyo3`, no `libtorch`.
- WebAssembly-compatible model code (plain `burn::nn` layers).
- Numerically verified against `timm` at `1e-5` relative L2 (see [Numerical parity](#numerical-parity)).

## Supported models

| Family  | Architectures                                                                                      |
| ------- | -------------------------------------------------------------------------------------------------- |
| ResNet  | `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`                                        |
| ResNet-D| `resnet18d`, `resnet34d`, `resnet50d`, `resnet101d`, `resnet152d`                                    |
| ResNeXt | `resnext50_32x4d`, `resnext101_32x8d`, `wide_resnet50_2`, `wide_resnet101_2`                        |
| AlexNet | `alexnet` (architecture only — no `timm` 1.0 checkpoint; torchvision weights can be loaded locally)  |

Use `list_models` / `list_pretrained` to query the registry at runtime:

```rust
for name in burn_im::list_pretrained(Some("*resnet*")) {
    println!("{name}");
}
```

## Installation

```toml
[dependencies]
burn-im = "0.1"
burn = { version = "0.22", features = ["flex"] } # pick your backend
```

| Feature | Default | Effect                                                        |
| ------- | ------- | ------------------------------------------------------------- |
| `image` | no      | `image`-crate integration for decoding JPEG/PNG inputs.        |
| `gpu`   | no      | Enables Burn's `wgpu`/Vulkan backend (`burn/vulkan`).          |

## Quick start

### Dynamic API

`create_model` resolves a `<arch>.<tag>` name through the registry and returns a
`Box<dyn VisionModel>`:

```rust
use burn_im::{CreateModelOptions, create_model, prelude::*};

let device = Device::default();
let model = create_model(
    "resnet50.a1_in1k",          // or "resnet50" for the default tag
    &CreateModelOptions::pretrained(),
    &device,
)?;

let input = Tensor::<4>::zeros([1, 3, 224, 224], &device);
let logits = model.forward(input);
assert_eq!(logits.dims(), [1, 1000]);
```

Weight sources:

```rust
// timm checkpoint from the Hugging Face Hub (cached locally, sync download).
CreateModelOptions::pretrained();

// Local `.safetensors` file.
CreateModelOptions::default().with_checkpoint("weights/resnet50.safetensors");

// Replace the classifier (the new head is freshly initialized).
CreateModelOptions::pretrained().with_num_classes(10);

// Non-RGB input: the first convolution is adapted the same way timm's
// `adapt_input_conv` does it (channel sum / mean).
CreateModelOptions::pretrained().with_in_chans(1);
```

### Static API

Each family also exposes its configuration and module directly:

```rust
use burn_im::models::resnet::{ResNet, resnet50};
use burn_im::prelude::*;

let model = ResNet::new(resnet50(1000), &device);
```

### Feature extraction

`forward_intermediates` mirrors `timm`: every model publishes a `FeatureInfo`
(reduction factor, channels, module) per stage and returns the requested levels.

```rust
use burn_im::prelude::*;

let features = model.forward_intermediates(
    input,
    &IntermediatesOptions::new().with_indices([2, 3, 4]).with_stop_early(true),
);
// features[i] is rank-4 [N, C, H, W] by default; see FeatureFormat for NHWC.
```

`model.feature_info()` returns the list of available levels, and
`model.forward_features(x)` gives the final backbone output before pooling.

## GPU

Model code is device-agnostic: it runs wherever the input tensors live. Burn does **not**
pick a GPU by default — `Device::default()`/`Device::flex()` is CPU. Enable the `gpu`
feature and select a Vulkan device explicitly:

```toml
burn-im = { version = "0.1", features = ["gpu"] }
```

```rust
use burn::tensor::{Device, DeviceKind};

let device = Device::vulkan(DeviceKind::IntegratedGpu(0)); // Intel / AMD integrated
// let device = Device::vulkan(DeviceKind::DiscreteGpu(0)); // discrete GPU
```

Weights load directly on the GPU device, and CPU/GPU outputs match to ~`4e-8` in `f32`.

Measured on an Intel Arc (Arrow Lake-P) iGPU with `resnet50` at `224×224`, batch 1:

| Backend                    | Debug     | Release   |
| -------------------------- | --------- | --------- |
| CPU (`Device::flex()`)     | ~16.8 s   | ~186 ms   |
| Vulkan (`Device::vulkan`)  | ~86 ms    | ~78 ms    |

In release the GPU is roughly **2.4× faster** than the CPU backend for this workload
(the GPU path is still dominated by kernel dispatch at batch 1; larger batches favour it more).

## Numerical parity

Golden tests compare logits, pooled features, intermediate levels and `pre_logits`
against reference `timm`/`torchvision` outputs on deterministic inputs:

| Model                     | Max abs diff (worst tensor) |
| ------------------------- | --------------------------- |
| `resnet18.a1_in1k`        | `2.3e-5`                    |
| `resnet50.a1_in1k`        | `2.8e-5`                    |
| `resnet50d.ra2_in1k`      | `6.1e-5`                    |
| `resnext50_32x4d.a1h_in1k`| `9.7e-5`                    |
| `alexnet` (torchvision)   | `5.5e-5`                    |
| `resnet18` `in_chans=1`   | `1.2e-5`                    |

Tolerance: **`1e-5` relative L2** for every compared tensor, plus an elementwise
`1e-4` ceiling so a single outlier cannot hide behind the norm. The elementwise value is
the f32 accumulation noise floor of these networks, not algorithmic slack — the per-element
differences are the ones listed above.

```bash
# Unit + registry + intermediates tests (no network).
cargo test
```

Golden fixtures are generated with `scripts/generate_fixtures.py`; see
[Development](#development) for the Python setup and the full test matrix.

## Development

Using the crate does **not** require Python. Python is only needed to regenerate the
reference outputs stored under `tests/fixtures/golden/`.

The generator runs the official `timm` (and `torchvision` for AlexNet) checkpoints.
[`uv`](https://docs.astral.sh/uv/) keeps that environment isolated and disposable:

```bash
# 1. Install uv (once).
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create an isolated Python environment.
uv venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install PyTorch for your platform:
#    https://pytorch.org/get-started/locally/
uv pip install torch torchvision

# 4. Install timm on top of it.
uv pip install timm

# 5. Regenerate the golden fixtures.
uv run python scripts/generate_fixtures.py
```

Then run the suites and lints:

```bash
cargo test                                            # no network
cargo test -- --ignored --nocapture                   # Hub weights + timm parity
cargo test --features gpu --test gpu -- --nocapture   # Vulkan smoke/parity/throughput
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
```

The golden tests download the same `timm/<arch>.<tag>` checkpoints as the library, then
compare logits, features, `pre_logits` and every intermediate level. Fixtures are
committed; weights are not.

Adding a model? See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the implementation, fixture
and parity-test workflow.

## Roadmap

- [x] ResNet / ResNet-D / ResNeXt families with timm weight loading
- [x] AlexNet
- [x] `forward_intermediates` + `FeatureInfo` (timm 1.0 semantics)
- [x] Non-RGB input adaptation, classifier replacement, headless models
- [ ] SE-ResNet and EfficientNet families
- [ ] ConvNeXt
- [ ] Vision Transformer (ViT) and `Nlc` feature format

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
