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

`burn-im` includes **40 architectures** across **7 families** and supports **150+ pretrained checkpoint configurations** directly downloadable from the Hugging Face Hub (or loaded locally from `.safetensors`).

| Family | Architectures | Default Tag | Checkpoint Recipes & Pretrained Weights |
| ------ | ------------- | ----------- | --------------------------------------- |
| **ResNet** | `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152` | `a1_in1k` | `a1_in1k`, `a2_in1k`, `a3_in1k`, `tv_in1k`, `tv2_in1k`, `gluon_in1k`, `fb_ssl_*`, `fb_swsl_*`, `am_in1k`, `ra_in1k`, `bt_in1k` |
| **ResNet-D** | `resnet18d`, `resnet34d`, `resnet50d`, `resnet101d`, `resnet152d` | `ra2_in1k` | `ra2_in1k`, `ra4_e3600_r224_in1k`, `a1_in1k`, `a2_in1k`, `a3_in1k`, `gluon_in1k` |
| **ResNeXt & Wide ResNet** | `resnext50_32x4d`, `resnext101_32x8d`, `wide_resnet50_2`, `wide_resnet101_2` | `a1h_in1k` / `tv_in1k` | `a1h_in1k`, `a1_in1k`, `a2_in1k`, `a3_in1k`, `tv_in1k`, `tv2_in1k`, `gluon_in1k`, `fb_ssl_*`, `fb_swsl_*`, `fb_wsl_*`, `racm_in1k` |
| **ConvNeXt** | `convnext_atto`, `convnext_femto`, `convnext_pico`, `convnext_nano`, `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`, `convnext_xlarge` | `fb_in22k_ft_in1k` / `d1_in1k` / `d2_in1k` / `d1h_in1k` | `fb_in22k_ft_in1k`, `fb_in1k`, `fb_in22k`, `d1_in1k`, `d2_in1k`, `d1h_in1k`, `in12k`, `dinov3_lvd1689m`, `eupe_lvd1689m`, `clip_laion2b*` |
| **ConvNeXt-V2** | `convnextv2_atto`, `convnextv2_femto`, `convnextv2_pico`, `convnextv2_nano`, `convnextv2_tiny`, `convnextv2_small`, `convnextv2_base`, `convnextv2_large`, `convnextv2_huge` | `fcmae_ft_in22k_in1k` / `fcmae_ft_in1k` | `fcmae_ft_in22k_in1k`, `fcmae_ft_in1k`, `fcmae` (pretrain), 384/512 resolutions (`convnextv2_small` is architecture only) |
| **Vision Transformer (ViT)** | `vit_tiny_patch16_224`, `vit_small_patch16_224`, `vit_base_patch16_224`, `vit_large_patch16_224` | `augreg2_in21k_ft_in1k` (base), `augreg_in21k_ft_in1k` | `augreg2_in21k_ft_in1k`, `augreg_in21k_ft_in1k`, `augreg_in1k`, `orig_in21k_ft_in1k`, `dino`, `dino3` |
| **DeiT-III** | `deit_base_patch16_224` | `b3_in1k` | `b3_in1k`, `fb_in1k` |
| **DINOv2** | `vit_small_patch14_dinov2` | `lvd142m` | `lvd142m` (self-supervised foundation representation) |
| **SigLIP** | `vit_base_patch16_siglip_224` | `v2_webli` | `v2_webli`, `webli` (multimodal vision encoder) |
| **AlexNet** | `alexnet` | — | Architecture only (torchvision `.safetensors` can be loaded locally via `with_checkpoint`) |

<details>
<summary><strong>View all available pretrained checkpoints by architecture</strong></summary>

### ResNet
- `resnet18`: `a1_in1k` (default), `a2_in1k`, `a3_in1k`, `fb_ssl_yfcc100m_ft_in1k`, `fb_swsl_ig1b_ft_in1k`, `gluon_in1k`, `tv_in1k`
- `resnet34`: `a1_in1k` (default), `a2_in1k`, `a3_in1k`, `bt_in1k`, `gluon_in1k`, `ra4_e3600_r224_in1k`, `tv_in1k`
- `resnet50`: `a1_in1k` (default), `a1h_in1k`, `a2_in1k`, `a3_in1k`, `am_in1k`, `b1k_in1k`, `b2k_in1k`, `bt_in1k`, `c1_in1k`, `c2_in1k`, `d_in1k`, `fb_ssl_yfcc100m_ft_in1k`, `fb_swsl_ig1b_ft_in1k`, `gluon_in1k`, `ra_in1k`, `ram_in1k`, `tv2_in1k`, `tv_in1k`
- `resnet101`: `a1_in1k` (default), `a1h_in1k`, `a2_in1k`, `a3_in1k`, `gluon_in1k`, `tv2_in1k`, `tv_in1k`
- `resnet152`: `a1_in1k` (default), `a1h_in1k`, `a2_in1k`, `a3_in1k`, `gluon_in1k`, `tv2_in1k`, `tv_in1k`

### ResNet-D
- `resnet18d`: `ra2_in1k` (default), `ra4_e3600_r224_in1k`
- `resnet34d`: `ra2_in1k` (default)
- `resnet50d`: `ra2_in1k` (default), `a1_in1k`, `a2_in1k`, `a3_in1k`, `gluon_in1k`, `ra4_e3600_r224_in1k`
- `resnet101d`: `ra2_in1k` (default), `gluon_in1k`
- `resnet152d`: `ra2_in1k` (default), `gluon_in1k`

### ResNeXt & Wide ResNet
- `resnext50_32x4d`: `a1h_in1k` (default), `a1_in1k`, `a2_in1k`, `a3_in1k`, `gluon_in1k`, `ra_in1k`, `tv2_in1k`, `tv_in1k`
- `resnext101_32x8d`: `tv_in1k` (default), `fb_ssl_yfcc100m_ft_in1k`, `fb_swsl_ig1b_ft_in1k`, `fb_wsl_ig1b_ft_in1k`, `tv2_in1k`
- `wide_resnet50_2`: `tv_in1k` (default), `racm_in1k`, `tv2_in1k`
- `wide_resnet101_2`: `tv_in1k` (default), `tv2_in1k`

### ConvNeXt (V1)
- `convnext_atto`: `d2_in1k` (default)
- `convnext_femto`: `d1_in1k` (default)
- `convnext_pico`: `d1_in1k` (default)
- `convnext_nano`: `d1h_in1k` (default), `in12k`, `in12k_ft_in1k`, `r384_ad_in12k`
- `convnext_tiny`: `fb_in22k_ft_in1k` (default), `dinov3_lvd1689m`, `eupe_lvd1689m`, `fb_in1k`, `fb_in22k`
- `convnext_small`: `fb_in22k_ft_in1k` (default), `dinov3_lvd1689m`, `eupe_lvd1689m`, `fb_in1k`, `fb_in22k`
- `convnext_base`: `fb_in22k_ft_in1k` (default), `clip_laion2b`, `clip_laion2b_augreg`, `clip_laion2b_augreg_ft_in1k`, `clip_laion2b_augreg_ft_in12k`, `dinov3_lvd1689m`, `eupe_lvd1689m`, `fb_in1k`, `fb_in22k`
- `convnext_large`: `fb_in22k_ft_in1k` (default), `dinov3_lvd1689m`, `fb_in1k`, `fb_in22k`
- `convnext_xlarge`: `fb_in22k_ft_in1k` (default), `fb_in22k`, `fb_in22k_ft_in1k_384`

### ConvNeXt-V2
- `convnextv2_atto`: `fcmae_ft_in1k` (default), `fcmae`
- `convnextv2_femto`: `fcmae_ft_in1k` (default), `fcmae`
- `convnextv2_pico`: `fcmae_ft_in1k` (default), `fcmae`
- `convnextv2_nano`: `fcmae_ft_in22k_in1k` (default), `fcmae`, `fcmae_ft_in1k`, `fcmae_ft_in22k_in1k_384`
- `convnextv2_tiny`: `fcmae_ft_in22k_in1k` (default), `fcmae`, `fcmae_ft_in1k`, `fcmae_ft_in22k_in1k_384`
- `convnextv2_small`: *(architecture only)*
- `convnextv2_base`: `fcmae_ft_in22k_in1k` (default), `fcmae`, `fcmae_ft_in1k`, `fcmae_ft_in22k_in1k_384`
- `convnextv2_large`: `fcmae_ft_in22k_in1k` (default), `fcmae`, `fcmae_ft_in1k`, `fcmae_ft_in22k_in1k_384`
- `convnextv2_huge`: `fcmae_ft_in1k` (default), `fcmae`, `fcmae_ft_in22k_in1k_384`, `fcmae_ft_in22k_in1k_512`

### Vision Transformer (ViT, DeiT-III, DINOv2, SigLIP)
- `vit_tiny_patch16_224`: `augreg_in21k_ft_in1k` (default), `augreg_in1k`
- `vit_small_patch16_224`: `augreg_in21k_ft_in1k` (default), `augreg_in1k`, `dino`
- `vit_base_patch16_224`: `augreg2_in21k_ft_in1k` (default), `augreg_in21k_ft_in1k`, `augreg_in1k`, `orig_in21k_ft_in1k`, `dino`, `dino3`
- `vit_large_patch16_224`: `augreg_in21k_ft_in1k` (default), `augreg_in1k`
- `deit_base_patch16_224`: `b3_in1k` (default), `fb_in1k`
- `vit_small_patch14_dinov2`: `lvd142m` (default)
- `vit_base_patch16_siglip_224`: `v2_webli` (default), `webli`

</details>

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
| `image` | no      | Image decoding and preprocessing (`ImageProcessor`).           |
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

### Classify an image

With the `image` feature, bundled examples preprocess a JPEG/PNG with the checkpoint's own
`PretrainedCfg` and print top ImageNet labels (weights download on first run):

```bash
# Basic ResNet-50 inference (defaults to an image in examples/data/):
cargo run --release --example resnet50 --features image

# Or specify a model and image via the CLI:
cargo run --release --example classify --features image -- path/to/image.jpg --model resnet50.a1_in1k --top 5
```

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
| `convnext_tiny.fb_in22k_ft_in1k`   | `8.0e-5` (features), `1.2e-5` (logits) |
| `convnextv2_tiny.fcmae_ft_in22k_in1k` | `5.5e-6` (logits) |

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
- [x] ConvNeXt and ConvNeXt-V2 families
- [x] Vision Transformer (ViT) and `Nlc` feature format
- [ ] SE-ResNet and EfficientNet families

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
