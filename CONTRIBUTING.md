# Contributing

Thanks for helping `burn-im` grow. The important part of this guide is the **golden
fixture** workflow: each supported model ships a small `.safetensors` file with the
reference outputs of the official `timm` checkpoint. That is what lets anyone run the
numerical-parity tests with only Rust and a network connection — no Python, no PyTorch.

## Setup

```bash
cargo build
cargo test
```

Python is only needed to **generate** fixtures. See the Development section of the
[README](README.md#development) for the `uv` setup.

## Adding a model

### 1. Implement the architecture

- Put the model in `src/models/<family>.rs`.
- Use `burn::nn` layers only: no hand-rolled convolutions, no Python at runtime, no
  `pyo3`. Keep the model code WASM-friendly.
- Reuse `crate::layers::{pool, head}` where possible.
- Expose a config with builders (`with_*`), a `new(config, device)` constructor, and
  presets such as `resnet50(num_classes)`.
- Implement `VisionModel` and document every public item (`#![warn(missing_docs)]`).

Feature metadata must match `timm` exactly: same number of levels, same channels and same
reduction factors, otherwise the intermediate-feature comparison will fail.

### 2. Register it

Add an entry to `MODELS` in `src/registry.rs`:

```rust
ModelInfo {
    arch: "resnet34",
    family: resnet(...),
    default_tag: Some("a1_in1k"),
    tags: &["a1_in1k", "tv_in1k"],
    num_features: 512,
},
```

`tags` powers `list_pretrained`; list only tags you have actually validated. If the
architecture has no Hub checkpoints (like AlexNet), use `default_tag: None, tags: &[]`.

### 3. Generate the reference fixture

Add the architecture to `MODELS` in `scripts/generate_fixtures.py`:

```python
MODELS = {
    "resnet18": "a1_in1k",
    # ...
    "resnet34": "a1_in1k",
}
```

Then:

```bash
uv run python scripts/generate_fixtures.py
```

The script runs the official checkpoint in `eval` mode on a deterministic input
(`torch.manual_seed(0)`, `1×3×64×64`) and writes
`tests/fixtures/golden/<arch>.<tag>.safetensors` with:

| key         | content                                     |
| ----------- | ------------------------------------------- |
| `input`     | deterministic input                         |
| `logits`    | classifier output                           |
| `features`  | `forward_features` output                   |
| `pre_logits`| pooled embedding                            |
| `level_N`   | each `forward_intermediates` level          |

Keep the input at 64×64 so the fixture stays small (a few hundred KB to ~1 MB). For
architectures that only exist in `torchvision` (AlexNet), follow the `generate_alexnet`
function: the golden tensors are small and committed, the 244 MB weights go to
`target/fixtures/` and are ignored.

### 4. Add the tests

In `tests/golden.rs`:

```rust
#[test]
#[ignore = "downloads pretrained weights from the Hugging Face Hub"]
fn resnet34_matches_timm() {
    check("resnet34", "a1_in1k");
}
```

Also add a weight-loading test in `tests/weights.rs` (load report, output shape,
head/input adaptation if relevant).

### 5. Validate

```bash
cargo test --test golden -- --ignored --nocapture
cargo test --test weights -- --ignored --nocapture
```

Parity gate, per compared tensor:

- relative L2 error `< 1e-5`,
- max absolute difference `< 1e-4` (the f32 accumulation noise floor).

If a model misses the gate, fix the implementation — do not loosen the constants. The
printed values tell you where the error appears (logits, features, or a specific level).

### 6. Commit the fixture

Commit the code **and** the `.safetensors` fixture. The fixture is the shared reference:
once it is on `main`, every contributor can run the parity tests without installing
Python or PyTorch.

Do **not** commit model weights. They are downloaded from the Hub at test time and cached
locally; `target/` is ignored.

### 7. Document it

- Add the architecture to the supported-models table in `README.md`.
- Update the roadmap if you complete an item.

## Before opening a PR

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --doc
```

The golden and Hub-loading tests are `#[ignore]`d because they need network access; run
them whenever you touch a model, the loader or the registry.

## Commit messages

Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `chore:`.
