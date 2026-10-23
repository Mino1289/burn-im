# Agent instructions

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before adding or changing a model.

## Ground rules

- Model code lives in `src/models/` and uses `burn::nn` layers only. No Python at
  runtime, no `pyo3`; keep it WASM-friendly.
- Keep the public API documented (`#![warn(missing_docs)]` is enabled).
- Golden fixtures (`tests/fixtures/golden/*.safetensors`) **are committed**: they are the
  shared `timm` reference. Model weights are never committed.
- Do not loosen the parity constants in `tests/golden.rs` (`1e-5` relative L2, `1e-4`
  max absolute). If a model misses the gate, fix the model.
- Adding a model means: implement, register in `src/registry.rs`, add it to
  `MODELS` in `scripts/generate_fixtures.py`, add a golden test, validate, commit the
  fixture, update the README.

## Checks before committing

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --doc
```

Network-dependent suites are ignored by default:

```bash
cargo test --test golden -- --ignored --nocapture
cargo test --test weights -- --ignored --nocapture
```
