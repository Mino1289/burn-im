"""Generate golden fixtures from timm for burn-im parity tests.

Usage:
    uv run python scripts/generate_fixtures.py

Writes one `.safetensors` file per architecture under `tests/fixtures/golden/`
containing the deterministic input, timm's logits, final features and every
intermediate feature level.
"""

from pathlib import Path

import torch
import timm
from safetensors.torch import save_file

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "golden"

# Architecture -> timm tag. Weights are pulled from the Hugging Face Hub.
MODELS = {
    "resnet18": "a1_in1k",
    "resnet50": "a1_in1k",
    "resnet50d": "ra2_in1k",
    "resnext50_32x4d": "a1h_in1k",
}

# Dilated variants: `output_stride` below 32 folds the stride of the last stages into the
# dilation of their convolutions. `resnet50d` also covers the average-pool downsample path.
OUTPUT_STRIDES = {
    ("resnet50", "a1_in1k"): [16, 8],
    ("resnet50d", "ra2_in1k"): [16, 8],
}



def generate(arch: str, tag: str) -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 3, 64, 64)

    model = timm.create_model(f"{arch}.{tag}", pretrained=True).eval()
    with torch.no_grad():
        logits = model(x)
        pre_logits = model.forward_head(model.forward_features(x), pre_logits=True)
        final, intermediates = model.forward_intermediates(
            x,
            indices=tuple(range(len(model.feature_info))),
            stop_early=False,
            output_fmt="NCHW",
            intermediates_only=False,
        )

    tensors = {
        "input": x.contiguous(),
        "logits": logits.contiguous(),
        "features": final.contiguous(),
        "pre_logits": pre_logits.contiguous(),
    }

    if arch == "resnet18":
        # One-channel input: timm adapts the 3-channel stem by summing the kernels.
        x1 = torch.randn(1, 1, 64, 64)
        model1 = timm.create_model(f"{arch}.{tag}", pretrained=True, in_chans=1).eval()
        with torch.no_grad():
            tensors["input_1ch"] = x1
            tensors["logits_1ch"] = model1(x1)
    for idx, level in enumerate(intermediates):
        tensors[f"level_{idx}"] = level.contiguous()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{arch}.{tag}.safetensors"
    save_file({k: v.float().clone() for k, v in tensors.items()}, str(out))
    print(f"{out.name}: logits {tuple(logits.shape)}, {len(intermediates)} levels")


def generate_output_stride(arch: str, tag: str, output_stride: int) -> None:
    """Golden activations of a dilated variant (`output_stride` 16 or 8)."""
    torch.manual_seed(0)
    x = torch.randn(1, 3, 64, 64)

    model = timm.create_model(
        f"{arch}.{tag}", pretrained=True, output_stride=output_stride
    ).eval()
    with torch.no_grad():
        logits = model(x)
        pre_logits = model.forward_head(model.forward_features(x), pre_logits=True)
        final, intermediates = model.forward_intermediates(
            x,
            indices=tuple(range(len(model.feature_info))),
            stop_early=False,
            output_fmt="NCHW",
            intermediates_only=False,
        )

    tensors = {
        "input": x,
        "logits": logits,
        "features": final,
        "pre_logits": pre_logits,
    }
    for idx, level in enumerate(intermediates):
        tensors[f"level_{idx}"] = level

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{arch}.{tag}.os{output_stride}.safetensors"
    save_file({k: v.float().clone() for k, v in tensors.items()}, str(out))
    print(
        f"{out.name}: logits {tuple(logits.shape)}, "
        f"{len(intermediates)} levels, {tuple(final.shape)} features"
    )


def generate_alexnet() -> None:
    """AlexNet was dropped from timm 1.0; torchvision is the reference implementation.

    The 244 MB checkpoint is written under `target/fixtures/` (not committed); the golden
    activations themselves are small and live with the other fixtures.
    """
    import torchvision

    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224)

    model = torchvision.models.alexnet(
        weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1
    ).eval()
    with torch.no_grad():
        logits = model(x)
        level_0 = model.features[:3](x)
        level_1 = model.features[3:6](level_0)
        level_2 = model.features[6:](level_1)

    golden = {
        "input": x,
        "logits": logits,
        "features": level_2,
        "level_0": level_0,
        "level_1": level_1,
        "level_2": level_2,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.float().clone() for k, v in golden.items()},
        str(OUT_DIR / "alexnet.tv_in1k.safetensors"),
    )

    weights_dir = OUT_DIR.parent.parent.parent / "target" / "fixtures"
    weights_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.float().clone() for k, v in model.state_dict().items()},
        str(weights_dir / "alexnet.safetensors"),
    )
    print("alexnet.tv_in1k: logits", tuple(logits.shape), "- weights in target/fixtures")


def main() -> None:
    for arch, tag in MODELS.items():
        generate(arch, tag)
    for (arch, tag), strides in OUTPUT_STRIDES.items():
        for output_stride in strides:
            generate_output_stride(arch, tag, output_stride)
    generate_alexnet()


if __name__ == "__main__":
    main()
