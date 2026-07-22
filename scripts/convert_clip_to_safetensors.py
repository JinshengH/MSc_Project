"""Convert the OpenAI CLIP checkpoint to safetensors, once, on the login node.

`openai/clip-vit-base-patch16` only ships `pytorch_model.bin`.  transformers
refuses to `torch.load` a .bin unless torch >= 2.6 (CVE-2025-32434), which
the `msc` env does not satisfy, so `CLIPModel.from_pretrained` fails at job
start.  ViT/BERT are unaffected because their repos ship safetensors.

This script downloads the snapshot, re-serialises the weights as
`model.safetensors`, and copies the config/tokenizer/preprocessor files into
a local model directory that `clip_fusion_seed42` loads from.  The load here
is an explicit `weights_only=True` state-dict load of an official OpenAI
checkpoint, not an arbitrary pickle.

Run once per machine, from the project root:
    python scripts/convert_clip_to_safetensors.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import paths  # noqa: E402

REPO_ID = "openai/clip-vit-base-patch16"
# Files needed alongside the weights for from_pretrained to work offline
SIDECAR_FILES = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]


def clip_model_dir() -> Path:
    return paths.scratch_project_path(
        "models", "clip-vit-base-patch16",
        local="models/clip-vit-base-patch16")


def main() -> None:
    out_dir = clip_model_dir()
    weights_path = out_dir / "model.safetensors"
    if weights_path.exists():
        print(f"Already converted: {weights_path}")
        return

    print(f"Downloading {REPO_ID} ...")
    snapshot = Path(snapshot_download(REPO_ID))

    bin_path = snapshot / "pytorch_model.bin"
    if not bin_path.exists():
        raise FileNotFoundError(f"{bin_path} not found in snapshot")

    state_dict = torch.load(bin_path, weights_only=True, map_location="cpu")
    # safetensors rejects shared storage; clone to be explicit and safe
    state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, weights_path, metadata={"format": "pt"})
    print(f"Wrote {len(state_dict)} tensors -> {weights_path}")

    for name in SIDECAR_FILES:
        src = snapshot / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
        else:
            print(f"  (skipped missing {name})")
    print(f"Model directory ready: {out_dir}")


if __name__ == "__main__":
    main()
