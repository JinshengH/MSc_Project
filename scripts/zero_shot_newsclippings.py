"""Zero-shot E/N/C inference on the NewsCLIPpings pilot sample.

Loads the SNLI-VE-trained checkpoints (results/<exp>/best.pt) and predicts
on the pilot packet produced by prepare_newsclippings_pilot.py.  This is the
pilot-week output-distribution check (final.md v2 §7.2): if a model collapses
to one class on news-domain data, that is a no-go signal.

Preprocessing (tokenizer, max_length, image transform) is deliberately
identical to src/train.py so the transfer gap is not confounded by input
processing.

Run from the project root, for example:
    conda run -n hf_latest python scripts/zero_shot_newsclippings.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import build_model  # noqa: E402

# packet dir name as optional CLI arg; defaults to the pilot packet
PACKET = sys.argv[1] if len(sys.argv) > 1 else "pilot_seed42"
PILOT_DIR = PROJECT_ROOT / "data" / "newsclippings" / PACKET
RESULTS_DIR = PROJECT_ROOT / "results"
BATCH_SIZE = 32
LABEL_NAMES = ["entailment", "neutral", "contradiction"]

# Every trained experiment with a local best.pt: one config module each
# (project convention), imported dynamically. camc_seed42 is the historical
# pre-regularisation run; the 5 current models each have seeds 42/43/44.
import importlib  # noqa: E402

EXPERIMENT_NAMES = ["camc_seed42"] + [
    f"{model}_seed{seed}"
    for model in ("camc_reg", "text_only", "late_fusion",
                  "clip_fusion", "image_only")
    for seed in (42, 43, 44)
] + [f"ablation_{v}_seed42" for v in ("a1", "a3", "a4")]  # Exp 4 (guide §5.2)
EXPERIMENT_NAMES += [f"clip_camc_seed{s}" for s in (42, 43, 44)]  # camc-on-CLIP

# Optional extra CLI args = experiment-name prefixes to run (incremental
# mode for newly trained checkpoints); summary.json is only rewritten on
# full runs so it always reflects every experiment.
ONLY = sys.argv[2:]
if ONLY:
    EXPERIMENT_NAMES = [n for n in EXPERIMENT_NAMES
                        if any(n.startswith(p) for p in ONLY)]
EXPERIMENTS = [
    (name, importlib.import_module(f"src.configs.{name}").config)
    for name in EXPERIMENT_NAMES
]

# Same normalisation table and text settings as src/train.py
# build_dataloaders; preprocessing is derived per experiment from the same
# config fields (tokenizer_name / image_norm) the training run used.
NORM_STATS = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "clip": ([0.48145466, 0.4578275, 0.40821073],
             [0.26862954, 0.26130258, 0.27577711]),
}


def build_preprocessing(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    max_length = min(128, tokenizer.model_max_length)
    mean, std = NORM_STATS[cfg.image_norm]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return tokenizer, max_length, transform


def pick_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def load_pilot_samples() -> list[dict]:
    samples_json = PILOT_DIR / "pilot_samples.json"
    if not samples_json.exists():
        raise FileNotFoundError(
            f"{samples_json} not found. Run prepare_newsclippings_pilot.py "
            "in full mode (VisualNews extracted) first.")
    samples = json.loads(samples_json.read_text())
    for sample in samples:
        if not sample.get("caption") or not sample.get("image_path"):
            raise RuntimeError(
                f"Sample {sample['sample_id']} lacks caption/image_path; "
                "the packet is incomplete")
    return samples


# Frozen from_pretrained encoders: their weights are identical to the
# checkpoint's, but transformers renames internal ViT/BERT modules across
# versions (HPC checkpoints use the newer q_proj naming), so we skip these
# prefixes and load only the trainable custom modules. "clip." is
# clip_fusion's frozen CLIP tower (loaded from the converted local copy,
# byte-identical to what the HPC run trained against).
FROZEN_PREFIXES = ("ie.", "te.", "clip.")


def load_checkpoint(exp_name: str, cfg, device: torch.device):
    checkpoint = RESULTS_DIR / exp_name / "best.pt"
    model = build_model(cfg)
    state_dict = torch.load(checkpoint, map_location=device)
    trainable_sd = {
        k: v for k, v in state_dict.items()
        if not k.startswith(FROZEN_PREFIXES)}
    missing, unexpected = model.load_state_dict(trainable_sd, strict=False)
    stray_missing = [k for k in missing if not k.startswith(FROZEN_PREFIXES)]
    if stray_missing or unexpected:
        raise RuntimeError(
            f"{exp_name}: trainable weights do not line up; "
            f"missing={stray_missing[:5]} unexpected={list(unexpected)[:5]}")
    trainable_keys = {
        k for k, p in model.named_parameters() if p.requires_grad}
    not_loaded = trainable_keys - trainable_sd.keys()
    if not_loaded:
        raise RuntimeError(
            f"{exp_name}: trainable params not in checkpoint: "
            f"{sorted(not_loaded)[:5]}")
    print(f"Loaded {len(trainable_sd)} trainable tensors from {checkpoint}")
    model.to(device)
    model.eval()
    return model


def predict(model, samples: list[dict], tokenizer, max_length: int,
            image_transform, device: torch.device) -> list[dict]:
    rows = []
    with torch.no_grad():
        for start in range(0, len(samples), BATCH_SIZE):
            batch = samples[start:start + BATCH_SIZE]
            encoded = tokenizer(
                [s["caption"] for s in batch],
                truncation=True, max_length=max_length,
                padding=True, return_tensors="pt")
            pixel_values = torch.stack([
                image_transform(
                    Image.open(PROJECT_ROOT / s["image_path"]).convert("RGB"))
                for s in batch])

            logits, _ = model(
                pixel_values.to(device),
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device))
            probs = torch.softmax(logits, dim=1).cpu()
            preds = probs.argmax(dim=1)

            for sample, pred, prob in zip(batch, preds, probs):
                rows.append({
                    "sample_id": sample["sample_id"],
                    "caption_id": sample["caption_id"],
                    "image_id": sample["image_id"],
                    "original_label": sample["original_label"],
                    "official_split": sample["official_split"],
                    "pred_label": LABEL_NAMES[int(pred)],
                    "p_entailment": f"{prob[0]:.6f}",
                    "p_neutral": f"{prob[1]:.6f}",
                    "p_contradiction": f"{prob[2]:.6f}",
                })
    return rows


def summarise(exp_name: str, rows: list[dict]) -> dict:
    summary = {"experiment": exp_name, "n": len(rows)}
    overall = Counter(r["pred_label"] for r in rows)
    summary["pred_distribution"] = {
        name: overall.get(name, 0) for name in LABEL_NAMES}
    for original in ("pristine", "falsified"):
        counts = Counter(
            r["pred_label"] for r in rows if r["original_label"] == original)
        summary[f"pred_distribution_{original}"] = {
            name: counts.get(name, 0) for name in LABEL_NAMES}
    # Objective stratification layer (addendum 2 §2): falsified predictions
    # broken down by the official NewsCLIPpings generation mechanism.
    by_split = {}
    for split in sorted({r["official_split"] for r in rows}):
        counts = Counter(
            r["pred_label"] for r in rows
            if r["official_split"] == split and r["original_label"] == "falsified")
        by_split[split] = {name: counts.get(name, 0) for name in LABEL_NAMES}
    summary["pred_distribution_falsified_by_official_split"] = by_split
    return summary


def main() -> None:
    device = pick_device()
    print("Using device:", device)
    samples = load_pilot_samples()
    print(f"Pilot samples: {len(samples)}")

    out_dir = PILOT_DIR / "zero_shot"
    out_dir.mkdir(exist_ok=True)

    summaries = []
    for exp_name, cfg in EXPERIMENTS:
        print(f"\n=== {exp_name} ===")
        tokenizer, max_length, image_transform = build_preprocessing(cfg)
        print(f"preprocessing: tokenizer={cfg.tokenizer_name} "
              f"max_length={max_length} image_norm={cfg.image_norm}")
        model = load_checkpoint(exp_name, cfg, device)
        rows = predict(
            model, samples, tokenizer, max_length, image_transform, device)
        del model

        csv_path = out_dir / f"{exp_name}_predictions.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        summary = summarise(exp_name, rows)
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    if not ONLY:
        (out_dir / "summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote predictions to {out_dir}"
          + ("" if ONLY else " (incl. summary.json)"))


if __name__ == "__main__":
    main()
