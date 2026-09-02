"""Qualitative analysis on the 500-sample annotated subset (final.md §6.5).

Two deliverables, both from the finalized annotation + zero-shot predictions:

1. Failure / boundary cases per evidence channel (2-3 each), selected by
   deterministic rules so the picks are reproducible, not cherry-picked:
     visible_conflict   gold C, camc_reg misses on all 3 seeds (pred N);
                        prefer cases clip_fusion catches 3/3 (contrast) —
                        plus one 3/3 hit for honest "what works"
     event_provenance / external_knowledge
                        gold N boundary cases: every model abstains (N) and
                        the original caption shows the mismatch — the
                        "evidence is not in the pixels" finding; plus one
                        clip_fusion 3/3 C catch for external_knowledge
                        (entity-knowledge signal) when available
   temporal_provenance / other have n=3 each: listed in the CSV, no figure.

2. Attention visualization (camc_reg_seed42, last cross-attn layer, text
   [CLS] query over the 196 image patches) overlaid on the image for the
   visible_conflict cases.  Presented as "where the claim attends", never
   as reasoning evidence (writing red line, final.md §9).  Interpretation
   limits (full list in DISSERTATION_NOTES §6h): softmax weights are not
   contribution magnitudes; weights are 8-head averages; single-layer view
   (cross-layer attribution would need rollout/gradient methods); the
   display-only max normalisation amplifies near-uniform maps, so read
   "does mass fall on the evidence region / do hit and miss differ", not
   "where is the brightest block".

Outputs -> analysis/qualitative/:
    failure_pool.csv     every case matching a selection rule (full pool)
    cases_selected.csv   the picked cases with all display fields
    fig_visible_conflict.png   image | attention | annotation text
    fig_event_provenance.png   image | annotation text
    fig_external_knowledge.png image | annotation text

Run from the project root:
    conda run -n hf_latest python scripts/make_qualitative.py
"""

from __future__ import annotations

import csv
import json
import sys
import textwrap
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from zero_shot_newsclippings import (  # noqa: E402
    build_preprocessing, load_checkpoint, pick_device)
from src.configs.camc_reg_seed42 import config as camc_reg_cfg  # noqa: E402

PACKETS = ["pilot_seed42", "formal_seed43"]
MODELS = ["camc_reg", "clip_fusion", "text_only"]
SEEDS = [42, 43, 44]
OUT = PROJECT_ROOT / "analysis" / "qualitative"
SHORT = {"entailment": "E", "neutral": "N", "contradiction": "C"}


def read_csv_rows(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_cases() -> list[dict]:
    """One merged record per annotated sample."""
    cases = {}
    for pk in PACKETS:
        base = PROJECT_ROOT / "data" / "newsclippings" / pk
        for s in json.loads((base / "pilot_samples.json").read_text()):
            cases[(pk, str(s["sample_id"]))] = {
                "packet": pk, "sample_id": str(s["sample_id"]),
                "caption": s["caption"],
                "original_caption": s.get("original_caption") or "",
                "original_label": s["original_label"],
                "official_split": s["official_split"],
                "image_path": s["image_path"],
            }
        for r in read_csv_rows(base / "pilot_annotations.csv"):
            c = cases[(pk, r["sample_id"])]
            c["relation"] = SHORT[r["relation_label"]]
            c["channel"] = r["evidence_channel"]
            c["confidence"] = r["annotator_confidence"]
    for pk in PACKETS:
        base = PROJECT_ROOT / "data" / "newsclippings" / pk / "zero_shot"
        for m in MODELS:
            for seed in SEEDS:
                rows = read_csv_rows(base / f"{m}_seed{seed}_predictions.csv")
                for r in rows:
                    c = cases[(pk, r["sample_id"])]
                    c[f"{m}_s{seed}"] = SHORT[r["pred_label"]]
                    c[f"{m}_s{seed}_pN"] = float(r["p_neutral"])
    out = list(cases.values())
    for c in out:
        for m in MODELS:
            votes = [c[f"{m}_s{s}"] for s in SEEDS]
            c[f"{m}_consensus"] = votes[0] if len(set(votes)) == 1 else "mixed"
            c[f"{m}_votes"] = "/".join(votes)
    return out


def select_cases(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    """Apply the documented selection rules; return (pool, selected)."""
    pool, selected = [], []

    def tag(c, role):
        r = dict(c)
        r["role"] = role
        pool.append(r)
        return r

    vc = [c for c in cases if c["channel"] == "visible_conflict"]
    misses = sorted(
        (c for c in vc if c["relation"] == "C"
         and c["camc_reg_consensus"] == "N"),
        key=lambda c: -sum(c[f"camc_reg_s{s}_pN"] for s in SEEDS))
    for c in misses:
        tag(c, "vc_miss_clip_catch"
            if c["clip_fusion_consensus"] == "C" else "vc_miss")
    # image-driven hits first: a hit where text_only also says C is a
    # text-artifact catch, not evidence of image use; camc-unique catches
    # (clip_fusion missing too) are the strongest interaction-layer cases
    rank = {"E": 0, "N": 0, "mixed": 1, "C": 2}  # clean non-C contrast first
    hits = sorted(
        (c for c in vc if c["relation"] == "C"
         and c["camc_reg_consensus"] == "C"),
        key=lambda c: (rank[c["text_only_consensus"]],
                       rank[c["clip_fusion_consensus"]]))
    for c in hits:
        tag(c, "vc_hit")
    # 2 misses (clip-contrast first) + 1 hit
    contrast = [r for r in pool if r["role"] == "vc_miss_clip_catch"]
    plain = [r for r in pool if r["role"] == "vc_miss"]
    selected += (contrast + plain)[:2]
    selected += [r for r in pool if r["role"] == "vc_hit"][:1]

    for ch, prefix in (("event_provenance", "ep"),
                       ("external_knowledge", "ek")):
        chc = [c for c in cases if c["channel"] == ch]
        # low-confidence excluded; high preferred but these channels are
        # medium-heavy by nature (blind relation labelling cannot see the
        # provenance mismatch either — that is the point of the channel)
        boundary = [c for c in chc if c["relation"] == "N"
                    and all(c[f"{m}_consensus"] == "N" for m in MODELS)
                    and c["confidence"] in ("high", "medium")]
        # high-confidence first, then strongest "looks plausible" examples
        boundary.sort(key=lambda c: (
            c["confidence"] != "high",
            -sum(c[f"camc_reg_s{s}_pN"] for s in SEEDS)))
        for c in boundary:
            tag(c, f"{prefix}_boundary")
        selected += [r for r in pool if r["role"] == f"{prefix}_boundary"][:2]
        if ch == "external_knowledge":
            catches = [c for c in chc
                       if c["clip_fusion_consensus"] == "C"]
            for c in catches:
                tag(c, "ek_clip_catch")
            selected += [r for r in pool if r["role"] == "ek_clip_catch"][:1]

    for c in cases:  # tiny channels: record only
        if c["channel"] in ("temporal_provenance", "other"):
            tag(c, f"{c['channel']}_all")
    return pool, selected


@torch.no_grad()
def attention_maps(selected: list[dict]) -> dict:
    """Last-layer text-[CLS] -> image-patch attention for camc_reg_seed42."""
    device = pick_device()
    tokenizer, max_length, transform = build_preprocessing(camc_reg_cfg)
    model = load_checkpoint("camc_reg_seed42", camc_reg_cfg, device)
    maps = {}
    for c in selected:
        if c["channel"] != "visible_conflict":
            continue
        img = Image.open(PROJECT_ROOT / c["image_path"]).convert("RGB")
        enc = tokenizer(c["caption"], truncation=True, max_length=max_length,
                        return_tensors="pt")
        _, attn = model(transform(img).unsqueeze(0).to(device),
                        enc["input_ids"].to(device),
                        enc["attention_mask"].to(device))
        # attn[-1]: [1, N_txt, 197] head-averaged; row 0 = text [CLS];
        # drop image CLS key, renormalise over the 14x14 patch grid
        cls_row = attn[-1][0, 0, 1:].reshape(14, 14).cpu()
        cls_row = cls_row / cls_row.max()
        maps[(c["packet"], c["sample_id"])] = cls_row.numpy()
    del model
    return maps


def wrap(s: str, width=58) -> str:
    return "\n".join(textwrap.wrap(s, width=width))


def case_text(c: dict) -> str:
    lines = [
        f"[{c['role']}]  {c['packet']}/{c['sample_id']}  "
        f"({c['official_split']})",
        f"claim: {wrap(c['caption'])}",
    ]
    if c["original_label"] == "falsified":
        lines.append(f"original: {wrap(c['original_caption'])}")
    lines.append(
        f"gold relation {c['relation']}  |  channel {c['channel']}")
    lines.append("  ".join(
        f"{m}: {c[f'{m}_votes']}" for m in MODELS))
    return "\n".join(lines)


def make_figures(selected: list[dict], maps: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    groups = {
        "visible_conflict": [c for c in selected
                             if c["channel"] == "visible_conflict"],
        "event_provenance": [c for c in selected
                             if c["channel"] == "event_provenance"],
        "external_knowledge": [c for c in selected
                               if c["channel"] == "external_knowledge"],
    }
    for ch, rows in groups.items():
        if not rows:
            continue
        with_attn = ch == "visible_conflict"
        ncols = 3 if with_attn else 2
        fig, axes = plt.subplots(
            len(rows), ncols,
            figsize=(4.2 * ncols + 1.5, 3.4 * len(rows)),
            gridspec_kw={"width_ratios": [1, 1, 1.7] if with_attn
                         else [1, 1.9]})
        axes = np.atleast_2d(axes)
        for ri, c in enumerate(rows):
            img = Image.open(
                PROJECT_ROOT / c["image_path"]).convert("RGB")
            axes[ri, 0].imshow(img)
            axes[ri, 0].set_title(
                f"{c['packet']}/{c['sample_id']}", fontsize=9)
            if with_attn:
                m = maps[(c["packet"], c["sample_id"])]
                axes[ri, 1].imshow(img.resize((224, 224)))
                axes[ri, 1].imshow(
                    np.kron(m, np.ones((16, 16))), cmap="inferno",
                    alpha=0.55, extent=(0, 224, 224, 0))
                axes[ri, 1].set_title(
                    "text[CLS] attention, last layer", fontsize=9)
            axes[ri, ncols - 1].text(
                0.0, 0.97, case_text(c), fontsize=8.5, family="monospace",
                va="top", transform=axes[ri, ncols - 1].transAxes)
            for ax in axes[ri]:
                ax.set_xticks([]), ax.set_yticks([])
            axes[ri, ncols - 1].axis("off")
        fig.tight_layout()
        fig.savefig(OUT / f"fig_{ch}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"fig_{ch}.png: {len(rows)} cases")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    pool, selected = select_cases(cases)

    cols = ["role", "packet", "sample_id", "official_split", "relation",
            "channel", "confidence", "original_label", "caption",
            "original_caption"] + [f"{m}_votes" for m in MODELS]
    for name, rows in (("failure_pool.csv", pool),
                       ("cases_selected.csv", selected)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    from collections import Counter
    print("pool roles:", dict(Counter(r["role"] for r in pool)))
    print("selected:", [(r["role"], r["packet"], r["sample_id"])
                        for r in selected])

    maps = attention_maps(selected)
    make_figures(selected, maps)
    print(f"\nWrote {OUT}/failure_pool.csv, cases_selected.csv + figures")


if __name__ == "__main__":
    main()
