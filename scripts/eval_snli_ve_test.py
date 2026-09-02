"""One-shot SNLI-VE test evaluation (Exp 1 final numbers).

Runs every trained checkpoint over the full original test split (17,901
pairs, local parquet shards in data/snli_ve_test/) and reports metrics on:

  1. corrected test (primary, final.md §3.1): the e-SNLI-VE re-annotated
     subset (14,740 pairs; neutral-class labels corrected, 6 duplicate keys
     with inconsistent labels dropped), joined by Flickr30kID + normalised
     hypothesis against data/esnlive_corrected/esnlive_test.csv;
  2. original test (comparison column): inherited SNLI labels.

Reuses the validated head-replication path from fsr_full_test.py (same
two-phase design: frozen towers encode unique images/hypotheses once into
a local cache; each checkpoint's trainable modules run over the cache).

Run from the project root:
    conda run -n hf_latest python scripts/eval_snli_ve_test.py
"""

from __future__ import annotations

import csv
import io
import sys
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image
from sklearn.metrics import f1_score, recall_score
from transformers import AutoTokenizer, CLIPModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import fsr_full_test as fsr  # noqa: E402  (validated head replication)
from src.models.encoders import ImageEncoder, TextEncoder  # noqa: E402

TEST_PARQUET_DIR = PROJECT_ROOT / "data" / "snli_ve_test"
CORRECTED_CSV = PROJECT_ROOT / "data" / "esnlive_corrected" / "esnlive_test.csv"
OUT_DIR = PROJECT_ROOT / "analysis" / "snli_ve_test_eval"
CACHE_DIR = OUT_DIR / "cache"

LABELS = ["entailment", "neutral", "contradiction"]
LAB_ID = {name: i for i, name in enumerate(LABELS)}
BATCH_IMAGES = 32
BATCH_TEXT = 256
BATCH_HEAD = 256


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).split()).lower()


def load_rows() -> list[dict]:
    rows = []
    for f in sorted(TEST_PARQUET_DIR.glob("*.parquet")):
        t = pq.read_table(f)
        for fn, img, hyp, lab in zip(
                t["filename"].to_pylist(), t["image"].to_pylist(),
                t["hypothesis"].to_pylist(), t["label"].to_pylist()):
            rows.append({"filename": fn, "image_bytes": img["bytes"],
                         "hypothesis": hyp, "label": int(lab)})
    if not rows:
        raise RuntimeError(f"No parquet shards in {TEST_PARQUET_DIR}")
    return rows


def attach_corrected(rows: list[dict]) -> int:
    """Set row['corrected_label'] where the corrected test provides one."""
    corrected = {}
    for r in csv.DictReader(CORRECTED_CSV.open()):
        corrected[(r["Flickr30kID"], norm(r["hypothesis"]))] = \
            LAB_ID[r["gold_label"]]
    # keys whose original rows disagree on the label are ambiguous: drop
    seen: dict[tuple, set] = {}
    for row in rows:
        key = (row["filename"], norm(row["hypothesis"]))
        seen.setdefault(key, set()).add(row["label"])
    ambiguous = {k for k, labs in seen.items() if len(labs) > 1}

    n = 0
    for row in rows:
        key = (row["filename"], norm(row["hypothesis"]))
        if key in corrected and key not in ambiguous:
            row["corrected_label"] = corrected[key]
            n += 1
        else:
            row["corrected_label"] = None
    return n


def build_caches(rows: list[dict], device: torch.device) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    todo = [n for n in ("vit_seq", "bert_seq", "clip_img", "clip_txt",
                        "clip_img_seq", "clip_txt_seq")
            if not (CACHE_DIR / f"{n}.pt").exists()]
    if not todo:
        print("Test caches already present")
        return
    print(f"Building test caches: {todo}")

    img_bytes = {}
    for r in rows:
        img_bytes.setdefault(r["filename"], r["image_bytes"])
    uniq_imgs = sorted(img_bytes)
    uniq_hyps = sorted({norm(r["hypothesis"]) for r in rows})
    hyp_text = {norm(r["hypothesis"]): r["hypothesis"] for r in rows}
    print(f"unique images: {len(uniq_imgs)} | unique hypotheses: {len(uniq_hyps)}")

    if "vit_seq" in todo or "clip_img" in todo or "clip_img_seq" in todo:
        vit = ImageEncoder().to(device).eval() if "vit_seq" in todo else None
        clip = (CLIPModel.from_pretrained(str(fsr.CLIP_DIR)).to(device).eval()
                if "clip_img" in todo or "clip_img_seq" in todo else None)
        vit_out, clip_out, clip_seq_out = [], [], []
        with torch.no_grad():
            for i in range(0, len(uniq_imgs), BATCH_IMAGES):
                pil = [Image.open(io.BytesIO(img_bytes[k])).convert("RGB")
                       for k in uniq_imgs[i:i + BATCH_IMAGES]]
                if vit is not None:
                    px = torch.stack([fsr.IMAGENET_TF(p) for p in pil]).to(device)
                    vit_out.append(vit(px).half().cpu())
                if clip is not None:
                    px = torch.stack([fsr.CLIP_TF(p) for p in pil]).to(device)
                    v = clip.vision_model(pixel_values=px)
                    if "clip_img" in todo:
                        clip_out.append(clip.visual_projection(
                            v.pooler_output).half().cpu())
                    if "clip_img_seq" in todo:
                        clip_seq_out.append(clip.visual_projection(
                            clip.vision_model.post_layernorm(
                                v.last_hidden_state)).half().cpu())
                if (i // BATCH_IMAGES) % 20 == 0:
                    print(f"  images {i}/{len(uniq_imgs)}")
        if vit is not None:
            torch.save({"ids": uniq_imgs, "feat": torch.cat(vit_out)},
                       CACHE_DIR / "vit_seq.pt")
        if "clip_img" in todo:
            torch.save({"ids": uniq_imgs, "feat": torch.cat(clip_out)},
                       CACHE_DIR / "clip_img.pt")
        if "clip_img_seq" in todo:
            torch.save({"ids": uniq_imgs, "feat": torch.cat(clip_seq_out)},
                       CACHE_DIR / "clip_img_seq.pt")
        del vit, clip

    if "bert_seq" in todo:
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        texts = [hyp_text[h] for h in uniq_hyps]
        max_len = min(128, max(len(tok(t)["input_ids"]) for t in texts))
        print(f"  BERT pad length: {max_len}")
        bert = TextEncoder().to(device).eval()
        feats = []
        with torch.no_grad():
            for i in range(0, len(texts), BATCH_TEXT):
                enc = tok(texts[i:i + BATCH_TEXT], truncation=True,
                          max_length=max_len, padding="max_length",
                          return_tensors="pt").to(device)
                feats.append(bert(enc["input_ids"],
                                  enc["attention_mask"]).half().cpu())
        torch.save({"ids": uniq_hyps, "feat": torch.cat(feats)},
                   CACHE_DIR / "bert_seq.pt")
        del bert

    if "clip_txt" in todo:
        tok = AutoTokenizer.from_pretrained(str(fsr.CLIP_DIR))
        texts = [hyp_text[h] for h in uniq_hyps]
        clip = CLIPModel.from_pretrained(str(fsr.CLIP_DIR)).to(device).eval()
        feats = []
        with torch.no_grad():
            for i in range(0, len(texts), BATCH_TEXT):
                enc = tok(texts[i:i + BATCH_TEXT], truncation=True,
                          max_length=77, padding="max_length",
                          return_tensors="pt").to(device)
                t = clip.text_model(input_ids=enc["input_ids"],
                                    attention_mask=enc["attention_mask"])
                feats.append(clip.text_projection(t.pooler_output).half().cpu())
        torch.save({"ids": uniq_hyps, "feat": torch.cat(feats)},
                   CACHE_DIR / "clip_txt.pt")
        del clip

    if "clip_txt_seq" in todo:
        # token-level text features for clip_camc (+ lengths for EOS pooling)
        tok = AutoTokenizer.from_pretrained(str(fsr.CLIP_DIR))
        texts = [hyp_text[h] for h in uniq_hyps]
        max_len = min(77, max(len(tok(t)["input_ids"]) for t in texts))
        print(f"  CLIP pad length: {max_len}")
        clip = CLIPModel.from_pretrained(str(fsr.CLIP_DIR)).to(device).eval()
        feats, lens = [], []
        with torch.no_grad():
            for i in range(0, len(texts), BATCH_TEXT):
                enc = tok(texts[i:i + BATCH_TEXT], truncation=True,
                          max_length=max_len, padding="max_length",
                          return_tensors="pt").to(device)
                t = clip.text_model(input_ids=enc["input_ids"],
                                    attention_mask=enc["attention_mask"])
                feats.append(clip.text_projection(
                    t.last_hidden_state).half().cpu())
                lens.append(enc["attention_mask"].sum(dim=-1).cpu())
        torch.save({"ids": uniq_hyps, "feat": torch.cat(feats),
                    "len": torch.cat(lens)}, CACHE_DIR / "clip_txt_seq.pt")
        del clip
    print("Test caches built")


class Cache:
    def __init__(self, name: str):
        blob = torch.load(CACHE_DIR / f"{name}.pt", map_location="cpu")
        self.row = {i: k for k, i in enumerate(blob["ids"])}
        self.feat = blob["feat"]
        self.len = blob.get("len")  # clip_txt_seq only: true lengths

    def take(self, ids: list) -> torch.Tensor:
        return self.feat[[self.row[i] for i in ids]].float()

    def take_len(self, ids: list) -> torch.Tensor:
        return self.len[[self.row[i] for i in ids]]


def predict_all(exp: str, arch: str, rows: list[dict], caches: dict,
                device: torch.device) -> list[int]:
    prefixes = fsr.ARCH_PREFIXES.get(arch, ("classifier.",))
    sd = fsr.load_trainable(exp, prefixes)
    preds = []
    with torch.no_grad():
        for i in range(0, len(rows), BATCH_HEAD):
            part = rows[i:i + BATCH_HEAD]
            img_ids = [r["filename"] for r in part]
            hyp_ids = [norm(r["hypothesis"]) for r in part]
            batch = {}
            if arch == "image_only" or fsr.uses_bert_vit(arch):
                batch["vit"] = caches["vit_seq"].take(img_ids)
            if arch == "text_only" or fsr.uses_bert_vit(arch):
                batch["bert"] = caches["bert_seq"].take(hyp_ids)
            if arch == "clip_fusion":
                batch["clip_img"] = caches["clip_img"].take(img_ids)
                batch["clip_txt"] = caches["clip_txt"].take(hyp_ids)
            if arch == "clip_camc":
                batch["clip_img_seq"] = caches["clip_img_seq"].take(img_ids)
                batch["clip_txt_seq"] = caches["clip_txt_seq"].take(hyp_ids)
                batch["clip_txt_len"] = caches["clip_txt_seq"].take_len(hyp_ids)
            logits = fsr.head_logits(arch, sd, batch, device)
            preds.extend(int(p) for p in logits.argmax(dim=1).cpu())
    return preds


def scores(y_true: list[int], y_pred: list[int]) -> dict:
    return {
        "n": len(y_true),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true),
        "contra_recall": recall_score(
            y_true, y_pred, labels=[2], average=None, zero_division=0)[0],
    }


def main() -> None:
    device = fsr.pick_device()
    print("Device:", device)
    rows = load_rows()
    n_corr = attach_corrected(rows)
    print(f"test rows: {len(rows)} | with corrected label: {n_corr}")

    build_caches(rows, device)
    caches = {n: Cache(n) for n in ("vit_seq", "bert_seq",
                                    "clip_img", "clip_txt",
                                    "clip_img_seq", "clip_txt_seq")}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for exp, model in fsr.EXPERIMENTS:
        arch = fsr.ARCH_OF.get(model, model)
        preds = predict_all(exp, arch, rows, caches, device)

        orig = scores([r["label"] for r in rows], preds)
        sub = [(r["corrected_label"], p) for r, p in zip(rows, preds)
               if r["corrected_label"] is not None]
        corr = scores([a for a, _ in sub], [b for _, b in sub])
        rec = {"experiment": exp, "model": model,
               "corrected_macro_f1": corr["macro_f1"],
               "corrected_accuracy": corr["accuracy"],
               "corrected_contra_recall": corr["contra_recall"],
               "original_macro_f1": orig["macro_f1"],
               "original_accuracy": orig["accuracy"],
               "n_corrected": corr["n"], "n_original": orig["n"]}
        summary.append(rec)
        print(f"{exp:<22} corrected F1 {corr['macro_f1']:.4f} "
              f"(acc {corr['accuracy']:.4f}, C-rec {corr['contra_recall']:.4f})"
              f" | original F1 {orig['macro_f1']:.4f}")

        with (OUT_DIR / f"{exp}_predictions.csv").open(
                "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "hypothesis", "original_label",
                        "corrected_label", "pred"])
            for r, p in zip(rows, preds):
                w.writerow([r["filename"], r["hypothesis"], r["label"],
                            "" if r["corrected_label"] is None
                            else r["corrected_label"], p])

    cols = list(summary[0].keys())
    with (OUT_DIR / "test_summary.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary)

    import statistics as st
    print("\n=== corrected test macro-F1, mean±std over seeds ===")
    for model in ("camc_reg", "text_only", "late_fusion",
                  "clip_fusion", "image_only", "clip_camc"):
        v = [r["corrected_macro_f1"] for r in summary if r["model"] == model]
        o = [r["original_macro_f1"] for r in summary if r["model"] == model]
        print(f"{model:<13} corrected {st.mean(v):.4f}±{st.stdev(v):.4f}"
              f" | original {st.mean(o):.4f}±{st.stdev(o):.4f}")
    print(f"\nWrote {OUT_DIR}/test_summary.csv and per-checkpoint predictions")


if __name__ == "__main__":
    main()
