"""Full-test detection metrics (FSR line) on NewsCLIPpings merged_balanced.

Zero-shot inference of every trained checkpoint over all 7,264 test pairs,
read against the official veracity labels (no human annotation involved):

    FSR                 = P(pred E | falsified)   "false support rate"
    E_recall_pristine   = P(pred E | pristine)    guards the trivial-zero read
    C_rate_falsified/pristine                     direction check at scale

Two-phase design. The frozen towers (ViT / BERT / CLIP) are identical across
all seeds and all experiments, so phase 1 encodes the test set ONCE per tower
and caches features to disk; phase 2 rebuilds only each checkpoint's
trainable modules and runs them over the cache (seconds per checkpoint).

Sanity check: the 500 pilot/formal samples are a subset of the test split,
so per checkpoint we compare cache-path predictions against the existing
zero_shot_newsclippings.py predictions on the overlap and report the match
rate (expect ~100%; tiny fp16-cache rounding flips are tolerated).

Run from the project root:
    conda run -n hf_latest python scripts/fsr_full_test.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer, CLIPModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.CAMC import (  # noqa: E402
    ContradictionAwareEncoder, ContradictionAwareLayer)
from src.models.encoders import ImageEncoder, TextEncoder  # noqa: E402
from src.utils import paths  # noqa: E402

TEST_JSON = PROJECT_ROOT / "data" / "newsclippings" / "merged_balanced" / "test.json"
VISUALNEWS_ORIGIN = PROJECT_ROOT / "data" / "visual_news" / "origin"
RESULTS_DIR = PROJECT_ROOT / "results"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
CACHE_DIR = ANALYSIS_DIR / "fsr_cache"
OUT_DIR = ANALYSIS_DIR / "fsr_full_test"
CLIP_DIR = paths.scratch_project_path(
    "models", "clip-vit-base-patch16", local="models/clip-vit-base-patch16")

BATCH_IMAGES = 32
BATCH_TEXT = 256
BATCH_HEAD = 256
MAX_LEN_BERT = 128
LABELS = ["entailment", "neutral", "contradiction"]

# (exp_name, architecture) — architecture selects the head replication path
EXPERIMENTS = [("camc_seed42", "camc")] + [
    (f"{model}_seed{seed}", model)
    for model in ("camc_reg", "text_only", "late_fusion",
                  "clip_fusion", "image_only")
    for seed in (42, 43, 44)
] + [(f"ablation_{v}_seed42", f"ablation_{v}") for v in ("a1", "a3", "a4")] \
  + [(f"clip_camc_seed{s}", "clip_camc") for s in (42, 43, 44)]  # camc-on-CLIP
ARCH_OF = {"camc_reg": "camc"}  # camc_reg shares the camc architecture

# Exp 4 ablation variants (guide §5.2): (use_cross_attn, fusion parts) —
# must mirror src/configs/ablation_*_seed42.py exactly
ABLATION_SPEC = {"ablation_a1": (False, "it"),
                 "ablation_a3": (True, "it"),
                 "ablation_a4": (True, "itd")}
ARCH_PREFIXES = {
    "camc": ("encoder.", "classifier."),
    "late_fusion": ("img_proj.", "classifier."),
    "ablation_a1": ("img_proj.", "classifier."),
    "ablation_a3": ("img_proj.", "layers.", "classifier."),
    "ablation_a4": ("img_proj.", "layers.", "classifier."),
    "clip_camc": ("img_proj.", "layers.", "classifier."),
}


def uses_bert_vit(arch: str) -> bool:
    return arch in ("camc", "late_fusion") or arch in ABLATION_SPEC

IMAGENET_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
CLIP_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                         [0.26862954, 0.26130258, 0.27577711]),
])


def pick_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu")


def load_test_set() -> list[dict]:
    payload = json.loads(TEST_JSON.read_text())
    split_names = payload["source_datasets"]
    anns = payload["annotations"]

    needed = {a["id"] for a in anns} | {a["image_id"] for a in anns}
    data_json = VISUALNEWS_ORIGIN / "data.json"
    vn = {e["id"]: e for e in json.loads(data_json.read_text())
          if e["id"] in needed}
    missing = needed - vn.keys()
    if missing:
        raise RuntimeError(f"{len(missing)} ids missing from VisualNews")

    samples = []
    for a in anns:
        img = VISUALNEWS_ORIGIN / vn[a["image_id"]]["image_path"].lstrip("./")
        samples.append({
            "caption_id": a["id"],
            "image_id": a["image_id"],
            "caption": vn[a["id"]]["caption"],
            "image_path": img,
            "falsified": a["falsified"],
            "official_split": split_names[str(a["source_dataset"])],
        })
    absent = [s for s in samples if not s["image_path"].exists()]
    if absent:
        raise RuntimeError(f"{len(absent)} image files missing on disk")
    return samples


# ---------------------------------------------------------------- phase 1

def build_caches(samples: list[dict], device: torch.device) -> None:
    """Encode unique images/captions once per frozen tower; cache fp16."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    todo = [n for n in ("vit_seq", "bert_seq", "clip_img", "clip_txt",
                        "clip_img_seq", "clip_txt_seq")
            if not (CACHE_DIR / f"{n}.pt").exists()]
    if not todo:
        print("Feature caches already present")
        return
    print(f"Building caches: {todo}")

    uniq_imgs = sorted({s["image_id"] for s in samples})
    img_path = {s["image_id"]: s["image_path"] for s in samples}
    uniq_caps = sorted({s["caption_id"] for s in samples})
    cap_text = {s["caption_id"]: s["caption"] for s in samples}
    print(f"unique images: {len(uniq_imgs)} | unique captions: {len(uniq_caps)}")

    if "vit_seq" in todo or "clip_img" in todo or "clip_img_seq" in todo:
        vit = ImageEncoder().to(device).eval() if "vit_seq" in todo else None
        clip = (CLIPModel.from_pretrained(str(CLIP_DIR)).to(device).eval()
                if "clip_img" in todo or "clip_img_seq" in todo else None)
        vit_out, clip_out, clip_seq_out = [], [], []
        with torch.no_grad():
            for i in range(0, len(uniq_imgs), BATCH_IMAGES):
                ids = uniq_imgs[i:i + BATCH_IMAGES]
                pil = [Image.open(img_path[j]).convert("RGB") for j in ids]
                if vit is not None:
                    px = torch.stack([IMAGENET_TF(p) for p in pil]).to(device)
                    vit_out.append(vit(px).half().cpu())
                if clip is not None:
                    px = torch.stack([CLIP_TF(p) for p in pil]).to(device)
                    v = clip.vision_model(pixel_values=px)
                    if "clip_img" in todo:
                        clip_out.append(clip.visual_projection(
                            v.pooler_output).half().cpu())
                    if "clip_img_seq" in todo:
                        # token-wise pooled path (clip_camc): post_layernorm
                        # then projection over all 197 tokens
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
        bert = TextEncoder().to(device).eval()
        feats = []
        with torch.no_grad():
            for i in range(0, len(uniq_caps), BATCH_TEXT):
                ids = uniq_caps[i:i + BATCH_TEXT]
                enc = tok([cap_text[j] for j in ids], truncation=True,
                          max_length=MAX_LEN_BERT, padding="max_length",
                          return_tensors="pt").to(device)
                feats.append(bert(enc["input_ids"],
                                  enc["attention_mask"]).half().cpu())
        torch.save({"ids": uniq_caps, "feat": torch.cat(feats)},
                   CACHE_DIR / "bert_seq.pt")
        del bert

    if "clip_txt" in todo:
        tok = AutoTokenizer.from_pretrained(str(CLIP_DIR))
        clip = CLIPModel.from_pretrained(str(CLIP_DIR)).to(device).eval()
        feats = []
        with torch.no_grad():
            for i in range(0, len(uniq_caps), BATCH_TEXT):
                ids = uniq_caps[i:i + BATCH_TEXT]
                enc = tok([cap_text[j] for j in ids], truncation=True,
                          max_length=77, padding="max_length",
                          return_tensors="pt").to(device)
                t = clip.text_model(input_ids=enc["input_ids"],
                                    attention_mask=enc["attention_mask"])
                feats.append(clip.text_projection(t.pooler_output).half().cpu())
        torch.save({"ids": uniq_caps, "feat": torch.cat(feats)},
                   CACHE_DIR / "clip_txt.pt")
        del clip

    if "clip_txt_seq" in todo:
        # token-level text features for clip_camc: projection over the full
        # (final-layernormed) sequence + true lengths for EOS pooling
        tok = AutoTokenizer.from_pretrained(str(CLIP_DIR))
        clip = CLIPModel.from_pretrained(str(CLIP_DIR)).to(device).eval()
        feats, lens = [], []
        with torch.no_grad():
            for i in range(0, len(uniq_caps), BATCH_TEXT):
                ids = uniq_caps[i:i + BATCH_TEXT]
                enc = tok([cap_text[j] for j in ids], truncation=True,
                          max_length=77, padding="max_length",
                          return_tensors="pt").to(device)
                t = clip.text_model(input_ids=enc["input_ids"],
                                    attention_mask=enc["attention_mask"])
                feats.append(clip.text_projection(
                    t.last_hidden_state).half().cpu())
                lens.append(enc["attention_mask"].sum(dim=-1).cpu())
        torch.save({"ids": uniq_caps, "feat": torch.cat(feats),
                    "len": torch.cat(lens)}, CACHE_DIR / "clip_txt_seq.pt")
        del clip
    print("Caches built")


class Cache:
    def __init__(self, name: str):
        blob = torch.load(CACHE_DIR / f"{name}.pt", map_location="cpu")
        self.row = {i: k for k, i in enumerate(blob["ids"])}
        self.feat = blob["feat"]
        self.len = blob.get("len")  # clip_txt_seq only: true lengths

    def take(self, ids: list[int]) -> torch.Tensor:
        return self.feat[[self.row[i] for i in ids]].float()

    def take_len(self, ids: list[int]) -> torch.Tensor:
        return self.len[[self.row[i] for i in ids]]


# ---------------------------------------------------------------- phase 2

def load_trainable(exp: str, prefixes: tuple[str, ...]) -> dict:
    sd = torch.load(RESULTS_DIR / exp / "best.pt", map_location="cpu")
    picked = {k: v for k, v in sd.items() if k.startswith(prefixes)}
    expected = {k for k in sd if not k.startswith(("ie.", "te.", "clip."))}
    if set(picked) != expected:
        raise RuntimeError(f"{exp}: unexpected trainable keys "
                           f"{sorted(expected - set(picked))[:5]}")
    return picked


def strip(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items()
            if k.startswith(prefix)}


def mlp_head(in_dim: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, 768), torch.nn.GELU(),
        torch.nn.Dropout(0.0), torch.nn.Linear(768, 3))


def head_logits(arch: str, sd: dict, batch: dict,
                device: torch.device) -> torch.Tensor:
    if arch == "camc":
        enc = ContradictionAwareEncoder(768, 8, 6).to(device).eval()
        enc.load_state_dict(strip(sd, "encoder."))
        cls = mlp_head(768 * 4).to(device).eval()
        cls.load_state_dict(strip(sd, "classifier."))
        fusion, _ = enc(batch["vit"].to(device), batch["bert"].to(device))
        return cls(fusion)
    if arch == "text_only":
        cls = mlp_head(768).to(device).eval()
        cls.load_state_dict(strip(sd, "classifier."))
        return cls(batch["bert"][:, 0].to(device))
    if arch == "image_only":
        cls = mlp_head(768).to(device).eval()
        cls.load_state_dict(strip(sd, "classifier."))
        return cls(batch["vit"][:, 0].to(device))
    if arch == "late_fusion":
        proj = torch.nn.Linear(768, 768).to(device).eval()
        proj.load_state_dict(strip(sd, "img_proj."))
        cls = mlp_head(768 * 4).to(device).eval()
        cls.load_state_dict(strip(sd, "classifier."))
        img = proj(batch["vit"][:, 0].to(device))
        txt = batch["bert"][:, 0].to(device)
        return cls(torch.cat(
            [img, txt, (txt - img).abs(), txt * img], dim=-1))
    if arch == "clip_fusion":
        cls = mlp_head(512 * 4).to(device).eval()
        cls.load_state_dict(strip(sd, "classifier."))
        img = F.normalize(batch["clip_img"].to(device), dim=-1)
        txt = F.normalize(batch["clip_txt"].to(device), dim=-1)
        return cls(torch.cat(
            [img, txt, (txt - img).abs(), txt * img], dim=-1))
    if arch == "clip_camc":
        proj = torch.nn.Linear(512, 512).to(device).eval()
        proj.load_state_dict(strip(sd, "img_proj."))
        layers = torch.nn.ModuleList([
            ContradictionAwareLayer(512, 8) for _ in range(6)
        ]).to(device).eval()
        layers.load_state_dict(strip(sd, "layers."))
        img_seq = proj(batch["clip_img_seq"].to(device))
        txt_seq = batch["clip_txt_seq"].to(device)
        for layer in layers:
            txt_seq, _ = layer(img_seq, txt_seq)
        eos = (batch["clip_txt_len"].to(device) - 1)
        txt = txt_seq[torch.arange(txt_seq.size(0), device=device), eos]
        img = img_seq[:, 0]
        cls = mlp_head(512 * 4).to(device).eval()
        cls.load_state_dict(strip(sd, "classifier."))
        return cls(torch.cat(
            [img, txt, (txt - img).abs(), txt * img], dim=-1))
    if arch in ABLATION_SPEC:
        use_attn, fusion = ABLATION_SPEC[arch]
        proj = torch.nn.Linear(768, 768).to(device).eval()
        proj.load_state_dict(strip(sd, "img_proj."))
        img_seq = proj(batch["vit"].to(device))
        txt_seq = batch["bert"].to(device)
        if use_attn:
            layers = torch.nn.ModuleList([
                ContradictionAwareLayer(768, 8) for _ in range(6)
            ]).to(device).eval()
            layers.load_state_dict(strip(sd, "layers."))
            for layer in layers:
                txt_seq, _ = layer(img_seq, txt_seq)
        img, txt = img_seq[:, 0], txt_seq[:, 0]
        parts = {"i": img, "t": txt,
                 "d": (txt - img).abs(), "p": txt * img}
        cls = mlp_head(768 * len(fusion)).to(device).eval()
        cls.load_state_dict(strip(sd, "classifier."))
        return cls(torch.cat([parts[c] for c in fusion], dim=-1))
    raise ValueError(arch)


def predict_all(exp: str, arch: str, samples: list[dict], caches: dict,
                device: torch.device) -> list[str]:
    prefixes = ARCH_PREFIXES.get(arch, ("classifier.",))
    sd = load_trainable(exp, prefixes)
    preds = []
    with torch.no_grad():
        for i in range(0, len(samples), BATCH_HEAD):
            part = samples[i:i + BATCH_HEAD]
            img_ids = [s["image_id"] for s in part]
            cap_ids = [s["caption_id"] for s in part]
            batch = {}
            if arch == "image_only" or uses_bert_vit(arch):
                batch["vit"] = caches["vit_seq"].take(img_ids)
            if arch == "text_only" or uses_bert_vit(arch):
                batch["bert"] = caches["bert_seq"].take(cap_ids)
            if arch == "clip_fusion":
                batch["clip_img"] = caches["clip_img"].take(img_ids)
                batch["clip_txt"] = caches["clip_txt"].take(cap_ids)
            if arch == "clip_camc":
                batch["clip_img_seq"] = caches["clip_img_seq"].take(img_ids)
                batch["clip_txt_seq"] = caches["clip_txt_seq"].take(cap_ids)
                batch["clip_txt_len"] = caches["clip_txt_seq"].take_len(cap_ids)
            logits = head_logits(arch, sd, batch, device)
            preds.extend(LABELS[int(p)] for p in logits.argmax(dim=1).cpu())
    return preds


def sanity_check(exp: str, samples: list[dict], preds: list[str]) -> str:
    """Compare against zero_shot_newsclippings predictions on the overlap."""
    by_pair = {(s["caption_id"], s["image_id"]): p
               for s, p in zip(samples, preds)}
    hit = tot = 0
    for packet in ("pilot_seed42", "formal_seed43"):
        f = (PROJECT_ROOT / "data" / "newsclippings" / packet /
             "zero_shot" / f"{exp}_predictions.csv")
        if not f.exists():
            continue
        for r in csv.DictReader(f.open()):
            key = (int(r["caption_id"]), int(r["image_id"]))
            if key in by_pair:
                tot += 1
                hit += by_pair[key] == r["pred_label"]
    return f"{hit}/{tot}" if tot else "n/a"


def metrics(samples: list[dict], preds: list[str]) -> dict:
    n_f = sum(s["falsified"] for s in samples)
    n_p = len(samples) - n_f
    c = Counter((s["falsified"], p) for s, p in zip(samples, preds))
    return {
        "FSR": c[(True, "entailment")] / n_f,
        "E_recall_pristine": c[(False, "entailment")] / n_p,
        "C_rate_falsified": c[(True, "contradiction")] / n_f,
        "C_rate_pristine": c[(False, "contradiction")] / n_p,
        "pred_E": c[(True, "entailment")] + c[(False, "entailment")],
        "pred_N": c[(True, "neutral")] + c[(False, "neutral")],
        "pred_C": c[(True, "contradiction")] + c[(False, "contradiction")],
    }


def main() -> None:
    device = pick_device()
    print("Device:", device)
    samples = load_test_set()
    print(f"Test pairs: {len(samples)} "
          f"(falsified {sum(s['falsified'] for s in samples)})")

    build_caches(samples, device)
    caches = {n: Cache(n) for n in ("vit_seq", "bert_seq",
                                    "clip_img", "clip_txt",
                                    "clip_img_seq", "clip_txt_seq")}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for exp, model in EXPERIMENTS:
        arch = ARCH_OF.get(model, model)
        preds = predict_all(exp, arch, samples, caches, device)
        match = sanity_check(exp, samples, preds)
        m = metrics(samples, preds)
        m.update(experiment=exp, model=model, overlap_match=match)
        summary.append(m)
        print(f"{exp:<22} FSR {m['FSR']:.4f} | E_rec_pri "
              f"{m['E_recall_pristine']:.4f} | C f/p "
              f"{m['C_rate_falsified']:.4f}/{m['C_rate_pristine']:.4f} "
              f"| E/N/C {m['pred_E']}/{m['pred_N']}/{m['pred_C']} "
              f"| overlap {match}")

        with (OUT_DIR / f"{exp}_predictions.csv").open(
                "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["caption_id", "image_id", "falsified",
                        "official_split", "pred_label"])
            for s, p in zip(samples, preds):
                w.writerow([s["caption_id"], s["image_id"],
                            int(s["falsified"]), s["official_split"], p])

    cols = ["experiment", "model", "FSR", "E_recall_pristine",
            "C_rate_falsified", "C_rate_pristine",
            "pred_E", "pred_N", "pred_C", "overlap_match"]
    with (OUT_DIR / "fsr_summary.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({k: r[k] for k in cols} for r in summary)

    import statistics as st
    print("\n=== mean±std over seeds (per model) ===")
    for model in ("camc_reg", "text_only", "late_fusion",
                  "clip_fusion", "image_only", "clip_camc"):
        rows = [r for r in summary if r["model"] == model]
        line = f"{model:<13}"
        for key in ("FSR", "E_recall_pristine",
                    "C_rate_falsified", "C_rate_pristine"):
            v = [r[key] for r in rows]
            line += f" {key} {st.mean(v):.4f}±{st.stdev(v):.4f}"
        print(line)
    print(f"\nWrote {OUT_DIR}/fsr_summary.csv and per-checkpoint predictions")


if __name__ == "__main__":
    main()
