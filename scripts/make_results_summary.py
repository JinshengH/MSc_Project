"""Unified results table + pivot + summary figures (final.md §6.3 item 7).

Single source of truth for every number quoted in the dissertation.  Reads
only machine-generated outputs (never hand-copied values):

  results/<exp>/metrics.csv                    best val macro-F1 / epoch
  results/snli_ve_test_eval/test_summary.csv   Exp 1 corrected/original test
  results/fsr_full_test/fsr_summary.csv        Exp 2 full-test probe
  data/newsclippings/*/zero_shot/*.csv         500-sample predictions
  data/newsclippings/*/pilot_annotations.csv   finalized two-layer labels

and derives the annotated-subset detection numbers, including the
guide-definition FSR (final.md §6.1: P(pred E | falsified with relation
gold in {N, C}) — computable only where relation gold exists).

Outputs:
  analysis/summary.csv          one row per checkpoint (16)
  analysis/summary_pivot.csv    mean±std per model over seeds 42/43/44
  analysis/summary_plots/summary.svg / .png   4-panel overview figure

Run from the project root:
    conda run -n hf_latest python scripts/make_results_summary.py
"""

from __future__ import annotations

import csv
import statistics as st
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
ANALYSIS = PROJECT_ROOT / "analysis"
PACKETS = ["pilot_seed42", "formal_seed43"]

MODELS = ["camc_reg", "text_only", "late_fusion", "clip_fusion", "image_only",
          "clip_camc"]  # clip_camc = camc-on-CLIP (2x2 missing cell)
SEEDS = [42, 43, 44]
EXPERIMENTS = (["camc_seed42"]
               + [f"{m}_seed{s}" for m in MODELS for s in SEEDS]
               + [f"ablation_{v}_seed42" for v in ("a1", "a3", "a4")])
# Trainable parameter counts (M), from training logs (train.py prints them)
PARAMS_M = {"camc": 45.5, "camc_reg": 45.5, "text_only": 0.6,
            "late_fusion": 3.0, "clip_fusion": 1.6, "image_only": 0.6,
            "ablation_a1": 1.8, "ablation_a3": 44.3, "ablation_a4": 44.9,
            "clip_camc": 20.8}

# Exp 4 chain (guide §5.2): variant -> (experiment row, mechanism label).
# A2/A6 reuse the already-trained baselines; note in writing that A2 was
# trained under the unified baseline recipe while A1/A3/A4/A6 share the
# camc_reg recipe (byte-identical configs, asserted at generation time).
EXP4_CHAIN = [
    ("A1", "ablation_a1_seed42", "concat only"),
    ("A2", "late_fusion_seed42", "concat + diff + product (no cross-attn)"),
    ("A3", "ablation_a3_seed42", "cross-attn only"),
    ("A4", "ablation_a4_seed42", "cross-attn + diff"),
    ("A6", "camc_reg_seed42", "full (cross-attn + diff + product)"),
]


def split_name(exp: str) -> tuple[str, int]:
    model, seed = exp.rsplit("_seed", 1)
    return model, int(seed)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_val(exp: str) -> dict:
    rows = read_csv(RESULTS / exp / "metrics.csv")
    best = max(rows, key=lambda r: float(r["val_macro_f1"]))
    return {"val_macro_f1": float(best["val_macro_f1"]),
            "best_epoch": int(best["epoch"]),
            "val_contra_recall": float(best["val_contradiction_recall"])}


def load_annotations() -> dict:
    """(packet, sample_id) -> {relation, channel, veracity}"""
    ann = {}
    for p in PACKETS:
        for r in read_csv(PROJECT_ROOT / "data" / "newsclippings" / p /
                          "pilot_annotations.csv"):
            ann[(p, r["sample_id"])] = r
    return ann


def subset_metrics(exp: str, ann: dict) -> dict:
    """Detection numbers on the 500 annotated samples, incl. guide-FSR."""
    rows = []
    for p in PACKETS:
        f = (PROJECT_ROOT / "data" / "newsclippings" / p / "zero_shot" /
             f"{exp}_predictions.csv")
        for r in read_csv(f):
            a = ann[(p, r["sample_id"])]
            rows.append((r["original_label"], r["pred_label"],
                         a["relation_label"], a["evidence_channel"]))
    from sklearn.metrics import f1_score
    lab = {"entailment": 0, "neutral": 1, "contradiction": 2}
    c = Counter()
    for veracity, pred, rel, chan in rows:
        falsified = veracity == "falsified"
        if falsified and rel in ("neutral", "contradiction"):
            c["fsr_denom"] += 1
            c["fsr_num"] += pred == "entailment"
        if pred == "contradiction":
            c["c_total"] += 1
            c["c_falsified"] += falsified
            c["c_visible"] += chan == "visible_conflict"
        c["visible_n"] += chan == "visible_conflict"
    return {
        "zs500_relation_f1": f1_score(
            [lab[r] for _, _, r, _ in rows],
            [lab[p] for _, p, _, _ in rows], average="macro"),
        "fsr500": c["fsr_num"] / c["fsr_denom"],
        "c500_falsified": c["c_falsified"],
        "c500_pristine": c["c_total"] - c["c_falsified"],
        "c500_precision": (c["c_falsified"] / c["c_total"]
                           if c["c_total"] else float("nan")),
        "visible_catch": c["c_visible"] / c["visible_n"],
    }


def collect() -> list[dict]:
    test = {r["experiment"]: r
            for r in read_csv(ANALYSIS / "snli_ve_test_eval" / "test_summary.csv")}
    probe = {r["experiment"]: r
             for r in read_csv(ANALYSIS / "fsr_full_test" / "fsr_summary.csv")}
    ann = load_annotations()

    out = []
    for exp in EXPERIMENTS:
        model, seed = split_name(exp)
        row = {"experiment": exp, "model": model, "seed": seed,
               "params_M": PARAMS_M[model]}
        row.update(best_val(exp))
        t = test[exp]
        row.update(corrected_test_f1=float(t["corrected_macro_f1"]),
                   corrected_test_contra_recall=float(
                       t["corrected_contra_recall"]),
                   original_test_f1=float(t["original_macro_f1"]))
        p = probe[exp]
        row.update(probe_c_rate_falsified=float(p["C_rate_falsified"]),
                   probe_c_rate_pristine=float(p["C_rate_pristine"]),
                   probe_e_rate_falsified=float(p["FSR"]),
                   probe_e_rate_pristine=float(p["E_recall_pristine"]))
        row.update(subset_metrics(exp, ann))
        out.append(row)
    return out


NUMERIC = ["val_macro_f1", "corrected_test_f1", "original_test_f1",
           "corrected_test_contra_recall", "zs500_relation_f1",
           "probe_c_rate_falsified", "probe_c_rate_pristine",
           "probe_e_rate_falsified", "probe_e_rate_pristine",
           "fsr500", "c500_precision", "visible_catch"]


def pivot(rows: list[dict]) -> list[dict]:
    out = []
    for model in MODELS:
        sub = [r for r in rows if r["model"] == model]
        rec = {"model": model, "n_seeds": len(sub),
               "params_M": PARAMS_M[model]}
        for key in NUMERIC:
            vals = [r[key] for r in sub]
            rec[f"{key}_mean"] = st.mean(vals)
            rec[f"{key}_std"] = st.stdev(vals)
        out.append(rec)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


ORDER = ["text_only", "image_only", "late_fusion", "clip_fusion", "camc_reg",
         "clip_camc"]
LABEL = {"text_only": "Text-only", "image_only": "Image-only",
         "late_fusion": "Diff Fusion\n(A2)", "clip_fusion": "CLIP Fusion",
         "camc_reg": "Proposed\n(CAMC)", "clip_camc": "CAMC-on-\nCLIP"}


def draw_indomain(ax, pv, np):
    x = np.arange(len(ORDER))
    for off, key, name in ((-0.18, "val_macro_f1", "SNLI-VE val"),
                           (0.18, "corrected_test_f1", "Corrected test")):
        ax.bar(x + off, [pv[m][f"{key}_mean"] for m in ORDER], 0.36,
               yerr=[pv[m][f"{key}_std"] for m in ORDER],
               capsize=3, label=name)
    ax.set_xticks(x, [LABEL[m] for m in ORDER])
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.25, 0.78)
    ax.legend()


def draw_direction(ax, pv, np):
    x = np.arange(len(ORDER))
    for off, key, name in (
            (-0.18, "probe_c_rate_falsified", "C-rate on falsified"),
            (0.18, "probe_c_rate_pristine", "C-rate on pristine")):
        ax.bar(x + off, [pv[m][f"{key}_mean"] for m in ORDER], 0.36,
               yerr=[pv[m][f"{key}_std"] for m in ORDER],
               capsize=3, label=name)
    ax.set_xticks(x, [LABEL[m] for m in ORDER])
    ax.set_ylabel("P(pred C)")
    ax.legend()


def draw_decoupling(ax, rows, np):
    for m in ORDER:
        xs = [r["corrected_test_f1"] for r in rows if r["model"] == m]
        ys = [r["c500_precision"] for r in rows if r["model"] == m]
        ax.scatter(xs, ys, s=60, label=LABEL[m].replace("\n", " "))
    ax.axhline(0.5, ls="--", lw=1, c="gray")
    ax.text(0.26, 0.51, "precision 0.5 (coin flip)", fontsize=9, c="gray")
    ax.set_xlabel("Corrected test Macro-F1 (in-domain)")
    ax.set_ylabel("C-prediction precision on 500 subset")
    ax.legend(fontsize=9)


def draw_fsr(ax, pv, np):
    x = np.arange(len(ORDER))
    for off, key, name in (
            (-0.18, "fsr500", "FSR (500, gold$\\in${N,C})"),
            (0.18, "probe_e_rate_pristine", "E-rate on pristine (full test)")):
        ax.bar(x + off, [pv[m][f"{key}_mean"] for m in ORDER], 0.36,
               yerr=[pv[m][f"{key}_std"] for m in ORDER],
               capsize=3, label=name)
    ax.set_xticks(x, [LABEL[m] for m in ORDER])
    ax.set_ylabel("Rate")
    ax.legend()


def make_figure(rows: list[dict], piv: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pv = {r["model"]: r for r in piv}
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4-panel overview (quick viewing; panel titles included)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    draw_indomain(axes[0, 0], pv, np)
    axes[0, 0].set_title("(a) In-domain performance (3 seeds)")
    draw_direction(axes[0, 1], pv, np)
    axes[0, 1].set_title("(b) Zero-shot direction, full test 7,264 (3 seeds)")
    draw_decoupling(axes[1, 0], rows, np)
    axes[1, 0].set_title("(c) In-domain F1 vs transfer precision (per seed)")
    draw_fsr(axes[1, 1], pv, np)
    axes[1, 1].set_title("(d) False Support Rate with its paired guard")
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(out_dir / f"summary.{ext}", dpi=150)
    plt.close(fig)

    # Publication figures. Only RQ-primary material gets a paper SVG; the
    # full-test probe stays dashboard-only (guide §7.1: behavioural probe,
    # supplementary).
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12,
                         "legend.fontsize": 11})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    draw_indomain(ax, pv, np)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_indomain_f1.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Figures -> {out_dir}/summary.svg|png + fig_indomain_f1.svg")


def exp4_table(rows: list[dict]) -> list[dict]:
    by_exp = {r["experiment"]: r for r in rows}
    out = []
    for variant, exp, mech in EXP4_CHAIN:
        r = by_exp[exp]
        out.append({"variant": variant, "mechanism": mech,
                    "experiment": exp, "params_M": r["params_M"],
                    "val_macro_f1": r["val_macro_f1"],
                    "corrected_test_f1": r["corrected_test_f1"],
                    "zs500_relation_f1": r["zs500_relation_f1"],
                    "drop": r["corrected_test_f1"] - r["zs500_relation_f1"],
                    "fsr500": r["fsr500"],
                    "c500_precision": r["c500_precision"]})
    return out


def main() -> None:
    rows = collect()
    piv = pivot(rows)
    write_csv(ANALYSIS / "summary.csv", rows)
    write_csv(ANALYSIS / "summary_pivot.csv", piv)
    exp4 = exp4_table(rows)
    write_csv(ANALYSIS / "exp4_ablation.csv", exp4)
    print(f"Wrote analysis/summary.csv ({len(rows)} checkpoints), "
          f"analysis/summary_pivot.csv ({len(piv)} models) "
          f"and analysis/exp4_ablation.csv (seed 42 chain)")
    print("\n=== Exp 4 ablation chain (seed 42) ===")
    for r in exp4:
        print(f"  {r['variant']}  {r['mechanism']:<42} "
              f"params {r['params_M']:>5}M  val {r['val_macro_f1']:.4f}  "
              f"corrected {r['corrected_test_f1']:.4f}")

    print(f"\n{'model':<13}{'val F1':<17}{'corr test F1':<17}"
          f"{'FSR(500)':<17}{'C500 precision'}")
    for r in piv:
        print(f"{r['model']:<13}"
              f"{r['val_macro_f1_mean']:.4f}±{r['val_macro_f1_std']:.4f}   "
              f"{r['corrected_test_f1_mean']:.4f}±{r['corrected_test_f1_std']:.4f}   "
              f"{r['fsr500_mean']:.4f}±{r['fsr500_std']:.4f}   "
              f"{r['c500_precision_mean']:.3f}±{r['c500_precision_std']:.3f}")

    make_figure(rows, piv, ANALYSIS / "summary_plots")


if __name__ == "__main__":
    main()
