"""RQ-primary tables and figures from the 500-sample annotated subset.

Everything here is the direct evidence for the research questions
(final.md §2), computed from per-sample zero-shot predictions
(data/newsclippings/*/zero_shot/) against the finalized two-layer
annotation (relation gold + evidence channel), 3 seeds per model:

  RQ1  analysis/rq/rq1_drop.csv       in-domain corrected-test macro-F1 vs
                                     zero-shot macro-F1 on relation gold,
                                     per model; text-only row = anchor
  RQ2  analysis/rq/rq2_h2.csv         H2 metric set per model: FSR
                                     (gold in {N,C}), contradiction recall,
                                     N<->C confusion (both directions),
                                     per-class zero-shot F1, drop
       analysis/rq/rq2_pred_dist.csv  prediction distribution P(E/N/C)
                                     pristine vs falsified, + human gold row
                                     (collapse diagnostic: E is dead
                                     everywhere, so FSR≈0 is vacuous)
  RQ3  analysis/rq/rq3_channel.csv    channel x model catch table with
                                     bootstrap 95% CI (guide §6.2 format)

Figures: fig_rq1_drop.svg (horizontal dumbbell, in-domain vs zero-shot),
fig_rq2_dist.svg (stacked prediction distribution vs human gold),
fig_rq3_channel.svg (grouped bars with CI).  Run from the project root:
    conda run -n hf_latest python scripts/make_rq_tables.py
"""

from __future__ import annotations

import csv
import random
import statistics as st
from pathlib import Path

from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
ANALYSIS = PROJECT_ROOT / "analysis"
OUT = ANALYSIS / "rq"
PACKETS = ["pilot_seed42", "formal_seed43"]
MODELS = ["camc_reg", "text_only", "late_fusion", "clip_fusion", "image_only",
          "clip_camc"]  # clip_camc = camc-on-CLIP (2x2 missing cell)
SEEDS = [42, 43, 44]
LAB = {"entailment": 0, "neutral": 1, "contradiction": 2}
CHANNELS = ["visible_conflict", "event_provenance", "external_knowledge",
            "temporal_provenance", "other"]
BOOT = 2000


def read_csv(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_samples() -> list[dict]:
    """One record per annotated sample with relation gold and channel."""
    out = []
    for p in PACKETS:
        base = PROJECT_ROOT / "data" / "newsclippings" / p
        for r in read_csv(base / "pilot_annotations.csv"):
            out.append({"packet": p, "sample_id": r["sample_id"],
                        "relation": LAB[r["relation_label"]],
                        "channel": r["evidence_channel"],
                        "split": r["official_split"]})
    return out


def load_preds(samples: list[dict]) -> dict:
    """preds[(model, seed)] = list of predicted label ids, sample-aligned."""
    preds = {}
    for m in MODELS:
        for s in SEEDS:
            by_key = {}
            for p in PACKETS:
                f = (PROJECT_ROOT / "data" / "newsclippings" / p /
                     "zero_shot" / f"{m}_seed{s}_predictions.csv")
                for r in read_csv(f):
                    by_key[(p, r["sample_id"])] = LAB[r["pred_label"]]
            preds[(m, s)] = [by_key[(x["packet"], x["sample_id"])]
                             for x in samples]
    return preds


def corrected_test_f1() -> dict:
    return {r["experiment"]: float(r["corrected_macro_f1"])
            for r in read_csv(ANALYSIS / "snli_ve_test_eval" /
                              "test_summary.csv")}


def mean_std(vals: list[float]) -> str:
    return f"{st.mean(vals):.4f}±{st.stdev(vals):.4f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    samples = load_samples()
    gold = [x["relation"] for x in samples]
    falsified = [x["channel"] != "n/a" for x in samples]
    preds = load_preds(samples)
    indomain = corrected_test_f1()

    # ---------------- RQ1: cross-domain drop ----------------
    rq1 = []
    for m in MODELS:
        ind = [indomain[f"{m}_seed{s}"] for s in SEEDS]
        zs = [f1_score(gold, preds[(m, s)], average="macro") for s in SEEDS]
        drops = [a - b for a, b in zip(ind, zs)]
        rq1.append({"model": m,
                    "indomain_corrected_f1": mean_std(ind),
                    "zeroshot_relation_f1": mean_std(zs),
                    "drop": mean_std(drops)})
    with (OUT / "rq1_drop.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rq1[0].keys()))
        w.writeheader(); w.writerows(rq1)

    # ---------------- RQ2: H2 metric set ----------------
    rq2 = []
    for m in MODELS:
        fsr, crec, c2n, n2c, zs = [], [], [], [], []
        f1c = {0: [], 1: [], 2: []}
        for s in SEEDS:
            pr = preds[(m, s)]
            idx_fnc = [i for i in range(len(gold))
                       if falsified[i] and gold[i] in (1, 2)]
            fsr.append(sum(pr[i] == 0 for i in idx_fnc) / len(idx_fnc))
            idx_c = [i for i in range(len(gold)) if gold[i] == 2]
            crec.append(sum(pr[i] == 2 for i in idx_c) / len(idx_c))
            c2n.append(sum(pr[i] == 1 for i in idx_c) / len(idx_c))
            idx_n = [i for i in range(len(gold)) if gold[i] == 1]
            n2c.append(sum(pr[i] == 2 for i in idx_n) / len(idx_n))
            zs.append(f1_score(gold, pr, average="macro"))
            per_class = f1_score(gold, pr, average=None, labels=[0, 1, 2])
            for k in (0, 1, 2):
                f1c[k].append(per_class[k])
        ind = [indomain[f"{m}_seed{s}"] for s in SEEDS]
        rq2.append({"model": m,
                    "FSR_gold_NC": mean_std(fsr),
                    "contradiction_recall": mean_std(crec),
                    "confusion_C_pred_N": mean_std(c2n),
                    "confusion_N_pred_C": mean_std(n2c),
                    "f1_E": mean_std(f1c[0]),
                    "f1_N": mean_std(f1c[1]),
                    "f1_C": mean_std(f1c[2]),
                    "drop": mean_std([a - b for a, b in zip(ind, zs)])})
    with (OUT / "rq2_h2.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rq2[0].keys()))
        w.writeheader(); w.writerows(rq2)

    # ---------------- RQ3: channel x model with bootstrap CI ----------------
    rng = random.Random(42)
    rq3 = []
    for ch in CHANNELS:
        idx = [i for i in range(len(samples)) if samples[i]["channel"] == ch]
        row = {"channel": ch, "n": len(idx)}
        for m in MODELS:
            per_seed = [[preds[(m, s)][i] == 2 for i in idx] for s in SEEDS]
            point = st.mean(sum(v) / len(v) for v in per_seed)
            boots = []
            for _ in range(BOOT):
                pick = [rng.randrange(len(idx)) for _ in idx]
                boots.append(st.mean(
                    sum(v[j] for j in pick) / len(pick) for v in per_seed))
            boots.sort()
            lo, hi = boots[int(0.025 * BOOT)], boots[int(0.975 * BOOT) - 1]
            row[m] = f"{point:.3f} [{lo:.3f},{hi:.3f}]"
            row[f"_{m}"] = (point, lo, hi)
        rq3.append(row)
    with (OUT / "rq3_channel.csv").open("w", newline="") as f:
        cols = ["channel", "n"] + MODELS
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rq3)

    # ---------------- Exp 3 objective layer: official split ----------------
    # Guide §6.2 objective stratification: one row per NewsCLIPpings
    # falsified-generation mechanism (machine label, zero annotation cost),
    # C-rate + bootstrap CI in the channel-table format, falsified rows
    # only.  Own rng so existing tables' bootstrap draws stay unchanged.
    # Companion: split x channel cross table = the free QC check (person
    # split should concentrate in external_knowledge).
    rng_off = random.Random(42)
    splits_off = sorted({x["split"] for i, x in enumerate(samples)
                         if falsified[i]})
    off_rows = []
    for sp in splits_off:
        idx = [i for i in range(len(samples))
               if samples[i]["split"] == sp and falsified[i]]
        row = {"official_split": sp, "n": len(idx)}
        for m in MODELS:
            per_seed = [[preds[(m, s)][i] == 2 for i in idx] for s in SEEDS]
            point = st.mean(sum(v) / len(v) for v in per_seed)
            boots = []
            for _ in range(BOOT):
                pick = [rng_off.randrange(len(idx)) for _ in idx]
                boots.append(st.mean(
                    sum(v[j] for j in pick) / len(pick) for v in per_seed))
            boots.sort()
            lo, hi = boots[int(0.025 * BOOT)], boots[int(0.975 * BOOT) - 1]
            row[m] = f"{point:.3f} [{lo:.3f},{hi:.3f}]"
        off_rows.append(row)
    with (OUT / "rq3_official_split.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["official_split", "n"] + MODELS)
        w.writeheader(); w.writerows(off_rows)

    qc_rows = []
    for sp in splits_off:
        row = {"official_split": sp}
        for ch in CHANNELS:
            row[ch] = sum(1 for i, x in enumerate(samples)
                          if x["split"] == sp and x["channel"] == ch)
        row["n"] = sum(row[ch] for ch in CHANNELS)
        qc_rows.append(row)
    with (OUT / "rq3_split_channel_qc.csv").open("w", newline="") as f:
        w = csv.DictWriter(f,
                           fieldnames=["official_split"] + CHANNELS + ["n"])
        w.writeheader(); w.writerows(qc_rows)

    for name, rows in (("RQ1 drop", rq1), ("RQ2 H2", rq2)):
        print(f"\n=== {name} ===")
        for r in rows:
            print("  " + " | ".join(f"{k}={v}" for k, v in r.items()))
    print("\n=== RQ3 channel x model (C-rate, 95% bootstrap CI) ===")
    for r in rq3:
        print(f"  {r['channel']:<20} n={r['n']:<4} " +
              " ".join(f"{m}:{r[m]}" for m in MODELS))
    print("\n=== Exp 3 objective layer: official split x model (C-rate) ===")
    for r in off_rows:
        print(f"  {r['official_split']:<28} n={r['n']:<4} " +
              " ".join(f"{m}:{r[m]}" for m in MODELS))
    print("\n=== QC: official split x evidence channel (counts) ===")
    for r in qc_rows:
        print(f"  {r['official_split']:<28} " +
              " ".join(f"{c}:{r[c]}" for c in CHANNELS) + f"  n={r['n']}")

    # ---------------- figures ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 12, "legend.fontsize": 10})
    label = {"camc_reg": "Proposed (CAMC)", "clip_fusion": "CLIP Fusion",
             "late_fusion": "Diff Fusion (A2)", "text_only": "Text-only",
             "image_only": "Image-only", "clip_camc": "CAMC-on-CLIP"}

    # RQ1/RQ2 main figure: horizontal dumbbell, one row per model sorted by
    # in-domain F1 (best on top).  The earlier slope chart tangled at the
    # zero-shot end (all models land within 0.06 of each other); rows plus a
    # printed drop column keep both stories legible — the in-domain spread
    # AND the collapse onto the all-N baseline.  Color encodes the
    # condition (2-slot validated categorical pair), never the model; the
    # connector and all text stay in neutral ink.
    order = ["text_only", "image_only", "late_fusion", "clip_fusion",
             "camc_reg", "clip_camc"]
    C_IND, C_ZS = "#2a78d6", "#eb6834"
    all_n_f1 = f1_score(gold, [1] * len(gold), average="macro")
    stats = {}
    for m in order:
        ind = [indomain[f"{m}_seed{s}"] for s in SEEDS]
        zs = [f1_score(gold, preds[(m, s)], average="macro") for s in SEEDS]
        stats[m] = (st.mean(ind), st.stdev(ind), st.mean(zs), st.stdev(zs))
    rows_m = sorted(order, key=lambda m: stats[m][0])   # best ends on top
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for y, m in enumerate(rows_m):
        im, istd, zm, zstd = stats[m]
        ax.plot([zm, im], [y, y], color="#c9c9c9", lw=1.6, zorder=1)
        ax.errorbar([im], [y], xerr=[istd], fmt="o", ms=8, color=C_IND,
                    ecolor="#9bb9e0", capsize=3, zorder=3,
                    label="In-domain (SNLI-VE corrected test)" if y == 0
                    else None)
        ax.errorbar([zm], [y], xerr=[zstd], fmt="o", ms=8, color=C_ZS,
                    ecolor="#f0b394", capsize=3, zorder=3,
                    label="Zero-shot (NewsCLIPpings 500)" if y == 0
                    else None)
        ax.text(im + 0.013, y, f"{im:.3f}", va="center", ha="left",
                fontsize=9, color="#444444")
        ax.text(0.875, y, f"−{im - zm:.3f}", va="center", ha="left",
                fontsize=9, color="#767676")
    ax.text(0.875, len(rows_m) - 0.45, "drop", fontsize=9, color="#767676",
            ha="left", style="italic")
    ax.axvline(all_n_f1, ls="--", c="0.55", lw=1, zorder=0)
    ax.text(all_n_f1 + 0.006, len(rows_m) - 0.45,
            f"all-N baseline ({all_n_f1:.2f})", fontsize=8.5, color="0.45",
            ha="left")
    ax.set_yticks(range(len(rows_m)), [label[m] for m in rows_m],
                  fontsize=10)
    ax.set_xlabel("Macro-F1")
    ax.set_xlim(0.10, 0.955)
    ax.set_ylim(-0.7, len(rows_m) - 0.1)
    ax.xaxis.grid(True, color="#ededed", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncols=2,
              fontsize=9, frameon=False, columnspacing=1.6,
              handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rq1_drop.svg", bbox_inches="tight")
    plt.close(fig)

    # RQ3 main figure: error rate vs human relation gold, per channel.
    # One direction everywhere (lower = better): on visible_conflict gold
    # is C, elsewhere mostly N — P(pred C) alone flips meaning per group,
    # this metric does not.  Main figure keeps only the three
    # narrative-carrying models; the full five-model P(pred C) version
    # goes to the appendix.
    big_ch = [c for c in CHANNELS
              if sum(x["channel"] == c for x in samples) >= 20]
    main_models = ["text_only", "clip_fusion", "camc_reg", "clip_camc"]
    err_rows = []
    for ch in big_ch:
        idx = [i for i in range(len(samples)) if samples[i]["channel"] == ch]
        row = {"channel": ch, "n": len(idx)}
        for m in MODELS:
            per_seed = [[preds[(m, s)][i] != gold[i] for i in idx]
                        for s in SEEDS]
            point = st.mean(sum(v) / len(v) for v in per_seed)
            boots = []
            for _ in range(BOOT):
                pick = [rng.randrange(len(idx)) for _ in idx]
                boots.append(st.mean(
                    sum(v[j] for j in pick) / len(pick) for v in per_seed))
            boots.sort()
            lo, hi = boots[int(0.025 * BOOT)], boots[int(0.975 * BOOT) - 1]
            row[m] = f"{point:.3f} [{lo:.3f},{hi:.3f}]"
            row[f"_{m}"] = (point, lo, hi)
        err_rows.append(row)
    with (OUT / "rq3_channel_error.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["channel", "n"] + MODELS,
                           extrasaction="ignore")
        w.writeheader(); w.writerows(err_rows)

    x = np.arange(len(err_rows))
    width = 0.8 / len(main_models)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for k, m in enumerate(main_models):
        pts = [r[f"_{m}"][0] for r in err_rows]
        errs = [[r[f"_{m}"][0] - r[f"_{m}"][1] for r in err_rows],
                [r[f"_{m}"][2] - r[f"_{m}"][0] for r in err_rows]]
        ax.bar(x + (k - (len(main_models) - 1) / 2) * width, pts, width,
               yerr=errs, capsize=3, label=label[m])
    ax.set_xticks(x, [f"{r['channel']}\n(n={r['n']})" for r in err_rows])
    ax.set_ylabel("Error rate vs human relation label")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rq3_channel.svg", bbox_inches="tight")
    plt.close(fig)

    # RQ2 figure: prediction distribution, pristine vs falsified, with the
    # human relation gold as reference row.  This replaces the old 4-panel
    # metric grid: the distributions show directly WHY FSR≈0 for everyone
    # (the E channel is dead out-of-domain — humans say E on 54% of
    # pristine, models ≤2%), that mass shifts to N, and that only
    # clip_fusion moves C mass in the manipulation direction.  Remaining
    # H2 metrics live in rq2_h2.csv (table, not figure).
    prist_idx = [i for i in range(len(samples)) if not falsified[i]]
    fals_idx = [i for i in range(len(samples)) if falsified[i]]
    splits = [("pristine", prist_idx), ("falsified", fals_idx)]

    def dist_of(vals: list[int], idx: list[int]) -> list[float]:
        return [sum(vals[i] == k for i in idx) / len(idx) for k in (0, 1, 2)]

    dist_num = {}   # (entity, split) -> [P(E), P(N), P(C)] means
    dist_rows = []
    for split, idx in splits:
        d = dist_of(gold, idx)
        dist_num[("human_gold", split)] = d
        dist_rows.append({"model": "human_gold", "split": split,
                          **{f"P_{c}": f"{d[k]:.4f}"
                             for k, c in enumerate("ENC")}})
    for m in MODELS:
        for split, idx in splits:
            per_seed = [dist_of(preds[(m, s)], idx) for s in SEEDS]
            means = [st.mean(ps[k] for ps in per_seed) for k in range(3)]
            stds = [st.stdev([ps[k] for ps in per_seed]) for k in range(3)]
            dist_num[(m, split)] = means
            dist_rows.append({"model": m, "split": split,
                              **{f"P_{c}": f"{means[k]:.4f}±{stds[k]:.4f}"
                                 for k, c in enumerate("ENC")}})
    with (OUT / "rq2_pred_dist.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dist_rows[0].keys()))
        w.writeheader(); w.writerows(dist_rows)

    # figure keeps the narrative models; image_only stays CSV-only (its
    # per-seed distributions collapse to different classes, std ±0.4)
    ents = [("human_gold", "Human gold"),
            ("camc_reg", label["camc_reg"]),
            ("clip_camc", label["clip_camc"]),
            ("clip_fusion", label["clip_fusion"]),
            ("late_fusion", label["late_fusion"]),
            ("text_only", label["text_only"])]
    seg_colors = {"E": "#6aa84f", "N": "#d9d9d9", "C": "#c44e52"}
    fig, ax = plt.subplots(figsize=(8, 5.6))
    yticks, ylabels = [], []
    for gi, (m, name) in enumerate(ents):
        for si, (split, _) in enumerate(splits):
            y = -(gi * 2.7 + si)
            left = 0.0
            for k, c in enumerate("ENC"):
                v = dist_num[(m, split)][k]
                ax.barh(y, v, left=left, height=0.85, color=seg_colors[c],
                        label=c if gi == 1 and si == 0 else None)
                if v >= 0.045:
                    ax.text(left + v / 2, y, f"{v * 100:.0f}", ha="center",
                            va="center", fontsize=8)
                left += v
            yticks.append(y)
            ylabels.append(f"{name} · {split}")
    ax.set_yticks(yticks, ylabels, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of E / N / C (model rows: mean of 3 seeds)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=3,
              fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rq2_dist.svg", bbox_inches="tight")
    plt.close(fig)

    # Appendix version: all five models, raw P(pred C) with CI
    big = [r for r in rq3 if r["n"] >= 20]
    x = np.arange(len(big))
    width = 0.8 / len(MODELS)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, m in enumerate(order):
        pts = [r[f"_{m}"][0] for r in big]
        errs = [[r[f"_{m}"][0] - r[f"_{m}"][1] for r in big],
                [r[f"_{m}"][2] - r[f"_{m}"][0] for r in big]]
        ax.bar(x + (k - (len(order) - 1) / 2) * width, pts, width,
               yerr=errs, capsize=2, label=label[m])
    ax.set_xticks(x, [f"{r['channel']}\n(n={r['n']})" for r in big])
    ax.set_ylabel("P(pred C)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rq3_channel_appendix.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {OUT}/rq1_drop.csv, rq2_h2.csv, rq2_pred_dist.csv, "
          f"rq3_channel.csv, rq3_official_split.csv, "
          f"rq3_split_channel_qc.csv, fig_rq1_drop.svg (dumbbell), "
          f"fig_rq2_dist.svg, fig_rq3_channel.svg")


if __name__ == "__main__":
    main()
