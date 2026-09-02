"""Stratified cross-analysis: zero-shot predictions vs pilot annotation layers.

Reads the human annotation CSV (pilot_annotations.csv if the reviewed export
exists, else falls back to the AI prelabel and marks output PRELIMINARY) and
each predictions CSV in pilot_seed42/zero_shot/, then prints per model:

  1. human x model confusion + overall agreement (with N-collision caveat)
  2. falsified: model prediction by evidence channel (the falsifiable test:
     C should concentrate in visible_conflict)
  3. where model C predictions land (visible / other channels / pristine)
  4. pristine behaviour split by human E/N (false-alarm accounting)
  5. detection signal: C-rate pristine vs falsified
  6. official_split x model prediction (objective stratification)

Run with: python3 scripts/analyze_pilot_zero_shot.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "newsclippings"
# packet dir names as CLI args; sample_ids are unique across packets by
# construction (pilot 1-200, formal extension 201-500), so pooling is a union
PACKETS = sys.argv[1:] or ["pilot_seed42"]
SHORT = {"entailment": "E", "neutral": "N", "contradiction": "C"}


def load_reference(packet: Path) -> tuple[dict[int, dict], str]:
    reviewed = packet / "pilot_annotations.csv"
    prelabel = packet / "pilot_annotations_agent_prelabel.csv"
    # the blank template has empty relation_label columns; only accept a
    # reviewed export if labels are actually filled in
    if reviewed.exists():
        rows = {int(r["sample_id"]): r for r in csv.DictReader(open(reviewed))}
        if all(r["relation_label"] for r in rows.values()):
            return rows, f"{packet.name}: reviewed"
    rows = {int(r["sample_id"]): r for r in csv.DictReader(open(prelabel))}
    return rows, f"{packet.name}: AI PRELABEL (PRELIMINARY)"


def main() -> None:
    human: dict[int, dict] = {}
    sources = []
    exp_preds: dict[str, dict[int, dict]] = defaultdict(dict)
    for name in PACKETS:
        packet = DATA / name
        rows, src = load_reference(packet)
        assert not (rows.keys() & human.keys()), "sample_id overlap across packets"
        human.update(rows)
        sources.append(src)
        for pred_file in sorted((packet / "zero_shot").glob("*_predictions.csv")):
            exp = pred_file.name.replace("_predictions.csv", "")
            exp_preds[exp].update(
                {int(r["sample_id"]): r for r in csv.DictReader(open(pred_file))})
    print(f"reference: {' + '.join(sources)}  (n={len(human)})")
    for exp, preds in sorted(exp_preds.items()):
        if preds.keys() < human.keys():
            print(f"\n=== {exp} === SKIPPED (predictions cover "
                  f"{len(preds)}/{len(human)} samples)")
            continue
        print(f"\n=== {exp} ===")

        conf = defaultdict(Counter)
        for sid, h in human.items():
            conf[SHORT[h["relation_label"]]][SHORT[preds[sid]["pred_label"]]] += 1
        print("confusion human\\model:", {k: dict(v) for k, v in sorted(conf.items())})
        agree = sum(conf[k][k] for k in conf)
        nn = conf["N"]["N"]
        print(f"overall agreement: {agree}/{len(human)} = {agree/len(human):.2f}"
              f"  (of which N-N collision: {nn} - report stratified, not this)")

        print("falsified - model pred by channel:")
        by_ch = defaultdict(Counter)
        for sid, h in human.items():
            if h["evidence_channel"] != "n/a":
                by_ch[h["evidence_channel"]][SHORT[preds[sid]["pred_label"]]] += 1
        for ch, c in sorted(by_ch.items(), key=lambda x: -sum(x[1].values())):
            print(f"  {ch:20s} n={sum(c.values()):3d}  {dict(c)}")

        c_preds = [sid for sid in human if preds[sid]["pred_label"] == "contradiction"]
        c_vis = sum(1 for s in c_preds if human[s]["evidence_channel"] == "visible_conflict")
        c_pri = sum(1 for s in c_preds if human[s]["evidence_channel"] == "n/a")
        print(f"model C predictions: {len(c_preds)} total; on visible_conflict: {c_vis}; "
              f"on pristine (false alarms): {c_pri}")

        pr = defaultdict(Counter)
        for sid, h in human.items():
            if h["evidence_channel"] == "n/a":
                pr[SHORT[h["relation_label"]]][SHORT[preds[sid]["pred_label"]]] += 1
        print("pristine - human\\model:", {k: dict(v) for k, v in sorted(pr.items())})

        n_pri = sum(1 for h in human.values() if h["evidence_channel"] == "n/a")
        n_fal = len(human) - n_pri
        cr_p = sum(1 for s, h in human.items()
                   if h["evidence_channel"] == "n/a" and preds[s]["pred_label"] == "contradiction")
        cr_f = sum(1 for s, h in human.items()
                   if h["evidence_channel"] != "n/a" and preds[s]["pred_label"] == "contradiction")
        print(f"C-rate: pristine {cr_p}/{n_pri} vs falsified {cr_f}/{n_fal}")

        by_split = defaultdict(Counter)
        for sid, h in human.items():
            by_split[h["official_split"]][SHORT[preds[sid]["pred_label"]]] += 1
        print("official_split x model pred:")
        for sp, c in sorted(by_split.items()):
            print(f"  {sp:28s} {dict(c)}")


if __name__ == "__main__":
    main()
