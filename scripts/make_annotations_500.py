"""Merge the two finalized annotation packets into one 500-row CSV.

Reads data/newsclippings/{pilot_seed42,formal_seed43}/pilot_annotations.csv
(the human-reviewed sources of truth; NOT modified) and writes
data/newsclippings/annotations_500.csv with a leading `packet` provenance
column.  The merged file is a convenience/appendix artifact: every analysis
script keeps reading the per-packet files, which remain authoritative.

Notes column: if data/newsclippings/notes_plain_500.csv exists (sample_id,
notes — a one-time plain-English rewrite of the original telegraphic
annotator shorthand; wording only, judgments untouched), its notes replace
the originals in the merged file.  The original notes always survive in the
per-packet CSVs.

Run from the project root:
    python3 scripts/make_annotations_500.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "data" / "newsclippings"
PACKETS = ["pilot_seed42", "formal_seed43"]
PLAIN_NOTES = BASE / "notes_plain_500.csv"
OUT = BASE / "annotations_500.csv"


def load_veracity() -> dict[int, str]:
    """sample_id -> official veracity (pristine/falsified), from the packet
    sample manifests (the annotation CSVs deliberately omit it: Stage 1 was
    labelled blind to veracity)."""
    ver = {}
    for p in PACKETS:
        for s in json.loads((BASE / p / "pilot_samples.json").read_text()):
            ver[int(s["sample_id"])] = s["original_label"]
    return ver


def main() -> None:
    rows, header = [], None
    for p in PACKETS:
        with (BASE / p / "pilot_annotations.csv").open(
                newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            h = next(r)
            if header is None:
                header = h
            elif h != header:
                raise RuntimeError(f"{p}: header mismatch {h} != {header}")
            rows.extend([p] + line for line in r)

    if len(rows) != 500:
        raise RuntimeError(f"expected 500 rows, got {len(rows)}")
    sid = header.index("sample_id") + 1
    ids = [int(r[sid]) for r in rows]
    if sorted(ids) != list(range(1, 501)):
        raise RuntimeError("sample_id is not exactly 1..500")

    notes_col = header.index("notes") + 1
    if PLAIN_NOTES.exists():
        with PLAIN_NOTES.open(newline="", encoding="utf-8") as f:
            plain = {int(r["sample_id"]): r["notes"]
                     for r in csv.DictReader(f)}
        missing = set(ids) - plain.keys()
        if missing:
            raise RuntimeError(
                f"notes_plain_500.csv lacks {len(missing)} sample_ids, "
                f"e.g. {sorted(missing)[:5]}")
        for r in rows:
            r[notes_col] = plain[int(r[sid])]
        print(f"Applied plain-English notes from {PLAIN_NOTES.name}")
    else:
        print("No notes_plain_500.csv; keeping original notes")

    # veracity column (after official_split): from the sample manifests
    ver = load_veracity()
    sp = header.index("official_split")   # header index (rows are offset +1)
    out_header = (["packet"] + header[:sp + 1] + ["veracity"]
                  + header[sp + 1:])
    out_rows = [r[:sp + 2] + [ver[int(r[sid])]] + r[sp + 2:] for r in rows]

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(out_header)
        w.writerows(out_rows)

    rel = out_header.index("relation_label")
    chan = out_header.index("evidence_channel")
    vcol = out_header.index("veracity")
    print(f"Wrote {OUT} ({len(out_rows)} rows)")
    print("veracity:", dict(Counter(r[vcol] for r in out_rows)))
    print("relation:", dict(Counter(r[rel] for r in out_rows)))
    print("channel: ", dict(Counter(r[chan] for r in out_rows)))


if __name__ == "__main__":
    main()
