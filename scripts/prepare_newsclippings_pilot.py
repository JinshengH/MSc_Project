"""Create the fixed NewsCLIPpings pilot packet for two-layer relabelling.

Draws a balanced, seeded sample from the merged_balanced test split
(final.md v2 §3.2: pilot 100-200; §3.3: two-layer annotation, layer 2
redefined by handoff addendum 2 as the evidence channel needed to detect
the mismatch).  Layer-1 relation annotation is blind: the annotator sees
only image and caption; original labels stay in the manifest.  Layer-2
evidence-channel annotation compares the mismatched caption against the
image's original caption (falsified samples only; pristine is n/a).

Runs in two modes depending on whether VisualNews has been downloaded:
- Annotations-only mode (VisualNews data.json missing): writes the sample
  manifest + metadata so the selection is frozen now.  Rerun after
  VisualNews arrives; the seeded selection is deterministic and is verified
  against the existing manifest before the packet is completed.
- Full mode (VisualNews present): joins captions and image paths, copies
  thumbnails, and writes review.html + the blank annotation CSV.

Run with an environment that provides Pillow, for example:
    conda run -n hf_latest python scripts/prepare_newsclippings_pilot.py
"""

from __future__ import annotations

import csv
import html
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_JSON = PROJECT_ROOT / "data" / "newsclippings" / "merged_balanced" / "test.json"
VISUALNEWS_ORIGIN = PROJECT_ROOT / "data" / "visual_news" / "origin"
OUTPUT_DIR = PROJECT_ROOT / "data" / "newsclippings" / "pilot_seed42"

SEED = 42
SAMPLES_PER_CLASS = 100  # 100 pristine + 100 falsified = 200 pilot samples
THUMBNAIL_SIZE = (480, 360)

RELATION_VALUES = "entailment / neutral / contradiction"
EVIDENCE_CHANNELS = (
    "visible_conflict / temporal_provenance / event_provenance / "
    "external_knowledge / other"
)
CONFIDENCE_VALUES = "high / medium / low"


def load_test_payload() -> tuple[list[dict], dict[str, str]]:
    with TEST_JSON.open() as f:
        payload = json.load(f)
    return payload["annotations"], payload["source_datasets"]


def select_pilot(annotations: list[dict],
                 split_names: dict[str, str]) -> list[dict]:
    """Seeded balanced selection with unique caption ids and image ids."""
    rng = random.Random(SEED)
    falsified_pool = [i for i, a in enumerate(annotations) if a["falsified"]]
    pristine_pool = [i for i, a in enumerate(annotations) if not a["falsified"]]

    used_caption_ids: set[int] = set()
    used_image_ids: set[int] = set()

    def pick(pool: list[int], count: int) -> list[int]:
        selected: list[int] = []
        for index in rng.sample(pool, len(pool)):
            entry = annotations[index]
            if entry["id"] in used_caption_ids:
                continue
            if entry["image_id"] in used_image_ids:
                continue
            selected.append(index)
            used_caption_ids.add(entry["id"])
            used_image_ids.add(entry["image_id"])
            if len(selected) == count:
                return selected
        raise ValueError(
            f"Pool exhausted: got {len(selected)}/{count} unique samples")

    # Falsified first, then pristine excluding any overlapping ids, so the
    # 200 pilot items share no caption or image.
    falsified_selected = pick(falsified_pool, SAMPLES_PER_CLASS)
    pristine_selected = pick(pristine_pool, SAMPLES_PER_CLASS)

    records = []
    for sample_id, index in enumerate(
            falsified_selected + pristine_selected, start=1):
        entry = annotations[index]
        records.append({
            "sample_id": sample_id,
            "ann_index": index,
            "caption_id": entry["id"],
            "image_id": entry["image_id"],
            "original_label": "falsified" if entry["falsified"] else "pristine",
            "official_split": split_names[str(entry["source_dataset"])],
            "similarity_score": entry["similarity_score"],
        })
    # Shuffle presentation order so annotators cannot infer the label from
    # position (first half falsified, second half pristine).
    rng.shuffle(records)
    for display_order, record in enumerate(records, start=1):
        record["display_order"] = display_order
    records.sort(key=lambda r: r["display_order"])
    return records


def write_manifest(records: list[dict]) -> None:
    fieldnames = ["display_order", "sample_id", "ann_index", "caption_id",
                  "image_id", "original_label", "similarity_score"]
    with (OUTPUT_DIR / "sample_manifest.csv").open(
            "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fieldnames})


def verify_manifest(records: list[dict]) -> None:
    """A rerun must reproduce the frozen selection exactly."""
    with (OUTPUT_DIR / "sample_manifest.csv").open(encoding="utf-8") as f:
        existing = list(csv.DictReader(f))
    if len(existing) != len(records):
        raise RuntimeError("Manifest length mismatch; selection drifted")
    for row, record in zip(existing, records):
        if (int(row["caption_id"]) != record["caption_id"]
                or int(row["image_id"]) != record["image_id"]
                or row["original_label"] != record["original_label"]):
            raise RuntimeError(
                f"Manifest mismatch at sample {record['sample_id']}; "
                "selection drifted, refusing to continue")


def write_metadata(records: list[dict], visualnews_ready: bool) -> None:
    metadata = {
        "purpose": ("NewsCLIPpings pilot two-layer relabelling packet "
                    "(final.md v2 §3.2/§3.3); evaluation-only, never training data"),
        "source": str(TEST_JSON.relative_to(PROJECT_ROOT)),
        "seed": SEED,
        "samples_per_class": SAMPLES_PER_CLASS,
        "selection": ("random.Random(seed): falsified then pristine, "
                      "rejecting duplicate caption ids and image ids; "
                      "presentation order shuffled with the same rng"),
        "blinding": ("Stage 1 (relation) is blind: image + caption only. "
                     "Stage 2 (evidence channel, falsified only) reveals the "
                     "image's original caption by design (addendum 2 §1.1: "
                     "annotation = caption comparison, not intent inference)"),
        "layers": {
            "relation_label": RELATION_VALUES,
            "evidence_channel": EVIDENCE_CHANNELS
                + " (falsified only; decision tree in addendum 2 §1.3; "
                  "priority: visible > temporal > event > external; "
                  "pristine recorded as n/a)",
            "annotator_confidence": CONFIDENCE_VALUES,
        },
        "official_split": "machine label from merged_balanced source_dataset "
                          "(objective stratification layer, addendum 2 §2)",
        "visualnews_joined": visualnews_ready,
        "n_falsified": sum(
            1 for r in records if r["original_label"] == "falsified"),
        "n_pristine": sum(
            1 for r in records if r["original_label"] == "pristine"),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_visualnews_index(needed_ids: set[int]) -> dict[int, dict]:
    data_json = VISUALNEWS_ORIGIN / "data.json"
    print(f"Loading VisualNews index: {data_json}")
    with data_json.open() as f:
        entries = json.load(f)
    index = {e["id"]: e for e in entries if e["id"] in needed_ids}
    missing = needed_ids - index.keys()
    if missing:
        raise RuntimeError(
            f"{len(missing)} ids missing from VisualNews data.json, "
            f"e.g. {sorted(missing)[:5]}")
    return index


def resolve_image_path(entry: dict) -> Path:
    return VISUALNEWS_ORIGIN / entry["image_path"].lstrip("./")


def write_samples_json(records: list[dict]) -> None:
    """Machine-readable join used by the zero-shot inference script."""
    (OUTPUT_DIR / "pilot_samples.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8")


def write_annotation_csv(records: list[dict]) -> None:
    path = OUTPUT_DIR / "pilot_annotations.csv"
    fieldnames = ["display_order", "sample_id", "caption_id", "image_id",
                  "official_split", "caption", "relation_label",
                  "evidence_channel", "annotator_confidence", "notes"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "display_order": record["display_order"],
                "sample_id": record["sample_id"],
                "caption_id": record["caption_id"],
                "image_id": record["image_id"],
                "official_split": record["official_split"],
                "caption": record["caption"],
            })


def write_review_html(records: list[dict]) -> None:
    cards: list[str] = []
    for record in records:
        cards.append(
            "<article>"
            f"<img src=\"thumbnails/{record['sample_id']:03d}.jpg\" "
            f"alt=\"Sample {record['sample_id']:03d}\" loading=\"lazy\">"
            f"<h2>#{record['display_order']:03d} "
            f"(sample {record['sample_id']:03d})</h2>"
            f"<p>{html.escape(str(record['caption']))}</p>"
            "</article>"
        )

    page = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NewsCLIPpings pilot: two-layer relabelling</title>
  <style>
    body { background: #f5f6f8; color: #1c2026; font: 16px/1.45 system-ui, sans-serif; margin: 0; }
    header { background: #fff; border-bottom: 1px solid #d9dee7; padding: 24px max(24px, calc((100% - 1120px)/2)); }
    h1 { margin: 0 0 6px; font-size: 1.45rem; }
    header p, header ul { margin: 6px 0 0; max-width: 900px; }
    main { display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); margin: 24px auto; max-width: 1120px; padding: 0 24px; }
    article { background: #fff; border: 1px solid #d9dee7; border-radius: 8px; overflow: hidden; padding-bottom: 14px; }
    img { display: block; height: 240px; object-fit: contain; width: 100%; background: #eceef2; }
    h2, article p { margin: 12px 14px 0; }
    h2 { font-size: 1rem; color: #606a78; }
    code { background: #eceef2; border-radius: 4px; padding: 0 4px; }
  </style>
</head>
<body>
  <header>
    <h1>NewsCLIPpings pilot: two-layer relabelling</h1>
    <p>Fixed seeded sample from the merged_balanced test split, presentation
    order shuffled. Browse-only page (image + caption, blind). For actual
    annotation use <code>annotate.html</code>: stage 1 relation
    (""" + RELATION_VALUES + """), stage 2 evidence channel
    (""" + EVIDENCE_CHANNELS + """, falsified only).</p>
  </header>
  <main>
""" + "\n".join(cards) + """
  </main>
</body>
</html>
"""
    (OUTPUT_DIR / "review.html").write_text(page, encoding="utf-8")


def main() -> None:
    annotation_csv = OUTPUT_DIR / "pilot_annotations.csv"
    if annotation_csv.exists():
        raise FileExistsError(
            f"Refusing to touch a packet with an annotation CSV: "
            f"{annotation_csv}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annotations, split_names = load_test_payload()
    records = select_pilot(annotations, split_names)

    manifest_path = OUTPUT_DIR / "sample_manifest.csv"
    if manifest_path.exists():
        verify_manifest(records)
        print("Existing manifest verified against the seeded selection")
    else:
        write_manifest(records)
        print(f"Wrote frozen selection: {manifest_path}")

    visualnews_ready = (VISUALNEWS_ORIGIN / "data.json").exists()
    write_metadata(records, visualnews_ready)

    if not visualnews_ready:
        print(f"VisualNews not found at {VISUALNEWS_ORIGIN}/data.json; "
              "wrote manifest + metadata only.")
        print("Rerun this script after extracting origin.tar to complete "
              "the packet (captions, thumbnails, review.html, blank CSV).")
        return

    from collections import Counter

    from PIL import Image, ImageOps  # only needed in full mode

    needed_ids = {r["caption_id"] for r in records}
    needed_ids |= {r["image_id"] for r in records}
    index = load_visualnews_index(needed_ids)

    for record in records:
        record["caption"] = index[record["caption_id"]]["caption"]
        image_entry = index[record["image_id"]]
        record["source"] = image_entry.get("source", "unknown")
        # Original caption of the image itself: the layer-2 comparison basis
        # (addendum 2 §1.1); equals `caption` for pristine samples.
        record["original_caption"] = image_entry.get("caption", "")
        record["image_path"] = str(
            resolve_image_path(image_entry).relative_to(PROJECT_ROOT))

    by_source = Counter(r["source"] for r in records)
    print("pilot images per source:", dict(by_source.most_common()))

    missing = [r for r in records
               if not (PROJECT_ROOT / r["image_path"]).exists()]
    if missing:
        # Partial sync: emit the tar member list for a selective extraction
        # (tar -xf origin.tar -T pilot_image_files.txt) instead of failing.
        filelist = OUTPUT_DIR / "pilot_image_files.txt"
        filelist.write_text("\n".join(sorted(
            "origin/" + index[r["image_id"]]["image_path"].lstrip("./")
            for r in missing)) + "\n", encoding="utf-8")
        print(f"{len(missing)}/{len(records)} pilot images not on disk; "
              f"wrote tar member list: {filelist}")
        print("Extract them from origin.tar, sync under data/visual_news/, "
              "then rerun to complete the packet.")
        return

    (OUTPUT_DIR / "thumbnails").mkdir(exist_ok=True)
    for record in records:
        thumbnail = ImageOps.contain(
            Image.open(PROJECT_ROOT / record["image_path"]).convert("RGB"),
            THUMBNAIL_SIZE)
        thumbnail.save(
            OUTPUT_DIR / "thumbnails" / f"{record['sample_id']:03d}.jpg",
            quality=88)

    write_samples_json(records)
    write_annotation_csv(records)
    write_review_html(records)
    print(f"Completed pilot packet ({len(records)} samples) in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
