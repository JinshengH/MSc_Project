"""Create the formal-annotation extension packet (300 samples, seed 43).

Extends the reviewed pilot (200) to the confirmed formal scope of 500:
draws a balanced 150/150 seeded sample from merged_balanced test,
**excluding every caption id and image id used by the pilot**, and builds
a packet with the same internal layout as pilot_seed42 (pilot_samples.json,
blank pilot_annotations.csv, thumbnails/, review-less), so the annotation
UI generator and the zero-shot/analysis tooling work unchanged on it.

sample_id runs 201-500 so ids stay unique across the combined 500 set.
Annotation follows ANNOTATION_GUIDELINES.md (finalised via the pilot).

Run with an environment that provides Pillow, for example:
    conda run -n hf_latest python scripts/prepare_newsclippings_formal300.py
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from prepare_newsclippings_pilot import (
    PROJECT_ROOT, VISUALNEWS_ORIGIN, load_test_payload, resolve_image_path)

PILOT_DIR = PROJECT_ROOT / "data" / "newsclippings" / "pilot_seed42"
OUTPUT_DIR = PROJECT_ROOT / "data" / "newsclippings" / "formal_seed43"

SEED = 43
SAMPLES_PER_CLASS = 150
SAMPLE_ID_START = 201  # pilot used 1-200
THUMBNAIL_SIZE = (480, 360)


def load_pilot_exclusions() -> tuple[set[int], set[int]]:
    pilot = json.loads((PILOT_DIR / "pilot_samples.json").read_text())
    return ({s["caption_id"] for s in pilot}, {s["image_id"] for s in pilot})


def select_extension(annotations: list[dict],
                     split_names: dict[str, str]) -> list[dict]:
    rng = random.Random(SEED)
    used_caption_ids, used_image_ids = load_pilot_exclusions()
    falsified_pool = [i for i, a in enumerate(annotations) if a["falsified"]]
    pristine_pool = [i for i, a in enumerate(annotations) if not a["falsified"]]

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

    falsified_selected = pick(falsified_pool, SAMPLES_PER_CLASS)
    pristine_selected = pick(pristine_pool, SAMPLES_PER_CLASS)

    records = []
    for sample_id, index in enumerate(
            falsified_selected + pristine_selected, start=SAMPLE_ID_START):
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


def write_metadata(records: list[dict]) -> None:
    metadata = {
        "purpose": ("NewsCLIPpings formal-annotation extension packet "
                    "(scope 500 = pilot 200 + this 300, decided 2026-07-18); "
                    "evaluation-only, never training data"),
        "seed": SEED,
        "samples_per_class": SAMPLES_PER_CLASS,
        "sample_id_range": [SAMPLE_ID_START,
                            SAMPLE_ID_START + 2 * SAMPLES_PER_CLASS - 1],
        "excludes": "all caption ids and image ids of pilot_seed42",
        "guideline": "ANNOTATION_GUIDELINES.md (finalised via pilot calibration)",
        "n_falsified": sum(
            1 for r in records if r["original_label"] == "falsified"),
        "n_pristine": sum(
            1 for r in records if r["original_label"] == "pristine"),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    annotation_csv = OUTPUT_DIR / "pilot_annotations.csv"
    if annotation_csv.exists():
        raise FileExistsError(
            f"Refusing to touch a packet with an annotation CSV: "
            f"{annotation_csv}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annotations, split_names = load_test_payload()
    records = select_extension(annotations, split_names)

    manifest_path = OUTPUT_DIR / "sample_manifest.csv"
    if manifest_path.exists():
        verify_manifest(records)
        print("Existing manifest verified against the seeded selection")
    else:
        write_manifest(records)
        print(f"Wrote frozen selection: {manifest_path}")
    write_metadata(records)

    from collections import Counter

    from PIL import Image, ImageOps

    needed_ids = {r["caption_id"] for r in records}
    needed_ids |= {r["image_id"] for r in records}
    data_json = VISUALNEWS_ORIGIN / "data.json"
    with data_json.open() as f:
        entries = json.load(f)
    index = {e["id"]: e for e in entries if e["id"] in needed_ids}
    missing_ids = needed_ids - index.keys()
    if missing_ids:
        raise RuntimeError(f"{len(missing_ids)} ids missing from data.json")

    for record in records:
        record["caption"] = index[record["caption_id"]]["caption"]
        image_entry = index[record["image_id"]]
        record["source"] = image_entry.get("source", "unknown")
        record["original_caption"] = image_entry.get("caption", "")
        record["image_path"] = str(
            resolve_image_path(image_entry).relative_to(PROJECT_ROOT))

    print("images per source:",
          dict(Counter(r["source"] for r in records).most_common()))

    missing = [r for r in records
               if not (PROJECT_ROOT / r["image_path"]).exists()]
    if missing:
        filelist = OUTPUT_DIR / "formal_image_files.txt"
        filelist.write_text("\n".join(sorted(
            "origin/" + index[r["image_id"]]["image_path"].lstrip("./")
            for r in missing)) + "\n", encoding="utf-8")
        print(f"{len(missing)}/{len(records)} images not on disk; "
              f"wrote tar member list: {filelist}")
        print("Extract from origin.tar, then rerun to complete the packet.")
        return

    (OUTPUT_DIR / "thumbnails").mkdir(exist_ok=True)
    for record in records:
        thumbnail = ImageOps.contain(
            Image.open(PROJECT_ROOT / record["image_path"]).convert("RGB"),
            THUMBNAIL_SIZE)
        thumbnail.save(
            OUTPUT_DIR / "thumbnails" / f"{record['sample_id']:03d}.jpg",
            quality=88)

    (OUTPUT_DIR / "pilot_samples.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8")

    fieldnames = ["display_order", "sample_id", "caption_id", "image_id",
                  "official_split", "caption", "relation_label",
                  "evidence_channel", "annotator_confidence", "notes"]
    with annotation_csv.open("w", newline="", encoding="utf-8") as f:
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
    print(f"Completed extension packet ({len(records)} samples) in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
