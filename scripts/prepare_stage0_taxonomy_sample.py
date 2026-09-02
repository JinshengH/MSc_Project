"""Create a fixed, blind review packet for the Stage 0 taxonomy check.

The packet is drawn only from SNLI-VE's official test split and is never a
training input.  It contains 200 randomly selected, gold-contradiction items
for manually checking whether the proposed taxonomy is viable in SNLI-VE.

Run with an environment that provides ``datasets`` and Pillow, for example:
    conda run -n hf_latest python scripts/prepare_stage0_taxonomy_sample.py
"""

from __future__ import annotations

import csv
import html
import json
import random
from pathlib import Path

from datasets import load_from_disk
from PIL import Image, ImageDraw, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "jupyter" / "snli-ve"
OUTPUT_DIR = PROJECT_ROOT / "jupyter" / "results" / "stage0_taxonomy_seed42"
SPLIT = "test"
CONTRADICTION_LABEL = 2
SAMPLE_SIZE = 200
SEED = 42

THUMBNAIL_SIZE = (360, 270)
SHEET_COLUMNS = 4
SHEET_ROWS = 5
CARD_SIZE = (430, 360)


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    max_width: int,
    max_lines: int,
) -> None:
    """Draw a compact, clipped hypothesis without relying on system fonts."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)

    for line_number, line in enumerate(lines[:max_lines]):
        if line_number == max_lines - 1 and len(lines) > max_lines:
            line = f"{line[:-3]}..." if len(line) > 3 else "..."
        draw.text((xy[0], xy[1] + line_number * 17), line, fill="black")


def make_contact_sheet(records: list[dict[str, object]], sheet_number: int) -> None:
    sheet_width = SHEET_COLUMNS * CARD_SIZE[0]
    sheet_height = SHEET_ROWS * CARD_SIZE[1]
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)

    for position, record in enumerate(records):
        column = position % SHEET_COLUMNS
        row = position // SHEET_COLUMNS
        left = column * CARD_SIZE[0]
        top = row * CARD_SIZE[1]

        card = Image.new("RGB", CARD_SIZE, "#f6f7f9")
        card_draw = ImageDraw.Draw(card)
        image = record["image"].convert("RGB")
        image = ImageOps.contain(image, THUMBNAIL_SIZE)
        image_left = (CARD_SIZE[0] - image.width) // 2
        card.paste(image, (image_left, 28))
        card_draw.rectangle(
            (0, 0, CARD_SIZE[0] - 1, CARD_SIZE[1] - 1), outline="#b8bec7"
        )
        card_draw.text(
            (12, 8),
            f"Sample {record['sample_id']:03d}  |  test index {record['test_index']}",
            fill="black",
        )
        draw_wrapped_text(
            card_draw,
            record["hypothesis"],
            (12, THUMBNAIL_SIZE[1] + 42),
            CARD_SIZE[0] - 24,
            max_lines=3,
        )
        sheet.paste(card, (left, top))

    sheet.save(OUTPUT_DIR / "contact_sheets" / f"sheet_{sheet_number:02d}.jpg", quality=90)


def write_review_html(records: list[dict[str, object]]) -> None:
    cards: list[str] = []
    for record in records:
        cards.append(
            "<article>"
            f"<img src=\"thumbnails/{record['sample_id']:03d}.jpg\" "
            f"alt=\"Sample {record['sample_id']:03d}\">"
            f"<h2>Sample {record['sample_id']:03d}</h2>"
            f"<p>{html.escape(str(record['hypothesis']))}</p>"
            "<p class=\"hint\">Record subtype, confidence, and notes in "
            "<code>stage0_annotations.csv</code>.</p>"
            "</article>"
        )

    page = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SNLI-VE Stage 0 taxonomy review</title>
  <style>
    body { background: #f5f6f8; color: #1c2026; font: 16px/1.45 system-ui, sans-serif; margin: 0; }
    header { background: #fff; border-bottom: 1px solid #d9dee7; padding: 24px max(24px, calc((100% - 1120px)/2)); }
    h1 { margin: 0 0 6px; font-size: 1.45rem; }
    header p { margin: 0; max-width: 900px; }
    main { display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); margin: 24px auto; max-width: 1120px; padding: 0 24px; }
    article { background: #fff; border: 1px solid #d9dee7; border-radius: 8px; overflow: hidden; padding-bottom: 14px; }
    img { display: block; height: 230px; object-fit: contain; width: 100%; }
    h2, article p { margin: 12px 14px 0; }
    h2 { font-size: 1rem; }
    .hint { color: #606a78; font-size: .85rem; }
  </style>
</head>
<body>
  <header>
    <h1>Stage 0: taxonomy feasibility check</h1>
    <p>200 fixed random samples from the official SNLI-VE test split. Each is a gold contradiction, but the review deliberately shows only the image and claim. Do not use this packet for training or model selection.</p>
  </header>
  <main>
""" + "\n".join(cards) + """
  </main>
</body>
</html>
"""
    (OUTPUT_DIR / "review.html").write_text(page, encoding="utf-8")


def main() -> None:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing review packet: {OUTPUT_DIR}"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "thumbnails").mkdir()
    (OUTPUT_DIR / "contact_sheets").mkdir()

    dataset = load_from_disk(DATA_DIR)
    test_split = dataset[SPLIT]
    candidates = [
        index for index, label in enumerate(test_split["label"])
        if label == CONTRADICTION_LABEL
    ]
    if len(candidates) < SAMPLE_SIZE:
        raise ValueError(
            f"Only {len(candidates)} contradiction samples are available; "
            f"need {SAMPLE_SIZE}."
        )

    selected_indices = random.Random(SEED).sample(candidates, SAMPLE_SIZE)
    records: list[dict[str, object]] = []
    for sample_id, test_index in enumerate(selected_indices, start=1):
        example = test_split[test_index]
        if example["label"] != CONTRADICTION_LABEL:
            raise RuntimeError(f"Sample {test_index} is not a contradiction")
        thumbnail = ImageOps.contain(example["image"].convert("RGB"), THUMBNAIL_SIZE)
        thumbnail.save(
            OUTPUT_DIR / "thumbnails" / f"{sample_id:03d}.jpg", quality=88
        )
        records.append(
            {
                "sample_id": sample_id,
                "test_index": test_index,
                "filename": example["filename"],
                "hypothesis": example["hypothesis"],
                "image": example["image"],
            }
        )

    with (OUTPUT_DIR / "stage0_annotations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id", "test_index", "filename", "hypothesis",
                "taxonomy", "confidence", "review_notes", "reviewed",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: record[key]
                    for key in ("sample_id", "test_index", "filename", "hypothesis")
                }
            )

    with (OUTPUT_DIR / "sample_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id", "test_index", "filename", "hypothesis", "source_label",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **{
                        key: record[key]
                        for key in ("sample_id", "test_index", "filename", "hypothesis")
                    },
                    "source_label": "contradiction",
                }
            )

    metadata = {
        "purpose": "Stage 0 taxonomy feasibility review only; never training data",
        "dataset_path": str(DATA_DIR),
        "split": SPLIT,
        "split_fingerprint": test_split._fingerprint,
        "source_label": "contradiction",
        "source_label_id": CONTRADICTION_LABEL,
        "candidate_count": len(candidates),
        "sample_size": SAMPLE_SIZE,
        "seed": SEED,
        "selection": "random.Random(seed).sample(candidate_test_indices, sample_size)",
        "review_blinding": "Images and hypotheses only; no premise or label appears in review.html.",
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    for offset in range(0, len(records), SHEET_COLUMNS * SHEET_ROWS):
        make_contact_sheet(
            records[offset:offset + SHEET_COLUMNS * SHEET_ROWS],
            sheet_number=offset // (SHEET_COLUMNS * SHEET_ROWS) + 1,
        )
    write_review_html(records)

    print(f"Created {len(records)} Stage 0 review samples in {OUTPUT_DIR}")
    print(f"Candidate contradiction count: {len(candidates)}; seed: {SEED}")


if __name__ == "__main__":
    main()
