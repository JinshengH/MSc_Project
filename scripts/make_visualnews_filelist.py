"""Generate the tar member list of every VisualNews image this project needs.

The project only ever touches images referenced by the merged_balanced test
split (pilot, formal annotation and full-test zero-shot all draw from it), so
origin.tar never needs a full extraction.  Workflow:

1. HPC: extract the index only
       tar -xf origin.tar origin/data.json
2. Sync origin/data.json to local data/visual_news/origin/data.json,
   run this script, sync the resulting list back to the HPC.
3. HPC: selective extraction, then verify before deleting the tar
       tar -xf origin.tar -T visualnews_test_files.txt
4. Sync the extracted files needed locally (pilot images at minimum).

Run with: conda run -n hf_latest python scripts/make_visualnews_filelist.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_JSON = PROJECT_ROOT / "data" / "newsclippings" / "merged_balanced" / "test.json"
VISUALNEWS_DATA_JSON = PROJECT_ROOT / "data" / "visual_news" / "origin" / "data.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "newsclippings" / "visualnews_test_files.txt"
TAR_PREFIX = "origin/"  # verify with: tar -tf origin.tar | head -3


def main() -> None:
    with TEST_JSON.open() as f:
        annotations = json.load(f)["annotations"]
    image_ids = {a["image_id"] for a in annotations}
    print(f"test split: {len(annotations)} samples, "
          f"{len(image_ids)} unique image ids")

    with VISUALNEWS_DATA_JSON.open() as f:
        entries = json.load(f)
    needed = {e["id"]: e for e in entries if e["id"] in image_ids}
    missing = image_ids - needed.keys()
    if missing:
        raise RuntimeError(
            f"{len(missing)} image ids missing from data.json, "
            f"e.g. {sorted(missing)[:5]}")

    by_source = Counter(e.get("source", "unknown") for e in needed.values())
    print("images per source:", dict(by_source.most_common()))

    members = sorted(
        TAR_PREFIX + e["image_path"].lstrip("./") for e in needed.values())
    OUTPUT_FILE.write_text("\n".join(members) + "\n", encoding="utf-8")
    print(f"Wrote {len(members)} tar members to {OUTPUT_FILE}")
    print("HPC: tar -xf origin.tar -T visualnews_test_files.txt "
          "(verify extraction before deleting the tar)")


if __name__ == "__main__":
    main()
