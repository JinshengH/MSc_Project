"""Write the preliminary, image-and-claim-only Stage 0 taxonomy audit.

This records an AI-assisted feasibility pass over the fixed packet produced by
``prepare_stage0_taxonomy_sample.py``.  It is deliberately kept separate from
``stage0_annotations.csv`` so a human annotator can independently review the
same blind packet before the taxonomy is treated as final.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "jupyter" / "results" / "stage0_taxonomy_seed42"

CATEGORY_TO_SAMPLE_IDS = {
    "action_state": {
        3, 6, 8, 9, 10, 11, 16, 18, 20, 24, 28, 30, 31, 33, 36, 43, 55,
        57, 64, 67, 71, 72, 74, 76, 77, 80, 81, 82, 87, 88, 90, 95, 106,
        107, 108, 109, 116, 119, 120, 122, 123, 125, 127, 134, 135, 140,
        141, 147, 151, 156, 161, 164, 168, 176, 178, 180, 181, 182, 183,
        185, 186, 188, 199, 200,
    },
    "object_entity": {
        4, 7, 13, 23, 26, 29, 34, 39, 41, 44, 46, 62, 69, 75, 89, 91, 94,
        97, 102, 104, 105, 110, 112, 117, 118, 124, 126, 128, 130, 136,
        142, 143, 150, 155, 158, 160, 166, 170, 171, 174, 175, 177, 187,
        192, 195, 198,
    },
    "attribute_quantity": {
        1, 2, 5, 14, 21, 25, 42, 56, 58, 59, 65, 79, 83, 84, 86, 96, 98,
        99, 101, 114, 115, 138, 139, 144, 145, 153, 157, 165, 167, 169,
        172, 189, 197,
    },
    "scene_spatial": {
        12, 15, 17, 19, 27, 32, 35, 37, 38, 40, 48, 49, 50, 51, 52, 53,
        60, 61, 66, 70, 73, 78, 92, 93, 100, 103, 111, 129, 131, 132, 146,
        148, 149, 152, 154, 159, 163, 173, 184, 191, 193, 194, 196,
    },
    "nonvisual_or_ambiguous": {
        22, 45, 47, 54, 63, 68, 85, 113, 121, 133, 137, 162, 179, 190,
    },
}

CATEGORY_DEFINITIONS = {
    "action_state": "Observable action, event, state, or its explicit negation.",
    "object_entity": "Incorrect object, person, animal, or identity/category.",
    "attribute_quantity": "Incorrect visible attribute, apparel, colour, age, or quantity.",
    "scene_spatial": "Incorrect scene, location, direction, or spatial relation.",
    "nonvisual_or_ambiguous": (
        "Requires an unshown relation, duration, intent, or other evidence not "
        "reliably recoverable from this image-claim pair."
    ),
}


def main() -> None:
    expected_ids = set(range(1, 201))
    assigned_ids = set().union(*CATEGORY_TO_SAMPLE_IDS.values())
    if assigned_ids != expected_ids:
        missing = sorted(expected_ids - assigned_ids)
        unexpected = sorted(assigned_ids - expected_ids)
        raise ValueError(f"Invalid coverage; missing={missing}, unexpected={unexpected}")
    total_assignments = sum(len(ids) for ids in CATEGORY_TO_SAMPLE_IDS.values())
    if total_assignments != len(assigned_ids):
        raise ValueError("At least one sample was assigned to multiple categories")

    sample_to_category = {
        sample_id: category
        for category, sample_ids in CATEGORY_TO_SAMPLE_IDS.items()
        for sample_id in sample_ids
    }
    source_path = OUTPUT_DIR / "stage0_annotations.csv"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Create the review packet first: {source_path} does not exist"
        )

    with source_path.open(encoding="utf-8", newline="") as file:
        source_rows = list(csv.DictReader(file))
    if len(source_rows) != 200:
        raise ValueError(f"Expected 200 review rows, found {len(source_rows)}")

    output_rows = []
    for row in source_rows:
        sample_id = int(row["sample_id"])
        output_rows.append(
            {
                **row,
                "taxonomy": sample_to_category[sample_id],
                "confidence": "preliminary",
                "review_notes": "AI-assisted blind feasibility audit; human check required.",
                "reviewed": "ai_preliminary",
            }
        )

    output_path = OUTPUT_DIR / "stage0_preliminary_audit.csv"
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=source_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    counts = Counter(sample_to_category.values())
    summary_lines = [
        "# Stage 0 preliminary taxonomy audit",
        "",
        "This is an AI-assisted blind review of the fixed 200-sample packet. "
        "It is evidence for taxonomy feasibility, not final human annotation. "
        "Keep `stage0_annotations.csv` for independent human verification.",
        "",
        "| Category | Definition | Count | Share |",
        "|---|---|---:|---:|",
    ]
    category_order = [
        "action_state", "object_entity", "attribute_quantity", "scene_spatial",
        "nonvisual_or_ambiguous",
    ]
    for category in category_order:
        count = counts[category]
        summary_lines.append(
            f"| {category} | {CATEGORY_DEFINITIONS[category]} | {count} | {count / 200:.1%} |"
        )
    summary_lines.extend(
        [
            "",
            "## Decision supported by this pass",
            "",
            "Use four visual contradiction subtypes—action/state, object/entity, "
            "attribute/quantity, and scene/spatial—for the 1k annotated test subset. "
            "Do not retain knowledge-required as a core subtype: the sampled residual "
            "is small and mostly non-visual or ambiguous rather than a coherent visual "
            "reasoning category. Flag such items as exclusions during final annotation.",
        ]
    )
    (OUTPUT_DIR / "stage0_preliminary_summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_path}")
    for category in category_order:
        print(f"{category}: {counts[category]} ({counts[category] / 200:.1%})")


if __name__ == "__main__":
    main()
