#!/usr/bin/env python3
"""Rebuild the prompt files and metadata from the dataset's construction files."""
from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["task_id", "category", "category_index", "prompt", "language",
          "target_words", "subset", "topic_id", "angle_id"]


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def build_records():
    original = load_json(ROOT / "construction/original_100.json")
    design = load_json(ROOT / "construction/design.json")
    categories = design["categories"]
    original_by_id = {r["task_id"]: r for r in original}
    assert len(original) == len(original_by_id) == 100
    assert Counter(r["category"] for r in original) == dict.fromkeys(categories, 25)
    records = []
    for category in categories:
        base = sorted((r for r in original if r["category"] == category),
                      key=lambda r: r["category_index"])
        assert [r["category_index"] for r in base] == list(range(1, 26))
        for row in base:
            records.append(dict(
                task_id=row["task_id"], category=category,
                category_index=row["category_index"], prompt=row["prompt"],
                language="en", target_words=design["target_words"],
                subset="original_100",
                topic_id=f'{category}_original_{row["category_index"]:02d}',
                angle_id=None))
        topics = design["topics"][category]
        angles = design["angles"][category]
        assert len(topics) == 45 and len(angles) == 5
        for topic_index, topic in enumerate(topics, 1):
            for angle_index, angle in enumerate(angles, 1):
                index = 25 + (topic_index - 1) * 5 + angle_index
                prompt = design["prefixes"][category] + topic + ", " + angle + design["suffix"]
                records.append(dict(
                    task_id=f"{category}_{index:02d}", category=category,
                    category_index=index, prompt=prompt, language="en",
                    target_words=design["target_words"], subset="extension_900",
                    topic_id=f"{category}_topic_{topic_index:02d}",
                    angle_id=f"{category}_angle_{angle_index}"))
    return records


def build_outputs():
    records = build_records()
    assert len(records) == 1000
    assert len({r["task_id"] for r in records}) == 1000
    assert len({r["prompt"] for r in records}) == 1000
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    lengths = [len(r["prompt"].split()) for r in records]
    metadata = dict(
        name="QuadWrite-1000", version="1.0.0",
        description="English long-form writing prompts across four genres.",
        language="en", dataset_type="writing_instructions",
        license="CC-BY-4.0",
        total_prompts=len(records),
        category_counts=dict(Counter(r["category"] for r in records)),
        subset_counts=dict(Counter(r["subset"] for r in records)),
        requested_output_words=1800,
        construction=dict(original_prompts=100, extension_prompts=900,
            extension_topics_per_category=45, angles_per_category=5,
            extension_topic_groups=180, total_construction_groups=280),
        prompt_length_words=dict(minimum=min(lengths), maximum=max(lengths),
                                 mean=round(mean(lengths), 3), median=median(lengths),
                                 counting_method="Whitespace-separated words."),
        canonical_file="data/prompts.jsonl", alternate_formats=["data/prompts.json", "data/prompts.csv"],
        fields=FIELDS)
    return {
        "data/prompts.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        "data/prompts.json": json_text(records),
        "data/prompts.csv": csv_buffer.getvalue(),
        "metadata.json": json_text(metadata),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check reproducibility without writing files.")
    args = parser.parse_args()
    outputs = build_outputs()
    if args.check:
        mismatches = [name for name, value in outputs.items()
                      if not (ROOT / name).exists() or (ROOT / name).read_text(encoding="utf-8") != value]
        if mismatches:
            raise SystemExit("Files differ from the declared construction: " + ", ".join(mismatches))
        print("Reproducibility check passed for all four generated files.")
    else:
        for name, value in outputs.items():
            path = ROOT / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8")
        print("Built 1,000 prompts in JSONL, JSON and CSV, plus metadata.json.")


if __name__ == "__main__":
    main()
