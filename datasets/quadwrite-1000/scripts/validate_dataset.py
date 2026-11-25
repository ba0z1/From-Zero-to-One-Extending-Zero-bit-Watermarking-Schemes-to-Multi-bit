#!/usr/bin/env python3
"""Check prompt identity, construction, and consistency across all three formats."""
from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter

from build_dataset import FIELDS, ROOT, build_outputs, build_records, load_json


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def validate():
    rows = [json.loads(line) for line in (ROOT / "data/prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    original = load_json(ROOT / "construction/original_100.json")
    design = load_json(ROOT / "construction/design.json")
    assert len(rows) == 1000, "Expected 1,000 rows"
    assert len({r["task_id"] for r in rows}) == 1000, "Duplicate task IDs"
    assert len({r["prompt"] for r in rows}) == 1000, "Duplicate prompt strings"
    assert len({normalize(r["prompt"]) for r in rows}) == 1000, "Normalized duplicate prompts"
    assert rows == load_json(ROOT / "data/prompts.json"), "JSON and JSONL differ"
    assert rows == build_records(), "Dataset differs from its declared construction"
    category_counts = Counter(r["category"] for r in rows)
    assert category_counts == dict.fromkeys(design["categories"], 250), category_counts
    subset_counts = Counter(r["subset"] for r in rows)
    assert subset_counts == {"original_100": 100, "extension_900": 900}, subset_counts
    by_id = {r["task_id"]: r for r in rows}
    for old in original:
        got = by_id[old["task_id"]]
        for key in ["task_id", "category", "category_index", "prompt"]:
            assert got[key] == old[key], (old["task_id"], key)
        assert got["subset"] == "original_100"
    topic_groups = Counter(r["topic_id"] for r in rows if r["subset"] == "extension_900")
    assert len(topic_groups) == 180 and set(topic_groups.values()) == {5}
    for r in rows:
        assert list(r) == FIELDS, r["task_id"]
        assert r["language"] == "en" and r["target_words"] == 1800
        assert r["prompt"].endswith(design["suffix"])
        assert r["prompt"].startswith(design["prefixes"][r["category"]])
        assert r["task_id"] == f'{r["category"]}_{r["category_index"]:02d}'
        assert (r["angle_id"] is None) == (r["subset"] == "original_100")
        assert " in English about " in r["prompt"], r["task_id"]
        assert "Aim for approximately 1800 words." in r["prompt"], r["task_id"]
        assert all(not ch.isalpha() or "LATIN" in unicodedata.name(ch, "")
                   for ch in r["prompt"]), f'Non-Latin letter in {r["task_id"]}'
    with (ROOT / "data/prompts.csv").open(encoding="utf-8", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(rows)
    for plain, table in zip(rows, csv_rows):
        assert table == {k: "" if v is None else str(v) for k, v in plain.items()}
    for name, expected in build_outputs().items():
        assert (ROOT / name).read_text(encoding="utf-8") == expected, name
    return dict(status="passed", total_prompts=1000, unique_ids=1000,
        unique_prompt_texts=1000, normalized_unique_texts=1000,
        category_counts=dict(category_counts), subset_counts=dict(subset_counts),
        original_100_preserved=True, extension_topic_groups=180, angles_per_topic=5,
        explicit_english_instruction_count=1000,
        approximate_1800_word_request_count=1000, non_latin_letter_count=0,
        formats_consistent=True, construction_reproducible=True)


if __name__ == "__main__":
    print(json.dumps(validate(), ensure_ascii=False, indent=2))
