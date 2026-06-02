"""Build a balanced IMDB movie-review dataset for the clustering app.

Reads the full 50 000-row IMDB Dataset CSV, strips HTML tags (``<br />``) and
other noise, balances the two sentiment classes (positive / negative), and
writes two CSVs in the pipeline's standard ``label,title,text`` shape:

    data/imdb_train.csv    (1 200 rows, balanced)
    data/imdb_frozen.csv   (  300 rows, balanced, held-out)

Design notes
------------
- ``title`` is intentionally left EMPTY (consistent with 20 Newsgroups). The
  review text alone carries the signal; leaving the title empty avoids any
  risk of label leakage.
- ``label`` mirrors the Amazon convention: **2 = positive, 1 = negative**.
  This makes the column meaning consistent across datasets and lets shared
  evaluation tooling handle both without special-casing.
- HTML tags (``<br />``, ``<br>``, etc.) are stripped before storing — they
  appear frequently in IMDB HTML-exported reviews and add no embedding signal.
- Short reviews (< 50 chars after stripping) are dropped for quality.
- Balanced sampling: equal number of positive and negative rows in both splits.

Run from the project root:

    PYTHONPATH=. python -m src.dataset_processing.sample_imdb_dataset \\
        --source "/path/to/IMDB Dataset.csv"

The default source path points to the Downloads folder (adjust as needed).
"""

from __future__ import annotations

import argparse
import csv
import re
import random
from pathlib import Path

SOURCE_CSV_DEFAULT = Path.home() / "Downloads" / "IMDB Dataset.csv"

TRAIN_OUT  = "data/imdb_train.csv"
FROZEN_OUT = "data/imdb_frozen.csv"

SAMPLE_SIZE   = 1500   # total rows before train/frozen split
TRAIN_SIZE    = 1200
RANDOM_STATE  = 42
MIN_TEXT_LEN  = 50

# Label convention — same as Amazon so shared eval tooling handles both.
LABEL_MAP = {"positive": 2, "negative": 1}


def _strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _load_rows(source_csv: Path) -> dict[int, list[dict]]:
    """Read the source CSV and return {label: [row, ...]} with clean text."""
    by_label: dict[int, list[dict]] = {1: [], 2: []}
    with open(source_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_sentiment = (row.get("sentiment") or "").strip().lower()
            label = LABEL_MAP.get(raw_sentiment)
            if label is None:
                continue
            text = _strip_html(row.get("review") or "")
            if len(text) < MIN_TEXT_LEN:
                continue
            by_label[label].append({"label": label, "title": "", "text": text})
    return by_label


def _balanced_sample(
    by_label: dict[int, list[dict]],
    total: int,
    rng: random.Random,
) -> list[dict]:
    """Sample ``total // n_classes`` rows per class, shuffle, return combined."""
    n_classes = len(by_label)
    per_class = total // n_classes
    sampled: list[dict] = []
    for rows in by_label.values():
        if len(rows) < per_class:
            raise ValueError(
                f"Not enough rows ({len(rows)}) to sample {per_class} per class."
            )
        sampled.extend(rng.sample(rows, per_class))
    rng.shuffle(sampled)
    return sampled


def _write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "title", "text"])
        writer.writeheader()
        writer.writerows(rows)


def main(source_csv: Path = SOURCE_CSV_DEFAULT) -> None:
    print(f"Loading IMDB dataset from: {source_csv}")
    by_label = _load_rows(source_csv)
    for lbl, rows in by_label.items():
        sentiment = {2: "positive", 1: "negative"}[lbl]
        print(f"  {sentiment} (label={lbl}): {len(rows)} usable rows")

    rng = random.Random(RANDOM_STATE)
    rows = _balanced_sample(by_label, SAMPLE_SIZE, rng)
    print(f"Balanced sample: {len(rows)} rows total "
          f"({SAMPLE_SIZE // 2} per class)")

    train = rows[:TRAIN_SIZE]
    frozen = rows[TRAIN_SIZE:]

    _write_csv(train, TRAIN_OUT)
    _write_csv(frozen, FROZEN_OUT)

    train_pos = sum(1 for r in train  if r["label"] == 2)
    frz_pos   = sum(1 for r in frozen if r["label"] == 2)
    print(f"Done!")
    print(f"  train  → {TRAIN_OUT}:  {len(train)} rows  "
          f"(pos={train_pos}, neg={len(train)-train_pos})")
    print(f"  frozen → {FROZEN_OUT}: {len(frozen)} rows  "
          f"(pos={frz_pos}, neg={len(frozen)-frz_pos})")
    print(f"Label convention: 2=positive, 1=negative  (same as Amazon)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        type=Path,
        default=SOURCE_CSV_DEFAULT,
        help=f"Path to the IMDB Dataset CSV (default: {SOURCE_CSV_DEFAULT})",
    )
    args = ap.parse_args()
    main(source_csv=args.source)
