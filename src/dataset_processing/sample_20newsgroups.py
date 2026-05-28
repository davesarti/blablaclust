"""Build a curated 20 Newsgroups dataset for the clustering app.

Fetches six well-separated newsgroups via scikit-learn, strips the email
headers / footers / quoted replies so only the post body remains, and writes
two CSVs in the pipeline's expected ``label,title,text`` shape:

    data/20newsgroups_train.csv   (~1200 rows)
    data/20newsgroups_frozen.csv  (~300 rows, held-out for generalization)

Design notes
------------
- ``title`` is intentionally left EMPTY. ``clean_text`` concatenates title+text
  for the embedding, so putting the category name in the title would leak the
  label into the vector and make clustering trivially easy — not a fair test
  of whether the system adapts to a new dataset. The post body alone carries
  the signal.
- ``label`` is the category index. It is stored for reference/evaluation only;
  it is NOT part of the embedded text.
- Short bodies (< 50 chars after stripping) are dropped — once headers/footers/
  quotes are removed, some posts collapse to almost nothing.

Run from the project root:

    PYTHONPATH=. python -m src.dataset_processing.sample_20newsgroups
"""

import csv

from sklearn.datasets import fetch_20newsgroups

CATEGORIES = [
    "sci.space",
    "rec.sport.baseball",
    "talk.politics.guns",
    "comp.graphics",
    "sci.med",
    "rec.autos",
]

TRAIN_OUT = "data/20newsgroups_train.csv"
FROZEN_OUT = "data/20newsgroups_frozen.csv"
SAMPLE_SIZE = 1500          # total kept rows before the train/frozen split
TRAIN_SIZE = 1200
RANDOM_STATE = 42
MIN_TEXT_LEN = 50


def _load_rows() -> list[dict]:
    """Fetch both subsets, strip noise, return [{label, title, text}, ...]."""
    bunch = fetch_20newsgroups(
        subset="all",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        random_state=RANDOM_STATE,
    )
    rows = []
    for body, target in zip(bunch.data, bunch.target):
        text = " ".join(body.split())          # collapse whitespace early
        if len(text) < MIN_TEXT_LEN:
            continue
        rows.append({"label": int(target), "title": "", "text": text})
    return rows


def main() -> None:
    import random

    print("Fetching 20 Newsgroups (6 categories)…")
    rows = _load_rows()
    print(f"Usable posts after stripping/filtering: {len(rows)}")

    rng = random.Random(RANDOM_STATE)
    rng.shuffle(rows)
    rows = rows[:SAMPLE_SIZE]

    train = rows[:TRAIN_SIZE]
    frozen = rows[TRAIN_SIZE:]

    for path, subset in ((TRAIN_OUT, train), (FROZEN_OUT, frozen)):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["label", "title", "text"])
            writer.writeheader()
            writer.writerows(subset)

    print(f"Done! Train: {len(train)} rows → {TRAIN_OUT}")
    print(f"       Frozen: {len(frozen)} rows → {FROZEN_OUT}")
    print(f"Categories (label index → name): "
          + ", ".join(f"{i}={c}" for i, c in enumerate(sorted(CATEGORIES))))


if __name__ == "__main__":
    main()
