"""Sample 1500 rows from the Amazon Review Polarity CSV already on disk.

Reads ``data/amazon_review_polarity_csv/train.csv``, filters out reviews
shorter than 50 chars, samples 1500 rows, then splits them into a 1200-row
training set and a 300-row frozen evaluation set.

Run from the project root:

    PYTHONPATH=. python -m src.dataset_processing.sample_amazon_dataset
"""

import pandas as pd

SOURCE_CSV = "data/amazon_review_polarity_csv/train.csv"
TRAIN_OUT = "data/train.csv"
FROZEN_OUT = "data/frozen_eval.csv"
SAMPLE_SIZE = 1500
TRAIN_SIZE = 1200
RANDOM_STATE = 42
MIN_TEXT_LEN = 50


def main() -> None:
    print("Caricando il dataset...")
    df = pd.read_csv(SOURCE_CSV, header=None, names=["label", "title", "text"])

    print(f"Record totali: {len(df)}")
    df = df[df["text"].str.len() > MIN_TEXT_LEN].dropna(subset=["text"])
    df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)

    train = df.sample(TRAIN_SIZE, random_state=RANDOM_STATE)
    frozen = df.drop(train.index)

    train.to_csv(TRAIN_OUT, index=False)
    frozen.to_csv(FROZEN_OUT, index=False)

    print(f"Fatto! Train: {len(train)} righe, Frozen: {len(frozen)} righe")
    print(train["text"].iloc[0])


if __name__ == "__main__":
    main()
