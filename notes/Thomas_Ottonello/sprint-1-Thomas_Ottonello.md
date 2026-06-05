# Sprint 1 — P2 (Data & Embeddings) - Thomas Ottonello

## What I did this sprint

Built the data pipeline from raw CSV to embedded, queryable data points.

- **DB schema alignment** — the legacy SQLite DB was missing the `embedding`
  column present in `src/models.py`; added it via a manual migration.
- **Seed script** (`seed_data.py`) — loads the Amazon Review Polarity dataset
  into `data_points`. Idempotent (skips if data exists, `--force` to wipe and
  reload). Result: 1500 points (1200 `amazon_reviews_train` + 300
  `amazon_reviews_eval`), each storing `label`, `title`, `text` as JSON.
- **Text cleaning** (`src/text_cleaning.py`) — `clean_text(title, text)` runs
  before any embedding: NFC unicode normalization, CSV double-quote artifact
  removal (`""` → `"`), repeated-char collapse (`!!!!!!` → `!!!`), whitespace
  normalization. Driven by a data-quality analysis of both CSVs.
- **Embedding generation** (`generate_embeddings.py`) — 384-dim vectors via
  `all-MiniLM-L6-v2` (Sentence Transformers), written to each point's
  `embedding`. Incremental: only rows where `embedding IS NULL`.
- **Verification** (`verify.py`) — checks DB integrity and embedding
  correctness. All green: 1500 points, **0 missing embeddings**, dim **384**,
  L2 norm **1.0**, stored-vs-regenerated cosine similarity **1.000000**.

We embed `title + " " + text` (settled after the data-quality analysis); the
model was chosen to be local (no cost, no key, fast enough for 1500 records).

## What blocked me

- First time using SQLAlchemy / SQLite at this level, so diagnosing the
  mismatch between the ORM models and the actual DB schema took time.
- No hard external blockers — the layer is self-contained.

## What I'm doing next

- Integrate `clean_text()` + the embedding pipeline with the rest of the system
  (P1 endpoints, P3 engine).
- Evaluate whether 384-dim vectors are enough for the clustering quality we
  need, or if a larger model is worth the tradeoff.
- Consider a `dataset_split` field on `DataPoint` to track train vs eval
  explicitly instead of relying on `dataset_name` string matching.

## Commits

Not individually recorded in this note (early sprint; work landed as the
initial data-pipeline drop).

## Issues

- **Opened & resolved by me:** #2 (dataset), #3 (set up SQLite DB + seed Amazon
  data points), #4 (add embedding generation for Amazon data points).
