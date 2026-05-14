# Sprint 1 — P2 (Data Pipeline & Embeddings)

## What I built

- **Database setup**: aligned the existing SQLite schema with the current SQLAlchemy models — the `data_points` table was missing the `embedding` column (present in `src/models.py` but not in the legacy DB). Added it via a manual migration.
- **Seed script**: `seed_data.py` — loads the Amazon Review Polarity dataset into `data_points`. Idempotent: skips insertion if data already exists, supports `--force` to wipe and reload. Result: 1500 data points (1200 `amazon_reviews_train`, 300 `amazon_reviews_eval`), each storing `label`, `title`, and `text` as JSON.
- **Text cleaning**: `src/text_cleaning.py` — `clean_text(title, text)` runs before any embedding. Steps: NFC unicode normalization, CSV double-quote artifact removal (`""` → `"`), collapse of repeated characters (`!!!!!!` → `!!!`), whitespace normalization. Based on a data quality analysis run on both CSVs.
- **Embedding generation**: `generate_embeddings.py` — generates 384-dim dense vectors using `all-MiniLM-L6-v2` (Sentence Transformers) and saves them to the `embedding` field of each data point. Incremental: only processes rows where `embedding IS NULL`.
- **Verification**: `verify.py` — checks DB integrity (row counts, missing embeddings) and embedding correctness (vector dim, L2 norm, cosine similarity between stored and regenerated vectors).

## Results

| Check | Result |
|---|---|
| Total data points | 1500 |
| Missing embeddings | 0 |
| Vector dimension | 384 |
| L2 norm | 1.0 (normalized) |
| Stored vs regenerated cosine sim | 1.000000 |

## Challenges

I had never worked with SQLAlchemy or SQLite at this level before, so understanding the mismatch between the ORM models and the actual DB schema took some time. Deciding what to embed (title only, text only, or both) also required some thought — we settled on `title + " " + text` after a data quality analysis. Choosing the right embedding model was straightforward once we ruled out API-based solutions (no cost, no key needed, fast enough for 1500 records).

## Next steps

- Integrate `clean_text()` and the embedding pipeline with the rest of the system (P1 endpoints, P3 engine)
- Evaluate whether 384-dim vectors are sufficient for the clustering quality we need, or if a larger model is worth the tradeoff
- Consider adding a `dataset_split` field to `DataPoint` to explicitly track train vs eval without relying on `dataset_name` string matching
