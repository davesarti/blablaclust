# CLAUDE.md — BlaBlaClust

Conversational clustering system for the UniTN DLSAS course (team: vibe-coders, 5 people).
The oracle (human or LLM persona) refines k-means/GMM clusters through natural language.
Matteo owns P3 (Core Engine). Other teammates own P1 (ingestion), P2 (cluster ops), P4 (eval), P5 (frontend).

## How to run

```bash
# Always use the vibe-coders conda env — base lacks sentence-transformers
conda activate vibe-coders

# Backend (auto-seeds amazon_reviews dataset on first run)
PYTHONPATH=. python scripts/serve_ui.py

# Frontend (separate terminal) — React + TypeScript + Vite
cd frontend && npm run dev
# Open http://localhost:5173  (Vite proxies /sessions /datasets /clusters /turns /umap → :8000)
```

Node must be ≥ 20 for Vite v8. On WSL2, `host: true` is set in `frontend/vite.config.ts` so the Windows browser can reach the dev server.

## Key architecture

- **Turn loop** (`backend/routers/turns.py`): oracle text → `f_output` (LLM parses ops) → `f_apply_operations` → `f_boundary_repair` → `f_next_best_step` (show/ask/stop) → snapshot committed
- **Clustering**: GMM (`covariance_type='diag'`, `USE_GMM=True` in `src/engine/initial_clustering.py`) with k-means fallback. Native posteriors replace the old softmax hack.
- **Soft assignments**: stored per (data_point, cluster, turn_number) — full snapshot each turn.
- **Snapshot model**: every merge/split/move/rename writes a new row at the current turn; dissolved clusters get `dissolved_at_turn` set.
- **LLM calls**: always through `src/harness.py`. Never import `anthropic`/`openai` directly in engine code. Current provider: OpenRouter (`google/gemini-3.1-flash-lite`).
- **Prompts**: live in `prompts/*.txt`, rendered via `render_prompt()`. Never hardcode as f-strings.
- **Transactions**: engine functions never call `db.commit()`. The router owns the transaction.

## DB — known schema drift pattern

Teammates add ORM columns without migrating the SQLite file. When the server crashes on startup with `OperationalError: no such column`, run:
```bash
sqlite3 data/demo_database.db ".schema <table>"
```
Then `ALTER TABLE <table> ADD COLUMN <col> <type> DEFAULT <val>`.

Current DB has legacy columns `data_points.data` (nullable JSON) and `data_points.dataset_name` / `sessions.dataset_name` that the ORM no longer maps — harmless, SQLAlchemy ignores them.

## Evaluation

```bash
# Needs server running
PYTHONPATH=. python scripts/run_scenario_eval.py \
  --scenarios scenarios/stable_oracle.json scenarios/sentiment_split.json \
  scenarios/contradictory_oracle.json scenarios/topic_merge.json \
  --out reports/<name>
```

Metrics: A1 (silhouette), A2 (turns/termination), A3 (cognitive load), B1 (overall quality), B2 (coherence), B3 (compliance), B4 (contradiction). `high_load_oracle` often times out due to LLM rate limits — exclude it from quick runs.

## Common pitfalls

- Running scripts from `base` conda env → `ModuleNotFoundError: sentence_transformers`. Always use `vibe-coders`.
- Stash conflicts on `f_next_best_step.py`: always take upstream (threshold=5, CognitiveLoad object).
- LLM mistyping UUIDs in operations: handled by `_normalize_cluster_ids` (difflib fuzzy match).
- `data_points.data NOT NULL` error on upload: run the nullable migration (already done on current DB).
- GMM convergence failure on tiny subsets (< k points): auto-falls back to k-means, logged as WARNING.

## Recent work (as of 2026-06-04)

- GMM replaces k-means as default (`USE_GMM=True`) — better soft-assignment posteriors, same end-to-end quality
- `f_boundary_repair`: post-merge/split LLM validation of uncertain boundary points (10 per affected cluster)
- Eval framework: `scripts/run_scenario_eval.py` + `src/eval/` + `scenarios/*.json`
- Frontend migrated to React + TypeScript + Vite (primary UI at :5173)
- Per-turn LLM cost accumulator in `src/harness.py`
- Cluster number badges in UI (1, 2, 3…)
- UUID fuzzy repair for LLM transcription errors
- Inline rename on merge/split ops
- Percentage-based representative sampling (15% cap 30 for naming, 10% for eval)
