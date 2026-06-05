# Conversational Clustering (BlaBlaClust)

An AI system that helps a user (the *oracle*) iteratively define how a text dataset should be grouped. There is no fixed ground truth — the oracle is the objective. The system proposes a clustering, explains its choices, and refines it through dialogue (merge / split / move / rename operations driven by the oracle's feedback).

## Quick start

```bash
# 1. Environment
conda create -n vibe-coders python=3.11
conda activate vibe-coders
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 2. Configure (defaults to the OpenRouter provider)
cp .env.example .env
#   edit .env and set the API key for your chosen LLM_PROVIDER

# 3. Backend — auto-seeds the demo dataset on first launch
PYTHONPATH=. python scripts/serve_ui.py

# 4. React frontend (separate terminal)
cd frontend && npm install && npm run dev
```

Then open:

- **React UI (primary):**  http://localhost:5173
- **Legacy HTML UI:**  http://localhost:8000/ui
- **API docs:**  http://localhost:8000/docs

On the **first** run the database is empty, so `serve_ui.py` loads `data/train.csv`
(~1,200 Amazon reviews) and generates sentence-transformer embeddings with
`all-MiniLM-L6-v2` (~60 s on CPU, model auto-downloads). Every later run sees a
populated DB and starts immediately.

## Other ways to run

```bash
# Backend only, with hot-reload (no auto-seed — seed manually first, see below)
uvicorn backend.main:app --reload

# Interactive terminal client against a running server
python scripts/cli.py                       # uses default dataset
python scripts/cli.py --session <id>        # resume an existing session
```

### Manual dataset seeding

`serve_ui.py` auto-seeds, so you normally don't need this. To seed by hand (e.g. when
running the bare API), use the helpers in `src/dataset_processing/`, or upload a dataset
at runtime via `POST /datasets/upload` (handles insert + embedding in one call).

### LLM-as-oracle (persona) sessions

Run a clustering session end-to-end with an LLM playing the oracle. Each persona is a
JSON file in `personas/` defining a goal, tone notes, dataset, `k_initial`, and an
optional model override. The runner creates a session per persona (visible in the UI
as `persona/<name>`), drives the turn loop via the live API, and writes a
`results.jsonl` + `summary.md` to `reports/<timestamp>-persona/`.

```bash
# 1. Start the API (in another terminal)
PYTHONPATH=. python scripts/serve_ui.py

# 2. Run one or more personas
python scripts/run_persona_eval.py \
    --personas personas/curious_explorer.json \
    --max-turns 12

# Globs work — run all shipped personas at once
python scripts/run_persona_eval.py --personas 'personas/*.json'

# Override the dataset for all personas (default: amazon_reviews)
python scripts/run_persona_eval.py --personas 'personas/*.json' --dataset imdb_train
```

Each run terminates per-persona on `oracle_satisfied`, `system_stop`, `max_turns`, or
on an error (`oracle_parse_error` / `api_error`). The end-of-session `/sessions/{sid}/eval`
metrics (A1–A3, B1–B4) are folded into each row automatically.

### Generalization stability eval

Answers the brief's *generalization* question — **once the oracle is happy, do new data
points that arrive keep the clustering coherent?** — on a *realistic* converged state.

The script (`scripts/run_generalization_stability_eval.py`) runs end-to-end in a single
terminal, **no server required** and **without touching `demo_database.db`**:

1. Spins up a **throwaway in-memory session** and clusters the `--base` split (GMM, with
   k-means fallback).
2. Drives a full **LLM-as-oracle conversation** (default persona `curious_explorer`)
   through the real engine, in-process, until the oracle is satisfied — so the converged
   state is oracle-shaped, with real turns and a real A1–B4 eval, not a bare clustering.
3. **Ingests the `--new` split** as a stream of new arrivals, assigning each against the
   *frozen* convergence geometry, and reports paired Δ + bootstrap 95% CIs on **A1**
   (silhouette) and **B2** (coherence), plus the **A4** Mahalanobis OOD rate.
4. **Closes and discards** the throwaway session (the in-memory DB vanishes on exit).

> Labels are out of scope — the script reads only `title,text` from every CSV. Generalization
> is consistency under growth, not accuracy against a hidden category.

```bash
# Real LLM judges/oracle require a non-dry-run provider (set in .env).
# 20 Newsgroups (k=6)
PYTHONPATH=. python scripts/run_generalization_stability_eval.py \
    --base data/20newsgroups_train.csv --new data/20newsgroups_frozen.csv --k 6

# Amazon reviews (k=4)
PYTHONPATH=. python scripts/run_generalization_stability_eval.py \
    --base data/train.csv --new data/frozen_eval.csv --k 4

# IMDB reviews (k=4)
PYTHONPATH=. python scripts/run_generalization_stability_eval.py \
    --base data/imdb_train.csv --new data/imdb_frozen.csv --k 4
```

Useful flags:

| Flag | Default | Purpose |
|---|---|---|
| `--k` | `6` | Number of clusters for the initial clustering. |
| `--persona` | `personas/curious_explorer.json` | LLM oracle that drives the throwaway conversation. |
| `--max-turns` | `12` | Hard cap on oracle turns. |
| `--batches N` | `1` | Split the new arrivals into N successive ingestions to trace a drift curve. |
| `--limit N` | none | Cap rows per split for a quick smoke run. |
| `--no-conversation` | off | Skip the oracle conversation and converge on the bare initial clustering (legacy A/B baseline). |

```bash
# Quick smoke run (small slice, short conversation)
PYTHONPATH=. python scripts/run_generalization_stability_eval.py \
    --base data/train.csv --new data/frozen_eval.csv --k 4 --limit 120 --max-turns 5

# Drift curve over 4 successive ingestion batches
PYTHONPATH=. python scripts/run_generalization_stability_eval.py \
    --base data/train.csv --new data/frozen_eval.csv --k 4 --batches 4
```

**Reading the report:** generalization *holds* when A1's paired Δ CI spans 0 (or is
positive), A4's OOD rate stays near its ~5% baseline, and B2 does not drop. A decline on
any of these means the new data is breaking the converged structure. With
`HARNESS_DRY_RUN=true` the LLM judges/oracle are mocked (B2 collapses to 0.0 and the
conversation is not meaningful) — use it only to smoke-test the plumbing.

## Project structure

```
├── backend/
│   ├── main.py                # FastAPI app + SQLite engine/session setup
│   ├── session_state.py       # Builds ChatSessionState from DB rows
│   ├── eval_cache.py          # Content-hash cache for /sessions/{id}/eval
│   └── routers/
│       ├── datasets.py        # upload / list datasets
│       ├── sessions.py        # create / read sessions, initial clustering
│       ├── turns.py           # the main oracle-interaction loop endpoint
│       ├── clusters.py        # read clusters and their points
│       └── umap.py            # 2-D UMAP projection for the analytics panel
├── frontend/                  # React UI (Vite + TypeScript) — primary interface
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── WelcomePage.tsx      # dataset / session selector
│   │   │   ├── WorkspacePage.tsx    # main chat + cluster workspace
│   │   │   ├── ChatPanel.tsx
│   │   │   ├── ClusterCard.tsx
│   │   │   ├── AnalyticsPanel.tsx   # UMAP + turn-history visualisation
│   │   │   └── modals/
│   │   ├── store/AppContext.tsx     # app-wide state (context)
│   │   └── api/client.ts           # typed API client
│   └── package.json
├── scripts/
│   ├── serve_ui.py                       # auto-seed + serve backend (recommended entrypoint)
│   ├── cli.py                            # interactive terminal client
│   ├── run_persona_eval.py               # LLM-as-oracle evaluation runner (persona files)
│   ├── run_scenario_eval.py              # scripted-oracle evaluation runner (scenario files)
│   ├── run_baseline_eval.py              # non-interactive baseline clustering eval
│   ├── run_generalization_stability_eval.py  # held-out generalization eval
│   ├── compare_embeddings.py             # embedding model comparison report
│   ├── verify_loop_20ng.py               # smoke-test the 20 Newsgroups loop
│   └── migrate_datapoint_text.py         # one-off DB migration utility
├── ui/
│   ├── index.html             # legacy single-file HTML UI (served at /ui)
│   └── DESIGN.md
├── data/
│   ├── train.csv                  # 1 200 Amazon reviews (demo dataset)
│   ├── frozen_eval.csv            # 300 Amazon eval records — do not modify
│   ├── 20newsgroups_train.csv     # 20 Newsgroups training split
│   ├── 20newsgroups_frozen.csv    # 20 Newsgroups frozen eval split
│   ├── imdb_train.csv             # IMDB training split
│   ├── imdb_frozen.csv            # IMDB frozen eval split
│   └── demo_database.db           # SQLite DB (created/seeded on first run)
├── scenarios/                 # Scripted oracle scenario files (JSON)
│   ├── stable_oracle.json
│   ├── contradictory_oracle.json
│   ├── high_load_oracle.json
│   ├── sentiment_split.json
│   ├── topic_merge.json
│   └── …
├── personas/                  # LLM-oracle persona definitions (JSON)
├── prompts/                   # One .txt file per LLM prompt (versioned)
│   ├── f_output.txt           # Intent classification: oracle text → operations
│   ├── f_next_best_step.txt   # Planner: decide show / ask / stop
│   ├── f_boundary_repair.txt  # Post-op boundary point validation
│   ├── f_update_preferences.txt
│   ├── f_eval.txt / f_eval_coherence.txt / f_eval_compliance.txt …
│   ├── semantic_reembed.txt   # Axis scoring (directional, pole-anchored)
│   ├── semantic_axis_poles.txt # LLM-generated axis pole examples
│   ├── cluster_naming.txt     # LLM cluster name + description generation
│   ├── parse_clustering_intent.txt
│   ├── dataset_description.txt
│   └── llm_oracle.txt         # LLM-as-oracle persona driver
├── src/
│   ├── engine/                # Core clustering logic
│   │   ├── turn_builder.py          # In-memory staging; single DB commit per turn
│   │   ├── f_output.py              # Executor: LLM → structured operations
│   │   ├── f_apply_operations.py    # Dispatch operations to cluster_operations
│   │   ├── f_next_best_step.py      # Planner: show / ask / stop
│   │   ├── f_uncertainty.py         # Cluster-level soft-assignment uncertainty
│   │   ├── f_parse_clustering_intent.py  # Free text → k + clustering axis
│   │   ├── f_semantic_reembed.py    # Hybrid axis-weighted embedding pipeline
│   │   ├── f_boundary_repair.py     # Post-op LLM-guided boundary correction
│   │   ├── f_cognitive_load.py      # Estimate conversation cognitive load
│   │   ├── f_update_preferences.py  # Rolling oracle preference summary
│   │   ├── f_eval.py                # Self-assess clustering quality (A/B metrics)
│   │   ├── semantic_clustering.py   # Global semantic re-clustering along an axis
│   │   ├── generalization.py        # Nearest-centroid assignment for held-out data
│   │   ├── initial_clustering.py    # GMM (k-means fallback) + soft assignments
│   │   ├── cluster_operations.py    # merge / split / move / rename / cluster_reembed
│   │   ├── cluster_naming.py        # LLM-generated cluster names + descriptions
│   │   └── cognitive_load_caps.py   # Turn / token / cluster thresholds (A3)
│   ├── eval/                  # LLM-oracle eval harness
│   │   ├── llm_oracle.py      # Drives sessions as a simulated user
│   │   ├── oracle_view.py     # Builds per-turn cluster panel for the oracle LLM
│   │   ├── persona.py         # Persona file loader + validation
│   │   └── eval_report.py     # Shared report writer (results.jsonl + summary.md)
│   ├── viz/
│   │   └── umap_projection.py # 2-D UMAP reduction (cached per dataset)
│   ├── harness/               # LLM provider abstraction
│   │   ├── harness.py         # Public API + provider dispatch
│   │   ├── harness_claude.py  # Anthropic Claude provider
│   │   ├── harness_openai.py  # OpenAI-compatible provider (OpenAI / Groq)
│   │   └── harness_openrouter.py  # OpenRouter provider
│   ├── dataset_processing/    # CSV ingest, embeddings, text cleaning
│   ├── logger.py              # Logging + structured LLM call log
│   ├── models.py              # SQLAlchemy ORM models
│   └── schemas.py             # Pydantic schemas (shared contracts)
├── docs/                      # Technical reports and literature reviews — see docs/README.md
│   ├── data-model.md                      # Database schema and ORM models
│   ├── quality_specs.md                   # Evaluation metric definitions (A1–A3, B1–B4)
│   ├── api-interface.md                   # Full HTTP/JSON API reference (all endpoints + schemas)
│   ├── semantic-reembed-report.md         # Semantic reembedding pipeline implementation
│   ├── semantic-reembed-diagrams.ipynb    # Slide diagrams: pipeline flowchart, embedding space, hybrid matrix
│   ├── gmm-vs-kmeans-report.md            # GMM vs k-means comparison
│   ├── generalization-stability-report.md # Online generalization across 6 user sessions (A1/A4/B2 + forest plots)
│   ├── final_experimental_report.md       # End-to-end results: B2 vs baseline, persona quality, real-user overlay
│   ├── related-work.md                    # Literature review — 5 related systems
│   ├── related-work-radar.ipynb           # Radar chart — design space comparison
│   ├── evaluation_literature_report.md    # Literature grounding for evaluation metrics
│   ├── embedding-model-comparison.md      # MiniLM vs BGE-base benchmark
│   └── instruct-reembed-comparison.md     # Test plan — instruct-tuned reembedding (GPU required)
├── notes/                     # Sprint notes per person
├── AGENTS.md                  # Planner/Executor architecture
├── .env.example               # Environment variable template
└── requirements.txt
```

## Environment variables

The provider is selected by `LLM_PROVIDER`; only the keys for the chosen provider are required.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openrouter` | `claude` \| `openai` \| `openrouter` |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=claude` |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Claude model |
| `OPENAI_API_KEY` | — | Required for `openai` (also used for Groq via `OPENAI_BASE_URL`) |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI-compatible model |
| `OPENAI_BASE_URL` | OpenAI default | Override to point at Groq, etc. |
| `OPENROUTER_API_KEY` | — | Required when `LLM_PROVIDER=openrouter` |
| `OPENROUTER_MODEL` | see `.env.example` | OpenRouter model slug |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter endpoint |
| `HARNESS_DRY_RUN` | `false` | `true` skips real API calls |
| `HARNESS_MAX_RETRIES` | `4` | Retries on transient API errors |
| `MAX_INPUT_TOKENS_PER_TURN` | `8000` | Context budget per turn (Claude only) |

## Development notes

- **Do not modify** `data/frozen_eval.csv` — reserved for final evaluation.
- All LLM calls go through `src/harness/` — never import a provider SDK (`anthropic`, `openai`) directly in engine code.
- Prompts live in `prompts/` as `.txt` files — never hardcode them as f-strings in Python.
- Engine functions never call `db.commit()` — the caller (the API router) owns the transaction so a turn that applies several operations stays atomic.
- Engine errors propagate (no silent skipping); the API surfaces them as HTTP 422 so failures are visible to the oracle.
- The DB layer is SQLite in dev; PostgreSQL in production.
