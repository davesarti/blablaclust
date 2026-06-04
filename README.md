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
```

Each run terminates per-persona on `oracle_satisfied`, `system_stop`, `max_turns`, or
on an error (`oracle_parse_error` / `api_error`). The end-of-session `/sessions/{sid}/eval`
metrics (A1–A3, B1–B4) are folded into each row automatically.

**A3 cognitive-load caps** (`src/engine/cognitive_load_caps.py`) — the three thresholds
(20 turns, 16 000 tokens, 25 clusters) are arbitrary engineering estimates of where LLM
output quality is expected to degrade, not empirically validated. They are intentionally
centralised in one file to make it easy to run a sweep: vary the caps, re-run the eval
suite, and measure the effect on A2 (turns to convergence) and B-metrics (clustering
quality) to find better-calibrated values.

## Project structure

```
├── backend/
│   ├── main.py                # FastAPI app + SQLite engine/session setup
│   ├── session_state.py       # Builds ChatSessionState from DB rows
│   └── routers/
│       ├── datasets.py        # upload / list datasets
│       ├── sessions.py        # create / read sessions, initial clustering
│       ├── turns.py           # the main oracle-interaction loop endpoint
│       └── clusters.py        # read clusters and their points
├── frontend/                  # React UI (Vite + TypeScript) — primary interface
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/        # UI components
│   │   ├── store/             # App state (context + reducers)
│   │   └── api/               # API client helpers
│   └── package.json
├── scripts/
│   ├── serve_ui.py            # auto-seed + serve backend (recommended entrypoint)
│   └── cli.py                 # interactive terminal client
├── ui/
│   ├── index.html             # legacy single-file HTML UI (served at /ui)
│   └── DESIGN.md
├── data/
│   ├── train.csv              # 1200 training records (demo dataset)
│   ├── frozen_eval.csv        # 300 evaluation records — do not modify
│   └── demo_database.db       # SQLite DB (created/seeded on first run)
├── prompts/                   # One .txt file per LLM prompt (versioned)
│   ├── f_output.txt
│   ├── f_next_best_step.txt
│   ├── f_eval.txt
│   ├── cluster_naming.txt
│   └── parse_clustering_intent.txt
├── src/
│   ├── engine/                # Core clustering logic (P3)
│   │   ├── f_output.py              # Executor: LLM → structured operations + usage
│   │   ├── f_apply_operations.py    # Dispatch operations to cluster_operations
│   │   ├── f_next_best_step.py      # Decide: show / ask / stop
│   │   ├── f_uncertainty.py         # Score data points by cluster ambiguity
│   │   ├── f_parse_clustering_intent.py  # Free text → k + clustering axis
│   │   ├── f_eval.py                # Self-assess clustering quality
│   │   ├── initial_clustering.py    # k-means + soft assignments
│   │   ├── cluster_operations.py    # merge / split / move / rename executors
│   │   └── cluster_naming.py        # LLM-generated cluster names + descriptions
│   ├── harness.py             # LLM wrapper (Claude) + provider dispatch
│   ├── harness_openai.py      # OpenAI-compatible provider (OpenAI / Groq)
│   ├── harness_openrouter.py  # OpenRouter provider
│   ├── logger.py              # Logging + structured LLM call log
│   ├── models.py              # SQLAlchemy ORM models
│   ├── schemas.py             # Pydantic schemas (shared contracts)
│   └── dataset_processing/    # CSV ingest, embeddings, text cleaning
├── notes/                     # Sprint notes per person
├── AGENTS.md                  # Planner/Executor architecture (grading component)
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
- All LLM calls go through `src/harness.py` — never import a provider SDK (`anthropic`, `openai`) directly in engine code.
- Prompts live in `prompts/` as `.txt` files — never hardcode them as f-strings in Python.
- Engine functions never call `db.commit()` — the caller (the API router) owns the transaction so a turn that applies several operations stays atomic.
- Engine errors propagate (no silent skipping); the API surfaces them as HTTP 422 so failures are visible to the oracle.
- The DB layer is SQLite in dev; PostgreSQL in production.
