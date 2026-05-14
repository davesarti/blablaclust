# Conversational Clustering

An AI system that helps a user (the *oracle*) iteratively define how a text dataset should be grouped. There is no fixed ground truth — the oracle is the objective. The system proposes a clustering, explains its choices, and refines through dialogue.

## Setup

### 1. Clone and create environment

```bash
git clone <repo-url>
cd vibe-coders

conda create -n vibe-coders python=3.11
conda activate vibe-coders
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in at minimum:

```
ANTHROPIC_API_KEY=your_key_here
```

### 3. Seed the database

```bash
python seed_data.py
```

This loads 1500 Amazon Reviews (1200 train + 300 frozen eval) into `data/demo_database.db`.

### 4. Generate embeddings

```bash
python generate_embeddings.py
```

Uses `all-MiniLM-L6-v2` to embed all data points and store them in the DB.

### 5. Start the API

```bash
uvicorn backend.main:app --reload
```

API available at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

## Project structure

```
├── backend/
│   └── main.py              # FastAPI app
├── data/
│   ├── train.csv            # 1200 training records
│   └── frozen_eval.csv      # 300 evaluation records — do not modify
├── prompts/                 # One file per LLM prompt (versioned)
│   └── f_output.txt
├── src/
│   ├── engine/              # Core clustering logic (P3)
│   │   ├── f_output.py      # Generate cluster names and descriptions
│   │   ├── f_next_state.py  # Apply oracle feedback to update clustering
│   │   ├── f_next_best_step.py  # Decide: show / ask / stop
│   │   ├── f_uncertainty.py # Score data points by cluster ambiguity
│   │   └── f_eval.py        # Self-assess clustering quality
│   ├── harness.py           # LLM wrapper — all Claude calls go through here
│   ├── models.py            # SQLAlchemy ORM models
│   ├── schemas.py           # Pydantic schemas (shared contracts)
│   └── text_cleaning.py     # Text preprocessing before embedding
├── notes/                   # Sprint notes per person
├── .env.example             # Environment variable template
└── requirements.txt
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Model to use |
| `HARNESS_DRY_RUN` | `false` | Set to `true` to skip real API calls |
| `HARNESS_MAX_RETRIES` | `4` | Retries on transient API errors |
| `MAX_INPUT_TOKENS_PER_TURN` | `8000` | Context window budget per turn |
| `LLM_PROVIDER` | `claude` | `claude` or `openai` |

## Development notes

- **Do not modify** `data/frozen_eval.csv` — reserved for final evaluation
- All LLM calls go through `src/harness.py` — never import `anthropic` directly in engine code
- Prompts live in `prompts/` as `.txt` files — never hardcode them as f-strings in Python
- The DB layer is SQLite in dev; PostgreSQL in production
