# AGENTS.md

## What this project does
This system helps a user (the "oracle") iteratively define how a text dataset
should be grouped into clusters. The oracle gives feedback in natural language
("this group is too broad", "merge these two"), and the system updates the
clustering accordingly. There is no fixed ground truth — the oracle's judgment
is the objective.

## Agentic Pattern: Planner / Executor

This system uses a Planner/Executor pattern. The agent does not respond to
oracle feedback in a single step — instead it separates the decision of *what
to do next* from the act of *doing it*. This makes the system easier to debug
and extend, since the decision logic and the execution logic never mix.

## Function map

| Role | Function | What it does |
|---|---|---|
| Sensor | `f_uncertainty` | Reads the DB and scores each data point by how ambiguous its cluster assignment is — feeds signal into the Planner |
| Planner | `f_next_best_step` | Reads state + uncertainty scores, decides the next action: show / ask / stop |
| Executor | `f_output` + `f_apply_operations` | Calls Claude to turn oracle feedback into structured operations, then applies them to the DB |
| Judge | `f_eval` | Self-assesses the quality of the current clustering at the end of a session |

## Architecture

```
Oracle natural language input
           │
           ▼
    f_output ──▶ Claude (via harness.py)
           │                  │
           │        operations (merge / split / move / rename)
           │◀─────────────────┘
           │
           ▼
    f_apply_operations ──writes──▶ clusters + soft_assignments (DB)
           │
           ▼
    f_uncertainty ──reads──▶ SoftAssignment table (DB)
           │
           ▼
    f_next_best_step ──decides──▶ show / ask / stop
           │
           ▼
       f_eval ──assesses──▶ SystemTurn.state_snapshot
```

## How to run

```bash
conda activate vibe-coders
uvicorn backend.main:app --reload
```

## Key rules
- All LLM calls go through `src/harness.py` — never import `anthropic` directly
- Prompts live in `prompts/` as `.txt` files — never hardcode them in Python
- Only `src/api/` reads and writes to the DB — engine functions transform state only