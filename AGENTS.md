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
| Sensor | `f_uncertainty` | Reads soft-assignment posteriors from the DB and computes cluster-level overlap and cohesion scores — feeds signal into the Planner |
| Planner | `f_next_best_step` | Reads state + uncertainty scores, decides the next action: show / ask / stop |
| Executor | `f_output` + `f_apply_operations` | Calls the LLM to turn oracle feedback into structured operations, then applies them via `TurnBuilder` |
| Reembedder | `f_semantic_reembed` | Projects data along a user-specified semantic axis (cosine anchor poles + LLM hybrid) |
| Repair | `f_boundary_repair` | Post-op LLM pass that corrects boundary points placed in the wrong cluster by geometry alone |
| Preference tracker | `f_update_preferences` | Distils the oracle's feedback history into a rolling 3–5 bullet summary injected into the next turn's prompt |
| Cognitive load | `f_cognitive_load` | Estimates conversation complexity; caps in `cognitive_load_caps.py` trigger A3 warnings |
| Judge | `f_eval` | Self-assesses clustering quality (A1–A3, B1–B4 metrics) at session end |
| Staging | `TurnBuilder` | In-memory staging layer; all ops read/write the builder; a single `commit()` writes the turn to the DB |

## Architecture

```
Oracle natural language input
           │
           ▼
    f_output ──▶ LLM (via src/harness/)
           │                  │
           │        operations (merge / split / move / rename /
           │                    semantic_reembed / cluster_reembed)
           │◀─────────────────┘
           │
           ├── semantic_reembed ──▶ semantic_clustering (whole dataset,
           │                        via f_semantic_reembed + GMM/k-means)
           │
           └── structural ops ──▶ f_apply_operations ──writes──▶ TurnBuilder (in-memory)
                                          │
                                          ├──▶ cluster_reembed ──▶ semantic_reembed_cluster
                                          │                         (single cluster,
                                          │                          f_semantic_reembed + GMM)
                                          │
                                          ├──▶ f_boundary_repair   (post-merge / post-split /
                                          │                          post-cluster_reembed)
                                          │
                                          ▼
                                  TurnBuilder.commit() ──writes──▶ clusters + soft_assignments (DB)
                                                                    (renormalized after every op)
           │
           ▼
    f_cluster_uncertainty ──reads──▶ SoftAssignment table (DB)
           │
           ▼
    f_next_best_step ──decides──▶ show / ask / stop
           │
           ├──▶ f_update_preferences  (rolling oracle summary)
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
- All LLM calls go through `src/harness/` — never import `anthropic` or `openai` directly in engine code
- Prompts live in `prompts/` as `.txt` files — never hardcode them in Python
- Engine functions write to `TurnBuilder`, not the DB directly; only `TurnBuilder.commit()` and API routers own DB transactions
- Engine errors propagate (no silent skipping); the API surfaces them as HTTP 422
