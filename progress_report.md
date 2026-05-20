# Progress Report — Conversational Clustering

*Course: Designing Large Scale AI Systems · Team: vibe-coders (5 members) · Date: 2026-05-20 · Sprint 2*

---

## Slide 1 — Problem statement & what's hard

**Problem.** Build an AI system that clusters a text dataset by *conversing* with a
human (the **oracle**): it proposes a grouping, explains it, and refines it through
natural-language feedback. The oracle is the sole judge of quality — there is no
fixed ground truth.

**What's hard.**
- Clustering is ill-defined: three people produce three valid groupings, and the
  oracle often doesn't know what they want until they see one.
- Preferences **emerge and contradict** mid-conversation ("merge A and B" → later
  "A and B are too different"). Latest intent must win, but the drift must be surfaced.
- Every turn has a **cognitive-load cost**: the system must be economical about what
  it shows and what it asks, not dump everything.
- No ground truth means **evaluation itself is a research problem**.

---

## Slide 2 — Focus, scope & key contributions

**Profile.** Build-heavy project (per the course brief), team of 5.

**In scope (minimum viable build).**
- End-to-end conversation over a fixed text dataset — Amazon Reviews, 1200 points.
- Initial clustering with LLM-generated cluster names and descriptions.
- Oracle feedback at multiple levels: global / cluster / point / instructional.
- Soft assignments — a probability distribution over clusters for each point.
- A Planner/Executor agentic loop orchestrated through structured JSON.
- A CLI interface.

**Key contributions we are targeting.**
- The **conversational refinement loop**: does dialogue actually move the clustering
  toward oracle acceptance, and how fast (turns, cognitive load)?
- **Uncertainty-driven interaction**: using soft-assignment boundary points to decide
  what to ask the oracle next.
- A **natural-language → clustering-parameter bridge**: the oracle states intent in
  free text, the system extracts `k` and the clustering axis.

**Deferred (scope discipline).** Hierarchy navigation, UMAP/t-SNE visualization,
multiple clustering backends (HDBSCAN) — listed explicitly, planned for later sprints.

---

## Slide 3 — Summary status

**Architecture.** Planner/Executor pattern. FastAPI + SQLite backend; a
provider-agnostic LLM harness (Claude / OpenAI).

| Component | Owner | Status |
|---|---|---|
| Data ingestion + embeddings (1500 Amazon reviews) | P2 | Done |
| Initial k-means clustering + soft assignments + LLM naming | P2 | Done |
| LLM harness — retry, dry-run, token/cost tracking | P4 | Done |
| `f_output` / `f_next_state` — Executor | P3 | Done |
| `f_uncertainty` — Sensor (boundary-point scoring) | P3 | Done |
| `f_next_best_step` — Planner (show / ask / stop) | P3/P4 | Done |
| `f_parse_clustering_intent` — NL intent → k + axis | P3 | Done |
| Backend routers (sessions / datasets / turns / clusters) | P1 | Done |
| Structured logger + interactive CLI | P5 | Done |
| `f_eval` — Judge (self-assessment) | P3 | In progress (stub) |
| Closed-loop wiring (re-clustering on feedback; planner on the API path) | team | Not yet |

**Honest headline.** All the building blocks now exist, but the loop is **not yet
wired end-to-end**: the turn endpoint calls the executor, but oracle feedback does
not yet re-cluster the database, and the planner is not yet on the API path.

---

## Slide 4 — Progress over the week

- **P1 — Backend & Data Modeling.** Split the API into dedicated routers; clusters
  and turns endpoints; session delete + status patch; cluster-points render API.
- **P2 — Data & Embeddings.** k-means initial clustering, soft assignments, LLM
  cluster naming, silhouette-based k tuning; fixed turn numbering (initial
  clustering recorded at turn 0, the pre-oracle state).
- **P3 — Core Engine.** Implemented `f_output`, `f_next_state`, `f_uncertainty`,
  `f_next_best_step`; added natural-language oracle-intent parsing.
- **P4 — LLM Harness.** Claude/OpenAI harness with retry + dry-run mode;
  `detect_contradiction`; token-usage and cost tracking exposed from `f_output`.
- **P5 — CLI & Logging.** Structured logger (`log_llm_call`, deviation tracking);
  interactive command-line interface.

---

## Slide 5 — Evaluation ideas

No ground truth exists, so we combine **process** and **outcome** signals.

- **LLM-as-oracle simulation** — a small LLM given a preference spec, a persona and a
  cognitive-load budget, to run many conversations at scale; validated against a
  small human study (N ≈ 5–10, within-subject).
- **Primary metric: turns-to-convergence** — weighted by feedback type (not all
  turns are equal).
- **Cognitive load per turn** — clusters/items shown, text length, question
  complexity — measured and minimized.
- **Baseline comparison** — conversational refinement vs. a default-parameter
  clustering: does the dialogue actually help?
- **Soft-assignment calibration** — when `f_uncertainty` flags a boundary point, is
  it genuinely ambiguous?
- **Contradiction / preference-drift tracking** — how often, how severe, and whether
  the system detects it.
- **Internal sanity check** — silhouette score tracked across turns (a secondary
  diagnostic, *not* the objective).
- **Generalization probe** — a frozen held-out set (300 reviews already reserved) to
  test whether oracle preferences can be codified into a reusable mapping function.

A written **quality spec** will be committed *before* the main experiment runs.
Candidate headline question: *"Does conversational refinement converge toward
oracle-accepted clusterings, and at what cognitive-load cost?"*

---

## Slide 6 — Plan for next week

- **Close the loop.** Implement `f_eval`; wire re-clustering so oracle feedback
  actually updates clusters + soft assignments; route the turn endpoint through
  `f_next_best_step` (the planner).
- **One full conversation end-to-end** via the CLI — demo-ready.
- Fix the CLI/API path mismatch (CLI calls `/sessions/{id}/turns`, API exposes
  `/turns`).
- Commit the **quality spec** and lock the headline research question.
- Start the **LLM-as-oracle** evaluation harness.
- Scaffolding: smoke test (`scripts/smoke_test.sh`) and an architecture / data-model
  document.

---

## Slide 7 — Demo

- **No hosted demo link yet.** The system runs locally.
- Run: `uvicorn backend.main:app --reload`, then `python scripts/cli.py`.
- Current demo path: create a session → run initial clustering (k-means + LLM names)
  → send oracle turns (logged). The full re-clustering loop demo is targeted for
  next week.
- Repository: `github.com/ai-design-2026-projects/vibe-coders`.
