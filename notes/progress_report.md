# Progress Report — Conversational Clustering

*Course: Designing Large Scale AI Systems · Team: vibe-coders (5 members) · Date: 2026-05-29 · Sprint 4*

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

**In scope (minimum viable build) — now delivered end-to-end.**
- Conversation over a text dataset — **two datasets**: Amazon Reviews (1200,
  sentiment axis) and 20 Newsgroups (1200, topic axis), to show the system
  adapts rather than being tuned to one.
- Initial clustering with LLM-generated cluster names and descriptions.
- Oracle feedback at multiple levels: global / cluster / point / instructional.
- Soft assignments — a probability distribution over clusters for each point.
- A Planner/Executor agentic loop orchestrated through structured JSON,
  **closing the loop**: oracle feedback re-clusters and persists new snapshots.
- A CLI and a single-page web UI.

**Key contributions we are targeting.**
- The **conversational refinement loop**: does dialogue actually move the clustering
  toward oracle acceptance, and how fast (turns, cognitive load)?
- **Uncertainty-driven interaction**: using soft-assignment boundary points to decide
  what to ask the oracle next.
- A **natural-language → clustering-parameter bridge**: the oracle states intent in
  free text, the system extracts `k` and the clustering axis.

**Deferred (scope discipline).** Hierarchy navigation, UMAP/t-SNE visualization,
multiple clustering backends (HDBSCAN) — listed explicitly, still deferred.

---

## Slide 3 — Summary status

**Architecture.** Planner/Executor pattern. FastAPI + SQLite backend; a
provider-agnostic LLM harness (Claude / OpenAI / OpenRouter).

| Component | Owner | Status |
|---|---|---|
| Data ingestion + embeddings — **two datasets** (Amazon + 20 Newsgroups) | P2 | Done |
| Initial k-means clustering + soft assignments + LLM naming | P2 | Done |
| Merge / split / move operations (re-clustering on feedback) | P2 | Done |
| LLM harness — retry, dry-run, token/cost tracking | P4 | Done |
| `f_output` / `f_apply_operations` — Executor + dispatch | P3 | Done |
| `f_uncertainty` — Sensor (boundary-point scoring) | P3 | Done |
| `f_next_best_step` — Planner (show / ask / stop), on the API path | P3/P4 | Done |
| `f_parse_clustering_intent` — NL intent → k + axis | P3 | Done |
| `f_eval` / `f_validate_point` — Judge (self-assessment, B4) | P3 | Done |
| Backend routers (sessions / datasets / turns / clusters) | P1 | Done |
| Structured logger + clustering-run log + CLI + web UI | P5 | Done |
| **Closed-loop**: oracle feedback re-clusters + persists; planner on API path | team | Done |
| **Generalization**: codified mapping evaluated on a held-out split | P2 | Done |
| LLM-as-oracle harness + human study | P5/P3 | Not yet |

**Honest headline.** The loop is now **wired end-to-end and verified live**: an
oracle turn is interpreted by the LLM, re-clusters the database (merge / split /
move), and persists a new soft-assignment snapshot — confirmed on both datasets.
The remaining gap is **evaluation at scale**: the LLM-as-oracle harness and the
human study are not yet done. (Generalization is already quantified — see Slide 5.)

---

## Slide 4 — Progress over the week

- **P1 — Backend & Data Modeling.** Routers for sessions / datasets / turns /
  clusters; session name + status; cluster-points render API.
- **P2 — Data & Embeddings.** Added a **second dataset** (20 Newsgroups) and
  proved adaptivity (87.9% purity, topic names); single-call LLM naming;
  fixed a merge bug that collapsed the dataset into one cluster; verified
  merge/split/move on the new geometry; built and ran the **generalization**
  eval (Slide 5); silhouette computed once and made crash-safe.
- **P3 — Core Engine.** Executor + operation dispatch (`f_apply_operations`),
  variable-arity split, naming-driven merge/split decisions, `f_eval` /
  `f_validate_point` judge; rename preserves existing names/descriptions.
- **P4 — LLM Harness.** Retry + dry-run; prompt versioning; tolerant JSON
  parsing (handles markdown-fenced LLM output); inline naming on ops.
- **P5 — Evaluation & UI.** Eval harness + scenarios + clustering-run log;
  web UI (dynamic dataset label, loading states, token/cost/load readout);
  semantic re-embedding feature (branch).

---

## Slide 5 — Evaluation ideas

No ground truth exists, so we combine **process** and **outcome** signals.

**Already measured.**
- **Generalization (quantified, with CIs)** — a finished clustering is codified
  into a nearest-centroid mapping and applied to a frozen held-out split. On
  20 Newsgroups (real category labels): **87.0% held-out accuracy, 95% CI
  [82.7%, 90.3%]**. Contrast on Amazon (vs. sentiment): **50.7% = base rate**,
  because the embeddings cluster by *topic*, not *sentiment* — an honest limit
  of what the representation encodes. Oracle refinement *holds* generalization
  (87.0%→88.3%, not significant, McNemar p=0.34 — a ceiling effect, since
  k-means already recovers the categories).
- **Internal sanity check** — silhouette score logged per clustering run.
- **Quality spec** — written and committed (`docs/quality_specs.md`).

**Pending (the main remaining work).**
- **LLM-as-oracle simulation** — a small LLM with a preference spec, persona and
  cognitive-load budget, to run many conversations at scale.
- **Turns-to-convergence** (primary process metric, type-weighted) and
  **cognitive load per turn** — harness + scenarios exist; needs the LLM-oracle.
- **Human study** (N ≈ 5–10, within-subject) to validate the simulated oracle.
- **Contradiction / drift tracking** and **soft-assignment calibration** —
  scaffolding present; to be measured systematically.

Candidate headline question: *"Does conversational refinement converge toward
oracle-accepted clusterings, and at what cognitive-load cost?"*

---

## Slide 6 — Plan for next week

The loop, `f_eval`, the quality spec, the second dataset and the generalization
result are **done**. The final week is about **evaluation at scale** and polish:

- **LLM-as-oracle harness** — run full conversations with a simulated oracle to
  produce turns-to-convergence and cognitive-load numbers with CIs.
- **Small human study** (N ≈ 5–10, scripted protocol) to validate the simulated
  oracle before trusting any quantitative claim.
- **Prompts dataset-agnostic** (#48) — remove residual "customer reviews" framing
  so naming/intent aren't biased toward Amazon.
- **Write-up**: fold the generalization result and the topic-vs-sentiment contrast
  into the final report; honest discussion of what the oracle signal can and
  cannot tell you.
- Optional if time allows: hierarchy, UMAP/t-SNE, a second backend (HDBSCAN).

---

## Slide 7 — Demo

- **No hosted demo link yet.** The system runs locally.
- Run: `PYTHONPATH=. python scripts/serve_ui.py`, then open `http://localhost:8000/ui`
  (auto-seeds the demo dataset on first launch). CLI also available via
  `python scripts/cli.py`.
- **Working demo path** (verified live on both datasets): pick a dataset →
  initial clustering (k-means + LLM topic names) → oracle turns that **merge /
  split / move** clusters and re-cluster the data in place → token/cost/cognitive-
  load shown per turn. The full conversational loop works end-to-end.
- Repository: `github.com/ai-design-2026-projects/vibe-coders`.
