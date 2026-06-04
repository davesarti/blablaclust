# Progress Report — Conversational Clustering

*Course: Designing Large Scale AI Systems · Team: vibe-coders (5 members) · Date: 2026-06-04 · Sprint 5*

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
- Conversation over a text dataset — **three datasets**: Amazon Reviews (1200,
  sentiment axis), 20 Newsgroups (1200, topic axis) and IMDB (1200, sentiment
  axis), to show the system adapts rather than being tuned to one.
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
| **Generalization (online, label-free)**: A1/B2 paired Δ around an ingestion event | P2 | Done |
| **No-dialogue baseline arm** (A1+B2 with bootstrap CI) | P2 | Done |
| **LLM-as-oracle harness + 3 personas** | P2/P5 | Done |
| Human study | team | Not yet (protocol pending) |

**Honest headline.** The loop is **wired end-to-end and verified live** on all
three datasets (Amazon, 20NG, IMDB): an oracle turn is interpreted by the LLM,
re-clusters the database (merge / split / move / semantic re-embed), and
persists a new soft-assignment snapshot. The **LLM-as-oracle harness now runs
end-to-end**: all 3 personas reach `oracle_satisfied` with parse-retry +
4xx-resilient runner. The **no-dialogue baseline arm** and **bootstrap 95% CIs**
on the eval aggregates are in place. The remaining gap is the **human study**
(protocol still to write).

---

## Slide 4 — Progress over the week

- **P1 — Backend & Data Modeling.** Routers for sessions / datasets / turns /
  clusters; session name + status; cluster-points render API.
- **P2 — Data & Embeddings.** Added a **second** (20 Newsgroups) and **third**
  (IMDB) dataset and proved adaptivity (topic names recovered, all-3 oracle
  loop verified); single-call LLM naming; reframed **generalization** as an
  online label-free A1/B2 eval with bootstrap CIs; built the **no-dialogue
  baseline arm** (`run_baseline_eval.py`) and added **bootstrap 95% CIs** to
  `eval_report.py`; UMAP visualization (Phase 1 + Phase 2 geometry-aware);
  per-turn LLM cost accumulator + complete audit logging; ran **3 personas
  end-to-end** with parse-retry and 4xx-resilient runner; DB migrations
  (Dataset model, `data`→`text`) without re-embedding; zombie-session fix +
  k≥2 enforcement.
- **P3 — Core Engine.** Executor + operation dispatch (`f_apply_operations`),
  variable-arity split, naming-driven merge/split decisions, `f_eval` /
  `f_validate_point` judge; rename preserves existing names/descriptions.
- **P4 — LLM Harness.** Retry + dry-run; prompt versioning; tolerant JSON
  parsing (handles markdown-fenced LLM output); inline naming on ops.
- **P5 — Evaluation & UI.** Full **React + TypeScript + Tailwind v4 rewrite**
  of the frontend (replaces HTML prototype): two-panel workspace (cluster grid
  + fixed chat sidebar), animated welcome page, Plotly UMAP evolution modal,
  eval scores modal, per-cluster expand/search/pin, dataset management UI
  (upload, preview, delete), stop + export modal, persona session UI; eval
  harness (LLM-as-oracle, 23 personas, 7 scripted scenarios) + clustering-run
  log; semantic re-embedding feature (cosine + LLM hybrid, clarify/confirm
  flow, axis-aware cluster naming); literature review and state-of-the-art
  positioning against 5 related systems (`docs/related-work.md`, radar chart);
  quantitative embedding model comparison MiniLM vs BGE-base (NMI +20%,
  ARI +24% — upgrade deferred on CPU latency, `docs/embedding-model-comparison.md`).

---

## Slide 5 — Evaluation ideas

No ground truth exists, so we combine **process** and **outcome** signals.

**Already measured.**
- **Generalization (online, label-free, with CIs)** — a converged clustering's
  centroids are frozen; new arrivals are assigned via nearest-centroid; we
  re-evaluate A1 (silhouette) and B2 (coherence) at t0 and t1 and report the
  paired Δ with bootstrap 95% CI. On 20 Newsgroups (1200 base + 300 new, k=6):
  **A1 paired Δ −0.0003, 95% CI [−0.0004, −0.0002]** — statistically detectable
  but practically negligible (~0.5% relative); **A1 holds**. **B2 paired Δ
  spans 0** across runs (judge non-determinism at k=6) — no significant
  change. *Labels out of scope:* the eval reads only `title,text`.
- **Honest limitation (qualitative).** On Amazon, k=2 splits by *topic*, not
  *sentiment* — both clusters are sentiment-mixed. The representation encodes
  topic; an oracle wanting a sentiment axis must steer it (semantic re-embed).
- **No-dialogue baseline arm** — the control arm for *"does dialogue improve
  clustering?"*. `run_baseline_eval.py` measures the initial k-means clustering
  with no oracle interaction, using the same A1+B2 metrics with bootstrap CIs.
  Read-only on the live DB (no pollution); A1 mean matches k-means' fit
  silhouette (sanity check). Baseline numbers: Amazon (k=4) A1 0.0385
  [0.0358, 0.0414], B2 0.788 [0.725, 0.850]; 20NG (k=6) A1 0.0555
  [0.0528, 0.0581], B2 0.525 [0.242, 0.783]; IMDB (k=4) A1 0.0032
  [0.0008, 0.0055], B2 0.787 [0.700, 0.875].
- **Bootstrap 95% CIs on the eval aggregates** — `eval_report.py` now reports
  percentile-bootstrap CIs (10k resamples) on every cross-scenario aggregate
  including **turns to convergence** (the convergence claim with CI requested
  by the brief). `n=1` is guarded.
- **LLM-as-oracle harness — 3 personas, all reach `oracle_satisfied`.**
  Parse-retry + provider-aware costs + 4xx-resilient runner. On Amazon:
  `satisfied_minimalist` 2 turns / $0.0014 / B1 0.85;
  `curious_explorer` 7 turns / $0.0093 / B1 0.75;
  `contradictory_oracle` 7 turns / $0.0081 / B1 0.75 (hit a 422 mid-session
  and recovered — proof the resilience works).
- **Early finding (n=3, scripted scenarios) — honest reading.** At matched k=5
  on Amazon, conversational B2 (0.69 [0.58, 0.78]) ≈ no-dialogue baseline B2
  (0.66 [0.33, 0.88]) — CIs overlap; **no significant intrinsic-quality gain
  from dialogue** — while B3 compliance = 1.00. The dialogue optimises
  **oracle preference** (B3), not the automated metric — consistent with the
  project's thesis ("the oracle is the objective").
- **Internal sanity checks** — silhouette score logged per clustering run;
  cluster naming and dialogue history validated end-to-end.
- **Quality spec** — written and kept current (`docs/quality_specs.md`).

**Pending (the remaining work).**
- **Human study** (N ≈ 5–10, within-subject, scripted protocol, consent) to
  validate the simulated oracle. Protocol still to write.
- **More scenario coverage** — the convergence-with-CI claim becomes stronger
  as we run more personas/scenarios; n=3 scripted scenarios is the floor.
- **Contradiction / drift tracking** and **soft-assignment calibration** —
  scaffolding present; to be measured systematically.

Candidate headline question: *"Does conversational refinement converge toward
oracle-accepted clusterings, and at what cognitive-load cost — compared to a
no-dialogue baseline?"*

---

## Slide 6 — Plan for next week

The loop, `f_eval`, the quality spec, the three datasets, the generalization
result, the **no-dialogue baseline arm**, **bootstrap CIs on the eval
aggregates** and the **LLM-as-oracle harness with 3 personas** are **done**.
The final week is about **the human study** and **the write-up**:

- **Human study** (N ≈ 5–10, scripted protocol, consent) to validate the
  simulated oracle. Protocol still to write — the prof flagged "3 friends with
  no protocol" as a risk.
- **Prompts dataset-agnostic** (#48) — remove residual "customer reviews" framing
  so naming/intent aren't biased toward Amazon (still pending).
- **Write-up**: fold the **baseline-vs-conversational comparison** with CIs,
  the online generalization result, the topic-vs-sentiment contrast, and the
  persona-eval finding into the final report; honest discussion that the
  dialogue serves oracle preference (B3=1.0), not the automated metric.
- Optional if time allows: hierarchy, a second backend (HDBSCAN). UMAP/t-SNE
  is already shipped.

---

## Slide 7 — Demo

- **No hosted demo link yet.** The system runs locally.
- Run (two terminals): backend `PYTHONPATH=. python scripts/serve_ui.py`
  (auto-seeds demo data on first launch), then React UI `cd frontend && npm run
  dev` → `http://localhost:5173`. Legacy single-page UI still served at
  `http://localhost:8000/ui`; CLI also available via `python scripts/cli.py`.
- **Working demo path** (verified live on all three datasets): pick a dataset →
  initial clustering (k-means + LLM topic names) → oracle turns that **merge /
  split / move** clusters and re-cluster the data in place → semantic re-embed
  along a user-stated axis → token/cost/cognitive-load shown per turn → UMAP
  panel with the per-turn projection. The full conversational loop works
  end-to-end and `oracle_satisfied` is reachable both manually and via the
  scripted personas.
- Repository: `github.com/ai-design-2026-projects/vibe-coders`.
