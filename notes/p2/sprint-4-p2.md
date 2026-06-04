# Sprint 4 — P2 (Data & Embeddings)

## What I did this sprint

Reframed generalization label-free, added a third dataset, made the eval
statistically honest, and shipped the GMM clustering engine plus a round of
infra/UI fixes.

**Generalization (the headline P2 deliverable)**
- **Mapping function** (`generalization.py`, #52) — codifies a finished
  clustering into a reusable assignment (`build_centroids` / `assign_nearest` /
  `gmm_params_from_snapshot` / `assign_gmm_posterior`), pure-numpy, DB/LLM-free,
  mutation-tested.
- **Online stability eval** (`run_generalization_stability_eval.py`, #56) — the
  *correct, label-free* framing: re-evaluate the already-frozen **A1 (silhouette)**
  and **B2 (coherence)** at two snapshots around an **ingestion event**, no new
  metric. Adds **A4** (Mahalanobis OOD rate on new arrivals) and paired Δ +
  bootstrap 95% CIs. `ingest_points` carries existing points forward verbatim
  (read-only assignment), so there is no separate "assignment stability" metric.
- **Oracle-converged rework** (§ this sprint, final) — the eval now drives a real
  in-process **LLM-oracle conversation** (default persona `curious_explorer`) to a
  realistic converged state (real turns + A1–B4 eval) before measuring
  generalization, instead of freezing a bare clustering. Throwaway in-memory
  session, demo DB untouched; banner reports the real backend (GMM, not k-means);
  transient provider failures (5xx turns, all-zero B2) are retried. Verified
  full-scale on **Amazon / 20NG / IMDB** — A1 and B2 hold on all three (A4 OOD
  8–14%, mildly elevated where the conversation grew k via splits). Commit `1ab49ba`.
- **Removed two label-driven scripts** (the central scope correction): the
  ground-truth-accuracy generalization eval (20NG ~87%, Amazon 50.7%) and the
  oracle-refined Option B. They measure embedding-space topic recovery, not the
  conversational system, and read the `label` column — out of scope for
  unsupervised clustering. Their honest finding survives label-free in the online
  eval.

**Datasets**
- **IMDB** added as a third dataset (1200 + 300 frozen, balanced, HTML stripped,
  full embeddings) — a second sentiment corpus alongside Amazon.
- **20 Newsgroups** carried over and used as the topic-axis adaptivity proof.
- **Frozen splits hidden from the UI** (`_frozen` filter in `GET /datasets`).

**Eval & metrics**
- **Bootstrap 95% CIs** in `eval_report.py` on every aggregate + a
  turns-to-convergence claim (the prof asked for CIs on claims).
- **No-dialogue baseline arm** (`run_baseline_eval.py`) — clusters read-only from
  the DB, A1+B2 with CI, zero DB pollution. **Key finding:** matched-k on Amazon,
  B2 conversational 0.69 [0.58, 0.78] ≈ baseline 0.66 [0.33, 0.88] — CIs overlap,
  **no intrinsic gain from dialogue on the automatic metric**; B3 compliance = 1.0.
  The dialogue serves the oracle's *preference*, not the metric — which is the
  project's thesis.
- **LLM-as-oracle / persona reports** made robust — oracle parse-retry,
  provider-aware cost (`_active_model`), 4xx-resilient runner. All 3 personas now
  reach `oracle_satisfied` with real costs (was 1/3 partial, cost $0).

**Engine & infra fixes**
- **GMM is now the primary initial-clustering backend** (diagonal covariance,
  k-means fallback); soft probabilities are native GMM posteriors.
- **k-means robustness** — `n_init=20` to avoid degenerate seeds (kept; the
  multi-seed *script* was dropped as redundant).
- **Per-turn cost accumulator + complete LLM-call logging** (#47) — thread-local
  accumulator in `call_llm` captures every call in a turn (was `f_output` only);
  4 previously-silent functions now log; model + turn on every site.
- **Token/cost pricing fix** — correct Gemini rates in `_PRICING`; `_active_model()`
  resolves the provider (was pricing Gemini at Claude rates → $0).
- **Token/cost persist on session resume** — accumulates from turn history instead
  of resetting to 0 (#65 opened for the same bug in the React reducer).
- **Zombie-session fix + backend k≥2** — client-side k validation +
  rollback-on-failure; `POST /clusters` k=1 → 422; 15 orphan sessions purged.

**DB migrations** (both non-destructive, idempotent, embeddings preserved)
- **Dataset model** (with P1) — `datasets` table + `dataset_id` FKs.
- **`data` JSON blob → flat `text` column** — 3900 rows migrated.

**UMAP & UI**
- **UMAP clustering-evolution viz** (#53) — single fixed 2-D layout, recoloured by
  per-turn hard cluster as a turn slider moves (points stay put); Phase 2 added
  geometry-aware projection, play/merge-split highlights, PNG/HTML export.
- **UMAP UI rework** (#58) — ghost circles removed, axis badge, no-shift layout;
  geom-aware toggle dropped from the React UI (feature stays internal).
- **React frontend** brought up to build (TypeScript fixes) and WorkspacePage
  production build repaired; eval modal opens on click, not on session resume.

**Test-suite stabilization** after the semantic-reembed (P5) and session-eval (P1)
branches merged — fixed a 3-tuple unpack break and a `DRY_RUN`-import-order race in
`conftest.py` (the suite was making real LLM calls); back to fully green and
deterministic across collection orders.

## What blocked me

- **Cross-team merge fallout** — the P5/P1 branches landed with the suite at
  210p/1f/22e; root-caused both (a 2-tuple→3-tuple unpack in P5's test, and a
  test-isolation race where `harness.DRY_RUN` froze to False and the suite spent
  real API credits). Fixed with permission.
- **Deprecated model** — `google/gemini-2.0-flash-001` 404s on OpenRouter; switched
  `OPENROUTER_MODEL` to `google/gemini-2.5-flash`. This shifts every Family-B judge
  number, so the quality_specs frozen-model reference needs updating (flagged).
- **Flaky provider** — intermittent 502s / empty responses corrupt LLM-judged
  metrics; addressed in the generalization eval with retries, but it's a standing
  risk for any B-metric run.
- **DB migration coordination with P1** on the Dataset model (resolved).

## What I'm doing next

- **Human study protocol** — at minimum a written protocol (within-subject,
  randomized, scripted, consented); the prof flagged "3 friends, no protocol" as a
  risk.
- **Update `notes/progress_report.md`** — still dated 2026-05-29, says
  "LLM-as-oracle: not yet" and cites the removed 87% accuracy claim; flag to P5/P1.
- **Fold P2 results into the final write-up** with P5 — turns-to-convergence,
  online-generalization A1/B2/A4, and the no-intrinsic-dialogue-gain baseline.

## Commits

- `6b4e177` — fix(db): non-destructive `data_points` text-column migration
- `04f448d` — feat(harness): per-turn LLM cost accumulator + complete call logging
- `a095253` — fix(sessions): prevent zombie sessions on invalid k + enforce k≥2
- `b5c3cf0` — feat(eval): bootstrap CIs + no-dialogue baseline comparison arm
- `1772d85` — fix(eval): repair generalization stability eval (data→text)
- `43a6a60` — fix(eval): robust LLM-oracle (parse retry, provider-aware cost, 4xx-resilient)
- `afedb95` — fix(ui): drop geom-aware toggle from UMAP modal (feature stays internal)
- `53e9570` — fix(ui): repair WorkspacePage production build
- `3989da5` — fix(ui): eval modal opens on click only (not on session resume)
- `c83dca9` — fix(frontend): TypeScript build errors after the React rewrite
- `556c97f` — docs(report): refresh team progress report + quality specs
- `3d0df94` — docs(notes): sprint-4 P2 update (eval CIs, baseline arm, generalization fix)
- `1ab49ba` — feat(generalization): oracle-converged throwaway session in stability eval

## Issues

- **Opened & resolved by me:** #53 (UMAP visualization of clustering evolution).
- **Resolved by me (opened by a teammate):** #56 (generalization eval should be
  online — opened by P1), #58 (UMAP enhancements — P1), #71 (A4 distance-based OOD
  rate — P1), #47 (wire token/cost/cognitive load to the UI — opened by me in
  sprint 3, the per-turn cost accumulator work).
- **Opened by me, assigned to a teammate:** #65 (token/cost reset on session
  load/switch in the React reducer — P5).
