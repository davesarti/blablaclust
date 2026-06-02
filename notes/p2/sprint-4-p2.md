# Sprint 4 — P2 (Data & Embeddings)

## Theme

Prove the system is genuinely **dataset-agnostic**, and answer the brief's
**generalization** question with a quantified, defensible claim (with CIs).
Everything below is verified; methodology notes are deliberately explicit
because these are research deliverables, not just code.

> **Revision (post-#52 follow-up): generalization reframed as an online,
> label-free eval.** The original sprint-4 generalization deliverable was a
> *label-driven held-out accuracy* eval (score the codified clustering against
> ground-truth category labels). A follow-up issue corrected the framing:
> clustering here is **unsupervised** — the oracle is the objective and the
> `label` column is out of scope — and "generalization" operationally means
> **consistency under growth** (new data arrives into the converged system and
> the clustering stays coherent), *not* accuracy against a hidden category. The
> four label-driven scripts (`run_generalization_eval`, `…_oracle_eval`,
> `…_multiseed`, `…_amazon_axis`) were **removed** and replaced by the online
> eval (§10). The pure nearest-centroid functions and the `n_init=20` k-means
> robustness fix are **kept** (still used); the topic-vs-sentiment insight is
> preserved as a report-ready limitation. Sections 2/4/8/9 are annotated with
> what changed.

## Delivered

### 1. Second dataset — 20 Newsgroups (adaptivity proof)

`src/dataset_processing/sample_20newsgroups.py` + `data/20newsgroups_{train,frozen}.csv`.

Curated 6 well-separated newsgroups (space, baseball, guns, graphics,
medicine, autos) via `sklearn`, headers/footers/quotes stripped, 1200 train +
300 frozen rows in the `label,title,text` schema. `title` is intentionally
empty so the category label is never leaked into the embedding. A topic-based
dataset is the opposite axis of Amazon's sentiment, so it stress-tests whether
the system adapts rather than being tuned to one dataset.

Verified live: k=6 clustering recovers all six topics with **topic-appropriate
names** ("Space Astronomy", "Baseball Strategy", …) and **87.9% purity** vs.
ground truth. Amazon is untouched; the two datasets coexist in the DB isolated
by `dataset_name`.

### 2. Generalization mapping function (#52 — closed)

`src/engine/generalization.py` + `tests/test_generalization.py`.

Codifies a finished clustering into a reusable **nearest-centroid mapping**
(`build_centroids` / `assign_nearest` / `centroids_from_snapshot`) — the same
Euclidean geometry k-means used, so a new item is assigned as if it had been in
the original fit. Pure-numpy core, DB/LLM-free, mutation-tested (argmin→argmax
fails 3 tests). **These functions are kept** — they are now the engine behind
`ingest_points` and the online generalization eval (§10).

> **Removed (label-driven):** the original `scripts/run_generalization_eval.py`
> scored this mapping against ground-truth labels on a held-out split
> (20NG ~87%, Amazon 50.7%). That measures **embedding-space topic recovery**,
> not the conversational system, and reads the `label` column — out of scope for
> unsupervised clustering. Replaced by §10.

**Preserved insight (report-ready):** on Amazon, k=2 clusters split by **topic,
not sentiment** — both clusters are sentiment-mixed. The representation encodes
topic; an oracle wanting a sentiment axis must *steer* it (semantic re-embed).
This topic-vs-sentiment contrast is exactly the kind of honest limitation the
write-up should foreground.

### 3. Loop verification on 20NG (#49 — closed)

`scripts/verify_loop_20ng.py` + **live API test**.

The oracle operations had only ever run on Amazon; the merge bug (22a5b0e) was
caused by flat soft-assignment geometry, which is dataset-dependent. Drove
merge / split(k=2) / split(k=3) / move through the real engine path
(`f_apply_operations` → `cluster_operations`) on 20NG, asserting after each op:
merge size == sum of merged (others unchanged, total conserved — no collapse);
split children sum to parent; move changes exactly the named point. **All
invariants hold.**

Also ran a **live session through the API + Gemini** (not just the
deterministic script): merge 187+187→374 with a sensible combined name, split
back into the two original topics, move of a single point — all correct. So
the loop is dataset-agnostic through the full LLM→executor→ops stack, not just
k-means.

### 4. Oracle-refined generalization (Option B) — REMOVED (label-driven)

`scripts/run_generalization_oracle_eval.py` **removed.** It simulated a competent
oracle that moves misassigned points to their *true category* (using labels),
rebuilt centroids, and re-scored held-out **accuracy** (87.0% → 88.3%, Δ not
significant, McNemar p=0.34). The paired-statistics methodology was sound, but the
eval is label-driven and frames generalization as accuracy-vs-truth — the framing
the follow-up issue rejects. Its honest finding (oracle refinement *holds*
generalization — a ceiling effect, not a loop failure) is now expressed
label-free by the online eval's paired-Δ-with-CI on A1/B2 (§10).

### 5. Supporting fixes

- **Silhouette perf + robustness** — `initial_clustering` now returns the
  silhouette it already computed for the log, so `POST /clusters` fits k-means
  once instead of twice; and `silhouette_score` is wrapped in try/except so a
  degenerate input (all-identical embeddings) can never abort a clustering run.
- **P3 stale test fix** (cross-team, with P2's go-ahead) — P3's #43 fix made
  rename preserve an existing description; the old test still asserted the
  cleared-to-"" behavior and was failing on main. Rewrote it to assert the new
  (correct) behavior, mutation-checked. Flagged to @MatteoPareto.

### 6. Test-suite stabilization after the cross-team merge

After the semantic-reembed (P5) and session-eval (P1) branches landed on main,
the suite was **210 pass / 1 fail / 22 errors**. Two distinct causes, both fixed:

- **22 errors** — `tests/test_semantic_clustering.py` (P5) unpacked
  `initial_clustering()` as a 2-tuple, but the silhouette change (§5) made it a
  3-tuple. Added `, _` at the single call-site. Cross-team file, fixed with P2's
  go-ahead since the break originated from my change; flagged @ariannaschiavi-AIS.
- **1 failure** (`test_multiple_target_clusters_accepted`) — NOT a side effect of
  the errors (the received diagnosis was wrong). A latent test-isolation bug:
  `harness.py` freezes `DRY_RUN` into a module constant **at import time**, and
  only `test_turns_endpoint.py` set `HARNESS_DRY_RUN`. When `test_semantic_*`
  (alphabetically earlier) imported `harness` first, `DRY_RUN` froze to False →
  the full suite made a **real LLM call** (verified: real token counts + ~$0.006
  in captured stdout) and produced a spurious 422. Fixed in `tests/conftest.py`
  with `os.environ.setdefault("HARNESS_DRY_RUN", "true")` before any import.
  Suite back to fully green, deterministic across collection orders, and tests no
  longer spend API credits or touch the network.

### 7. UMAP visualization of clustering evolution (#53)

`src/viz/umap_projection.py` + `backend/routers/umap.py` +
`tests/test_umap_projection.py` + UMAP modal in `ui/index.html` +
`umap-learn` in `requirements.txt`.

Lets the oracle **watch the clustering evolve** over a conversation. Design:
fit a **single 2-D UMAP layout** over the dataset embeddings (fixed coordinates),
then recolour the same points by their per-turn hard cluster (argmax of the soft
snapshot) as a **turn slider** moves — points stay put, only the partition
changes (re-fitting UMAP per turn would teleport points and is the trap we
avoid). Pluggable reducer (umap-learn, **PCA fallback** so it never hard-depends
on the heavier numba stack); UMAP runs once per dataset and is cached. The
endpoint also returns per-point **assignment confidence** (winning soft prob) and
the **silhouette-per-turn** overlay from the clustering log.

Phase-2 UI: **Play ▶** auto-advance; **merge/split highlight** (clusters born
this turn marked ✦ + outlined, with `✦N new / −M gone` deltas); **Export
PNG/HTML**; professional hover (bold cluster name + wrapped/truncated text +
confidence). Plotly.js via CDN, rendering client-side.

Verified live (real Amazon session, browser-driven): 1200 points, per-turn
snapshots correct, **0 unassigned**, the slider shows the topic layout being
re-partitioned each turn. 10 unit tests on the projection module; full suite
green.

**Three bugs found by real use and fixed (root-caused, not patched blind):**
- *Stale after merge/split* — the browser cached the `GET …/umap` response,
  hiding new/modified clusters. Proven server-side correct (injected a real merge
  → endpoint returns the new turn + the merged cluster's 357 points). Fixed with
  `Cache-Control: no-store` (endpoint) + `fetch(cache:'no-store')`.
- *Blank plot after many opens* — `scattergl` leaked a WebGL context on every
  open; after enough opens the browser dropped the oldest and the plot rendered
  blank (this is what "UMAP empty after a split" actually was). Switched point
  traces to **SVG `scatter`** (1200 pts render fine, no context limit) + purge on
  reopen. Verified: 12 consecutive reopens → 1200 points every time.
- *Cluster-count overcount* — the info bar counted overlay traces (ghost
  centroids + axis arrow from a parallel re-embed-viz feature) as clusters on
  re-embed turns; now counts real point-groups only.

Cross-team: `ui/index.html` (P5/Arianna), router registration in
`backend/main.py` (P1/Dave) — additive, flagged. Built on top of a parallel
"semantic axis arrow / ghost centroid" overlay (committed by another session)
without clobbering it; reconciled cleanly with P1's #51 push (no conflicts —
different regions).

### 8. k-means robustness fix (`n_init=20`) — KEPT; multi-seed *script* removed

`seed` param added to `initial_clustering()` (retrocompatible, default = 42) and
**`n_init` raised from "auto" (10) to 20** in `_fit_kmeans`.

**The bug (kept fix):** with `n_init="auto"`, seeds 2 and 7 produced a
*degenerate partition* on 20NG k=6 — k-means merged two geometrically-close
topics into one cluster. 20 independent k-means++ restarts (keep lowest inertia,
the standard production recommendation) eliminate it. This is a real
**clustering-quality** improvement, independent of any labels, so it stays. No
API change; all call-sites unaffected.

> **Removed (label-driven):** `scripts/run_generalization_multiseed.py` measured
> held-out **accuracy** across 10 seeds (20NG 86.9% ± 0.2 pp) to show the headline
> was seed-stable. Sound methodology, but label-driven and tied to the removed
> accuracy eval. The robustness it demonstrated now lives in the engine fix above;
> seed-stability of the *online* eval can be re-checked label-free via §10.

### 9. Amazon sentiment-axis generalization — REMOVED (label-driven)

`scripts/run_generalization_amazon_axis.py` **removed.** It re-embedded Amazon
along a deterministic cosine **sentiment axis** before clustering, then scored
held-out **sentiment accuracy** (50.7% → 56.7%, CI [51.0, 62.2]) to show steering
breaks the degenerate topic split. The *qualitative* result — steering works but
the cosine sentiment signal on MiniLM is weak and saturates, which is why the
live re-embed keeps an **LLM-scoring fallback** — is preserved as a note for the
semantic-re-embed feature. The labeled-accuracy number is dropped (out of scope).

### 10. Online generalization eval (the replacement) — `ingest_points` + stability eval

`src/engine/generalization.py::ingest_points` + `tests/test_generalization.py`
(8 new DB-backed tests) + `scripts/run_generalization_stability_eval.py` +
`docs/quality_specs.md` ("Generalization (procedure)" subsection).

**The correct framing, label-free.** Generalization = re-evaluate the already-
frozen metrics **A1 (silhouette)** and **B2 (coherence)** at two snapshots around
an **ingestion event** — *no new metric*. Procedure:
1. Cluster a base corpus → **converged** snapshot (frozen centroids).
2. **t0** eval: A1 + B2.
3. **Ingest** new arrivals: embed → `assign_nearest` vs the frozen centroids →
   full snapshot at `turn+1`. Pre-existing points carried forward **verbatim**
   (read-only assignment); new points get soft probs so ill-fitting ones earn a
   low max-prob and surface in B2's bottom-2 sample.
4. **t1** eval: A1 + B2 again + an A1 sub-aggregate over the new batch.
5. Report **paired Δ + bootstrap 95% CI** on A1 (common points) and B2
   (per-cluster); `--batches N` traces a drift curve.

`ingest_points` documents the **read-only-assignment policy** and is why no
separate "assignment stability" metric is added (old-point stability is 100% by
construction). The eval reads **only `title,text`** — never `label`.

**Verified — full-scale run** (20NG: 1200 base + 300 new, k=6; A1 real + B2 via
real LLM judge `gemini-2.5-flash`). Converged silhouette 0.0562.
- **A1**: t0 0.0562 → t1 0.0551; paired Δ on the 1200 pre-existing points
  **−0.0003, 95% CI [−0.0004, −0.0002]** — statistically detectable (tight CI at
  n=1200) but **practically negligible** (~0.5% relative): A1 *holds*. New-batch
  sub-aggregate 0.0516 [0.0469, 0.0563]. Drift curve gently down 0.0562 → 0.0551.
- **B2** (LLM judge, **non-deterministic** — two real runs): run 1 mean
  0.700→0.558 (Δ −0.142 [−0.383, +0.033]); run 2 mean 0.683→0.733 (Δ +0.050
  [−0.125, +0.208]). The point estimate **flips sign between runs** (LLM sampling
  at k=6); the stable conclusion is **Δ CI spans 0 → no significant change** in
  coherence under ingestion. Run 1's min→0 showed the bottom-2 stress sample can
  catch an ill-fitting new member (the designed mechanism).
- **Honest read**: A1 generalization holds (negligible Δ); B2 shows no
  significant change at k=6 — don't over-read a single-run sign. 21
  `test_generalization.py` tests pass; A1 degrades gracefully to "B2 unavailable"
  if the judge errors.

> Model note: the original `google/gemini-2.0-flash-001` is **deprecated on
> OpenRouter (404)**; `.env` `OPENROUTER_MODEL` is now `google/gemini-2.5-flash`
> (works). quality_specs' frozen-model (v1) reference should be updated to match —
> the model swap affects all Family-B judge numbers, not just generalization.

### 11. UMAP Phase 2 — geometry-aware projection (#53 follow-up)

`src/viz/umap_projection.py` (new: `_build_hybrid_space`, `compute_geometry_aware_coords`) +
`backend/routers/umap.py` (`?geometry_aware=true` query param) +
`tests/test_umap_projection.py` (8 new tests, 18 total).

On semantic_reembed turns the partition appeared to "cut across" the original
topic layout — the original UMAP was fit on the raw embeddings, not on the
re-oriented hybrid space k-means actually used. Phase 2 adds a **second UMAP
per reembed turn**, fit on the `(N, D+1)` hybrid space reconstructed
**deterministically** (`"very {axis}"` / `"not {axis} at all"` poles, no LLM,
zero API cost, reproducible). Cached in memory per `(session, turn)`.

**Drive-by fix:** the old `reembed_turns` detection used a "≥2 clusters
dissolved AND ≥2 created same turn" heuristic that silently missed real
re-embeds (e.g. session `b502918a` t9 axis="happy and hungry" — engine
dissolved 1 / created many). Replaced with direct read of the persisted
`semantic_reembed` op from `state_snapshot`. `reembed_turns` and `axis_arrows`
now reflect the truth.

### 12. UMAP UI rework (issue #58)

`ui/index.html` — three changes, all surgical:

1. **Ghost circles removed.** The gray `circle-open` markers (dissolved cluster
   centroids from the prior turn) were confusing and positionally wrong. Removed
   entirely; the re-embed event is already communicated by ✦ new-cluster markers.

2. **Semantic axis now readable.** Replaced the PCA-arrow overlay (a line in
   data coordinates, hard to interpret) with: (a) a **paper-anchored annotation**
   `↻ re-embed axis: "sentiment"` at the top of the plot, and (b) a
   **coloured badge** in the info bar — both use the real `axis_label` from the
   DB.

3. **Layout shift fixed.** Info bar and slider were on the same flex row;
   changing info text width shifted the slider. New layout: **two rows** — row 1
   is play + slider + "Turn 3 / 5" + PNG/HTML + Close (stable width); row 2 is
   `.umap-info-bar` with `min-height: 18px` (text changes, layout never shifts).
   Removed the confusing "geom-aware" checkbox from the UI (hidden, still
   accessible programmatically for dev/eval).

### 13. History bar enrichment

`ui/index.html` — per-turn meta line under each system message:
- Badge `↻ re-embed: "sentiment"` (yellow) when the turn ran a semantic_reembed
- Turn number, token count, cost per-turn
- Works on both live turns (`handleSystemTurn`) and historical turns loaded on
  session resume

### 14. Token/cost cost fix — correct provider pricing

`src/harness.py`:
- Added `google/gemini-2.5-flash` ($0.30/$2.50 per M) and
  `google/gemini-2.5-flash-lite` ($0.10/$0.40) to `_PRICING`
- `estimate_cost_usd` now resolves the model from the active provider via
  `_active_model()` instead of always defaulting to Claude Sonnet

Previously every turn was priced at Claude Sonnet rates (~$0.025) instead of
Gemini rates (~$0.0025) — **10× over-reported cost**.

### 15. Token/cost persist on session resume

`ui/index.html` — resume block now accumulates `resumeTokens` / `resumeCost` /
`resumeLoad` from the historical turn list (same loop that builds the chat),
then assigns them to state instead of hard-coded 0. Opened issue #65 for
Arianna with the same bug in the new React frontend (`UPDATE_METRICS` reducer
replaces totals instead of accumulating).

### 16. IMDB dataset added (third dataset)

`src/dataset_processing/sample_imdb_dataset.py` + `data/imdb_train.csv` (1200
rows) + `data/imdb_frozen.csv` (300 rows).

Reads the 50k-row IMDB Dataset CSV, strips HTML (`<br />`), balances 750
positive + 750 negative, writes `label,title,text` (2=positive, 1=negative —
same convention as Amazon). Injected into the DB via the standard pipeline;
both splits have full embeddings. The dataset does **not** appear in the UI
dropdown (covered by the `_frozen` filter in `GET /datasets` for the frozen
split).

### 17. Frozen dataset hidden from UI

`backend/routers/datasets.py` — added `.filter(~Dataset.name.like("%_frozen"))`
to `list_datasets`. Held-out splits (`imdb_reviews_frozen`, etc.) are now
excluded from `GET /datasets` and therefore from the new-session dropdown.

### 18. DB migration — Dataset model (P1 cross-team alignment)

P1 introduced `Dataset` as a proper table (`datasets`), changing `DataPoint`
and `ChatSession` from a plain `dataset_name` string to a FK `dataset_id →
datasets.id`. The DB still had the old schema.

Migration executed (non-destructive):
1. Created 4 `Dataset` rows from the distinct `dataset_name` values already in
   the DB
2. Added `dataset_id` columns to `data_points` (3900 rows) and `sessions` (9
   rows)
3. Recreated `sessions` table to remove the `NOT NULL` constraint on
   `dataset_name` (SQLite doesn't support `ALTER COLUMN`; the old constraint
   blocked any new session creation with the updated router)

All 295 tests pass after migration; API and ORM verified.

### 19. LLM-as-oracle end-to-end verified

Fixed two broken personas (`"dataset": "Amazon"` → `"amazon_reviews"`) and
confirmed the full runner works: `satisfied_minimalist` → `oracle_satisfied` in
1 turn, $0.006, errors=0. All 3 personas (`satisfied_minimalist`,
`curious_explorer`, `contradictory_oracle`) are ready to run.

```bash
# Both terminals needed
PYTHONPATH=. python scripts/serve_ui.py          # terminal 1
PYTHONPATH=. python scripts/run_persona_eval.py \
  --personas 'personas/*.json' \
  --out reports/run_$(date +%Y%m%d) --max-turns 12   # terminal 2
```

### 20. React frontend bootstrap (P5 merge + build fix)

After merging Arianna's full React + TypeScript + Tailwind v4 rewrite
(`frontend/`), fixed two TypeScript build errors:
- `frontend/tsconfig.app.json`: `noUnusedLocals: false` (React 18 JSX transform
  doesn't require `import React`, but files still had it — TS strict mode broke
  the build)
- `frontend/src/plotly.d.ts`: `declare module 'plotly.js-dist-min'` (no
  `@types` package available)

Build now compiles clean. Dev server: `cd frontend && npm run dev` →
`http://localhost:5173` (proxies API calls to `:8000`).

## Results

| Check | Result |
|---|---|
| 20NG initial clustering (qualitative) | 6 topic-named clusters; `n_init=20` fixes degenerate seeds |
| Online generalization eval (A1/B2 around ingestion) | A1 Δ −0.0003 [−0.0004,−0.0002] holds (negligible); B2 Δ n.s. (CI spans 0, sign varies across runs, k=6) |
| `ingest_points` read-only ingestion | 8 DB tests: verbatim carry-forward, frozen centroids, calibration |
| Amazon topic-vs-sentiment (qualitative) | k=2 splits by topic, not sentiment (report-ready limitation) |
| Loop verification (merge/split/move) on 20NG | all invariants hold |
| Live API session (merge/split/move + Gemini) | all correct end-to-end |
| `generalization.py` mutation test | argmin→argmax caught (3 tests fail) |
| Post-merge suite stabilization | 210p/1f/22e → all green |
| UMAP projection module | 18 unit tests (10 baseline + 8 geometry-aware) |
| UMAP geometry-aware Phase 2 | second UMAP on hybrid (D+1) space, deterministic, cached |
| UMAP reembed detection fix | reads op directly from state_snapshot (was heuristic) |
| UMAP UI rework (#58) | ghost circles removed; axis badge; no-shift 2-row layout |
| Token/cost pricing fix | Gemini rates in `_PRICING`; `_active_model()` resolves provider |
| Token/cost persist on resume | accumulates from turn history instead of reset to 0 |
| IMDB dataset | 1200 + 300 frozen, balanced, HTML stripped, full embeddings |
| Frozen datasets hidden from UI | `_frozen` filter in `GET /datasets` |
| DB migration (P1 Dataset model) | non-destructive; 3900 data_points + 9 sessions migrated |
| LLM-as-oracle | personas fixed; `satisfied_minimalist` → oracle_satisfied in 1 turn |
| React frontend build | TypeScript errors fixed; `npm run dev` → localhost:5173 |
| Full test suite | **295 passed**, 0 fail |

## Issues opened this sprint

#42 (`f_eval` Gemini crash — already fixed by P3), #43 (rename/merge/split
descriptions — P3, fixed), #47 (token/cost/load to UI — me), #48 (prompts
dataset-agnostic — P4), #49 (loop verification — closed), #50 (eval scenarios
for 20NG — P5), #51 (datasets API record count — P1, merged), #52
(generalization — closed), #53 (UMAP clustering-evolution viz — me, delivered),
**#56** (generalization reframe — me, closed), **#58** (UMAP UI — me, closed),
**#65** (token/cost reset on session load — opened for P5/Arianna).

## Honest scope notes (for the report)

- **Labels are out of scope** (the central correction): clustering is
  unsupervised, the oracle is the objective, and the generalization eval reads
  only `title,text` — never `label`. Any ground-truth-accuracy number measures
  embedding-space topic recovery, not the conversational system. The 20NG ~87%
  and Amazon 50.7% figures from the removed scripts are *not* part of the claim.
- **Determinism boundary**: A1 (silhouette) in the online eval is deterministic
  (embeddings + k-means + nearest-centroid). B2 (coherence) needs the LLM judge,
  so its numbers require a real run (dry-run mocks it to 0.0). The full
  conversational loop's non-determinism is a separate layer (#50/LLM-as-oracle,
  P5/P3) we have not measured.
- **Topic-vs-sentiment (kept qualitatively)**: on Amazon the k=2 split is by
  topic, not sentiment (both clusters sentiment-mixed). Stated as a limitation,
  no labeled accuracy attached — exactly the honest contrast for the write-up.
- **UMAP is a qualitative aid, not a metric**: it is a non-linear 2-D projection,
  so absolute distances/areas are not faithful and the layout is fit on the
  *original* embeddings — on semantic-re-embed turns the partition is shown
  cutting across the topic layout, not in the re-oriented geometry (see the
  geometry-aware item under Pending). Use it to *read* evolution, not to measure
  it; the quantified claims stay silhouette + coherence (A1/B2), now also
  re-evaluated around an ingestion event for generalization.

## Pending / plan

- [x] Commit the verified-but-uncommitted files (#49 script, Option B script,
      P3 test fix) — done; @MatteoPareto notified.
- [x] Post-merge suite stabilization (22 errors + dry-run import-order race) —
      done, suite green.
- [x] UMAP viz (#53) — delivered + verified (Phase 1 + Phase 2 + UI rework #58).
- [x] `n_init=20` k-means robustness fix — kept.
- [x] **Generalization reframed** (#56, closed) — `ingest_points` + online eval
      + 21 tests + quality_specs subsection. Full-scale verified (20NG).
- [x] UMAP Phase 2 — geometry-aware projection, deterministic, cached, 8 tests.
- [x] UMAP UI rework (#58) — ghost circles removed, axis badge, no-shift layout.
- [x] Token/cost pricing fix — Gemini rates; `_active_model()` provider-aware.
- [x] Token/cost persist on resume — accumulates from history, not reset to 0.
      Issue #65 opened for Arianna (same bug in React `UPDATE_METRICS` reducer).
- [x] IMDB dataset — 1200+300 rows, balanced, full embeddings, hidden _frozen.
- [x] DB migration — P1 Dataset model aligned; `sessions` table recreated.
- [x] LLM-as-oracle end-to-end verified; personas dataset names fixed.
- [x] React frontend bootstrap — TypeScript build fixed; `npm run dev` works.
- [ ] **Commit pending:** `frontend/tsconfig.app.json` + `frontend/src/plotly.d.ts`
      (two small build-fix files, awaiting Thomas's ok).
- [ ] **Run personas at scale** — produce `reports/` with results.jsonl +
      summary.md across all 3 personas; commit as deliverable for the prof.
- [ ] **CI on convergence claim** — `eval_report.py` aggregates mean±std but
      lacks bootstrap CI. Needed for the prof's "1 claim with CI" requirement.
- [ ] **Baseline comparison** — clustering without dialogue for the
      trio/quartet headline experiment (prof's spec).
- [ ] **Human study protocol** — at minimum a written protocol (within-subject,
      randomized, scripted, consented). Prof flags "3 friends with no protocol"
      as a risk.
- [ ] **Update `notes/progress_report.md`** — still dated 2026-05-29, says
      "LLM-as-oracle: Not yet" and cites old 87% accuracy claim. Both wrong now.
      P5/P1 authored — flag to them.
- [ ] Coordinate with P5 on folding into the final report:
      turns-to-convergence, online generalization A1/B2 result,
      topic-vs-sentiment contrast, UMAP figures.
