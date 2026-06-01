# Sprint 4 — P2 (Data & Embeddings)

## Theme

Prove the system is genuinely **dataset-agnostic**, and answer the brief's
**generalization** question with a quantified, defensible claim (with CIs).
Everything below is verified; methodology notes are deliberately explicit
because these are research deliverables, not just code.

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

`src/engine/generalization.py` + `tests/test_generalization.py` +
`scripts/run_generalization_eval.py`.

Codifies a finished clustering into a reusable **nearest-centroid mapping**
(the same Euclidean geometry k-means used, so a held-out item is assigned as if
it had been in the original fit). Pure-numpy core, DB/LLM-free, mutation-tested
(argmin→argmax fails 3 tests).

**Headline result (20NG, real ground truth):** held-out accuracy **87.0%**
(261/300), 95% CI Wilson **[82.7%, 90.3%]** (bootstrap agrees). Matches the
87.9% training purity → the mapping generalizes to unseen items about as well
as it fits training.

**Contrast (Amazon, vs sentiment labels): 50.7%** = base rate. Per-cluster
sentiment is ~50/56% → the k=2 split is orthogonal to sentiment. Honest
insight: the representation captures **topic, not sentiment**; generalization
holds when the oracle's labels align with what the embeddings encode.

Methodology checks: zero train∩frozen leakage; balanced labels (87% ≫ 17%
random baseline); cluster→label map is a clean bijection (no majority-vote
inflation); runs through the real `initial_clustering` engine; deterministic
(identical 87.0% on re-run).

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

### 4. Oracle-refined generalization (Option B)

`scripts/run_generalization_oracle_eval.py`.

The brief asks about generalization *once the oracle is happy*, not just the
initial clustering. Simulated a competent oracle whose intent is the 6
categories: it moves misassigned train points to their true-category cluster
(verified `move` op, no destructive merges), rebuilds centroids from the final
snapshot, re-evaluates held-out.

**Result: 87.0% → 88.3%, but NOT significant** (paired McNemar p=0.34, paired
Δ 95% CI [-0.7, +3.3] pts includes 0). Honest conclusion: oracle refinement
**holds** generalization (does not degrade it); k-means already recovers the
categories so well there is little headroom — a **ceiling effect**, not a loop
failure. Uses the correct paired statistic, not a comparison of overlapping
marginal CIs. The refined clustering is 6 perfectly pure clusters (verified).

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

### 8. Multi-seed robustness of the generalization claim

`scripts/run_generalization_multiseed.py` + `seed` param added to
`initial_clustering()` (retrocompatible, default = 42).

**The problem found:** with `n_init="auto"` (sklearn default = 10 restarts),
seeds 2 and 7 produced a *degenerate partition* on 20NG k=6: k-means merged
labels 1 and 4 (geometrically close topics) into one 392-pt cluster, leaving
label 4 without a majority cluster → 75% accuracy.  Std across 10 seeds was
4.8 pp with a 12 pp range — the 87% headline was a lucky seed.

**The fix:** raised `n_init` from "auto" (10) to **20** in `_fit_kmeans`.
20 independent k-means++ initialisations keep the best (lowest inertia), the
standard production recommendation. No API change; all call-sites unaffected.

**Result after fix (10 seeds [0,1,2,3,4,7,13,21,42,99]):**

| Dataset | mean ± std | min | max | range | baseline |
|---|---|---|---|---|---|
| 20NG k=6 | **86.9% ± 0.2 pp** | 86.7% | 87.3% | 0.7 pp | 16.7% |
| Amazon k=2 (topic) | 50.7% ± 0.0 pp | 50.7% | 50.7% | 0.0 pp | 50.0% |

The 20NG claim is now **stable** (std = 0.2 pp, range = 0.7 pp, all 10 seeds
> 86.5%, all 6 labels covered by every seed). Amazon collapses to base rate
deterministically on every seed (both clusters always map to the positive
label — a structural property of topic-vs-sentiment, not a fluke).

The canonical seed-42 result is now **87.3%** (262/300, one extra correct
prediction freed by a better init). Wilson CI widens slightly to [82.7, 90.3]
at the canonical seed, but the full multi-seed range [86.7, 87.3] lies
comfortably above the 20NG random baseline (16.7%) on all seeds.

### 9. Amazon generalization along the sentiment axis (steered re-embed)

`scripts/run_generalization_amazon_axis.py`.

The §2 Amazon contrast (50.7%) measures *topic* clustering against *sentiment*
labels — both k=2 clusters collapse onto the positive label, so it is exactly
the base rate. This eval asks the **steered** question: re-orient the space
along a **sentiment axis before clustering** (the semantic-re-embed capability,
kept deterministic here — cosine anchor poles, no LLM — so the number is
reproducible), then codify + generalize with the same nearest-centroid + CI
machinery. Held-out items are projected into the same hybrid space (same poles,
train-derived standardization — no peeking).

**Result: 50.7% → 56.7%** (170/300), Wilson CI **[51.0%, 62.2%]**. The CI lower
bound just clears the base rate, and — crucially — the two clusters now map to
**different** labels (positive vs negative): the partition is genuinely
sentiment-aligned rather than degenerate. Held-out by label: neg 50.7%, pos
62.5%.

Honest read: steering **works qualitatively** (it breaks the degenerate topic
split), but the gain is **modest and saturates** (identical at axis_weight
0.7 / 0.9 / 0.95) because the *cosine* sentiment score on MiniLM is a weak
signal (train std ≈ 0.057). This is exactly why the live re-embed keeps an
**LLM-scoring fallback** for axes the embedding model doesn't capture
geometrically — that path (non-deterministic, costs LLM calls) would likely do
better. The strong generalization claim stays 20NG (87.0%, real ground truth);
this is a secondary, steered-case data point.

## Results

| Check | Result |
|---|---|
| 20NG initial clustering vs ground truth | 87.9% purity, topic names |
| Generalization held-out accuracy (20NG, seed-42) | 87.3% (262/300) after n_init=20 fix |
| Multi-seed robustness 20NG (10 seeds) | 86.9% ± 0.2 pp, range 86.7–87.3% — STABLE |
| Multi-seed robustness Amazon (10 seeds) | 50.7% ± 0.0 pp — deterministically base rate |
| Amazon contrast (vs sentiment) | 50.7% = base rate (topic ≠ sentiment) |
| Amazon, sentiment-axis re-embed (steered) | 56.7% (CI [51.0, 62.2]); clusters now sentiment-aligned |
| Oracle-refined generalization | 88.3%, Δ not significant (McNemar p=0.34) |
| Loop verification (merge/split/move) on 20NG | all invariants hold |
| Live API session (merge/split/move + Gemini) | all correct end-to-end |
| `generalization.py` mutation test | argmin→argmax caught (3 tests fail) |
| Post-merge suite stabilization | 210p/1f/22e → all green |
| UMAP projection module | 10 unit tests; merge snapshot → new cluster, 0 unassigned |
| UMAP blank-plot fix (SVG) | 12 reopens → 1200 points each time |
| UMAP stale fix (no-store) | merge → endpoint returns new turn live |
| Full test suite (post-merge + UMAP + P1 #51) | 248 pass, 0 fail, 0 errors |

## Issues opened this sprint

#42 (`f_eval` Gemini crash — already fixed by P3), #43 (rename/merge/split
descriptions — P3, fixed), #47 (token/cost/load to UI — me), #48 (prompts
dataset-agnostic — P4), #49 (loop verification — closed), #50 (eval scenarios
for 20NG — P5), #51 (datasets API record count — P1, merged), #52
(generalization — closed), #53 (UMAP clustering-evolution viz — me, delivered).

## Honest scope notes (for the report)

- **Determinism boundary**: the generalization numbers are deterministic
  because they depend only on embeddings + k-means. The full conversational
  system (with LLM oracle interpretation) introduces non-determinism we have
  NOT measured — that's a separate layer (#50/LLM-as-oracle, P5/P3).
- **Majority-vote metric** is clean only because k = #categories (bijection);
  with a different k the accuracy isn't directly comparable.
- **Amazon generalization** as a real accuracy is not meaningful (no topic
  ground truth); it would need oracle/LLM-judge validation (`f_validate_point`,
  B4) on a sample. Deferred — the strong claim is 20NG with real ground truth.
- **UMAP is a qualitative aid, not a metric**: it is a non-linear 2-D projection,
  so absolute distances/areas are not faithful and the layout is fit on the
  *original* embeddings — on semantic-re-embed turns the partition is shown
  cutting across the topic layout, not in the re-oriented geometry (see the
  geometry-aware item under Pending). Use it to *read* evolution, not to measure
  it; the quantified claims stay silhouette + purity + held-out accuracy.

## Pending / plan

- [x] Commit the verified-but-uncommitted files (#49 script, Option B script,
      P3 test fix) — done; @MatteoPareto notified.
- [x] Post-merge suite stabilization (22 errors + dry-run import-order race) —
      done, suite green.
- [x] UMAP viz (#53) — delivered + verified (Phase 1 + Phase 2 + 3 bug fixes).
- [ ] **Push** the local UMAP SVG-render fix (`6401a5f`, ahead 1) and **notify
      @ariannaschiavi-AIS** — the UMAP modal lives in her `ui/index.html`.
- [ ] (Optional, UMAP polish) 23-cluster sessions overflow the legend and the
      12-colour palette repeats — cap / group small clusters or widen the palette.
- [ ] (Optional) Geometry-aware UMAP: a second layout fit on the hybrid (D+1)
      re-embed space to show the re-oriented geometry. Deferred (needs an LLM
      pole call per session); the parallel axis-arrow overlay partly covers this.
- [x] Multi-seed robustness (`run_generalization_multiseed.py`, `n_init=20` fix) —
      done; 20NG stable at 86.9% ± 0.2 pp across 10 seeds; also fixed a real
      bug (degenerate partition at seeds 2+7 with n_init=10/auto).
- [x] Amazon generalization on the **sentiment axis** (steered re-embed) —
      done, 50.7% → 56.7% (`run_generalization_amazon_axis.py`, deterministic).
- [ ] (Optional) Stronger Amazon sentiment generalization via the **LLM-scoring**
      re-embed path (what the live system uses) — would likely beat the cosine
      56.7%, but is non-deterministic and costs LLM calls.
- [ ] (Optional) Amazon generalization via `f_validate_point` on a sample, if we
      want yet another data point (weaker, no ground truth).
- [ ] Coordinate with P5 on folding these into the final report / notebook
      (turns-to-convergence, generalization accuracy, the topic-vs-sentiment
      contrast, and the UMAP figures are all report-ready).
