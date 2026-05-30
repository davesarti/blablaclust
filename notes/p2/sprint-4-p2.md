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

## Results

| Check | Result |
|---|---|
| 20NG initial clustering vs ground truth | 87.9% purity, topic names |
| Generalization held-out accuracy (20NG) | 87.0% (261/300), CI [82.7, 90.3] |
| Amazon contrast (vs sentiment) | 50.7% = base rate (topic ≠ sentiment) |
| Oracle-refined generalization | 88.3%, Δ not significant (McNemar p=0.34) |
| Loop verification (merge/split/move) on 20NG | all invariants hold |
| Live API session (merge/split/move + Gemini) | all correct end-to-end |
| `generalization.py` mutation test | argmin→argmax caught (3 tests fail) |
| Full P2 test suite | 184 pass |

## Issues opened this sprint

#42 (`f_eval` Gemini crash — already fixed by P3), #43 (rename/merge/split
descriptions — P3, fixed), #47 (token/cost/load to UI — me), #48 (prompts
dataset-agnostic — P4), #49 (loop verification — closed), #50 (eval scenarios
for 20NG — P5), #51 (datasets API record count — P1), #52 (generalization —
closed).

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

## Pending / plan

- [ ] Commit the 3 verified-but-uncommitted files (#49 script, Option B script,
      P3 test fix) and notify @MatteoPareto about the test edit.
- [ ] (Optional) Amazon generalization via `f_validate_point` on a sample, if
      we want a second generalization data point (weaker claim).
- [ ] Coordinate with P5 on folding these numbers into the final report /
      notebook (turns-to-convergence, generalization accuracy, the topic-vs-
      sentiment contrast are all report-ready).
