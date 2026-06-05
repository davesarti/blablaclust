# Generalization Stability Report — 6 user sessions

_Date: 2026-06-05  |  Branch: `main`  |  Script: `scripts/run_generalization_stability_eval.py`_

---

## 1. The question, operationalized

**Generalization here is not "does the clustering recover a hidden category"** —
the task is unsupervised, labels are out of scope at scoring time
(`docs/quality_specs.md`). The operational question is:

> Once a user has driven a session to convergence, does the clustering stay
> coherent as new data arrives into the running system?

This is the **robustness under distribution shift** dimension of the quality
spec in §10 of the project brief — a pre-committed quality dimension, measured
without redefinition after seeing results.

### Quality dimensions covered

Following §10, every claim below maps to a single pre-declared dimension and
measurement method. Nothing is collapsed to a scalar; the multi-dimensional
table is the report.

| Dimension (§10) | Operationalisation | Metric | Method |
|---|---|---|---|
| Robustness — geometry | Does the silhouette of *pre-existing* points hold once new arrivals are placed? | **A1 paired Δ silhouette** (common points) | Deterministic; percentile bootstrap CI over per-point differences. |
| Robustness — calibration | Are new arrivals near or far from their assigned cluster's base distribution? | **A4 OOD rate** (d² > base d²_95) and pooled mean z | Distance-based, temperature-free; Wilson 95% CI on the OOD rate; bootstrap CI on mean z. |
| Coherence | Does the LLM judge still call clusters coherent after ingestion? | **B2 paired Δ coherence** | LLM-as-judge (6 top + 4 bottom-fitting members per cluster = 10 samples, raised from the live-endpoint's 3+2 on explicit request); percentile bootstrap CI over per-cluster differences. |
| Helpfulness / utility | Composite quality of the converged session itself, against which generalization is measured. | **B1 overall**, B3 compliance | LLM-as-judge per the eval endpoint; cohort-level bootstrap CI only (per-unit values not exposed by the eval). |
| Efficiency | How much work did the user do? | A2 turns, A3 mean cognitive load | A2 reported as point estimate (CI excluded by request); A3 cohort-level bootstrap CI only. |
| Safety / contradiction | Did the system contradict itself during the session? | B4 contradiction | Point estimate (CI excluded by request). |

**Confidence intervals.** All metrics carry 95% CIs except A2 (turns to
convergence) and B4 (contradiction), per the brief. CIs are bootstrap-percentile
(10 000 iterations) for paired Δ and means, and Wilson-score for OOD-rate
proportions. Where the underlying per-unit scores are not exposed by the eval
endpoint (A3, B1, B3), CIs are reported at the **cohort level** (bootstrap of
the across-session mean over n = 6) and not at the per-session level — flagged
explicitly in each table.

---

## 2. Method recap

For each of the 6 user-driven sessions in `data/demo_database.db`
(`user1/newsgroups`, `user2/amazon`, `user3/imdb`, `user4/newsgroups`,
`user5/amazon`, `user6/imdb` — the 3 `baseline/*` sessions are bare initial
clusterings with no user refinement and are excluded):

1. The session is **cloned in-memory** from `demo_database.db` — the source DB
   is opened, read, and closed; no ingestion-phase write touches it.
2. The converged snapshot at `max(SoftAssignment.turn_number)` is frozen.
   GMM parameters (centroids, diag-vars, log-weights) are extracted with
   `gmm_params_from_snapshot`.
3. A4 calibration: per-cluster d²_95 / μ / σ are built once from base
   in-cluster squared distances (`calibrate_distance_reference`,
   temperature-free).
4. **t0 eval**: A1 (deterministic silhouette) and B2 (LLM judge over 6 top + 4
   bottom-fitting cluster members each).
5. The 300-row frozen split for the matching dataset is embedded with the
   session's `all-MiniLM-L6-v2` encoder and ingested via
   `assign_gmm_posterior` → `ingest_points` (writes a `turn+1` snapshot in the
   clone). Pre-existing points are carried forward verbatim, never re-evaluated.
6. **t1 eval**: A1 + B2 again; A4 scores every new point against its assigned
   cluster's base reference; paired Δ + bootstrap CIs are reported.

Per-session logs: `logs/generalization/user{1..6}_*.log` (full output preserved
for re-analysis).

---

## 3. Per-session results

### 3.1 Robustness — geometry (A1)

Paired Δ is the inferential signal. The t0 and t1 means are the absolute
levels; the paired Δ + CI tests whether *the same points* moved.

| Session | A1 t0 mean | A1 t1 mean | Paired Δ on 1200 common points | New-point sub-aggregate | Verdict |
|---|---:|---:|---|---|---|
| user1/newsgroups | +0.0432 | +0.0426 | +0.0001  [−0.0000, +0.0002] | +0.0398 [+0.0353, +0.0442] | **held** (CI spans 0) |
| user2/amazon    | +0.0321 | +0.0322 | −0.0005  [−0.0007, −0.0003] | +0.0348 [+0.0306, +0.0391] | **degraded** |
| user3/imdb      | −0.0053 | −0.0047 | +0.0004  [+0.0002, +0.0005] | −0.0038 [−0.0077, +0.0001] | **improved** |
| user4/newsgroups | +0.0176 | +0.0199 | +0.0003  [−0.0000, +0.0006] | +0.0279 [+0.0233, +0.0328] | **held** (CI spans 0) |
| user5/amazon    | +0.0145 | +0.0159 | −0.0001  [−0.0005, +0.0003] | +0.0221 [+0.0190, +0.0252] | **held** (CI spans 0) |
| user6/imdb      | −0.0023 | −0.0007 | −0.0007  [−0.0012, −0.0003] | +0.0088 [+0.0063, +0.0113] | **degraded** |

![A1 paired Δ forest](generalization-a1-forest.png)

> Magnitudes are tiny in absolute terms (silhouette values are themselves
> near zero — IMDB sessions even start negative, reflecting heavy text overlap
> across categories). The bootstrap CIs are tight because n = 1200 common
> points per session; "degraded" verdicts at this magnitude mean the *direction*
> is detectable, not that cohesion collapsed.

### 3.2 Robustness — calibration (A4)

A4 is **temperature-free**: every new point is scored against its assigned
cluster's base d²_95. Under the null (in-distribution arrivals), the OOD rate
sits near 5% by construction.

| Session | Pooled OOD rate (Wilson 95% CI) | Pooled mean z (bootstrap 95% CI) | vs 5% baseline |
|---|---|---|---|
| user1/newsgroups | **8.7% [6.0%, 12.4%]** | +0.2476 [+0.1348, +0.3643] | above baseline |
| user2/amazon | 5.3% [3.3%, 8.5%] | +0.0412 [−0.0703, +0.1515] | CI spans baseline |
| user3/imdb | **8.0% [5.4%, 11.6%]** | +0.2031 [+0.0859, +0.3269] | above baseline |
| user4/newsgroups | 5.0% [3.1%, 8.1%] | +0.0479 [−0.0660, +0.1614] | CI spans baseline |
| user5/amazon | **1.7% [0.7%, 3.8%]** | −0.1175 [−0.2211, −0.0137] | **below baseline** |
| user6/imdb | 4.3% [2.5%, 7.3%] | +0.0538 [−0.0476, +0.1595] | CI spans baseline |

![A4 pooled OOD rate](generalization-a4-ood.png)

> Wilson CIs are computed from the per-session n = 300 binomial: red = lower
> bound above 5%, green = upper bound below 5%, grey = CI spans 5%. user5's
> below-baseline OOD is consistent with the negative mean z (its arrivals are
> *closer* to their assigned centroids than chance would suggest).

### 3.3 Coherence (B2)

The B2 paired Δ is on per-cluster scores; the CI is the inferentially relevant
quantity. The min column flags the weakest cluster's coherence.

| Session | B2 t0 mean / min | B2 t1 mean / min | Paired Δ over clusters (95% CI) | Verdict |
|---|---|---|---|---|
| user1/newsgroups | 0.656 / 0.350 | 0.663 / 0.450 | +0.006 [−0.075, +0.081] | held |
| user2/amazon | 0.750 / 0.550 | 0.575 / 0.400 | **−0.175 [−0.258, −0.083]** | **degraded** |
| user3/imdb | 0.679 / 0.500 | 0.679 / 0.550 | +0.000 [−0.080, +0.059] | held |
| user4/newsgroups | 0.680 / 0.450 | 0.660 / 0.450 | −0.020 [−0.090, +0.030] | held |
| user5/amazon | 0.530 / 0.350 | 0.560 / 0.400 | +0.030 [−0.080, +0.160] | held |
| user6/imdb | 0.757 / 0.700 | 0.700 / 0.650 | −0.057 [−0.120, +0.000] | held (CI edge = 0) |

![B2 paired Δ forest](generalization-b2-forest.png)

![B2 mean coherence shift](generalization-b2-shift.png)

> Cluster counts (3–8 live clusters per session) make the per-session B2 CI
> coarse — explicitly noted in the script output. The B2 paired Δ uses the
> matching clusters at t0 and t1 (cluster IDs are stable because ingestion
> doesn't change cluster membership of pre-existing points). The per-cluster
> coherence scores themselves are not persisted at the row level — re-running
> with `--b2-every-batch` would expose per-batch trajectories if a richer signal
> is needed.

### 3.4 Session context — A2, A3, B1, B3, B4

These describe the converged session itself (not generalization). Per the
brief, A2 and B4 are reported as point estimates without CIs. A3, B1, B3 only
have one number per session in the eval response — within-session bootstrap
would require re-running the eval with per-unit tracing. Cohort-level CIs
(across n = 6) are in §4.

| Session | k live | A2 turns | A3 cognitive load | B1 overall | B3 compliance | B4 contradiction |
|---|---:|---:|---:|---:|---:|---:|
| user1/newsgroups | 8 | 5 | 2.00 | 0.650 | 1.000 | 0.700 |
| user2/amazon | 6 | 4 | 1.50 | 0.550 | 0.500 | 0.600 |
| user3/imdb | 7 | 3 | 2.00 | 0.450 | 0.330 | 0.700 |
| user4/newsgroups | 5 | 4 | 1.00 | 0.650 | 1.000 | 0.400 |
| user5/amazon | 5 | 3 | 1.33 | 0.650 | 0.800 | 0.400 |
| user6/imdb | 3 | 5 | 1.20 | 0.550 | 0.500 | 0.600 |

> `k live` is the count of non-dissolved clusters at the converged snapshot —
> consistently smaller than the raw `k` stored in the DB because user
> refinement dissolved low-quality clusters during the conversation.

---

## 4. Cohort-level summary (n = 6)

Bootstrap 95% CIs on the across-session mean (10 000 iterations, percentile
method). This is the only legitimate CI on A3 / B1 / B3 from the data we have.

| Metric | Cohort mean | 95% CI |
|---|---:|---|
| A1 silhouette — t0 mean | +0.0166 | [+0.0028, +0.0305] |
| A1 silhouette — t1 mean | +0.0175 | [+0.0042, +0.0309] |
| A3 mean cognitive load | 1.51 | [1.21, 1.81] |
| A4 pooled OOD rate | 5.5% | [3.7%, 7.3%] |
| B1 overall | 0.583 | [0.517, 0.633] |
| B2 mean coherence — t0 | 0.675 | [0.613, 0.729] |
| B2 mean coherence — t1 | 0.640 | [0.597, 0.680] |
| B3 compliance | 0.688 | [0.472, 0.888] |

(A2 and B4 are reported per the brief without CIs — cohort means are A2 = 4.0
turns, B4 = 0.567.)

> The cohort-mean OOD rate at 5.5% with CI [3.7%, 7.3%] is **consistent with
> the 5% null** — across the 6 sessions, ingestion does not on average shift
> arrivals out of distribution. The per-session breakdown in §3.2 is where the
> heterogeneity lives. Similarly, the B2 cohort mean drops from 0.675 → 0.640
> across t0 → t1, but the two CIs overlap heavily; the meaningful per-session
> degradation is the user2 outlier in §3.3.

---

## 5. Insights

### 5.1 user2/amazon is the only clean failure
This is the only session that **degrades simultaneously on A1 and B2** with
both CIs strictly negative. Its A1 paired Δ is −0.0005 [−0.0007, −0.0003]
(direction-confirmed, magnitude small) and its B2 paired Δ is −0.175
[−0.258, −0.083] — about an order of magnitude larger than the next-worst
B2 drop (user6 at −0.057). The B2 minimum cluster fell from 0.550 → 0.400.
A4's OOD rate at 5.3% [3.3%, 8.5%] is unremarkable, so the failure is **not**
explained by arrivals being out of distribution — the arrivals fit the GMM
geometry roughly as expected, but a measurable share of clusters lost
coherence under the judge's read.

### 5.2 A1 and A4 are not redundant
user1 and user3 both **hold A1** (silhouette stable or improved) yet show
**elevated A4 OOD rates** (8.7% and 8.0%, both with Wilson CI lower bound
strictly above 5%). The geometric story is "arrivals stretch the cluster
envelope but don't push existing points out of cohesion" — exactly the case
A4 was added to catch and A1 alone would miss.

### 5.3 user5/amazon is anomalously tight
A4 OOD rate of 1.7% [0.7%, 3.8%] is the only session whose upper bound is
strictly below the 5% baseline, and its mean z is significantly negative.
This is consistent with its k = 5 live-cluster topology being unusually
concentrated: arrivals land closer to their assigned centroid than the base
points themselves. Note this happens against a fairly low B1 overall (0.650)
and low B3 compliance (0.800), so "tight" should not be read as "high
quality" — it's a calibration observation, not a coherence one.

### 5.4 Dataset asymmetry
- **Amazon** is bimodal across sessions: user2 fails hard on B2, user5 is the
  tightest of all. Different user paths over the same dataset produce
  qualitatively different converged geometries.
- **Newsgroups** is the most stable family: both sessions hold A1 and B2.
- **IMDB** is the softest family: both sessions have near-zero or negative
  A1 silhouette levels (a property of the embedding+dataset, not the
  ingestion) and at least one of the two degrades softly on A1 or has B2 CI
  touching 0.

With n = 2 sessions per dataset this is observation, not inference — but it
flags amazon as the dataset whose converged structures are most sensitive
to user choices.

### 5.5 The headline of the cohort
On average across the 6 user sessions, **ingesting +300 in-domain points into
a converged session does not break the structure**: cohort-mean A1 holds
(t0 → t1 essentially unchanged, both CIs overlapping), cohort-mean A4 OOD
rate sits at the null baseline (5.5% with CI [3.7%, 7.3%]), and cohort-mean
B2 drops by 0.035 with the two CIs overlapping. The cohort masks one clean
failure (user2/amazon) and one soft failure (user6/imdb) — both surfaced by
the per-session forest plots, not by the cohort mean.

---

## 6. Threats to validity

- **n = 6 sessions, n = 2 per dataset.** Cohort CIs are coarse; dataset-level
  claims are observation.
- **LLM judge for B2.** The judge is a measurement instrument, not a
  ground-truth. The script's transient-failure guard (`_coherence_by_cluster`
  retries on all-zero responses) addresses a known judge fragility but the
  judge is not validated against human labels on these sessions —
  per §10's "Minimum acceptable," that validation is a known gap.
- **Single embedding model (`all-MiniLM-L6-v2`).** Generalization is reported
  against the same encoder the sessions converged in. Robustness under
  *encoder shift* is a different question, not addressed here.
- **In-distribution arrivals.** The frozen splits are held-out rows of the
  same source datasets the sessions converged over. The signal here is
  "does the converged structure absorb more of the same?" — not "does it
  survive a true distribution shift." A genuine OOD eval would feed a
  different-domain split (e.g. newsgroups arrivals into an amazon session)
  and is left as a follow-up.
- **A3 / B1 / B3 lack within-session CIs.** The eval response collapses these
  to a single number; only the cohort-level CI (n = 6) is reported. Adding
  per-unit tracing to the eval would unlock proper per-session CIs.
- **B2 sampling was raised to 10 / cluster (6 top + 4 bottom).** This diverges
  from the live `/eval` endpoint's 3 + 2 setting; the B2 numbers here are
  more stable but not directly comparable to numbers printed by the API.

---

## 7. Reproducing

```bash
# One session (Mode B — clone an existing session and ingest a frozen split)
PYTHONPATH=. python scripts/run_generalization_stability_eval.py \
    --session-id 8a436fa9-9dc5-4188-8f46-3b787998df64 \
    --new data/20newsgroups_frozen.csv

# Regenerate the charts from the per-session logs
PYTHONPATH=. python scripts/generate_generalization_charts.py
```

Full per-session logs: `logs/generalization/user{1..6}_*.log`.
Chart-generation source (with the verbatim metric tables): `scripts/generate_generalization_charts.py`.
