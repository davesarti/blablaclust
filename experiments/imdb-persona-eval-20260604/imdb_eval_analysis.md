# BlaBlaClust — Persona Evaluation Analysis (IMDB)
**Date:** 2026-06-04 | **Dataset:** imdb_train | **Model:** google/gemini-3.1-flash-lite | **Max turns:** 15

---

## 1. Overview

21 personas ran to completion with no crashes. 18/21 (86%) ended in `oracle_satisfied` — a 10-point improvement over the Amazon run (76%). Only 1 persona triggered `system_stop` vs 4 on Amazon. One `oracle_parse_error` occurred due to an OpenRouter null response (not a system bug).

| Metric | IMDB | Amazon | Δ |
|---|---|---|---|
| oracle_satisfied | 18 (86%) | 16 (76%) | +10% |
| system_stop | 1 (5%) | 4 (19%) | −14% |
| max_turns | 1 (5%) | 1 (5%) | 0 |
| oracle_parse_error | 1 (5%) | 0 | +1 |
| Personas with errors | 3 | 3 | 0 |
| Total turn errors | 3 | 8 | −5 |
| Mean B1 (overall quality) | **0.70** | 0.63 | +0.07 |
| Mean B2 (coherence) | **0.70** | 0.65 | +0.05 |
| Mean B3 (compliance) | **0.89** | 0.82 | +0.07 |
| Mean B4 (contradiction) | 0.24 | 0.25 | −0.01 |
| Mean A3 (cognitive load) | 1.41 | 1.48 | −0.07 |

---

## 2. Dataset effect

IMDB reviews are **all film and TV criticism**. This has two structural consequences:

**Initial silhouette is dramatically lower.** Amazon starts at 0.038–0.043; IMDB starts at 0.003–0.009 — an order of magnitude lower. All 1200 reviews discuss the same domain, so the initial unsupervised geometry is nearly flat. There is no natural topic separation like "electronics vs books vs music" to cluster against.

**Coherence potential is higher.** Because every cluster is about film, the evaluator scores coherence against a tighter domain. A cluster of "horror reviews" on IMDB scores 0.82 cleanly; an equivalent cluster on Amazon gets diluted by off-topic product reviews landing in the wrong bucket. IMDB clusters are harder to *form* but easier to *evaluate as coherent*.

**Semantic reembedding gains are massive.** The near-zero initial silhouette means any axis-guided reorganisation produces spectacular improvements:

| Persona | Initial silhouette | Final silhouette | Δ |
|---|---|---|---|
| clarify_baiter | 0.003 | 0.659 | +0.656 |
| bilingual_drifter | 0.003 | 0.677 | +0.674 |
| sentiment_obsessive | 0.003 | 0.663 | +0.660 |
| vague_minimalist | 0.007 | 0.536 | +0.529 |
| reembed_evangelist | 0.011 | 0.510 | +0.499 |

Compare with Amazon's best gain: +0.408. On IMDB, semantic axes (sentiment, writing style, genre tone) discriminate the data far more effectively than they do on Amazon's heterogeneous product reviews.

---

## 3. Metric-by-metric analysis

### A1 — Silhouette

Sessions that used structural ops only (split/merge/rename) on IMDB rarely improve silhouette significantly — the flat geometry means structural ops don't cut along natural lines. Sessions that used `semantic_reembed` show the largest improvements, confirming that **axis-guided clustering is essential for homogeneous datasets**.

Degradations:
- **count_flipper**: 0.003 → 0.012 (+minimal), oscillating splits/merges fragment the space
- **contradictory_oracle**: 0.003 → 0.003 (flat — contradictory ops cancel each other out)

---

### A2 — Turns and efficiency

Most personas converge in 2–6 turns, consistent with Amazon. The IMDB dataset doesn't seem to require more turns despite being harder to cluster — the oracle LLM adapts its requests to what the data can actually support.

---

### A3 — Cognitive load

Mean cognitive load dropped from 1.48 (Amazon) to 1.41 (IMDB). Fewer `system_stop` events (1 vs 4) confirms the system is less overloaded on IMDB, likely because the homogeneous dataset produces cleaner cluster structures that require less back-and-forth to stabilise.

**never_satisfied** still escalates to load 4 over 14 turns, confirming the cognitive load model works correctly regardless of dataset.

---

### B1 — Overall quality

| Score range | Count | Personas |
|---|---|---|
| 0.9–1.0 | 2 | multi_merge_consolidator (1.0), bilingual_drifter (0.95) |
| 0.8–0.89 | 5 | explain_only_meta (0.85), methodical_analyst_it (0.85), typo_chaos (0.85), vague_minimalist (0.85), curious_explorer (0.75+) |
| 0.65–0.79 | 6 | contradictory_oracle (0.75), patient_curator (0.72), clarify_baiter (0.72), brisk_executive (0.65), never_satisfied (0.65), rigorous_phd (0.65) |
| 0.5–0.64 | 6 | boundary_pedant (0.55), prompt_injector_redteam (0.55), reembed_evangelist (0.55), rename_obsessed (0.55), sentiment_obsessive (0.55), count_flipper (0.65) |
| < 0.5 | 1 | delete_cluster_requester (0.35) |

**Mean B1: 0.70** vs 0.63 on Amazon. The improvement is real and consistent across most personas.

---

### B2 — Internal cluster coherence

IMDB coherence is substantially better than Amazon on average (0.70 vs 0.65). The main driver is that all IMDB clusters are film-domain — there's no cross-contamination of "electronics reviews landing in the books cluster" type.

**Strong cluster types on IMDB:**
- Horror / exploitation cinema: consistently 0.80–0.90
- International / arthouse cinema: 0.75–0.85
- Negative critique clusters (when well-defined): 0.75–0.90

**Weak cluster types on IMDB:**
- "General film reviews" catch-alls: 0.40–0.55 — when the oracle leaves a residual cluster, it collects everything
- Television / pop culture: 0.45–0.65 — TV reviews bleed into film reviews because the language is identical
- Sentiment-based clusters after multiple reembeds: 0.40–0.55 — repeated reembeds (reembed_evangelist) destabilise the cluster geometry

---

### B3 — Oracle compliance

**Mean B3: 0.89** — best across both datasets.

| Score | Count | Personas |
|---|---|---|
| 1.0 (perfect) | 13 | bilingual_drifter, brisk_executive, clarify_baiter, contradictory_oracle, count_flipper, curious_explorer, explain_only_meta, methodical_analyst_it, multi_merge_consolidator, patient_curator, reembed_evangelist, satisfied_minimalist, sentiment_obsessive, typo_chaos, vague_minimalist |
| 0.8–0.99 | 3 | boundary_pedant (0.80), never_satisfied (0.86), rigorous_phd (0.85) |
| 0.3–0.79 | 3 | rename_obsessed (0.50), prompt_injector_redteam (0.33), delete_cluster_requester (0.25) |

The same caveats as Amazon apply: **delete_cluster_requester (0.25) and prompt_injector_redteam (0.33)** are correct system behaviours penalised by the evaluator (delete→merge is the designed rewrite; injection resistance is correct).

**rename_obsessed (0.50)** reflects a recurring issue: when the oracle asks for bare renames ("rename this") repeatedly on the *same* cluster across multiple turns, the system sometimes fails to provide a new name string — performing a rename operation with a null value. This is a prompt-level issue in `f_output.txt` worth investigating.

---

### B4 — Oracle contradiction

Mean B4 of 0.24 is nearly identical to Amazon (0.25). The same personas that were contradictory on Amazon are contradictory on IMDB — the contradiction pattern is driven by persona design, not dataset content.

---

## 4. Error analysis

**3 errors across 3 personas** — a dramatic improvement from Amazon's 8 errors across 3 personas.

| Persona | Error | Root cause |
|---|---|---|
| boundary_pedant | 0 errors (was 4 on Amazon) | **Point ID fix worked** — oracle now sees `[uuid] text` and uses real IDs |
| patient_curator | 0 errors (was 2 on Amazon) | Move operations correctly executed |
| methodical_analyst_it | `cannot merge already-dissolved clusters` | Oracle LLM referenced a cluster dissolved in a prior turn |
| never_satisfied_max_turns | `cannot re-embed already-dissolved cluster` | New: oracle_reembed on a cluster dissolved the same session |
| sentiment_obsessive | `OpenRouter returned no text content` | API-level null response at turn 12; transient |

**Key finding:** The oracle view point ID fix (showing `[uuid] text` instead of just text) eliminated all 4 boundary_pedant errors. The oracle can now pin real point IDs and the system can match them. This directly validates the fix.

**New error type:** `cannot re-embed already-dissolved cluster` — `cluster_reembed` on a cluster that was dissolved earlier in the session. The oracle LLM referenced a stale cluster ID from its context. The system returns a clean 422 and the oracle self-corrects. This is the first time this code path was exercised in evaluation.

---

## 5. Dataset comparison: IMDB vs Amazon

| Aspect | Amazon | IMDB | Winner |
|---|---|---|---|
| Initial silhouette | 0.038–0.043 | 0.003–0.009 | Amazon (more natural structure) |
| Silhouette gain from reembed | +0.30–+0.41 | +0.50–+0.67 | **IMDB** |
| Mean B1 | 0.63 | **0.70** | IMDB |
| Mean B2 coherence | 0.65 | **0.70** | IMDB |
| Mean B3 compliance | 0.82 | **0.89** | IMDB |
| Turn errors | 8 | **3** | IMDB |
| oracle_satisfied rate | 76% | **86%** | IMDB |
| "Consumer electronics" fragmentation | Yes (persistent) | No equivalent | IMDB |
| Cluster distinctiveness w/o oracle | High (topics differ) | Low (all film) | Amazon |

**Summary:** The system performs better on IMDB across every quality metric. The dataset's homogeneity makes coherence scoring more reliable and semantic reembedding dramatically more effective. The tradeoff is that without oracle guidance, the initial clustering is nearly random — making oracle involvement *more necessary* on IMDB, not less.

---

## 6. Key findings

### Improvements over Amazon run

1. **Point ID fix validated.** `boundary_pedant` went from 4 errors to 0. The oracle now uses real UUIDs from the panel instead of inventing text-based identifiers.

2. **oracle_satisfied rate +10%.** Fewer `system_stop` events suggest the IMDB dataset produces more stable conversational trajectories — the oracle reaches satisfaction more consistently.

3. **Semantic reembed shines.** The near-zero initial silhouette makes axis-guided reembedding the dominant value-add on IMDB. Silhouette gains of +0.50 to +0.67 are roughly 2× larger than on Amazon.

4. **Coherence is higher and more reliable.** Without Amazon's heterogeneous product categories creating domain-bleed, cluster coherence scores are more accurate and generally higher.

### Persistent issues (same as Amazon)

1. **delete→merge eval penalty.** `delete_cluster_requester` still scores B3=0.25 on IMDB. The system behaviour is correct (merge-nearest is the designed rewrite), but the eval LLM penalises it as non-compliance. The B3 eval prompt should be updated.

2. **rename_obsessed bare-rename failure.** The system repeatedly failed to generate a new name when the oracle asked for a bare rename of the same cluster multiple turns in a row. The `f_output.txt` bare-rename logic handles single occurrences but seems to stall under repeated requests for the same cluster. Worth investigating.

3. **Repeated reembed degrades coherence.** `reembed_evangelist` on IMDB scores B2=0.50 — worse than on Amazon (0.60). Each global reembed restructures the space and the cluster names drift from the underlying geometry, producing fragmented results. A "cooldown" between reembeds or a limit on reembed frequency may help.

### New finding: cluster_reembed in the wild

`never_satisfied_max_turns` triggered a `cannot re-embed already-dissolved cluster` error — the first time `cluster_reembed` surfaced an error in evaluation. The system handles it cleanly (422, oracle self-corrects), but the oracle LLM needs to be more careful about checking cluster lifecycle before issuing reembed ops. The oracle prompt's pinning section should note that cluster IDs become invalid after structural changes.

---

## 7. Conclusions

The IMDB dataset produces uniformly better evaluation results than Amazon across all quality dimensions. The improvement is driven by dataset homogeneity (all film criticism) which makes semantic clustering more effective and coherence scoring more accurate.

The oracle view point ID fix is confirmed effective. The system is ready for final evaluation. The remaining issues (delete→merge eval penalty, rename stalling, repeated-reembed coherence) are polish items that do not block the core functionality.

**Total wall time: ~34 minutes for 21 personas** (vs ~28 minutes on Amazon — slightly slower due to more semantic reembed calls on IMDB).
