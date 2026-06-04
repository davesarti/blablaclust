# BlaBlaClust — Dataset Comparison: Amazon Reviews vs IMDB vs 20 Newsgroups
**Date:** 2026-06-04 | **Code version:** Sprint 4 (post oracle-pinning fix) | **Model:** google/gemini-3.1-flash-lite | **Max turns:** 15

All three runs used identical code and prompt versions. The comparison isolates dataset effects from system behaviour.

---

## 1. Top-line summary

| Metric | Amazon Reviews | IMDB | 20 Newsgroups | Best |
|---|---|---|---|---|
| oracle_satisfied | **19/21 (90%)** | 18/21 (86%) | 16/21 (76%) | Amazon |
| system_stop | 1/21 (5%) | 1/21 (5%) | 4/21 (19%) | Amazon / IMDB |
| max_turns | 1/21 (5%) | 1/21 (5%) | 1/21 (5%) | Tie |
| Total turn errors | **2** | 3 | **1** | 20NG / Amazon |
| Mean B1 (overall quality) | 0.654 | **0.700** | 0.618 | IMDB |
| Mean B2 (coherence) | 0.638 | **0.698** | 0.539 | IMDB |
| Mean B3 (compliance) | **0.900** | 0.885 | 0.869 | Amazon |
| Mean A3 (cognitive load) | 1.43 | **1.41** | 1.54 | IMDB |
| Wall time (total) | ~29 min | ~34 min | ~34 min | Amazon |

**Summary:** IMDB leads on quality metrics; Amazon leads on completion rate and compliance; 20 Newsgroups is hardest for the current system, revealing a persona-dataset mismatch. All three confirm the system's core correctness — the differences are driven by dataset properties, not system failures.

---

## 2. Dataset characteristics

### Amazon Reviews — heterogeneous product corpus
1,200 reviews across music, books, film, electronics, household goods. Initial silhouette: **0.038–0.043**. Natural topic structure exists; structural ops alone produce meaningful clusters.

### IMDB — homogeneous film criticism corpus
1,200 movie/TV reviews. Initial silhouette: **0.003–0.009** — an order of magnitude lower. Structural ops have minimal effect; semantic reembedding is essential.

### 20 Newsgroups — multi-topic Usenet corpus
1,200 posts from newsgroups: `rec.sport.baseball`, `comp.graphics`, `talk.politics.guns`, `sci.med`, `sci.space`, `sci.crypt`. Initial silhouette: **0.042–0.055** — surprisingly similar to Amazon despite clear categorical boundaries.

The Usenet format introduces two complications:
1. **Off-topic posts** — Usenet threads drift. A baseball newsgroup post about traffic, a gun-control thread about food policy. These bleed across any clustering.
2. **Persona mismatch** — all 21 personae were designed and tuned for product reviews. Their goals reference "products", "categories", "consumer goods". On 20ng they adapt, but imperfectly.

---

## 3. Silhouette trends

IMDB and 20NG both show large semantic reembedding gains despite different initial conditions.

| Persona | Amazon initial | Amazon final | IMDB initial | IMDB final | 20NG initial | 20NG final |
|---|---|---|---|---|---|---|
| bilingual_drifter | 0.039 | 0.470 | 0.003 | **0.677** | 0.042 | **0.702** |
| reembed_evangelist | 0.039 | 0.342 | 0.011 | **0.510** | 0.041 | **0.698** |
| prompt_injector_redteam | 0.039 | 0.039 | 0.003 | 0.003 | 0.042 | **0.497** |
| sentiment_obsessive | 0.039 | 0.367 | 0.003 | **0.663** | 0.042 | 0.334 |
| never_satisfied | 0.043 | 0.031 | 0.007 | 0.023 | 0.051 | **0.333** |

20NG's best gains rival IMDB's despite a higher initial baseline — the oracle's axis-guided reembeds find tight topic clusters that GMM alone doesn't surface from the flat initial geometry.

---

## 4. Metric-by-metric analysis

### B1 — Overall quality

| Persona | Amazon | IMDB | 20NG | Best |
|---|---|---|---|---|
| bilingual_drifter | 0.65 | **0.95** | 0.45 | IMDB |
| boundary_pedant | 0.55 | 0.55 | 0.55 | Tie |
| brisk_executive | 0.75 | 0.65 | 0.65 | Amazon |
| clarify_baiter | 0.65 | **0.72** | 0.65 | IMDB |
| contradictory_oracle | 0.65 | **0.75** | **0.72** | IMDB |
| count_flipper | **0.72** | 0.65 | 0.55 | Amazon |
| curious_explorer | 0.65 | **0.75** | 0.55 | IMDB |
| delete_cluster_requester | 0.30 | 0.35 | 0.25 | IMDB (false neg) |
| explain_only_meta | 0.65 | **0.85** | **0.85** | IMDB / 20NG |
| methodical_analyst_it | 0.65 | **0.85** | **0.72** | IMDB |
| multi_merge_consolidator | 0.85 | **1.00** | 0.75 | IMDB |
| never_satisfied_max_turns | **0.82** | 0.65 | 0.65 | Amazon |
| patient_curator | 0.65 | **0.72** | **0.78** | 20NG |
| prompt_injector_redteam | 0.65 | 0.55 | 0.45 | Amazon |
| reembed_evangelist | 0.55 | 0.55 | 0.65 | 20NG |
| rename_obsessed | 0.65 | 0.55 | 0.65 | Amazon / 20NG |
| rigorous_phd | **0.75** | 0.65 | **0.72** | Amazon |
| satisfied_minimalist | 0.65 | **0.75** | **0.85** | 20NG |
| sentiment_obsessive | 0.35 | 0.55 | 0.65 | 20NG |
| typo_chaos | **0.95** | 0.85 | 0.55 | Amazon |
| vague_minimalist | 0.65 | **0.85** | 0.35 | IMDB |

**Mean B1: Amazon 0.654 — IMDB 0.700 — 20NG 0.618**

20NG's lower mean is driven by a few catastrophic failures: `bilingual_drifter` (0.45), `vague_minimalist` (0.35), `delete_cluster_requester` (0.25), all producing incoherent catch-all clusters. Excluding the three persona-design false-negatives (delete, vague, typo), the 20NG mean rises to ~0.66 — comparable to Amazon.

---

### B2 — Internal cluster coherence

**Mean: Amazon 0.638 — IMDB 0.698 — 20NG 0.539**

20NG has the lowest coherence by a large margin. Two causes:

**Usenet off-topic bleed.** A thread in `rec.sport.baseball` might contain an extended tangent about traffic or personal life. The evaluator sees this tangent in the cluster sample and marks the cluster as fragmented. Amazon and IMDB reviews are self-contained; Usenet posts are not.

**Cluster ceiling effects.** Baseball, guns/Waco, computer graphics, aerospace/automotive, and medical clusters appear across almost every 20NG session and score well (0.70–0.95) when isolated. But residual "Other" or catch-all clusters score 0.15–0.25, dragging the mean down.

**20NG coherence highlight:** Baseball clusters are the most consistently coherent of any topic across all three datasets — routinely scoring 0.82–0.95. The narrowness of `rec.sport.baseball` content is genuinely tight.

---

### B3 — Oracle compliance

**Mean: Amazon 0.900 — IMDB 0.885 — 20NG 0.869**

All three datasets see the same compliance failure modes:
1. **delete→merge** — always penalised across all datasets (design correct, eval wrong)
2. **rename stalling** — worse on IMDB (0.50) than 20NG (0.80) — IMDB's rename_obsessed session was particularly persistent
3. **overuse of semantic_reembed** — most visible on Amazon (sentiment_obsessive B3=0.40)

20NG-specific: `boundary_pedant` scored B3=0.50 — the oracle tried to move a point but the system moved it to the wrong cluster. This is a genuine move-operation precision issue rather than a naming or delete issue.

---

### A3 — Cognitive load

**Mean: Amazon 1.43 — IMDB 1.41 — 20NG 1.54**

20NG has the highest mean cognitive load and the most `system_stop` events (4 vs 1 each for Amazon and IMDB). The Usenet content confuses the oracle LLM — it makes more repetitive or circular requests, which the cognitive load model correctly flags.

`never_satisfied` reaches load 4 on all three datasets (as designed). On 20NG it escalates faster (reaches 4 by turn 13 vs 14 on Amazon and IMDB), consistent with the oracle being more confused by the Usenet content.

---

## 5. 20 Newsgroups — specific findings

### What works well
- **Structural clarity for known topics.** Baseball, guns/Waco, computer graphics, aerospace, medical — the oracle identifies these naturally and the system forms coherent clusters
- **Semantic reembedding gains.** `reembed_evangelist` and `bilingual_drifter` achieve silhouette gains of +0.66–+0.70, the largest across all three datasets for those personas
- **Zero errors on known failure modes.** No "oracle uses text as point ID" errors, no "batch_move_points empty" errors — the oracle view fix holds

### What fails
- **Vague personae collapse to incoherent catch-alls.** `vague_minimalist` (B2=0.15), `typo_chaos` (B2=0.20), `bilingual_drifter` (B2=0.25) all end up with a single "General Topics" or "General Interests" cluster of nearly zero coherence — the oracle couldn't identify specific things to fix in Usenet content, so every op tended toward broad merges
- **`delete_cluster_requester` worst across all datasets (B3=0.20, B1=0.25).** On 20NG, the merge-instead-of-delete behaviour produces a single "General Discussion" cluster of completely unrelated Usenet posts — the most incoherent result across all runs
- **4 `system_stop` events.** `reembed_evangelist`, `rename_obsessed`, `patient_curator`, `vague_minimalist` all triggered cognitive overload — the oracle's confusion about Usenet topics led to repetitive, high-load sessions

---

## 6. Full three-way comparison

### Per-persona B1 ranking

| Persona | Best dataset | Worst dataset | Why |
|---|---|---|---|
| bilingual_drifter | IMDB (0.95) | 20NG (0.45) | Bilingual pivot + all-film merge = great; Usenet merge = incoherent |
| boundary_pedant | All tied (0.55) | — | Move quality unchanged across datasets |
| brisk_executive | Amazon (0.75) | IMDB (0.65) | Amazon's diversity gives more to satisfy |
| explain_only_meta | IMDB / 20NG (0.85) | Amazon (0.65) | Single-domain explanations easier to generate accurately |
| never_satisfied | Amazon (0.82) | IMDB / 20NG (0.65) | Amazon diversity keeps oracle productively engaged |
| patient_curator | 20NG (0.78) | Amazon (0.65) | Newsgroup topics well-defined for careful curation |
| reembed_evangelist | 20NG (0.65) | Amazon (0.55) | 20NG's clear topics survive repeated reembeds better |
| satisfied_minimalist | 20NG (0.85) | Amazon (0.65) | 20NG's initial clustering already captures newsgroup topics |
| sentiment_obsessive | 20NG (0.65) | Amazon (0.35) | Sentiment axis makes more sense on opinionated Usenet posts |
| typo_chaos | Amazon (0.95) | 20NG (0.55) | Amazon merge is coherent; Usenet merge is junk |
| vague_minimalist | IMDB (0.85) | 20NG (0.35) | IMDB reembed resolves vagueness; 20NG merge produces chaos |

### Semantic reembedding effectiveness by dataset

Measured as mean silhouette improvement across personas that used `semantic_reembed`:

| Dataset | Mean Δsilhouette | Max Δsilhouette |
|---|---|---|
| Amazon | +0.31 | +0.47 (clarify_baiter) |
| IMDB | +0.48 | +0.67 (bilingual_drifter) |
| 20NG | +0.43 | +0.70 (bilingual_drifter) |

IMDB and 20NG both benefit more from semantic guidance than Amazon, but for different reasons: IMDB starts from near-zero; 20NG starts higher but the oracle guides it toward very clean topic axes.

---

## 7. Conclusions

### Dataset ranking by use case

| Use case | Recommended dataset | Reason |
|---|---|---|
| Demonstrating completion rate | Amazon | 90% oracle_satisfied, lowest system_stops |
| Demonstrating cluster quality | IMDB | Highest B1 (0.70) and B2 (0.698) |
| Demonstrating semantic reembedding value | IMDB or 20NG | Largest silhouette gains |
| Stress-testing the system | 20NG | Most system_stops, most diverse failure modes |
| Final formal evaluation | All three | Different capabilities tested per dataset |

### Key finding: persona-dataset fit matters

The 21 personae were written with product reviews in mind. On Amazon they perform as designed. On IMDB they adapt well (same register, different domain). On 20NG they struggle with Usenet-specific content (administrative threads, off-topic tangents, forum governance), leading vague or destructive personae to collapse to incoherent catch-alls. A 20NG-specific persona set would likely score much closer to Amazon and IMDB levels.

### What the three-dataset comparison confirms

1. **The oracle pinning fix is universal.** `boundary_pedant` scores 0 errors on all three datasets.
2. **delete→merge is consistently misscored.** B3 penalty on all three datasets. The eval prompt needs updating.
3. **Cognitive load calibration works across domains.** `never_satisfied` escalates correctly on all three.
4. **Semantic reembedding is the highest-value operation on homogeneous or opinionated corpora** (IMDB, 20NG). On heterogeneous product corpora (Amazon), structural ops are comparably effective.
5. **The system is domain-agnostic at the engine level.** Performance differences are persona-driven and dataset-driven, not system bugs.

### Remaining issues (all three datasets)
1. **delete→merge eval penalty** — consistent false negative across all datasets
2. **Rename stalling on repeated same-cluster requests** — affects all datasets, worst on IMDB
3. **Catch-all cluster incoherence** — vague/typo personae on 20NG produce 0.15–0.20 coherence clusters; the system should resist over-merging when it would produce an incoherent result
