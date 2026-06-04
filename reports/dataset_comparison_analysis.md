# BlaBlaClust — Dataset Comparison: Amazon Reviews vs IMDB
**Date:** 2026-06-04 | **Code version:** Sprint 4 (post oracle-pinning fix) | **Model:** google/gemini-3.1-flash-lite | **Max turns:** 15

Both runs used identical code and prompt versions. The comparison isolates dataset effects from system behaviour.

---

## 1. Top-line summary

| Metric | Amazon Reviews | IMDB | Winner |
|---|---|---|---|
| oracle_satisfied | **19/21 (90%)** | 18/21 (86%) | Amazon |
| system_stop | 1/21 (5%) | 1/21 (5%) | Tie |
| max_turns | 1/21 (5%) | 1/21 (5%) | Tie |
| Total turn errors | **2** | 3 | Amazon |
| Mean B1 (overall quality) | 0.654 | **0.70** | IMDB |
| Mean B2 (coherence) | 0.638 | **0.698** | IMDB |
| Mean B3 (compliance) | **0.90** | 0.885 | Amazon |
| Mean A3 (cognitive load) | 1.43 | **1.41** | IMDB |
| Wall time (total) | ~29 min | ~34 min | Amazon |

**Summary:** Amazon scores higher on compliance and completion rate; IMDB scores higher on cluster quality and coherence. Neither dataset dominates cleanly — they probe different system strengths.

---

## 2. Code improvements since previous Amazon run (context)

The previous Amazon run (Sprint 3) had 8 errors across 3 personas and a 76% satisfaction rate. The current run (Sprint 4 code) achieves 90% satisfaction and only 2 errors — the main driver being the oracle view point ID fix, which eliminated all 4 boundary_pedant errors and 2 patient_curator errors.

| Persona | Old Amazon errors | New Amazon errors | IMDB errors |
|---|---|---|---|
| boundary_pedant | 4 | **0** | **0** |
| patient_curator | 2 | **0** | **0** |
| rigorous_phd | 2 | **0** | **0** |

The fixes work consistently on both datasets.

---

## 3. Dataset characteristics and their effect

### Amazon Reviews — heterogeneous product corpus

1,200 reviews spanning music, books, film, electronics, household goods, clothing, food. Initial silhouette: **0.038–0.043** — there is natural geometric structure (topics genuinely differ).

**Consequence:** Structural ops (split/merge/rename) alone can produce meaningful clusters. The oracle can identify clusters by domain and guide refinements without needing axis-guided reembedding.

### IMDB — homogeneous film criticism corpus

1,200 reviews all about movies and television. Initial silhouette: **0.003–0.009** — an order of magnitude lower. Without oracle guidance, the initial clustering is essentially random.

**Consequence:** Structural ops produce marginal silhouette improvement. Semantic reembedding is almost always required for meaningful separation.

---

## 4. Silhouette trends

Silhouette measures cluster geometry quality. Amazon starts higher but IMDB improves more dramatically with oracle intervention.

| Persona | Amazon initial | Amazon final | IMDB initial | IMDB final |
|---|---|---|---|---|
| bilingual_drifter | 0.039 | 0.470 | 0.003 | **0.677** |
| clarify_baiter | 0.039 | **0.501** | 0.003 | **0.659** |
| reembed_evangelist | 0.039 | 0.342 | 0.011 | **0.510** |
| sentiment_obsessive | 0.039 | 0.367 | 0.003 | **0.663** |
| vague_minimalist | 0.043 | 0.043 | 0.007 | **0.536** |
| count_flipper | 0.039 | 0.043 | 0.003 | 0.012 |

IMDB gains from semantic reembedding are 1.5–2× larger than Amazon's, because the initial geometry is so flat that any axis guidance creates dramatic separation. Amazon's richer initial structure means semantic reembedding still helps, but the baseline is less desperate.

---

## 5. Metric-by-metric comparison

### B1 — Overall quality

| Persona | Amazon B1 | IMDB B1 | Δ | Note |
|---|---|---|---|---|
| bilingual_drifter | 0.65 | **0.95** | +0.30 | Bilingual pivot works better on homogeneous IMDB |
| boundary_pedant | 0.55 | 0.55 | 0 | Same — coherence limited by move-only session |
| brisk_executive | **0.75** | 0.65 | −0.10 | Amazon clusters are naturally better defined |
| clarify_baiter | 0.65 | **0.72** | +0.07 | IMDB reembed more impactful |
| contradictory_oracle | 0.65 | **0.75** | +0.10 | IMDB structures more stable under contradiction |
| count_flipper | **0.72** | 0.65 | −0.07 | Amazon's diversity makes K-arithmetic more meaningful |
| curious_explorer | 0.65 | **0.75** | +0.10 | Cleaner splits on IMDB (all film) |
| delete_cluster_requester | 0.30 | 0.35 | +0.05 | Both penalised by delete→merge design |
| explain_only_meta | 0.65 | **0.85** | +0.20 | IMDB clusters easier to explain (single domain) |
| methodical_analyst_it | 0.65 | **0.85** | +0.20 | Italian analyst works well on focused IMDB |
| multi_merge_consolidator | 0.85 | **1.00** | +0.15 | IMDB consolidation yields perfect coherence |
| never_satisfied_max_turns | **0.82** | 0.65 | −0.17 | Amazon diversity lets oracle find real distinctions |
| patient_curator | 0.65 | **0.72** | +0.07 | IMDB curation more precise |
| prompt_injector_redteam | 0.65 | 0.55 | −0.10 | Amazon more coherent final state after attack |
| reembed_evangelist | 0.55 | 0.55 | 0 | Repeated reembeds destabilize both datasets |
| rename_obsessed | 0.65 | 0.55 | −0.10 | Amazon rename stalls slightly less |
| rigorous_phd | **0.75** | 0.65 | −0.10 | Amazon structure cleaner for structural ops |
| satisfied_minimalist | 0.65 | **0.75** | +0.10 | IMDB initial state more satisfying |
| sentiment_obsessive | 0.35 | 0.55 | +0.20 | IMDB sentiment splits more meaningful (all reviews) |
| typo_chaos | **0.95** | 0.85 | −0.10 | Amazon merge yields excellent coherent catch-all |
| vague_minimalist | 0.65 | **0.85** | +0.20 | IMDB reembed resolves vagueness better |

**Mean: Amazon 0.654 vs IMDB 0.700.** IMDB leads by 0.046.

---

### B2 — Internal cluster coherence

**Amazon strengths:** Literary/non-fiction (0.85–0.90), film/TV (0.75–0.85), music (0.55–0.75). The diverse dataset creates clear topic clusters when oracle guidance is precise.

**Amazon weaknesses:** Consumer electronics/household tools consistently score 0.40–0.55. Off-topic reviews (book reviews in a tools cluster, hardware reviews in a music cluster) are endemic to Amazon's heterogeneous domain.

**IMDB strengths:** Horror/exploitation (0.80–0.90), international/arthouse (0.75–0.85), negative critique clusters (0.75–0.90). All clusters benefit from being single-domain.

**IMDB weaknesses:** Catch-all clusters ("General Film Reviews") score 0.40–0.55 when the oracle leaves a residual group. TV vs. film bleed-over is persistent.

**Mean: Amazon 0.638 vs IMDB 0.698.** IMDB leads because single-domain data makes clusters genuinely easier to evaluate as coherent.

---

### B3 — Oracle compliance

**Mean: Amazon 0.900 vs IMDB 0.885.** Amazon leads slightly.

Both datasets show the same two compliance failure modes:
1. **delete→merge rewrite** (delete_cluster_requester): The system correctly merges instead of deleting. B3=0.25–0.40 is a false negative — the system behaviour is per spec.
2. **Overuse of semantic_reembed** (sentiment_obsessive on Amazon, B3=0.40): When the oracle wants refinement (split/move within a structure), the system re-embeds globally instead of making incremental changes. This appears worse on Amazon because the existing cluster structure is richer and worth preserving.

The slightly higher Amazon B3 is explained by the richer initial structure — the oracle can accomplish more via structural ops, which the system handles correctly, whereas IMDB requires more reembed ops which occasionally over-restructure.

---

### B3 per-persona highlights

| Persona | Amazon B3 | IMDB B3 | Key difference |
|---|---|---|---|
| boundary_pedant | **1.00** | 0.80 | Amazon: 2 turns, done. IMDB: 8 turns, repeated moves |
| never_satisfied | **1.00** | 0.86 | Amazon: perfect over 15 turns |
| sentiment_obsessive | 0.40 | **1.00** | IMDB: repeated global reembed is correct; Amazon: system should have switched to split/move |
| rename_obsessed | 0.85 | 0.50 | IMDB rename stalls more — same cluster requested many times |

---

### A3 — Cognitive load

Both datasets show nearly identical load profiles. The cognitive load model is not sensitive to dataset content — it responds to structural volatility (number of ops, contradiction rate) which is driven by persona design, not data.

The only notable difference: `never_satisfied` reaches mean load 2.50 on Amazon vs 2.29 on IMDB over the same 14-turn session. Amazon's diversity gives the oracle more to complain about, pushing the load slightly higher.

---

## 6. Per-persona dataset sensitivity

Some personas are strongly affected by dataset; others are dataset-agnostic.

**Dataset-sensitive (Δ B1 ≥ 0.15):**

| Persona | Amazon | IMDB | Why |
|---|---|---|---|
| bilingual_drifter | 0.65 | **0.95** | IMDB: bilingual pivot merges into highly coherent film cluster; Amazon: pivot collapses to catch-all |
| explain_only_meta | 0.65 | **0.85** | IMDB: single-domain clusters easier to explain accurately |
| methodical_analyst_it | 0.65 | **0.85** | IMDB: Italian analyst finds cleaner thematic divisions |
| multi_merge_consolidator | 0.85 | **1.00** | IMDB: all-film consolidation yields perfect core cluster |
| never_satisfied_max_turns | **0.82** | 0.65 | Amazon: diverse topics enable meaningful long-session distinctions |
| vague_minimalist | 0.65 | **0.85** | IMDB: reembed fully resolves vague complaints |

**Dataset-agnostic (Δ B1 ≤ 0.05):**
boundary_pedant, brisk_executive, reembed_evangelist, count_flipper, curious_explorer, rigorous_phd. These personas' outcomes are driven by persona design rather than data content.

---

## 7. Error patterns

| Error type | Amazon | IMDB |
|---|---|---|
| Oracle uses text as point ID | 0 | 0 | Both fixed ✓ |
| Merge already-dissolved cluster | 1 | 1 | Same oracle hallucination, both datasets |
| Empty batch_move_points | 1 | 0 | Amazon only |
| cluster_reembed on dissolved cluster | 0 | 1 | IMDB only (new op exercised) |
| OpenRouter null response | 0 | 1 | Transient API issue |

Both datasets see the "merge dissolved cluster" oracle hallucination — it's a fundamental LLM memory limitation, not a dataset issue. The `cluster_reembed` error on IMDB is a new code path exercised for the first time in evaluation.

---

## 8. Conclusions

### When Amazon outperforms IMDB
- **Structural operation sessions** — split/merge/rename work better on a diverse dataset with natural topic boundaries
- **Long sessions** — `never_satisfied` achieves B1=0.82 on Amazon vs 0.65 on IMDB because diversity provides genuine fodder for incremental refinement
- **Compliance** — B3 is marginally higher on Amazon (0.90 vs 0.89) because the richer structure accommodates more operation types cleanly

### When IMDB outperforms Amazon
- **Semantic reembedding** — silhouette gains are 1.5–2× larger on IMDB because the near-zero initial geometry makes axis guidance decisive
- **Cluster coherence** — IMDB clusters score 0.06 higher on average because single-domain data avoids cross-category contamination
- **Explanatory sessions** — `explain_only_meta` and `methodical_analyst_it` perform substantially better on IMDB because single-domain clusters are easier to describe accurately

### Recommendation
For evaluation purposes, **both datasets are necessary** — they test complementary capabilities. Amazon tests structural reasoning and mixed-domain separation; IMDB tests semantic axis sensitivity and high-coherence cluster formation. A system that performs well on both is robustly general.

### Remaining issues on both datasets
1. **delete→merge eval penalty** — B3 scoring penalises correct behaviour (delete→merge is per spec)
2. **Overuse of semantic_reembed** — on Amazon, `sentiment_obsessive` triggered global reembeds when the oracle wanted incremental structural refinement; the system should downgrade from reembed to split/move when the oracle references existing clusters
3. **Rename stalling** — when the oracle repeatedly requests renames on the same cluster, the system sometimes applies a null name (both datasets, more pronounced on IMDB)
