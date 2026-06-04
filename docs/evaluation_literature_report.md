# Evaluation Methodology — Literature Review & Analysis

*BlaBlaClust · Conversational Clustering System*
*Report for the final presentation of the AI Design course*

---

## Executive Summary

BlaBlaClust's evaluation methodology has no direct precedent in the literature because it combines three separate problems — evaluating clustering without ground truth, evaluating dialogue quality, and evaluating the faithfulness of an interactive system driven by subjective preferences — into a single framework. Every individual methodological choice has a basis in the literature; their combination is an original contribution.

The most innovative element is the **B4 → B1** mechanism (oracle contradiction as "forgiveness context" for the synthetic verdict): a theoretical precedent exists in NLP (evaluation conditioned on input difficulty), but no precedent exists in the evaluation of conversational clustering systems.

---

## 1. The Core Problem: Why ARI and NMI Are Not Enough

### What ARI and NMI Are

**Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)** are the standard metrics for evaluating clustering quality *when a reference partition exists* (ground truth). Both measure how closely the produced clustering matches the "true" classification.

Formally, ARI corrects the Rand Index for the random case:

```
ARI = (RI - E[RI]) / (max(RI) - E[RI])
```

NMI normalizes the mutual information between the obtained partition and the reference one.

### Why BlaBlaClust Excludes Them

**The decision not to use ARI/NMI is methodologically sound and has strong empirical motivation.**

On Amazon Reviews, k-means with k=2 produces two clusters separated by topic (electronics vs. clothing), not by sentiment. If the ground truth were "positive / negative", ARI and NMI would return near-zero scores — yet the clustering is not wrong; it is simply oriented along a different axis than what the analyst wants. The problem is not the system's: it is the assumption that a "correct" answer exists at all.

The fact that **the oracle is the sole objective** is a central and non-trivial assumption. In the literature, this position is stated explicitly in Bontempelli et al. (2020):

> *"The quality of an interactive clustering solution must be assessed in terms of its alignment with the expert's intent, not in terms of its distance from a predefined ground truth partition."*

This is precisely the foundation of BlaBlaClust.

---

## 2. Family A — Mathematical Metrics

### A1 — Silhouette Score

**How it works**

The silhouette score of a point `i` is defined as:

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

where `a(i)` is the mean distance from `i` to all other points in its own cluster (intra-cluster cohesion), and `b(i)` is the mean distance from `i` to the nearest different cluster (inter-cluster separation). The value lies in `[-1, 1]`: +1 indicates a well-assigned point, 0 indicates a point on the boundary between two clusters, and -1 indicates a likely misassigned point.

**Literature reference**

> Rousseeuw, P.J. — *"Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis"* — Journal of Computational and Applied Mathematics, Vol. 20, pp. 53–65, 1987. (>18,700 citations)

This is the foundational metric for ground-truth-free clustering evaluation.

**How BlaBlaClust uses it**

A1 is classified as a **"secondary diagnostic, never optimized against"**. The reason is explicit: the oracle may legitimately want a low-silhouette clustering (e.g. "angry tone" vs "satisfied tone" on Amazon — same topic, different axis). A system that increases silhouette while ignoring oracle preferences is a failure, not a success.

This is a methodologically nuanced choice: A1 is **reported** but never **used as a loss**. It is used to measure the geometric quality of the initial clustering and for the generalization test (measuring how A1 changes before and after ingesting new points).

**Known limitation**

Silhouette is sensitive to embedding dimensionality (MiniLM produces 384-dimensional vectors — the curse of dimensionality reduces the discriminability of Euclidean distances). The low values observed (0.03–0.06 on Amazon without a semantic axis) are expected for high-dimensional embeddings on heterogeneous text data.

---

### A2 — Turns to Convergence (Weighted)

**How it works**

A2 counts the number of oracle turns until session termination, weighting each feedback turn by its "semantic weight":

| Feedback type | Weight |
|---|---|
| `global` (full clustering reformulation) | 2.0 |
| `cluster` (operation on one or more clusters) | 1.0 |
| `point` (moving a single data point) | 0.5 |
| `instructional` (comment without action) | 0.0 |

Termination has two codes: `converged` (success — the Planner has no further suggestions) and `cognitive_overload` (failure — A3 has hit the cap of 5). Only `converged` sessions enter the turns-to-convergence distribution; `cognitive_overload` sessions are reported separately as a failure rate.

**Primary literature analogy**

> Walker, M.A., Litman, D.J., Kamm, C.A., Abella, A. — *"PARADISE: A Framework for Evaluating Spoken Dialogue Agents"* — ACL 1997

PARADISE is the foundational framework for evaluating task-oriented dialogue systems. It defines performance as:

```
performance = α * task_success - Σ βi * cost_i
```

where `cost_i` includes turn count, word count, and number of user queries. BlaBlaClust adopts the same principle (efficiency = fewer weighted turns to reach convergence) and extends it with differentiated weighting by feedback type. This weighting has no direct precedent in PARADISE or its successors.

**Originality**

The differentiated weighting (`global` weighs 4× more than `point`) has no direct precedent in dialogue evaluation literature. The motivation is: a global feedback turn ("I want to completely reorganize the clusters") requires a far more complex system response than "move this point to cluster B", and therefore carries more weight in measuring oracle cognitive effort. This is an assumption that should be validated empirically in a human study.

---

### A3 — Cognitive Load Score

**How it works**

A3 is a deterministic proxy of the system's cognitive load (not the human user's). It is computed as:

```
score = max(
  round(turns / 20 * 5),             # caps at 5 after 20 turns
  round(tokens_pre_trim / 8000 * 5), # caps at 5 after 8k tokens
  round(active_clusters / 10 * 5)    # caps at 5 after 10 clusters
)
```

The resulting value is in `[1, 5]`. The Planner halts the session when score = 5 (termination code `cognitive_overload`). The `cognitive_load_driver` field records which of the three signals saturated first.

**Theoretical reference**

> Hart, S.G., Staveland, L.E. — *"Development of NASA-TLX (Task Load Index): Results of Empirical and Theoretical Research"* — Human Mental Workload (Hancock & Meshkati, eds.), Elsevier, 1988

The NASA Task Load Index is the standard measure of cognitive workload in HCI. It is a six-subscale questionnaire (mental demand, physical demand, temporal demand, performance, effort, frustration) completed by the user after a task. A3 is conceptually inspired by NASA-TLX but operationalized **deterministically from observable signals** rather than from user self-report.

This introduces a necessary but important limitation: A3 measures the **system's cognitive load** (how heavy the conversation is becoming for the LLM), not the **human user's cognitive load** (how tiring the interface is to use). This distinction is made explicit in the project's quality spec.

**Originality**

Formalizing a deterministic cognitive load score for a conversational clustering system has no direct precedent in the literature. The closest work uses NASA-TLX with human subjects; A3 instead produces an automatic signal for the automated evaluation pipeline.

---

## 3. Family B — LLM-as-Judge

Family B uses separate LLMs as judges. All judges are **out-of-band**: they never participate in the live conversational loop, preventing the system from grading itself.

### Foundations: LLM-as-Judge in the Literature

The paradigm of using an LLM as a judge in place of human raters emerged in 2023 and has rapidly come to dominate the NLP evaluation landscape.

> Zheng, L., et al. — *"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"* — NeurIPS 2023 (Datasets and Benchmarks)
> arXiv: 2306.05685

MT-Bench and Chatbot Arena demonstrate that GPT-4 as a judge achieves over 80% agreement with human judgments on multi-turn conversation tasks. They identify three systematic biases that any LLM-as-judge implementation must account for:

1. **Position bias**: the judge tends to prefer the response presented first (in comparative tasks)
2. **Verbosity bias**: longer responses are preferred regardless of quality
3. **Self-enhancement bias**: an LLM tends to prefer its own responses when it is the judge

BlaBlaClust does not perform comparative scoring (it does not rank two sessions against each other), so position bias does not apply directly. Verbosity bias is relevant for B2 (cluster coherence): a judge receiving verbose cluster descriptions may over-score well-described but poorly-formed clusters. Self-enhancement bias is mitigated by using a different model as judge than the one driving the live session (e.g. Gemini as judge on sessions conducted with Claude).

---

### B2 — Cluster Coherence

**How it works**

The B2 judge receives the top-3 and bottom-2 members (by text) of each cluster and assigns a thematic coherence score in `[0, 1]`. The bottom-2 serve an explicit role: stress-testing the cluster's boundaries, not just its centre.

Reported aggregates:
- `coherence_mean`: average score across all clusters
- `coherence_min`: score of the worst-performing cluster (a single malformed cluster penalizes the entire session)

**Literature analogy**

> Liu, Y., et al. — *"G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"* — EMNLP 2023
> ACL Anthology: 2023.emnlp-main.153

G-Eval is the closest methodological precedent. It proposes using GPT-4 with chain-of-thought to evaluate NLG outputs (summarization, dialogue, data-to-text) on dimensions such as coherence, faithfulness, and fluency. The Spearman correlation with human judges is 0.514 on summarization — higher than automatic metrics such as BERTScore. B2 adopts the same approach: the judge explicitly reasons about coherence before assigning a score.

**The bottom-2 design choice**

Including the bottom-2 members (the points farthest from the cluster centroid) has no explicit precedent in G-Eval, but draws from **stress-testing** methods in NLP benchmarks: rather than showing only representative examples, the worst-case members are included to probe the construct's boundaries. This is an original methodological contribution of BlaBlaClust.

**Limitation**

B2 is non-deterministic: two consecutive runs with the same judge on the same dataset have produced Δ values with opposite signs (sprint 4 p2, §10). This is not a bug — it is the intrinsic variance of the LLM as a judge. The correct conclusion is to report a CI (which spans zero) rather than a point estimate.

---

### B3 — Oracle Compliance

**How it works**

The B3 judge receives the list of oracle turns (with natural-language requests) and the operations actually executed by the system, and scores the fidelity of the request → operation translation in `[0, 1]`.

**Literature analogy**

B3 is conceptually analogous to **faithfulness / grounding** metrics in NLP, which measure how well the information in an output is supported by its input. In translation and summarization, this is commonly referred to as **factual consistency** or **source faithfulness**.

The most directly relevant work is:

> Malaviya, C., et al. — *"Contextualized Evaluations: Judging LLM Responses to Underspecified Queries"* — TACL 2025
> ACL Anthology: 2025.tacl-1.41

This work demonstrates that providing **context** to the LLM judge improves agreement with human judges by 3–10%. In the context of B3, the relevant context is the state of the clustering at the moment of the oracle's request — the correct operation depends on what the system "knows" at that point. BlaBlaClust passes this context to the judge explicitly.

**Known limitation (explicit caveat in the quality spec)**

B3 receives the `target_cluster_ids` provided by the oracle. In sessions where the oracle does not specify explicit IDs (e.g. `contradictory_oracle.json`), the operation-target matching score is 0 even when the textual intent was clear. In those cases, B3 measures the **system's robustness to ambiguous instructions**, not its faithfulness. This caveat is documented in the quality spec and must be reported in the paper.

---

### B4 — Oracle Contradiction

**How it works**

The B4 judge analyses the full oracle feedback history and scores how difficult it would have been for the system to correctly interpret the instructions. It evaluates: self-contradictions, drift in evaluation criteria, vague targets, and ambiguity. The score is in `[0, 1]`, where higher values indicate a harder (more contradictory) oracle.

B4 does not measure system quality — it measures **input difficulty**. Its role is to provide context for B1.

**Literature analogy**

> Malaviya, C., et al. — *"Contextualized Evaluations"* — TACL 2025 (cited above)

This work demonstrates empirically that ignoring the degree of underspecification of a query leads to unfair evaluations: systems that responded reasonably to ambiguous instructions are penalized as if they had received clear ones. B4 is the formalization of this principle in the context of conversational clustering evaluation.

**Originality**

The B4 mechanism as a "forgiveness context" in a conversational clustering system has no direct precedent in the literature. The principle exists (evaluation conditioned on input difficulty), but its specific application to an interactive oracle with iterative feedback is an original contribution of BlaBlaClust.

---

### B1 — Overall Verdict (Reasoning-Based Synthesis)

**How it works**

B1 is the synthetic judge that combines B2, B3, and B4 **not through a formula, but through chain-of-thought**. The judge receives the three scores and reasons about their combination, producing a final score in `[0, 1]`.

The reasoning logic is:
- If the oracle was clear (B4 low) and coherence (B2) or compliance (B3) are low → the system failed → B1 low
- If the oracle was contradictory (B4 high) and coherence/compliance are low → the system faithfully executed difficult instructions → B1 partially forgiven
- If both coherence and compliance are high → B1 high regardless of B4

**Literature analogy**

> Liu, Y., et al. — *"G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"* — EMNLP 2023

G-Eval uses chain-of-thought to evaluate NLG outputs, producing scores with higher correlation to human judges than formula-based metrics. BlaBlaClust's decision to use reasoning rather than a weighted average (e.g. `B1 = 0.5 * B2 + 0.3 * B3 + 0.2 * (1-B4)`) is exactly the same methodological choice as G-Eval, with the same motivation: *a formula cannot distinguish "bad system" from "difficult oracle"*.

**Originality**

The conditional forgiveness mechanism (B4 as a weight on B1's judgment) has no direct equivalent in G-Eval or MT-Bench. It is the most original element of the entire evaluation framework.

---

## 4. Generalization Procedure

**How it works**

Generalization is not a new metric but a procedure that re-applies A1 and B2 at two snapshots: before ingesting new data (t0) and after (t1). The centroids of the converged clustering are **frozen** — new points are assigned to the nearest centroid without re-optimizing. `Δ A1` and `Δ B2` are reported with bootstrap 95% CIs.

The "success" of generalization is operationalized as: the Δ CI spans zero (no significant degradation), not as "accuracy against a held-out test set" — a deliberate and methodologically correct choice for unsupervised clustering.

**Literature analogies**

The procedure draws from two distinct principles:

1. **Distribution shift testing** (well-established in ML): the ability of a model trained on one distribution to maintain performance on a different one. Here, the "different distribution" is the batch of new incoming points.

2. **Nearest-centroid classification** (Voronoi assignment): the assignment method for new points is the simplest possible — Euclidean nearest centroid, without re-fitting k-means. This is equivalent to a 1-NN classifier in embedding space, a standard technique.

No direct precedent exists for "generalization of a conversational clustering" with this exact procedure. It is an original formalization of the question: "does the learned structure hold when new data arrives?"

---

## 5. Framework Validation: the Human Study

**Current state**

The quality spec calls for a human study (N ≈ 5–10) to validate the LLM judges. The protocol is:

- Human raters receive the same payload as the judge (final clusters + oracle feedback history)
- They score `coherence` using the same 1–5 rubric
- Spearman correlation between human and B2 judge scores is reported
- Threshold: correlation < 0.6 invalidates the judge for that metric

This protocol is exactly the one recommended by Zheng et al. (2023) and Liu et al. (2023) for validating an LLM-as-judge: agreement is validated on a subset before applying the judge at scale.

**Current limitation**

The human study has not been conducted. The current B1–B4 numbers have **LLM-only validation**. This is a limitation that must be stated explicitly in the paper. The evaluation framework is methodologically sound; its empirical calibration awaits the human study.

---

## 6. Summary: Originality vs. State of the Art

| Component | Literature basis | Original extension |
|---|---|---|
| A1 (Silhouette) | Rousseeuw 1987 (standard) | Used as diagnostic, not as objective |
| A2 (Turns to convergence) | PARADISE 1997 (weighted dialogue cost) | Per-feedback-type weighting |
| A3 (Cognitive load) | NASA-TLX 1988 (conceptual) | Deterministic proxy from observable signals |
| B2 (Cluster coherence) | G-Eval 2023 (LLM scoring) | Bottom-2 stress test + min as aggregate |
| B3 (Oracle compliance) | NLP faithfulness metrics | Applied to request → operation translation |
| B4 (Oracle contradiction) | Malaviya 2025 (context-aware eval) | Formalized for interactive oracle |
| B1 (Overall verdict) | G-Eval 2023 (chain-of-thought) | Conditional forgiveness via B4 |
| Generalization | Distribution shift testing | Label-free procedure for conversational clustering |
| Dual termination codes | — | `converged` vs `cognitive_overload` with driver tracking |

---

## 7. Implications for the Presentation

### What to emphasize

1. **The decision not to use ARI/NMI is principled**: cite Bontempelli 2020 ("the quality must be assessed in terms of alignment with expert's intent").

2. **B1 via reasoning is not an arbitrary choice**: cite G-Eval (Liu 2023). A formula cannot distinguish "bad system" from "difficult oracle".

3. **The B4 mechanism has a theoretical precedent**: cite Malaviya 2025 (context-aware evaluation).

4. **A2 positions itself relative to PARADISE**: per-feedback-type weighting is an original extension of a well-known paradigm.

### What to acknowledge proactively

1. **The human study was not conducted**: B1–B4 numbers are LLM-only validated. "We know how to validate them, the protocol is written, we did not have time to run it."

2. **B2 is non-deterministic**: two runs on the same dataset produced Δ values with opposite signs. "The CI spanning zero is the correct result. A directional claim cannot be made from a single judge run."

3. **A3 measures system load, not user load**: "Calling it cognitive load is slightly imprecise — it is more accurately a proxy for the LLM system's contextual-computational load."

---

## References

1. Rousseeuw, P.J. (1987). *Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis.* Journal of Computational and Applied Mathematics, 20, 53–65.

2. Hart, S.G., Staveland, L.E. (1988). *Development of NASA-TLX: Results of Empirical and Theoretical Research.* In Human Mental Workload, Elsevier.

3. Walker, M.A., Litman, D.J., Kamm, C.A., Abella, A. (1997). *PARADISE: A Framework for Evaluating Spoken Dialogue Agents.* ACL 1997.

4. Bontempelli, A., et al. (2020). *Interactive Clustering: A Comprehensive Review.* ACM Computing Surveys, 53(1).

5. Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023. arXiv:2306.05685.

6. Liu, Y., et al. (2023). *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.* EMNLP 2023. ACL Anthology: 2023.emnlp-main.153.

7. Malaviya, C., et al. (2025). *Contextualized Evaluations: Judging LLM Responses to Underspecified Queries.* TACL 2025. ACL Anthology: 2025.tacl-1.41.
