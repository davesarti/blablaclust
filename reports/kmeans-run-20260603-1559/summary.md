# Evaluation summary
_Generated 2026-06-03T16:08:33_

**Scenarios run:** 4

> **Metric cheat-sheet**
> | Metric | What it measures | Good looks like |
> |--------|-----------------|-----------------|
> | **A1 silhouette** | Geometric separation of clusters (−1 to 1). Higher = tighter, more distinct groups. | Stable or rising across turns. |
> | **A2 turns / weighted turns** | How much dialogue it took to converge. Weighted turns penalise heavy global feedback more than fine-grained point edits. | Fewer turns, ends in `converged`. |
> | **A3 cognitive load** | Per-turn deterministic LLM-side load (1–5) from turns / pre-trim tokens / cluster count. Session halts at 5; the `driver` says which signal saturated. | Stays low (1–2). |
> | **B1 overall** | Synthesis judge over B2, B3, B4 — gives one fair verdict on the session. | ≥ 0.7 |
> | **B2 coherence** | Per-cluster internal focus (LLM judge). Reports mean and the weakest cluster. | Mean ≥ 0.7, no single cluster below 0.5. |
> | **B3 compliance** | Fidelity of system operations to oracle requests (LLM judge). | ≥ 0.7 |
> | **B4 contradiction** | How hard the oracle was to understand: contradictions, drift, vague targets. Higher = harder. Feeds B1 as forgiveness context. | Low (≤ 0.3). High values excuse low B2/B3 in B1. |

## Aggregates across all scenarios
- **B1 overall  _(synthesis judge, 0–1)_**: mean=0.775 median=0.775 n=4
- **B2 coherence mean  _(0–1)_**: mean=0.669 median=0.682 n=4
- **B2 coherence min  _(weakest cluster, 0–1)_**: mean=0.463 median=0.475 n=4
- **B3 compliance  _(0–1)_**: mean=1.000 median=1.000 n=4
- **B4 contradiction  _(0–1, higher = oracle harder to understand)_**: mean=0.150 median=0.000 n=4
- **A1 silhouette (final)  _(cluster separation)_**: mean=0.033 median=0.032 n=4
- **A3 mean cognitive load  _(1–5)_**: mean=1.083 median=1.000 n=4
- **A3 stop-driver distribution**: (no overload terminations)
- **A2 termination breakdown**: `converged` × 4

---

## Per-scenario detail

### stable_oracle
_Oracle approves the initial clustering with minimal tweaks. Expects fast termination and high coherence._

**A1 — Cluster quality (silhouette):** 0.043 → 0.043 (stable)
> Measures whether the final clusters are geometrically tight and well-separated. A drop can happen when adding a cluster splits a previously cohesive group.

**A2 — Dialogue efficiency:** 3 turns (weighted 4.0), ended as `converged`
> Operations per turn: turn 1: no ops; turn 2: no ops; turn 3: no ops

**A3 — Cognitive load:** mean 1.0/5 per turn: [1, 1, 1]
> Deterministic 1–5 score from turns / pre-trim tokens / cluster count. High load (5) triggers early termination; the driver names which signal saturated.

**B1 — Overall quality (synthesis judge):** 0.85/1.0
> The session was highly successful, driven by perfect oracle compliance and zero contradiction, which allowed the system to maintain a stable, well-defined structure. While the overall verdict is positive, the moderate coherence issues within Personal Electronic Device Reviews and Consumer Household and Hobby Products prevent a perfect score, representing localized quality concerns rather than a systemic failure.

**B2 — Coherence:** mean 0.72, min 0.50 — weakest: `Personal Electronic Device Reviews` @ 0.50 (While centered on hardware, it is fragmented by specific off-topic queries about video game cheats and a customer service complaint regarding baby bedding, which do not align with personal electronic device performance reports.)

**B3 — Compliance:** 1.00/1.0
> The system correctly identified that all oracle requests in this session were instructional or conversational rather than operational. It faithfully maintained the existing cluster structure as requested, performing no unnecessary modifications.

**B4 — Oracle contradiction:** 0.00/1.0 _(higher = oracle harder to understand)_
> The oracle provided highly consistent feedback throughout the session. The user maintained a steady level of satisfaction with the existing clustering, focusing on summarizing the state rather than requesting structural modifications.
> - Turn 1 stated the clustering was reasonable, turn 2 requested summaries without changes, and turn 3 confirmed satisfaction with those same groupings.

**Clusters:** 5 → 5  |  **Wall time:** 105.4s


### sentiment_split
_Oracle refines the clustering by splitting a broad cluster on sentiment, renaming, then accepting. Should exercise split + rename + show in one healthy session._

**A1 — Cluster quality (silhouette):** 0.039 → 0.024 (dropped)
> Measures whether the final clusters are geometrically tight and well-separated. A drop can happen when adding a cluster splits a previously cohesive group.

**A2 — Dialogue efficiency:** 3 turns (weighted 6.0), ended as `converged`
> Operations per turn: turn 1: split; turn 2: rename; turn 3: no ops

**A3 — Cognitive load:** mean 1.0/5 per turn: [1, 1, 1]
> Deterministic 1–5 score from turns / pre-trim tokens / cluster count. High load (5) triggers early termination; the driver names which signal saturated.

**B1 — Overall quality (synthesis judge):** 0.70/1.0
> The session was highly successful in terms of oracle compliance and clarity, though the overall score is tempered by poor internal coherence in the 'Positive Reviews' and 'Negative Reviews' clusters. These thematic failures indicate that while the system followed instructions perfectly, it struggled to maintain domain-specific integrity when grouping primarily by sentiment.

**B2 — Coherence:** mean 0.59, min 0.35 — weakest: `Positive Reviews` @ 0.35 (The name 'Positive Reviews' is too generic, resulting in a fragmented cluster that mixes household tools and appliances with literature reviews. The theme is defined by sentiment rather than subject matter, which fails to create a specific domain.)

**B3 — Compliance:** 1.00/1.0
> The system correctly interpreted and executed all requests provided by the oracle across every turn. The split of the mixed cluster was handled accurately, and the renaming of the Book Reviews cluster was applied faithfully.

**B4 — Oracle contradiction:** 0.00/1.0 _(higher = oracle harder to understand)_
> The oracle provided a consistent and clear progression of instructions. The commands demonstrated a logical flow from specific structural changes to finalizing the categorization.
> - Turn 1 established a sentiment-based splitting goal, and Turn 3 confirmed this specific objective had been met, showing alignment throughout the session.

**Clusters:** 4 → 5  |  **Wall time:** 104.8s


### contradictory_oracle
_Oracle first splits a cluster by sentiment, then reverses course and asks to merge the sentiment clusters back together. Tests that the agent handles contradictory intent gracefully and keeps producing valid state. NOTE: the built-in B3 detector fires only when target_cluster_ids overlap across turns — since scenarios use empty target lists, B3 will read 0 even though the intent is contradictory. This scenario instead validates agent robustness (no crashes, valid final state) rather than the contradiction counter._

**A1 — Cluster quality (silhouette):** 0.039 → 0.024 (dropped)
> Measures whether the final clusters are geometrically tight and well-separated. A drop can happen when adding a cluster splits a previously cohesive group.

**A2 — Dialogue efficiency:** 4 turns (weighted 8.0), ended as `converged`
> Operations per turn: turn 1: split; turn 2: merge; turn 3: split; turn 4: no ops

**A3 — Cognitive load:** mean 1.0/5 per turn: [1, 1, 1, 1]
> Deterministic 1–5 score from turns / pre-trim tokens / cluster count. High load (5) triggers early termination; the driver names which signal saturated.

**B1 — Overall quality (synthesis judge):** 0.75/1.0
> The session was successful overall, as the system demonstrated perfect compliance despite significant oracle volatility. While the internal coherence suffered due to the fragmented Negative Consumer Product Reviews cluster, the system's performance is largely excused by the unstable and contradictory instructions regarding sentiment-based grouping.

**B2 — Coherence:** mean 0.69, min 0.45 — weakest: `Negative Consumer Product Reviews` @ 0.45 (This cluster is highly fragmented; while described as negative feedback for electronics, it contains a highly positive review for a GPS device and general customer service complaints about furniture retail.)

**B3 — Compliance:** 1.00/1.0
> The system followed the oracle's fluctuating instructions perfectly across all four turns. It correctly executed the requested splits and merges according to the provided instructions, ending with the appropriate division of product reviews.

**B4 — Oracle contradiction:** 0.60/1.0 _(higher = oracle harder to understand)_
> The oracle exhibited a pattern of indecision, oscillating between splitting and merging the sentiment clusters twice within the session. While the rationale for reverting was explicitly stated, the rapid flip-flopping indicates a lack of stable intent that would cause significant processing churn for a clustering system.
> - Turn 1 requested a split, turn 2 requested a merge, and turn 3 reversed the merge, effectively undoing the previous instruction without new analytical scope.

**Clusters:** 4 → 5  |  **Wall time:** 212.3s


### topic_merge
_Oracle starts with an over-fragmented clustering (k=7) and progressively merges related topic clusters down to a cleaner set. Tests that merge operations fire correctly and coherence improves._

**A1 — Cluster quality (silhouette):** 0.040 → 0.040 (stable)
> Measures whether the final clusters are geometrically tight and well-separated. A drop can happen when adding a cluster splits a previously cohesive group.

**A2 — Dialogue efficiency:** 3 turns (weighted 6.0), ended as `converged`
> Operations per turn: turn 1: merge; turn 2: no ops; turn 3: merge; turn 4: no ops

**A3 — Cognitive load:** mean 1.33/5 per turn: [2, 1, 1]
> Deterministic 1–5 score from turns / pre-trim tokens / cluster count. High load (5) triggers early termination; the driver names which signal saturated.

**B1 — Overall quality (synthesis judge):** 0.80/1.0
> The session is a success driven by perfect Oracle compliance and a clear, consistent instruction set. While minor internal coherence issues exist in 'Apparel and Lifestyle Product Reviews', 'Music Albums and Collections', and 'Electronics Reviews' due to slight category drift, the overall performance is high given the system's precise execution of the Oracle's logic.

**B2 — Coherence:** mean 0.68, min 0.55 — weakest: `Apparel and Lifestyle Product Reviews` @ 0.55 (While centered on lifestyle products, the cluster is fragmented by the inclusion of automotive parts and kitchen appliances, which sit outside the defined theme of 'Apparel and Lifestyle'. The name is overly generic for such a broad mix of consumer goods.)

**B3 — Compliance:** 1.00/1.0
> The system correctly identified the intended clusters for both the electronics and the media-related merges. All operations were executed accurately according to the instructions provided in the oracle's requests across all turns.

**B4 — Oracle contradiction:** 0.00/1.0 _(higher = oracle harder to understand)_
> The oracle provided a consistent, logical progression of instructions to refine the clustering structure. Each request was unambiguous, had clear intent, and concluded with a final declaration of satisfaction.
> - Turn 1 initiated a merge based on a thematic observation, followed by a Turn 2 merge for related media categories, leading logically to the Turn 3 completion signal.

**Clusters:** 7 → 4  |  **Wall time:** 101.8s

> **Errors:** ["turn[2] http=422 body={'detail': 'Engine produced an invalid operation: merge_clusters needs at least 2 distinct clusters'}"]

