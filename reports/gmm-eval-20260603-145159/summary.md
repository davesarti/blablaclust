# Evaluation summary
_Generated 2026-06-03T14:58:55_

**Scenarios run:** 5

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
- **B1 overall  _(synthesis judge, 0–1)_**: mean=0.610 median=0.650 n=5
- **B2 coherence mean  _(0–1)_**: mean=0.578 median=0.600 n=5
- **B2 coherence min  _(weakest cluster, 0–1)_**: mean=0.400 median=0.400 n=5
- **B3 compliance  _(0–1)_**: mean=0.964 median=1.000 n=5
- **B4 contradiction  _(0–1, higher = oracle harder to understand)_**: mean=0.190 median=0.000 n=5
- **A1 silhouette (final)  _(cluster separation)_**: mean=0.034 median=0.039 n=5
- **A3 mean cognitive load  _(1–5)_**: mean=1.402 median=1.000 n=5
- **A3 stop-driver distribution**: `turns` × 1
- **A2 termination breakdown**: `cognitive_overload` × 1, `converged` × 4

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

**B1 — Overall quality (synthesis judge):** 0.65/1.0
> The session receives a moderate score as the strong oracle compliance and zero contradiction were undermined by significant coherence issues in 'Household and Personal Goods' and 'Consumer Electronics and PC Accessories'. While the system followed instructions perfectly, the internal logic of these two clusters failed to maintain thematic purity, representing a localized quality concern rather than a total system failure.

**B2 — Coherence:** mean 0.67, min 0.40 — weakest: `Consumer Electronics and PC Accessories` @ 0.40 (This cluster is fragmented due to the inclusion of a query about video game cheats and a complaint about customer service for a nursery/baby bed set, both of which are unrelated to consumer electronics.)

**B3 — Compliance:** 1.00/1.0
> The system correctly interpreted all oracle requests across the session. In each turn, the oracle indicated that no changes were needed, and the system accurately performed no operations, maintaining the cluster structure as requested.

**B4 — Oracle contradiction:** 0.00/1.0 _(higher = oracle harder to understand)_
> The oracle's feedback is entirely consistent throughout the session. The intent remained focused on validating the existing clustering structure, with subsequent requests only asking for descriptive summaries.
> - Turn 1, 2, and 3 consistently validate the current state without requesting any structural changes or reformulations.

**Clusters:** 5 → 5  |  **Wall time:** 25.9s


### sentiment_split
_Oracle refines the clustering by splitting a broad cluster on sentiment, renaming, then accepting. Should exercise split + rename + show in one healthy session._

**A1 — Cluster quality (silhouette):** 0.039 → 0.024 (dropped)
> Measures whether the final clusters are geometrically tight and well-separated. A drop can happen when adding a cluster splits a previously cohesive group.

**A2 — Dialogue efficiency:** 3 turns (weighted 6.0), ended as `converged`
> Operations per turn: turn 1: split; turn 2: rename; turn 3: no ops

**A3 — Cognitive load:** mean 1.0/5 per turn: [1, 1, 1]
> Deterministic 1–5 score from turns / pre-trim tokens / cluster count. High load (5) triggers early termination; the driver names which signal saturated.

**B1 — Overall quality (synthesis judge):** 0.50/1.0
> While the system demonstrated perfect oracle compliance throughout the session, the internal coherence is severely lacking, particularly in the broad and mislabeled Positive Reviews and Negative Reviews clusters. The verdict is driven by the poor grouping quality; despite clear instructions from the oracle, the system failed to organize the content into logically consistent or well-defined thematic categories.

**B2 — Coherence:** mean 0.47, min 0.35 — weakest: `Positive Reviews` @ 0.35 (The cluster is poorly defined and overly broad, containing a chaotic mix of household tools, software, and apparel. The name is too generic, which significantly penalizes the score.)

**B3 — Compliance:** 1.00/1.0
> The system correctly identified the mixed cluster and performed the split into Positive Reviews and Negative Reviews in the first turn. It accurately mapped the request to rename the book cluster in the second turn, and properly interpreted the final turn request as a signal to cease operations.

**B4 — Oracle contradiction:** 0.00/1.0 _(higher = oracle harder to understand)_
> The oracle provided clear, linear instructions that successfully guided the clustering process. The feedback transitions logically from a structural splitting request to a labeling request, culminating in a clear signal that the desired state was achieved.

**Clusters:** 4 → 5  |  **Wall time:** 65.0s


### contradictory_oracle
_Oracle first splits a cluster by sentiment, then reverses course and asks to merge the sentiment clusters back together. Tests that the agent handles contradictory intent gracefully and keeps producing valid state. NOTE: the built-in B3 detector fires only when target_cluster_ids overlap across turns — since scenarios use empty target lists, B3 will read 0 even though the intent is contradictory. This scenario instead validates agent robustness (no crashes, valid final state) rather than the contradiction counter._

**A1 — Cluster quality (silhouette):** 0.039 → 0.024 (dropped)
> Measures whether the final clusters are geometrically tight and well-separated. A drop can happen when adding a cluster splits a previously cohesive group.

**A2 — Dialogue efficiency:** 4 turns (weighted 8.0), ended as `converged`
> Operations per turn: turn 1: split; turn 2: merge; turn 3: split; turn 4: no ops

**A3 — Cognitive load:** mean 1.0/5 per turn: [1, 1, 1, 1]
> Deterministic 1–5 score from turns / pre-trim tokens / cluster count. High load (5) triggers early termination; the driver names which signal saturated.

**B1 — Overall quality (synthesis judge):** 0.75/1.0
> The session resulted in a successful outcome driven by high oracle compliance despite significant environmental challenges. While the overall coherence was dragged down by the Negative Consumer Product Reviews and Positive Consumer Product Reviews clusters, these irregularities are primarily attributed to the high-contradiction oracle that forced the system into reactive, oscillating grouping strategies. Given the system's perfect execution of the oracle's complex and fluctuating demands, the internal inconsistencies are viewed as a reflection of the input rather than a systemic failure.

**B2 — Coherence:** mean 0.61, min 0.40 — weakest: `Negative Consumer Product Reviews` @ 0.40 (The cluster is internally inconsistent, as one of its representative 'top members' is a highly positive GPS review, which contradicts the 'Negative' label and the focus on technical defects found in other members.)

**B3 — Compliance:** 1.00/1.0
> The system correctly navigated the oracle's changing preferences across the turns, executing the requested splits and merges accurately. The final turn correctly identified that no further operations were required based on the oracle's request.

**B4 — Oracle contradiction:** 0.60/1.0 _(higher = oracle harder to understand)_
> The oracle exhibited a pattern of self-contradiction by oscillating between opposing structural requirements. While the instructions were explicitly defined, the oracle reversed its stance twice regarding the positive and negative review clusters without providing a substantive reason for the back-and-forth.
> - Turn 1 requested to split positive and negative reviews, Turn 2 ordered them merged, and Turn 3 requested to split them again.

**Clusters:** 4 → 5  |  **Wall time:** 41.6s


### topic_merge
_Oracle starts with an over-fragmented clustering (k=7) and progressively merges related topic clusters down to a cleaner set. Tests that merge operations fire correctly and coherence improves._

**A1 — Cluster quality (silhouette):** 0.040 → 0.040 (stable)
> Measures whether the final clusters are geometrically tight and well-separated. A drop can happen when adding a cluster splits a previously cohesive group.

**A2 — Dialogue efficiency:** 4 turns (weighted 8.0), ended as `converged`
> Operations per turn: turn 1: merge; turn 2: merge; turn 3: merge; turn 4: no ops

**A3 — Cognitive load:** mean 1.25/5 per turn: [2, 1, 1, 1]
> Deterministic 1–5 score from turns / pre-trim tokens / cluster count. High load (5) triggers early termination; the driver names which signal saturated.

**B1 — Overall quality (synthesis judge):** 0.50/1.0
> The session resulted in an underwhelming outcome primarily due to low internal coherence, as the system failed to organize items into meaningful thematic groups despite a perfectly clear and consistent oracle. While the system achieved perfect compliance by following all instructions, the resulting Electronics Reviews and Personal Wellness and Apparel clusters proved largely disjointed and conceptually mismatched.

**B2 — Coherence:** mean 0.54, min 0.40 — weakest: `Electronics Reviews` @ 0.40 (The cluster is highly fragmented; despite the name 'Electronics Reviews', it contains reviews for books and instructional materials, which belong in a different cluster entirely.)

**B3 — Compliance:** 1.00/1.0
> The system correctly executed all merge requests across the turns, mapping the oracle's instructions to the appropriate cluster operations. The system accurately identified the intent to group related contents and performed each action as requested without any omissions.

**B4 — Oracle contradiction:** 0.00/1.0 _(higher = oracle harder to understand)_
> The oracle provided a consistent and logical progression of tasks aimed at consolidating clusters under broader, intuitive thematic categories. Each instruction was explicit and followed a clear strategy of reducing complexity until the final state was reached.

**Clusters:** 7 → 4  |  **Wall time:** 43.6s


### high_load_oracle
_Oracle sends 21 turns of small, unrelenting nudges without ever converging. Designed to exhaust the cognitive-load budget (score reaches 5 at turn 20, triggering termination=cognitive_overload) before the script ends. Validates the A2 cognitive_overload termination code and that B2 load scores rise predictably with turn count._

**A1 — Cluster quality (silhouette):** 0.043 → 0.039 (stable)
> Measures whether the final clusters are geometrically tight and well-separated. A drop can happen when adding a cluster splits a previously cohesive group.

**A2 — Dialogue efficiency:** 17 turns (weighted 21.5), ended as `cognitive_overload`
> Operations per turn: turn 1: rename; turn 2: rename; turn 3: split; turn 4: split; turn 5: no ops; turn 6: rename, rename, rename, rename, rename, rename, rename; turn 7: semantic_reembed; turn 8: merge; turn 9: rename; turn 10: semantic_reembed; turn 11: split; turn 12: merge; turn 13: split; turn 14: rename; turn 15: semantic_reembed; turn 16: merge; turn 17: split; turn 18: rename, rename

**A3 — Cognitive load:** mean 2.76/5 per turn: [1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5]
> Driver per turn: ['turns']
> Deterministic 1–5 score from turns / pre-trim tokens / cluster count. High load (5) triggers early termination; the driver names which signal saturated.

**B1 — Overall quality (synthesis judge):** 0.65/1.0
> The session resulted in mediocre internal coherence due to the fragmented nature of the Mixed Quality Consumer Goods cluster, though this is partially redeemed by the effective grouping in Media And Literature Critiquing. While the oracle was clear (low contradiction), the system's failure to apply title case formatting across all groups represents a minor compliance failure that prevents a higher score. Overall, the performance was adequate but demonstrated inconsistent attention to formatting and thematic purity.

**B2 — Coherence:** mean 0.60, min 0.45 — weakest: `Mixed Quality Consumer Goods` @ 0.45 (This cluster is highly fragmented, mixing hardware/electronics performance reviews with music album and economic book reviews. The name 'Mixed Quality Consumer Goods' is generic and fails to explain the inclusion of media reviews, which are clearly off-topic for a functional consumer goods cluster.)

**B3 — Compliance:** 0.82/1.0
> The system followed nearly all instructions, effectively executing splits, merges, and renames. However, the system failed to fulfill the Turn 17 request to rename 'every' cluster to use title case, as it only performed this action on two final clusters while leaving others unnamed or ignored.

**B4 — Oracle contradiction:** 0.35/1.0 _(higher = oracle harder to understand)_
> The oracle displays a 'trial-and-error' pattern where structural decisions like split/merge cycles are retracted based on empirical preference rather than logical contradiction. While this creates some administrative churn, the intent remains clear throughout, and the feedback directly addresses the outcomes of previous operations.
> - Turn 9 requested splitting book reviews by genre, which was then immediately undone by the instruction in turn 11.
> - Turn 14 requested splitting positive reviews into five-star and four-star groups, which was reversed in turn 15 after the Oracle deemed it too granular.

**Clusters:** 5 → 2  |  **Wall time:** 240.2s

> **Errors:** ["turn[5] http=422 body={'detail': 'Engine produced an invalid operation: batch_move_points needs at least 1 move'}"]

