# Evaluation summary
_Generated 2026-06-03T14:46:54_

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
- **B1 overall  _(synthesis judge, 0–1)_**: (no data)
- **B2 coherence mean  _(0–1)_**: (no data)
- **B2 coherence min  _(weakest cluster, 0–1)_**: (no data)
- **B3 compliance  _(0–1)_**: (no data)
- **B4 contradiction  _(0–1, higher = oracle harder to understand)_**: (no data)
- **A1 silhouette (final)  _(cluster separation)_**: (no data)
- **A3 mean cognitive load  _(1–5)_**: (no data)
- **A3 stop-driver distribution**: (no overload terminations)
- **A2 termination breakdown**: `None` × 5

---

## Per-scenario detail

### stable_oracle
_Oracle approves the initial clustering with minimal tweaks. Expects fast termination and high coherence._

**A1 — Cluster quality (silhouette):** (no data)
> Measures whether the final clusters are geometrically tight and well-separated. A drop can happen when adding a cluster splits a previously cohesive group.

**A2 — Dialogue efficiency:** ? turns (weighted ?), ended as `?`
**A3 — Cognitive load:** mean (no data) per turn: []
> Deterministic 1–5 score from turns / pre-trim tokens / cluster count. High load (5) triggers early termination; the driver names which signal saturated.

**B1 — Overall quality (synthesis judge):** None/1.0
**B2 — Coherence:** (no data)

**B3 — Compliance:** (no data)

**B4 — Oracle contradiction:** (no data)

**Clusters:** 5 → None  |  **Wall time:** 10.6s

> **Errors:** ['turn[1] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[2] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[3] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'eval http=500 body=Internal Server Error']


### sentiment_split
_Oracle refines the clustering by splitting a broad cluster on sentiment, renaming, then accepting. Should exercise split + rename + show in one healthy session._

**A1 — Cluster quality (silhouette):** (no data)
> Measures whether the final clusters are geometrically tight and well-separated. A drop can happen when adding a cluster splits a previously cohesive group.

**A2 — Dialogue efficiency:** ? turns (weighted ?), ended as `?`
**A3 — Cognitive load:** mean (no data) per turn: []
> Deterministic 1–5 score from turns / pre-trim tokens / cluster count. High load (5) triggers early termination; the driver names which signal saturated.

**B1 — Overall quality (synthesis judge):** None/1.0
**B2 — Coherence:** (no data)

**B3 — Compliance:** (no data)

**B4 — Oracle contradiction:** (no data)

**Clusters:** 4 → None  |  **Wall time:** 8.2s

> **Errors:** ['turn[1] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[2] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[3] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'eval http=500 body=Internal Server Error']


### contradictory_oracle
_Oracle first splits a cluster by sentiment, then reverses course and asks to merge the sentiment clusters back together. Tests that the agent handles contradictory intent gracefully and keeps producing valid state. NOTE: the built-in B3 detector fires only when target_cluster_ids overlap across turns — since scenarios use empty target lists, B3 will read 0 even though the intent is contradictory. This scenario instead validates agent robustness (no crashes, valid final state) rather than the contradiction counter._

**A1 — Cluster quality (silhouette):** (no data)
> Measures whether the final clusters are geometrically tight and well-separated. A drop can happen when adding a cluster splits a previously cohesive group.

**A2 — Dialogue efficiency:** ? turns (weighted ?), ended as `?`
**A3 — Cognitive load:** mean (no data) per turn: []
> Deterministic 1–5 score from turns / pre-trim tokens / cluster count. High load (5) triggers early termination; the driver names which signal saturated.

**B1 — Overall quality (synthesis judge):** None/1.0
**B2 — Coherence:** (no data)

**B3 — Compliance:** (no data)

**B4 — Oracle contradiction:** (no data)

**Clusters:** 4 → None  |  **Wall time:** 7.3s

> **Errors:** ['turn[1] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[2] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[3] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[4] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'eval http=500 body=Internal Server Error']


### topic_merge
_Oracle starts with an over-fragmented clustering (k=7) and progressively merges related topic clusters down to a cleaner set. Tests that merge operations fire correctly and coherence improves._

**A1 — Cluster quality (silhouette):** (no data)
> Measures whether the final clusters are geometrically tight and well-separated. A drop can happen when adding a cluster splits a previously cohesive group.

**A2 — Dialogue efficiency:** ? turns (weighted ?), ended as `?`
**A3 — Cognitive load:** mean (no data) per turn: []
> Deterministic 1–5 score from turns / pre-trim tokens / cluster count. High load (5) triggers early termination; the driver names which signal saturated.

**B1 — Overall quality (synthesis judge):** None/1.0
**B2 — Coherence:** (no data)

**B3 — Compliance:** (no data)

**B4 — Oracle contradiction:** (no data)

**Clusters:** 7 → None  |  **Wall time:** 10.0s

> **Errors:** ['turn[1] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[2] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[3] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[4] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'eval http=500 body=Internal Server Error']


### high_load_oracle
_Oracle sends 21 turns of small, unrelenting nudges without ever converging. Designed to exhaust the cognitive-load budget (score reaches 5 at turn 20, triggering termination=cognitive_overload) before the script ends. Validates the A2 cognitive_overload termination code and that B2 load scores rise predictably with turn count._

**A1 — Cluster quality (silhouette):** (no data)
> Measures whether the final clusters are geometrically tight and well-separated. A drop can happen when adding a cluster splits a previously cohesive group.

**A2 — Dialogue efficiency:** ? turns (weighted ?), ended as `?`
**A3 — Cognitive load:** mean (no data) per turn: []
> Deterministic 1–5 score from turns / pre-trim tokens / cluster count. High load (5) triggers early termination; the driver names which signal saturated.

**B1 — Overall quality (synthesis judge):** None/1.0
**B2 — Coherence:** (no data)

**B3 — Compliance:** (no data)

**B4 — Oracle contradiction:** (no data)

**Clusters:** 5 → None  |  **Wall time:** 13.9s

> **Errors:** ['turn[1] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[2] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[3] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[4] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[5] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[6] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[7] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[8] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[9] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[10] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[11] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[12] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[13] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[14] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[15] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[16] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[17] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[18] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[19] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[20] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'turn[21] http=502 body={\'detail\': "Engine call failed: Error code: 401 - {\'error\': {\'message\': \'Missing Authentication header\', \'code\': 401}}"}', 'eval http=500 body=Internal Server Error']

