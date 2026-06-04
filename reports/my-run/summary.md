# Evaluation summary
_Generated 2026-05-27T16:58:49_

**Scenarios run:** 2

## Aggregates
- **B1 coherence_score**: (no data)
- **B4 endorsement_rate**: mean=0.317 median=0.317 n=2
- **A1 silhouette_final**: mean=0.027 median=0.027 n=2
- **A2 termination breakdown**: {converged=2}

## Per-scenario detail

### sentiment_split
_Oracle refines the clustering by splitting a broad cluster on sentiment, renaming, then accepting. Should exercise split + rename + show in one healthy session._
- A1: silhouette 0.04046419635415077 → 0.025595834478735924  (trend: [0.04046419635415077, 0.025595834478735924])
- A2: 3 turns (weighted 6.0), termination=`converged`
- B1: coherence=None — __
- B2: mean cog load = 1  per turn: [1, 1, 1]
- B3: contradictions detected = 0
- B4: endorsement rate = 0.5, mean confidence = 0.95 (n_valid=14)
- k: 4 → 5; wall_time=51.1s

### stable_oracle
_Oracle approves the initial clustering with minimal tweaks. Expects fast termination and high coherence._
- A1: silhouette 0.02922011725604534 → 0.02922011725604534  (trend: [0.02922011725604534])
- A2: 3 turns (weighted 4.0), termination=`converged`
- B1: coherence=None — __
- B2: mean cog load = 1  per turn: [1, 1, 1]
- B3: contradictions detected = 0
- B4: endorsement rate = 0.133, mean confidence = 0.923 (n_valid=15)
- k: 5 → 5; wall_time=36.1s
