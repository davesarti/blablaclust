"""Caps for the deterministic cognitive-load proxy.

The caps in this file define when each signal saturates at score 5. They are
defensible defaults (no empirical model-quality study), tunable in place.
"""

MAX_TURNS: int = 20
TOKEN_BUDGET: int = 16000
CLUSTER_BUDGET: int = 25
