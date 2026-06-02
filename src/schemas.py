"""
Pydantic models and example payloads for the JSON schemas.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Cluster(BaseModel):
    id: str
    session_id: str
    name: str
    description: str
    created_at_turn: int
    dissolved_at_turn: Optional[int] = None
    size: int
    representative_points: List[str]


class FeedbackEntry(BaseModel):
    turn: int
    type: Literal["global", "cluster", "point", "instructional"]
    content: str
    target_cluster_ids: List[str] = Field(default_factory=list)
    target_point_ids: List[str] = Field(default_factory=list)


class ChatSessionState(BaseModel):
    session_id: str
    turn_number: int = Field(ge=0)
    dataset_name: str
    embedding_model: str
    status: Literal["active", "converged", "closed"]
    clusters: List[Cluster]
    feedback_history: List[FeedbackEntry]


class ClusterPoint(BaseModel):
    id: str
    data: Dict[str, Any]
    probability: float = Field(ge=0.0, le=1.0)


class ClusterPointsResponse(BaseModel):
    cluster_id: str
    session_id: str
    turn_number: Optional[int] = Field(default=None, ge=0)
    points: List[ClusterPoint]


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    dataset_name: str
    inserted: int
    skipped: int
    embeddings_generated: int
    description: str = ""


class InputOracle(BaseModel):
    session_id: str
    raw_text: str
    feedback_type: Literal["global", "cluster", "point", "instructional"]
    target_cluster_ids: List[str] = Field(default_factory=list)
    target_point_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Display(BaseModel):
    type: Literal["text"] = "text"
    content: str
    items: List[Dict[str, Any]] = Field(default_factory=list)


class CognitiveLoad(BaseModel):
    """Deterministic LLM-side load score (A3).

    `score` is the composite read by the Planner and surfaced through
    `SystemTurn.cognitive_load_score`. `driver` names which signal hit the
    composite. The per-signal scores and raw values are kept for the eval
    pipeline.
    """
    score: int = Field(ge=1, le=5)
    driver: Literal["turns", "tokens", "clusters"]
    turns_score: int = Field(ge=1, le=5)
    tokens_score: int = Field(ge=1, le=5)
    clusters_score: int = Field(ge=1, le=5)
    turns_used: int = Field(ge=0)
    tokens_used: int = Field(ge=0)
    clusters_count: int = Field(ge=0)


class SystemTurn(BaseModel):
    session_id: str
    turn_number: int = Field(ge=0)
    action: Literal["show", "ask", "stop"]
    clusters_updated: bool
    display: Display
    cognitive_load_score: int = Field(ge=1, le=5)
    state_snapshot: Dict[str, Any] = Field(default_factory=dict)
    token_usage: Optional[Dict[str, int]] = None
    cost_usd: Optional[float] = None


class TurnRead(BaseModel):
    session_id: str
    turn_number: int
    oracle_input: InputOracle
    system_output: SystemTurn