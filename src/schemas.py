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
    turn_number: int
    dataset_name: str
    status: Literal["active", "converged", "closed"]
    clusters: List[Cluster]
    feedback_history: List[FeedbackEntry]
    contradictions: List[Any] = Field(default_factory=list)


class SoftAssignment(BaseModel):
    data_point_id: str
    cluster_id: str
    turn_number: int = Field(ge=0)
    probability: float = Field(ge=0.0, le=1.0)


class ClusterPoint(BaseModel):
    id: str
    data: Dict[str, Any]
    probability: float = Field(ge=0.0, le=1.0)


class ClusterPointsResponse(BaseModel):
    cluster_id: str
    session_id: str
    turn_number: Optional[int] = None
    points: List[ClusterPoint]


class DatasetUploadResponse(BaseModel):
    dataset_name: str
    inserted: int
    skipped: int
    embeddings_generated: int


class InputOracle(BaseModel):
    session_id: str
    raw_text: str
    feedback_type: Literal["global", "cluster", "point", "instructional"]
    target_cluster_ids: List[str] = Field(default_factory=list)
    target_point_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Display(BaseModel):
    type: str
    content: str
    items: List[Any] = Field(default_factory=list)


class SystemTurn(BaseModel):
    session_id: str
    turn_number: int
    action: Literal["show", "ask", "stop"] #necessary?
    clusters_updated: bool
    display: Display
    contradiction_detected: bool
    contradiction_detail: Optional[str] = None
    cognitive_load_score: int = Field(ge=1, le=5)
    state_snapshot: Dict[str, Any] = Field(default_factory=dict)
    token_usage: Optional[Dict[str, int]] = None
    cost_usd: Optional[float] = None


class TurnRead(BaseModel):
    session_id: str
    turn_number: int
    oracle_input: InputOracle
    system_output: SystemTurn