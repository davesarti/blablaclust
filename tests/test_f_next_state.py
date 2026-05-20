import os
os.environ["HARNESS_DRY_RUN"] = "true"

from src.schemas import ChatSessionState, InputOracle, Cluster
from src.harness import ConversationContext
from src.engine.f_next_state import f_next_state


# --- fixtures ---

def make_cluster(id: str, session_id: str = "sess-1") -> Cluster:
    return Cluster(
        id=id,
        session_id=session_id,
        name=f"Cluster {id}",
        description="A test cluster",
        size=10,
        representative_points=[],
        created_at_turn=1,
    )


def make_state(clusters: list[Cluster] | None = None) -> ChatSessionState:
    return ChatSessionState(
        session_id="sess-1",
        turn_number=1,
        dataset_name="test_dataset",
        status="active",
        clusters=clusters or [make_cluster("c1"), make_cluster("c2")],
        feedback_history=[],
        contradictions=[],
    )


def make_oracle_turn(feedback_type: str = "global", target_cluster_ids: list[str] | None = None) -> InputOracle:
    return InputOracle(
        session_id="sess-1",
        raw_text="merge the two clusters",
        feedback_type=feedback_type,
        target_cluster_ids=target_cluster_ids or [],
        target_point_ids=[],
        metadata={},
    )


# --- tests ---

def test_turn_number_increments():
    state = make_state()
    context = ConversationContext(session_id="sess-1")
    new_state = f_next_state(state, make_oracle_turn(), context, total_points=100)
    assert new_state.turn_number == state.turn_number + 1


def test_output_is_valid_chat_session_state():
    state = make_state()
    context = ConversationContext(session_id="sess-1")
    new_state = f_next_state(state, make_oracle_turn(), context, total_points=100)
    assert isinstance(new_state, ChatSessionState)
    assert new_state.session_id == state.session_id
    assert new_state.dataset_name == state.dataset_name


def test_fallback_to_current_clusters_when_llm_returns_none():
    # dry run returns clusters_updated=[] so f_next_state must keep existing clusters
    state = make_state()
    context = ConversationContext(session_id="sess-1")
    new_state = f_next_state(state, make_oracle_turn(), context, total_points=100)
    assert new_state.clusters == state.clusters


def test_feedback_history_grows():
    state = make_state()
    context = ConversationContext(session_id="sess-1")
    new_state = f_next_state(state, make_oracle_turn(), context, total_points=100)
    assert len(new_state.feedback_history) == len(state.feedback_history) + 1


def test_feedback_entry_content_matches_oracle_input():
    state = make_state()
    context = ConversationContext(session_id="sess-1")
    oracle_turn = make_oracle_turn(feedback_type="cluster", target_cluster_ids=["c1"])
    new_state = f_next_state(state, oracle_turn, context, total_points=100)
    entry = new_state.feedback_history[-1]
    assert entry.content == oracle_turn.raw_text
    assert entry.type == "cluster"
    assert entry.target_cluster_ids == ["c1"]


def test_no_crash_with_empty_clusters():
    state = make_state(clusters=[])
    context = ConversationContext(session_id="sess-1")
    new_state = f_next_state(state, make_oracle_turn(), context, total_points=100)
    assert isinstance(new_state, ChatSessionState)


def test_contradiction_stub_does_not_crash():
    # detect_contradiction is a stub that returns None — contradictions list stays empty
    state = make_state()
    context = ConversationContext(session_id="sess-1")
    new_state = f_next_state(state, make_oracle_turn(), context, total_points=100)
    assert new_state.contradictions == []