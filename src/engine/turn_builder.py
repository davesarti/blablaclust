"""In-memory staging of a single conversation turn's clustering changes.

The engine's clustering ops (merge / split / move / semantic_reembed / boundary
repair) all mutate a :class:`TurnBuilder` rather than writing to the DB directly.
The router commits the builder ONCE at the end of the conversation turn, which
keeps the invariant ``Turn.turn_number == SoftAssignment.turn_number ==
Cluster.created_at_turn / dissolved_at_turn``.

Why a builder instead of per-op DB writes
-----------------------------------------
Pre-refactor, each op flushed a full SoftAssignment snapshot at its own
``turn_number`` so the next op could read it back via ``_load_latest_snapshot``.
That required strictly increasing snapshot turns inside a single conv turn,
which diverged the snapshot axis from the conversation axis (and forced a
``final_snapshot_turn`` bridge for the UMAP viewer to make sense of it).

The builder restores 1:1 by keeping the in-progress snapshot in Python. Each op
reads and writes ``builder.snapshot`` (``point_id -> {cluster_id: probability}``),
stages new clusters in ``builder.new_clusters``, and records dissolutions in
``builder.dissolved_ids``. ``commit`` is the only thing that touches the DB:
one batch of INSERTs for the new snapshot at ``builder.turn_number``, mutations
to existing ``Cluster`` rows for dissolutions, and ``db.add`` for new clusters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.models import (
    Cluster as DbCluster,
    SoftAssignment as DbSoftAssignment,
)


@dataclass
class TurnBuilder:
    session_id: str
    turn_number: int
    db: Session
    snapshot: dict[str, dict[str, float]] = field(default_factory=dict)
    new_clusters: dict[str, DbCluster] = field(default_factory=dict)
    dissolved_ids: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, session_id: str, turn_number: int, db: Session) -> "TurnBuilder":
        """Construct a builder pre-loaded with the latest persisted snapshot.

        The first conv turn (no clusters yet) gets an empty snapshot — ops that
        need a prior partition (merge/split/move) will raise on it, which is
        the correct behaviour since they have no clustering to operate on.

        Raises:
            ValueError: ``turn_number`` is not strictly greater than the latest
                persisted snapshot turn. The invariant exists because each conv
                turn writes exactly one snapshot at its own ``turn_number`` —
                writing into the past would collide on the soft-assignment PK.
        """
        cluster_ids = [
            cid
            for (cid,) in db.query(DbCluster.id)
            .filter(DbCluster.session_id == session_id)
            .all()
        ]
        snapshot: dict[str, dict[str, float]] = {}
        prev_turn: int | None = None
        if cluster_ids:
            prev_turn = (
                db.query(func.max(DbSoftAssignment.turn_number))
                .filter(DbSoftAssignment.cluster_id.in_(cluster_ids))
                .scalar()
            )
            if prev_turn is not None:
                if turn_number <= prev_turn:
                    raise ValueError(
                        f"turn_number ({turn_number}) must be greater than the "
                        f"latest snapshot turn ({prev_turn})"
                    )
                rows = (
                    db.query(DbSoftAssignment)
                    .filter(
                        DbSoftAssignment.cluster_id.in_(cluster_ids),
                        DbSoftAssignment.turn_number == prev_turn,
                    )
                    .all()
                )
                for r in rows:
                    snapshot.setdefault(r.data_point_id, {})[r.cluster_id] = (
                        r.probability
                    )
        return cls(
            session_id=session_id,
            turn_number=turn_number,
            db=db,
            snapshot=snapshot,
        )

    # ── Cluster lookup (combined view of DB + staged) ─────────────────────────

    def get_cluster(self, cluster_id: str) -> DbCluster | None:
        """Return a cluster, preferring this turn's staged additions over DB."""
        if cluster_id in self.new_clusters:
            return self.new_clusters[cluster_id]
        return (
            self.db.query(DbCluster)
            .filter(
                DbCluster.id == cluster_id,
                DbCluster.session_id == self.session_id,
            )
            .first()
        )

    def active_clusters(self) -> list[DbCluster]:
        """All clusters considered active mid-turn: DB-active minus our staged
        dissolutions, plus our staged new clusters."""
        db_active = (
            self.db.query(DbCluster)
            .filter(
                DbCluster.session_id == self.session_id,
                DbCluster.dissolved_at_turn.is_(None),
            )
            .all()
        )
        kept = [c for c in db_active if c.id not in self.dissolved_ids]
        kept.extend(self.new_clusters.values())
        return kept

    def is_dissolved(self, cluster_id: str) -> bool:
        """True if the cluster is either dissolved in DB or staged-dissolved."""
        if cluster_id in self.dissolved_ids:
            return True
        cluster = self.get_cluster(cluster_id)
        return cluster is not None and cluster.dissolved_at_turn is not None

    # ── Mutators called by ops ────────────────────────────────────────────────

    def add_cluster(self, cluster: DbCluster) -> None:
        self.new_clusters[cluster.id] = cluster

    def dissolve(self, cluster_id: str) -> None:
        self.dissolved_ids.add(cluster_id)

    # ── Single point-of-DB-write ──────────────────────────────────────────────

    def commit(self) -> None:
        """Persist all staged state as one snapshot at ``self.turn_number``."""
        for cluster in self.new_clusters.values():
            self.db.add(cluster)
        # Mutate existing Cluster rows that we dissolved this turn. We touch
        # each at most once — already-dissolved (in DB) clusters keep whatever
        # turn they were dissolved at; only newly-dissolved ones get our turn.
        for cluster_id in self.dissolved_ids:
            cluster = self.get_cluster(cluster_id)
            if cluster is not None and cluster.dissolved_at_turn is None:
                cluster.dissolved_at_turn = self.turn_number
        for point_id, distribution in self.snapshot.items():
            for cluster_id, probability in distribution.items():
                self.db.add(
                    DbSoftAssignment(
                        data_point_id=point_id,
                        cluster_id=cluster_id,
                        turn_number=self.turn_number,
                        probability=probability,
                    )
                )
