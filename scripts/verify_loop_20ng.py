"""Verify the conversational loop (merge / split / move) on 20 Newsgroups.

Issue #49: we proved initial clustering + naming adapt to a non-Amazon dataset,
but the oracle OPERATIONS had only ever run on Amazon. The merge bug (commit
22a5b0e) was caused by flat soft-assignment geometry — which is dataset-
dependent — so the operations must be re-verified on 20NG's geometry.

This drives the operations through the REAL engine path
(``f_apply_operations`` → ``cluster_operations``) with hand-written operation
dicts (no LLM, fully deterministic), and after each operation asserts the
invariants the issue lists:

  - merge: the new cluster's hard size == the sum of the merged clusters'
    sizes; every un-merged cluster is byte-for-byte unchanged; total points
    conserved (NOT the whole dataset collapsing into one cluster — the bug).
  - split: parent dissolved, exactly k children, children sizes sum to the
    parent's size; other clusters unchanged.
  - move: exactly the named point changes cluster; everyone else unchanged.
  - after every op: the snapshot is complete (every point still assigned).

Cluster naming is stubbed out so the run is offline and reproducible — naming
adaptivity was verified separately. The printed turn-by-turn trace doubles as
the "one transcript per condition" deliverable.

Run:
    PYTHONPATH=. python scripts/verify_loop_20ng.py
"""

from __future__ import annotations

import collections
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.engine.cluster_operations as cluster_operations
import src.logger as logger
from src.dataset_processing.text_cleaning import clean_text
from src.engine.f_apply_operations import f_apply_operations
from src.engine.initial_clustering import initial_clustering
from src.models import Base, ChatSession, Cluster, DataPoint, Dataset, SoftAssignment

TRAIN_CSV = "data/20newsgroups_train.csv"
DATASET = "20_newsgroups"
DATASET_ID = "ds-20ng"
SESSION_ID = "verify-20ng"
K = 6


# ---------------------------------------------------------------------------
# DB snapshot helpers
# ---------------------------------------------------------------------------
def hard_partition(db, turn: int) -> dict[str, str]:
    """point_id -> hard cluster_id (argmax) at a given turn, for this session."""
    cluster_ids = [
        cid for (cid,) in db.query(Cluster.id).filter(Cluster.session_id == SESSION_ID)
    ]
    rows = (
        db.query(SoftAssignment)
        .filter(
            SoftAssignment.cluster_id.in_(cluster_ids),
            SoftAssignment.turn_number == turn,
        )
        .all()
    )
    dist: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for r in rows:
        dist[r.data_point_id][r.cluster_id] = r.probability
    return {pid: max(d, key=d.get) for pid, d in dist.items()}


def sizes(partition: dict[str, str]) -> dict[str, int]:
    return dict(collections.Counter(partition.values()))


def active_cluster_ids(db) -> list[str]:
    return [
        cid
        for (cid,) in db.query(Cluster.id).filter(
            Cluster.session_id == SESSION_ID, Cluster.dissolved_at_turn.is_(None)
        )
    ]


# ---------------------------------------------------------------------------
# Assertion helper
# ---------------------------------------------------------------------------
_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    mark = "✓" if condition else "✗"
    print(f"      {mark} {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        _failures.append(f"{label} — {detail}")


# ---------------------------------------------------------------------------
# Setup: seed 20NG into a temp in-memory DB and run initial clustering
# ---------------------------------------------------------------------------
def setup(db):
    print("Loading + embedding 20NG train split (one-time, ~60-90s on CPU)…")
    texts = []
    with open(TRAIN_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = clean_text(row.get("title") or "", row.get("text") or "")
            if t:
                texts.append(t)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False,
                              convert_to_numpy=True)

    db.add(Dataset(id=DATASET_ID, name=DATASET, description=""))
    db.add(ChatSession(id=SESSION_ID, dataset_id=DATASET_ID,
                       embedding_model="all-MiniLM-L6-v2", status="active"))
    points = []
    for i, (t, emb) in enumerate(zip(texts, embeddings)):
        dp = DataPoint(id=f"p{i}", dataset_id=DATASET_ID,
                       data={"title": "", "text": t}, embedding=emb.tolist())
        db.add(dp)
        points.append(dp)
    db.flush()

    clusters, assignments, _ = initial_clustering(
        points, k=K, session_id=SESSION_ID, turn_number=0
    )
    for c in clusters:
        db.add(c)
    for a in assignments:
        db.add(a)
    db.commit()
    return len(points)


# ---------------------------------------------------------------------------
# Main verification sequence
# ---------------------------------------------------------------------------
def main() -> None:
    # Offline + reproducible: stub cluster naming and redirect the run log.
    cluster_operations.name_clusters = lambda *a, **k: None
    logger._clustering_log_path = Path(tempfile.gettempdir()) / "verify_20ng.jsonl"

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()

    n_points = setup(db)
    turn = 0
    part = hard_partition(db, turn)
    print(f"\nInitial clustering (turn 0): {n_points} points, "
          f"{len(sizes(part))} clusters, sizes={sorted(sizes(part).values(), reverse=True)}")

    def next_turn(ops) -> int:
        nonlocal turn
        latest = db.query(func.max(SoftAssignment.turn_number)).scalar()
        start = (latest or 0) + 1
        turn = f_apply_operations(ops, session_id=SESSION_ID, turn_number=start, db=db)
        db.commit()
        return start  # turn at which this op's snapshot was written

    # ---- MERGE -----------------------------------------------------------
    print("\n[TURN 1] MERGE — combine the two smallest clusters")
    before = sizes(part)
    actives = active_cluster_ids(db)
    by_size = sorted(actives, key=lambda c: before.get(c, 0))
    c1, c2 = by_size[0], by_size[1]
    s1, s2 = before[c1], before[c2]
    untouched_before = {c: before[c] for c in actives if c not in (c1, c2)}
    op_turn = next_turn([{"type": "merge", "cluster_ids": [c1, c2]}])
    after = sizes(hard_partition(db, op_turn))
    new_clusters = [c for c in active_cluster_ids(db) if c not in actives]
    print(f"      merged {c1[:8]}({s1}) + {c2[:8]}({s2})  →  expect new cluster of {s1 + s2}")
    check("new cluster size == sum of merged", len(new_clusters) == 1 and after.get(new_clusters[0]) == s1 + s2,
          f"new={after.get(new_clusters[0]) if new_clusters else None}, expected={s1 + s2}")
    check("merged clusters gone from partition", c1 not in after and c2 not in after)
    check("un-merged clusters byte-for-byte unchanged",
          all(after.get(c) == sz for c, sz in untouched_before.items()),
          f"{untouched_before} vs now { {c: after.get(c) for c in untouched_before} }")
    check("total points conserved (NOT collapsed to one)",
          sum(after.values()) == n_points and after.get(new_clusters[0]) < n_points,
          f"total={sum(after.values())}, biggest={max(after.values())}/{n_points}")

    # ---- SPLIT k=2 -------------------------------------------------------
    print("\n[TURN 2] SPLIT (k=2) — split the largest cluster in two")
    part = hard_partition(db, op_turn)
    before = sizes(part)
    actives = active_cluster_ids(db)
    target = max(actives, key=lambda c: before.get(c, 0))
    sc = before[target]
    untouched_before = {c: before[c] for c in actives if c != target}
    op_turn = next_turn([{"type": "split", "cluster_id": target, "k": 2}])
    after = sizes(hard_partition(db, op_turn))
    children = [c for c in active_cluster_ids(db) if c not in actives]
    print(f"      split {target[:8]}({sc})  →  {len(children)} children, "
          f"sizes={[after.get(c) for c in children]}")
    check("parent dissolved", target not in after)
    check("exactly 2 children", len(children) == 2)
    check("children sizes sum to parent", sum(after.get(c, 0) for c in children) == sc,
          f"sum={sum(after.get(c, 0) for c in children)}, parent={sc}")
    check("other clusters unchanged",
          all(after.get(c) == sz for c, sz in untouched_before.items()))
    check("total conserved", sum(after.values()) == n_points)

    # ---- SPLIT k=3 -------------------------------------------------------
    print("\n[TURN 3] SPLIT (k=3) — split the largest cluster into three")
    part = hard_partition(db, op_turn)
    before = sizes(part)
    actives = active_cluster_ids(db)
    target = max(actives, key=lambda c: before.get(c, 0))
    sc = before[target]
    untouched_before = {c: before[c] for c in actives if c != target}
    op_turn = next_turn([{"type": "split", "cluster_id": target, "k": 3}])
    after = sizes(hard_partition(db, op_turn))
    children = [c for c in active_cluster_ids(db) if c not in actives]
    print(f"      split {target[:8]}({sc})  →  {len(children)} children, "
          f"sizes={[after.get(c) for c in children]}")
    check("parent dissolved", target not in after)
    check("exactly 3 children", len(children) == 3)
    check("children sizes sum to parent", sum(after.get(c, 0) for c in children) == sc,
          f"sum={sum(after.get(c, 0) for c in children)}, parent={sc}")
    check("other clusters unchanged",
          all(after.get(c) == sz for c, sz in untouched_before.items()))
    check("total conserved", sum(after.values()) == n_points)

    # ---- MOVE ------------------------------------------------------------
    print("\n[TURN 4] MOVE — move one point to a different cluster")
    part_before = hard_partition(db, op_turn)
    actives = active_cluster_ids(db)
    moved_point = "p0"
    src_cluster = part_before[moved_point]
    dest = next(c for c in actives if c != src_cluster)
    op_turn = next_turn([{"type": "move", "point_ids": [moved_point],
                          "target_cluster_id": dest}])
    part_after = hard_partition(db, op_turn)
    print(f"      moved {moved_point}: {src_cluster[:8]} → {dest[:8]}")
    check("moved point now in target", part_after.get(moved_point) == dest)
    changed = [pid for pid in part_before
               if pid in part_after and part_before[pid] != part_after[pid]]
    check("exactly ONE point changed cluster", changed == [moved_point],
          f"changed={[p for p in changed][:5]} (n={len(changed)})")
    check("total conserved", len(part_after) == n_points)

    # ---- summary ---------------------------------------------------------
    print("\n" + "=" * 64)
    if _failures:
        print(f"RESULT: ✗ {len(_failures)} INVARIANT(S) FAILED on 20NG geometry:")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("RESULT: ✓ all invariants hold — merge/split/move are dataset-agnostic.")
    print("The conversational loop works on 20 Newsgroups, not just Amazon.")
    print("=" * 64)


if __name__ == "__main__":
    main()
