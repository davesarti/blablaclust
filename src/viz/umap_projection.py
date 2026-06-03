"""UMAP projection of a session's clustering evolution (P2 — Data & Embeddings).

The goal is to *watch the clustering evolve* over the conversation. We fit a
single 2-D layout over the session's dataset embeddings — so every point gets a
**fixed** coordinate — and then reconstruct the hard cluster assignment of each
point at every snapshot turn. The frontend animates the partition over turns by
recolouring the same points; nothing teleports.

Design notes:
- The dimensionality reducer is **pluggable**: ``umap-learn`` when available,
  otherwise a PCA fallback (scikit-learn), so the pipeline never hard-depends on
  the heavier umap/numba stack. ``compute_coords`` picks automatically, or the
  caller can force one via ``reducer=``.
- UMAP runs once per dataset (the embeddings don't change mid-session), so the
  expensive step is cacheable by ``dataset_name`` — pass a dict as
  ``coords_cache`` and it will be reused across requests.
- Hard label at turn t = argmax soft-assignment probability for that point at t.
- The silhouette overlay is read from the clustering-runs log (best-effort).
- **Geometry-aware mode** (Phase 2): for turns where a ``semantic_reembed`` op
  ran, we additionally fit a *second* UMAP on the hybrid (D+1) re-embed space —
  the actual geometry k-means saw at that turn — so the projection visually
  re-orients alongside the new partition instead of cutting across the original
  topic layout. The poles are deterministic abstract phrases (``very {axis}`` /
  ``not {axis} at all``) — the same fallback ``_generate_axis_poles`` already
  uses on LLM failure — so the view is reproducible and free of API calls. This
  trades faithfulness (the live clustering may have used LLM-judged poles) for
  determinism; the cluster *partition* shown is still the real one from the DB.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import numpy as np

import src.logger as logger
from src.models import ChatSession, Cluster, DataPoint, SoftAssignment, Turn

try:  # umap-learn is optional; PCA is the always-available fallback.
    import umap  # noqa: F401

    _UMAP_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only where umap is absent
    _UMAP_AVAILABLE = False

# Reproducibility for both reducers.
RANDOM_STATE = 42


def compute_coords(
    X: np.ndarray, reducer: str | None = None, random_state: int = RANDOM_STATE
) -> tuple[np.ndarray, str]:
    """Reduce an (N, D) embedding matrix to (N, 2).

    Args:
        X: Float embedding matrix, shape (N, D).
        reducer: ``"umap"``, ``"pca"`` or ``None`` (auto: umap if installed).
        random_state: Seed for deterministic layouts.

    Returns:
        ``(coords, reducer_name)`` where coords is (N, 2) float64 and
        reducer_name is the reducer actually used (``"umap"`` falls back to
        ``"pca"`` on any failure, e.g. too few points).
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    if reducer is None:
        reducer = "umap" if _UMAP_AVAILABLE else "pca"

    if reducer == "umap":
        try:
            import umap

            # n_neighbors must be < n_samples; cosine suits sentence embeddings.
            n_neighbors = min(15, max(2, n - 1))
            mapper = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric="cosine",
                random_state=random_state,
            )
            coords = mapper.fit_transform(X)
            return np.asarray(coords, dtype=np.float64), "umap"
        except Exception as exc:  # degrade gracefully to PCA
            logger.deviation(
                "umap_projection: UMAP failed — falling back to PCA",
                n_points=n,
                error=str(exc),
            )

    from sklearn.decomposition import PCA

    coords = PCA(n_components=2, random_state=random_state).fit_transform(X)
    return np.asarray(coords, dtype=np.float64), "pca"


def _silhouette_by_turn(session_id: str) -> dict[str, float | None]:
    """Map ``turn_number -> silhouette`` from the clustering-runs log.

    Best-effort: a missing or malformed log yields ``{}`` rather than raising.
    Reads ``logger._clustering_log_path`` dynamically so test patches apply.
    """
    path = logger._clustering_log_path
    result: dict[str, float | None] = {}
    try:
        if not path.exists():
            return result
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("session_id") == session_id and "turn_number" in entry:
                result[str(entry["turn_number"])] = entry.get("silhouette")
    except OSError:
        pass
    return result


def _axis_label_at_turn(
    db, session_id: str, turn_number: int
) -> str | None:
    """Read the ``axis_label`` of the semantic_reembed op recorded at ``turn_number``.

    Returns None when the turn has no semantic_reembed op (so the caller knows
    to skip the geometry-aware layout for that turn). Tolerates the older
    ``axis_hint`` key for resumed sessions, matching ``_active_axis_from_history``
    in the turns router.
    """
    row = (
        db.query(Turn)
        .filter(Turn.session_id == session_id, Turn.turn_number == turn_number)
        .one_or_none()
    )
    if row is None:
        return None
    ops = ((row.system_output or {}).get("state_snapshot") or {}).get("operations") or []
    for op in ops:
        if isinstance(op, dict) and op.get("type") == "semantic_reembed":
            label = op.get("axis_label") or op.get("axis_hint")
            if label:
                return str(label)
    return None


def _encode_poles_default(axis_label: str) -> tuple[np.ndarray, np.ndarray]:
    """Encode the two abstract pole phrases via MiniLM (the engine's encoder)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    pos = model.encode(f"very {axis_label}", convert_to_numpy=True).astype(np.float64)
    neg = model.encode(f"not {axis_label} at all", convert_to_numpy=True).astype(np.float64)
    return pos, neg


def _build_hybrid_space(
    point_ids: list[str],
    embeddings: np.ndarray,
    axis_label: str,
    axis_weight: float = 0.7,
    *,
    pole_encoder=None,
) -> np.ndarray:
    """Reconstruct the (N, D+1) hybrid space for a ``semantic_reembed`` axis,
    deterministically (no LLM calls).

    Mirrors ``f_semantic_reembed.reembed_for_axis`` but with **abstract phrase
    poles** ("very {axis}" / "not {axis} at all") — the same fallback that
    ``_generate_axis_poles`` returns on LLM failure. Cosine scoring against
    those poles is fully deterministic, so the geometry-aware UMAP is
    reproducible across requests and costs zero API calls.

    Args:
        point_ids: ids parallel to ``embeddings`` rows (unused here, kept so the
            caller can build a session-stable cache key).
        embeddings: (N, D) float32/64 matrix of the *original* dataset embeddings.
            D must match the pole encoder's output dimensionality (384 for the
            default MiniLM encoder, i.e. the production case).
        axis_label: axis text, e.g. "positive sentiment".
        axis_weight: same default 0.7 as the engine's ``reembed_for_axis``.
        pole_encoder: optional callable ``axis_label -> (pos, neg)`` returning two
            (D,) vectors. Defaults to MiniLM (matches the engine). Injection
            exists so unit tests can run without downloading MiniLM and against
            arbitrary D.

    Returns:
        (N, D+1) float32 matrix in the hybrid space — what k-means saw at the
        reembed turn (modulo the LLM-judged variant when the engine used it).

    Raises:
        ValueError: when the pole encoding's dimension does not match ``D``.
    """
    encoder = pole_encoder or _encode_poles_default
    pole_pos, pole_neg = (np.asarray(v, dtype=np.float64) for v in encoder(axis_label))

    X = np.asarray(embeddings, dtype=np.float64)
    if pole_pos.shape[0] != X.shape[1]:
        raise ValueError(
            f"pole dimension ({pole_pos.shape[0]}) does not match embeddings dim "
            f"({X.shape[1]}) — encoder/embedding model mismatch?"
        )

    pole_pos /= np.linalg.norm(pole_pos) + 1e-8
    pole_neg /= np.linalg.norm(pole_neg) + 1e-8
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    orig_norm = X / (row_norms + 1e-8)
    scores = orig_norm @ pole_pos - orig_norm @ pole_neg  # (N,) signed cosine axis
    axis_norm = (scores - scores.mean()) / (scores.std() + 1e-8)

    orig_scale = float(np.sqrt(1.0 - axis_weight))
    ax_scale = float(np.sqrt(axis_weight))
    return np.hstack(
        [orig_norm * orig_scale, axis_norm.reshape(-1, 1) * ax_scale]
    ).astype(np.float32)


def compute_geometry_aware_coords(
    db,
    session_id: str,
    turn_number: int,
    embeddings: np.ndarray,
    point_ids: list[str],
    *,
    reducer: str | None = None,
    cache: dict | None = None,
    pole_encoder=None,
) -> tuple[np.ndarray, str, str] | None:
    """Geometry-aware 2-D layout for a single ``semantic_reembed`` turn.

    Fits a *second* UMAP on the hybrid (D+1) space the engine clustered at this
    turn, so the partition shown by the slider is laid out in the re-oriented
    geometry instead of the original topic layout. Returns ``None`` when this
    turn has no semantic_reembed op — the caller then keeps the baseline coords.

    Cached by ``(session_id, turn_number)`` because the hybrid space depends on
    the session's axis. Pass a shared ``cache`` dict to amortise across requests.

    Returns:
        ``(coords_2d, reducer_name, axis_label)`` or ``None``.
    """
    axis_label = _axis_label_at_turn(db, session_id, turn_number)
    if axis_label is None:
        return None

    key = (session_id, int(turn_number))
    if cache is not None and key in cache:
        return cache[key]

    X_hybrid = _build_hybrid_space(
        point_ids, embeddings, axis_label, pole_encoder=pole_encoder
    )
    coords, reducer_name = compute_coords(X_hybrid, reducer=reducer)
    result = (coords, reducer_name, axis_label)
    if cache is not None:
        cache[key] = result
    return result


def project_session(
    db,
    session_id: str,
    coords_cache: dict[str, tuple[list[str], np.ndarray, str]] | None = None,
    reducer: str | None = None,
    *,
    geometry_aware: bool = False,
    geometry_cache: dict | None = None,
) -> dict[str, Any]:
    """Build the full UMAP projection payload for one session.

    Args:
        db: SQLAlchemy session.
        session_id: The session to project.
        coords_cache: Optional ``{dataset_name: (point_ids, coords, reducer)}``
            cache for the expensive reduction step. Mutated in place on miss.
        reducer: Force a reducer (see :func:`compute_coords`); ``None`` = auto.

    Returns:
        A JSON-serialisable dict::

            {
              "session_id", "dataset_name", "reducer", "n_points",
              "points":   [{"id", "x", "y", "text"}],          # fixed order
              "turns":    [int, ...],                          # snapshot turns
              "assignments": {"<turn>": [cluster_id|None, ...]},  # parallel to points
              "clusters": {"<id>": {"name", "created_at_turn", "dissolved_at_turn"}},
              "silhouette_by_turn": {"<turn>": float|None},
              "centroids_by_turn": {"<turn>": {"<cluster_id>": [cx, cy]}},
              "reembed_turns": [int, ...],
              "axis_arrows": {"<turn>": {"from": [x,y], "to": [x,y], "label": str}},
              # Only present when geometry_aware=True; one entry per reembed_turn
              # containing the secondary UMAP of the hybrid (D+1) space.
              "geometry_aware": {
                  "<turn>": {
                      "axis_label": str,
                      "reducer": str,
                      "points": [{"id", "x", "y"}],            # parallel to points
                      "centroids": {"<cluster_id>": [cx, cy]},
                  },
                  ...
              },
            }

    Raises:
        ValueError: unknown session, or the dataset has no embedded points.
    """
    session = db.get(ChatSession, session_id)
    if session is None:
        raise ValueError(f"session '{session_id}' not found")
    dataset_id = session.dataset_id
    dataset_name = session.dataset.name if session.dataset else dataset_id

    points = (
        db.query(DataPoint)
        .filter(DataPoint.dataset_id == dataset_id, DataPoint.embedding.isnot(None))
        .order_by(DataPoint.id)
        .all()
    )
    # The SQL filter catches SQL NULL; a Python guard also catches JSON ``null``
    # (the JSON column serialises Python None as JSON null, not SQL NULL).
    points = [p for p in points if p.embedding is not None]
    if not points:
        raise ValueError(f"dataset '{dataset_name}' has no embedded points")
    point_ids = [p.id for p in points]

    # ── Reduce to 2-D (cached per dataset — embeddings are immutable) ──────────
    cached = coords_cache.get(dataset_id) if coords_cache is not None else None
    if cached is not None and cached[0] == point_ids:
        coords, reducer_name = cached[1], cached[2]
    else:
        X = np.array([p.embedding for p in points], dtype=np.float32)
        coords, reducer_name = compute_coords(X, reducer=reducer)
        if coords_cache is not None:
            coords_cache[dataset_id] = (point_ids, coords, reducer_name)

    idx = {pid: i for i, pid in enumerate(point_ids)}

    # ── Cluster metadata for this session ─────────────────────────────────────
    clusters = db.query(Cluster).filter(Cluster.session_id == session_id).all()
    clusters_meta = {
        c.id: {
            "name": c.name,
            "created_at_turn": c.created_at_turn,
            "dissolved_at_turn": c.dissolved_at_turn,
        }
        for c in clusters
    }

    # ── Per-turn hard assignment (argmax soft probability) ────────────────────
    # turn -> point_id -> (best_prob, cluster_id)
    best: dict[int, dict[str, tuple[float, str]]] = defaultdict(dict)
    rows = (
        db.query(SoftAssignment)
        .join(Cluster, SoftAssignment.cluster_id == Cluster.id)
        .filter(Cluster.session_id == session_id)
        .all()
    )
    for r in rows:
        cur = best[r.turn_number].get(r.data_point_id)
        if cur is None or r.probability > cur[0]:
            best[r.turn_number][r.data_point_id] = (r.probability, r.cluster_id)

    all_turns = sorted(best.keys())
    full_assignments: dict[int, list[str | None]] = {}
    full_confidence: dict[int, list[float | None]] = {}
    for t in all_turns:
        arr: list[str | None] = [None] * len(point_ids)
        conf: list[float | None] = [None] * len(point_ids)
        for pid, (prob, cid) in best[t].items():
            if pid in idx:
                arr[idx[pid]] = cid
                conf[idx[pid]] = round(float(prob), 4)
        full_assignments[t] = arr
        full_confidence[t] = conf

    # One slider frame per conversation turn. The engine commits each conv
    # turn's changes as a single snapshot at turn_number == conv turn number,
    # so ``Turn.turn_number`` is also the snapshot key in ``full_assignments``.
    # Frame 0 is the initial clustering (pre-conversation); conv turns with no
    # snapshot-writing ops inherit the previous frame's partition so the slider
    # stays 1:1 with the conversation.
    conv_rows = sorted(
        db.query(Turn).filter(Turn.session_id == session_id).all(),
        key=lambda r: r.turn_number,
    )

    snapshot_for_frame: dict[int, int] = {}
    turns: list[int] = []
    prev_snap: int | None = None
    if 0 in full_assignments:
        turns.append(0)
        snapshot_for_frame[0] = 0
        prev_snap = 0

    for row in conv_rows:
        N = row.turn_number
        snap = N if N in full_assignments else prev_snap
        if snap is None:
            continue
        turns.append(N)
        snapshot_for_frame[N] = snap
        prev_snap = snap

    assignments: dict[str, list[str | None]] = {
        str(frame): full_assignments[snap]
        for frame, snap in snapshot_for_frame.items()
    }
    # Per-point assignment confidence = the winning soft-probability. Lets the
    # UI convey the underlying distribution (e.g. how decisively a point sits in
    # its cluster) instead of only the hard argmax.
    confidence: dict[str, list[float | None]] = {
        str(frame): full_confidence[snap]
        for frame, snap in snapshot_for_frame.items()
    }

    points_out = [
        {
            "id": p.id,
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "text": (p.text or "")[:160],
        }
        for i, p in enumerate(points)
    ]

    # ── Centroids per frame in 2-D UMAP space ───────────────────────────────
    centroids_by_turn: dict[str, dict[str, list[float]]] = {}
    for frame, snap in snapshot_for_frame.items():
        cluster_point_idxs: dict[str, list[int]] = defaultdict(list)
        for pid, (_, cid) in best[snap].items():
            if pid in idx:
                cluster_point_idxs[cid].append(idx[pid])
        centroids_by_turn[str(frame)] = {
            cid: [float(np.mean(coords[idxs, 0])), float(np.mean(coords[idxs, 1]))]
            for cid, idxs in cluster_point_idxs.items()
        }

    # ── Semantic re-embed frames: read from persisted ops ────────────────────
    axis_labels: dict[int, str] = {}
    for row in conv_rows:
        ops = ((row.system_output or {}).get("state_snapshot") or {}).get("operations") or []
        for op in ops:
            if isinstance(op, dict) and op.get("type") == "semantic_reembed":
                label = op.get("axis_label") or op.get("axis_hint") or ""
                axis_labels[row.turn_number] = str(label)
    reembed_frames = sorted(f for f in axis_labels if f in snapshot_for_frame)

    axis_arrows: dict[str, dict] = {}
    for frame in reembed_frames:
        snap = snapshot_for_frame[frame]
        # Created clusters are keyed by SNAPSHOT turn in the cluster table, not
        # conv turn — match against the resolved snapshot.
        created = [cid for cid, m in clusters_meta.items() if m["created_at_turn"] == snap]
        cents_dict = centroids_by_turn.get(str(frame), {})
        pts_2d = np.array([cents_dict[cid] for cid in created if cid in cents_dict])
        if len(pts_2d) < 2:
            continue
        mean_2d = pts_2d.mean(axis=0)
        centered = pts_2d - mean_2d
        # First principal component = axis direction in 2D UMAP space.
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]
        span = float(np.max(np.abs(centered @ direction))) * 1.2
        axis_arrows[str(frame)] = {
            "from":  (mean_2d - span * direction).tolist(),
            "to":    (mean_2d + span * direction).tolist(),
            "label": axis_labels.get(frame, ""),
        }

    # ── Phase 2: geometry-aware projection for semantic_reembed frames ───────
    # Opt-in (geometry_aware=True). For each reembed frame we fit a SECOND UMAP
    # on the hybrid (D+1) space the engine clustered in at that turn, so the
    # partition is shown in the re-oriented geometry instead of cutting across
    # the original topic layout. Deterministic (abstract-phrase poles, no LLM).
    geometry_aware_payload: dict[str, dict[str, Any]] = {}
    if geometry_aware and reembed_frames:
        X_orig = np.array([p.embedding for p in points], dtype=np.float32)
        for frame in reembed_frames:
            snap = snapshot_for_frame[frame]
            res = compute_geometry_aware_coords(
                db,
                session_id=session_id,
                turn_number=frame,
                embeddings=X_orig,
                point_ids=point_ids,
                reducer=reducer,
                cache=geometry_cache,
            )
            if res is None:
                continue
            ga_coords, ga_reducer, axis_label_t = res
            ga_centroids: dict[str, list[float]] = {}
            for cid, idxs in {
                cid: [idx[pid] for pid, (_, cid_p) in best[snap].items()
                      if cid_p == cid and pid in idx]
                for cid in {c for _, c in best[snap].values()}
            }.items():
                if idxs:
                    ga_centroids[cid] = [
                        float(np.mean(ga_coords[idxs, 0])),
                        float(np.mean(ga_coords[idxs, 1])),
                    ]
            geometry_aware_payload[str(frame)] = {
                "axis_label": axis_label_t,
                "reducer": ga_reducer,
                "points": [
                    {"id": point_ids[i], "x": float(ga_coords[i, 0]),
                     "y": float(ga_coords[i, 1])}
                    for i in range(len(point_ids))
                ],
                "centroids": ga_centroids,
            }

    # Silhouette: stored against snapshot turns in the clustering log; re-key
    # to the slider's frame numbers so the UI overlay lines up.
    sil_by_snap = _silhouette_by_turn(session_id)
    silhouette_by_frame: dict[str, float | None] = {}
    for frame, snap in snapshot_for_frame.items():
        val = sil_by_snap.get(str(snap))
        if val is not None:
            silhouette_by_frame[str(frame)] = val

    return {
        "session_id": session_id,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "reducer": reducer_name,
        "n_points": len(points),
        "points": points_out,
        "turns": turns,
        "assignments": assignments,
        "confidence": confidence,
        "clusters": clusters_meta,
        "silhouette_by_turn": silhouette_by_frame,
        "centroids_by_turn": centroids_by_turn,
        "reembed_turns": reembed_frames,
        "axis_arrows": axis_arrows,
        "geometry_aware": geometry_aware_payload,
    }
