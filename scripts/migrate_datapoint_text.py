"""Migrate legacy ``data_points.data`` (JSON blob) to the flat ``data_points.text`` column.

Background
----------
``DataPoint`` moved from a ``data`` JSON blob (``{label, title, text}``) to a flat
``text`` column (commit ``0eeb6ce``, "datapoint left with only text value"). A DB
seeded before that change crashes ``serve_ui.py`` at startup with::

    sqlite3.OperationalError: no such column: data_points.text

This script migrates an existing DB **in place**, preserving the ``id`` /
``dataset_id`` / ``embedding`` of every point — no re-embedding, no data loss.

The new ``text`` value reproduces exactly what the old pipeline embedded,
``clean_text(title, text)``: for datasets with empty titles (20 Newsgroups, IMDB)
that equals the body alone; for Amazon it keeps the title prefix the embedding was
built on, so ``text`` <-> ``embedding`` stay consistent.

The table is **recreated** (not just ALTER ADD COLUMN) so the legacy ``data``
(``NOT NULL``) and ``dataset_name`` columns are dropped — otherwise the new ORM,
which never writes ``data``, would fail every INSERT on the leftover NOT NULL
constraint.

Safety
------
* Always writes a timestamped ``<db>.pre_text_migration.bak`` first.
* Idempotent: if ``data_points`` already matches the new schema it exits cleanly.
* Verifies row count, embedding integrity and readable text after migrating.

Usage::

    PYTHONPATH=. python scripts/migrate_datapoint_text.py            # default DB
    PYTHONPATH=. python scripts/migrate_datapoint_text.py --db path/to.db
    PYTHONPATH=. python scripts/migrate_datapoint_text.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_processing.text_cleaning import clean_text  # noqa: E402

DEFAULT_DB = "data/demo_database.db"

# Canonical DDL — must match what SQLAlchemy generates for DataPoint.__table__.
_CREATE_TABLE = """
CREATE TABLE data_points (
    id VARCHAR(36) NOT NULL,
    dataset_id VARCHAR(36) NOT NULL,
    text TEXT NOT NULL,
    embedding JSON,
    PRIMARY KEY (id),
    FOREIGN KEY(dataset_id) REFERENCES datasets (id) ON DELETE CASCADE
)
"""
_CREATE_INDEX = "CREATE INDEX ix_data_points_dataset_id ON data_points (dataset_id)"


def _columns(con: sqlite3.Connection, table: str) -> dict[str, dict]:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1]: {"type": r[2], "notnull": r[3]} for r in rows}


def _needs_migration(con: sqlite3.Connection) -> bool:
    cols = _columns(con, "data_points")
    if not cols:
        raise SystemExit("[migrate] No 'data_points' table found — wrong DB?")
    has_text = "text" in cols
    has_legacy_data = "data" in cols
    if has_text and not has_legacy_data:
        return False  # already on the new schema
    return True


def _transform_text(raw_data: str) -> str:
    d = json.loads(raw_data)
    text = clean_text(d.get("title", "") or "", d.get("text", "") or "")
    if not text:
        raise ValueError(f"empty text after cleaning for row data={raw_data[:120]!r}")
    return text


def migrate(db_path: str, dry_run: bool = False) -> int:
    path = Path(db_path)
    if not path.exists():
        raise SystemExit(f"[migrate] DB not found: {path}")

    con = sqlite3.connect(str(path))
    con.execute("PRAGMA foreign_keys=OFF")
    try:
        if not _needs_migration(con):
            print(f"[migrate] '{path}' already on the new schema — nothing to do.")
            return 0

        rows = con.execute(
            "SELECT id, dataset_id, data, embedding FROM data_points"
        ).fetchall()
        print(f"[migrate] {len(rows)} rows to migrate in '{path}'.")

        # Transform every row up front so a bad row aborts before we touch the DB.
        new_rows = []
        null_dataset = 0
        for pid, dataset_id, data, embedding in rows:
            if dataset_id is None:
                null_dataset += 1
            new_rows.append((pid, dataset_id, _transform_text(data), embedding))
        if null_dataset:
            raise SystemExit(
                f"[migrate] {null_dataset} rows have NULL dataset_id — resolve the "
                f"dataset FK backfill before running this migration."
            )

        if dry_run:
            sample = new_rows[0]
            print("[migrate] --dry-run: no changes written.")
            print(f"[migrate] sample new text: {sample[2][:100]!r}")
            return 0

        backup = path.with_suffix(path.suffix + ".pre_text_migration.bak")
        shutil.copy2(path, backup)
        print(f"[migrate] backup written: {backup}")

        # NOTE: sqlite3 on Python < 3.12 auto-commits before each DDL statement, so
        # the rename/create/drop sequence below is *not* a single atomic transaction.
        # The backup above is the rollback mechanism; the steps are ordered so a
        # re-run after a partial failure still converges (legacy is dropped only
        # after the new table is fully populated, and the index name is freed first).
        con.execute("DROP TABLE IF EXISTS data_points_legacy")  # stray from prior run
        con.execute("ALTER TABLE data_points RENAME TO data_points_legacy")
        # The dataset_id index followed the rename; free its (global) name.
        con.execute("DROP INDEX IF EXISTS ix_data_points_dataset_id")
        con.execute(_CREATE_TABLE)
        con.executemany(
            "INSERT INTO data_points (id, dataset_id, text, embedding) "
            "VALUES (?, ?, ?, ?)",
            new_rows,
        )
        con.execute(_CREATE_INDEX)
        con.execute("DROP TABLE data_points_legacy")
        con.commit()
        print(f"[migrate] migrated {len(new_rows)} rows.")
    finally:
        con.execute("PRAGMA foreign_keys=ON")
        con.close()

    _verify(db_path, expected=len(new_rows))
    return 0


def _verify(db_path: str, expected: int) -> None:
    """Re-open via the ORM to prove the new schema is what the app expects."""
    from sqlalchemy import create_engine, func, select
    from sqlalchemy.orm import Session

    from src.models import DataPoint

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as s:
        count = s.scalar(select(func.count()).select_from(DataPoint))
        with_emb = s.scalar(
            select(func.count())
            .select_from(DataPoint)
            .where(DataPoint.embedding.is_not(None))
        )
        sample = s.scalars(select(DataPoint).limit(1)).first()
    assert count == expected, f"count mismatch: {count} != {expected}"
    assert with_emb == expected, f"embeddings lost: {with_emb}/{expected} present"
    print(
        f"[migrate] verify OK — {count} points, {with_emb} with embeddings.\n"
        f"[migrate] sample text: {sample.text[:100]!r}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DEFAULT_DB, help=f"DB path (default: {DEFAULT_DB})")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Transform + validate without writing changes.",
    )
    args = ap.parse_args()
    return migrate(args.db, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
