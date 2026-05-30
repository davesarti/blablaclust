import datetime

from sqlalchemy import (
	JSON,
	CheckConstraint,
	DateTime,
	Float,
	ForeignKey,
	Integer,
	String,
	Text,
)
from typing import Optional
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
	pass


class DataPoint(Base):
	__tablename__ = "data_points"

	id: Mapped[str] = mapped_column(String(36), primary_key=True)
	dataset_name: Mapped[str] = mapped_column(String(255), index=True)
	data: Mapped[dict] = mapped_column(JSON)
	embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
	
	soft_assignments: Mapped[list["SoftAssignment"]] = relationship(
		back_populates="data_point", cascade="all, delete-orphan"
	)


class ChatSession(Base):
	__tablename__ = "sessions"
	__table_args__ = (
		CheckConstraint(
			"status IN ('active','converged','closed')",
			name="ck_sessions_status",
		),
	)

	id: Mapped[str] = mapped_column(String(36), primary_key=True)
	name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
	dataset_name: Mapped[str] = mapped_column(String(255), index=True)
	embedding_model: Mapped[str] = mapped_column(String(255))
	status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
	turns: Mapped[list["Turn"]] = relationship(
		back_populates="session", cascade="all, delete-orphan"
	)
	clusters: Mapped[list["Cluster"]] = relationship(
		back_populates="session", cascade="all, delete-orphan"
	)

class Turn(Base):
	__tablename__ = "turns"
	__table_args__ = (
		CheckConstraint("turn_number >= 0", name="ck_turns_turn_number"),
	)

	session_id: Mapped[str] = mapped_column(
		ForeignKey("sessions.id"), primary_key=True
	)
	turn_number: Mapped[int] = mapped_column(Integer, primary_key=True)
	oracle_input: Mapped[dict] = mapped_column(JSON)
	system_output: Mapped[dict] = mapped_column(JSON)

	session: Mapped[ChatSession] = relationship(back_populates="turns")


class Cluster(Base):
	__tablename__ = "clusters"
	__table_args__ = (
		CheckConstraint(
			"dissolved_at_turn IS NULL OR dissolved_at_turn >= created_at_turn",
			name="ck_clusters_turn_order",
		),
	)

	id: Mapped[str] = mapped_column(String(36), primary_key=True)
	session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"))
	name: Mapped[str] = mapped_column(String(255))
	description: Mapped[str] = mapped_column(Text)
	created_at_turn: Mapped[int] = mapped_column(Integer)
	dissolved_at_turn: Mapped[int | None] = mapped_column(Integer, nullable=True)

	session: Mapped[ChatSession] = relationship(back_populates="clusters")
	soft_assignments: Mapped[list["SoftAssignment"]] = relationship(
		back_populates="cluster", cascade="all, delete-orphan"
	)


class SoftAssignment(Base):
	__tablename__ = "soft_assignments"
	__table_args__ = (
		CheckConstraint(
			"probability >= 0.0 AND probability <= 1.0",
			name="ck_soft_assignments_probability",
		),
		CheckConstraint("turn_number >= 0", name="ck_soft_assignments_turn_number"),
	)

	data_point_id: Mapped[str] = mapped_column(
		ForeignKey("data_points.id"), primary_key=True
	)
	cluster_id: Mapped[str] = mapped_column(
		ForeignKey("clusters.id"), primary_key=True
	)
	turn_number: Mapped[int] = mapped_column(Integer, primary_key=True)
	probability: Mapped[float] = mapped_column(Float)

	data_point: Mapped[DataPoint] = relationship(back_populates="soft_assignments")
	cluster: Mapped[Cluster] = relationship(back_populates="soft_assignments")


class EvalCache(Base):
	__tablename__ = "eval_cache"

	key: Mapped[str] = mapped_column(String(64), primary_key=True)
	session_id: Mapped[str] = mapped_column(
		ForeignKey("sessions.id", ondelete="CASCADE"), index=True
	)
	response_json: Mapped[str] = mapped_column(Text)
	created_at: Mapped[datetime.datetime] = mapped_column(
		DateTime, default=datetime.datetime.utcnow
	)
