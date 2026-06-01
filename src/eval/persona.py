"""Persona file loader.

A persona file describes the goal and tone of an LLM-driven oracle that will
drive a clustering session. The shape is validated up front so a typo in the
JSON fails loudly before any LLM call is made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class PersonaNotes(BaseModel):
    """Free-form notes about how the oracle should behave.

    Extra keys are tolerated so a persona file can add a one-off note without
    a code change; the prompt renders the whole object as a bulleted list.
    """

    model_config = ConfigDict(extra="allow")

    tone: Optional[str] = None
    language: Optional[str] = None
    should_contradict: bool = False
    extra: Optional[str] = None


class Persona(BaseModel):
    """A persona file, validated.

    Top-level extra keys are rejected to catch typos. Notes are permissive.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: Optional[str] = None
    dataset: str
    k_initial: int = Field(ge=2)
    model: Optional[str] = None
    goal: str
    notes: PersonaNotes = Field(default_factory=PersonaNotes)


def load_persona(path: str | Path) -> Persona:
    """Read a persona JSON file from disk and validate it."""
    text = Path(path).read_text(encoding="utf-8")
    raw: Dict[str, Any] = json.loads(text)
    return Persona.model_validate(raw)
