"""LLMOracle — drives a clustering session as a simulated human user.

Holds its own ConversationContext (separate from the system-under-test's) so
the oracle sees the dialogue history exactly as a user would. Each call to
`next_turn` reads the system's last reply + the current cluster view, calls
the configured LLM, parses the JSON response, and returns an InputOracle
payload ready to POST to /turns.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from src.eval.oracle_view import OracleView, filter_invented_ids, render_notes
from src.eval.persona import Persona
from src.harness import (
    ConversationContext,
    call_llm,
    estimate_cost_usd,
    hash_prompt,
    load_prompt,
    loads_llm_json,
)
from src.logger import log_llm_call


_DEFAULT_FEEDBACK_TYPE = "global"
_ALLOWED_FEEDBACK_TYPES = {"global", "cluster", "point", "instructional"}


@dataclass
class OracleTurn:
    """The decoded oracle reply + the InputOracle body to POST.

    `body` is the dict to send to /turns. `satisfied` and `reasoning` are
    surfaced separately because the runner needs `satisfied` to decide
    termination and `reasoning` for logs.
    """

    body: Dict[str, Any]
    satisfied: bool
    reasoning: str
    usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    model: str = ""


class OracleResponseError(ValueError):
    """Raised when the LLM oracle's reply cannot be parsed into a valid turn."""


class LLMOracle:
    """LLM-as-oracle. Stateless w.r.t. the DB; stateful w.r.t. dialogue."""

    def __init__(
        self,
        persona: Persona,
        session_id: str,
        max_turns: int,
    ) -> None:
        self.persona = persona
        self.session_id = session_id
        self.max_turns = max_turns
        self.context = ConversationContext(session_id=session_id)
        # Resolve the oracle's model up front so it's recorded on every turn
        # even when the persona inherits from env.
        self._model = persona.model or os.environ.get(
            "ANTHROPIC_MODEL", "claude-sonnet-4-6"
        )
        self._total_usage = {"input_tokens": 0, "output_tokens": 0}
        self._total_cost_usd = 0.0
        self._prompt_name = "llm_oracle"
        self._prompt_hash = hash_prompt(self._prompt_name)

    # ------------------------------------------------------------------ public

    def observe_system(self, display_text: str) -> None:
        """Record the system's reply so the next prompt sees it as history."""
        self.context.add_system_turn({"display": display_text})

    def next_turn(
        self,
        view: OracleView,
        system_display: str,
        turn_number: int,
    ) -> OracleTurn:
        """Decide the oracle's next message based on the current view."""
        system_prompt = self._render_system_prompt(view, system_display, turn_number)
        messages = self.context.build_messages()
        # Anthropic treats a trailing `assistant` message as a prefill and
        # continues generating from it. The runner always observes the latest
        # system reply before next_turn, so our context always ends in an
        # assistant turn — without dropping it, the model parrots/continues the
        # system's text instead of replying. The current system_display is
        # already inlined into the system prompt via {system_display}, so the
        # trailing assistant turn is redundant anyway.
        if messages and messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        # The oracle's POV puts the clustering system in the `assistant` role,
        # so after dropping the trailing turn the list may still start with an
        # assistant message (or be empty). Anthropic requires conversations to
        # start with `user`; prepend a synthetic seed when that's not the case.
        if not messages or messages[0]["role"] != "user":
            messages = [{"role": "user", "content": "Begin the session."}] + messages

        try:
            response = call_llm(messages, system=system_prompt)
        except ValueError as exc:
            # Provider returned no usable text (e.g. OpenRouter finish_reason='length'
            # with empty content, or a similar empty-output condition). Surface as an
            # OracleResponseError so the runner records it and moves to the next
            # persona instead of crashing the whole run.
            raise OracleResponseError(f"LLM call produced no usable response: {exc}")
        log_llm_call(
            session_id=self.session_id,
            prompt_name=self._prompt_name,
            prompt_hash=self._prompt_hash,
            usage=response.usage,
            cost_usd=estimate_cost_usd(response.usage, self._model),
        )

        oracle_json = self._parse_response(response.text)
        cost = estimate_cost_usd(response.usage, self._model)
        body = self._build_turn_body(oracle_json, view, response.usage, cost)

        # Track the oracle's own message in dialogue history so subsequent
        # turns see what it just said.
        self.context.add_oracle_turn({"raw_text": body["raw_text"]})

        self._total_usage["input_tokens"] += response.usage.get("input_tokens", 0)
        self._total_usage["output_tokens"] += response.usage.get("output_tokens", 0)
        self._total_cost_usd += cost

        return OracleTurn(
            body=body,
            satisfied=bool(oracle_json.get("satisfied", False)),
            reasoning=str(oracle_json.get("reasoning", "")),
            usage=response.usage,
            cost_usd=cost,
            model=self._model,
        )

    @property
    def totals(self) -> Tuple[Dict[str, int], float]:
        """(running_token_usage, running_cost_usd) for the report row."""
        return dict(self._total_usage), self._total_cost_usd

    @property
    def model(self) -> str:
        return self._model

    # ----------------------------------------------------------------- private

    def _render_system_prompt(
        self,
        view: OracleView,
        system_display: str,
        turn_number: int,
    ) -> str:
        template = load_prompt(self._prompt_name)
        return template.format(
            goal=self.persona.goal,
            notes_bulleted=render_notes(self.persona.notes.model_dump()),
            turn_number=turn_number,
            max_turns=self.max_turns,
            system_display=system_display.replace('"', '\\"'),
            clusters_view=view.render(),
        )

    def _parse_response(self, text: str) -> Dict[str, Any]:
        try:
            parsed = loads_llm_json(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise OracleResponseError(
                f"Oracle returned non-JSON or unrecoverable JSON: {exc}; "
                f"raw text: {text[:300]!r}"
            )
        if not isinstance(parsed, dict):
            raise OracleResponseError(
                f"Oracle JSON is not an object: got {type(parsed).__name__}"
            )
        if "raw_text" not in parsed or not isinstance(parsed["raw_text"], str):
            raise OracleResponseError(
                "Oracle JSON is missing a string 'raw_text' field"
            )
        return parsed

    def _build_turn_body(
        self,
        oracle_json: Dict[str, Any],
        view: OracleView,
        usage: Dict[str, int],
        cost_usd: float,
    ) -> Dict[str, Any]:
        feedback_type = oracle_json.get("feedback_type", _DEFAULT_FEEDBACK_TYPE)
        if feedback_type not in _ALLOWED_FEEDBACK_TYPES:
            feedback_type = _DEFAULT_FEEDBACK_TYPE

        raw_cluster_ids = oracle_json.get("target_cluster_ids") or []
        raw_point_ids = oracle_json.get("target_point_ids") or []
        if not isinstance(raw_cluster_ids, list):
            raw_cluster_ids = []
        if not isinstance(raw_point_ids, list):
            raw_point_ids = []
        filtered = filter_invented_ids(
            [str(x) for x in raw_cluster_ids],
            [str(x) for x in raw_point_ids],
            view,
        )

        return {
            "session_id": self.session_id,
            "raw_text": oracle_json["raw_text"],
            "feedback_type": feedback_type,
            "target_cluster_ids": filtered["target_cluster_ids"],
            "target_point_ids": filtered["target_point_ids"],
            "metadata": {
                "oracle_kind": "persona",
                "persona_name": self.persona.name,
                "satisfied": bool(oracle_json.get("satisfied", False)),
                "reasoning": str(oracle_json.get("reasoning", "")),
                "llm": {
                    "model": self._model,
                    "input_tokens": int(usage.get("input_tokens", 0)),
                    "output_tokens": int(usage.get("output_tokens", 0)),
                    "cost_usd": cost_usd,
                },
            },
        }
