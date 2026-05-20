"""
Run with:
    conda activate vibe-coders
    cd vibe-coders
    PYTHONPATH=. python scripts/serve_ui.py

UI:  http://localhost:8000/ui
API: http://localhost:8000/docs
"""
import pathlib
import uvicorn
from fastapi.responses import FileResponse
from backend.main import app

# ── Schema bridge ─────────────────────────────────────────────────────────────
# P3's f_output prompt now returns:
#   {action: merge|split|rename|no_change|explain, operations: [...], display: "string", ...}
#
# P1's SystemTurn expects:
#   {action: show|ask|stop, clusters_updated: bool, display: {type,content,items}, session_id, ...}
#
# P1's f_apply_operations reads `clusters_updated` as a list of cluster dicts,
# but the prompt now puts operations in `operations` — so cluster updates are a no-op
# until P1 and P3 align their schemas.
#
# This adapter reconciles the two formats without touching anyone else's code.

from src.engine.f_output import f_output as _real_f_output
import backend.routers.turns as _turns_module

_ACTION_MAP = {
    "merge":     "show",
    "split":     "show",
    "rename":    "show",
    "no_change": "show",
    "explain":   "ask",
}

def _f_output_adapted(state, oracle_turn, context, total_points):
    try:
        result = _real_f_output(state, oracle_turn, context, total_points)
        raw, usage = result if isinstance(result, tuple) else (result, {})
    except Exception as exc:
        print(f"[serve_ui] LLM call failed ({exc}), using mock response")
        raw, usage = {}, {}

    display_text = raw.get("display", "")
    if not isinstance(display_text, str):
        display_text = str(display_text)

    adapted = {
        "session_id":           state.session_id,
        "turn_number":          state.turn_number + 1,
        "action":               _ACTION_MAP.get(raw.get("action", "no_change"), "show"),
        "clusters_updated":     False,  # f_apply_operations not yet aligned with new prompt format
        "display": {
            "type":    "message",
            "content": display_text or "[demo] No response from model.",
            "items":   [],
        },
        "contradiction_detected": bool(raw.get("contradiction_detected", False)),
        "contradiction_detail":   None,
        "cognitive_load_score":   int(raw.get("cognitive_load_score", 1)),
        "state_snapshot":         {},
        "token_usage":            usage or None,
        "cost_usd":               None,
    }
    return adapted, usage

_turns_module.f_output = _f_output_adapted
# ─────────────────────────────────────────────────────────────────────────────

UI_PATH = pathlib.Path(__file__).parent.parent / "ui" / "index.html"


@app.get("/ui", include_in_schema=False)
def serve_ui():
    return FileResponse(UI_PATH, media_type="text/html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
