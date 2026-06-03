"""Tests for the per-turn LLM cost accumulator in src.harness.

The accumulator lets backend/routers/turns.py report the combined token usage
and cost of ALL LLM calls made during one oracle turn (f_output, cluster_naming,
f_update_preferences, boundary repair, …) rather than just f_output's. It is a
thread-local that begin_turn_tracking() resets, call_llm() feeds via
_accumulate_turn(), and pop_turn_tracking() drains.

The contract these tests pin down:
* multiple calls within a turn sum up;
* pop drains and resets (a second pop yields zeros);
* a turn that errors out before pop (begin without pop) never contaminates the
  next turn, because begin resets unconditionally;
* call_llm outside any turn (no active tracker) is a harmless no-op;
* in DRY_RUN (forced by conftest) call_llm accumulates zero usage cleanly.
"""

from src.harness import (
    begin_turn_tracking,
    pop_turn_tracking,
    _accumulate_turn,
    call_llm,
)


def test_single_call_accumulates():
    begin_turn_tracking()
    _accumulate_turn({"input_tokens": 100, "output_tokens": 40}, 0.05)
    usage, cost = pop_turn_tracking()
    assert usage == {"input_tokens": 100, "output_tokens": 40}
    assert cost == 0.05


def test_multiple_calls_sum():
    begin_turn_tracking()
    _accumulate_turn({"input_tokens": 100, "output_tokens": 40}, 0.05)
    _accumulate_turn({"input_tokens": 200, "output_tokens": 10}, 0.03)
    _accumulate_turn({"input_tokens": 50, "output_tokens": 5}, 0.01)
    usage, cost = pop_turn_tracking()
    assert usage == {"input_tokens": 350, "output_tokens": 55}
    assert abs(cost - 0.09) < 1e-9


def test_pop_drains_and_resets():
    begin_turn_tracking()
    _accumulate_turn({"input_tokens": 100, "output_tokens": 40}, 0.05)
    pop_turn_tracking()
    # A second pop without a new begin sees an empty (drained) accumulator.
    usage, cost = pop_turn_tracking()
    assert usage == {"input_tokens": 0, "output_tokens": 0}
    assert cost == 0.0


def test_failed_turn_does_not_contaminate_next():
    """A turn that raises before pop must not leak into the following turn."""
    # Turn A starts, accumulates, then "fails" — pop is never reached.
    begin_turn_tracking()
    _accumulate_turn({"input_tokens": 9999, "output_tokens": 8888}, 1.23)
    # Turn B starts fresh; begin must wipe Turn A's leftovers.
    begin_turn_tracking()
    _accumulate_turn({"input_tokens": 10, "output_tokens": 2}, 0.001)
    usage, cost = pop_turn_tracking()
    assert usage == {"input_tokens": 10, "output_tokens": 2}
    assert abs(cost - 0.001) < 1e-9


def test_accumulate_without_active_tracker_is_noop():
    """call_llm outside a turn (no begin) must not blow up or retain state."""
    # Drain any state from prior tests on this thread.
    pop_turn_tracking()
    _accumulate_turn({"input_tokens": 500, "output_tokens": 500}, 9.99)
    # Nothing was tracked because no turn was active.
    usage, cost = pop_turn_tracking()
    assert usage == {"input_tokens": 0, "output_tokens": 0}
    assert cost == 0.0


def test_call_llm_dry_run_accumulates_zero():
    """Integration: in DRY_RUN call_llm feeds zero usage into the tracker."""
    begin_turn_tracking()
    resp = call_llm([{"role": "user", "content": "hi"}], system="be terse")
    usage, cost = pop_turn_tracking()
    # Dry-run responses carry zero usage, so the accumulator stays at zero —
    # but the call must have gone through the accumulation path without error.
    assert resp.usage == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }
    assert usage == {"input_tokens": 0, "output_tokens": 0}
    assert cost == 0.0
