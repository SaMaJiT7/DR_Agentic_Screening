"""Router node: decides whether to stop with screening advice or continue.

This node contains no ML and no LLM, so its behaviour is fully unit-testable.

Rule (from the README): stop early **only** for a confident No DR result with
no safety flag. Everything else -- including a low-confidence No DR -- goes to
the explainer.
"""

from __future__ import annotations

from agents.state import DR_State, ROUTER_STOP_CONFIDENCE, Route


def decide_route(state: DR_State) -> tuple[Route, str]:
    """Return ``(route, reason)`` for the given state.

    Raises:
        ValueError: if the detector has not populated ``grade`` yet.
    """

    if state.grade is None or state.confidence is None:
        raise ValueError("router requires detector output (grade, confidence)")

    # Safety flags always force a full explanation.
    if state.urgent or state.referable or state.flags:
        return "explain", "safety flag raised by the detector"

    if state.grade != 0:
        return "explain", f"detector reported {state.grade_label}"

    if state.confidence < ROUTER_STOP_CONFIDENCE:
        return (
            "explain",
            f"low-confidence No DR ({state.confidence:.2f} "
            f"< {ROUTER_STOP_CONFIDENCE:.2f})",
        )

    return "screen_only", "confident No DR with no safety flag"


def router_node(state: DR_State) -> dict:
    """LangGraph node: write ``route`` and ``route_reason`` into the state."""

    route, reason = decide_route(state)
    return {"route": route, "route_reason": reason}


def route_after_detector(state: DR_State) -> str:
    """Conditional-edge selector used by the graph.

    Returns a node name: ``"screen_only"`` or ``"explainer"``.
    """

    route, _ = decide_route(state)
    return "screen_only" if route == "screen_only" else "explainer"
