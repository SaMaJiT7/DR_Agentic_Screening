"""Tests for the router's pure decision logic."""

from agents.nodes.router import decide_route, route_after_detector, router_node
from agents.state import make_initial_state


def _state(**kwargs):
    return make_initial_state("scan.png", "Kolkata").model_copy(update=kwargs)


def test_confident_no_dr_stops() -> None:
    state = _state(grade=0, raw_grade=0, confidence=0.91)
    assert decide_route(state) == (
        "screen_only",
        "confident No DR with no safety flag",
    )
    assert router_node(state) == {
        "route": "screen_only",
        "route_reason": "confident No DR with no safety flag",
    }


def test_low_confidence_no_dr_explains() -> None:
    state = _state(grade=0, raw_grade=0, confidence=0.59)
    route, reason = decide_route(state)
    assert route == "explain"
    assert "low-confidence" in reason


def test_any_dr_grade_explains() -> None:
    for grade in (1, 2, 3, 4):
        state = _state(grade=grade, raw_grade=grade, confidence=0.99)
        route, _ = decide_route(state)
        assert route == "explain", grade


def test_flags_force_explain_even_at_grade_zero() -> None:
    for flags in ({"urgent": True}, {"referable": True}, {"flags": ["x"]}):
        state = _state(grade=0, raw_grade=0, confidence=0.99, **flags)
        route, reason = decide_route(state)
        assert route == "explain", flags
        assert "safety flag" in reason


def test_missing_detector_output_raises() -> None:
    try:
        decide_route(make_initial_state("scan.png", "Kolkata"))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError before detection")


def test_conditional_edge_targets() -> None:
    stop = _state(grade=0, confidence=0.99)
    go = _state(grade=2, confidence=0.99)
    assert route_after_detector(stop) == "screen_only"
    assert route_after_detector(go) == "explainer"


if __name__ == "__main__":
    test_confident_no_dr_stops()
    test_low_confidence_no_dr_explains()
    test_any_dr_grade_explains()
    test_flags_force_explain_even_at_grade_zero()
    test_missing_detector_output_raises()
    test_conditional_edge_targets()
    print("router tests passed")
