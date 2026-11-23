"""Tests for the LangGraph wiring, using fake nodes (no ML, LLM or network)."""

from agents.graph import build_graph, run_screening
from agents.state import DR_State


def make_fakes(grade: int, confidence: float, flags=None):
    """Return node fakes plus a call log."""

    calls: list[str] = []
    flags = flags or []

    def detector(state: DR_State) -> dict:
        calls.append("detector")
        return {
            "grade": grade,
            "raw_grade": grade,
            "label": ["No DR", "Mild", "Moderate", "Severe", "Proliferative"][grade],
            "confidence": confidence,
            "probs": [0.0] * 5,
            "urgent": grade >= 3,
            "referable": grade >= 2,
            "flags": list(flags),
            "t_severe": 0.084,
            "t_referable": 0.368,
        }

    def router(state: DR_State) -> dict:
        calls.append("router")
        from agents.nodes.router import decide_route

        route, reason = decide_route(state)
        return {"route": route, "route_reason": reason}

    def screen_only(state: DR_State) -> dict:
        calls.append("screen_only")
        return {
            "explanation": "No DR (no diabetic retinopathy) was seen. This is a screening result, not a diagnosis.",
            "next_steps": ["Book a routine dilated eye exam."],
            "consult_note": "You must still consult a doctor to confirm this result.",
        }

    def explainer(state: DR_State) -> dict:
        calls.append("explainer")
        return {"explanation": "Explained stage."}

    def cost(state: DR_State) -> dict:
        calls.append("cost")
        return {"cost_options": [], "cost_note": "No injection cost."}

    def caregiver(state: DR_State) -> dict:
        calls.append("caregiver")
        return {"care_plan": "Care plan.", "next_steps": ["See a doctor."]}

    def hospital_search(state: DR_State) -> dict:
        calls.append("hospital_search")
        return {"hospitals": []}

    def reporter(state: DR_State) -> dict:
        calls.append("report")
        return {"report": f"report for grade {state.grade}"}

    return {
        "detector": detector,
        "router": router,
        "screen_only": screen_only,
        "explainer": explainer,
        "cost": cost,
        "caregiver": caregiver,
        "hospital_search": hospital_search,
        "reporter": reporter,
    }, calls


def _run(grade, confidence, flags=None):
    nodes, calls = make_fakes(grade, confidence, flags)
    graph = build_graph(**nodes)
    result = graph.invoke({"image_path": "scan.png", "city": "Kolkata"})
    return result, calls


def test_confident_no_dr_stops_early() -> None:
    result, calls = _run(0, 0.92)
    assert calls == ["detector", "router", "screen_only", "report"]
    assert result["route"] == "screen_only"
    assert "No DR" in result["explanation"]
    assert result["report"] == "report for grade 0"


def test_low_confidence_no_dr_gets_explained() -> None:
    result, calls = _run(0, 0.41)
    assert calls == [
        "detector",
        "router",
        "explainer",
        "cost",
        "caregiver",
        "hospital_search",
        "report",
    ]
    assert result["route"] == "explain"


def test_no_dr_with_flag_gets_explained() -> None:
    result, calls = _run(0, 0.91, flags=["referable: threshold met"])
    assert "explainer" in calls
    assert result["route"] == "explain"


def test_dr_follows_full_path() -> None:
    result, calls = _run(2, 0.70)
    assert calls == [
        "detector",
        "router",
        "explainer",
        "cost",
        "caregiver",
        "hospital_search",
        "report",
    ]
    assert result["explanation"] == "Explained stage."
    assert result["care_plan"] == "Care plan."
    assert result["report"] == "report for grade 2"


def test_severe_grade_is_urgent() -> None:
    result, _ = _run(3, 0.55)
    assert result["urgent"] is True


def test_run_screening_returns_state_object() -> None:
    nodes, calls = make_fakes(0, 0.95)
    state = run_screening("scan.png", "Kolkata", graph=build_graph(**nodes))
    assert isinstance(state, DR_State)
    assert state.report is not None
    assert "detector" in calls


def test_default_graph_compiles_with_all_nodes() -> None:
    graph = build_graph()
    assert set(graph.get_graph().nodes) >= {
        "detector",
        "router",
        "screen_only",
        "explainer",
        "cost",
        "caregiver",
        "hospital_search",
        "report",
    }


if __name__ == "__main__":
    test_confident_no_dr_stops_early()
    test_low_confidence_no_dr_gets_explained()
    test_no_dr_with_flag_gets_explained()
    test_dr_follows_full_path()
    test_severe_grade_is_urgent()
    test_run_screening_returns_state_object()
    test_default_graph_compiles_with_all_nodes()
    print("graph tests passed")
