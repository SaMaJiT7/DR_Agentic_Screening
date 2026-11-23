"""Tests for the screen-only (confident No DR) node."""

from agents.nodes.screen_only import screen_only_node, screening_advice
from agents.state import DR_State


def _state(**overrides) -> DR_State:
    values = {
        "image_path": "scan.png",
        "city": "Kolkata",
        "grade": 0,
        "raw_grade": 0,
        "label": "No DR",
        "confidence": 0.91,
        "probs": [0.91, 0.04, 0.03, 0.01, 0.01],
    }
    values.update(overrides)
    return DR_State(**values)


def test_advice_says_no_dr_and_is_not_a_diagnosis() -> None:
    advice = screening_advice(_state())
    assert "No DR" in advice
    assert "not a diagnosis" in advice.lower()
    assert "OCT" in advice


def test_advice_includes_confidence() -> None:
    assert "0.91" in screening_advice(_state())


def test_node_fills_grounded_fields() -> None:
    update = screen_only_node(_state())
    assert update["urgent"] is False
    assert "No DR" in update["explanation"]
    assert "care_plan" not in update
    assert update["next_steps"]
    assert update["diet_advice"]
    assert update["red_flags"]
    assert update["high_blood_sugar_steps"]
    assert update["high_blood_pressure_steps"]
    assert update["reports_to_bring"]
    assert "consult a doctor" in update["consult_note"].lower()


def test_node_rejects_non_zero_grade() -> None:
    try:
        screen_only_node(_state(grade=2, label="Moderate"))
    except ValueError as exc:
        assert "grade 0" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_node_requires_grade() -> None:
    try:
        screen_only_node(_state(grade=None, raw_grade=None, label=None))
    except ValueError as exc:
        assert "grade" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    test_advice_says_no_dr_and_is_not_a_diagnosis()
    test_advice_includes_confidence()
    test_node_fills_grounded_fields()
    test_node_rejects_non_zero_grade()
    test_node_requires_grade()
    print("screen_only tests passed")
