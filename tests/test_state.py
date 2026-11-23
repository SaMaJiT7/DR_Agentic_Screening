"""Smoke tests for the shared graph state.

Runs with either ``pytest`` or plain ``python tests/test_state.py``.
"""

from pydantic import ValidationError

from agents.state import (
    DR_State,
    Hospital,
    grade_label,
    make_initial_state,
    severity_band,
)


def test_initial_state() -> None:
    state = make_initial_state("scan.png", "Kolkata")
    assert state.image_path == "scan.png"
    assert state.city == "Kolkata"
    assert state.flags == []
    assert state.errors == []
    assert state.grade_label == "Unknown"
    assert state.severity_band == "unknown"


def test_missing_inputs_rejected() -> None:
    for args in (("", "Kolkata"), ("scan.png", "")):
        try:
            make_initial_state(*args)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"expected ValueError for {args!r}")


def test_grade_validation() -> None:
    try:
        DR_State(image_path="scan.png", city="Kolkata", grade=5)
    except ValidationError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValidationError for grade 5")

    try:
        DR_State(image_path="scan.png", city="Kolkata", probs=[0.1, 0.9])
    except ValidationError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValidationError for short probs")


def test_labels_and_bands() -> None:
    assert grade_label(0) == "No DR"
    assert grade_label(4) == "Proliferative"
    assert severity_band(0) == "none"
    assert severity_band(2) == "moderate"
    assert severity_band(4) == "severe"

    try:
        grade_label(5)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for grade 5")


def test_detector_result_round_trip() -> None:
    state = make_initial_state("scan.png", "Kolkata")
    state.grade = 2
    state.raw_grade = 2
    state.label = "Moderate"
    state.confidence = 0.72
    state.probs = [0.05, 0.13, 0.72, 0.07, 0.03]
    state.urgent = False
    state.referable = True
    state.flags = ["referable"]
    state.hospitals = [Hospital(name="Eye Care Centre", address="Salt Lake")]
    state.add_error("example warning")

    assert state.grade_label == "Moderate"
    assert state.severity_band == "moderate"
    assert state.hospitals[0].name == "Eye Care Centre"
    assert state.errors == ["example warning"]

    dumped = state.model_dump()
    assert dumped["probs"][2] == 0.72
    assert dumped["hospitals"][0]["name"] == "Eye Care Centre"


if __name__ == "__main__":
    test_initial_state()
    test_missing_inputs_rejected()
    test_grade_validation()
    test_labels_and_bands()
    test_detector_result_round_trip()
    print("state smoke tests passed")
