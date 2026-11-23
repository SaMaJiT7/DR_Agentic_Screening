"""Tests for the cost node and the cost KB validator."""

import json
import tempfile
from pathlib import Path

from agents.nodes.cost import (
    COST_KB,
    NO_TREATMENT_NOTE,
    cost_node,
    cost_plan,
    format_cost_options,
    load_cost_kb,
)
from agents.state import CostOption, make_initial_state

GOOD_KB = {
    "currency": "INR",
    "as_of": "2026-11",
    "source": "test",
    "notes": ["Prices are per the unit shown."],
    "treatments": [
        {
            "hospital": "Test Eye Hospital",
            "kind": "injection",
            "name": "Eylea",
            "active_ingredient": "aflibercept",
            "unit": "one package, per eye",
            "cost_inr": 39900,
        },
        {
            "hospital": "Test Eye Hospital",
            "kind": "laser",
            "name": "PRP laser",
            "unit": "one session",
            "cost_inr": 4300,
        },
    ],
}


def _state(grade=None):
    state = make_initial_state("scan.png", "Kolkata")
    if grade is not None:
        state.grade = grade
        state.raw_grade = grade
        state.label = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"][grade]
        state.confidence = 0.8
    return state


def _write(directory: str, payload) -> Path:
    path = Path(directory) / "cost_kb.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_no_cost_for_no_dr_and_mild() -> None:
    for grade in (0, 1):
        update = cost_node(_state(grade))
        assert update["cost_options"] == []
        assert update["cost_note"] == NO_TREATMENT_NOTE


def test_referable_grades_get_the_whole_table() -> None:
    for grade in (2, 3, 4):
        update = cost_node(_state(grade))
        assert len(update["cost_options"]) == len(COST_KB.options)
        assert all(isinstance(option, CostOption) for option in update["cost_options"])
        assert update["cost_note"] == COST_KB.note_text()


def test_cost_node_requires_grade() -> None:
    try:
        cost_node(_state())
    except ValueError as exc:
        assert "grade" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_table_lines_cover_every_treatment() -> None:
    lines = format_cost_options(COST_KB)
    assert len(lines) == len(COST_KB.options)
    assert any("Eylea" in line for line in lines)
    assert any("Accentrix" in line for line in lines)
    assert any("PRP laser" in line for line in lines)


def test_loads_a_valid_file() -> None:
    with tempfile.TemporaryDirectory() as directory:
        table = load_cost_kb(_write(directory, GOOD_KB))
    assert table.currency == "INR"
    assert {option.kind for option in table.options} == {"injection", "laser"}
    assert table.options[0].cost_inr == 39900


def test_rejects_missing_file() -> None:
    try:
        load_cost_kb("/nonexistent/cost_kb.json")
    except FileNotFoundError as exc:
        assert "COST_KB_PATH" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected FileNotFoundError")


def test_rejects_non_inr_currency() -> None:
    payload = dict(GOOD_KB, currency="USD")
    with tempfile.TemporaryDirectory() as directory:
        try:
            load_cost_kb(_write(directory, payload))
        except ValueError as exc:
            assert "INR" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


def test_rejects_missing_sections() -> None:
    payload = {key: value for key, value in GOOD_KB.items() if key != "treatments"}
    with tempfile.TemporaryDirectory() as directory:
        try:
            load_cost_kb(_write(directory, payload))
        except ValueError as exc:
            assert "missing sections" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


def test_rejects_non_positive_price() -> None:
    payload = json.loads(json.dumps(GOOD_KB))
    payload["treatments"][0]["cost_inr"] = 0
    with tempfile.TemporaryDirectory() as directory:
        try:
            load_cost_kb(_write(directory, payload))
        except ValueError as exc:
            assert "invalid" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


def test_rejects_unknown_kind() -> None:
    payload = json.loads(json.dumps(GOOD_KB))
    payload["treatments"][0]["kind"] = "surgery"
    with tempfile.TemporaryDirectory() as directory:
        try:
            load_cost_kb(_write(directory, payload))
        except ValueError as exc:
            assert "invalid" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


def test_rejects_duplicate_entries() -> None:
    payload = json.loads(json.dumps(GOOD_KB))
    payload["treatments"].append(dict(payload["treatments"][0], cost_inr=1000))
    with tempfile.TemporaryDirectory() as directory:
        try:
            load_cost_kb(_write(directory, payload))
        except ValueError as exc:
            assert "duplicate" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


def test_cost_plan_matches_node() -> None:
    assert cost_plan(_state(3)) == cost_node(_state(3))


if __name__ == "__main__":
    test_no_cost_for_no_dr_and_mild()
    test_referable_grades_get_the_whole_table()
    test_cost_node_requires_grade()
    test_table_lines_cover_every_treatment()
    test_loads_a_valid_file()
    test_rejects_missing_file()
    test_rejects_non_inr_currency()
    test_rejects_missing_sections()
    test_rejects_non_positive_price()
    test_rejects_unknown_kind()
    test_rejects_duplicate_entries()
    test_cost_plan_matches_node()
    print("cost tests passed")
