"""Tests for loading the care knowledge base from JSON."""

import json
import tempfile
from pathlib import Path

from agents.config import settings
from agents.nodes.caregiver import CARE_KB, load_care_kb


def _valid_payload() -> dict:
    common = {
        "diet": ["diet tip"],
        "red_flags": ["red flag"],
        "high_blood_sugar": ["sugar step"],
        "high_blood_pressure": ["bp step"],
        "reports_to_bring": ["report to bring"],
    }
    grades = {
        str(grade): {
            "follow_up": f"follow up {grade}",
            "next_steps": [f"next step {grade}"],
            "consult_note": f"consult note {grade}",
            "emphasis": f"emphasis {grade}",
        }
        for grade in range(5)
    }
    return {"common": common, "grades": grades}


def _write_kb(payload, directory: str) -> Path:
    path = Path(directory) / "care_kb.json"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_default_file_is_configured_and_loaded() -> None:
    assert settings.care_kb_path.is_file()
    assert settings.care_kb_path.name == "care_kb.json"
    assert set(CARE_KB) == {0, 1, 2, 3, 4}
    for facts in CARE_KB.values():
        assert facts.follow_up and facts.consult_note and facts.emphasis
        assert facts.next_steps
        assert facts.diet and facts.red_flags
        assert facts.high_blood_sugar and facts.high_blood_pressure
        assert facts.reports_to_bring


def test_custom_path_is_loaded() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = _write_kb(_valid_payload(), directory)
        kb = load_care_kb(path)
    assert kb[2].follow_up == "follow up 2"
    assert kb[0].consult_note == "consult note 0"
    assert kb[4].high_blood_sugar == ("sugar step",)


def test_missing_file_has_clear_error() -> None:
    try:
        load_care_kb("/nonexistent/care_kb.json")
    except FileNotFoundError as exc:
        assert "CARE_KB_PATH" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected FileNotFoundError")


def test_invalid_json_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = _write_kb("{not json", directory)
        try:
            load_care_kb(path)
        except ValueError as exc:
            assert "valid JSON" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError for malformed JSON")


def test_missing_sections_rejected() -> None:
    payload = _valid_payload()
    del payload["common"]
    with tempfile.TemporaryDirectory() as directory:
        path = _write_kb(payload, directory)
        try:
            load_care_kb(path)
        except ValueError as exc:
            assert "missing sections" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError for missing sections")


def test_missing_grade_rejected() -> None:
    payload = _valid_payload()
    del payload["grades"]["3"]
    with tempfile.TemporaryDirectory() as directory:
        path = _write_kb(payload, directory)
        try:
            load_care_kb(path)
        except ValueError as exc:
            assert "missing grades" in str(exc)
            assert "3" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError for missing grade")


def test_empty_common_list_rejected() -> None:
    payload = _valid_payload()
    payload["common"]["reports_to_bring"] = []
    with tempfile.TemporaryDirectory() as directory:
        path = _write_kb(payload, directory)
        try:
            load_care_kb(path)
        except ValueError as exc:
            assert "reports_to_bring" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError for empty common list")


if __name__ == "__main__":
    test_default_file_is_configured_and_loaded()
    test_custom_path_is_loaded()
    test_missing_file_has_clear_error()
    test_invalid_json_rejected()
    test_missing_sections_rejected()
    test_missing_grade_rejected()
    test_empty_common_list_rejected()
    print("care KB tests passed")
