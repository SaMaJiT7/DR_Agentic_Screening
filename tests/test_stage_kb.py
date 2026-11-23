"""Tests for loading the stage knowledge base from JSON."""

import json
import tempfile
from pathlib import Path

from agents.config import settings
from agents.nodes.explainer import STAGE_KB, load_stage_kb


def _write_kb(payload, directory: str, name: str = "stage_kb.json") -> Path:
    path = Path(directory) / name
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _valid_payload() -> dict:
    return {
        str(grade): {
            "summary": f"summary {grade}",
            "follow_up": f"follow up {grade}",
            "treatment": f"treatment {grade}",
            "recovery": f"recovery {grade}",
        }
        for grade in range(5)
    }


def test_default_file_is_configured_and_loaded() -> None:
    assert settings.stage_kb_path.is_file()
    assert settings.stage_kb_path.name == "stage_kb.json"
    assert set(STAGE_KB) == {0, 1, 2, 3, 4}
    for facts in STAGE_KB.values():
        assert facts.summary and facts.follow_up
        assert facts.treatment and facts.recovery


def test_custom_path_is_loaded() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = _write_kb(_valid_payload(), directory, "custom.json")
        kb = load_stage_kb(path)
    assert kb[2].summary == "summary 2"
    assert kb[4].follow_up == "follow up 4"


def test_missing_file_has_clear_error() -> None:
    try:
        load_stage_kb("/nonexistent/stage_kb.json")
    except FileNotFoundError as exc:
        assert "STAGE_KB_PATH" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected FileNotFoundError")


def test_invalid_json_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = _write_kb("{not json", directory)
        try:
            load_stage_kb(path)
        except ValueError as exc:
            assert "valid JSON" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError for malformed JSON")


def test_missing_grade_rejected() -> None:
    payload = _valid_payload()
    del payload["3"]
    with tempfile.TemporaryDirectory() as directory:
        path = _write_kb(payload, directory)
        try:
            load_stage_kb(path)
        except ValueError as exc:
            assert "missing grades" in str(exc)
            assert "3" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError for missing grade")


def test_bad_field_rejected() -> None:
    payload = _valid_payload()
    del payload["2"]["summary"]
    with tempfile.TemporaryDirectory() as directory:
        path = _write_kb(payload, directory)
        try:
            load_stage_kb(path)
        except ValueError as exc:
            assert "missing fields" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError for missing field")


if __name__ == "__main__":
    test_default_file_is_configured_and_loaded()
    test_custom_path_is_loaded()
    test_missing_file_has_clear_error()
    test_invalid_json_rejected()
    test_missing_grade_rejected()
    test_bad_field_rejected()
    print("stage KB tests passed")
