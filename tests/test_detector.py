"""Tests for the detector's scoring logic using fake model bundles."""

import tempfile
from pathlib import Path

from agents.nodes.detector import detect, load_bundle, predict
from agents.state import make_initial_state


class FakeModel:
    def __init__(self, probs) -> None:
        self.probs = probs
        self.seen = None

    def predict_proba(self, rows):
        self.seen = rows
        return [self.probs]


class FakeScaler:
    def __init__(self) -> None:
        self.called = False

    def transform(self, rows):
        self.called = True
        return [[value * 2 for value in row] for row in rows]


def _bundle(probs, *, t_severe=None, t_referable=None, scaler=None, key="svm"):
    bundle = {key: FakeModel(probs)}
    if scaler is not None:
        bundle["scaler"] = scaler
    if t_severe is not None:
        bundle["t_severe"] = t_severe
    if t_referable is not None:
        bundle["t_referable"] = t_referable
    return bundle


def test_no_escalation() -> None:
    result = predict(
        [0.0] * 4096,
        _bundle([0.80, 0.10, 0.05, 0.03, 0.02], t_severe=0.5, t_referable=0.5),
    )
    assert result["raw_grade"] == 0
    assert result["grade"] == 0
    assert result["label"] == "No DR"
    assert result["confidence"] == 0.80
    assert result["urgent"] is False
    assert result["referable"] is False
    assert result["flags"] == []
    assert result["t_severe"] == 0.5
    assert result["t_referable"] == 0.5


def test_urgent_escalates_grade() -> None:
    result = predict(
        [0.0] * 4096,
        _bundle([0.05, 0.05, 0.10, 0.50, 0.30], t_severe=0.5, t_referable=0.5),
    )
    assert result["raw_grade"] == 3
    assert result["grade"] == 3
    assert result["urgent"] is True
    assert result["referable"] is True
    assert any(flag.startswith("urgent:") for flag in result["flags"])
    assert any(flag.startswith("referable:") for flag in result["flags"])


def test_referable_escalates_low_raw_grade() -> None:
    result = predict(
        [0.0] * 4096,
        _bundle([0.40, 0.30, 0.20, 0.05, 0.05], t_severe=0.50, t_referable=0.25),
    )
    assert result["raw_grade"] == 0
    assert result["grade"] == 2
    assert result["label"] == "Moderate"
    assert result["urgent"] is False
    assert result["referable"] is True


def test_bundle_thresholds_override_defaults() -> None:
    strict = predict(
        [0.0] * 4096,
        _bundle([0.05, 0.05, 0.10, 0.50, 0.30], t_severe=0.95, t_referable=0.95),
    )
    assert strict["urgent"] is False
    assert strict["referable"] is False
    assert strict["grade"] == 3


def test_thresholds_fall_back_to_settings() -> None:
    # settings.t_severe default is 0.30; P(Severe+) here is 0.35.
    result = predict(
        [0.0] * 4096,
        _bundle([0.31, 0.29, 0.05, 0.20, 0.15]),
    )
    assert result["urgent"] is True
    assert result["t_severe"] == 0.30
    assert result["t_referable"] == 0.50


def test_threshold_arguments_override_bundle() -> None:
    result = predict(
        [0.0] * 4096,
        _bundle([0.05, 0.05, 0.10, 0.50, 0.30], t_severe=0.5, t_referable=0.5),
        t_severe=0.99,
        t_referable=0.99,
    )
    assert result["urgent"] is False
    assert result["referable"] is False


def test_scaler_and_bundle_aliases() -> None:
    scaler = FakeScaler()
    bundle = {
        "classifier": FakeModel([0.80, 0.10, 0.05, 0.03, 0.02]),
        "standard_scaler": scaler,
        "T_SEVERE": 0.5,
        "T_REFERABLE": 0.5,
    }
    result = predict([1.0, 2.0, 3.0], bundle)

    assert scaler.called is True
    assert bundle["classifier"].seen == [[2.0, 4.0, 6.0]]
    assert result["grade"] == 0


def test_model_without_probabilities_rejected() -> None:
    class NoProba:
        pass

    try:
        predict([0.0] * 4096, {"svm": NoProba()})
    except TypeError as exc:
        assert "probability=True" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected TypeError")


def test_wrong_class_count_rejected() -> None:
    try:
        predict([0.0] * 4096, _bundle([0.5, 0.5]))
    except ValueError as exc:
        assert "expected 5" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_detect_uses_injected_extractor() -> None:
    calls = []

    def fake_extractor(path):
        calls.append(path)
        return [1.0, 2.0]

    result = detect(
        "scan.png",
        bundle=_bundle([0.80, 0.10, 0.05, 0.03, 0.02], t_severe=0.5, t_referable=0.5),
        extractor=fake_extractor,
    )
    assert calls == ["scan.png"]
    assert result["grade"] == 0


def test_detect_propagates_bad_image() -> None:
    def bad_extractor(path):
        raise ValueError(f"could not read image: {path}")

    try:
        detect("missing.png", bundle=_bundle([0.2] * 5), extractor=bad_extractor)
    except ValueError as exc:
        assert "missing.png" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_load_bundle_missing_file() -> None:
    try:
        load_bundle("/nonexistent/dr_svm_bundle.joblib")
    except FileNotFoundError as exc:
        assert "DR_MODEL_PATH" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected FileNotFoundError")


def test_load_bundle_round_trip() -> None:
    import joblib

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bundle.joblib"
        joblib.dump({"t_severe": 0.4, "t_referable": 0.6}, path)
        loaded = load_bundle(path)
    assert loaded == {"t_severe": 0.4, "t_referable": 0.6}


if __name__ == "__main__":
    test_no_escalation()
    test_urgent_escalates_grade()
    test_referable_escalates_low_raw_grade()
    test_bundle_thresholds_override_defaults()
    test_thresholds_fall_back_to_settings()
    test_threshold_arguments_override_bundle()
    test_scaler_and_bundle_aliases()
    test_model_without_probabilities_rejected()
    test_wrong_class_count_rejected()
    test_detect_uses_injected_extractor()
    test_detect_propagates_bad_image()
    test_load_bundle_missing_file()
    test_load_bundle_round_trip()
    print("detector tests passed")
