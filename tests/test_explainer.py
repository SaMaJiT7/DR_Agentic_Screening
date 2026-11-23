"""Tests for the explainer agent using an injected fake LLM."""

from types import SimpleNamespace

from agents.nodes.explainer import (
    STAGE_KB,
    build_prompt,
    compute_urgent,
    explain,
)
from agents.state import make_initial_state


class FakeLLM:
    def __init__(self, text: str = "This is an AI screening result.") -> None:
        self.text = text
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=self.text)


def _detected(**kwargs):
    defaults = dict(
        grade=2,
        raw_grade=2,
        label="Moderate",
        confidence=0.72,
        probs=[0.05, 0.13, 0.72, 0.07, 0.03],
    )
    defaults.update(kwargs)
    return make_initial_state("scan.png", "Kolkata").model_copy(update=defaults)


def test_stage_table_covers_all_grades() -> None:
    assert set(STAGE_KB) == {0, 1, 2, 3, 4}
    for facts in STAGE_KB.values():
        assert facts.summary and facts.follow_up
        assert facts.treatment and facts.recovery


def test_urgency_is_deterministic() -> None:
    assert compute_urgent(_detected(grade=0)) is False
    assert compute_urgent(_detected(grade=2)) is False
    assert compute_urgent(_detected(grade=3)) is True
    assert compute_urgent(_detected(grade=4)) is True
    # A detector flag wins even if the grade is lower.
    assert compute_urgent(_detected(grade=1, urgent=True)) is True


def test_prompt_is_grounded_and_includes_safety_limits() -> None:
    state = _detected(grade=3, raw_grade=2, label="Severe", confidence=0.81)
    messages = build_prompt(state)
    assert [role for role, _ in messages] == ["system", "human"]

    system = messages[0][1]
    human = messages[1][1]

    assert "AI screening result" in system
    assert "OCT" in system
    assert "DME" in system
    assert "do not invent" in system.lower()

    assert "Grade: 3" in human
    assert "Severe" in human
    assert "0.810" in human
    assert STAGE_KB[3].summary in human
    assert "Urgent (severe or worse): yes" in human


def test_prompt_requires_detector_output() -> None:
    state = make_initial_state("scan.png", "Kolkata")
    try:
        build_prompt(state)
    except ValueError as exc:
        assert "grade" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError before detection")


def test_explain_fills_state_slice() -> None:
    llm = FakeLLM("Your scan shows moderate changes. This is an AI screening result.")
    state = _detected(grade=2)
    update = explain(state, llm)

    assert update["explanation"].startswith("Your scan shows moderate changes.")
    assert update["urgent"] is False
    assert llm.messages == build_prompt(state)


def test_explain_marks_severe_results_urgent() -> None:
    update = explain(_detected(grade=3, raw_grade=3), FakeLLM())
    assert update["urgent"] is True


def test_empty_llm_response_rejected() -> None:
    try:
        explain(_detected(), FakeLLM("   "))
    except RuntimeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError for empty LLM output")


def test_truncated_llm_output_rejected() -> None:
    class Truncated:
        content = "Partial explanation"
        response_metadata = {"finish_reason": "length"}

    class LLM:
        def invoke(self, messages):
            return Truncated()

    try:
        explain(_detected(), LLM())
    except RuntimeError as exc:
        assert "truncated" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError for truncated output")


if __name__ == "__main__":
    test_stage_table_covers_all_grades()
    test_urgency_is_deterministic()
    test_prompt_is_grounded_and_includes_safety_limits()
    test_prompt_requires_detector_output()
    test_explain_fills_state_slice()
    test_explain_marks_severe_results_urgent()
    test_empty_llm_response_rejected()
    test_truncated_llm_output_rejected()
    print("explainer tests passed")
