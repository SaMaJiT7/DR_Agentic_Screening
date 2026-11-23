"""Tests for the care-giver agent using an injected fake LLM."""

from types import SimpleNamespace

from agents.nodes.caregiver import (
    CARE_KB,
    COMMON_DIET,
    HIGH_BLOOD_PRESSURE_STEPS,
    HIGH_BLOOD_SUGAR_STEPS,
    RED_FLAGS,
    REPORTS_TO_BRING,
    build_care_prompt,
    run_caregiver,
)
from agents.state import CostOption, TreatmentBrief, make_initial_state


class FakeLLM:
    def __init__(self, text: str = "Here is your care plan.") -> None:
        self.text = text
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=self.text)


def _state(**kwargs):
    defaults = dict(grade=2, raw_grade=2, label="Moderate", confidence=0.72)
    defaults.update(kwargs)
    return make_initial_state("scan.png", "Kolkata").model_copy(update=defaults)


def test_care_table_covers_all_grades() -> None:
    assert set(CARE_KB) == {0, 1, 2, 3, 4}
    for facts in CARE_KB.values():
        assert facts.follow_up
        assert facts.next_steps and facts.diet and facts.red_flags
        for step in facts.next_steps:
            assert step.strip()


def test_severity_changes_follow_up_urgency() -> None:
    assert "12 months" in CARE_KB[1].follow_up
    assert "6 months" in CARE_KB[2].follow_up
    assert "weeks" in CARE_KB[3].follow_up
    assert "Urgent" in CARE_KB[4].follow_up


def test_prompt_is_grounded_in_tables() -> None:
    state = _state(grade=2, explanation="You have moderate changes.")
    messages = build_care_prompt(state)
    assert [role for role, _ in messages] == ["system", "human"]

    system, human = messages[0][1], messages[1][1]

    assert "NOT a doctor" in system
    assert "do not invent" in system.lower()
    assert "OCT" in human
    assert "Ask whether you need an OCT scan" in human
    assert COMMON_DIET[0] in human
    assert RED_FLAGS[0] in human
    assert "You have moderate changes." in human
    assert "Grade: 2" in human


def test_prompt_includes_high_sugar_bp_and_reports() -> None:
    human = build_care_prompt(_state(grade=2))[1][1]
    assert HIGH_BLOOD_SUGAR_STEPS[0] in human
    assert HIGH_BLOOD_PRESSURE_STEPS[0] in human
    assert REPORTS_TO_BRING[0] in human
    assert CARE_KB[2].consult_note in human


def test_no_dr_still_requires_doctor_consultation() -> None:
    messages = build_care_prompt(_state(grade=0, label="No DR"))
    system, human = messages[0][1], messages[1][1]

    assert "NO DR" in human
    assert "must still consult a doctor" in human.lower()
    assert "No diabetic retinopathy was seen" in CARE_KB[0].consult_note
    assert "Never say that no medical follow-up is needed" in system


def test_prompt_requires_grade() -> None:
    state = make_initial_state("scan.png", "Kolkata")
    try:
        build_care_prompt(state)
    except ValueError as exc:
        assert "grade" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError before detection")


def test_run_caregiver_fills_state_slice() -> None:
    llm = FakeLLM("Book an eye appointment and eat more vegetables.")
    update = run_caregiver(_state(grade=3), llm)

    assert update["care_plan"].startswith("Book an eye appointment")
    assert update["next_steps"] == list(CARE_KB[3].next_steps)
    assert update["diet_advice"] == list(COMMON_DIET)
    assert update["red_flags"] == list(RED_FLAGS)
    assert update["high_blood_sugar_steps"] == list(HIGH_BLOOD_SUGAR_STEPS)
    assert update["high_blood_pressure_steps"] == list(HIGH_BLOOD_PRESSURE_STEPS)
    assert update["reports_to_bring"] == list(REPORTS_TO_BRING)
    assert update["consult_note"] == CARE_KB[3].consult_note
    assert llm.messages == build_care_prompt(_state(grade=3))


def test_prompt_includes_quoted_costs_and_briefs() -> None:
    state = _state(
        grade=3,
        cost_options=[
            CostOption(
                hospital="ASG Eye Hospital",
                kind="injection",
                name="Eylea",
                active_ingredient="aflibercept",
                unit="one package, per eye",
                cost_inr=39900,
            ),
            CostOption(
                hospital="ASG Eye Hospital",
                kind="laser",
                name="PRP laser (panretinal photocoagulation)",
                unit="one session",
                cost_inr=4300,
            ),
        ],
        cost_note="Prices exclude consultation and OCT scans.",
        treatment_briefs=[
            TreatmentBrief(
                name="Vitrectomy",
                purpose="bleeding that does not clear",
                summary="Surgery to remove blood from inside the eye.",
            )
        ],
    )
    system, human = build_care_prompt(state)[0][1], build_care_prompt(state)[1][1]

    assert "₹39,900" in human
    assert "₹4,300" in human
    assert "per eye" in human
    assert "Prices exclude consultation" in human
    assert "Vitrectomy" in human
    assert "NO prices" in human
    assert "Never invent a price" in system


def test_prompt_has_no_cost_block_without_costs() -> None:
    human = build_care_prompt(_state(grade=2))[1][1]
    assert "Possible injection costs" not in human
    assert "Other treatments the eye doctor" not in human


def test_urgent_grade_prompt_stresses_urgency() -> None:
    human = build_care_prompt(_state(grade=4, label="Proliferative"))[1][1]
    assert "Urgent ophthalmology review" in human
    assert "within days" in human


def test_truncated_llm_output_rejected() -> None:
    class Truncated:
        content = "Partial care plan that stops mid-sentence, price ₹39,"
        response_metadata = {"finish_reason": "length"}

    class LLM:
        def invoke(self, messages):
            return Truncated()

    try:
        run_caregiver(_state(grade=3), LLM())
    except RuntimeError as exc:
        assert "truncated" in str(exc)
        assert "LLM_MAX_TOKENS" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError for truncated output")


def test_empty_llm_response_rejected() -> None:
    try:
        run_caregiver(_state(), FakeLLM("   "))
    except RuntimeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError for empty LLM output")


if __name__ == "__main__":
    test_care_table_covers_all_grades()
    test_severity_changes_follow_up_urgency()
    test_prompt_is_grounded_in_tables()
    test_prompt_includes_high_sugar_bp_and_reports()
    test_no_dr_still_requires_doctor_consultation()
    test_prompt_requires_grade()
    test_prompt_includes_quoted_costs_and_briefs()
    test_prompt_has_no_cost_block_without_costs()
    test_run_caregiver_fills_state_slice()
    test_truncated_llm_output_rejected()
    test_urgent_grade_prompt_stresses_urgency()
    test_empty_llm_response_rejected()
    print("care-giver tests passed")
