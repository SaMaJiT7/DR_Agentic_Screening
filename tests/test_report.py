"""Tests for the deterministic report template."""

from agents.nodes.report import DISCLAIMER, build_report, report_node
from agents.state import CostOption, DR_State, Hospital, TreatmentBrief


def _state(**overrides) -> DR_State:
    values = {
        "image_path": "scan.png",
        "city": "Kolkata",
        "grade": 3,
        "raw_grade": 2,
        "label": "Severe",
        "confidence": 0.61,
        "probs": [0.05, 0.10, 0.24, 0.45, 0.16],
        "urgent": True,
        "referable": True,
        "flags": ["urgent: P(Severe or worse)=0.61 >= T_SEVERE=0.08"],
        "t_severe": 0.084,
        "t_referable": 0.368,
        "explanation": "Severe changes were seen.",
        "care_plan": "See an ophthalmologist within weeks.",
        "next_steps": ["Consult an ophthalmologist within weeks."],
        "diet_advice": ["Fill half the plate with vegetables."],
        "red_flags": ["Sudden loss of vision."],
        "high_blood_sugar_steps": ["Check your blood sugar more often."],
        "high_blood_pressure_steps": ["Reduce salt."],
        "reports_to_bring": ["Recent HbA1c report."],
        "consult_note": "Severe changes were seen. Consult an ophthalmologist.",
        "hospitals": [Hospital(name="AIIMS", address="Baghajatin, Kolkata")],
        "cost_options": [
            CostOption(
                hospital="ASG Eye Hospital",
                kind="injection",
                name="Eylea",
                active_ingredient="aflibercept",
                unit="one package, per eye",
                cost_inr=39900,
            ),
            CostOption(
                hospital="Renuka",
                kind="laser",
                name="PRP laser (panretinal photocoagulation)",
                unit="one session for both eyes",
                cost_inr=3000,
            ),
        ],
        "cost_note": "Prices are per the unit shown and exclude consultation.",
        "treatment_briefs": [
            TreatmentBrief(
                name="Vitrectomy",
                purpose="bleeding that does not clear",
                summary="Surgery to remove blood from inside the eye.",
            )
        ],
    }
    values.update(overrides)
    return DR_State(**values)


def test_report_has_all_sections() -> None:
    report = build_report(_state())
    for heading in (
        "# Diabetic Retinopathy Screening Report",
        "## What this result means",
        "## Care plan",
        "## Consultation advice",
        "## Next steps",
        "## Diet and lifestyle",
        "## If your blood sugar is high",
        "## If your blood pressure is high",
        "## Reports and tests to bring to the consultation",
        "## Warning signs",
        "## Eye hospitals near Kolkata",
        "## Important",
    ):
        assert heading in report, heading


def test_report_shows_quoted_prices_and_briefs() -> None:
    report = build_report(_state())
    assert "## Possible treatment costs" in report
    assert "only if an ophthalmologist recommends" in report
    assert "₹39,900" in report
    assert "₹3,000" in report
    assert "both eyes" in report
    assert "Prices are per the unit shown" in report

    assert "## Other treatments your eye doctor may discuss" in report
    assert "Vitrectomy" in report
    assert "No price is quoted for these treatments" in report


def test_report_omits_cost_sections_when_empty() -> None:
    report = build_report(
        _state(cost_options=[], cost_note=None, treatment_briefs=[])
    )
    assert "## Possible treatment costs" not in report
    assert "## Other treatments your eye doctor may discuss" not in report


def test_report_shows_escalation_and_thresholds() -> None:
    report = build_report(_state())
    assert "grade 3" in report
    assert "Severe" in report
    assert "raised the grade" in report
    assert "T_SEVERE=0.084" in report


def test_report_lists_only_returned_hospitals() -> None:
    report = build_report(_state())
    assert "AIIMS" in report
    assert "no ratings" in report


def test_report_without_hospitals_offers_advice() -> None:
    report = build_report(_state(hospitals=[]))
    assert "No eye hospitals were found" in report
    assert "referral" in report


def test_no_dr_report_is_explicit_and_requires_doctor() -> None:
    report = build_report(
        _state(
            grade=0,
            raw_grade=0,
            label="No DR",
            confidence=0.88,
            urgent=False,
            referable=False,
            flags=[],
        )
    )
    assert "No DR (no diabetic retinopathy) was seen" in report
    assert "must still consult a doctor" in report
    assert "not a diagnosis" in report.lower()


def test_report_always_contains_disclaimer() -> None:
    assert DISCLAIMER in build_report(_state())
    assert "OCT" in build_report(_state())


def test_report_surfaces_errors() -> None:
    report = build_report(_state(errors=["hospital search failed: timeout"]))
    assert "Notes about this run" in report
    assert "hospital search failed" in report


def test_report_requires_grade() -> None:
    try:
        build_report(_state(grade=None, raw_grade=None, label=None))
    except ValueError as exc:
        assert "grade" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_report_node_returns_report_field() -> None:
    update = report_node(_state())
    assert list(update) == ["report"]
    assert "Severe" in update["report"]


if __name__ == "__main__":
    test_report_has_all_sections()
    test_report_shows_quoted_prices_and_briefs()
    test_report_omits_cost_sections_when_empty()
    test_report_shows_escalation_and_thresholds()
    test_report_lists_only_returned_hospitals()
    test_report_without_hospitals_offers_advice()
    test_no_dr_report_is_explicit_and_requires_doctor()
    test_report_always_contains_disclaimer()
    test_report_surfaces_errors()
    test_report_requires_grade()
    test_report_node_returns_report_field()
    print("report tests passed")
