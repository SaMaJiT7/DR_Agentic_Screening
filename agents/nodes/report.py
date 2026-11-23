"""Report node: deterministic, plain-language report from the shared state.

No LLM is involved, so the layout, the numbers and the disclaimer are stable
and testable. Every report:
  * states the grade and what it means,
  * repeats the consultation advice,
  * lists the care steps, diet, high-sugar/BP steps, reports to bring and
    emergency warning signs,
  * lists only hospitals the search actually returned,
  * says it is a screening aid, not a diagnosis, and that macular swelling
    (DME) needs an OCT scan.
"""

from __future__ import annotations

from agents.nodes.cost import format_cost_option, format_treatment_brief
from agents.state import DR_State, GRADE_LABELS

DISCLAIMER = (
    "This is an AI screening result, not a diagnosis. It cannot see macular "
    "swelling (DME), which needs an OCT scan, and it can miss disease. Always "
    "confirm the result with a qualified ophthalmologist."
)


def _bullet_list(items: list[str] | tuple[str, ...]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _section(title: str, body: str, level: int = 2) -> str:
    body = (body or "").strip()
    if not body:
        return ""
    return f"{'#' * level} {title}\n\n{body}\n"


def _result_lines(state: DR_State) -> str:
    if state.grade is None:
        raise ValueError("report requires detector output (grade)")

    lines = [f"**Result: grade {state.grade} — {GRADE_LABELS[state.grade]}**"]
    if state.confidence is not None:
        lines.append(f"- Confidence in the model's top class: {state.confidence:.2f}")
    if state.raw_grade is not None and state.raw_grade != state.grade:
        lines.append(
            f"- The model's raw top class was {GRADE_LABELS[state.raw_grade]}; "
            "the safety thresholds raised the grade."
        )
    if state.t_severe is not None and state.t_referable is not None:
        lines.append(
            f"- Thresholds used: T_SEVERE={state.t_severe:.3f}, "
            f"T_REFERABLE={state.t_referable:.3f}"
        )
    if state.flags:
        lines.append(f"- Safety flags: {'; '.join(state.flags)}")
    return "\n".join(lines)


def _hospital_lines(state: DR_State) -> str:
    if not state.hospitals:
        return (
            f"No eye hospitals were found automatically for **{state.city}**. "
            "Ask your doctor for a referral, or search for an eye hospital or "
            "ophthalmologist near you."
        )
    lines = [f"Places found near **{state.city}** (from OpenStreetMap, no ratings):"]
    for hospital in state.hospitals:
        if hospital.address:
            lines.append(f"- **{hospital.name}** — {hospital.address}")
        else:
            lines.append(f"- **{hospital.name}**")
    return "\n".join(lines)


def _cost_lines(state: DR_State) -> str:
    """Render the quoted prices, or nothing when no injection applies."""

    if not state.cost_options:
        return ""

    lines = [
        "These are real hospital quotes for anti-VEGF injections and PRP laser "
        "sessions. They apply **only if an ophthalmologist recommends that "
        "treatment**, which this screening tool cannot decide."
    ]
    for option in state.cost_options:
        lines.append(f"- {format_cost_option(option)}")
    if state.cost_note:
        lines.append(f"\n{state.cost_note}")
    return "\n".join(lines)


def _other_treatment_lines(state: DR_State) -> str:
    """Describe the unpriced treatments without attaching any figure."""

    if not state.treatment_briefs:
        return ""

    lines = ["Your eye doctor may also discuss these, depending on what they find:"]
    for brief in state.treatment_briefs:
        lines.append(f"- {format_treatment_brief(brief)}")
    lines.append("\nNo price is quoted for these treatments in this report.")
    return "\n".join(lines)


def build_report(state: DR_State) -> str:
    """Render the whole report for one finished run."""

    if state.grade is None:
        raise ValueError("report requires detector output (grade)")

    parts: list[str] = ["# Diabetic Retinopathy Screening Report\n"]

    if state.grade == 0:
        parts.append(
            "**No DR (no diabetic retinopathy) was seen in this image. You must "
            "still consult a doctor to confirm this result.**\n"
        )

    parts.append(_result_lines(state) + "\n")
    parts.append(_section("What this result means", state.explanation or ""))
    parts.append(_section("Care plan", state.care_plan or ""))
    parts.append(
        _section(
            "Possible treatment costs (injections and PRP laser, only if recommended)",
            _cost_lines(state),
        )
    )
    parts.append(
        _section(
            "Other treatments your eye doctor may discuss",
            _other_treatment_lines(state),
        )
    )

    if state.consult_note:
        parts.append(_section("Consultation advice", state.consult_note))

    parts.append(_section("Next steps", _bullet_list(state.next_steps)))
    parts.append(_section("Diet and lifestyle", _bullet_list(state.diet_advice)))
    parts.append(
        _section(
            "If your blood sugar is high",
            _bullet_list(state.high_blood_sugar_steps),
        )
    )
    parts.append(
        _section(
            "If your blood pressure is high",
            _bullet_list(state.high_blood_pressure_steps),
        )
    )
    parts.append(
        _section(
            "Reports and tests to bring to the consultation",
            _bullet_list(state.reports_to_bring),
        )
    )
    parts.append(
        _section(
            "Warning signs — seek emergency eye care today",
            _bullet_list(state.red_flags),
        )
    )
    parts.append(_section(f"Eye hospitals near {state.city}", _hospital_lines(state)))
    parts.append(_section("Important", DISCLAIMER))

    if state.errors:
        parts.append(
            _section(
                "Notes about this run",
                _bullet_list(state.errors),
                level=3,
            )
        )

    return "\n".join(part for part in parts if part.strip()).strip() + "\n"


def report_node(state: DR_State) -> dict:
    """LangGraph node: fill ``report``."""

    return {"report": build_report(state)}
