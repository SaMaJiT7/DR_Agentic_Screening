"""Screen-only node: fast, deterministic advice for a confident No DR result.

Reached when the router sees ``grade == 0`` with confidence of at least
``ROUTER_STOP_CONFIDENCE`` and no safety flag. No LLM is called: the advice is
assembled from the grade-0 entry of the care knowledge base, so it cannot
invent facts and costs nothing to run.

Safety rule: a No DR result must say "No DR" explicitly **and** tell the
patient to consult a doctor to confirm it. It must never imply that no medical
follow-up is needed.
"""

from __future__ import annotations

from agents.nodes.caregiver import CARE_KB
from agents.state import DR_State, GRADE_LABELS


def screening_advice(state: DR_State) -> str:
    """Build the short No DR summary shown in the report."""

    if state.grade != 0:
        raise ValueError("screen_only node expects grade 0")

    confidence = f"{state.confidence:.2f}" if state.confidence is not None else "unknown"
    return (
        "No DR (no diabetic retinopathy) was seen in this image. "
        f"The model's confidence in this result is {confidence}. "
        "This is a screening result, not a diagnosis. Macular swelling (DME) "
        "cannot be seen by this model and needs an OCT scan."
    )


def screen_only_node(state: DR_State) -> dict:
    """LangGraph node: fill the care fields without calling an LLM."""

    if state.grade is None:
        raise ValueError("screen_only requires detector output (grade)")
    if state.grade != 0:
        raise ValueError(
            f"screen_only is only valid for grade 0, got {GRADE_LABELS.get(state.grade)!r}"
        )

    facts = CARE_KB[0]

    # No care_plan here: the deterministic lists below plus the consultation
    # note already carry the advice, and a third paragraph would just repeat it.
    return {
        "explanation": screening_advice(state),
        "next_steps": list(facts.next_steps),
        "diet_advice": list(facts.diet),
        "red_flags": list(facts.red_flags),
        "high_blood_sugar_steps": list(facts.high_blood_sugar),
        "high_blood_pressure_steps": list(facts.high_blood_pressure),
        "reports_to_bring": list(facts.reports_to_bring),
        "consult_note": facts.consult_note,
        "urgent": False,
    }
