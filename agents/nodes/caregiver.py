"""Care-giver agent: next steps, diet, high sugar/BP guidance and reports.

This is the first half of the care-navigator role in the README. All clinical
content is loaded from ``care_kb.json`` so the LLM cannot invent advice, diet
rules, hospitals, prices or medicine dosages.

Input: ``grade`` (required), ``label``, ``explanation`` (optional).
Output: ``care_plan`` (LLM prose) plus deterministic lists for next steps,
diet, high blood sugar, high blood pressure, reports to bring and red flags.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from agents.config import settings
from agents.llm import check_finish_reason, get_llm
from agents.nodes.cost import COST_KB, format_cost_option, format_treatment_brief
from agents.state import DR_State, GRADE_LABELS


@dataclass(frozen=True)
class CareFacts:
    """Grounded facts the care-giver may use for one grade."""

    follow_up: str
    next_steps: tuple[str, ...]
    diet: tuple[str, ...]
    red_flags: tuple[str, ...]
    high_blood_sugar: tuple[str, ...]
    high_blood_pressure: tuple[str, ...]
    reports_to_bring: tuple[str, ...]
    consult_note: str
    emphasis: str


_COMMON_LISTS = (
    "diet",
    "red_flags",
    "high_blood_sugar",
    "high_blood_pressure",
    "reports_to_bring",
)
_GRADE_STRINGS = ("follow_up", "consult_note", "emphasis")
_VALID_GRADES = frozenset(range(5))


def _require_string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string")
    return value.strip()


def _require_string_list(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} must be a non-empty list of strings")
    return tuple(_require_string(item, f"{where}[{index}]") for index, item in enumerate(value))


@lru_cache(maxsize=8)
def _load_care_kb(path: str) -> dict[int, CareFacts]:
    """Load, validate and cache one care KB file."""

    kb_path = Path(path)
    if not kb_path.is_file():
        raise FileNotFoundError(
            f"care knowledge base not found: {kb_path} "
            "(set CARE_KB_PATH to a valid JSON file)"
        )

    try:
        raw = json.loads(kb_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"care knowledge base is not valid JSON: {kb_path}") from exc

    if not isinstance(raw, dict):
        raise ValueError("care knowledge base must be a JSON object")

    missing_sections = {"common", "grades"} - raw.keys()
    if missing_sections:
        raise ValueError(f"care knowledge base is missing sections: {sorted(missing_sections)}")

    common = raw["common"]
    grades = raw["grades"]
    if not isinstance(common, dict):
        raise ValueError("care knowledge base 'common' must be an object")
    if not isinstance(grades, dict):
        raise ValueError("care knowledge base 'grades' must be an object")

    common_values = {
        name: _require_string_list(common.get(name), f"common.{name}")
        for name in _COMMON_LISTS
    }

    kb: dict[int, CareFacts] = {}
    for grade_key, facts in grades.items():
        try:
            grade = int(grade_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid grade key {grade_key!r}") from exc
        if grade not in _VALID_GRADES:
            raise ValueError(f"grade key out of range 0-4: {grade}")
        if not isinstance(facts, dict):
            raise ValueError(f"facts for grade {grade} must be an object")

        missing = set(_GRADE_STRINGS) - facts.keys()
        extra = facts.keys() - set(_GRADE_STRINGS) - {"next_steps"}
        if missing:
            raise ValueError(f"grade {grade} is missing fields: {sorted(missing)}")
        if extra:
            raise ValueError(f"grade {grade} has unknown fields: {sorted(extra)}")

        kb[grade] = CareFacts(
            follow_up=_require_string(facts["follow_up"], f"grades.{grade}.follow_up"),
            next_steps=_require_string_list(
                facts.get("next_steps"), f"grades.{grade}.next_steps"
            ),
            consult_note=_require_string(
                facts["consult_note"], f"grades.{grade}.consult_note"
            ),
            emphasis=_require_string(facts["emphasis"], f"grades.{grade}.emphasis"),
            **common_values,
        )

    missing_grades = _VALID_GRADES - kb.keys()
    if missing_grades:
        raise ValueError(f"care knowledge base is missing grades: {sorted(missing_grades)}")

    return kb


def load_care_kb(path: str | Path | None = None) -> dict[int, CareFacts]:
    """Load and validate the care KB.

    The path is ``CARE_KB_PATH`` unless one is passed explicitly. Results are
    cached per resolved path.
    """

    kb_path = Path(path) if path is not None else settings.care_kb_path
    return dict(_load_care_kb(str(kb_path.expanduser().resolve())))


#: Default KB loaded at import time; kept for convenience and back-compat.
CARE_KB: dict[int, CareFacts] = load_care_kb()

COMMON_DIET: tuple[str, ...] = CARE_KB[0].diet
RED_FLAGS: tuple[str, ...] = CARE_KB[0].red_flags
HIGH_BLOOD_SUGAR_STEPS: tuple[str, ...] = CARE_KB[0].high_blood_sugar
HIGH_BLOOD_PRESSURE_STEPS: tuple[str, ...] = CARE_KB[0].high_blood_pressure
REPORTS_TO_BRING: tuple[str, ...] = CARE_KB[0].reports_to_bring

SYSTEM_PROMPT = """You are a careful care-giver assistant for an AI \
diabetic-retinopathy screening tool. You are NOT a doctor and you must NOT \
diagnose, prescribe medicines, give dosages, or promise an outcome.

Use only the facts provided below. Do not invent hospitals, prices, phone \
numbers, statistics, treatments or foods that are not in the list.

Write for a patient with no medical background, in plain language, under 350 \
words. Include:
1. The result and consultation advice. If the grade is 0, say clearly that no \
diabetic retinopathy was seen, then state that they MUST still consult a doctor \
to confirm the result. Never say that no medical follow-up is needed.
2. The most important next steps, starting with the eye appointment and its \
urgency.
3. Practical diet guidance from the list, in your own simple words.
4. A short "if your blood sugar is high" section with the steps provided.
5. A short "if your blood pressure is high" section with the steps provided.
6. The reports and tests to do and bring to the consultation.
7. Possible treatment costs, but only if a cost list is provided below. Quote \
only the listed figures, say they are per eye or per session as stated, and say \
that they exclude consultation, OCT scans, the procedure fee and follow-up \
visits. Never invent a price, a total or a discount.
8. Other treatments, but only if that list is provided below. Describe each in \
one short sentence and attach NO price to them.
9. The warning signs that mean they should seek emergency eye care today.
10. A reminder that this is an AI screening result, not a diagnosis.
"""


def _require_grade(state: DR_State) -> int:
    if state.grade is None:
        raise ValueError("care-giver requires detector output (grade)")
    return state.grade


def _bullets(items: tuple[str, ...]) -> str:
    return "\n".join(f"- {item}" for item in items)


def build_care_prompt(state: DR_State) -> list[tuple[str, str]]:
    """Build the chat messages for the care-giver."""

    grade = _require_grade(state)
    facts = CARE_KB[grade]
    label = state.label or GRADE_LABELS[grade]

    explanation = (state.explanation or "").strip()
    explanation_block = (
        f"\nThe explainer told the patient:\n{explanation}\n"
        if explanation
        else ""
    )
    no_dr_block = (
        "\nIMPORTANT: the detector found NO DR (no diabetic retinopathy). "
        "Say this clearly, but you MUST still tell the patient to consult a "
        "doctor to confirm the result. Do not imply that no medical follow-up "
        "is needed.\n"
        if grade == 0
        else ""
    )

    if state.cost_options:
        cost_lines = "\n".join(
            f"- {format_cost_option(option, COST_KB.currency)}"
            for option in state.cost_options
        )
        cost_block = (
            "\nPossible injection costs (the only prices you may quote):\n"
            f"{cost_lines}\n"
            f"Price caveats (must be repeated in your own words): "
            f"{state.cost_note or COST_KB.note_text()}\n"
        )
    else:
        cost_block = ""

    if state.treatment_briefs:
        brief_lines = "\n".join(
            f"- {format_treatment_brief(brief)}" for brief in state.treatment_briefs
        )
        brief_block = (
            "\nOther treatments the eye doctor may discuss (describe each in one "
            f"short sentence, with NO prices):\n{brief_lines}\n"
        )
    else:
        brief_block = ""

    human = f"""Patient's screening result:
- Grade: {grade} ({label})
- Follow-up: {facts.follow_up}
- Consultation note (must be included): {facts.consult_note}
- Diet and lifestyle emphasis: {facts.emphasis}
{explanation_block}{no_dr_block}{cost_block}{brief_block}
Next steps you may use:
{_bullets(facts.next_steps)}

Diet guidance you may use:
{_bullets(facts.diet)}

If blood sugar is high, the steps to take are:
{_bullets(facts.high_blood_sugar)}

If blood pressure is high, the steps to take are:
{_bullets(facts.high_blood_pressure)}

Reports and tests to do and bring to the consultation:
{_bullets(facts.reports_to_bring)}

Emergency warning signs you must include:
{_bullets(facts.red_flags)}

Write the care plan now."""

    return [("system", SYSTEM_PROMPT), ("human", human)]


def run_caregiver(state: DR_State, llm) -> dict:
    """Run the care-giver with an injected chat model (testable)."""

    grade = _require_grade(state)
    facts = CARE_KB[grade]

    response = llm.invoke(build_care_prompt(state))
    check_finish_reason(response, "care-giver")
    text = (getattr(response, "content", response) or "").strip()
    if not text:
        raise RuntimeError("care-giver LLM returned an empty response")

    return {
        "care_plan": text,
        "next_steps": list(facts.next_steps),
        "diet_advice": list(facts.diet),
        "red_flags": list(facts.red_flags),
        "high_blood_sugar_steps": list(facts.high_blood_sugar),
        "high_blood_pressure_steps": list(facts.high_blood_pressure),
        "reports_to_bring": list(facts.reports_to_bring),
        "consult_note": facts.consult_note,
    }


def caregiver_node(state: DR_State) -> dict:
    """LangGraph node: fill the care plan and its grounded lists."""

    return run_caregiver(state, get_llm())
