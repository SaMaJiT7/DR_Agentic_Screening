"""Explainer agent: plain-language explanation of a detected DR stage.

Input: the detector's ``grade``, ``raw_grade``, ``label``, ``confidence``
(plus probabilities and safety flags, when present).

Output: ``explanation`` (LLM-written) and ``urgent`` (deterministic).

The LLM is grounded in a small stage fact table. It is told not to invent
facts, not to diagnose, and to disclose the limits of an AI screening result
(especially that macular oedema/DME cannot be seen and needs an OCT scan).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from agents.config import settings
from agents.llm import check_finish_reason, get_llm
from agents.state import DR_State, GRADE_LABELS, SEVERE_GRADE


@dataclass(frozen=True)
class StageFacts:
    """Ground-truth facts the explainer may draw on for one grade."""

    summary: str
    follow_up: str
    treatment: str
    recovery: str


_REQUIRED_FIELDS = frozenset(StageFacts.__dataclass_fields__)
_VALID_GRADES = frozenset(range(5))


@lru_cache(maxsize=8)
def _load_stage_kb(path: str) -> dict[int, StageFacts]:
    """Load, validate and cache one stage KB file."""

    kb_path = Path(path)
    if not kb_path.is_file():
        raise FileNotFoundError(
            f"stage knowledge base not found: {kb_path} "
            "(set STAGE_KB_PATH to a valid JSON file)"
        )

    try:
        raw = json.loads(kb_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"stage knowledge base is not valid JSON: {kb_path}") from exc

    if not isinstance(raw, dict):
        raise ValueError("stage knowledge base must be a JSON object keyed by grade")

    kb: dict[int, StageFacts] = {}
    for grade_key, facts in raw.items():
        try:
            grade = int(grade_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid grade key {grade_key!r}") from exc
        if grade not in _VALID_GRADES:
            raise ValueError(f"grade key out of range 0-4: {grade}")
        if not isinstance(facts, dict):
            raise ValueError(f"facts for grade {grade} must be an object")

        missing = _REQUIRED_FIELDS - facts.keys()
        extra = facts.keys() - _REQUIRED_FIELDS
        if missing:
            raise ValueError(f"grade {grade} is missing fields: {sorted(missing)}")
        if extra:
            raise ValueError(f"grade {grade} has unknown fields: {sorted(extra)}")

        for field in _REQUIRED_FIELDS:
            if not isinstance(facts[field], str) or not facts[field].strip():
                raise ValueError(f"grade {grade} field {field!r} must be a non-empty string")

        kb[grade] = StageFacts(**{field: facts[field] for field in _REQUIRED_FIELDS})

    missing_grades = _VALID_GRADES - kb.keys()
    if missing_grades:
        raise ValueError(f"stage knowledge base is missing grades: {sorted(missing_grades)}")

    return kb


def load_stage_kb(path: str | Path | None = None) -> dict[int, StageFacts]:
    """Load and validate the stage KB.

    The path is ``STAGE_KB_PATH`` unless one is passed explicitly. Results are
    cached per resolved path.
    """

    kb_path = Path(path) if path is not None else settings.stage_kb_path
    return dict(_load_stage_kb(str(kb_path.expanduser().resolve())))


#: Default KB loaded at import time; kept for convenience and back-compat.
STAGE_KB: dict[int, StageFacts] = load_stage_kb()

SYSTEM_PROMPT = """You are a careful medical explainer for an AI diabetic-retinopathy \
screening tool. You are NOT a doctor and you must NOT diagnose or prescribe.

Use only the stage facts and numbers provided. Do not invent statistics, \
treatments, costs, hospital names or phone numbers.

Write for a patient with no medical background, in plain language, under 250 \
words. Cover, in this order:
1. What the condition is and what the grade means.
2. What it may mean for their vision, without scaring or reassuring falsely.
3. How urgent it is.
4. What they can do now: control glucose, blood pressure and cholesterol, keep \
regular eye exams, and see an ophthalmologist.

You must state that this is an AI screening result, not a diagnosis, and that \
macular swelling (DME) cannot be seen by this model and needs an OCT scan.
"""


def compute_urgent(state: DR_State) -> bool:
    """Deterministic urgency: severe-or-worse, or a detector urgent flag.

    Never delegated to the LLM.
    """

    return bool(state.urgent or (state.grade is not None and state.grade >= SEVERE_GRADE))


def _require_detector_fields(state: DR_State) -> None:
    missing = [
        name
        for name in ("grade", "confidence")
        if getattr(state, name) is None
    ]
    if missing:
        raise ValueError(
            "explainer requires detector output: " + ", ".join(missing)
        )


def build_prompt(state: DR_State) -> list[tuple[str, str]]:
    """Build the chat messages for the explainer.

    Kept separate from the LLM call so the prompt can be unit-tested.
    """

    _require_detector_fields(state)
    assert state.grade is not None and state.confidence is not None

    facts = STAGE_KB[state.grade]
    label = state.label or GRADE_LABELS[state.grade]
    probs = ", ".join(f"{p:.3f}" for p in state.probs) if state.probs else "not available"

    human = f"""Screening result for this patient:
- Grade: {state.grade} ({label})
- Model's raw top class: {GRADE_LABELS.get(state.raw_grade, "unknown") if state.raw_grade is not None else "unknown"}
- Confidence in the top class: {state.confidence:.3f}
- Class probabilities (No DR, Mild, Moderate, Severe, Proliferative): {probs}
- Safety flags: {", ".join(state.flags) if state.flags else "none"}
- Urgent (severe or worse): {"yes" if compute_urgent(state) else "no"}

Stage facts (the only clinical facts you may use):
- Summary: {facts.summary}
- Follow-up: {facts.follow_up}
- Usual treatment: {facts.treatment}
- Expected recovery: {facts.recovery}

Write the explanation now."""

    return [("system", SYSTEM_PROMPT), ("human", human)]


def explain(state: DR_State, llm) -> dict:
    """Run the explainer with an injected chat model (testable)."""

    messages = build_prompt(state)
    response = llm.invoke(messages)
    check_finish_reason(response, "explainer")
    text = (getattr(response, "content", response) or "").strip()
    if not text:
        raise RuntimeError("explainer LLM returned an empty response")

    return {"explanation": text, "urgent": compute_urgent(state)}


def explainer_node(state: DR_State) -> dict:
    """LangGraph node: fill ``explanation`` and ``urgent``."""

    return explain(state, get_llm())
