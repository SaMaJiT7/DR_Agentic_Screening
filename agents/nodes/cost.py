"""Cost node: real, hospital-quoted treatment prices from ``cost_kb.json``.

This node contains no ML and no LLM. It loads and validates the price table
(path from ``COST_KB_PATH``) and attaches the quoted options to the state.
The caregiver and the report may only quote these figures; nothing else in the
pipeline knows a price.

The table covers intravitreal anti-VEGF injections (per eye) and PRP laser
sessions (per session), because both are used for diabetic retinopathy.

Scope rule: prices are attached only for referable grades (Moderate, Severe,
Proliferative). For No DR and Mild the state carries no options and an explicit
note that an ophthalmologist decides whether treatment is needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from agents.config import settings
from agents.state import CostOption, DR_State, TreatmentBrief

#: Grades for which an anti-VEGF injection may be part of treatment.
COST_GRADES = frozenset({2, 3, 4})

_REQUIRED_TOP_LEVEL = {"currency", "as_of", "source", "notes", "treatments"}


@dataclass(frozen=True)
class CostTable:
    """Validated contents of one cost knowledge base."""

    currency: str
    as_of: str
    source: str
    notes: tuple[str, ...]
    options: tuple[CostOption, ...]
    briefs: tuple[TreatmentBrief, ...] = ()
    """Other treatments described without a price, to avoid inventing figures."""

    def note_text(self) -> str:
        """Deterministic caveats shown next to the prices, one per line."""

        return "\n".join(f"- {note}" for note in self.notes)


def _require_text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string")
    return value.strip()


def _require_text_list(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} must be a non-empty list of strings")
    return tuple(
        _require_text(item, f"{where}[{index}]") for index, item in enumerate(value)
    )


@lru_cache(maxsize=8)
def _load_cost_kb(path: str) -> CostTable:
    kb_path = Path(path)
    if not kb_path.is_file():
        raise FileNotFoundError(
            f"cost knowledge base not found: {kb_path} "
            "(set COST_KB_PATH to a valid JSON file)"
        )

    try:
        raw = json.loads(kb_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"cost knowledge base is not valid JSON: {kb_path}") from exc

    if not isinstance(raw, dict):
        raise ValueError("cost knowledge base must be a JSON object")

    missing = _REQUIRED_TOP_LEVEL - raw.keys()
    if missing:
        raise ValueError(f"cost knowledge base is missing sections: {sorted(missing)}")

    currency = _require_text(raw["currency"], "currency")
    if currency != "INR":
        raise ValueError(
            f"cost knowledge base currency must be INR (prices use cost_inr), got {currency!r}"
        )

    raw_options = raw["treatments"]
    if not isinstance(raw_options, list) or not raw_options:
        raise ValueError("treatments must be a non-empty list")

    options: list[CostOption] = []
    for index, entry in enumerate(raw_options):
        if not isinstance(entry, dict):
            raise ValueError(f"treatments[{index}] must be an object")
        try:
            option = CostOption(**entry)
        except Exception as exc:  # pydantic ValidationError
            raise ValueError(f"treatments[{index}] is invalid: {exc}") from exc
        options.append(option)

    seen: set[tuple[str, str, str]] = set()
    for option in options:
        key = (option.hospital.lower(), option.name.lower(), option.unit.lower())
        if key in seen:
            raise ValueError(f"duplicate cost entry: {key}")
        seen.add(key)

    raw_briefs = raw.get("other_treatments", [])
    if not isinstance(raw_briefs, list):
        raise ValueError("other_treatments must be a list")
    briefs: list[TreatmentBrief] = []
    for index, entry in enumerate(raw_briefs):
        if not isinstance(entry, dict):
            raise ValueError(f"other_treatments[{index}] must be an object")
        try:
            brief = TreatmentBrief(**entry)
        except Exception as exc:  # pydantic ValidationError
            raise ValueError(f"other_treatments[{index}] is invalid: {exc}") from exc
        briefs.append(brief)

    return CostTable(
        currency=currency,
        as_of=_require_text(raw["as_of"], "as_of"),
        source=_require_text(raw["source"], "source"),
        notes=_require_text_list(raw["notes"], "notes"),
        options=tuple(options),
        briefs=tuple(briefs),
    )


def load_cost_kb(path: str | Path | None = None) -> CostTable:
    """Load and validate the cost KB (``COST_KB_PATH`` or an explicit path)."""

    kb_path = Path(path) if path is not None else settings.cost_kb_path
    return _load_cost_kb(str(kb_path.expanduser().resolve()))


#: Default table loaded at import time; kept for convenience and back-compat.
COST_KB: CostTable = load_cost_kb()

NO_TREATMENT_NOTE = (
    "No treatment cost is shown for this result. An ophthalmologist decides "
    "whether an injection or laser is needed; ask them about the cost if it is "
    "recommended."
)


def format_cost_option(option: CostOption, currency: str = "INR") -> str:
    """One readable line, used by both the prompt and the report."""

    symbol = "₹" if currency == "INR" else f"{currency} "
    ingredient = f" ({option.active_ingredient})" if option.active_ingredient else ""
    return (
        f"{option.hospital} — {option.name}{ingredient}, {option.unit}: "
        f"{symbol}{option.cost_inr:,}"
    )


def format_cost_options(table: CostTable) -> list[str]:
    """All priced rows of the table as readable lines."""

    return [format_cost_option(option, table.currency) for option in table.options]


def format_treatment_brief(brief: TreatmentBrief) -> str:
    """One readable line for an unpriced treatment."""

    return f"{brief.name} (for {brief.purpose}): {brief.summary}"


def format_treatment_briefs(table: CostTable) -> list[str]:
    """All unpriced treatment descriptions as readable lines."""

    return [format_treatment_brief(brief) for brief in table.briefs]


def cost_plan(state: DR_State) -> dict:
    """Return the cost slice for the given state (no LLM, no network)."""

    if state.grade is None:
        raise ValueError("cost node requires detector output (grade)")

    if state.grade not in COST_GRADES:
        return {
            "cost_options": [],
            "cost_note": NO_TREATMENT_NOTE,
            "treatment_briefs": [],
        }

    return {
        "cost_options": list(COST_KB.options),
        "cost_note": COST_KB.note_text(),
        "treatment_briefs": list(COST_KB.briefs),
    }


def cost_node(state: DR_State) -> dict:
    """LangGraph node: fill ``cost_options`` and ``cost_note``."""

    return cost_plan(state)
