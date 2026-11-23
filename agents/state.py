"""Shared state for the diabetic-retinopathy multi-agent graph.

Each node in the LangGraph pipeline reads and updates pieces of one
``DR_State``. Keep this module free of ML/LLM imports so it can be tested
without TensorFlow, scikit-learn or an API key.

Flow of a run::

    inputs ──► detector ──► router ──┬──► screen_only ──────────────────────► report
                                     └──► explainer ──► cost ──► caregiver
                                                         ──► navigator ────► report
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Domain constants (APTOS 2019 labels)
# ---------------------------------------------------------------------------

GRADE_LABELS: dict[int, str] = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative",
}

GRADE_NAMES: dict[int, str] = {
    0: "No diabetic retinopathy",
    1: "Mild non-proliferative diabetic retinopathy",
    2: "Moderate non-proliferative diabetic retinopathy",
    3: "Severe non-proliferative diabetic retinopathy",
    4: "Proliferative diabetic retinopathy",
}

#: Grades at or above this value are "severe or worse".
SEVERE_GRADE = 3
#: Grades at or above this value are "referable".
REFERABLE_GRADE = 2
#: Confidence needed before a "No DR" result may stop the pipeline early.
ROUTER_STOP_CONFIDENCE = 0.60

Route = Literal["screen_only", "explain"]


class Hospital(BaseModel):
    """One hospital record returned by the OpenStreetMap lookup."""

    name: str
    address: str | None = None
    lat: float | None = None
    lon: float | None = None


class CostOption(BaseModel):
    """One real price from ``cost_kb.json``.

    Covers both intravitreal injections and PRP laser sessions. Only the
    treatment price is stored. Consultation, scans, procedure fees and
    follow-up visits are deliberately absent, so the report can never present
    a total that nobody quoted.
    """

    model_config = ConfigDict(extra="forbid")

    hospital: str
    kind: Literal["injection", "laser"]
    name: str
    active_ingredient: str | None = None
    unit: str
    """What the price buys, e.g. "one package, per eye" or "one session"."""
    cost_inr: int = Field(gt=0)
    notes: str | None = None


class TreatmentBrief(BaseModel):
    """One possible treatment, described briefly and without a price.

    Used for options the project has no real quote for. Keeping them separate
    from ``CostOption`` means the report never puts a price on a treatment
    nobody quoted.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    purpose: str
    summary: str


class DR_State(BaseModel):
    """State shared by every node of the DR graph.

    Inputs are required; everything else is filled progressively by the
    detector, router, explainer, navigator and report nodes.
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # --- inputs -----------------------------------------------------------
    image_path: str
    city: str

    # --- detector output --------------------------------------------------
    grade: int | None = None
    """Final grade after safety escalation; never lower than ``raw_grade``."""
    raw_grade: int | None = None
    """Top class predicted by the model, kept for auditing."""
    label: str | None = None
    confidence: float | None = None
    probs: list[float] = Field(default_factory=list)
    """Five probabilities; index == grade."""
    urgent: bool = False
    """P(Severe) + P(Proliferative) >= T_SEVERE."""
    referable: bool = False
    """P(Moderate) + P(Severe) + P(Proliferative) >= T_REFERABLE."""
    flags: list[str] = Field(default_factory=list)
    t_severe: float | None = None
    """Threshold used for P(Severe or worse) on this run."""
    t_referable: float | None = None
    """Threshold used for P(Moderate or worse) on this run."""

    # --- router output ----------------------------------------------------
    route: Route | None = None
    route_reason: str | None = None

    # --- explainer output -------------------------------------------------
    explanation: str | None = None

    # --- care giver / navigator output ------------------------------------
    hospitals: list[Hospital] = Field(default_factory=list)
    care_plan: str | None = None
    next_steps: list[str] = Field(default_factory=list)
    diet_advice: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    high_blood_sugar_steps: list[str] = Field(default_factory=list)
    high_blood_pressure_steps: list[str] = Field(default_factory=list)
    reports_to_bring: list[str] = Field(default_factory=list)
    consult_note: str | None = None

    # --- cost output ------------------------------------------------------
    cost_options: list[CostOption] = Field(default_factory=list)
    """Real quoted prices from ``cost_kb.json``; empty when no treatment applies."""
    cost_note: str | None = None
    """Deterministic caveats about what the prices do and do not include."""
    treatment_briefs: list[TreatmentBrief] = Field(default_factory=list)
    """Other treatments with no quote attached (described, never priced)."""

    # --- report output ----------------------------------------------------
    report: str | None = None

    # --- diagnostics ------------------------------------------------------
    errors: list[str] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @field_validator("grade", "raw_grade")
    @classmethod
    def _grade_in_range(cls, value: int | None) -> int | None:
        if value is not None and not 0 <= value <= 4:
            raise ValueError("grade must be between 0 and 4")
        return value

    @field_validator("confidence")
    @classmethod
    def _confidence_in_range(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return value

    @field_validator("probs")
    @classmethod
    def _probs_shape(cls, value: list[float]) -> list[float]:
        if value and len(value) != 5:
            raise ValueError("probs must contain exactly five values")
        return value

    # ------------------------------------------------------------------
    # Convenience helpers (safe to call from any node or template)
    # ------------------------------------------------------------------
    @property
    def grade_label(self) -> str:
        """Short APTOS label, or ``"Unknown"`` before detection."""
        if self.grade is None:
            return "Unknown"
        return GRADE_LABELS[self.grade]

    @property
    def severity_band(self) -> str:
        """Coarse band used by the report: none / mild / moderate / severe."""
        if self.grade is None:
            return "unknown"
        if self.grade <= 0:
            return "none"
        if self.grade == 1:
            return "mild"
        if self.grade == 2:
            return "moderate"
        return "severe"

    def add_error(self, message: str) -> None:
        """Record a non-fatal problem while keeping the run alive."""
        self.errors.append(message)


def make_initial_state(image_path: str, city: str) -> DR_State:
    """Build a minimal, valid starting state for a run."""

    if not image_path or not city:
        raise ValueError("image_path and city are required")
    return DR_State(image_path=image_path, city=city)


def grade_label(grade: int) -> str:
    """Return the short APTOS label for ``grade``."""

    try:
        return GRADE_LABELS[grade]
    except KeyError as exc:
        raise ValueError(f"grade must be 0-4, got {grade!r}") from exc


def severity_band(grade: int) -> str:
    """Coarse severity band used by the report layout."""

    if grade <= 0:
        return "none"
    if grade == 1:
        return "mild"
    if grade == 2:
        return "moderate"
    return "severe"
