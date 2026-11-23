"""Runtime configuration for the DR pipeline.

Everything here can be overridden with environment variables. Keep this
module import-light: it is read by nodes that may run without an API key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path = _PROJECT_ROOT / ".env") -> None:
    """Load ``KEY=value`` lines from a local ``.env`` if the keys are unset.

    Deliberately tiny: no new dependency, no overriding of real environment
    variables, no support for multi-line values.
    """

    if not path.is_file():
        return

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_STAGE_KB_PATH = _PACKAGE_DIR / "data" / "stage_kb.json"


@dataclass(frozen=True)
class Settings:
    """Resolved settings for one process."""

    # --- knowledge base ---------------------------------------------------
    stage_kb_path: Path = Path(
        os.getenv("STAGE_KB_PATH", str(DEFAULT_STAGE_KB_PATH))
    )
    """JSON file with the grade 0-4 stage facts used by the explainer."""

    care_kb_path: Path = Path(
        os.getenv(
            "CARE_KB_PATH", str(_PACKAGE_DIR / "data" / "care_kb.json")
        )
    )
    """JSON file with next steps, diet, high-sugar/BP and report guidance."""

    cost_kb_path: Path = Path(
        os.getenv("COST_KB_PATH", str(_PACKAGE_DIR / "data" / "cost_kb.json"))
    )
    """JSON file with real, hospital-quoted injection prices (INR)."""

    # --- detector ---------------------------------------------------------
    model_path: Path = Path(os.getenv("DR_MODEL_PATH", "dr_svm_bundle.joblib"))
    """Joblib bundle written by the training notebook (scaler + SVM + thresholds)."""

    t_severe: float = _float_env("T_SEVERE", 0.30)
    """Fallback P(Severe)+P(Proliferative) threshold when the bundle has none.

    PLACEHOLDER: tune on the official validation split and record the value.
    """
    t_referable: float = _float_env("T_REFERABLE", 0.50)
    """Fallback P(Moderate)+P(Severe)+P(Proliferative) threshold.

    PLACEHOLDER: tune on the official validation split and record the value.
    """

    # --- LLM (explainer + care navigator) --------------------------------
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    llm_temperature: float = _float_env("LLM_TEMPERATURE", 0.2)
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2000"))

    # --- hospital search (OpenStreetMap Nominatim) ------------------------
    nominatim_url: str = os.getenv(
        "NOMINATIM_URL", "https://nominatim.openstreetmap.org/search"
    )
    nominatim_user_agent: str = os.getenv(
        "NOMINATIM_USER_AGENT", "dr-screening-agent/0.1 (portfolio project)"
    )
    hospital_limit: int = int(os.getenv("HOSPITAL_LIMIT", "8"))
    http_timeout: float = _float_env("HTTP_TIMEOUT", 15.0)


settings = Settings()
