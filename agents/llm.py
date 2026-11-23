"""Shared Groq chat-model factory for the LLM-backed nodes."""

from __future__ import annotations

from agents.config import settings


def get_llm(*, temperature: float | None = None, max_tokens: int | None = None):
    """Create a Groq chat model, importing ``langchain_groq`` lazily.

    Raises:
        RuntimeError: when ``GROQ_API_KEY`` or ``langchain-groq`` is missing.
    """

    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set; export it or pass an llm to the node."
        )

    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "langchain-groq is not installed; pip install langchain-groq"
        ) from exc

    return ChatGroq(
        api_key=settings.groq_api_key, #type:ignore
        model=settings.llm_model,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=settings.llm_max_tokens if max_tokens is None else max_tokens,
    )


def check_finish_reason(response, what: str) -> None:
    """Raise when the provider stopped because it hit the token limit.

    A truncated care plan or explanation is worse than a failed run: it can end
    mid-sentence, sometimes mid-price. Surfacing it loudly is the safer choice.
    """

    metadata = getattr(response, "response_metadata", None) or {}
    reason = metadata.get("finish_reason") or metadata.get("stop_reason")
    if reason in ("length", "max_tokens"):
        raise RuntimeError(
            f"{what} LLM output was truncated at the token limit "
            f"({reason}); raise LLM_MAX_TOKENS (currently "
            f"{settings.llm_max_tokens}) or shorten the prompt."
        )
