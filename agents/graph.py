"""LangGraph assembly for the diabetic-retinopathy screening pipeline.

Wiring::

    START ──► detector ──► router ──┬──► screen_only ──────────────────────────┐
                                    └──► explainer ──► cost ──► caregiver      │
                                                       ──► hospital_search ───┴──► report ──► END

* ``detector``        fundus image -> grade, probabilities, safety flags
* ``router``          confident No DR with no flag -> stop early, else explain
* ``screen_only``     deterministic No DR advice (no LLM, no hospital search)
* ``explainer``       LLM stage explanation grounded in ``stage_kb.json``
* ``cost``            real injection quotes from ``cost_kb.json`` (no LLM)
* ``caregiver``       LLM care plan grounded in ``care_kb.json`` + cost table
* ``hospital_search`` OpenStreetMap lookup for eye hospitals in the city
* ``report``          deterministic combined report

Every dependency can be injected into ``build_graph`` so the wiring can be
tested without TensorFlow, a Groq key or network access.
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from agents.nodes.caregiver import caregiver_node
from agents.nodes.cost import cost_node
from agents.nodes.detector import detector_node
from agents.nodes.explainer import explainer_node
from agents.nodes.hospital_search import hospital_search_node
from agents.nodes.report import report_node
from agents.nodes.router import route_after_detector, router_node
from agents.nodes.screen_only import screen_only_node
from agents.state import DR_State, make_initial_state


def build_graph(
    *,
    detector=detector_node,
    router=router_node,
    screen_only=screen_only_node,
    explainer=explainer_node,
    cost=cost_node,
    caregiver=caregiver_node,
    hospital_search=hospital_search_node,
    reporter=report_node,
    route_fn=route_after_detector,
):
    """Build and compile the screening graph.

    All arguments are node callables (``DR_State -> dict``) and can be replaced
    by fakes in tests.
    """

    builder = StateGraph(DR_State)

    builder.add_node("detector", detector)
    builder.add_node("router", router)
    builder.add_node("screen_only", screen_only)
    builder.add_node("explainer", explainer)
    builder.add_node("cost", cost)
    builder.add_node("caregiver", caregiver)
    builder.add_node("hospital_search", hospital_search)
    builder.add_node("report", reporter)

    builder.add_edge(START, "detector")
    builder.add_edge("detector", "router")
    builder.add_conditional_edges(
        "router",
        route_fn,
        {"screen_only": "screen_only", "explainer": "explainer"},
    )

    # Early stop: confident No DR, no flags -> advice, no LLM, no hospital search.
    builder.add_edge("screen_only", "report")

    # Full path: explain -> cost table -> care plan -> find hospitals -> report.
    builder.add_edge("explainer", "cost")
    builder.add_edge("cost", "caregiver")
    builder.add_edge("caregiver", "hospital_search")
    builder.add_edge("hospital_search", "report")

    builder.add_edge("report", END)

    return builder.compile()


@lru_cache(maxsize=1)
def get_graph():
    """Return the default compiled graph (built once per process)."""

    return build_graph()


def _as_state(result) -> DR_State:
    """Normalise a LangGraph result to ``DR_State``."""

    if isinstance(result, DR_State):
        return result
    return DR_State(**result)


def run_screening(
    image_path: str,
    city: str,
    *,
    graph=None,
    recursion_limit: int = 25,
) -> DR_State:
    """Run the whole pipeline for one image and city."""

    compiled = graph if graph is not None else get_graph()
    initial = make_initial_state(image_path, city)
    result = compiled.invoke(initial, config={"recursion_limit": recursion_limit})
    return _as_state(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the diabetic-retinopathy screening pipeline on one image."
    )
    parser.add_argument("image", type=Path, help="retinal fundus image")
    parser.add_argument("city", help="city used for the eye-hospital search")
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="write the report markdown to this file",
    )
    args = parser.parse_args(argv)

    if not args.image.is_file():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 2

    try:
        state = run_screening(str(args.image), args.city)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(state.report or "(no report produced)")

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(state.report or "", encoding="utf-8")
        print(f"\nwrote report to {args.report_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
