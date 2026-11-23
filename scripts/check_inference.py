"""Smoke-test real detector inference on one fundus image.

Usage:
    python scripts/check_inference.py path/to/fundus.png
    python scripts/check_inference.py path/to/fundus.png --model dr_svm_bundle.joblib

This exercises the real preprocessing, feature extraction, scaler/SVM and
threshold escalation. It needs tensorflow, scikit-learn, opencv and numpy.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as `python scripts/check_inference.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.config import settings
from agents.nodes.detector import detect, load_bundle
from agents.state import GRADE_LABELS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="retinal fundus image")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=f"joblib bundle (default: {settings.model_path})",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 2

    bundle_path = args.model or settings.model_path
    print(f"image:  {args.image}")
    print(f"bundle: {bundle_path}")

    started = time.perf_counter()
    try:
        bundle = load_bundle(args.model)
        result = detect(str(args.image), bundle=bundle)
    except Exception as exc:  # keep the CLI message short and actionable
        print(f"error: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started

    print(f"\nfinished in {elapsed:.1f}s")
    print(
        f"raw grade:   {result['raw_grade']} "
        f"({GRADE_LABELS[result['raw_grade']]})  "
        f"confidence={result['confidence']:.3f}"
    )
    print(
        f"final grade: {result['grade']} ({result['label']})"
        + (
            "  <- escalated by safety thresholds"
            if result["grade"] != result["raw_grade"]
            else ""
        )
    )

    print("\nprobabilities:")
    for grade in range(5):
        print(f"  {grade} {GRADE_LABELS[grade]:14s} {result['probs'][grade]:.3f}")

    p_severe = sum(result["probs"][3:])
    p_referable = sum(result["probs"][2:])
    print(
        f"\nurgent:    {result['urgent']}  "
        f"P(Severe+)={p_severe:.3f}  T_SEVERE={result['t_severe']:.3f}"
    )
    print(
        f"referable: {result['referable']}  "
        f"P(Moderate+)={p_referable:.3f}  T_REFERABLE={result['t_referable']:.3f}"
    )

    if result["flags"]:
        print("\nflags:")
        for flag in result["flags"]:
            print(f"  - {flag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
