"""Validate a dr_svm_bundle.joblib before running inference.

Usage:
    python scripts/inspect_bundle.py dr_svm_bundle.joblib

Checks the keys the detector expects, the classifier type/classes and the
hybrid feature width. Also prints the library versions used to load the file,
which helps diagnose scikit-learn pickle incompatibilities.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.config import settings
from agents.state import GRADE_LABELS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bundle",
        nargs="?",
        type=Path,
        default=settings.model_path,
        help=f"joblib bundle (default: {settings.model_path})",
    )
    args = parser.parse_args()

    if not args.bundle.is_file():
        print(f"error: bundle not found: {args.bundle}", file=sys.stderr)
        return 2

    import joblib
    import numpy
    import sklearn
    import tensorflow

    print(f"bundle:     {args.bundle.resolve()}")
    print(f"python:     {sys.version.split()[0]}")
    print(f"numpy:      {numpy.__version__}")
    print(f"scikit:     {sklearn.__version__}")
    print(f"tensorflow: {tensorflow.__version__}")

    try:
        bundle = joblib.load(args.bundle)
    except Exception as exc:
        print(f"\nerror: could not load bundle: {exc}", file=sys.stderr)
        print(
            "hint: the bundle was probably saved with a different "
            "scikit-learn version; re-save it with the same or a newer version.",
            file=sys.stderr,
        )
        return 1

    if not isinstance(bundle, dict):
        print("\nerror: bundle must be a dict", file=sys.stderr)
        return 1

    print(f"\nkeys: {sorted(bundle)}")

    model = None
    for key in ("model", "svm", "classifier"):
        if key in bundle:
            model = bundle[key]
            print(f"classifier key: {key}")
            break
    if model is None:
        print("\nerror: bundle has no model/svm/classifier", file=sys.stderr)
        return 1

    print(f"classifier type: {type(model).__name__}")
    if not hasattr(model, "predict_proba"):
        print("\nerror: classifier has no predict_proba (fit with probability=True)", file=sys.stderr)
        return 1

    classes = getattr(model, "classes_", None)
    if classes is not None:
        class_order = [int(c) for c in classes]
        print(f"classes_: {class_order}")
        if sorted(class_order) != list(range(5)):
            print("\nerror: classifier classes are not 0-4", file=sys.stderr)
            return 1

    n_features = getattr(model, "n_features_in_", None)
    print(f"n_features_in_: {n_features}")
    if n_features is not None and int(n_features) != 4096:
        print(
            "\nerror: model expects a different feature width; the detector "
            "produces 4096 (ResNet50 + InceptionV3).",
            file=sys.stderr,
        )
        return 1

    t_severe = bundle.get("t_severe")
    t_referable = bundle.get("t_referable")
    print(f"t_severe: {t_severe}   (settings fallback: {settings.t_severe})")
    print(f"t_referable: {t_referable}   (settings fallback: {settings.t_referable})")
    if t_severe is None or t_referable is None:
        print("warning: thresholds missing; the detector will use settings/env values.")

    if "classes" in bundle:
        print(f"bundle classes: {bundle['classes']}")
    print("\nlabels: " + ", ".join(f"{g}={GRADE_LABELS[g]}" for g in range(5)))
    print("\nbundle looks compatible with the detector.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
