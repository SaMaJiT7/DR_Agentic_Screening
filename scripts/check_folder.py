"""Run the detector over a folder of fundus images and summarise the output.

Usage:
    python scripts/check_folder.py test_images
    python scripts/check_folder.py test_images --limit 100
    python scripts/check_folder.py test_images --csv predictions.csv

Prints the raw grade distribution (the SVM's own prediction), the final
distribution after ``t_severe`` / ``t_referable`` escalation, and how often
escalation changed the grade. Needs tensorflow, scikit-learn, opencv, numpy.
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.config import settings
from agents.nodes.detector import detect, load_bundle
from agents.state import GRADE_LABELS

EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_images(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for extension in EXTENSIONS:
        paths.extend(folder.glob(f"*{extension}"))
        paths.extend(folder.glob(f"*{extension.upper()}"))
    return sorted(set(paths))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="folder of retinal fundus images")
    parser.add_argument(
        "--limit", type=int, default=0, help="only score the first N images (0 = all)"
    )
    parser.add_argument(
        "--model", type=Path, default=None, help="joblib bundle override"
    )
    parser.add_argument(
        "--csv", type=Path, default=None, help="write per-image predictions here"
    )
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"error: not a folder: {args.folder}", file=sys.stderr)
        return 2

    paths = find_images(args.folder)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"error: no images found in {args.folder}", file=sys.stderr)
        return 2

    bundle = load_bundle(args.model)
    raw_hist: Counter[int] = Counter()
    final_hist: Counter[int] = Counter()
    rows: list[dict] = []
    failures = 0
    escalated = 0
    used_thresholds: tuple[float, float] | None = None

    print(f"folder: {args.folder}")
    print(f"images: {len(paths)}")

    started = time.perf_counter()
    for path in paths:
        try:
            result = detect(str(path), bundle=bundle)
        except Exception as exc:
            failures += 1
            print(f"  skipped {path.name}: {exc}", file=sys.stderr)
            continue

        raw_hist[result["raw_grade"]] += 1
        final_hist[result["grade"]] += 1
        if result["grade"] != result["raw_grade"]:
            escalated += 1
        if used_thresholds is None:
            used_thresholds = (result["t_severe"], result["t_referable"])

        rows.append(
            {
                "image": path.name,
                "raw_grade": result["raw_grade"],
                "grade": result["grade"],
                "label": result["label"],
                "confidence": round(result["confidence"], 4),
                "p_no_dr": round(result["probs"][0], 4),
                "p_mild": round(result["probs"][1], 4),
                "p_moderate": round(result["probs"][2], 4),
                "p_severe": round(result["probs"][3], 4),
                "p_proliferative": round(result["probs"][4], 4),
                "urgent": result["urgent"],
                "referable": result["referable"],
            }
        )
    elapsed = time.perf_counter() - started

    scored = sum(raw_hist.values())
    if not scored:
        print("error: every image failed", file=sys.stderr)
        return 1

    print(f"\nscored {scored} images in {elapsed:.1f}s ({elapsed / scored:.2f}s each)")
    if failures:
        print(f"failed: {failures}")

    print("\nraw grades (SVM argmax):")
    for grade in range(5):
        count = raw_hist[grade]
        print(
            f"  {grade} {GRADE_LABELS[grade]:14s} {count:5d}  {count / scored:6.1%}"
        )

    print("\nfinal grades (after t_severe / t_referable):")
    for grade in range(5):
        count = final_hist[grade]
        if count:
            print(
                f"  {grade} {GRADE_LABELS[grade]:14s} {count:5d}  {count / scored:6.1%}"
            )

    print(
        f"\nescalated by thresholds: {escalated}/{scored} ({escalated / scored:.1%})"
    )
    if used_thresholds is not None:
        print(
            f"thresholds used: T_SEVERE={used_thresholds[0]:.3f}  "
            f"T_REFERABLE={used_thresholds[1]:.3f}"
        )
    print(
        f"settings fallback: T_SEVERE={settings.t_severe}  "
        f"T_REFERABLE={settings.t_referable} (bundle values win when present)"
    )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
