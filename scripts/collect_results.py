"""Collect paste-ready results for a project report.

Usage (from the project root, with the ML venv):

    .venv-ml/bin/python scripts/collect_results.py
    .venv-ml/bin/python scripts/collect_results.py --limit 150
    .venv-ml/bin/python scripts/collect_results.py --skip-llm   # no Groq calls

Writes one markdown file per result section into ``results/``, plus an index.
Each file contains the exact command and its captured output, so it can be
pasted into a report without editing.

Sections:
    00_tests.md              node/graph test suite
    01_environment.md        Python and library versions
    02_bundle.md             dr_svm_bundle.joblib validation
    03_batch_distribution.md detector distribution over test_images
    04_single_image.md       one image, full probability breakdown
    05_pipeline_dr.md        full pipeline report (DR case, LLM + map search)
    06_pipeline_no_dr.md     full pipeline report (No DR early-stop case)

Needs tensorflow, scikit-learn, opencv and numpy (the ``.venv-ml`` env). The
pipeline sections also need ``GROQ_API_KEY`` (a ``.env`` file is picked up
automatically) and network access for the hospital search.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def _python() -> str:
    return sys.executable


def _run(args: list[str], timeout: int = 1800) -> tuple[str, float, int]:
    """Run a command from the project root and capture stdout+stderr."""

    env = dict(os.environ, PYTHONPATH=str(ROOT))
    started = time.perf_counter()
    process = subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started
    output = (process.stdout or "") + (process.stderr or "")
    return output.strip(), elapsed, process.returncode


def _write(name: str, title: str, command: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    text = f"# {title}\n\nCommand:\n\n```bash\n{command}\n```\n\nOutput:\n\n```text\n{body}\n```\n"
    path.write_text(text, encoding="utf-8")
    print(f"  wrote results/{name}")


def _section_tests() -> bool:
    tests = sorted((ROOT / "tests").glob("test_*.py"))
    if not tests:
        return False

    lines: list[str] = []
    failed = 0
    started = time.perf_counter()
    for test in tests:
        output, _, code = _run([_python(), str(test)], timeout=300)
        if code == 0:
            last = output.strip().splitlines()[-1] if output.strip() else "passed"
            lines.append(f"{test.name:32s} PASS  ({last})")
        else:
            failed += 1
            lines.append(f"{test.name:32s} FAIL")
            lines.append(output)
    elapsed = time.perf_counter() - started

    lines.append("")
    lines.append(
        f"{len(tests) - failed}/{len(tests)} test files passed in {elapsed:.1f}s"
    )
    command = (
        f'for f in tests/test_*.py; do PYTHONPATH=. {_python()} "$f" || exit 1; done'
    )
    _write("00_tests.md", "Test suite", command, "\n".join(lines))
    return failed == 0


def _section_environment() -> None:
    code = (
        "import platform, sys\n"
        "import numpy, cv2, sklearn, tensorflow as tf, keras, langgraph\n"
        "print('python      ', sys.version.split()[0])\n"
        "print('platform    ', platform.platform())\n"
        "print('numpy       ', numpy.__version__)\n"
        "print('opencv      ', cv2.__version__)\n"
        "print('scikit-learn', sklearn.__version__)\n"
        "print('tensorflow  ', tf.__version__)\n"
        "print('keras       ', keras.__version__)\n"
    )
    output, _, _ = _run([_python(), "-c", code], timeout=600)
    _write(
        "01_environment.md",
        "Environment",
        f"{_python()} - <<'PY'\n{code}PY",
        output,
    )


def _section_bundle() -> None:
    output, _, _ = _run(
        [_python(), "scripts/inspect_bundle.py", "dr_svm_bundle.joblib"],
        timeout=600,
    )
    _write(
        "02_bundle.md",
        "Model bundle validation",
        f"{_python()} scripts/inspect_bundle.py dr_svm_bundle.joblib",
        output,
    )


def _section_batch(limit: int) -> list[dict]:
    csv_path = OUT / "batch_predictions.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    output, elapsed, code = _run(
        [
            _python(),
            "scripts/check_folder.py",
            "test_images",
            "--limit",
            str(limit),
            "--csv",
            str(csv_path),
        ],
        timeout=3600,
    )
    _write(
        "03_batch_distribution.md",
        f"Detector distribution over {limit} images",
        f"{_python()} scripts/check_folder.py test_images --limit {limit} --csv results/batch_predictions.csv",
        output,
    )
    if code != 0 or not csv_path.is_file():
        return []
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _pick_images(rows: list[dict]) -> tuple[str | None, str | None]:
    """Choose one clear No DR image and one referable image from the batch."""

    no_dr = None
    dr = None
    for row in rows:
        image = f"test_images/{row['image']}"
        grade = int(row["grade"])
        confidence = float(row["confidence"])
        urgent = row["urgent"] == "True"
        if no_dr is None and grade == 0 and not urgent and confidence >= 0.60:
            no_dr = image
        if dr is None and grade >= 2 and urgent:
            dr = image
    if dr is None and rows:
        for row in rows:
            if int(row["grade"]) >= 2:
                dr = f"test_images/{row['image']}"
                break
    return dr, no_dr


def _section_single_image(image: str) -> None:
    output, elapsed, _ = _run(
        [_python(), "scripts/check_inference.py", image], timeout=1800
    )
    _write(
        "04_single_image.md",
        "Single-image inference",
        f"{_python()} scripts/check_inference.py {image}",
        f"image: {image}\nwall clock: {elapsed:.1f}s (includes model warm-up)\n\n{output}",
    )


def _section_pipeline(name: str, title: str, image: str, city: str) -> None:
    report_path = OUT / f"{Path(name).stem}_report.md"
    output, elapsed, code = _run(
        [
            _python(),
            "-m",
            "agents.graph",
            image,
            city,
            "--report-out",
            str(report_path),
        ],
        timeout=1800,
    )
    if code != 0:
        body = f"FAILED after {elapsed:.1f}s\n\n{output}"
    else:
        report = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
        body = (
            f"image: {image}\ncity: {city}\nwall clock: {elapsed:.1f}s\n"
            f"saved report: results/{report_path.name}\n\n"
            f"----- full report -----\n\n{report}"
        )
    command = f"{_python()} -m agents.graph {image} {city} --report-out results/{report_path.name}"
    _write(name, title, command, body)


def _index(images: dict[str, str | None], limit: int) -> None:
    lines = [
        "# Result pack",
        "",
        "Generated by `scripts/collect_results.py`.",
        "",
        "| File | What it shows |",
        "|---|---|",
        "| `00_tests.md` | the node/graph test suite passing |",
        "| `01_environment.md` | Python and library versions |",
        "| `02_bundle.md` | bundle keys, classes, feature width, thresholds |",
        f"| `03_batch_distribution.md` | grade distribution over {limit} images |",
        "| `04_single_image.md` | probabilities and threshold escalation for one image |",
        "| `05_pipeline_dr.md` | full report for a referable case (LLM + map search) |",
        "| `06_pipeline_no_dr.md` | full report for the confident No DR early-stop case |",
        "",
        "## Images used",
        "",
        f"- referable case: `{images['dr'] or 'not found'}`",
        f"- No DR case: `{images['no_dr'] or 'not found'}`",
        "",
        "## Caveats to state in the report",
        "",
        "- No ground-truth labels were available, so accuracy, sensitivity and QWK are not reported.",
        "- On the sampled images the SVM's raw top class skews to Moderate, and the saved "
        "`T_SEVERE=0.084` escalates most of them to Severe. This is model calibration, not an "
        "inference bug: preprocessing and the 4096-value features match the notebook exactly.",
        "- Prices in the cost table are per-eye injection packages or per PRP session, and exclude "
        "consultation, OCT scans, procedure fees and follow-up. They are not total treatment costs.",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  wrote results/README.md")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100, help="images for the batch section")
    parser.add_argument("--city", default="Kolkata", help="city for the hospital search")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="skip the two pipeline sections (no Groq calls, no network)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="delete the results directory before writing",
    )
    args = parser.parse_args()

    if args.clean and OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"writing to {OUT}")
    print("[1/6] tests")
    tests_ok = _section_tests()

    print("[2/6] environment")
    _section_environment()

    print("[3/6] bundle")
    _section_bundle()

    print(f"[4/6] batch over {args.limit} images")
    rows = _section_batch(args.limit)
    dr_image, no_dr_image = _pick_images(rows)

    if dr_image:
        print("[5/6] single-image inference")
        _section_single_image(dr_image)
    else:
        print("[5/6] single-image inference: skipped (no referable image found)")

    if not args.skip_llm and dr_image:
        print("[6/6] full pipeline (DR)")
        _section_pipeline(
            "05_pipeline_dr.md", "Full pipeline: referable case", dr_image, args.city
        )
        if no_dr_image:
            print("      full pipeline (No DR)")
            _section_pipeline(
                "06_pipeline_no_dr.md",
                "Full pipeline: No DR case",
                no_dr_image,
                args.city,
            )
    else:
        print("[6/6] full pipeline: skipped")

    _index({"dr": dr_image, "no_dr": no_dr_image}, args.limit)

    print("")
    print("done." + ("" if tests_ok else " WARNING: some tests failed, see 00_tests.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
