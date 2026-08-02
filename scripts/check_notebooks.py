#!/usr/bin/env python3
"""Validate repository notebooks and optionally execute them from clean kernels."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_NOTEBOOKS = [
    *(f"{number:02d}_{name}.ipynb" for number, name in [
        (1, "perceptron"),
        (2, "mlp_backprop"),
        (3, "cnn"),
        (4, "rnn"),
        (5, "lstm"),
        (6, "seq2seq"),
        (7, "attention"),
        (8, "transformer"),
        (9, "gpt"),
        (10, "bert"),
        (11, "superposition"),
        (12, "sparse_autoencoder"),
    ]),
    "13_linear_algebra_mastery.ipynb",
    "14_calculus_optimization_mastery.ipynb",
    "15_probability_statistics_mastery.ipynb",
    "16_information_theory_mastery.ipynb",
    "17_autodiff_from_scratch.ipynb",
]
FAST_NOTEBOOKS = [
    "01_perceptron.ipynb",
    "02_mlp_backprop.ipynb",
    "13_linear_algebra_mastery.ipynb",
    "14_calculus_optimization_mastery.ipynb",
    "15_probability_statistics_mastery.ipynb",
    "16_information_theory_mastery.ipynb",
    "17_autodiff_from_scratch.ipynb",
]
PEDAGOGICAL_NOTEBOOKS = CANONICAL_NOTEBOOKS[:12]
IMPORT_TO_REQUIREMENT = {
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "torch": "torch",
    "torchvision": "torchvision",
}


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def validate_structure() -> list[str]:
    failures: list[str] = []
    expected = set(CANONICAL_NOTEBOOKS)
    actual = {path.name for path in ROOT.glob("*.ipynb")}

    for name in sorted(expected - actual):
        failures.append(f"missing curriculum notebook: {name}")
    for name in sorted(actual - expected):
        failures.append(f"unexpected top-level notebook: {name}")

    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    if re.search(r"\[\[[^\]]+\]\]", readme_text):
        failures.append("README.md contains Obsidian-style links that GitHub cannot resolve")
    for name in CANONICAL_NOTEBOOKS:
        if name not in readme_text:
            failures.append(f"README.md does not inventory curriculum notebook: {name}")
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme_text):
        parsed = urlparse(target)
        if parsed.scheme or target.startswith("#"):
            continue
        local_target = unquote(parsed.path)
        if local_target and not (ROOT / local_target).exists():
            failures.append(f"README.md contains missing relative link target: {target}")

    imported_modules: set[str] = set()
    notebook_paths = [ROOT / name for name in CANONICAL_NOTEBOOKS]
    notebook_paths.extend(sorted((ROOT / "scratch").glob("*.ipynb")))

    for path in notebook_paths:
        if not path.exists():
            continue
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"{path.relative_to(ROOT)}: invalid notebook JSON: {exc}")
            continue

        if notebook.get("nbformat") != 4:
            failures.append(f"{path.relative_to(ROOT)}: expected nbformat 4")
        if not isinstance(notebook.get("cells"), list):
            failures.append(f"{path.relative_to(ROOT)}: cells must be a list")
            continue

        cell_ids: set[str] = set()
        notebook_source = "\n".join(source_text(cell) for cell in notebook["cells"])
        if path.name in PEDAGOGICAL_NOTEBOOKS:
            lesson_number = PEDAGOGICAL_NOTEBOOKS.index(path.name) + 1
            expected_prompt_id = f"pedagogy-{lesson_number:02d}-prompts"
            expected_check_id = f"pedagogy-{lesson_number:02d}-checks"
            cells_by_id = {cell.get("id"): cell for cell in notebook["cells"]}
            if "## Pedagogical checks" not in notebook_source:
                failures.append(f"{path.relative_to(ROOT)}: missing pedagogical-check section")
            if expected_prompt_id not in cells_by_id:
                failures.append(f"{path.relative_to(ROOT)}: missing pedagogical prompt cell")
            check_cell = cells_by_id.get(expected_check_id)
            if check_cell is None:
                failures.append(f"{path.relative_to(ROOT)}: missing pedagogical assertion cell")
            elif not re.search(r"^\s*assert\s+", source_text(check_cell), re.MULTILINE):
                failures.append(f"{path.relative_to(ROOT)}: pedagogical checks contain no assertions")

        for index, cell in enumerate(notebook["cells"], start=1):
            cell_id = cell.get("id")
            if not cell_id:
                failures.append(f"{path.relative_to(ROOT)} cell {index}: missing cell id")
            elif cell_id in cell_ids:
                failures.append(f"{path.relative_to(ROOT)} cell {index}: duplicate cell id {cell_id}")
            else:
                cell_ids.add(cell_id)

            if cell.get("cell_type") != "code":
                continue
            code = source_text(cell)
            for module in re.findall(r"^(?:from|import)\s+([A-Za-z_][\w.]*)", code, re.MULTILINE):
                imported_modules.add(module.split(".", 1)[0])
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    name = output.get("ename", "Error")
                    message = output.get("evalue", "")
                    failures.append(
                        f"{path.relative_to(ROOT)} cell {index}: committed {name}: {message}"
                    )

    requirement_text = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for module, requirement in IMPORT_TO_REQUIREMENT.items():
        if module in imported_modules and not re.search(
            rf"^{re.escape(requirement)}(?:[=<>!~]|$)", requirement_text, re.MULTILINE
        ):
            failures.append(f"requirements.txt does not declare imported package: {requirement}")

    return failures


def execute_notebooks(names: list[str], timeout: int) -> list[str]:
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:
        return [f"execution dependencies are missing: {exc}; install requirements-ci.txt"]

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="llm-wiki-notebooks-") as temp_dir:
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("MPLCONFIGDIR", temp_dir)
        for name in names:
            path = ROOT / name
            try:
                notebook = nbformat.read(path, as_version=4)
                client = NotebookClient(
                    notebook,
                    timeout=timeout,
                    kernel_name="python3",
                    resources={"metadata": {"path": str(ROOT)}},
                    allow_errors=False,
                )
                client.execute()
                print(f"executed {name}")
            except Exception as exc:  # nbclient wraps cell context in its exception
                failures.append(f"{name}: clean-kernel execution failed: {exc}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        choices=("none", "fast", "all"),
        default="none",
        help="execute no notebooks, the deterministic fast suite, or all curriculum notebooks",
    )
    parser.add_argument("--timeout", type=int, default=600, help="per-cell timeout in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures = validate_structure()

    if args.execute == "fast":
        failures.extend(execute_notebooks(FAST_NOTEBOOKS, args.timeout))
    elif args.execute == "all":
        failures.extend(execute_notebooks(CANONICAL_NOTEBOOKS, args.timeout))

    if failures:
        print("Notebook integrity check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    suffix = "" if args.execute == "none" else f" with {args.execute} execution"
    print(f"Notebook integrity check passed{suffix}: {len(CANONICAL_NOTEBOOKS)} curriculum notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
