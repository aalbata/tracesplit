"""Fill the article templates from the result files and typeset both PDFs.

The script copies the generated figures next to the templates, runs the
number-filling script, and then runs pdflatex and bibtex. It reports the page
count of each PDF and fails on undefined references or overfull boxes.
"""
from __future__ import annotations
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command, cwd):
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        print(completed.stdout[-4000:])
        raise SystemExit(f"command failed: {' '.join(command)}")


def typeset(paper, name, bibliography):
    latex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"]
    run(latex, paper)
    if bibliography:
        run(["bibtex", name], paper)
        run(latex, paper)
    run(latex, paper)
    log = (paper / f"{name}.log").read_text(encoding="utf-8", errors="replace")
    problems = [line for line in log.splitlines() if "Overfull" in line or "undefined" in line.lower() and "reference" in line.lower()]
    if "There were undefined references" in log or "Citation" in log and "undefined" in log:
        problems.append("undefined references")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "full")
    parser.add_argument("--paper", type=Path, default=ROOT / "paper")
    args = parser.parse_args(argv)
    paper = args.paper.resolve()
    figures = paper / "figures"
    figures.mkdir(exist_ok=True)
    for name in ["fault_matrix.pdf", "runtime_differences.pdf"]:
        shutil.copyfile(args.results.resolve() / "figures" / name, figures / name)
    run([sys.executable, str(ROOT / "scripts" / "fill_manuscript.py"), "--results", str(args.results), "--paper", str(paper)], ROOT)
    report = {}
    report["Manuscript"] = typeset(paper, "Manuscript", True)
    report["Supplementary_Methods"] = typeset(paper, "Supplementary_Methods", False)
    for name, problems in report.items():
        pdf = paper / f"{name}.pdf"
        pages = max((int(m) for m in re.findall(rb"/Type\s*/Pages[^>]*?/Count\s+(\d+)", pdf.read_bytes(), re.S)), default=None)
        print(f"{name}.pdf: {pages if pages else '?'} pages; problems: {problems or 'none'}")
    if any(report.values()):
        raise SystemExit("typesetting problems were reported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
