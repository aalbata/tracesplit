"""The same protocol-violation families expressed as TraceSplit programs.

Families that the closed language cannot express are recorded as such rather
than counted as rejections.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from tracesplit import compile_source, LanguageError

ROOT = Path(__file__).resolve().parents[2]
INEXPRESSIBLE = {"test_rows_in_training": "No statement can fit an estimator on rows outside the training partition.",
                 "select_by_test": "Selection has no partition argument; it always uses validation scores."}


def variants(source):
    lines = source.splitlines(keepends=True)
    split_line = next(i for i, line in enumerate(lines) if line.startswith("split "))
    prepare_line = next(i for i, line in enumerate(lines) if line.startswith("prepare "))
    first_train = next(i for i, line in enumerate(lines) if line.startswith("train "))
    return {
        "valid_workflow": source,
        "fit_validation": source.replace("on train\n", "on validation\n", 2).replace("categorical most_frequent on validation", "categorical most_frequent on train"),
        "fit_test": source.replace("train linear logistic C=1 on train", "train linear logistic C=1 on test").replace("train linear ridge alpha=1 on train", "train linear ridge alpha=1 on test"),
        "prepare_all": source.replace("categorical most_frequent on train", "categorical most_frequent on all"),
        "preprocess_before_split": "".join(lines[:split_line] + [lines[prepare_line], lines[split_line]] + lines[split_line + 1:prepare_line] + lines[prepare_line + 1:]),
        "evaluate_test": source.replace("evaluate linear on validation", "evaluate linear on test"),
        "assess_validation": source.replace("assess best on test", "assess best on validation"),
        "second_assessment": source.replace("explain\n", "assess best on test\n"),
        "post_assessment_fit": source.replace("explain\n", "evaluate linear on validation\n"),
        "unsplit_fit": "".join(lines[:split_line] + [lines[first_train]] + lines[split_line:]),
        "missing_assessment": source.replace("assess best on test\n", "").replace("explain\n", ""),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = []
    for path in sorted((ROOT / "examples").glob("*.tsml")):
        source = path.read_text()
        cases = variants(source)
        for family, text in cases.items():
            if family != "valid_workflow" and text == source:
                raise AssertionError(f"Ineffective variant {family} for {path.stem}")
            try:
                compile_source(text); accepted, code = True, ""
            except LanguageError as exc:
                accepted, code = False, exc.code
            rows.append({"dataset": path.stem, "family": family, "expressible": True, "tracesplit_accepted": accepted, "tracesplit_error": code})
        for family, reason in INEXPRESSIBLE.items():
            rows.append({"dataset": path.stem, "family": family, "expressible": False, "tracesplit_accepted": False, "tracesplit_error": reason})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for row in rows:
        if row["dataset"] == "iris": print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
