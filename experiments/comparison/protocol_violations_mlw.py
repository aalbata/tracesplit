"""Matched protocol violations submitted to the mlw package.

Each case expresses, as a sequence of mlw calls, one of the source-level
protocol violations that the TraceSplit checker rejects, together with the
valid workflow. The script records whether mlw raises before or during the
offending call. It runs in a separate environment because mlw pins pandas
below 3.0. Results are written as CSV for the comparison table.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
import pandas as pd
import ml

warnings.simplefilter("ignore")
ALGORITHMS = {"classification": "logistic", "regression": "linear"}
SPLITS = {"iris": {}, "wine": {}, "breast_cancer": {}, "digits": {}, "diabetes": {}, "mixed": {},
          "grouped": {"groups": "subject"}, "temporal": {"time": "time"}}
TASKS = {"diabetes": "regression", "temporal": "regression"}


def workflow(df, task, extra, seed=42):
    """Valid development loop: split, fit on train, evaluate on validation, assess once on test."""
    s = ml.split(df, "target", seed=seed, ratio=(0.6, 0.2, 0.2), task=task, **extra)
    model = ml.fit(s.train, "target", algorithm=ALGORITHMS[task], seed=seed, engine="sklearn", task=task)
    ml.evaluate(model, s.valid)
    return s, model


def cases(df, task, extra):
    fit = lambda frame, seed=42: ml.fit(frame, "target", algorithm=ALGORITHMS[task], seed=seed, engine="sklearn", task=task)
    split = lambda: ml.split(df, "target", seed=42, ratio=(0.6, 0.2, 0.2), task=task, **extra)

    def valid():
        s, model = workflow(df, task, extra)
        ml.assess(model, test=s.test)

    def fit_validation():
        s = split(); fit(s.valid)

    def fit_test():
        s = split(); fit(s.test)

    def prepare_all():
        prepared = ml.prepare(df, "target", task=task)
        frame = prepared.data.copy(); frame["target"] = df["target"].to_numpy()
        s = ml.split(frame, "target", seed=42, ratio=(0.6, 0.2, 0.2), task=task, **({} if extra else {}))
        fit(s.train)

    def preprocess_before_split():
        frame = df.copy()
        numeric = [c for c in frame.columns if c != "target" and pd.api.types.is_numeric_dtype(frame[c]) and c not in extra.values()]
        frame[numeric] = (frame[numeric] - frame[numeric].mean()) / frame[numeric].std(ddof=0).replace(0, 1)
        s = ml.split(frame, "target", seed=42, ratio=(0.6, 0.2, 0.2), task=task, **extra)
        fit(s.train)

    def evaluate_test():
        s, model = workflow(df, task, extra); ml.evaluate(model, s.test)

    def assess_validation():
        s, model = workflow(df, task, extra); ml.assess(model, test=s.valid)

    def second_assessment():
        s, model = workflow(df, task, extra); ml.assess(model, test=s.test); ml.assess(model, test=s.test)

    def select_by_test():
        s, first = workflow(df, task, extra); second = fit(s.train, seed=43)
        ml.assess(first, test=s.test); ml.assess(second, test=s.test)

    def post_assessment_fit():
        s, model = workflow(df, task, extra); ml.assess(model, test=s.test); ml.evaluate(model, s.valid)

    def test_rows_in_training():
        s = split(); fit(pd.concat([s.train, s.test]))

    def unsplit_fit():
        fit(df)

    def missing_assessment():
        workflow(df, task, extra)

    return {"valid_workflow": (True, valid), "fit_validation": (False, fit_validation), "fit_test": (False, fit_test),
            "prepare_all": (False, prepare_all), "preprocess_before_split": (False, preprocess_before_split),
            "evaluate_test": (False, evaluate_test), "assess_validation": (False, assess_validation),
            "second_assessment": (False, second_assessment), "select_by_test": (False, select_by_test),
            "post_assessment_fit": (False, post_assessment_fit), "test_rows_in_training": (False, test_rows_in_training),
            "unsplit_fit": (False, unsplit_fit), "missing_assessment": (False, missing_assessment)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = []
    for name, extra in SPLITS.items():
        df = pd.read_csv(args.datasets / f"{name}.csv")
        task = TASKS.get(name, "classification")
        for family, (expected_accept, action) in cases(df, task, extra).items():
            ml._provenance._registry.clear()
            try:
                action(); accepted, error, message = True, "", ""
            except Exception as exc:
                accepted, error, message = False, type(exc).__name__, str(exc).splitlines()[0][:200]
            rows.append({"dataset": name, "family": family, "expected_accept": expected_accept, "mlw_accepted": accepted,
                         "mlw_error": error, "mlw_message": message})
            print(f"{name:14s} {family:26s} accepted={accepted} {error}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    chain = ml._provenance._chain
    environment = {"mlw": ml.__version__, "pandas": pd.__version__, "python": sys.version.split()[0],
                   "audit_chain_entries_after_run": len(chain), "audit_chain_intact": chain.verify()}
    (args.output.with_suffix(".environment.json")).write_text(json.dumps(environment, indent=2) + "\n")
    print(json.dumps(environment, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
