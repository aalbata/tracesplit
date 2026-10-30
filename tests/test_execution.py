import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from tracesplit import compile_source, run_plan, run_source
from tracesplit.engine import ExecutionError, make_split
from tracesplit.exporter import export_bundle, compare_results
from tracesplit.verifier import verify, independent_metrics
from tracesplit.evidence import digest

@pytest.mark.parametrize("name", ["iris", "wine", "breast_cancer", "digits", "diabetes", "mixed", "grouped", "temporal"])
def test_run_verify_export(name, root, tmp_path):
    text = (root/"examples"/(name+".tsml")).read_text()
    result = run_source(text, root)
    report = verify(result, root/"datasets"/(name+".csv"))
    assert report["ok"], report
    out = export_bundle(result, root, tmp_path/name)
    spec = importlib.util.spec_from_file_location("replay_"+name, out/"experiment.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    assert compare_results(result, mod.run())["ok"]
    assert "import tracesplit" not in (out/"experiment.py").read_text()

def test_same_seed_same_state(source, root):
    a, b = run_source(source, root), run_source(source, root)
    assert a["partitions"] == b["partitions"]
    assert a["model_fingerprints"] == b["model_fingerprints"]
    assert a["events"] == b["events"]

def test_tie_rule(root, source):
    source = source.replace("train forest forest trees=40 depth=6", "train forest logistic C=1")
    source = source.replace("train tree tree depth=5", "train tree logistic C=1")
    result = run_source(source, root)
    assert result["selected"] == "linear"
    assert verify(result, root/"datasets"/"iris.csv")["ok"]

def test_workspace_escape(source, tmp_path):
    with pytest.raises(ExecutionError): run_source(source.replace('datasets/iris.csv', '../outside.csv'), tmp_path)

def test_missing_column(source, root):
    with pytest.raises(ExecutionError): run_source(source.replace('"target"', '"unknown"'), root)

def test_bad_csv_header(source, root, tmp_path):
    raw = (root/"datasets"/"iris.csv").read_text().replace("x0,x1", "x0,x0", 1)
    (tmp_path/"data.csv").write_text(raw)
    with pytest.raises(ExecutionError): run_source(source.replace("datasets/iris.csv", "data.csv"), tmp_path)

@pytest.mark.parametrize("fault", ["missing_target", "infinity", "one_class", "too_small"])
def test_data_validation(fault, source, root, tmp_path):
    frame = pd.read_csv(root/"datasets"/"iris.csv")
    if fault == "missing_target": frame.loc[0, "target"] = np.nan
    if fault == "infinity": frame.loc[0, "x0"] = np.inf
    if fault == "one_class": frame["target"] = 0
    if fault == "too_small": frame = frame.iloc[:10]
    frame.to_csv(tmp_path/"data.csv", index=False)
    with pytest.raises((ExecutionError, ValueError)): run_source(source.replace("datasets/iris.csv", "data.csv"), tmp_path)

def test_group_exclusion(root):
    r = run_source((root/"examples"/"grouped.tsml").read_text(), root)
    assert "subject" not in r["schema"]["numeric"] + r["schema"]["categorical"]

def test_temporal_tied_boundary(root, tmp_path):
    frame = pd.read_csv(root/"datasets"/"temporal.csv"); frame["time"] = 1
    frame.to_csv(tmp_path/"data.csv", index=False)
    source = (root/"examples"/"temporal.tsml").read_text().replace("datasets/temporal.csv", "data.csv")
    with pytest.raises(ExecutionError): run_source(source, tmp_path)

def test_holdout_noninterference_frozen(source, root, tmp_path):
    base = run_source(source, root)
    frame = pd.read_csv(root/"datasets"/"iris.csv"); ids = base["partitions"]["test"]
    frame.loc[ids, "x0"] = frame.loc[ids, "x0"]*100 + 17
    frame.to_csv(tmp_path/"data.csv", index=False)
    other = run_plan(compile_source(source.replace("datasets/iris.csv", "data.csv")), tmp_path, frozen_partitions=base["partitions"])
    assert base["preprocessing"] == other["preprocessing"]
    assert base["model_fingerprints"] == other["model_fingerprints"]
    assert base["validation"] == other["validation"]
    assert base["selected"] == other["selected"]

def test_export_changed_data_refused(source, root, tmp_path):
    result = run_source(source, root); out = export_bundle(result, root, tmp_path)
    with open(out/"data.csv", "a") as f: f.write("\n")
    completed = subprocess.run([sys.executable, str(out/"experiment.py"), "--output", str(out/"result.json")], capture_output=True, text=True)
    assert completed.returncode != 0 and "fingerprint" in completed.stderr

@pytest.mark.parametrize("task", ["classification", "regression"])
def test_independent_metrics(task):
    from tracesplit.engine import score
    y, pred = ([0, 1, 2, 0, 1, 2], [0, 1, 1, 0, 2, 2]) if task == "classification" else ([1, 3, 5, 7], [2, 2, 6, 8])
    assert independent_metrics(y, pred, task) == pytest.approx(score(y, pred, task))
